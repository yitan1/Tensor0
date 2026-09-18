"""Update contract: independent numerical and boundary regressions."""

from itertools import product
from math import prod

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensor0._stride import StridedView, add, scale
from tensor0._stride._ffi._calls import execute_update
from tensor0._stride._ffi._descriptor import encode_layout
from tensor0._stride._ffi._registration import operation_target
from tensor0._stride._jax import reduction_p, update_p
from tensor0._stride._layout import AffineRecord

from ._support import native_available
from tests.stride.test_reduction_ad import values


pytestmark = pytest.mark.skipif(not native_available(), reason="native CPU stride is unavailable")

@pytest.fixture(autouse=True)
def enable_x64():
    with jax.enable_x64():
        yield

FLOAT_PAIRS = [
    ('float16', 'float32'),
    ('float32', 'complex64'),
    ('complex64', 'float32'),
    ('float64', 'complex128'),
    ('complex128', 'float64'),
    ('float32', 'float16'),
]

INTEGER_PAIRS = [('int32', 'bool'), ('int32', 'int8'), ('int32', 'int16'), ('uint32', 'uint8')]

PAIRS = FLOAT_PAIRS + INTEGER_PAIRS

def mixed_operands(source_dtype, base_dtype, batch_shape):
    source = (jnp.arange(prod(batch_shape) * 5) % 4 + 1).astype(source_dtype).reshape((*batch_shape, 5))
    base = (jnp.arange(prod(batch_shape) * 10) % 5 + 1).astype(base_dtype).reshape((*batch_shape, 10))
    if jnp.iscomplexobj(source):
        source = source + 2j
    if jnp.iscomplexobj(base):
        base = base - 3j
    return (source, base)

MIXED_RECORDS = (AffineRecord((2, 2), (0, -1), 3, (4, 2), 1), AffineRecord((), (), 2, (), 8))

def reference(source, base, alpha, beta, records=MIXED_RECORDS):
    expected = np.asarray(base).copy()
    for record in records:
        for coordinates in np.ndindex(record.logical_shape):
            source_index = record.source_offset + sum((index * stride for index, stride in zip(coordinates, record.source_strides, strict=True)))
            destination_index = record.destination_offset + sum((index * stride for index, stride in zip(coordinates, record.destination_strides, strict=True)))
            source_term = 0 if alpha == 0 else np.asarray(source)[..., source_index] * alpha
            base_term = 0 if beta == 0 else np.asarray(base)[..., destination_index] * beta
            value = source_term + base_term
            if not jnp.iscomplexobj(base):
                value = np.real(value)
            expected[..., destination_index] = np.asarray(value).astype(base.dtype)
    return expected

@pytest.mark.parametrize('source_dtype,base_dtype', PAIRS)
@pytest.mark.parametrize('batch_shape', [(), (2, 3), (0,), (2, 0)])
@pytest.mark.parametrize('alpha,beta', list(product((0, 1, 2), repeat=2)))
def test_mixed_storage_branches_records_and_batches(source_dtype, base_dtype, batch_shape, alpha, beta):
    with jax.enable_x64():
        source, base = mixed_operands(source_dtype, base_dtype, batch_shape)
        layout = encode_layout(MIXED_RECORDS, source_size=5, output_size=10)
        run = jax.jit(lambda values, original, first, second: execute_update(values, original, first, second, layout=layout))
        result = run(source, base, jnp.float32(alpha), jnp.full(batch_shape, beta, dtype=jnp.int32))
        assert result.dtype == base.dtype and result.shape == base.shape
        np.testing.assert_allclose(result, reference(source, base, alpha, beta), rtol=1e-06, atol=1e-06)

def test_mixed_complex_term_is_converted_after_addition():
    source = jnp.asarray([1 + 2j], jnp.complex64)
    base = jnp.asarray([3], jnp.float32)
    layout = encode_layout((AffineRecord((1,), (1,), 0, (1,), 0),), source_size=1, output_size=1)
    result = execute_update(source, base, jnp.complex64(1j), jnp.float32(1), layout=layout)
    np.testing.assert_array_equal(result, [1])

@pytest.mark.parametrize('alpha,beta', [(0, 0), (0, 2), (2, 0)])
def test_mixed_zero_terms_skip_nonfinite_inputs(alpha, beta):
    source = jnp.asarray([np.nan if alpha == 0 else 2], jnp.complex64)
    base = jnp.asarray([np.inf if beta == 0 else 3], jnp.float32)
    layout = encode_layout((AffineRecord((1,), (1,), 0, (1,), 0),), source_size=1, output_size=1)
    np.testing.assert_array_equal(execute_update(source, base, jnp.int32(alpha), jnp.int32(beta), layout=layout), [alpha * 2 + beta * 3])

@pytest.mark.parametrize('source_dtype,base_dtype', INTEGER_PAIRS)
def test_integer_storage_has_zero_coefficient_tangent(source_dtype, base_dtype):
    source = StridedView(jnp.ones(2, dtype=source_dtype), (2,), (1,), 0)
    base = StridedView(jnp.ones(2, dtype=base_dtype), (2,), (1,), 0)
    run = lambda factor: add(base, source, beta=factor).data
    np.testing.assert_array_equal(run(jnp.float32(2)), jnp.full(2, 3, dtype=base_dtype))
    result, tangent = jax.jvp(run, (jnp.float32(2),), (jnp.float32(1),))
    assert tangent.dtype == jax.dtypes.float0 and tangent.shape == result.shape
    np.testing.assert_array_equal(jax.grad(lambda value: jnp.sum(run(value).astype(jnp.float32)))(jnp.float32(2)), 0)

@pytest.mark.parametrize('source_dtype,base_dtype', PAIRS)
@pytest.mark.parametrize('output_size', [0, 3])
def test_empty_mixed_source_preserves_base(source_dtype, base_dtype, output_size):
    with jax.enable_x64():
        source = jnp.empty((2, 0), dtype=source_dtype)
        base = jnp.ones((2, output_size), dtype=base_dtype)
        layout = encode_layout((AffineRecord((0,), (1,), 0, (1,), 0),), source_size=0, output_size=output_size)
        np.testing.assert_array_equal(execute_update(source, base, jnp.int32(0), jnp.int32(0), layout=layout), base)

@pytest.mark.parametrize('source_dtype,base_dtype', [('float16', 'float32'), ('float32', 'float16')])
def test_native_mixed_source_rank_and_base_alias_checks(source_dtype, base_dtype):
    layout = encode_layout(MIXED_RECORDS, source_size=5, output_size=10)
    source, base = mixed_operands(source_dtype, base_dtype, (1,))
    target = operation_target('update', base.dtype)
    call = jax.ffi.ffi_call(target, jax.ShapeDtypeStruct(base.shape, base.dtype))
    with pytest.raises(Exception, match='rank-two'):
        call(source[0], base, jnp.int32(1), jnp.int32(1), layout=layout).block_until_ready()
    alias_call = jax.ffi.ffi_call(target, jax.ShapeDtypeStruct(base.shape, base.dtype), input_output_aliases={1: 0})
    result = alias_call(source, base, jnp.int32(1), jnp.int32(1), layout=layout)
    np.testing.assert_array_equal(result, reference(source, base, 1, 1))

def test_source_gradient_preserves_record_accumulation_order():
    records = tuple((AffineRecord((), (), 0, (), destination) for destination in range(3)))
    run = lambda data: update_p.bind(data, jnp.zeros(3, dtype=jnp.float32), jnp.float32(1), jnp.float32(0), records=records)
    cotangent = jnp.asarray([2 ** 24, 1, -2 ** 24], dtype=jnp.float32)
    np.testing.assert_array_equal(jax.linear_transpose(run, jnp.zeros(1, dtype=jnp.float32))(cotangent)[0], [0])

def address_pairs(record):
    return tuple(((record.source_offset + sum((axis * stride for axis, stride in zip(coordinates, record.source_strides))), record.destination_offset + sum((axis * stride for axis, stride in zip(coordinates, record.destination_strides)))) for coordinates in np.ndindex(record.logical_shape)))

def assert_disjoint_writes(records):
    destinations = [destination for record in records for _, destination in address_pairs(record)]
    assert len(destinations) == len(set(destinations))

@pytest.mark.parametrize('shape,strides,offset', [((2, 2), (4, 1), 1), ((3,), (-2,), 7), ((), (), 3), ((0,), (1,), 16)])
def test_algebra_generators_produce_disjoint_writes(monkeypatch, shape, strides, offset):
    generated = []

    def capture(source, base, alpha, beta, *, records):
        generated.append(records)
        return base
    monkeypatch.setattr(update_p, 'bind', capture)
    view = StridedView(jnp.arange(16, dtype=jnp.float32), shape, strides, offset)
    repeated = StridedView(view.data, shape, (0,) * len(shape), 0)
    scale(view, 2)
    add(view, repeated, alpha=2, beta=3)
    assert len(generated) == 2
    for records in generated:
        assert len(records) == 1
        assert_disjoint_writes(records)

def single_record_operands(dtype, batch_shape, coefficient_shapes):
    source = (jnp.arange(prod(batch_shape) * 5) % 5).astype(dtype).reshape((*batch_shape, 5))
    base = (jnp.arange(prod(batch_shape) * 10) % 7).astype(dtype).reshape((*batch_shape, 10))
    if dtype in ('complex64', 'complex128'):
        source, base = (source + 1j * (source - 2), base + 1j * (3 - base))
    alpha = jnp.full(coefficient_shapes[0], 2 + 1j if dtype in ('complex64', 'complex128') else 2, dtype=dtype)
    beta = jnp.full(coefficient_shapes[1], 3 - 2j if dtype in ('complex64', 'complex128') else 3, dtype=dtype)
    return (source, base, alpha, beta)

def single_record_functions(record):
    source_indices = jnp.asarray([record.source_offset + sum((axis * stride for axis, stride in zip(coordinates, record.source_strides))) for coordinates in np.ndindex(record.logical_shape)], dtype=jnp.int32)
    base_indices = jnp.asarray([record.destination_offset + sum((axis * stride for axis, stride in zip(coordinates, record.destination_strides))) for coordinates in np.ndindex(record.logical_shape)], dtype=jnp.int32)

    def execute(source, base, alpha, beta):
        return update_p.bind(source, base, alpha, beta, records=(record,))

    def reference(source, base, alpha, beta):
        factors = tuple((value.reshape(()) if value.shape == (1,) else value for value in (alpha, beta)))
        alpha, beta = tuple((value[..., None] if value.ndim else value for value in factors))
        return base.at[..., base_indices].set(alpha * source[..., source_indices] + beta * base[..., base_indices])
    return (execute, reference)

update_coefficient_ad_RECORDS = [
    AffineRecord((2, 2), (1, 1), 1, (2, 1), 1),
    AffineRecord((2, 2, 2), (0, -1, -1), 3, (-4, -2, -1), 8),
    AffineRecord((0,), (1,), 5, (1,), 10),
    AffineRecord((), (), 2, (), 3),
]

@pytest.mark.parametrize('dtype', ['float32', 'float64', 'complex64', 'complex128'])
@pytest.mark.parametrize('active', [(2, 3), (0, 3), (1, 2)])
def test_direct_transpose_of_independent_terms(dtype, active):
    arguments = single_record_operands(dtype, (2, 3), ((), (2, 3)))
    execute, reference = single_record_functions(update_coefficient_ad_RECORDS[0])

    def bind(function, *values):
        operands = list(arguments)
        for index, value in zip(active, values, strict=True):
            operands[index] = value
        return function(*operands)
    primals = tuple((arguments[index] for index in active))
    run, oracle = (lambda *values: bind(execute, *values), lambda *values: bind(reference, *values))
    for actual, expected in zip(jax.jit(jax.linear_transpose(run, *primals))(arguments[1]), jax.vjp(oracle, *primals)[1](arguments[1]), strict=True):
        np.testing.assert_array_equal(actual, expected)

def test_selected_dot_lowering_and_unselected_nonfinite_values():
    execute, _ = single_record_functions(update_coefficient_ad_RECORDS[0])
    source = jnp.asarray([np.nan, 2, 3, 4, np.inf], jnp.float32)
    base = jnp.asarray([np.nan, 5, 6, 7, 8, np.inf, np.nan, np.inf, np.nan, np.inf], jnp.float32)
    cotangent = jnp.asarray([np.nan, 1, 2, 3, 4, np.inf, np.nan, np.inf, np.nan, np.inf], jnp.float32)
    reverse = jax.jit(lambda first, second, value: jax.vjp(lambda alpha, beta: execute(first, second, alpha, beta), jnp.float32(0), jnp.float32(1))[1](value))
    alpha_gradient, beta_gradient = reverse(source, base, cotangent)
    np.testing.assert_array_equal(alpha_gradient, 33)
    np.testing.assert_array_equal(beta_gradient, 70)
    text = reverse.lower(source, base, cotangent).as_text()
    assert text.count('stablehlo.custom_call @tensor0_stride_dot_f32_cpu_v1') == 2
    for operation in ('tensor0_stride_update', 'stablehlo.gather', 'stablehlo.scatter'):
        assert operation not in text

def test_joint_bilinear_transpose_remains_rejected():
    source, base, alpha, beta = single_record_operands('float32', (), ((), ()))
    execute, _ = single_record_functions(update_coefficient_ad_RECORDS[0])
    with pytest.raises(NotImplementedError, match='known operand'):
        jax.linear_transpose(lambda inputs, factor: execute(inputs, base, factor, beta), source, alpha)(base)
    with pytest.raises(NotImplementedError, match='known operand'):
        jax.linear_transpose(lambda inputs, factor: execute(source, inputs, alpha, factor), base, beta)(base)

def test_coefficient_projection_after_multiplication():
    record = AffineRecord((1,), (1,), 0, (1,), 1)
    source = jnp.asarray([2 + 3j], jnp.complex64)
    base = jnp.asarray([7 + 2j, 4 + 5j, 9 + 1j], jnp.complex64)
    alpha, beta = (jnp.float64(2), jnp.float32(3))
    run = lambda data, original, first, second: update_p.bind(data, original, first, second, records=(record,))
    cotangent = jnp.asarray([3 + 1j, 1 + 2j, 5 + 4j], jnp.complex64)
    gradients = jax.jit(jax.vjp(run, source, base, alpha, beta)[1])(cotangent)
    np.testing.assert_array_equal(gradients[0], [2 + 4j])
    np.testing.assert_array_equal(gradients[1], [3 + 1j, 3 + 6j, 5 + 4j])
    np.testing.assert_array_equal(gradients[2], -4)
    np.testing.assert_array_equal(gradients[3], -6)
    assert gradients[2].dtype == alpha.dtype and gradients[3].dtype == beta.dtype
    text = jax.jit(lambda data, original, value: jax.vjp(lambda first, second: run(data, original, first, second), alpha, beta)[1](value)).lower(source, base, cotangent).as_text()
    assert text.count('stablehlo.custom_call @tensor0_stride_dot_') == 2
    for operation in ('stablehlo.real', 'stablehlo.gather', 'stablehlo.scatter', 'tensor0_stride_update'):
        assert operation not in text

def update_functions(records):
    assert_disjoint_writes(records)
    indices = tuple(((jnp.asarray([source for source, _ in address_pairs(record)], dtype=jnp.int32), jnp.asarray([destination for _, destination in address_pairs(record)], dtype=jnp.int32)) for record in records))

    def execute(source, base, alpha, beta):
        return update_p.bind(source, base, alpha, beta, records=records)

    def reference(source, base, alpha, beta):
        factors = tuple((value.reshape(()) if value.shape == (1,) else value for value in (alpha, beta)))
        alpha, beta = tuple((value[..., None] if value.ndim else value for value in factors))
        result = base
        for source_indices, destination_indices in indices:
            contribution = alpha * source[..., source_indices] + beta * base[..., destination_indices]
            if not jnp.iscomplexobj(base):
                contribution = jnp.real(contribution)
            result = result.at[..., destination_indices].set(contribution.astype(base.dtype))
        return result
    return (execute, reference)

LAYOUTS = [
    (AffineRecord((3,), (1,), 0, (2,), 1), AffineRecord((3,), (1,), 0, (2,), 2)),
    (AffineRecord((2, 2), (1, 1), 0, (4, 1), 1), AffineRecord((2,), (0,), 3, (-4,), 7)),
    (AffineRecord((3,), (-1,), 5, (-2,), 6), AffineRecord((2,), (1,), 0, (2,), 1)),
    (AffineRecord((), (), 0, (), 2), AffineRecord((), (), 0, (), 5)),
    (AffineRecord((2,), (1,), 1, (1,), 2), AffineRecord((0,), (1,), 6, (1,), 10)),
    (AffineRecord((1, 1), (2 ** 63 - 1,) * 2, 0, (2 ** 63 - 1,) * 2, 1), AffineRecord((), (), 3, (), 4)),
    (AffineRecord((0,), (1,), 6, (1,), 10), AffineRecord((0,), (1,), 6, (1,), 10)),
    (),
]

@pytest.mark.parametrize('dtype,coefficient_dtype', [('float32', 'complex128'), ('complex64', 'float64')])
def test_sequential_updates_and_vmap(dtype, coefficient_dtype):
    first, first_reference = update_functions(LAYOUTS[0])
    second, second_reference = update_functions(LAYOUTS[2])
    run = lambda source, base, alpha, beta: second(source, first(source, base, alpha, beta), beta, alpha)
    reference = lambda source, base, alpha, beta: second_reference(source, first_reference(source, base, alpha, beta), beta, alpha)
    arguments = (values((2, 6), dtype), values((2, 10), dtype), jnp.full((2,), 2, coefficient_dtype), jnp.full((2,), 0.5, coefficient_dtype))
    run, reference = (jax.vmap(run), jax.vmap(reference))
    directions = tuple((jnp.ones_like(value) for value in arguments))
    for actual, expected in zip(jax.jit(lambda *inputs: jax.jvp(run, inputs, directions))(*arguments), jax.jvp(reference, arguments, directions), strict=True):
        np.testing.assert_allclose(actual, expected, rtol=3e-06, atol=3e-06)
    cotangent = jnp.ones_like(arguments[1])
    for actual, expected, original in zip(jax.jit(jax.vjp(run, *arguments)[1])(cotangent), jax.vjp(reference, *arguments)[1](cotangent), arguments, strict=True):
        assert actual.dtype == original.dtype and actual.shape == original.shape
        np.testing.assert_allclose(actual, expected, rtol=3e-06, atol=3e-06)

def compare(run, reference, arguments):
    directions = tuple((jnp.ones_like(value) for value in arguments))
    for actual, expected in zip(jax.jit(lambda *inputs: jax.jvp(run, inputs, directions))(*arguments), jax.jvp(reference, arguments, directions), strict=True):
        np.testing.assert_allclose(actual, expected, rtol=3e-06, atol=3e-06)
    cotangent = jnp.full_like(run(*arguments), 1 + 2j if jnp.iscomplexobj(arguments[1]) else 2)
    for actual, expected, original in zip(jax.jit(jax.vjp(run, *arguments)[1])(cotangent), jax.vjp(reference, *arguments)[1](cotangent), arguments, strict=True):
        assert actual.dtype == original.dtype and actual.shape == original.shape
        np.testing.assert_allclose(actual, expected, rtol=3e-06, atol=3e-06)
    coefficient_run = lambda alpha, beta: run(*arguments[:2], alpha, beta)
    for actual, expected in zip(jax.jit(jax.linear_transpose(coefficient_run, *arguments[2:]))(cotangent), jax.vjp(reference, *arguments)[1](cotangent)[2:], strict=True):
        np.testing.assert_allclose(actual, expected, rtol=3e-06, atol=3e-06)

@pytest.mark.parametrize('dtype,coefficient_dtype', [('float32', 'complex128'), ('complex64', 'float64')])
def test_public_add_coefficient_order(dtype, coefficient_dtype):
    source, base = (values((6,), dtype), values((10,), dtype))
    alpha = jnp.asarray(2 + 1j if coefficient_dtype.startswith('complex') else 2, coefficient_dtype)
    beta = jnp.asarray(0.5, dtype)
    record = AffineRecord((3,), (0,), 2, (2,), 1)
    _, reference = update_functions((record,))

    def run(data, original, source_factor, base_factor):
        left = StridedView(original, (3,), (2,), 1)
        right = StridedView(data, (3,), (0,), 2)
        return add(left, right, alpha=base_factor, beta=source_factor).data
    compare(run, reference, (source, base, alpha, beta))

DTYPES = ('float32', 'float64', 'complex64', 'complex128')

@pytest.mark.parametrize('source_dtype,base_dtype', FLOAT_PAIRS[1:])
@pytest.mark.parametrize('alpha_dtype,beta_dtype', list(product(DTYPES, repeat=2)))
@pytest.mark.parametrize('batch_shape', [(), (2, 3), (0,), (2, 0)])
def test_joint_input_and_coefficient_derivatives(source_dtype, base_dtype, alpha_dtype, beta_dtype, batch_shape):
    run, reference = update_functions(LAYOUTS[1])
    arguments = (values((*batch_shape, 6), source_dtype), values((*batch_shape, 10), base_dtype), jnp.asarray(2 + 1j if alpha_dtype.startswith('complex') else 2, alpha_dtype), jnp.full(batch_shape, 0.5 - 1j if beta_dtype.startswith('complex') else 0.5, beta_dtype))
    compare(run, reference, arguments)

def test_source_projection_after_multiplication():
    record = AffineRecord((2,), (0,), 0, (1,), 1)
    source = jnp.asarray([2], jnp.float32)
    base = jnp.asarray([7, 4, 9, 3], jnp.complex64)
    run = lambda data, original: update_p.bind(data, original, jnp.complex64(3 + 4j), jnp.float32(2), records=(record,))
    cotangent = jnp.asarray([3 + 1j, 1 + 2j, 1 + 2j, 5 + 4j], jnp.complex64)
    source_gradient, base_gradient = jax.jit(jax.vjp(run, source, base)[1])(cotangent)
    np.testing.assert_array_equal(source_gradient, [-10])
    np.testing.assert_array_equal(base_gradient, [3 + 1j, 2 + 4j, 2 + 4j, 5 + 4j])
    text = jax.jit(lambda data, original, value: jax.vjp(run, data, original)[1](value)).lower(source, base, cotangent).as_text()
    assert 'tensor0_stride_accumulation_' in text
    for name in ('stablehlo.real', 'stablehlo.gather', 'stablehlo.scatter'):
        assert name not in text

def test_low_precision_coefficient_ad():
    source, base = (jnp.ones(1, jnp.float16), jnp.ones(1, jnp.float32))
    record = AffineRecord((1,), (1,), 0, (1,), 0)
    run = lambda factor: update_p.bind(source, base, factor, jnp.float32(1), records=(record,))
    primal, tangent = jax.jvp(run, (jnp.float32(2),), (jnp.float32(1),))
    np.testing.assert_array_equal(primal, [3])
    np.testing.assert_array_equal(tangent, [1])
    np.testing.assert_array_equal(jax.linear_transpose(run, jnp.float32(2))(base)[0], 1)

def test_f16_fused_reduction_avoids_intermediate_storage_rounding():
    left, right = (jnp.zeros(3, dtype=jnp.float16), jnp.ones(1, dtype=jnp.float16))
    factor = jnp.float32(1.5)
    execute = lambda inputs: add(StridedView(left, (3,), (1,), 0), StridedView(inputs, (3,), (0,), 0), alpha=0, beta=factor).data
    cotangent = jnp.asarray([2048, 1, -2048], dtype=jnp.float16)
    actual = jax.jit(jax.vjp(execute, right)[1])(cotangent)[0]
    np.testing.assert_array_equal(actual, [1.5])
    fused = reduction_p.bind(cotangent, factor, records=(AffineRecord((3,), (1,), 0, (1,), 0),), output_shapes=((1,),), reduction_axes=((True,),), coefficient_records=(0,), output_size=1, dtype=jnp.dtype(jnp.float16))
    np.testing.assert_array_equal(fused, [1.5])

def add_ad_functions(shape, left_strides, left_offset, right_strides, right_offset, alpha, beta):
    left_indices = jnp.asarray([left_offset + sum((index * stride for index, stride in zip(coordinate, left_strides, strict=True))) for coordinate in np.ndindex(shape)], dtype=jnp.int32)
    right_indices = jnp.asarray([right_offset + sum((index * stride for index, stride in zip(coordinate, right_strides, strict=True))) for coordinate in np.ndindex(shape)], dtype=jnp.int32)

    def execute(left, right):
        return add(StridedView(left, shape, left_strides, left_offset), StridedView(right, shape, right_strides, right_offset), alpha=alpha, beta=beta).data

    def reference(left, right):
        left_factor, right_factor = (jnp.asarray(alpha), jnp.asarray(beta))
        if left_factor.ndim:
            left_factor = left_factor[..., None]
        if right_factor.ndim:
            right_factor = right_factor[..., None]
        left_values, right_values = (left[..., left_indices], right[..., right_indices])
        left_term = jnp.where(left_factor == 0, jnp.zeros_like(left_values), jnp.where(left_factor == 1, left_values, left_factor * left_values))
        right_term = jnp.where(right_factor == 0, jnp.zeros_like(right_values), jnp.where(right_factor == 1, right_values, right_factor * right_values))
        return left.at[..., left_indices].set((left_term + right_term).astype(left.dtype))
    return (execute, reference)

def test_nonzero_stride_overlap_and_second_derivative():
    left, right = (jnp.ones(8), jnp.ones(3))
    execute, reference = add_ad_functions((2, 2, 2), (4, 2, 1), 0, (0, 1, 1), 0, 1, 2)
    np.testing.assert_array_equal(jax.jit(jax.grad(lambda inputs: execute(left, inputs).sum()))(right), [4, 8, 4])
    np.testing.assert_array_equal(jax.jit(jax.hessian(lambda inputs: (execute(left, inputs) ** 2).sum()))(right), jax.hessian(lambda inputs: (reference(left, inputs) ** 2).sum())(right))

DISCRETE = ('bool', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64')

STORAGE_PAIRS = [(dtype, dtype) for dtype in DISCRETE] + INTEGER_PAIRS

COEFFICIENTS = ('float16', 'bfloat16', 'float32', 'float64', 'complex64', 'complex128')

def check(run, arguments):
    directions = tuple((jnp.ones_like(value) if jnp.issubdtype(value.dtype, jnp.inexact) else np.zeros(value.shape, dtype=jax.dtypes.float0) for value in arguments))
    expected = run(*arguments)
    primal, tangent = jax.jit(lambda *values: jax.jvp(run, values, directions))(*arguments)
    assert primal.dtype == expected.dtype
    np.testing.assert_array_equal(primal, expected)
    assert tangent.dtype == jax.dtypes.float0 and tangent.shape == primal.shape
    _, pullback = jax.vjp(run, *arguments)
    gradients = jax.jit(pullback)(np.zeros(primal.shape, dtype=jax.dtypes.float0))
    for value, original in zip(gradients, arguments, strict=True):
        assert value.shape == original.shape
        if jnp.issubdtype(original.dtype, jnp.inexact):
            assert value.dtype == original.dtype
            np.testing.assert_array_equal(value, jnp.zeros_like(original))
        else:
            assert value.dtype == jax.dtypes.float0

@pytest.mark.parametrize('source_dtype,base_dtype', STORAGE_PAIRS)
@pytest.mark.parametrize('coefficient_dtype', COEFFICIENTS)
def test_joint_primals(source_dtype, base_dtype, coefficient_dtype):
    source = jnp.asarray([1, 2, 3, 1, 2, 3], dtype=source_dtype)
    base = jnp.asarray([0, 1, 2, 3, 4, 0, 1, 2, 3, 4], dtype=base_dtype)
    alpha, beta = (jnp.asarray(2, dtype=coefficient_dtype), jnp.asarray(0.5, dtype=coefficient_dtype))
    run = lambda *values: update_p.bind(*values, records=LAYOUTS[1])
    check(run, (source, base, alpha, beta))

@pytest.mark.parametrize('factor', [0.0, 1.0, 2.0])
@pytest.mark.parametrize('empty', [False, True])
def test_composed_derivatives(factor, empty):
    storage = jnp.asarray([1, 2, 3], dtype=jnp.int32)
    view = StridedView(storage, (0 if empty else 2,), (1,), 1)

    def loss(coefficient):
        return jnp.sum(scale(view, coefficient).data.astype(jnp.float32))
    coefficient = jnp.float32(factor)
    gradient = jax.grad(loss)
    np.testing.assert_array_equal(jax.jit(gradient)(coefficient), 0)
    np.testing.assert_array_equal(jax.jit(jax.grad(gradient))(coefficient), 0)
    batch = jnp.asarray([0, 1, 2], dtype=jnp.float32)
    np.testing.assert_array_equal(jax.jit(jax.vmap(gradient))(batch), jnp.zeros_like(batch))
    text = jax.jit(gradient).lower(coefficient).as_text()
    assert 'tensor0_stride_dot_' not in text
    assert 'tensor0_stride_accumulation_' not in text
