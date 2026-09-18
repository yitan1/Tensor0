"""Accumulation contract: independent numerical and boundary regressions."""

from math import prod

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensor0._stride._ffi._calls import execute_accumulation, execute_copy
from tensor0._stride._ffi._descriptor import encode_layout, encode_reduction_layout
from tensor0._stride._ffi._registration import operation_target
from tensor0._stride._jax import accumulation_p, dot_p
from tensor0._stride._layout import AffineRecord

from ._support import native_available
from tests.stride.test_reduction_ad import values


pytestmark = pytest.mark.skipif(not native_available(), reason="native CPU stride is unavailable")

@pytest.fixture(autouse=True)
def enable_x64():
    with jax.enable_x64():
        yield

OVERLAP = AffineRecord((2, 2), (1, 2), 0, (1, 1), 1)

@pytest.mark.parametrize('batch_shape', [(), (2,), (2, 3), (0,), (2, 0)])
@pytest.mark.parametrize('record', [OVERLAP, AffineRecord((2, 2, 2), (-1, -2, -4), 7, (1, -1, 0), 2), AffineRecord((2, 2, 2), (1, 2, 4), 0, (0, 1, 1), 2), AffineRecord((0,), (-(1 << 63),), 8, ((1 << 63) - 1,), 8), AffineRecord((), (), 2, (), 3), AffineRecord((1,) * 70, ((1 << 63) - 1,) * 70, 2, (-(1 << 63),) * 70, 3)])
def test_layouts_and_batches(batch_shape, record):
    source = jnp.arange(prod(batch_shape) * 8, dtype=jnp.float32).reshape((*batch_shape, 8))
    expected = np.zeros((*batch_shape, 8), np.float32)
    for coordinates in np.ndindex(record.logical_shape):
        source_index = record.source_offset + sum((axis * stride for axis, stride in zip(coordinates, record.source_strides)))
        output_index = record.destination_offset + sum((axis * stride for axis, stride in zip(coordinates, record.destination_strides)))
        expected[..., output_index] += np.asarray(source)[..., source_index]
    layout = encode_layout((record,), source_size=8, output_size=8)
    execute = lambda data: execute_accumulation(data, layout=layout, output_size=8)
    for call in (execute, jax.jit(execute)):
        np.testing.assert_array_equal(call(source), expected)

@pytest.mark.parametrize('batch_shape', [(), (3,), (2, 3), (0,), (2, 0)])
def test_record_coefficients_and_reuse(batch_shape):
    records = (OVERLAP, AffineRecord((0,), (1,), 8, (1,), 8), AffineRecord((2, 2), (-1, -2), 7, (-1, -1), 3))
    source = jnp.arange(prod(batch_shape) * 8, dtype=jnp.float32).reshape((*batch_shape, 8)) + 1
    original = np.asarray(source).copy()
    factors = jnp.asarray(np.arange(prod(batch_shape)).reshape(batch_shape) % 3, dtype=jnp.int32)
    layout = encode_layout(records, source_size=8, output_size=8)
    execute = jax.jit(lambda data, first, second: execute_accumulation(data, (first, second), coefficient_records=(0, 2), layout=layout, output_size=8))
    for shift in (0, 2):
        first, second = (jnp.float32(2 + shift), factors + shift)
        expected = np.zeros((*batch_shape, 8), np.float32)
        for record, coefficient in ((records[0], first), (records[2], second)):
            for coordinates in np.ndindex(record.logical_shape):
                source_index = record.source_offset + sum((axis * stride for axis, stride in zip(coordinates, record.source_strides)))
                output_index = record.destination_offset + sum((axis * stride for axis, stride in zip(coordinates, record.destination_strides)))
                expected[..., output_index] += original[..., source_index] * np.asarray(coefficient)
        np.testing.assert_array_equal(execute(source, first, second), expected)
    np.testing.assert_array_equal(source, original)
    hlo = execute.lower(source, jnp.float32(2), factors).compiler_ir(dialect='hlo').as_hlo_text()
    assert hlo.count('custom-call(') == (1 if prod(batch_shape) else 0)

def test_map_and_axis_reduction_still_reject_overlap():
    layout = encode_layout((OVERLAP,), source_size=4, output_size=5)
    with pytest.raises(Exception, match='injective'):
        execute_copy(jnp.ones(4), layout=layout, output_size=5).block_until_ready()
    axis_layout = encode_reduction_layout(
        (AffineRecord((2, 2), (1, 2), 0, (1, 1), 1),),
        output_shapes=((2, 2),), reduction_axes=((False, False),),
        source_size=4, output_size=5)
    with pytest.raises(Exception, match='injective'):
        jax.ffi.ffi_call(operation_target('reduction', jnp.dtype('float32')), jax.ShapeDtypeStruct((1, 5), jnp.float32))(jnp.ones((1, 4)), layout=axis_layout, coefficient_records=np.asarray([], np.int64)).block_until_ready()

def raw_call(source, layout, *, coefficients=(), indices=(), shape=(1, 5), dtype=jnp.float32, aliases=None):
    return jax.ffi.ffi_call(operation_target('accumulation', jnp.dtype(dtype)), jax.ShapeDtypeStruct(shape, dtype), vmap_method='sequential', input_output_aliases=aliases)(source, *coefficients, layout=layout, coefficient_records=np.asarray(indices, dtype=np.int64))

def test_invalid_protocol_and_buffers():
    layout = encode_layout((OVERLAP,), source_size=4, output_size=5)
    for length in range(layout.size):
        with pytest.raises(Exception, match='descriptor|record count|rank'):
            raw_call(jnp.ones((1, 4)), layout[:length]).block_until_ready()
    malformed = [np.append(layout, 0)]
    for index, value in ((0, 2), (1, -1), (2, 2), (3, -1), (4, -1), (5, -1), (6, 4), (7, -1)):
        words = layout.copy()
        words[index] = value
        malformed.append(words)
    for words in malformed:
        with pytest.raises(Exception, match='layout|address|descriptor'):
            raw_call(jnp.ones((1, 4)), words).block_until_ready()
    for source, shape in ((jnp.ones(4), (1, 5)), (jnp.ones((1, 3)), (1, 5)), (jnp.ones((1, 4)), (2, 5)), (jnp.ones((1, 4)), (1, 4))):
        with pytest.raises(Exception, match='rank-two|dimensions'):
            raw_call(source, layout, shape=shape).block_until_ready()
    for indices, coefficients in (((0,), ()), ((), (jnp.float32(2),)), ((1,), (jnp.float32(2),)), ((-1,), (jnp.float32(2),)), ((0, 0), (jnp.float32(2), jnp.float32(3)))):
        with pytest.raises(Exception, match='coefficient.*(count|indices)'):
            raw_call(jnp.ones((1, 4)), layout, coefficients=coefficients, indices=indices).block_until_ready()
    for coefficient in (jnp.ones(2), jnp.ones((1, 1)), jnp.float8_e4m3fn(1)):
        with pytest.raises(Exception, match='batch count|unsupported scalar dtype'):
            raw_call(jnp.ones((1, 4)), layout, coefficients=(coefficient,), indices=(0,)).block_until_ready()
    with pytest.raises(Exception, match='unsupported scalar dtype'):
        raw_call(jnp.ones((1, 4), jnp.float8_e4m3fn), layout).block_until_ready()

def test_alias_and_python_boundaries():
    layout = encode_layout((OVERLAP,), source_size=4, output_size=4)
    with pytest.raises(Exception, match='overlap'):
        raw_call(jnp.ones((1, 4), dtype=jnp.float32), layout, shape=(1, 4), aliases={0: 0}).block_until_ready()
    with pytest.raises(TypeError, match='JAX'):
        execute_accumulation(np.ones(4), layout=layout, output_size=4)
    with pytest.raises(ValueError, match='storage'):
        execute_accumulation(jnp.float32(1), layout=layout, output_size=4)
    with pytest.raises(TypeError, match='explicit dtypes'):
        execute_accumulation(jnp.ones(4), (2,), coefficient_records=(0,), layout=layout, output_size=4)
    with pytest.raises(ValueError, match='batch shape'):
        execute_accumulation(jnp.ones(4), (jnp.ones(2),), coefficient_records=(0,), layout=layout, output_size=4)
    with pytest.raises(ValueError, match='nonnegative int64'):
        execute_accumulation(jnp.ones(4), coefficient_records=(-1,), layout=layout, output_size=4)
    with pytest.raises(ValueError, match='cannot be differentiated'):
        jax.grad(lambda source: execute_accumulation(source, layout=layout, output_size=4).sum())(jnp.ones(4))

def test_empty_storage_and_no_records():
    for records in ((), (AffineRecord((0,), (1,), 0, (1,), 5),)):
        np.testing.assert_array_equal(execute_accumulation(jnp.empty(0, jnp.float32), layout=encode_layout(records, source_size=0, output_size=5), output_size=5), np.zeros(5))
    assert execute_accumulation(jnp.empty((2, 0)), layout=encode_layout((), source_size=0, output_size=0), output_size=0).shape == (2, 0)

@pytest.mark.parametrize('batch_shape', [(), (2,), (2, 3), (0,), (2, 0, 3)])
@pytest.mark.parametrize('per_record', [False, True])
def test_dynamic_record_factors_batches_and_empty_records(batch_shape, per_record):
    records = (AffineRecord((0,), (1,), 0, (1,), 0), AffineRecord((2, 3), (1, 2), 0, (3, 1), 1), AffineRecord((3,), (0,), 6, (1,), 7))
    layout = encode_layout(records, source_size=7, output_size=11)
    source = jnp.arange(prod(batch_shape) * 7, dtype=jnp.int16).reshape((*batch_shape, 7))
    factors = jnp.asarray([99, 0.5, -0.25] if per_record else 0.5, dtype=jnp.float32)
    execute = jax.jit(lambda values, coefficients: execute_accumulation(values, tuple(coefficients) if per_record else (coefficients,) * 3, coefficient_records=(0, 1, 2), layout=layout, output_size=11, dtype='float32'))
    for coefficients in (factors, factors + 1):
        expected = np.zeros((*batch_shape, 11), dtype=np.float32)
        factor = np.asarray(coefficients)
        expected[..., 1:7] = np.asarray(source)[..., [0, 2, 4, 1, 3, 5]] * (factor[1] if per_record else factor)
        expected[..., 7:10] = np.asarray(source)[..., 6:7] * (factor[2] if per_record else factor)
        np.testing.assert_array_equal(execute(source, coefficients), expected)

@pytest.mark.parametrize('tiny_source', [False, True])
def test_no_conversion_before_multiply(tiny_source):
    with jax.enable_x64():
        source = jnp.asarray([1e-46 if tiny_source else 1e+30], dtype=jnp.float64)
        factor = jnp.asarray(1e+30 if tiny_source else 1e-46, dtype=jnp.float64)
        layout = encode_layout((AffineRecord((), (), 0, (), 0),), source_size=1, output_size=1)
        result = execute_accumulation(source, (factor,), coefficient_records=(0,), layout=layout, output_size=1, dtype='float32')
        np.testing.assert_allclose(result, [np.float32(1e-16)], rtol=1e-06)

def test_integer_product_wraps_before_output_conversion():
    layout = encode_layout((AffineRecord((2,), (1,), 0, (1,), 0),), source_size=2, output_size=2)
    result = execute_accumulation(jnp.asarray([100, -100], dtype=jnp.int8), (jnp.int8(2),), coefficient_records=(0,), layout=layout, output_size=2, dtype='int32')
    np.testing.assert_array_equal(result, [-56, 56])

@pytest.mark.parametrize('dtype', ['float16', 'bfloat16'])
def test_low_precision_product_rounding_precedes_widening(dtype):
    source = jnp.asarray([0.33325, 0.7, 1.01], dtype=dtype)
    factor = jnp.asarray(0.33325, dtype=dtype)
    layout = encode_layout((AffineRecord((3,), (1,), 0, (1,), 0),), source_size=3, output_size=3)
    expected = (source * factor).astype('float32')
    np.testing.assert_array_equal(execute_accumulation(source, (factor,), coefficient_records=(0,), layout=layout, output_size=3, dtype='float32'), expected)

def test_zero_and_one_follow_accumulation_binding():
    with jax.enable_x64():
        layout = encode_layout((AffineRecord((3,), (1,), 0, (1,), 0),), source_size=3, output_size=3)
        source = jnp.asarray([-0.0, np.inf, np.nan], dtype=jnp.float32)
        result = np.asarray(execute_accumulation(source, (jnp.float32(0),), coefficient_records=(0,), layout=layout, output_size=3, dtype='float64'))
        np.testing.assert_array_equal(result, np.zeros(3))
        integers = jnp.asarray([16777217, -16777217, 3], dtype=jnp.int32)
        np.testing.assert_array_equal(execute_accumulation(integers, (jnp.float32(1),), coefficient_records=(0,), layout=layout, output_size=3, dtype='int64'), [16777217, -16777217, 3])

def test_real_coefficient_form_is_preserved():
    with jax.enable_x64():
        layout = encode_layout((AffineRecord((1,), (1,), 0, (1,), 0),), source_size=1, output_size=1)
        source = jnp.asarray([complex(1, np.inf)], dtype=jnp.complex64)
        real_product = np.asarray(execute_accumulation(source, (jnp.float32(2),), coefficient_records=(0,), layout=layout, output_size=1, dtype='complex128'))
        complex_product = np.asarray(execute_accumulation(source, (jnp.complex64(2),), coefficient_records=(0,), layout=layout, output_size=1, dtype='complex128'))
        assert real_product[0].real == 2 and np.isposinf(real_product[0].imag)
        assert np.isnan(complex_product[0].real) and np.isposinf(complex_product[0].imag)

def test_empty_source_initializes_real_output():
    layout = encode_layout((AffineRecord((0,), (1,), 0, (1,), 2),), source_size=0, output_size=4)
    result = execute_accumulation(jnp.zeros(0, dtype=jnp.int16), (jnp.float32(np.nan),), coefficient_records=(0,), layout=layout, output_size=4, dtype='float32')
    np.testing.assert_array_equal(result, np.zeros(4))

def test_unsupported_source_and_result_dtypes():
    layout = encode_layout((AffineRecord((1,), (1,), 0, (1,), 0),), source_size=1, output_size=1)
    with pytest.raises(Exception, match='unsupported scalar dtype'):
        execute_accumulation(jnp.ones(1, dtype=jnp.float8_e4m3fn), (jnp.float32(1),), coefficient_records=(0,), layout=layout, output_size=1, dtype='float32').block_until_ready()
    with pytest.raises(NotImplementedError, match='does not support'):
        execute_accumulation(jnp.ones(1), (jnp.float32(1),), coefficient_records=(0,), layout=layout, output_size=1, dtype=jnp.float8_e4m3fn)

TWO_RECORDS = (
    AffineRecord((2, 2), (1, 1), 0, (1, 1), 1),
    AffineRecord((0,), (1,), 5, (1,), 5),
    AffineRecord((2, 2), (-1, -1), 3, (1, -1), 2),
)

def two_record_reference(source, first, second):
    result = jnp.zeros((*source.shape[:-1], 5), source.dtype)
    for record, factor in ((TWO_RECORDS[0], first), (TWO_RECORDS[2], second)):
        if factor.shape == (1,):
            factor = factor.reshape(())
        for coordinates in np.ndindex(record.logical_shape):
            source_index = record.source_offset + sum((axis * stride for axis, stride in zip(coordinates, record.source_strides)))
            output_index = record.destination_offset + sum((axis * stride for axis, stride in zip(coordinates, record.destination_strides)))
            result = result.at[..., output_index].add(source[..., source_index] * factor)
    return result

def two_record_accumulation(source, first, second):
    return accumulation_p.bind(source, first, second, records=TWO_RECORDS, coefficient_records=(0, 2), output_size=5, dtype=source.dtype)

@pytest.mark.parametrize('dtype', ['float32', 'float64', 'complex64', 'complex128'])
@pytest.mark.parametrize('batch_shape', [(), (2,), (2, 3), (0,), (2, 0)])
@pytest.mark.parametrize('factor_shape', ['scalar', 'one', 'batch'])
def test_jvp_and_vjp(dtype, batch_shape, factor_shape):
    source = (jnp.arange(prod(batch_shape) * 5) % 5).astype(dtype).reshape((*batch_shape, 5))
    if dtype in ('complex64', 'complex128'):
        source = source + 1j * (source - 2)
    shape = () if factor_shape == 'scalar' else (1,) if factor_shape == 'one' else batch_shape
    first = jnp.full(shape, 2 + 1j if dtype in ('complex64', 'complex128') else 2, dtype=dtype)
    second = jnp.full(shape, 3 - 2j if dtype in ('complex64', 'complex128') else 3, dtype=dtype)
    arguments = (source, first, second)
    tangents = tuple((jnp.ones_like(value) for value in arguments))
    for actual, expected in zip(jax.jit(lambda *values: jax.jvp(two_record_accumulation, values, tangents))(*arguments), jax.jvp(two_record_reference, arguments, tangents)):
        np.testing.assert_array_equal(actual, expected)
    cotangent = jnp.full((*batch_shape, 5), 2 - 1j if dtype in ('complex64', 'complex128') else 2, dtype=dtype)
    for actual, expected in zip(jax.jit(jax.vjp(two_record_accumulation, *arguments)[1])(cotangent), jax.vjp(two_record_reference, *arguments)[1](cotangent)):
        np.testing.assert_array_equal(actual, expected)
    for active in range(3):

        def bound(function, value):
            values = list(arguments)
            values[active] = value
            return function(*values)
        np.testing.assert_array_equal(jax.vjp(lambda value: bound(two_record_accumulation, value), arguments[active])[1](cotangent)[0], jax.vjp(lambda value: bound(two_record_reference, value), arguments[active])[1](cotangent)[0])

@pytest.mark.parametrize('factor', [0.0, 1.0, 2.0])
def test_coefficient_derivative_uses_algebraic_formula(factor):
    source = jnp.asarray([1, 2, 3, 4, 5], jnp.float32)
    for function in (two_record_accumulation, two_record_reference):
        result = jax.grad(lambda coefficient: function(source, coefficient, jnp.float32(2)).sum())(jnp.float32(factor))
        assert result == 8

def test_unscaled_records_and_cross_kind_coefficient_derivative():
    source = jnp.arange(5, dtype=jnp.float32)

    def unscaled(data):
        return accumulation_p.bind(data, records=TWO_RECORDS, coefficient_records=(), output_size=5, dtype=data.dtype)
    np.testing.assert_array_equal(jax.grad(lambda data: unscaled(data).sum())(source), [1, 3, 3, 1, 0])
    np.testing.assert_array_equal(jax.grad(lambda data: accumulation_p.bind(data, jnp.int32(2), records=(TWO_RECORDS[0],), coefficient_records=(0,), output_size=5, dtype=data.dtype).sum())(source), [2, 4, 2, 0, 0])
    with jax.enable_x64():
        gradient = jax.grad(lambda factor: accumulation_p.bind(source, factor, records=TWO_RECORDS, coefficient_records=(0,), output_size=5, dtype=source.dtype).sum())(jnp.complex128(2))
        np.testing.assert_array_equal(gradient, 4 + 0j)
        assert gradient.dtype == jnp.complex128

def test_multirecord_dot_transpose():
    source = jnp.arange(5, dtype=jnp.float32)
    records = (TWO_RECORDS[0], TWO_RECORDS[0], TWO_RECORDS[1], TWO_RECORDS[2])

    def function(left, right):
        return dot_p.bind(left, right, records=records, conjugate_left=False)
    left_gradient, right_gradient = jax.grad(function, argnums=(0, 1))(source, source)
    expected_left = np.zeros(5, np.float32)
    expected_right = np.zeros(5, np.float32)
    for record in records:
        for coordinates in np.ndindex(record.logical_shape):
            left_index = record.source_offset + sum((axis * stride for axis, stride in zip(coordinates, record.source_strides)))
            right_index = record.destination_offset + sum((axis * stride for axis, stride in zip(coordinates, record.destination_strides)))
            expected_left[left_index] += float(source[right_index])
            expected_right[right_index] += float(source[left_index])
    np.testing.assert_array_equal(left_gradient, expected_left)
    np.testing.assert_array_equal(right_gradient, expected_right)

def test_reverse_product_uses_source_gradient_range():
    with jax.enable_x64():
        record = AffineRecord((1,), (1,), 0, (1,), 0)
        source = jnp.asarray([0.001], dtype=jnp.float64)
        factor = jnp.float16(1000)
        run = lambda values: accumulation_p.bind(values, factor, records=(record,), output_size=1, dtype=jnp.dtype('float16'), coefficient_records=(0,))
        np.testing.assert_array_equal(run(source), [1])
        actual = jax.jit(jax.vjp(run, source)[1])(jnp.asarray([1000], dtype=jnp.float16))[0]
        assert actual.dtype == source.dtype
        np.testing.assert_allclose(actual, [1000000], rtol=1e-14, atol=0)

@pytest.mark.parametrize('factor', [0, 1])
def test_zero_one_shortcuts_and_empty_records(factor):
    source = jnp.ones(1, dtype=jnp.float16)
    records = (AffineRecord((1,), (1,), 0, (1,), 0), AffineRecord((0,), (1,), 1, (1,), 1))
    run = lambda values: accumulation_p.bind(values, jnp.float32(factor), records=records, output_size=2, dtype=jnp.dtype('float32'), coefficient_records=(0,))
    cotangent = jnp.asarray([jnp.inf, jnp.nan], dtype=jnp.float32)
    actual = jax.linear_transpose(run, source)(cotangent)[0]
    if factor == 0:
        np.testing.assert_array_equal(actual, [0])
    else:
        assert jnp.isposinf(actual[0])
    _, tangent = jax.jvp(lambda values: run(jax.lax.stop_gradient(values)), (source,), (source,))
    np.testing.assert_array_equal(tangent, [0, 0])

MIXED_RECORDS = (
    AffineRecord((2, 2), (1, -1), 1, (1, 1), 1),
    AffineRecord((2, 2), (0, 1), 1, (1, -1), 2),
    AffineRecord((0,), (1,), 5, (1,), 5),
    AffineRecord((1,), (1,), 4, (1,), 4),
)

def mixed_accumulation(source, factors, dtype):
    return accumulation_p.bind(source, *factors, records=MIXED_RECORDS, output_size=6, dtype=jnp.dtype(dtype), coefficient_records=(0, 1, 2))

def mixed_reference(source, factors, dtype):
    result = jnp.zeros((*source.shape[:-1], 6), dtype=dtype)
    for record, factor in zip(MIXED_RECORDS, (*factors, None), strict=True):
        for coordinates in np.ndindex(record.logical_shape):
            source_index = record.source_offset + sum((index * stride for index, stride in zip(coordinates, record.source_strides)))
            destination_index = record.destination_offset + sum((index * stride for index, stride in zip(coordinates, record.destination_strides)))
            value = source[..., source_index]
            if factor is not None:
                factor = factor.reshape(()) if factor.shape == (1,) else factor
                value = jnp.where(factor == 0, 0, jnp.where(factor == 1, value, value * factor))
            if not jnp.issubdtype(jnp.dtype(dtype), jnp.complexfloating):
                value = jnp.real(value)
            result = result.at[..., destination_index].add(value.astype(dtype))
    return result

@pytest.mark.parametrize('source_dtype,result_dtype', [('float16', 'complex64'), ('complex64', 'float16')])
def test_low_precision_cross_kind_source_ad(source_dtype, result_dtype):
    source = jnp.ones(5, dtype=source_dtype)
    run = lambda values: mixed_accumulation(values, (jnp.float32(2),) * 3, result_dtype)
    oracle = lambda values: mixed_reference(values, (jnp.float32(2),) * 3, result_dtype)
    for actual, expected in zip(jax.jvp(run, (source,), (source,)), jax.jvp(oracle, (source,), (source,)), strict=True):
        np.testing.assert_array_equal(actual, expected)

def test_outer_vmap_and_nested_derivatives():
    source = (jnp.arange(15, dtype=jnp.float32) / 4).astype(jnp.float16).reshape(3, 5)
    factors = jnp.asarray([0, 1, -2], dtype=jnp.float32)
    run = jax.vmap(lambda values, factor: mixed_accumulation(values, (factor, jnp.float32(1.5), jnp.float32(0)), 'float32'))
    oracle = jax.vmap(lambda values, factor: mixed_reference(values, (factor, jnp.float32(1.5), jnp.float32(0)), 'float32'))
    np.testing.assert_allclose(jax.jit(run)(source, factors), oracle(source, factors), rtol=0.001, atol=0.001)
    cotangent = jnp.ones((3, 6), dtype=jnp.float32)
    reverse = lambda values: jax.linear_transpose(lambda inputs: run(inputs, factors), source)(values)[0]
    expected = jax.vjp(lambda inputs: oracle(inputs, factors), source)[1](cotangent)[0]
    np.testing.assert_allclose(jax.jit(reverse)(cotangent), expected, rtol=0.001, atol=0.001)
    np.testing.assert_allclose(jax.jit(jax.linear_transpose(reverse, cotangent))(jnp.ones_like(source))[0], oracle(jnp.ones_like(source), factors), rtol=0.001, atol=0.001)
    actual_hessian = jax.jacfwd(jax.grad(lambda values: jnp.sum(run(values, factors) ** 2)))(source)
    expected_hessian = jax.jacfwd(jax.grad(lambda values: jnp.sum(oracle(values, factors) ** 2)))(source)
    np.testing.assert_allclose(actual_hessian, expected_hessian, rtol=0.001, atol=0.001)

def test_project_after_multiply_and_coefficient_gradient_dtype():
    record = AffineRecord((2,), (0,), 0, (0,), 1)

    def run(source, factor):
        return accumulation_p.bind(source, factor, records=(record,), coefficient_records=(0,), output_size=3, dtype=jnp.dtype('complex128'))
    source, factor = (jnp.float32(2).reshape(1), jnp.complex128(3 + 4j))
    cotangent = jnp.asarray([0, 1 + 2j, 0], jnp.complex128)
    source_gradient, factor_gradient = jax.jit(jax.vjp(run, source, factor)[1])(cotangent)
    np.testing.assert_array_equal(source_gradient, [-10])
    np.testing.assert_array_equal(factor_gradient, 4 + 8j)
    assert factor_gradient.dtype == factor.dtype
    text = jax.jit(lambda data, alpha, value: jax.vjp(run, data, alpha)[1](value)).lower(source, factor, cotangent).as_text()
    assert 'tensor0_stride_accumulation_' in text and 'tensor0_stride_dot_' in text
    for name in ('stablehlo.real', 'stablehlo.gather', 'stablehlo.scatter', 'tensor0_stride_update'):
        assert name not in text

@pytest.mark.parametrize('source_dtype,dtype', [('float32', 'complex128'), ('complex64', 'float64')])
def test_unscaled_empty_and_symbolic_zero(source_dtype, dtype):
    source = values((3,), source_dtype)
    for record in (AffineRecord((2, 2), (1, 1), 0, (1, 1), 1), AffineRecord((0,), (1,), 3, (1,), 5)):
        run = lambda data: accumulation_p.bind(data, records=(record,), output_size=5, dtype=jnp.dtype(dtype))
        frozen = lambda data: run(jax.lax.stop_gradient(data))
        np.testing.assert_array_equal(jax.jvp(frozen, (source,), (source,))[1], jnp.zeros(5, dtype=dtype))
        result, pullback = jax.vjp(run, source)
        gradient = jax.jit(pullback)(jnp.ones_like(result))[0]
        np.testing.assert_array_equal(gradient, [1, 2, 1] if record.logical_shape != (0,) else [0, 0, 0])
