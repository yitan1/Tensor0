"""JAX AD update contracts."""

from itertools import product
from math import prod

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensor0._stride import StridedView, get_num_threads, scale, set_num_threads
from tensor0._stride._ffi._registration import operation_target
from tensor0._stride._jax import accumulation_p, update_p
from tensor0._stride._layout import AffineRecord

from tests.stride.support.availability import native_available
from tests.stride.support.data import FLOAT_PAIRS, INTEGER_PAIRS
from tests.stride.support.layouts import PARTIAL as RAW_UPDATE_PARTIAL
from tests.stride.support.oracles.affine import (
    MIXED as AFFINE_MIXED,
    assert_close as affine_assert_close,
    broadcast_record,
    native,
    reference,
    restore_threads,
    signed_record,
    values as affine_values,
)
from tests.stride.support.oracles.dispatch import (
    addresses,
    assert_close as dispatch_assert_close,
    assert_native as dispatch_assert_native,
    fresh,
    selected_reference,
)
from tests.stride.support.oracles.dtype_family import (
    assert_native as dtype_family_assert_native,
    assert_result,
    values as dtype_family_values,
)
from tests.stride.support.oracles.generic import (
    HALF_LAYOUTS,
    PARTITIONS,
    assert_close as generic_assert_close,
    assert_native as generic_assert_native,
    blocked_record,
    cast,
    dtype_values,
    mapped,
    reference_map,
)
from tests.stride.support.oracles.product_stages import PARTIAL as PRODUCT_STAGE_PARTIAL
from tests.stride.support.oracles.raw_update import reference_update
from tests.stride.support.oracles.scalar_paths import (
    PARTIAL as SCALAR_PATH_PARTIAL,
    _CASES,
    assert_close as scalar_path_assert_close,
    store,
)
from tests.stride.support.oracles.update import LAYOUTS as UPDATE_LAYOUTS, compare, update_functions
from tests.stride.support.samples import values as samples_values


@pytest.mark.skipif(not native_available(), reason='native CPU stride unavailable')
@pytest.mark.parametrize("source_dtype,result_dtype,factor", AFFINE_MIXED)
@pytest.mark.parametrize("broadcast", [False, True])
def test_signed_and_broadcast_batch_vmap_jvp_vjp(source_dtype, result_dtype, factor, broadcast):
    if broadcast:
        record, size = broadcast_record(broadcast_axes=(1,), destination_signs=(-1, 1))
    else:
        record = signed_record(physical_shape=(4, 3), destination_fastest=(0, 1), destination_signs=(-1, 1))
        size = 12
    source = affine_values(source_dtype, size)
    tangent = (source * .25 + .5).astype(source.dtype)
    batch = jnp.stack((source, -source, 2 * source))
    execute = lambda value: native(value, record, factor, result_dtype)
    oracle = lambda value: reference(value, record, factor, result_dtype)
    affine_assert_close(jax.jit(execute)(batch), oracle(batch))
    affine_assert_close(jax.jit(jax.vmap(execute))(batch), oracle(batch))
    affine_assert_close(jax.jvp(execute, (source,), (tangent,))[1], jax.jvp(oracle, (source,), (tangent,))[1])
    cotangent = affine_values(result_dtype, prod(record.logical_shape))
    affine_assert_close(jax.vjp(execute, source)[1](cotangent)[0], jax.vjp(oracle, source)[1](cotangent)[0])


@pytest.mark.skipif(not native_available(), reason='native CPU stride unavailable')
def test_large_repeated_source_forward_and_vjp_accumulate():
    rows, columns = 1024, 512
    record = AffineRecord((rows, columns), (1, 1), 0, (columns, 1), 0)
    source = affine_values("float32", rows + columns - 1)
    addresses = np.arange(rows)[:, None] + np.arange(columns)[None, :]
    expected = (source[addresses] * jnp.float32(1.25)).reshape(-1)
    counts = np.bincount(addresses.ravel(), minlength=source.size)
    execute = lambda value: native(value, record, 1.25, "float32")
    compiled = jax.jit(execute)
    pullback = jax.jit(jax.vjp(execute, source)[1])
    previous = get_num_threads()
    try:
        for limit in (1, 4):
            set_num_threads(limit)
            affine_assert_close(compiled(source), expected)
            affine_assert_close(pullback(jnp.ones(rows * columns, dtype=jnp.float32))[0],
                         jnp.asarray(counts * 1.25, dtype=source.dtype))
    finally:
        restore_threads(previous)


GENERIC_MIXED = [(jnp.float16, jnp.float32, -1.25), (jnp.float32, jnp.complex64, 1.25 - .75j),
         (jnp.complex64, jnp.float32, -.75), (jnp.float64, jnp.complex128, .5 + .75j),
         (jnp.complex128, jnp.float64, -.75)]


@pytest.mark.skipif(not native_available(), reason='native CPU stride unavailable')
@pytest.mark.parametrize("source_dtype,result_dtype,factor", GENERIC_MIXED)
@pytest.mark.parametrize("shape", [(35,), (5, 7), (8, 7, 4)])
def test_mixed_mapping_batch_jvp_vjp(source_dtype, result_dtype, factor, shape):
    with jax.enable_x64():
        record = blocked_record(shape)
        source = dtype_values(source_dtype, prod(shape))
        coefficient = jnp.asarray(factor, dtype=jnp.complex128 if isinstance(factor, complex) else jnp.float64)
        operation = lambda value: mapped(value, record, coefficient, result_dtype, source.size)
        reference = lambda value: reference_map(value, record, coefficient, result_dtype, source.size)
        direction = jnp.ones_like(source) * .25
        cotangent = dtype_values(result_dtype, source.size) * .5
        actual = jax.jit(lambda value, tangent, cot: (
            jax.jvp(operation, (value,), (tangent,)), jax.vjp(operation, value)[1](cot)[0]))(source, direction, cotangent)
        expected = (jax.jvp(reference, (source,), (direction,)), jax.vjp(reference, source)[1](cotangent)[0])
        for result, wanted in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
            generic_assert_close(result, wanted)
        batch = jnp.stack((source, source * .5, -source))
        generic_assert_close(jax.jit(jax.vmap(operation))(batch), reference(batch))


@pytest.mark.skipif(not native_available(), reason='native CPU stride unavailable')
@pytest.mark.parametrize("dtype", [jnp.float16, jnp.float32])
@pytest.mark.parametrize("record,size", [(HALF_LAYOUTS[0][0], 257),
                                        (AffineRecord((64, 128), (1, 64), 0, (128, 1), 0), 8192)])
def test_half_forward_and_reverse_with_batching(dtype, record, size):
    source = jnp.asarray(np.linspace(-1, 1, size, dtype=np.float16))
    operation = lambda value: mapped(value, record, jnp.float32(-1.25), dtype, size)
    reference = lambda value: reference_map(value, record, jnp.float32(-1.25), dtype, size)
    direction, cotangent = source[::-1], jnp.linspace(-2, 3, size, dtype=dtype)
    for result, wanted in zip(jax.jvp(operation, (source,), (direction,)),
                              jax.jvp(reference, (source,), (direction,)), strict=True):
        generic_assert_close(result, wanted)
    generic_assert_close(jax.jit(lambda cot: jax.vjp(operation, source)[1](cot)[0])(cotangent),
                 jax.vjp(reference, source)[1](cotangent)[0])
    batch = jnp.stack((source, source * .5))
    generic_assert_close(jax.jit(jax.vmap(operation))(batch), reference(batch))


@pytest.mark.skipif(not native_available(), reason='native CPU stride unavailable')
@pytest.mark.parametrize("source_dtype,result_dtype", [
    (jnp.float16, jnp.float32), (jnp.float32, jnp.complex64), (jnp.complex64, jnp.float32),
])
def test_mixed_multirecord_batch_derivatives(source_dtype, result_dtype):
    raw = jnp.arange(48, dtype=jnp.float32).reshape(3, 16)
    source = (raw * (1 + .25j) if source_dtype == jnp.complex64 else raw).astype(source_dtype)
    coefficients = (jnp.float32(1.25), jnp.float32(-.5))
    operation = lambda value: accumulation_p.bind(value, *coefficients, records=PARTITIONS,
        coefficient_records=(0, 1), output_size=16, dtype=jnp.dtype(result_dtype))

    def reference(value):
        blocks = value.reshape(*value.shape[:-1], 4, 4)
        selected = jnp.concatenate((blocks[..., 2:] * coefficients[1], blocks[..., :2] * coefficients[0]), axis=-1)
        return cast(selected.reshape(value.shape), result_dtype)

    direction = source / 8
    cotangent = jnp.ones(source.shape, dtype=result_dtype)
    if result_dtype == jnp.complex64:
        cotangent = cotangent * jnp.complex64(1 + .25j)
    actual = jax.jit(lambda value, tangent, cot: (
        jax.jvp(operation, (value,), (tangent,)), jax.vjp(operation, value)[1](cot)[0]))(source, direction, cotangent)
    expected = (jax.jvp(reference, (source,), (direction,)), jax.vjp(reference, source)[1](cotangent)[0])
    for result, wanted in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        generic_assert_close(result, wanted)
    generic_assert_native(jax.jit(operation).lower(source))


@pytest.mark.skipif(not native_available(), reason='native CPU stride unavailable')
@pytest.mark.parametrize("explicit", [False, True])
def test_half_storage_derivatives_do_not_narrow_factor(explicit):
    source = jnp.ones(1, dtype=jnp.float16)
    factor = jnp.float32(1e-8)
    direction = jnp.full_like(source, 65504)
    operation = lambda values, coefficient: scale(StridedView(values, (1,), (1,), 0), coefficient).data

    def derivative(values, coefficient):
        if explicit:
            tangent = jax.jvp(operation, (values, coefficient), (direction, jnp.zeros_like(coefficient)))[1]
        else:
            tangent = jax.jvp(lambda data: operation(data, coefficient), (values,), (direction,))[1]
        cotangent = jax.vjp(lambda data: operation(data, coefficient), values)[1](direction)[0]
        return tangent, cotangent

    for actual in jax.jit(derivative)(source, factor):
        dispatch_assert_close(actual, (factor * direction).astype(source.dtype))
        assert bool(jnp.all(actual != 0))


@pytest.mark.skipif(not native_available(), reason='native CPU stride unavailable')
@pytest.mark.parametrize("layout", ["contiguous", "forward", "reverse"])
def test_half_permutation_tails_and_zero_cotangents(layout):
    shape = (11, 19)
    record = AffineRecord(shape, (1, 11) if layout == "forward" else (19, 1), 0,
                          (1, 11) if layout == "reverse" else (19, 1), 0)
    source = jnp.resize(jnp.asarray([-0., 1., -2., .125, -3.5, 128., 65504], dtype=jnp.float16), (209,))
    factor = jnp.float16(.75)
    operation = lambda values: fresh(values, record, factor, 209, values.dtype)
    _, destinations = addresses(record)
    expected = jnp.zeros_like(source).at[destinations].set(selected_reference(source, record, factor, source.dtype))
    dispatch_assert_close(jax.jit(operation)(source), expected)
    pullback = jax.jit(lambda values, cot: jax.vjp(operation, values)[1](cot)[0])
    np.testing.assert_array_equal(pullback(source, jnp.full_like(source, -0.)), jnp.zeros_like(source))


@pytest.mark.skipif(not native_available(), reason='native CPU stride unavailable')
@pytest.mark.parametrize("dtype", ["float16", "complex64"])
@pytest.mark.parametrize("factor_value", [0., 1.])
def test_data_derivative_zero_one_scaling(dtype, factor_value):
    source = jnp.ones(153, dtype=dtype)
    direction = jnp.full_like(source, 1.25 - .5j if dtype == "complex64" else 1.25)
    coefficient = jnp.asarray(factor_value, dtype=dtype)
    derivative = lambda values, factor, tangent: jax.jvp(
        lambda data: scale(StridedView(data, (153,), (1,), 0), factor).data, (values,), (tangent,))[1]
    lowered = jax.jit(derivative).lower(source, coefficient, direction)
    dispatch_assert_native(lowered)
    dispatch_assert_close(lowered.compile()(source, coefficient, direction), coefficient * direction)


@pytest.mark.parametrize("source_dtype,result_dtype", [(jnp.float32, jnp.float32),
                                                      (jnp.float16, jnp.float32)])
@pytest.mark.parametrize("source_factor,base_factor", [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_explicit_coefficient_composition_keeps_original_derivatives(
    source_dtype, result_dtype, source_factor, base_factor,
):
    records = (AffineRecord((2,), (1,), 0, (2,), 1),)
    base = jnp.arange(5, dtype=result_dtype)
    source = jnp.asarray([3, 5], dtype=source_dtype)
    function = lambda old, new, lhs, rhs: update_p.bind(
        new, old, lhs * 2, rhs, records=records,
    )
    oracle = lambda old, new, lhs, rhs: old.at[1::2].set(
        (lhs * 2) * new + rhs * old[1::2],
    )
    primals = (base, source, jnp.asarray(source_factor, dtype=result_dtype),
               jnp.asarray(base_factor, dtype=result_dtype))
    tangents = tuple(jnp.ones_like(value) for value in primals)
    actual = jax.jit(lambda *values: jax.jvp(function, values, tangents))(*primals)
    expected = jax.jvp(oracle, primals, tangents)
    for observed, reference in zip(actual, expected, strict=True):
        np.testing.assert_allclose(observed, reference)
    actual_vjp = jax.jit(jax.grad(lambda *values: function(*values).sum(),
                                  argnums=(0, 1, 2, 3)))(*primals)
    expected_vjp = jax.grad(lambda *values: oracle(*values).sum(),
                            argnums=(0, 1, 2, 3))(*primals)
    for observed, reference in zip(actual_vjp, expected_vjp, strict=True):
        np.testing.assert_allclose(observed, reference)
    second = jax.jacfwd(jax.grad(lambda new, factor:
                                function(base, new, factor, primals[3]).sum(), 1), 0)
    np.testing.assert_allclose(second(source, primals[2]), jnp.full((2,), 2))
    jax.jit(function)(*primals).block_until_ready()
    lowered = jax.jit(function).lower(*primals).as_text()
    assert lowered.count("stablehlo.custom_call") == 1
    assert "tensor0_stride_update_f32_cpu_v1" in lowered


def test_broadcast_source_transpose_sums_repeated_reads():
    records = (AffineRecord((3,), (0,), 0, (1,), 1),)
    base = jnp.arange(5, dtype=jnp.float32)
    source = jnp.asarray([3.0])
    function = lambda old, new, factor: update_p.bind(
        new, old, factor * 2, jnp.int32(0), records=records,
    )
    gradients = jax.grad(lambda old, new, factor: function(old, new, factor).sum(),
                        argnums=(0, 1, 2))(base, source, jnp.asarray(1.0))
    np.testing.assert_array_equal(gradients[0], [1, 0, 0, 0, 1])
    np.testing.assert_array_equal(gradients[1], [6])
    np.testing.assert_array_equal(gradients[2], 18)


def test_mixed_dtype_direct_coefficient_transpose():
    records = (AffineRecord((2,), (1,), 0, (1,), 0),)
    base = jnp.ones(2, dtype=jnp.float32)
    source = jnp.asarray([3, 5], dtype=jnp.float16)
    function = lambda factor: update_p.bind(
        source, base, factor, jnp.int32(0), records=records,
    )
    np.testing.assert_array_equal(jax.linear_transpose(function, jnp.asarray(0.))(
        jnp.ones_like(base))[0], 8)


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
@pytest.mark.parametrize("source_dtype,result_dtype,factor", [
    (jnp.float64, jnp.float64, -1.25), (jnp.complex128, jnp.complex128, 1.25 - .75j),
    (jnp.float64, jnp.complex128, .5 + .75j), (jnp.complex128, jnp.float64, -.75),
])
def test_wide_broadcast_vjp_accumulates_without_hidden_narrowing(source_dtype, result_dtype, factor):
    with jax.enable_x64():
        source = dtype_family_values(source_dtype, 3) + jnp.asarray(2**-40, dtype=source_dtype)
        cotangent = dtype_family_values(result_dtype, 12, shift=5) + jnp.asarray(2**-39, dtype=result_dtype)
        coefficient = jnp.asarray(factor, dtype=result_dtype)
        record = AffineRecord((4, 3), (0, 1), 0, (3, 1), 0)
        native = lambda value: accumulation_p.bind(value, coefficient, records=(record,), coefficient_records=(0,),
                                                   output_size=12, dtype=np.dtype(result_dtype))

        def reference(value):
            result = jnp.broadcast_to(value, (4, 3)) * coefficient
            if result_dtype == jnp.float64:
                result = jnp.real(result)
            return result.astype(result_dtype).ravel()

        assert_result(jax.jit(native)(source), reference(source))
        pullback = jax.jit(lambda cot: jax.vjp(native, source)[1](cot)[0])
        assert_result(pullback(cotangent), jax.vjp(reference, source)[1](cotangent)[0])
        dtype_family_assert_native(pullback.lower(cotangent), "accumulation", source_dtype)


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
@pytest.mark.parametrize("factor", [1.25 - .75j, .125 + 1j, complex(2.5, -0.), 3 + 3j])
def test_complex128_finite_values_and_bilinear_transpose(factor):
    with jax.enable_x64():
        source = jnp.asarray([0., -0., 1.25, -2.5, .125, 2., -2.], dtype=jnp.float64)
        cotangent = jnp.asarray([complex(-0., -0.), 0j, 1.25, 2.5j, .125 + 1j, 3 + 3j, -3 - 3j], dtype=jnp.complex128)
        coefficient = jnp.asarray(factor, dtype=jnp.complex128)
        record = AffineRecord((7,), (1,), 0, (1,), 0)
        native = lambda value: accumulation_p.bind(value, coefficient, records=(record,), coefficient_records=(0,),
                                                   output_size=7, dtype=np.dtype(jnp.complex128))
        reference = lambda value: value * coefficient
        actual, gradient = jax.jit(lambda value, cot: (native(value), jax.vjp(native, value)[1](cot)[0]))(source, cotangent)
        assert_result(actual, reference(source))
        assert_result(gradient, jax.vjp(reference, source)[1](cotangent)[0])
        assert_result(jax.jit(native)(cotangent), reference(cotangent))


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
@pytest.mark.parametrize("source_dtype,result_dtype,coefficient_dtype", _CASES)
def test_native_computation_explicit_differentials(source_dtype, result_dtype, coefficient_dtype):
    with jax.enable_x64():
        source = jnp.asarray([1.25, -2.5, 3.75], dtype=source_dtype)
        base = jnp.arange(7, dtype=jnp.float32).astype(result_dtype)
        first, second = jnp.asarray(.75, dtype=coefficient_dtype), jnp.asarray(.5, dtype=coefficient_dtype)
        def operation(old, values, alpha, beta):
            return update_p.bind(values, old, alpha, beta, records=SCALAR_PATH_PARTIAL)
        def reference(old, values, alpha, beta):
            return store(old, alpha * values + beta * old[1::2])
        arguments = (base, source, first, second)
        directions = tuple(jnp.ones_like(value) for value in arguments)
        def derivative(function, *values):
            return jax.jvp(function, values, directions)[1]
        actual = jax.jit(lambda *values: derivative(operation, *values))(*arguments)
        expected = derivative(reference, *arguments)
        scalar_path_assert_close(actual, expected)
        for actual_value, expected_value in zip(
            jax.vjp(operation, *arguments)[1](jnp.ones_like(base)),
            jax.vjp(reference, *arguments)[1](jnp.ones_like(base)), strict=True,
        ):
            scalar_path_assert_close(actual_value, expected_value)
        actual = jax.jvp(lambda *values: derivative(operation, *values), arguments, directions)[1]
        expected = jax.jvp(lambda *values: derivative(reference, *values), arguments, directions)[1]
        scalar_path_assert_close(actual, expected)


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
@pytest.mark.parametrize("mode", ["vjp", "linear_transpose"])
def test_selected_scale_multirecord_factor_gradient_uses_native_dot(mode):
    records = (AffineRecord((3,), (2,), 0, (2,), 0), AffineRecord((2,), (3,), 6, (3,), 6))
    base = jnp.linspace(-2, 3, 12, dtype=jnp.float32)
    factor = jnp.float32(1.25)
    cotangent = jnp.linspace(4, -1, 12, dtype=jnp.float32)
    operation = lambda value: update_p.bind(base, base, value, jnp.int32(0), records=records)
    if mode == "vjp":
        pullback = jax.jit(lambda cot: jax.vjp(operation, factor)[1](cot)[0])
    else:
        pullback = jax.jit(lambda cot: jax.linear_transpose(operation, factor)(cot)[0])
    indices = jnp.asarray([0, 2, 4, 6, 9])
    np.testing.assert_allclose(pullback(cotangent), jnp.sum(cotangent[indices] * base[indices]), rtol=2e-6)
    text = pullback.lower(cotangent).as_text()
    count = len(records) if mode == "vjp" else 1
    assert text.count("custom_call") == count
    assert text.count(operation_target("dot", np.dtype(jnp.float32))) == count
    assert "gather" not in text and "scatter" not in text


@pytest.mark.parametrize("beta", [0, 1])
@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("transpose_layout", [False, True])
def test_update_jvp_vjp_and_linear_transpose(beta, mixed, transpose_layout):
    records = (AffineRecord((17, 13), (1, 17), 0, (13, 1), 0),) if transpose_layout else RAW_UPDATE_PARTIAL
    source_size, output_size = (221, 221) if transpose_layout else (32, 50)
    source = jnp.linspace(-2, 2, source_size)
    base = jnp.linspace(-3, 4, output_size)
    if mixed:
        base = base + 1j * base[::-1]
    directions = (jnp.ones_like(base) * .5, jnp.linspace(-1, 1, source_size))
    cotangent = jnp.linspace(-4, 3, output_size).astype(base.dtype)
    if mixed:
        cotangent = cotangent + 1j * cotangent[::-1]
    operation = lambda old, new: update_p.bind(new, old, jnp.float32(.5), jnp.int32(beta), records=records)
    reference = lambda old, new: reference_update(new, old, records, .5, beta)
    actual_jvp = jax.jit(lambda old, new: jax.jvp(operation, (old, new), directions))(base, source)
    expected_jvp = jax.jvp(reference, (base, source), directions)
    for actual, expected in zip(actual_jvp, expected_jvp, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-6)
    actual_vjp = jax.jit(jax.vjp(operation, base, source)[1])(cotangent)
    expected_vjp = jax.vjp(reference, base, source)[1](cotangent)
    actual_transpose = jax.jit(jax.linear_transpose(operation, base, source))(cotangent)
    for actual, expected, transposed in zip(actual_vjp, expected_vjp, actual_transpose, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-6)
        np.testing.assert_allclose(transposed, expected, rtol=2e-6, atol=1e-6)
    if beta:
        np.testing.assert_array_equal(actual_vjp[0], cotangent)
    else:
        selected = actual_vjp[0] if transpose_layout else actual_vjp[0][2:50:3]
        np.testing.assert_array_equal(selected, jnp.zeros_like(selected))


MULTIRECORD_LAYOUTS = (
    (AffineRecord((3,), (1,), 0, (2,), 1), AffineRecord((3,), (1,), 0, (2,), 2)),
    (AffineRecord((2, 2), (1, 1), 0, (4, 1), 1), AffineRecord((2,), (0,), 3, (-4,), 7)),
    (AffineRecord((3,), (-1,), 5, (-2,), 6), AffineRecord((2,), (1,), 0, (2,), 1)),
    (AffineRecord((0,), (1,), 6, (1,), 10),),
)


def multirecord_update_functions(records):
    indices = []
    destinations = []
    for record in records:
        pairs = [(
            record.source_offset + sum(index * stride for index, stride in zip(coordinate, record.source_strides)),
            record.destination_offset + sum(index * stride for index, stride in zip(coordinate, record.destination_strides)),
        ) for coordinate in np.ndindex(record.logical_shape)]
        indices.append((jnp.asarray([source for source, _ in pairs], dtype=jnp.int32),
                        jnp.asarray([destination for _, destination in pairs], dtype=jnp.int32)))
        destinations.extend(destination for _, destination in pairs)
    assert len(destinations) == len(set(destinations))

    def execute(source, base, alpha, beta):
        return update_p.bind(source, base, alpha, beta, records=records)

    def reference(source, base, alpha, beta):
        alpha = alpha[..., None] if alpha.ndim else alpha
        beta = beta[..., None] if beta.ndim else beta
        result = base
        for source_indices, destination_indices in indices:
            contribution = alpha * source[..., source_indices] + beta * base[..., destination_indices]
            result = result.at[..., destination_indices].set(contribution.astype(base.dtype))
        return result

    return execute, reference


def operands(dtype, batch_shape):
    source = (jnp.arange(prod(batch_shape) * 6) % 5).astype(dtype).reshape((*batch_shape, 6))
    base = (jnp.arange(prod(batch_shape) * 10) % 7).astype(dtype).reshape((*batch_shape, 10))
    if dtype == "complex64":
        source = source + 1j * (source - 2)
        base = base + 1j * (3 - base)
    return source, base, jnp.asarray(2, dtype=dtype), jnp.full(batch_shape, 3, dtype=dtype)


def check_joint_derivatives(execute, reference, arguments):
    directions = tuple(jnp.ones_like(value) for value in arguments)
    actual = jax.jit(lambda *values: jax.jvp(execute, values, directions))(*arguments)
    expected = jax.jvp(reference, arguments, directions)
    for observed, wanted in zip(actual, expected, strict=True):
        np.testing.assert_allclose(observed, wanted, rtol=2e-6, atol=1e-6)
    cotangent = arguments[1] + 1
    actual_gradient = jax.jit(jax.vjp(execute, *arguments)[1])(cotangent)
    expected_gradient = jax.vjp(reference, *arguments)[1](cotangent)
    for observed, wanted in zip(actual_gradient, expected_gradient, strict=True):
        np.testing.assert_allclose(observed, wanted, rtol=2e-6, atol=1e-6)


@pytest.mark.parametrize("records", MULTIRECORD_LAYOUTS)
@pytest.mark.parametrize("dtype", ["float32", "complex64"])
@pytest.mark.parametrize("batch_shape", [(), (2, 3), (0,)])
def test_multirecord_update_joint_jvp_vjp(records, dtype, batch_shape):
    check_joint_derivatives(*multirecord_update_functions(records), operands(dtype, batch_shape))


@pytest.mark.parametrize("dtype", ["float32", "complex64"])
def test_sequential_updates_with_overlapping_selections(dtype):
    first, reference_first = multirecord_update_functions((AffineRecord((3,), (1,), 0, (1,), 1),))
    second, reference_second = multirecord_update_functions((AffineRecord((3,), (-1,), 5, (1,), 2),))

    def execute(source, base, alpha, beta):
        return second(source, first(source, base, alpha, beta), alpha, beta)

    def reference(source, base, alpha, beta):
        return reference_second(source, reference_first(source, base, alpha, beta), alpha, beta)

    check_joint_derivatives(execute, reference, operands(dtype, (2, 3)))


@pytest.mark.parametrize("active,target", [(0, "accumulation"), (1, "update"), (2, "dot"), (3, "dot")])
def test_update_transpose_uses_existing_native_operations(active, target):
    arguments = operands("float32", ())
    execute, reference = multirecord_update_functions(MULTIRECORD_LAYOUTS[0])

    def bind(function, value):
        values = list(arguments)
        values[active] = value
        return function(*values)

    transpose = jax.jit(jax.linear_transpose(lambda value: bind(execute, value), arguments[active]))
    expected = jax.vjp(lambda value: bind(reference, value), arguments[active])[1](arguments[1])[0]
    np.testing.assert_allclose(transpose(arguments[1])[0], expected, rtol=2e-6, atol=1e-6)
    lowered = transpose.lower(arguments[1]).as_text()
    assert f"tensor0_stride_{target}_f32_cpu_v1" in lowered
    assert "stablehlo.gather" not in lowered
    assert "stablehlo.scatter" not in lowered


@pytest.fixture(autouse=False)
def enable_x64():
    with jax.enable_x64():
        yield


@pytest.mark.usefixtures("enable_x64")
@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
def test_source_gradient_preserves_record_accumulation_order():
    records = tuple((AffineRecord((), (), 0, (), destination) for destination in range(3)))
    run = lambda data: update_p.bind(data, jnp.zeros(3, dtype=jnp.float32), jnp.float32(1), jnp.float32(0), records=records)
    cotangent = jnp.asarray([2 ** 24, 1, -2 ** 24], dtype=jnp.float32)
    np.testing.assert_array_equal(jax.linear_transpose(run, jnp.zeros(1, dtype=jnp.float32))(cotangent)[0], [0])


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


@pytest.mark.usefixtures("enable_x64")
@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
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


@pytest.mark.usefixtures("enable_x64")
@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
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


@pytest.mark.usefixtures("enable_x64")
@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
def test_joint_bilinear_transpose_remains_rejected():
    source, base, alpha, beta = single_record_operands('float32', (), ((), ()))
    execute, _ = single_record_functions(update_coefficient_ad_RECORDS[0])
    with pytest.raises(NotImplementedError, match='known operand'):
        jax.linear_transpose(lambda inputs, factor: execute(inputs, base, factor, beta), source, alpha)(base)
    with pytest.raises(NotImplementedError, match='known operand'):
        jax.linear_transpose(lambda inputs, factor: execute(source, inputs, alpha, factor), base, beta)(base)


@pytest.mark.usefixtures("enable_x64")
@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
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


@pytest.mark.usefixtures("enable_x64")
@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
@pytest.mark.parametrize('dtype,coefficient_dtype', [('float32', 'complex128'), ('complex64', 'float64')])
def test_sequential_updates_and_vmap(dtype, coefficient_dtype):
    first, first_reference = update_functions(UPDATE_LAYOUTS[0])
    second, second_reference = update_functions(UPDATE_LAYOUTS[2])
    run = lambda source, base, alpha, beta: second(source, first(source, base, alpha, beta), beta, alpha)
    reference = lambda source, base, alpha, beta: second_reference(source, first_reference(source, base, alpha, beta), beta, alpha)
    arguments = (samples_values((2, 6), dtype), samples_values((2, 10), dtype), jnp.full((2,), 2, coefficient_dtype), jnp.full((2,), 0.5, coefficient_dtype))
    run, reference = (jax.vmap(run), jax.vmap(reference))
    directions = tuple((jnp.ones_like(value) for value in arguments))
    for actual, expected in zip(jax.jit(lambda *inputs: jax.jvp(run, inputs, directions))(*arguments), jax.jvp(reference, arguments, directions), strict=True):
        np.testing.assert_allclose(actual, expected, rtol=3e-06, atol=3e-06)
    cotangent = jnp.ones_like(arguments[1])
    for actual, expected, original in zip(jax.jit(jax.vjp(run, *arguments)[1])(cotangent), jax.vjp(reference, *arguments)[1](cotangent), arguments, strict=True):
        assert actual.dtype == original.dtype and actual.shape == original.shape
        np.testing.assert_allclose(actual, expected, rtol=3e-06, atol=3e-06)


DTYPES = ('float32', 'float64', 'complex64', 'complex128')


# Fixed forward representatives retain same-kind and both mixed-kind coefficients.
# Wide storage keeps nonempty/scalar AD; empty AD uses narrow/half representatives.
JOINT_AD_COEFFICIENTS = [('float32', 'float32'), ('complex128', 'complex128'),
                         ('float32', 'complex128'), ('complex128', 'float64')]


JOINT_FORWARD_COEFFICIENTS = [('float64', 'float64'), ('complex64', 'complex64'),
                              ('float64', 'complex64'), ('complex64', 'float32')]


JOINT_AD_CASES = [
    (*storage, alpha, beta, (2, 3))
    for storage in FLOAT_PAIRS[1:] for alpha, beta in product(DTYPES, repeat=2)
    if (alpha, beta) in JOINT_AD_COEFFICIENTS + JOINT_FORWARD_COEFFICIENTS
] + [
    (*storage, alpha, beta, batch)
    for storage in FLOAT_PAIRS[1:]
    for alpha, beta in (('float32', 'complex128'), ('complex128', 'float64'))
    for batch in ((), (0,), (2, 0))
    if not batch or storage in (FLOAT_PAIRS[1], FLOAT_PAIRS[2], FLOAT_PAIRS[5])
]


@pytest.mark.usefixtures("enable_x64")
@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
@pytest.mark.parametrize('source_dtype,base_dtype,alpha_dtype,beta_dtype,batch_shape', JOINT_AD_CASES)
def test_joint_input_and_coefficient_contract(source_dtype, base_dtype, alpha_dtype, beta_dtype, batch_shape):
    run, reference = update_functions(UPDATE_LAYOUTS[1])
    arguments = (samples_values((*batch_shape, 6), source_dtype), samples_values((*batch_shape, 10), base_dtype), jnp.asarray(2 + 1j if alpha_dtype.startswith('complex') else 2, alpha_dtype), jnp.full(batch_shape, 0.5 - 1j if beta_dtype.startswith('complex') else 0.5, beta_dtype))
    if (alpha_dtype, beta_dtype) in JOINT_AD_COEFFICIENTS:
        compare(run, reference, arguments)
    else:
        actual, expected = jax.jit(run)(*arguments), reference(*arguments)
        assert actual.dtype == expected.dtype and actual.shape == expected.shape
        np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=3e-6)


@pytest.mark.usefixtures("enable_x64")
@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
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


@pytest.mark.usefixtures("enable_x64")
@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
def test_low_precision_coefficient_ad():
    source, base = (jnp.ones(1, jnp.float16), jnp.ones(1, jnp.float32))
    record = AffineRecord((1,), (1,), 0, (1,), 0)
    run = lambda factor: update_p.bind(source, base, factor, jnp.float32(1), records=(record,))
    primal, tangent = jax.jvp(run, (jnp.float32(2),), (jnp.float32(1),))
    np.testing.assert_array_equal(primal, [3])
    np.testing.assert_array_equal(tangent, [1])
    np.testing.assert_array_equal(jax.linear_transpose(run, jnp.float32(2))(base)[0], 1)


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


@pytest.mark.usefixtures("enable_x64")
@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
@pytest.mark.parametrize('source_dtype,base_dtype', STORAGE_PAIRS)
@pytest.mark.parametrize('coefficient_dtype', COEFFICIENTS)
def test_joint_primals(source_dtype, base_dtype, coefficient_dtype):
    source = jnp.asarray([1, 2, 3, 1, 2, 3], dtype=source_dtype)
    base = jnp.asarray([0, 1, 2, 3, 4, 0, 1, 2, 3, 4], dtype=base_dtype)
    alpha, beta = (jnp.asarray(2, dtype=coefficient_dtype), jnp.asarray(0.5, dtype=coefficient_dtype))
    run = lambda *values: update_p.bind(*values, records=UPDATE_LAYOUTS[1])
    check(run, (source, base, alpha, beta))


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
@pytest.mark.parametrize("source_is_narrow", [False, True])
@pytest.mark.parametrize("coefficient_values", [(0, .25), (1, .25), (.75, 1), (.75, .25)])
def test_mixed_product_stage_differentials(source_is_narrow, coefficient_values):
    with jax.enable_x64():
        source = jnp.asarray([1.25, -2.5, 3.75], dtype=jnp.float32)
        base = jnp.arange(7, dtype=jnp.float32)
        alpha = jnp.asarray(coefficient_values[0], dtype=jnp.float32 if source_is_narrow else jnp.float64)
        beta = jnp.asarray(coefficient_values[1], dtype=jnp.float64 if source_is_narrow else jnp.float32)
        def operation(old, values, first, second):
            return update_p.bind(values, old, first, second, records=PRODUCT_STAGE_PARTIAL)

        def reference(old, values, first, second):
            return old.at[1::2].set((first * values + second * old[1::2]).astype(old.dtype))

        arguments = (base, source, alpha, beta)
        directions = tuple(jnp.ones_like(value) for value in arguments)
        expected_jvp = jax.jvp(reference, arguments, directions)[1]
        expected_vjp = jax.vjp(reference, *arguments)[1](jnp.ones_like(base))
        expected_nested_jvp = jax.jvp(lambda *values: jax.jvp(reference, values, directions)[1], arguments, directions)[1]
        for function in (operation, jax.jit(operation)):
            actual = jax.jvp(function, arguments, directions)[1]
            np.testing.assert_allclose(actual, expected_jvp, rtol=2e-6, atol=1e-7)
            actual_vjp = jax.vjp(function, *arguments)[1](jnp.ones_like(base))
            for actual, expected in zip(actual_vjp, expected_vjp, strict=True):
                np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-7)
            actual = jax.jvp(lambda *values: jax.jvp(function, values, directions)[1], arguments, directions)[1]
            np.testing.assert_allclose(actual, expected_nested_jvp, rtol=2e-6, atol=1e-7)
