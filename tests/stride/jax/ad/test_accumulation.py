"""JAX AD accumulation contracts."""

from math import prod

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensor0._stride._jax import accumulation_p, copy_p
from tensor0._stride._layout import AffineRecord

from tests.stride.support.availability import native_available
from tests.stride.support.oracles.accumulation import TWO_RECORDS
from tests.stride.support.oracles.affine_ad import (
    PARTITIONS,
    assert_close as affine_ad_assert_close,
    assert_native_calls,
    reference_transform,
)
from tests.stride.support.oracles.compact_complex import (
    assert_close as compact_assert_close,
    assert_native as compact_assert_native,
    compact_record,
    complex_values,
    mapped as compact_mapped,
    reference,
)
from tests.stride.support.oracles.complex_kernels import (
    assert_close as complex_kernel_assert_close,
    assert_native as complex_kernel_assert_native,
    mapped as complex_kernel_mapped,
    values as complex_kernel_values,
)
from tests.stride.support.oracles.generic import (
    assert_close as generic_assert_close,
    dtype_values,
    mapped as generic_mapped,
    reference_map,
)
from tests.stride.support.samples import values as samples_values


@pytest.fixture(autouse=False)
def enable_x64():
    with jax.enable_x64():
        yield


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


@pytest.mark.usefixtures("enable_x64")
@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
# Float32 carries geometry; other types retain selected precision/complex crosses.
@pytest.mark.parametrize('dtype,batch_shape,factor_shape', [
    pytest.param(dtype, batch, form, id=f'{form}-batch_shape{index}-{dtype}')
    for form in ('scalar', 'one', 'batch')
    for index, batch in enumerate([(), (2,), (2, 3), (0,), (2, 0)])
    for dtype in ('float32', 'float64', 'complex64', 'complex128')
    if not (batch == () and form == 'batch')
    if dtype == 'float32' or (batch, form) in (
        (((), 'scalar'), ((2, 3), 'one'), ((2,), 'batch'),
         ((2, 3), 'batch'), ((2, 0), 'scalar'), ((2, 0), 'batch'))
        if dtype == 'complex64' else (((), 'scalar'), ((2, 3), 'batch'))
    )
])
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


@pytest.mark.usefixtures("enable_x64")
@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
@pytest.mark.parametrize('factor', [0.0, 1.0, 2.0])
def test_coefficient_derivative_uses_algebraic_formula(factor):
    source = jnp.asarray([1, 2, 3, 4, 5], jnp.float32)
    for function in (two_record_accumulation, two_record_reference):
        result = jax.grad(lambda coefficient: function(source, coefficient, jnp.float32(2)).sum())(jnp.float32(factor))
        assert result == 8


@pytest.mark.usefixtures("enable_x64")
@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
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


@pytest.mark.usefixtures("enable_x64")
@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
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


@pytest.mark.usefixtures("enable_x64")
@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
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


@pytest.mark.usefixtures("enable_x64")
@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
@pytest.mark.parametrize('source_dtype,result_dtype', [('float16', 'complex64'), ('complex64', 'float16')])
def test_low_precision_cross_kind_source_ad(source_dtype, result_dtype):
    source = jnp.ones(5, dtype=source_dtype)
    run = lambda values: mixed_accumulation(values, (jnp.float32(2),) * 3, result_dtype)
    oracle = lambda values: mixed_reference(values, (jnp.float32(2),) * 3, result_dtype)
    for actual, expected in zip(jax.jvp(run, (source,), (source,)), jax.jvp(oracle, (source,), (source,)), strict=True):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.usefixtures("enable_x64")
@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
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


@pytest.mark.usefixtures("enable_x64")
@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
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


@pytest.mark.usefixtures("enable_x64")
@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
@pytest.mark.parametrize('source_dtype,dtype', [('float32', 'complex128'), ('complex64', 'float64')])
def test_unscaled_empty_and_symbolic_zero(source_dtype, dtype):
    source = samples_values((3,), source_dtype)
    for record in (AffineRecord((2, 2), (1, 1), 0, (1, 1), 1), AffineRecord((0,), (1,), 3, (1,), 5)):
        run = lambda data: accumulation_p.bind(data, records=(record,), output_size=5, dtype=jnp.dtype(dtype))
        frozen = lambda data: run(jax.lax.stop_gradient(data))
        np.testing.assert_array_equal(jax.jvp(frozen, (source,), (source,))[1], jnp.zeros(5, dtype=dtype))
        result, pullback = jax.vjp(run, source)
        gradient = jax.jit(pullback)(jnp.ones_like(result))[0]
        np.testing.assert_array_equal(gradient, [1, 2, 1] if record.logical_shape != (0,) else [0, 0, 0])


PARTIAL = (AffineRecord((16,), (2,), 1, (3,), 2),)


LAYOUTS = [
    pytest.param(PARTITIONS, (1.25, -.5), 16, 16, id="partitions"),
    pytest.param(PARTITIONS, (1.25, -.5), 20, 16, id="unread-tail"),
    pytest.param(PARTIAL, (.5,), 32, 50, id="partial"),
    pytest.param((AffineRecord((2, 8, 3, 12), (96, 1, 192, 8), 0, (288, 36, 12, 1), 0),),
                 (1.25,), 576, 576, id="rank4"),
    pytest.param((AffineRecord((5, 7), (1, 5), 0, (7, 1), 0),),
                 (.75,), 35, 35, id="transpose"),
    pytest.param((AffineRecord((2, 2), (1, 1), 0, (2, 1), 1),),
                 (1.25,), 5, 6, id="repeated-reads"),
    pytest.param((AffineRecord((3, 2), (0, -1), 1, (2, 1), 0),),
                 (-.5,), 3, 6, id="broadcast-negative"),
]


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
# Keep every layout in float32 and explicit mixed-type projection/read cases.
@pytest.mark.parametrize('records,factors,source_size,output_size,source_dtype,result_dtype', [
    pytest.param(*layout.values, source, result,
                 id=f'{source.__name__}-{result.__name__}-{layout.id}')
    for source, result, layouts in (
        (jnp.float32, jnp.float32, ('partitions', 'unread-tail', 'partial', 'rank4',
                                  'transpose', 'repeated-reads', 'broadcast-negative')),
        (jnp.float16, jnp.float32, ('unread-tail', 'partial')),
        (jnp.float32, jnp.complex64, ('partial',)),
        (jnp.complex64, jnp.float32, ('partial', 'transpose')),
        (jnp.complex64, jnp.complex64, ('partial', 'repeated-reads', 'broadcast-negative')),
    )
    for layout in LAYOUTS if layout.id in layouts
])
def test_affine_forward_jvp_vjp_and_linear_transpose(
    records, factors, source_size, output_size, source_dtype, result_dtype,
):
    factors = tuple(jnp.asarray(value, dtype=jnp.float32) for value in factors)
    source = jnp.linspace(-3, 4, source_size).astype(source_dtype)
    direction = jnp.full_like(source, .5)
    cotangent = jnp.linspace(-2, 3, output_size).astype(result_dtype)
    if source_dtype == jnp.complex64:
        source = source * jnp.complex64(1 + .375j)
        direction = direction * jnp.complex64(.75 - .5j)
    if result_dtype == jnp.complex64:
        cotangent = cotangent * jnp.complex64(.75 + 1.25j)
    native = lambda value: accumulation_p.bind(
        value, *factors, records=records, coefficient_records=tuple(range(len(records))),
        output_size=output_size, dtype=np.dtype(result_dtype),
    )
    reference = lambda value: reference_transform(value, records, factors, output_size, result_dtype)
    forward = jax.jit(lambda value, tangent: jax.jvp(native, (value,), (tangent,)))
    for actual, expected in zip(forward(source, direction),
                                jax.jvp(reference, (source,), (direction,)), strict=True):
        affine_ad_assert_close(actual, expected)
    assert_native_calls(forward.lower(source, direction), 2)
    reverse = jax.jit(lambda value, cot: (native(value), jax.vjp(native, value)[1](cot)[0]))
    actual, gradient = reverse(source, cotangent)
    affine_ad_assert_close(actual, reference(source))
    expected_gradient = jax.vjp(reference, source)[1](cotangent)[0]
    affine_ad_assert_close(gradient, expected_gradient)
    affine_ad_assert_close(jax.jit(lambda cot: jax.linear_transpose(native, source)(cot)[0])(cotangent),
                 expected_gradient)
    assert_native_calls(reverse.lower(source, cotangent), 2)
    used = np.zeros(source_size, dtype=bool)
    for record in records:
        for index in np.ndindex(record.logical_shape):
            used[record.source_offset + sum(item * stride for item, stride in
                                           zip(index, record.source_strides, strict=True))] = True
    np.testing.assert_array_equal(np.asarray(gradient)[~used], 0)


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
@pytest.mark.parametrize("factor", [1.25 - .75j, .125 + 1j, complex(2.5, -0.), 3 + 3j])
@pytest.mark.parametrize("source_dtype", [jnp.float32, jnp.complex64])
def test_complex_factor_uses_bilinear_not_conjugated_transpose(factor, source_dtype):
    source = jnp.asarray([0, -0., 1.25, -2.5, .125, 2, -2], dtype=source_dtype)
    if source_dtype == jnp.complex64:
        source = source * jnp.complex64(1 + .375j)
    cotangent = jnp.asarray([complex(-0., -0.), 1j, 1.25, 2.5j, .125 + 1j, 3 + 3j, -3 - 3j])
    coefficient = jnp.asarray(factor, dtype=jnp.complex64)
    native = lambda value: accumulation_p.bind(
        value, coefficient, records=(AffineRecord((7,), (1,), 0, (1,), 0),),
        coefficient_records=(0,), output_size=7, dtype=np.dtype(jnp.complex64),
    )
    reference = lambda value: (value * coefficient).astype(jnp.complex64)
    actual, gradient = jax.jit(lambda value, cot: (native(value), jax.vjp(native, value)[1](cot)[0]))(
        source, cotangent,
    )
    affine_ad_assert_close(actual, reference(source))
    affine_ad_assert_close(gradient, jax.vjp(reference, source)[1](cotangent)[0])


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
def test_real_to_complex_transpose_has_explicit_bilinear_result():
    native = lambda value: accumulation_p.bind(
        value, jnp.complex64(2 + 3j), records=(AffineRecord((1,), (1,), 0, (1,), 0),),
        coefficient_records=(0,), output_size=1, dtype=np.dtype(jnp.complex64),
    )
    source, cotangent = jnp.asarray([1.25]), jnp.asarray([5 + 7j])
    np.testing.assert_array_equal(jax.jit(native)(source), [2.5 + 3.75j])
    np.testing.assert_array_equal(jax.jit(lambda cot: jax.vjp(native, source)[1](cot)[0])(cotangent), [-11])


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
@pytest.mark.parametrize("large_product", [False, True])
def test_complex_finite_product_accuracy(large_product):
    if large_product:
        maximum = np.finfo(np.float32).max / 16
        source = jnp.asarray([complex(maximum, -maximum)], dtype=jnp.complex64)
        factor = jnp.complex64(3 + 3j)
    else:
        rng = np.random.default_rng(20260825)
        source = jnp.asarray(rng.standard_normal(131072) + 1j * rng.standard_normal(131072), dtype=jnp.complex64)
        factor = jnp.complex64(1.25 - .75j)
    actual = jax.jit(lambda value: accumulation_p.bind(
        value, factor, records=(AffineRecord((source.size,), (1,), 0, (1,), 0),),
        coefficient_records=(0,), output_size=source.size, dtype=np.dtype(jnp.complex64),
    ))(source)
    affine_ad_assert_close(actual, source * factor)


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
@pytest.mark.parametrize("scaled", [False, True])
@pytest.mark.parametrize("source_dtype,result_dtype", [
    (jnp.float32, jnp.float32), (jnp.float16, jnp.float32), (jnp.complex64, jnp.complex64),
])
def test_zero_cotangent_pullback_remains_linear(scaled, source_dtype, result_dtype):
    parameters = dict(records=(AffineRecord((3,), (1,), 0, (1,), 0),),
                      output_size=3, dtype=np.dtype(result_dtype))
    if scaled:
        native = lambda value: accumulation_p.bind(value, jnp.float32(1), coefficient_records=(0,), **parameters)
    else:
        native = lambda value: copy_p.bind(value, **parameters)
    source = jnp.ones(3, dtype=source_dtype)
    cotangent = jnp.asarray([-0., 0., -0.], dtype=result_dtype)
    if result_dtype == jnp.complex64:
        cotangent = jnp.asarray([complex(-0., 1), complex(1, -0.), complex(-0., -0.)], dtype=result_dtype)
    direction = jnp.ones_like(cotangent)
    pullback = lambda cot: jax.vjp(native, source)[1](cot)[0]
    expected = lambda cot: jax.vjp(lambda value: value.astype(result_dtype), source)[1](cot)[0]
    for actual, reference in zip(jax.jit(lambda cot, tangent: jax.jvp(pullback, (cot,), (tangent,)))(cotangent, direction),
                                 jax.jvp(expected, (cotangent,), (direction,)), strict=True):
        affine_ad_assert_close(actual, reference)
    output_cotangent = jnp.ones_like(source)
    affine_ad_assert_close(jax.jit(lambda cot: jax.linear_transpose(pullback, cotangent)(cot)[0])(output_cotangent),
                 jax.linear_transpose(expected, cotangent)(output_cotangent)[0])
    lowered = jax.jit(pullback).lower(cotangent)
    assert_native_calls(lowered, 1)
    text = lowered.as_text().lower()
    assert "stablehlo.compare" not in text and "stablehlo.select" not in text


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.int32])
def test_zero_and_integer_tangents(dtype):
    source = jnp.arange(35, dtype=dtype)
    record = AffineRecord((5, 7), (1, 5), 0, (7, 1), 0)
    native = lambda value: accumulation_p.bind(
        value, jnp.asarray(-1, dtype=dtype), records=(record,), coefficient_records=(0,),
        output_size=35, dtype=np.dtype(dtype),
    )
    tangent_dtype = jax.dtypes.float0 if dtype == jnp.int32 else dtype
    direction = jnp.zeros(source.shape, dtype=tangent_dtype)
    primal, tangent = jax.jit(lambda value, dot: jax.jvp(native, (value,), (dot,)))(source, direction)
    np.testing.assert_array_equal(primal, -source.reshape(7, 5).T.ravel())
    assert tangent.shape == (35,) and tangent.dtype == tangent_dtype
    if dtype == jnp.float32:
        np.testing.assert_array_equal(tangent, 0)


@pytest.mark.skipif(not native_available(), reason='native CPU stride unavailable')
def test_compact_vmap_and_float32_derivatives_use_native():
    record = compact_record((4096,), source_offset=3)
    source = jnp.arange(4 * 4104, dtype=jnp.float32).reshape(4, 4104)
    function = jax.vmap(lambda value: compact_mapped(value, record, .75))
    oracle = lambda value: reference(value, record, .75)
    compact_assert_native(function, source)
    actual = jax.jvp(function, (source,), (source / 8,))
    expected = jax.jvp(oracle, (source,), (source / 8,))
    for result, wanted in zip(actual, expected, strict=True):
        compact_assert_close(result, wanted)
    cotangent = jnp.ones_like(actual[0])
    pullback = lambda value: jax.vjp(function, source)[1](value)[0]
    gradient = jax.jit(pullback)(cotangent)
    compact_assert_close(gradient, jax.vjp(oracle, source)[1](cotangent)[0])
    np.testing.assert_array_equal(gradient[:, :3], 0)
    np.testing.assert_array_equal(gradient[:, -5:], 0)
    compact_assert_native(pullback, cotangent)


@pytest.mark.skipif(not native_available(), reason='native CPU stride unavailable')
def test_complex_nonleading_vmap_jvp_and_vjp():
    record = compact_record((128, 64), source_offset=3)
    source = complex_values((8200, 3))
    direction = source * jnp.complex64(.25 + .125j)
    function = jax.vmap(lambda value: compact_mapped(value, record, .75 - .5j), in_axes=1, out_axes=1)
    oracle = jax.vmap(lambda value: reference(value, record, .75 - .5j), in_axes=1, out_axes=1)
    compact_assert_native(function, source)
    actual = jax.jvp(function, (source,), (direction,))
    expected = jax.jvp(oracle, (source,), (direction,))
    for result, wanted in zip(actual, expected, strict=True):
        compact_assert_close(result, wanted)
    cotangent = complex_values(actual[0].shape)
    pullback = lambda value: jax.vjp(function, source)[1](value)[0]
    gradient = jax.jit(pullback)(cotangent)
    compact_assert_close(gradient, jax.vjp(oracle, source)[1](cotangent)[0])
    np.testing.assert_array_equal(gradient[:3], 0)
    np.testing.assert_array_equal(gradient[-5:], 0)
    compact_assert_native(pullback, cotangent)


@pytest.mark.skipif(not native_available(), reason='native CPU stride unavailable')
@pytest.mark.parametrize("complex_output", [False, True])
def test_broadcast_pullback_accumulates_repeated_reads(complex_output):
    record = AffineRecord((2, 3), (0, -1), 2, (3, 1), 0)
    source = jnp.asarray([1., -2., 3.], dtype=jnp.float32)
    dtype = jnp.complex64 if complex_output else jnp.float32
    coefficient = jnp.asarray(1.25 - .5j if complex_output else 1.25, dtype=dtype)
    operation = lambda value: generic_mapped(value, record, coefficient, dtype, 6)
    reference = lambda value: reference_map(value, record, coefficient, dtype, 6)
    cotangent = dtype_values(dtype, 6)
    generic_assert_close(jax.jit(lambda cot: jax.vjp(operation, source)[1](cot)[0])(cotangent),
                 jax.vjp(reference, source)[1](cotangent)[0])


@pytest.mark.skipif(not native_available(), reason='native CPU stride unavailable')
def test_mixed_repeated_source_reduction_differentials():
    count, width = 1024, 16
    record = AffineRecord((count, width), (1, 0), 0, (1, 0), 0)
    coefficient = jnp.complex64(1.25 - .5j)
    source = jnp.linspace(-2, 3, count, dtype=jnp.float32)
    operation = lambda value: accumulation_p.bind(value, coefficient, records=(record,),
        coefficient_records=(0,), output_size=count, dtype=jnp.dtype(jnp.complex64))
    reference = lambda value: value * coefficient * width
    direction = source * -.25 + 1
    cotangent = dtype_values(jnp.complex64, count) / 8
    actual = jax.jit(lambda value, tangent, cot: (
        jax.jvp(operation, (value,), (tangent,)), jax.vjp(operation, value)[1](cot)[0]))(source, direction, cotangent)
    expected = (jax.jvp(reference, (source,), (direction,)), jax.vjp(reference, source)[1](cotangent)[0])
    for result, wanted in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        generic_assert_close(result, wanted)


@pytest.mark.skipif(not native_available(), reason='native CPU stride unavailable')
@pytest.mark.parametrize("source_dtype,result_dtype,factor", [
    (jnp.float16, jnp.float32, -1.25), (jnp.complex64, jnp.complex64, 1),
    (jnp.complex64, jnp.float32, -1.25 + .75j), (jnp.float32, jnp.complex64, -.75),
])
def test_mixed_contiguous_pullback_finite_and_tiny_values(source_dtype, result_dtype, factor):
    size = 1025
    source = jnp.ones(size, dtype=source_dtype)
    record = AffineRecord((size,), (1,), 0, (1,), 0)
    coefficient = jnp.asarray(factor, dtype=jnp.complex64 if isinstance(factor, complex) else jnp.float32)
    operation = lambda value: generic_mapped(value, record, coefficient, result_dtype, size)
    reference = lambda value: reference_map(value, record, coefficient, result_dtype, size)
    raw = np.resize(np.asarray([0., -0., np.nextafter(np.float32(0), np.float32(1)), 1., -1., 1.25, -.125],
                                dtype=np.float32), size)
    cotangent = jnp.asarray(raw, dtype=result_dtype)
    if jnp.issubdtype(result_dtype, jnp.complexfloating):
        cotangent = cotangent + .5j * cotangent[::-1]
    actual = jax.jit(lambda cot: jax.vjp(operation, source)[1](cot)[0])(cotangent)
    generic_assert_close(actual, jax.vjp(reference, source)[1](cotangent)[0])


@pytest.mark.skipif(not native_available(), reason='native CPU stride unavailable')
def test_complex_compact_batch_jvp_and_vjp():
    size = 65539
    record = AffineRecord((size,), (1,), 0, (1,), 0)
    source = jnp.stack((complex_kernel_values(size), complex_kernel_values(size) + 1j))
    direction = source * jnp.complex64(.25 + .125j)
    cotangent = source * jnp.complex64(-.5 + .25j)
    operation = lambda value: complex_kernel_mapped(value, record, .75 - .5j)
    oracle = lambda value: value * jnp.complex64(.75 - .5j)
    complex_kernel_assert_close(jax.jit(operation)(source), oracle(source))
    complex_kernel_assert_close(jax.jit(jax.vmap(operation))(source), oracle(source))
    actual = jax.jit(lambda value, tangent: jax.jvp(operation, (value,), (tangent,)))(source, direction)
    expected = jax.jvp(oracle, (source,), (direction,))
    for result, wanted in zip(actual, expected, strict=True):
        complex_kernel_assert_close(result, wanted)
    pullback = jax.jit(lambda cot: jax.vjp(operation, source)[1](cot)[0])
    complex_kernel_assert_close(pullback(cotangent), jax.vjp(oracle, source)[1](cotangent)[0])
    complex_kernel_assert_native(pullback.lower(cotangent))
