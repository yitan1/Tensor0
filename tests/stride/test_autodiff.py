from __future__ import annotations

from math import prod

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensor0._stride import StridedView, materialize
from tensor0._stride._jax import accumulation_p, copy_p
from tensor0._stride._layout import AffineRecord

from ._support import native_available


pytestmark = pytest.mark.skipif(not native_available(), reason="native CPU stride is unavailable")

PARTITIONS = (
    AffineRecord((4, 2), (4, 1), 0, (4, 1), 2),
    AffineRecord((4, 2), (4, 1), 2, (4, 1), 0),
)
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


def reference_transform(source, records, factors, output_size, dtype):
    result = jnp.zeros(source.shape[:-1] + (output_size,), dtype=dtype)
    for record, factor in zip(records, factors, strict=True):
        source_indices = np.full(record.logical_shape, record.source_offset, dtype=np.int32)
        destination_indices = np.full(record.logical_shape, record.destination_offset, dtype=np.int32)
        for axis, size in enumerate(record.logical_shape):
            shape = (1,) * axis + (size,) + (1,) * (len(record.logical_shape) - axis - 1)
            indices = np.arange(size, dtype=np.int32).reshape(shape)
            source_indices += indices * record.source_strides[axis]
            destination_indices += indices * record.destination_strides[axis]
        values = source[..., source_indices.ravel()] * factor
        if not jnp.issubdtype(dtype, jnp.complexfloating):
            values = jnp.real(values)
        result = result.at[..., destination_indices.ravel()].set(values.astype(dtype))
    return result


def assert_close(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    tolerance = 2e-3 if actual.dtype == jnp.float16 else 2e-6
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)


def assert_native_calls(lowered, count):
    text = lowered.as_text().lower()
    assert text.count("custom_call") == count
    assert "tensor0_stride_" in text
    assert "gather" not in text and "scatter" not in text


@pytest.mark.parametrize("records,factors,source_size,output_size", LAYOUTS)
@pytest.mark.parametrize("source_dtype,result_dtype", [
    (jnp.float32, jnp.float32), (jnp.float16, jnp.float32),
    (jnp.float32, jnp.complex64), (jnp.complex64, jnp.float32),
    (jnp.complex64, jnp.complex64),
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
        assert_close(actual, expected)
    assert_native_calls(forward.lower(source, direction), 2)
    reverse = jax.jit(lambda value, cot: (native(value), jax.vjp(native, value)[1](cot)[0]))
    actual, gradient = reverse(source, cotangent)
    assert_close(actual, reference(source))
    expected_gradient = jax.vjp(reference, source)[1](cotangent)[0]
    assert_close(gradient, expected_gradient)
    assert_close(jax.jit(lambda cot: jax.linear_transpose(native, source)(cot)[0])(cotangent),
                 expected_gradient)
    assert_native_calls(reverse.lower(source, cotangent), 2)
    used = np.zeros(source_size, dtype=bool)
    for record in records:
        for index in np.ndindex(record.logical_shape):
            used[record.source_offset + sum(item * stride for item, stride in
                                           zip(index, record.source_strides, strict=True))] = True
    np.testing.assert_array_equal(np.asarray(gradient)[~used], 0)


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
    assert_close(actual, reference(source))
    assert_close(gradient, jax.vjp(reference, source)[1](cotangent)[0])


def test_real_to_complex_transpose_has_explicit_bilinear_result():
    native = lambda value: accumulation_p.bind(
        value, jnp.complex64(2 + 3j), records=(AffineRecord((1,), (1,), 0, (1,), 0),),
        coefficient_records=(0,), output_size=1, dtype=np.dtype(jnp.complex64),
    )
    source, cotangent = jnp.asarray([1.25]), jnp.asarray([5 + 7j])
    np.testing.assert_array_equal(jax.jit(native)(source), [2.5 + 3.75j])
    np.testing.assert_array_equal(jax.jit(lambda cot: jax.vjp(native, source)[1](cot)[0])(cotangent), [-11])


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
    assert_close(actual, source * factor)


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
        assert_close(actual, reference)
    output_cotangent = jnp.ones_like(source)
    assert_close(jax.jit(lambda cot: jax.linear_transpose(pullback, cotangent)(cot)[0])(output_cotangent),
                 jax.linear_transpose(expected, cotangent)(output_cotangent)[0])
    lowered = jax.jit(pullback).lower(cotangent)
    assert_native_calls(lowered, 1)
    text = lowered.as_text().lower()
    assert "stablehlo.compare" not in text and "stablehlo.select" not in text


@pytest.mark.parametrize("mode", ["jvp", "vjp"])
def test_vmap_batches_affine_derivatives_without_unrolling_records(mode):
    factors = (jnp.float32(1.25), jnp.float32(-.5))
    native = lambda value: accumulation_p.bind(
        value, *factors, records=PARTITIONS, coefficient_records=(0, 1), output_size=16, dtype=np.dtype(jnp.float32),
    )
    reference = lambda value: reference_transform(value, PARTITIONS, factors, 16, jnp.float32)
    source = jnp.arange(48, dtype=jnp.float32).reshape(3, 16)
    direction = jnp.linspace(-2, 3, 48).reshape(3, 16)
    if mode == "jvp":
        derivative = lambda operation, value, tangent: jax.jvp(operation, (value,), (tangent,))
    else:
        derivative = lambda operation, value, tangent: jax.vjp(operation, value)[1](tangent)[0]
    compiled = jax.jit(jax.vmap(lambda value, tangent: derivative(native, value, tangent)))
    actual = compiled(source, direction)
    expected = jax.vmap(lambda value, tangent: derivative(reference, value, tangent))(source, direction)
    for result, wanted in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        assert_close(result, wanted)
    assert_native_calls(compiled.lower(source, direction), 2 if mode == "jvp" else 1)


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


COPY_FLOATS = ("float16", "bfloat16", "float32", "float64")



COPY_CROSS_PAIRS = [(source, result) for real in COPY_FLOATS for complex_dtype in ("complex64", "complex128")
               for source, result in ((real, complex_dtype), (complex_dtype, real))]



COPY_PAIRS = [(source, result) for source in COPY_FLOATS for result in COPY_FLOATS if source != result] + [
    ("complex64", "complex128"), ("complex128", "complex64"), *COPY_CROSS_PAIRS]



def copy_converted(value, dtype):
    if jnp.issubdtype(jnp.dtype(dtype), jnp.floating):
        value = jnp.real(value)
    return value.astype(dtype)



@pytest.mark.parametrize("source_dtype,result_dtype", COPY_PAIRS)
@pytest.mark.parametrize("batch_shape", [(), (2,), (2, 0)])
@pytest.mark.parametrize("shape,strides,offset", [((2, 2), (1, -1), 1), ((4,), (0,), 2)])
def test_copy_mixed_jvp_vjp_transpose(source_dtype, result_dtype, batch_shape, shape, strides, offset):
    with jax.enable_x64():
        source = jnp.arange(prod(batch_shape) * 4, dtype=jnp.float32).astype(source_dtype).reshape((*batch_shape, 4))
        if source_dtype.startswith("complex"):
            source = source * (1 + 2j)
        addresses = np.asarray([offset + sum(index * stride for index, stride in zip(coordinate, strides))
                                for coordinate in np.ndindex(shape)]).reshape(shape)
        run = lambda values: materialize(StridedView(values, shape, strides, offset), dtype=result_dtype)
        reference = lambda values: copy_converted(values[..., addresses], result_dtype)
        tangent = jnp.ones_like(source)
        for actual, expected in zip(jax.jit(lambda values: jax.jvp(run, (values,), (tangent,)))(source),
                                    jax.jvp(reference, (source,), (tangent,)), strict=True):
            assert actual.dtype == expected.dtype
            np.testing.assert_array_equal(actual, expected)
        cotangent = jnp.ones((*batch_shape, *shape), dtype=result_dtype)
        if result_dtype.startswith("complex"):
            cotangent = cotangent * (2 + 3j)
        expected = jax.vjp(reference, source)[1](cotangent)[0]
        for reverse in (jax.vjp(run, source)[1], jax.linear_transpose(run, source)):
            actual = jax.jit(reverse)(cotangent)[0]
            assert actual.dtype == source.dtype
            np.testing.assert_array_equal(actual, expected)
        transpose = lambda values: jax.linear_transpose(run, source)(values)[0]
        np.testing.assert_array_equal(jax.jvp(transpose, (cotangent,), (cotangent,))[1], expected)



@pytest.mark.parametrize("source_dtype,result_dtype,cotangent,expected", [
    ("float32", "float64", [1 + 2**-24, -1], 0),
    ("float64", "float32", [16777216, 1, -16777216], 1),
])
def test_transpose_cast_precedes_accumulation(source_dtype, result_dtype, cotangent, expected):
    with jax.enable_x64():
        source = jnp.zeros(1, dtype=source_dtype)
        run = lambda values: materialize(StridedView(values, (len(cotangent),), (0,), 0), dtype=result_dtype)
        cotangent = jnp.asarray(cotangent, dtype=result_dtype)
        actual = jax.jit(jax.linear_transpose(run, source))(cotangent)[0]
        np.testing.assert_array_equal(actual, [expected])
        text = jax.jit(jax.linear_transpose(run, source)).lower(cotangent).as_text()
        assert "stablehlo.convert" in text
        assert "stride_accumulation_" in text



def test_multiple_records_and_outer_vmap():
    records = (AffineRecord((2,), (1,), 0, (1,), 0), AffineRecord((2,), (-1,), 1, (1,), 2))
    run = lambda values: copy_p.bind(values, records=records, output_size=5, dtype=jnp.dtype("float16"))
    reference = lambda values: jnp.concatenate((values[..., [0, 1, 1, 0]].astype(jnp.float16),
                                               jnp.zeros((*values.shape[:-1], 1), dtype=jnp.float16)), axis=-1)
    source = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)
    cotangent = jnp.ones((2, 5), dtype=jnp.float16)
    np.testing.assert_array_equal(jax.jit(jax.vmap(run))(source), reference(source))
    actual = jax.jit(jax.vjp(jax.vmap(run), source)[1])(cotangent)[0]
    np.testing.assert_array_equal(actual, jax.vjp(reference, source)[1](cotangent)[0])



def test_empty_layout_and_zero_tangent():
    run = lambda source: materialize(StridedView(source, (0,), (1,), 0), dtype=jnp.float16)
    source = jnp.zeros(0, dtype=jnp.float32)
    np.testing.assert_array_equal(jax.vjp(run, source)[1](jnp.zeros(0, dtype=jnp.float16))[0], source)
    result, tangent = jax.jvp(lambda unused: run(source), (jnp.float32(1),), (jnp.float32(1),))
    assert result.dtype == tangent.dtype == jnp.float16
    assert result.shape == tangent.shape == (0,)



@pytest.mark.parametrize("source_dtype,result_dtype", [("float32", "bool"),
                                                       ("complex64", "int32"), ("float32", "int32")])
def test_discrete_conversion_has_zero_tangent(source_dtype, result_dtype):
    source = jnp.ones(1, dtype=source_dtype)
    run = lambda values: materialize(StridedView(values, (1,), (1,), 0), dtype=result_dtype)
    result, tangent = jax.jvp(run, (source,), (source,))
    assert tangent.dtype == jax.dtypes.float0 and tangent.shape == result.shape
    np.testing.assert_array_equal(jax.grad(lambda value: jnp.sum(run(value).astype(jnp.float32)))(source),
                                  jnp.zeros_like(source))



@pytest.mark.parametrize("source_dtype,result_dtype", COPY_CROSS_PAIRS)
@pytest.mark.parametrize("batch_shape", [(), (2, 3), (0,), (2, 0)])
def test_cross_kind_multirecord_higher_derivatives(source_dtype, result_dtype, batch_shape):
    with jax.enable_x64():
        source = (jnp.arange(prod(batch_shape) * 3) % 4 / 4).astype(source_dtype).reshape((*batch_shape, 3))
        if source_dtype.startswith("complex"):
            source = source + 1j * (source + .5)
        records = (AffineRecord((2,), (1,), 0, (2,), 0),
                   AffineRecord((2,), (-1,), 1, (2,), 1),
                   AffineRecord((0,), (1,), 3, (1,), 5))
        def run(values):
            return copy_p.bind(values, records=records, output_size=5, dtype=jnp.dtype(result_dtype))
        def reference(values):
            selected = copy_converted(values[..., [0, 1, 1, 0]], result_dtype)
            return jnp.concatenate((selected, jnp.zeros((*values.shape[:-1], 1), dtype=result_dtype)), axis=-1)
        cotangent = jnp.full((*batch_shape, 5), 2 + 3j if result_dtype.startswith("complex") else 2, dtype=result_dtype)
        reverse = lambda values: jax.linear_transpose(run, source)(values)[0]
        expected_reverse = lambda values: jax.vjp(reference, source)[1](values)[0]
        for actual, expected in zip(jax.jvp(reverse, (cotangent,), (cotangent,)),
                                    jax.jvp(expected_reverse, (cotangent,), (cotangent,)), strict=True):
            np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-3)
        actual = jax.jit(jax.linear_transpose(reverse, cotangent))(source)[0]
        expected = jax.vjp(expected_reverse, cotangent)[1](source)[0]
        assert actual.dtype == cotangent.dtype
        np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-3)
        def objective(function, values):
            result = function(values)
            return jnp.real(jnp.sum(result * jnp.conj(result)))
        gradient = jax.grad(lambda values: objective(run, values))
        reference_gradient = jax.grad(lambda values: objective(reference, values))
        directions = jnp.ones_like(source)
        np.testing.assert_allclose(jax.jit(gradient)(source), reference_gradient(source), rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(jax.jvp(gradient, (source,), (directions,))[1],
                                   jax.jvp(reference_gradient, (source,), (directions,))[1], rtol=1e-3, atol=1e-3)



@pytest.mark.parametrize("source_dtype,result_dtype", COPY_CROSS_PAIRS)
def test_cross_kind_empty_and_symbolic_zero(source_dtype, result_dtype):
    with jax.enable_x64():
        source = jnp.empty(0, dtype=source_dtype)
        run = lambda values: materialize(StridedView(values, (0,), (1,), 0), dtype=result_dtype)
        result, pullback = jax.vjp(run, source)
        gradient = jax.jit(pullback)(result)[0]
        assert gradient.shape == (0,) and gradient.dtype == source.dtype
        frozen = lambda values: run(jax.lax.stop_gradient(values))
        tangent = jax.jvp(frozen, (source,), (source,))[1]
        assert tangent.shape == (0,) and tangent.dtype == result.dtype



@pytest.mark.parametrize("source_dtype,result_dtype", [("float32", "complex128"), ("complex128", "float32")])
def test_cross_kind_vmap_and_lowering(source_dtype, result_dtype):
    with jax.enable_x64():
        source = jnp.arange(6).astype(source_dtype).reshape((2, 3))
        if source_dtype.startswith("complex"):
            source = source + 2j
        run = lambda values: materialize(StridedView(values, (2, 2), (1, 1), 0), dtype=result_dtype)
        reference = lambda values: copy_converted(values[jnp.asarray([[0, 1], [1, 2]])], result_dtype)
        cotangent = jnp.full((2, 2, 2), 2 + 3j if result_dtype.startswith("complex") else 2, dtype=result_dtype)
        mapped = jax.vmap(run)
        gradient = jax.jit(jax.vjp(mapped, source)[1])(cotangent)[0]
        np.testing.assert_allclose(gradient, jax.vjp(jax.vmap(reference), source)[1](cotangent)[0])
        text = jax.jit(jax.linear_transpose(mapped, source)).lower(cotangent).as_text()
        assert "tensor0_stride_accumulation_" in text
        for operation in ("stablehlo.gather", "stablehlo.scatter", "tensor0_stride_update"):
            assert operation not in text



def test_imaginary_cotangent_is_not_a_real_source_contribution():
    source = jnp.ones(1, dtype=jnp.float32)
    run = lambda values: materialize(StridedView(values, (2,), (0,), 0), dtype=jnp.complex64)
    gradient = jax.jit(jax.vjp(run, source)[1])(jnp.asarray([1 + 2j, 3 - 4j], jnp.complex64))[0]
    np.testing.assert_array_equal(gradient, [4])



def test_real_output_cotangent_embeds_with_zero_imaginary_part():
    source = jnp.asarray([1 + 2j], dtype=jnp.complex64)
    run = lambda values: materialize(StridedView(values, (2,), (0,), 0), dtype=jnp.float32)
    gradient = jax.jit(jax.vjp(run, source)[1])(jnp.asarray([1, 3], jnp.float32))[0]
    np.testing.assert_array_equal(gradient, [4 + 0j])
