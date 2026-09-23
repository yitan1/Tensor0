"""JAX AD copy contracts."""

from math import prod

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensor0._stride import StridedView, materialize
from tensor0._stride._jax import copy_p
from tensor0._stride._layout import AffineRecord

from tests.stride.support.availability import native_available
from tests.stride.support.oracles.reduction import addresses, assert_close


pytestmark = pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')


COPY_FLOATS = ("float16", "bfloat16", "float32", "float64")


COPY_CROSS_PAIRS = [(source, result) for real in COPY_FLOATS for complex_dtype in ("complex64", "complex128")
               for source, result in ((real, complex_dtype), (complex_dtype, real))]


COPY_PAIRS = [(source, result) for source in COPY_FLOATS for result in COPY_FLOATS if source != result] + [
    ("complex64", "complex128"), ("complex128", "complex64"), *COPY_CROSS_PAIRS]


def copy_converted(value, dtype):
    if jnp.issubdtype(jnp.dtype(dtype), jnp.floating):
        value = jnp.real(value)
    return value.astype(dtype)


# Each real dtype retains both projection directions, not both complex widths.
COPY_REPRESENTATIVE_CROSS_PAIRS = (
    ("float16", "complex64"), ("complex64", "float16"),
    ("bfloat16", "complex128"), ("complex128", "bfloat16"),
    ("float32", "complex128"), ("complex128", "float32"),
    ("float64", "complex64"), ("complex64", "float64"),
)

COPY_BATCH_PAIRS = [("float16", "float32"), ("float64", "bfloat16"),
                    ("float32", "complex128"), ("complex128", "float32")]


# Preserve original IDs while coupling nonempty/empty batches to one layout.
@pytest.mark.parametrize("source_dtype,result_dtype,batch_shape,shape,strides,offset", [
    pytest.param(source, result, batch, shape, strides, offset,
                 id=f'shape{layout}-strides{layout}-{offset}-{source}-{result}-batch_shape{index}')
    for layout, (shape, strides, offset) in enumerate([((2, 2), (1, -1), 1), ((4,), (0,), 2)])
    for index, (source, result, batch) in enumerate(
        [(*pair, ()) for pair in COPY_PAIRS]
        + [(*pair, batch) for pair in COPY_BATCH_PAIRS for batch in ((2,), (2, 0))])
    if (batch == () and (source, result) in (
        *COPY_REPRESENTATIVE_CROSS_PAIRS,
        ("float16", "bfloat16"), ("bfloat16", "float16"),
        ("float16", "float32"), ("float32", "float16"),
        ("bfloat16", "float64"), ("float64", "bfloat16"),
        ("complex64", "complex128"), ("complex128", "complex64")))
    or (batch, layout) in (((2,), 0), ((2, 0), 1))
])
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


# Nonempty rank-two AD remains for every batch pair; retain one empty rank each.
@pytest.mark.parametrize("source_dtype,result_dtype,batch_shape", [
    pytest.param(source, result, batch, id=f'{source}-{result}-batch_shape{index}')
    for index, (source, result, batch) in enumerate(
        [(*pair, ()) for pair in COPY_CROSS_PAIRS]
        + [(*pair, batch) for pair in (
            ("float16", "complex64"), ("bfloat16", "complex128"),
            ("complex64", "float64"), ("complex128", "float32"))
           for batch in ((2, 3), (0,), (2, 0))])
    if (batch == () and (source, result) in COPY_REPRESENTATIVE_CROSS_PAIRS)
    or batch == (2, 3)
    or (source, result, batch) in (
        ("float16", "complex64", (0,)), ("bfloat16", "complex128", (2, 0)),
        ("complex64", "float64", (2, 0)), ("complex128", "float32", (0,)))
])
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


@pytest.mark.parametrize("source_dtype,result_dtype", COPY_REPRESENTATIVE_CROSS_PAIRS)
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


def test_rank70_broadcast_forward_jvp_and_transpose():
    record = AffineRecord((1,) * 69 + (3,), (0,) * 70, 0, (0,) * 69 + (1,), 0)
    operation = lambda value: copy_p.bind(value, records=(record,), output_size=3, dtype=value.dtype)
    source, direction = jnp.asarray([3.5]), jnp.asarray([-1.25])
    actual, tangent = jax.jit(lambda value, dot: jax.jvp(operation, (value,), (dot,)))(source, direction)
    np.testing.assert_array_equal(actual, np.full(3, 3.5))
    np.testing.assert_array_equal(tangent, np.full(3, -1.25))
    np.testing.assert_array_equal(jax.jit(lambda cot: jax.vjp(operation, source)[1](cot)[0])(jnp.asarray([1., -2., 4.])), [3.])


def test_broadcast_map_transpose_accumulates_and_supports_nested_ad():
    record = AffineRecord((3, 4), (0, -1), 3, (-1, 3), 2)
    source = jnp.arange(4, dtype=jnp.float32)
    native = lambda value: copy_p.bind(value, records=(record,), output_size=12, dtype=value.dtype)
    reference = lambda value: jnp.zeros(12).at[addresses((3, 4), (-1, 3), 2)].set(value[addresses((3, 4), (0, -1), 3)])
    native_pullback = lambda cot: jax.vjp(native, source)[1](cot)[0]
    reference_pullback = lambda cot: jax.vjp(reference, source)[1](cot)[0]
    cotangent = jnp.arange(36, dtype=jnp.float32).reshape(3, 12) / 4
    assert_close(jax.jit(jax.vmap(native_pullback))(cotangent), jax.vmap(reference_pullback)(cotangent))
    assert_close(jax.linear_transpose(native_pullback, cotangent[0])(jnp.ones_like(source))[0],
                 jax.linear_transpose(reference_pullback, cotangent[0])(jnp.ones_like(source))[0])
