"""Public scale coefficients, short circuits and selected storage."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensor0._stride import StridedView, scale
from tensor0._stride._ffi._registration import operation_target

from tests.stride.support.availability import native_available
from tests.stride.support.oracles.scalar import assert_components
from tests.stride.support.oracles.scale import reference_scale


pytestmark = pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')


@pytest.mark.parametrize("factor", [1e-8, np.float32(1e-8), np.float64(1e-8)])
def test_closed_and_dynamic_scale_do_not_narrow_strong_coefficients(factor):
    with jax.enable_x64():
        source = jnp.asarray([65504, 32752, -65504], dtype=jnp.float16)
        expected = (factor * source).astype(source.dtype)
        closed = jax.jit(lambda value: scale(StridedView(value, (3,), (1,), 0), factor).data)
        dynamic = jax.jit(lambda value, coefficient: scale(StridedView(value, (3,), (1,), 0), coefficient).data)
        assert_components(closed(source), expected)
        assert_components(dynamic(source, factor), expected)


def test_weak_and_strong_coefficients_are_normalized_before_zero_one_binding():
    with jax.enable_x64():
        operation = jax.jit(lambda data, factor: scale(StridedView(data, (3,), (1,), 0), factor).data)
        source = jnp.asarray([1e30, np.inf, np.nan], dtype=jnp.float32)
        np.testing.assert_array_equal(operation(source, 1e-46), jnp.zeros_like(source))
        strong = operation(source, jnp.float64(1e-46))
        expected = (source.astype(jnp.float64) * jnp.float64(1e-46)).astype(source.dtype)
        assert_components(strong, expected)
        finite = jnp.asarray([65504, 32752, -65504], dtype=jnp.float32)
        np.testing.assert_array_equal(operation(finite, 1.00000001), finite)


@pytest.mark.parametrize("dtype,factor", [
    (jnp.float16, -1.25), (jnp.bfloat16, -1.25), (jnp.float32, -1.25),
    (jnp.float64, -1.25), (jnp.complex64, 1.25 - .75j),
    (jnp.complex128, 1.25 - .75j), (jnp.int32, -3),
])
def test_selected_scale_scalar_policies_and_input_protection(dtype, factor):
    with jax.enable_x64():
        base = jnp.arange(50, dtype=jnp.float32).astype(dtype) - 7
        original = np.asarray(base).copy()
        coefficient = jnp.asarray(factor, dtype=dtype)
        operation = jax.jit(lambda old, value: scale(StridedView(old, (16,), (3,), 2), value))
        actual = operation(base, coefficient)
        np.testing.assert_array_equal(actual.data, reference_scale(base, coefficient))
        np.testing.assert_array_equal(base, original)
        assert (actual.sizes, actual.strides, actual.offset) == ((16,), (3,), 2)
        selected = np.zeros(50, dtype=bool)
        selected[2:50:3] = True
        assert np.asarray(actual.data)[~selected].tobytes() == original[~selected].tobytes()
        text = operation.lower(base, coefficient).as_text()
        assert text.count("custom_call") == 1
        assert operation_target("update", np.dtype(dtype)) in text


def test_selected_scale_validates_public_inputs_and_injective_selection():
    base = jnp.arange(50, dtype=jnp.float32)
    with pytest.raises(TypeError, match="StridedView"):
        scale(base, 2)
    with pytest.raises(ValueError, match="scalar or match the batch shape"):
        scale(StridedView(base, (16,), (3,), 2), jnp.ones(2))
    overlapping = StridedView(base, (2, 2), (1, 1), 0)
    original = np.asarray(base).copy()
    with pytest.raises(Exception, match="injective|overlap"):
        jax.jit(lambda view: scale(view, 2))(overlapping).data.block_until_ready()
    np.testing.assert_array_equal(base, original)


@pytest.mark.parametrize("batch_shape", [(2,), (2, 3), (0,)])
@pytest.mark.parametrize("size", [0, 32])
def test_selected_scale_complete_and_empty_batches(batch_shape, size):
    base = jnp.arange(np.prod(batch_shape) * size, dtype=jnp.float32).reshape(*batch_shape, size)
    factors = (jnp.arange(np.prod(batch_shape), dtype=jnp.float32) + 2).reshape(batch_shape)
    actual = jax.jit(lambda old, value: scale(StridedView(old, (size,), (1,), 0), value).data)(base, factors)
    np.testing.assert_array_equal(actual, base * factors[..., None])


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.complex64])
def test_selected_scale_finite_zero_one_and_general_factors(dtype):
    pattern = [0., -0., 1.25, -1.25, .125, 1e10, -1e10, 1.25]
    factors = [0., -0., 1., -2., 1.25, -1.25, .125, 3.5]
    if dtype == jnp.complex64:
        pattern = [0j, complex(-0., 0.), 1.25 + 2j, -1.25 - 3j, .125 + 1j,
                   complex(1e10, -1e10), 1.25 - .75j, -2.5 + 4j]
        factors = [0j, complex(-0., 0.), 1 + 0j, -2 + 3j, 1.25 + 1j, .125 - 2j, 3.5 + 3.5j]
    base = jnp.resize(jnp.asarray(pattern, dtype=dtype), (50,))
    operation = jax.jit(lambda old, value: scale(StridedView(old, (16,), (3,), 2), value).data)
    for factor in factors:
        coefficient = jnp.asarray(factor, dtype=dtype)
        np.testing.assert_allclose(operation(base, coefficient), reference_scale(base, coefficient), rtol=2e-6, atol=1e-6)


def test_selected_scale_batched_factors_and_vmap_axes():
    base = jnp.linspace(-3, 4, 50, dtype=jnp.float32)
    factors = jnp.asarray([2., -.5, 3.], dtype=jnp.float32)
    bases = jnp.stack((base, 2 * base, -base))
    operation = lambda old, value: scale(StridedView(old, (16,), (3,), 2), value).data
    np.testing.assert_array_equal(jax.jit(operation)(bases, factors), reference_scale(bases, factors))
    for axes in ((0, 0), (0, None), (None, 0)):
        arguments = (bases if axes[0] == 0 else base, factors if axes[1] == 0 else factors[0])
        np.testing.assert_array_equal(jax.jit(jax.vmap(operation, in_axes=axes))(*arguments),
                                      jax.vmap(reference_scale, in_axes=axes)(*arguments))
