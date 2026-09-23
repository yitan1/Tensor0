"""Public Update coefficients, storage and validation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensor0._stride import StridedView, add, scale
from tensor0._stride._jax import update_p

from tests.stride.support.availability import native_available
from tests.stride.support.oracles.update import assert_disjoint_writes


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16, jnp.float32,
                                    jnp.float64, jnp.complex64, jnp.complex128])
@pytest.mark.parametrize("source_factor,base_factor",
                         [(source, base) for source in (0, 1, 2)
                          for base in (0, 1, -2)])
def test_finite_zero_one_contract_is_identical_for_static_and_batched_factors(
    dtype, source_factor, base_factor,
):
    with jax.enable_x64():
        previous = jnp.asarray([[7, -0.0, -3, -0.0],
                                [-2, -0.0, 5, -0.0]], dtype=dtype)
        source = jnp.asarray([[3, -0.0], [-0.0, 4]], dtype=dtype)
        left = StridedView(previous, (2,), (2,), 0)
        right = StridedView(source, (2,), (1,), 0)
        factors = (jnp.full((2,), base_factor, dtype=dtype),
                   jnp.full((2,), source_factor, dtype=dtype))
        function = jax.jit(lambda old, new, lhs, rhs:
                           add(old, new, alpha=lhs, beta=rhs).data)
        outputs = (add(left, right, alpha=base_factor, beta=source_factor).data,
                   function(left, right, *factors))
        terms = []
        for factor, values in ((source_factor, source),
                               (base_factor, previous[:, ::2])):
            if factor != 0:
                terms.append(values if factor == 1 else factor * values)
        selected = (jnp.zeros_like(source) if not terms else terms[0]
                    if len(terms) == 1 else terms[0] + terms[1])
        expected = previous.at[:, ::2].set(selected)
        for output in outputs:
            np.testing.assert_allclose(output, expected, equal_nan=True)
            np.testing.assert_array_equal(
                np.asarray(output)[:, 1::2].copy().view(np.uint8),
                np.asarray(previous)[:, 1::2].copy().view(np.uint8),
            )


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16, jnp.float32,
                                    jnp.float64, jnp.complex64, jnp.complex128])
@pytest.mark.parametrize("source_factor,base_factor", [(0, 0), (0, 1), (1, 0)])
def test_zero_terms_skip_nonfinite_input_and_preserve_unselected_bits(dtype, source_factor, base_factor):
    with jax.enable_x64():
        source = jnp.asarray([[1, 2], [3, 4]] if source_factor else
                             [[jnp.nan, jnp.inf], [jnp.inf, jnp.nan]], dtype=dtype)
        selected_base = jnp.asarray([[7, 8], [9, 10]] if base_factor else
                                    [[jnp.inf, jnp.nan], [jnp.nan, jnp.inf]], dtype=dtype)
        base = jnp.asarray([[0, -0.0, 0, jnp.nan], [0, jnp.nan, 0, -0.0]], dtype=dtype)
        base = base.at[:, ::2].set(selected_base)
        left = StridedView(base, (2,), (2,), 0)
        right = StridedView(source, (2,), (1,), 0)
        expected = source if source_factor else selected_base if base_factor else jnp.zeros_like(source)
        dynamic = jax.jit(lambda old, new, alpha, beta: add(old, new, alpha=alpha, beta=beta).data)
        results = (add(left, right, alpha=base_factor, beta=source_factor).data,
                   dynamic(left, right, jnp.full((2,), base_factor, dtype=dtype),
                           jnp.full((2,), source_factor, dtype=dtype)))
        for result in results:
            np.testing.assert_array_equal(result[:, ::2], expected)
            np.testing.assert_array_equal(np.asarray(result)[:, 1::2].copy().view(np.uint8),
                                          np.asarray(base)[:, 1::2].copy().view(np.uint8))


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.complex64])
@pytest.mark.parametrize("factor", [0, 1])
def test_scale_short_circuit_preserves_unit_bits_and_zeroes_nonfinite(dtype, factor):
    values = jnp.asarray([complex(jnp.inf, -0.0), complex(jnp.nan, 2),
                          complex(-0.0, -0.0)], dtype=jnp.complex64)
    if dtype == jnp.float32:
        values = values.real
    view = StridedView.from_dense(values, (3,))
    expected = values if factor else jnp.zeros_like(values)
    for output in (scale(view, factor).data,
                   jax.jit(lambda value: scale(view, value).data)(
                       jnp.asarray(factor, dtype=dtype))):
        np.testing.assert_array_equal(np.asarray(output).view(np.uint8),
                                      np.asarray(expected).view(np.uint8))


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
def test_strided_updates_reject_non_jax_array_operands():
    base = jnp.arange(8, dtype=jnp.float32)
    numpy_base = np.arange(8, dtype=np.float32)
    numpy_source = np.asarray([10, 20], dtype=np.float32)
    with pytest.raises(TypeError, match="data must be a JAX Array"):
        StridedView(numpy_base, (2,), (2,), 0)
    with pytest.raises(TypeError, match="data must be a JAX Array"):
        StridedView.from_dense(numpy_source, (2,))
    destination = StridedView(base, (2,), (2,), 0)
    for alpha in (0, 1):
        with pytest.raises(TypeError, match="right must be a StridedView"):
            add(destination, numpy_source, alpha=alpha, beta=1)
    with pytest.raises(TypeError, match="view must be a StridedView"):
        scale(numpy_base, 2)


@pytest.fixture(autouse=False)
def enable_x64():
    with jax.enable_x64():
        yield


@pytest.mark.usefixtures("enable_x64")
@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
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
