from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensor0._stride import StridedView, scale
from tensor0._stride._ffi._registration import operation_target
from tensor0._stride._jax import update_p
from tensor0._stride._layout import AffineRecord

from ._support import native_available


pytestmark = pytest.mark.skipif(not native_available(), reason="native CPU stride is unavailable")


def reference_scale(base, factor):
    return base.at[..., 2:50:3].set(base[..., 2:50:3] * factor[..., None])


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


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16, jnp.float32,
                                   jnp.float64, jnp.complex64, jnp.complex128])
def test_selected_scale_factor_jvp_uses_one_accumulation(dtype):
    with jax.enable_x64():
        base = jnp.arange(50, dtype=jnp.float32).astype(dtype) - 7
        complex_storage = jnp.issubdtype(dtype, jnp.complexfloating)
        factor = jnp.asarray(1.25 - .5j if complex_storage else -1.25, dtype=dtype)
        direction = jnp.asarray(.75 + .25j if complex_storage else .75, dtype=dtype)
        operation = lambda value: scale(StridedView(base, (16,), (3,), 2), value).data
        tangent = jax.jit(lambda value, dot: jax.jvp(operation, (value,), (dot,))[1])
        expected = jax.jvp(lambda value: reference_scale(base, value), (factor,), (direction,))[1]
        np.testing.assert_allclose(tangent(factor, direction), expected, rtol=5e-3, atol=5e-3)
        text = tangent.lower(factor, direction).as_text()
        assert text.count("custom_call") == 1
        assert operation_target("accumulation", np.dtype(dtype)) in text


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


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.complex64])
def test_selected_scale_ad_transposes_and_higher_order_compositions(dtype):
    base = jnp.linspace(-3, 4, 50, dtype=jnp.float32).astype(dtype)
    cotangent = jnp.linspace(-4, 3, 50, dtype=jnp.float32).astype(dtype)
    factor = jnp.asarray(-1.25 if dtype == jnp.float32 else 1.25 - .75j, dtype=dtype)
    if dtype == jnp.complex64:
        base = base * (1 + .5j)
        cotangent = cotangent * (.75 - 1.25j)
    directions = (jnp.full_like(base, .5), jnp.asarray(.75, dtype=dtype))
    operation = lambda old, value: scale(StridedView(old, (16,), (3,), 2), value).data

    def derivatives(function, old, value, cot):
        primal, tangent = jax.jvp(function, (old, value), directions)
        gradients = jax.vjp(function, old, value)[1](cot)
        base_transpose = jax.linear_transpose(lambda data: function(data, value), old)(cot)[0]
        factor_transpose = jax.linear_transpose(lambda coefficient: function(old, coefficient), value)(cot)[0]
        pullback = lambda output_cot: jax.vjp(lambda data: function(data, value), old)[1](output_cot)[0]
        nested = jax.jvp(pullback, (jnp.zeros_like(cot),), (jnp.ones_like(cot),))
        factor_gradient = lambda data: jax.vjp(lambda coefficient: function(data, coefficient), value)[1](cot)[0]
        mixed = jax.jvp(factor_gradient, (old,), (directions[0],))
        return primal, tangent, gradients, base_transpose, factor_transpose, nested, mixed

    actual = jax.jit(lambda old, value, cot: derivatives(operation, old, value, cot))(base, factor, cotangent)
    expected = derivatives(reference_scale, base, factor, cotangent)
    for result, wanted in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(result, wanted, rtol=2e-6, atol=2e-6)


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


@pytest.mark.parametrize("donate", [False, True])
def test_selected_scale_dynamic_operand_and_input_reuse(donate):
    operation = jax.jit(lambda old, factor: scale(StridedView(old, (16,), (3,), 2), factor).data,
                        donate_argnums=(0,) if donate else ())
    template = jnp.arange(50, dtype=jnp.float32)
    lowered = operation.lower(template, jnp.float32(2))
    text = lowered.as_text()
    assert text.count("custom_call") == 1 and "output_operand_aliases" in text
    assert operation_target("update", np.dtype(jnp.float32)) in text
    executable = lowered.compile()
    memory = executable.memory_analysis()
    assert memory is not None
    assert memory.alias_size_in_bytes == (template.nbytes if donate else 0)
    for coefficient in (jnp.float32(2), jnp.float32(-3)):
        base = jnp.array(template, copy=True)
        expected = reference_scale(template, coefficient)
        np.testing.assert_array_equal(executable(base, coefficient), expected)
        if donate:
            assert base.is_deleted()
        else:
            np.testing.assert_array_equal(base, template)


@pytest.mark.parametrize("stride", [1, 2])
def test_selected_scale_large_layout_lowers_without_address_arrays(stride):
    size = 1_100_000
    base = jax.ShapeDtypeStruct((stride * (size - 1) + 1,), jnp.float32)
    operation = jax.jit(lambda old, factor: scale(StridedView(old, (size,), (stride,), 0), factor).data)
    text = operation.lower(base, jax.ShapeDtypeStruct((), jnp.float32)).as_text().lower()
    assert text.count("custom_call") == 1
    assert operation_target("update", np.dtype(jnp.float32)) in text
    assert "gather" not in text and "scatter" not in text


def test_selected_scale_non_cpu_lowering_fails_closed():
    operation = jax.jit(lambda old, factor: scale(StridedView(old, (16,), (3,), 2), factor).data)
    traced = operation.trace(jax.ShapeDtypeStruct((50,), jnp.float32), jax.ShapeDtypeStruct((), jnp.float32))
    with pytest.raises((RuntimeError, NotImplementedError, ValueError), match="tpu|TPU"):
        traced.lower(lowering_platforms=("tpu",))


def _check_sharding():
    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    assert len(jax.devices()) == 2
    mesh = Mesh(np.asarray(jax.devices()), ("device",))
    batches = NamedSharding(mesh, PartitionSpec("device", None))
    partitioned = NamedSharding(mesh, PartitionSpec("device"))
    replicated = NamedSharding(mesh, PartitionSpec())
    base_host = np.arange(100, dtype=np.float32).reshape(2, 50)
    base = jax.device_put(base_host, batches)
    factors = jax.device_put(np.asarray([2., -3.], dtype=np.float32), partitioned)
    operation = lambda old, factor: scale(StridedView(old, (16,), (3,), 2), factor).data
    for coefficient in (factors, jax.device_put(np.float32(2), replicated)):
        executable = jax.jit(operation, in_shardings=(batches, coefficient.sharding),
                             out_shardings=batches).lower(base, coefficient).compile()
        actual = executable(base, coefficient)
        np.testing.assert_array_equal(actual, reference_scale(jnp.asarray(base_host), jnp.asarray(coefficient)))
        assert actual.sharding.is_equivalent_to(batches, 2)
        tangent = jax.jit(lambda old, value: jax.jvp(lambda data: operation(data, value),
                           (old,), (jnp.ones_like(old),))[1],
                          in_shardings=(batches, coefficient.sharding), out_shardings=batches)
        derivative = tangent.lower(base, coefficient).compile()
        np.testing.assert_array_equal(derivative(base, coefficient), reference_scale(jnp.ones_like(base_host), jnp.asarray(coefficient)))
        for compiled in (executable, derivative):
            text = compiled.as_text().lower()
            assert operation_target("update", np.dtype(jnp.float32)) in text
            assert "all-gather" not in text and "all-reduce" not in text
    storage = jax.device_put(base_host[0], partitioned)
    coefficient = jax.device_put(np.float32(2), replicated)
    with pytest.raises(Exception, match="cannot shard the packed storage axis"):
        jax.jit(operation, in_shardings=(partitioned, replicated), out_shardings=partitioned).lower(storage, coefficient).compile()
    assert not any(name == "tensor0._stride1" or name.startswith("tensor0._stride1.") or
                   name == "tensor0.operations._strided" for name in sys.modules)


def test_selected_scale_multi_device_batch_sharding_and_storage_rejection():
    environment = os.environ.copy()
    environment["JAX_PLATFORMS"] = "cpu"
    environment["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    completed = subprocess.run(
        [sys.executable, "-c", "from tests.stride.test_selected_scale import _check_sharding; _check_sharding()"],
        cwd=Path(__file__).resolve().parents[2], env=environment, capture_output=True, text=True, timeout=180,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
