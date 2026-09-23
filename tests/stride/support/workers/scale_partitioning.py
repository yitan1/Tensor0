"""Isolated scale partitioning checks."""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensor0._stride import StridedView, scale
from tensor0._stride._ffi._registration import operation_target

from tests.stride.support.oracles.scale import reference_scale


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
