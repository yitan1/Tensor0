"""Isolated mapping partitioning checks."""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensor0._stride._jax import accumulation_p, copy_p

from tests.stride.support.layouts import PARTITIONS


def _check_sharding(mode):
    from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec

    assert len(jax.devices()) == 2
    mesh = Mesh(np.asarray(jax.devices()), ("device",),
                axis_types=(AxisType.Explicit if mode == "explicit" else AxisType.Auto,))
    batches = NamedSharding(mesh, PartitionSpec("device", None))
    replicated = NamedSharding(mesh, PartitionSpec())
    partitioned = NamedSharding(mesh, PartitionSpec("device"))
    host = np.arange(64, dtype=np.float32).reshape(4, 16)
    order = np.arange(16).reshape(4, 4)[:, [2, 3, 0, 1]].ravel()
    for scaled in (False, True):
        def operation(value):
            if scaled:
                return accumulation_p.bind(value, jnp.float32(.75), jnp.float32(.75), records=PARTITIONS,
                    coefficient_records=(0, 1), output_size=16, dtype=value.dtype)
            return copy_p.bind(value, records=PARTITIONS, output_size=16, dtype=value.dtype)

        for sharding in (batches, replicated):
            source = jax.device_put(host, sharding)
            executable = jax.jit(operation, in_shardings=sharding, out_shardings=sharding).lower(source).compile()
            actual = executable(source)
            np.testing.assert_array_equal(actual, host[:, order] * (.75 if scaled else 1))
            assert actual.sharding.is_equivalent_to(sharding, 2)
            text = executable.as_text().lower()
            assert "tensor0_stride_" in text
            assert "all-gather" not in text and "all-reduce" not in text
        source = jax.device_put(host[0], partitioned)
        with pytest.raises(Exception, match="cannot shard the packed storage axis"):
            jax.jit(operation, in_shardings=partitioned, out_shardings=replicated).lower(source).compile()
    assert not any(name == "tensor0._stride1" or name.startswith("tensor0._stride1.") or
                   name == "tensor0.operations._strided" for name in sys.modules)
