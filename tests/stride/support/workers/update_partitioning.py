"""Isolated update partitioning checks."""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensor0._stride._jax import update_p

from tests.stride.support.layouts import PARTIAL


def _check_sharding(mode):
    from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec

    mesh = Mesh(np.asarray(jax.devices()), ("device",),
                axis_types=(AxisType.Explicit if mode == "explicit" else AxisType.Auto,))
    assert len(jax.devices()) == 2
    batches = NamedSharding(mesh, PartitionSpec("device", None))
    base_host = np.arange(100, dtype=np.float32).reshape(2, 50)
    source_host = np.arange(64, dtype=np.float32).reshape(2, 32)
    base, source = jax.device_put(base_host, batches), jax.device_put(source_host, batches)
    for beta in (0, 1):
        operation = lambda old, new: update_p.bind(new, old, jnp.float32(.5), jnp.int32(beta), records=PARTIAL)
        executable = jax.jit(operation, in_shardings=(batches, batches), out_shardings=batches).lower(base, source).compile()
        actual = executable(base, source)
        expected = base_host.copy()
        expected[:, 2:50:3] = .5 * source_host[:, 1:32:2] + beta * base_host[:, 2:50:3]
        np.testing.assert_array_equal(actual, expected)
        assert actual.sharding.is_equivalent_to(batches, 2)
        text = executable.as_text().lower()
        assert "tensor0_stride_update_f32_cpu_v1" in text
        assert "all-gather" not in text and "all-reduce" not in text
    for operand in ("source", "base", "output"):
        partitioned = NamedSharding(mesh, PartitionSpec("device"))
        replicated = NamedSharding(mesh, PartitionSpec())
        shardings = (partitioned if operand == "base" else replicated,
                     partitioned if operand == "source" else replicated)
        output = partitioned if operand == "output" else replicated
        arguments = tuple(jax.device_put(value, sharding) for value, sharding in
                          zip((base_host[0], source_host[0]), shardings, strict=True))
        compiled = jax.jit(operation, in_shardings=shardings, out_shardings=output)
        if operand == "output" and mode == "explicit":
            executable = compiled.lower(*arguments).compile()
            actual = executable(*arguments)
            expected = base_host[0].copy()
            expected[2:50:3] += .5 * source_host[0, 1:32:2]
            np.testing.assert_array_equal(actual, expected)
            assert actual.sharding.is_equivalent_to(output, 1)
            calls = [line for line in executable.as_text().splitlines()
                     if 'custom_call_target="tensor0_stride_update_f32_cpu_v1"' in line]
            assert len(calls) == 1 and "f32[1,50]" in calls[0]
        else:
            with pytest.raises(Exception, match="cannot shard the packed storage axis"):
                compiled.lower(*arguments).compile()
    assert not any(name == "tensor0._stride1" or name.startswith("tensor0._stride1.") or
                   name == "tensor0.operations._strided" for name in sys.modules)
