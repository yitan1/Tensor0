"""Isolated copy partitioning checks."""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensor0._stride._jax import copy_p
from tensor0._stride._layout import AffineRecord


def _check_sharding():
    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    assert len(jax.devices()) == 2
    mesh = Mesh(np.asarray(jax.devices()), ("device",))
    batches = NamedSharding(mesh, PartitionSpec("device", None))
    replicated = NamedSharding(mesh, PartitionSpec())
    partitioned = NamedSharding(mesh, PartitionSpec("device"))
    size = 131072
    record = AffineRecord((size,), (1,), 0, (1,), 0)
    apply = lambda value: copy_p.bind(value, records=(record,), output_size=size, dtype=value.dtype)
    host = np.arange(2 * size, dtype=np.float32).reshape(2, size)
    for values, sharding in ((host, batches), (host[0], replicated)):
        source = jax.device_put(values, sharding)
        compiled = jax.jit(apply, in_shardings=sharding, out_shardings=sharding).lower(source).compile()
        actual = compiled(source)
        np.testing.assert_array_equal(actual, values)
        assert actual.sharding.is_equivalent_to(sharding, values.ndim)
        text = compiled.as_text().lower()
        assert "tensor0_stride_copy_f32_cpu_v1" in text
        assert "all-gather" not in text and "all-reduce" not in text
    np.testing.assert_array_equal(jax.pmap(apply)(jnp.asarray(host)), host)
    source = jax.device_put(host[0], partitioned)
    with pytest.raises(Exception, match="cannot shard the packed storage axis"):
        jax.jit(apply, in_shardings=partitioned, out_shardings=partitioned).lower(source).compile()
    assert not any(name == "tensor0._stride1" or name.startswith("tensor0._stride1.") or
                   name == "tensor0.operations._strided" for name in sys.modules)
