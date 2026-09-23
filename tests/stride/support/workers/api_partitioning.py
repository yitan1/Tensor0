"""Isolated api partitioning checks."""

import sys

import jax
import numpy as np
import pytest

from tensor0._stride import StridedView, add, materialize, scale


def _check_eager_sharding():
    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    assert len(jax.devices()) == 2
    mesh = Mesh(np.asarray(jax.devices()), ("device",))
    partitioned = NamedSharding(mesh, PartitionSpec("device"))
    replicated = NamedSharding(mesh, PartitionSpec())
    storage = jax.device_put(np.arange(8, dtype=np.float32), partitioned)
    local_storage = jax.device_put(np.arange(8, dtype=np.float32), replicated)
    update = jax.device_put(np.asarray([10, 20], dtype=np.float32), replicated)
    sharded_update = jax.device_put(np.asarray([10, 20], dtype=np.float32), partitioned)
    factor = jax.device_put(np.float32(2), replicated)
    operations = [
        lambda: materialize(StridedView(storage, (2,), (2,), 0)),
        lambda: scale(StridedView(storage, (2,), (2,), 0), factor).data,
    ]
    for operation in operations:
        with pytest.raises(Exception, match="cannot shard the packed storage axis"):
            operation().block_until_ready()
    for alpha in (0, 1):
        for base, source in ((storage, update), (local_storage, sharded_update)):
            with pytest.raises(Exception, match="cannot shard the packed storage axis"):
                add(StridedView(base, (2,), (2,), 0), StridedView.from_dense(source, (2,)),
                    alpha=alpha, beta=1).data.block_until_ready()
    assert not any(name == "tensor0._stride1" or name.startswith("tensor0._stride1.") or
                   name == "tensor0.operations._strided" for name in sys.modules)
