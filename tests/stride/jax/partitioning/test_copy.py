"""JAX partitioning copy contracts."""

import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tensor0._stride import _jax

from tests.stride.support.availability import native_available
from tests.stride.support.paths import REPO_ROOT


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
def test_multi_device_copy_batch_replication_pmap_and_storage_boundary():
    environment = os.environ.copy()
    environment["JAX_PLATFORMS"] = "cpu"
    environment["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    completed = subprocess.run([sys.executable, "-c", "from tests.stride.support.workers.copy_partitioning import _check_sharding; _check_sharding()"],
                               cwd=REPO_ROOT, env=environment,
                               capture_output=True, text=True, timeout=180)
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_unknown_sharding_is_rejected_at_storage_boundary():
    with pytest.raises(ValueError, match="requires NamedSharding"):
        _jax._batch_storage_sharding(SimpleNamespace(shape=(2, 16), sharding=None))


@pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")
@pytest.mark.parametrize("mode", ["auto", "explicit"])
def test_mapping_batch_replication_and_storage_sharding_boundaries(mode):
    environment = os.environ.copy()
    environment["JAX_PLATFORMS"] = "cpu"
    environment["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    script = f"from tests.stride.support.workers.mapping_partitioning import _check_sharding; _check_sharding({mode!r})"
    completed = subprocess.run([sys.executable, "-c", script], cwd=REPO_ROOT,
                               env=environment, capture_output=True, text=True, timeout=180)
    assert completed.returncode == 0, completed.stdout + completed.stderr
