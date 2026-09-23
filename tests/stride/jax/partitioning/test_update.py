"""JAX partitioning update contracts."""

import os
import subprocess
import sys

import pytest

from tests.stride.support.availability import native_available
from tests.stride.support.paths import REPO_ROOT


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
def test_selected_scale_multi_device_batch_sharding_and_storage_rejection():
    environment = os.environ.copy()
    environment["JAX_PLATFORMS"] = "cpu"
    environment["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    completed = subprocess.run(
        [sys.executable, "-c", "from tests.stride.support.workers.scale_partitioning import _check_sharding; _check_sharding()"],
        cwd=REPO_ROOT, env=environment, capture_output=True, text=True, timeout=180,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize("mode", ["auto", "explicit"])
def test_update_batch_sharding_and_storage_boundary(mode):
    environment = os.environ.copy()
    environment["JAX_PLATFORMS"] = "cpu"
    environment["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    script = f"from tests.stride.support.workers.update_partitioning import _check_sharding; _check_sharding({mode!r})"
    completed = subprocess.run([sys.executable, "-c", script], cwd=REPO_ROOT,
                               env=environment, capture_output=True, text=True, timeout=180)
    assert completed.returncode == 0, completed.stdout + completed.stderr
