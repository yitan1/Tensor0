"""JAX partitioning api contracts."""

import os
import subprocess
import sys

import pytest

from tests.stride.support.availability import native_available
from tests.stride.support.paths import REPO_ROOT


pytestmark = pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')


def test_concrete_eager_stride_operations_reject_packed_storage_sharding():
    environment = os.environ.copy()
    environment["JAX_PLATFORMS"] = "cpu"
    environment["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    completed = subprocess.run(
        [sys.executable, "-c", "from tests.stride.support.workers.api_partitioning import _check_eager_sharding; _check_eager_sharding()"],
        cwd=REPO_ROOT, env=environment, capture_output=True, text=True, timeout=180,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
