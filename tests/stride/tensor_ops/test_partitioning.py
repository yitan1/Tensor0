"""Two-device public transform contracts in isolated processes."""

import os
import subprocess
import sys

import pytest

from tests.stride.support.availability import native_available
from tests.stride.support.paths import REPO_ROOT


@pytest.mark.parametrize('operation,mode', [('transform', 'auto'), ('transform', 'explicit')])
def test_distributed_transform(operation, mode):
    if not native_available():
        pytest.skip("native CPU stride is unavailable")
    environment = dict(os.environ, JAX_PLATFORMS="cpu", XLA_FLAGS="--xla_force_host_platform_device_count=2")
    script = f"from tests.stride.support.workers.{operation}_partitioning import run_{operation}; run_{operation}({mode!r})"
    completed = subprocess.run([sys.executable, "-c", script],
                               cwd=REPO_ROOT, env=environment,
                               capture_output=True, text=True, timeout=240)
    assert completed.returncode == 0, completed.stdout + completed.stderr
