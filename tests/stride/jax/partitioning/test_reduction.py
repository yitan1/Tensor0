"""JAX partitioning reduction contracts."""

import os
import subprocess
import sys

import pytest

from tests.stride.support.paths import REPO_ROOT


@pytest.mark.parametrize("mode", ["auto", "explicit", "input_storage", "output_storage"])
def test_two_cpu_reduction(mode):
    environment = os.environ.copy()
    environment["JAX_PLATFORMS"] = "cpu"
    environment["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    script = f"from tests.stride.support.workers.reduction_partitioning import _run_reduction_worker; _run_reduction_worker({mode!r})"
    completed = subprocess.run([sys.executable, "-c", script], cwd=REPO_ROOT,
                               env=environment, capture_output=True, text=True, timeout=180)
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize("operation", ["reduction", "accumulation"])
@pytest.mark.parametrize("mode", ["auto", "explicit"])
def test_two_cpu_mapped_coefficients(operation, mode):
    environment = os.environ.copy()
    environment["JAX_PLATFORMS"] = "cpu"
    environment["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    script = f"from tests.stride.support.workers.reduction_partitioning import _run_mapped_coefficients_worker; _run_mapped_coefficients_worker({operation!r}, {mode!r})"
    result = subprocess.run([sys.executable, "-c", script], cwd=REPO_ROOT,
                            env=environment, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
