"""Compile and run the standalone CUDA Copy prepared-state/boundary contract.

Uses the current interpreter's JAX headers and an existing CUDA toolkit only.
No extension rebuild, dependency installation, or JAX GPU backend is required.
"""

import subprocess

import pytest

from tests.stride.support.cuda_contract import build_contract, run_contract


@pytest.fixture(scope="module")
def copy_contract_binary(tmp_path_factory):
    return build_contract("copy", tmp_path_factory)


def test_cuda_copy_native_contract(copy_contract_binary):
    run_contract(copy_contract_binary, "copy")


def test_cuda_copy_grid_fallback(copy_contract_binary):
    completed = subprocess.run([str(copy_contract_binary), "--fallback"],
                               text=True, capture_output=True, timeout=90)
    if completed.returncode == 77:
        pytest.skip(completed.stdout.strip())
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "CUDA Copy contract passed" in completed.stdout
