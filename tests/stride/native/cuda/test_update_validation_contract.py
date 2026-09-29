"""Native CUDA graph replay and combined-grid fallback for multi-record Update."""

import subprocess

import pytest

from tests.stride.support.cuda_contract import build_contract


@pytest.fixture(scope="module")
def update_validation_binary(tmp_path_factory):
    return build_contract("update_validation", tmp_path_factory)


@pytest.mark.parametrize("args", [[], ["fallback"]])
def test_cuda_update_validation(update_validation_binary, args):
    result = subprocess.run(
        [str(update_validation_binary), *args], text=True, capture_output=True, timeout=90
    )
    if result.returncode == 77:
        pytest.skip(result.stdout.strip())
    assert result.returncode == 0, result.stdout + result.stderr
    assert ("fallback 65536 blocks" if args else "graph 3 replay") in result.stdout
