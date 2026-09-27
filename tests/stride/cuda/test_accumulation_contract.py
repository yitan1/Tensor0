"""Compile and run the standalone CUDA Accumulation alias/boundary contract.

Uses the current interpreter's JAX headers and an existing CUDA toolkit only.
No extension rebuild, dependency installation, or JAX GPU backend is required.
"""

import pytest

from tests.stride.support.cuda_contract import build_contract, run_contract


@pytest.fixture(scope="module")
def accumulation_contract_binary(tmp_path_factory):
    return build_contract("accumulation", tmp_path_factory)


def test_cuda_accumulation_native_contract(accumulation_contract_binary):
    run_contract(accumulation_contract_binary, "accumulation")
