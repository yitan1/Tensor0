"""Compile and run the standalone CUDA Dot alias/boundary contract.

Uses the current interpreter's JAX headers and an existing CUDA toolkit only.
No extension rebuild, dependency installation, or JAX GPU backend is required.
"""

import pytest

from tests.stride.support.cuda_contract import build_contract, run_contract


@pytest.fixture(scope="module")
def dot_contract_binary(tmp_path_factory):
    return build_contract("dot", tmp_path_factory)


def test_cuda_dot_native_contract(dot_contract_binary):
    run_contract(dot_contract_binary, "dot")
