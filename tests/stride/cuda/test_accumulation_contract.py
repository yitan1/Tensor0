"""Compile and run the standalone CUDA Accumulation alias/boundary contract.

Uses the current interpreter's JAX headers and an existing CUDA toolkit only.
No extension rebuild, dependency installation, or JAX GPU backend is required.
"""

import os
from pathlib import Path
import shutil
import subprocess

import jax.ffi
import pytest


@pytest.fixture(scope="module")
def accumulation_contract_binary(tmp_path_factory):
    configured = os.environ.get("NVCC")
    cuda_home = os.environ.get("CUDA_HOME")
    nvcc = configured or (str(Path(cuda_home) / "bin" / "nvcc") if cuda_home else "nvcc")
    compiler = shutil.which(nvcc)
    if compiler is None:
        pytest.skip("CUDA toolkit nvcc is unavailable")

    root = Path(__file__).resolve().parents[3]
    native = root / "crates" / "tensor0-py" / "native"
    executable = tmp_path_factory.mktemp("cuda-accumulation-contract") / "accumulation_contract"
    command = [
        compiler, "-std=c++20", "-O0", "--cudart=static",
        "-arch=" + os.environ.get("TENSOR0_CUDA_ARCH", "sm_80"),
        "-I", str(native), "-I", jax.ffi.include_dir(),
        str(Path(__file__).with_name("accumulation_contract.cu")),
        str(native / "ffi" / "prepared.cc"),
        str(native / "layout" / "record.cc"),
        str(native / "layout" / "traversal.cc"),
        "-o", str(executable),
    ]
    compiled = subprocess.run(command, text=True, capture_output=True, timeout=180)
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    return executable


def test_cuda_accumulation_native_contract(accumulation_contract_binary):
    completed = subprocess.run(
        [str(accumulation_contract_binary)], text=True, capture_output=True, timeout=60,
    )
    if completed.returncode == 77:
        pytest.skip(completed.stdout.strip())
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "CUDA Accumulation contract passed" in completed.stdout
