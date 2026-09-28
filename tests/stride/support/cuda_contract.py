"""Build and execute standalone CUDA FFI contracts without installing an extension."""

import os
from pathlib import Path
import shutil
import subprocess

import jax.ffi
import pytest

from tests.stride.support.paths import REPO_ROOT


def build_contract(name, tmp_path_factory):
    configured = os.environ.get("NVCC")
    cuda_home = os.environ.get("CUDA_HOME")
    nvcc = configured or (str(Path(cuda_home) / "bin" / "nvcc") if cuda_home else "nvcc")
    compiler = shutil.which(nvcc)
    if compiler is None:
        pytest.skip("CUDA toolkit nvcc is unavailable")

    native = REPO_ROOT / "crates" / "tensor0-py" / "native"
    executable = tmp_path_factory.mktemp(f"cuda-{name}-contract") / f"{name}_contract"
    command = [
        compiler, "-std=c++20", "-O0", "--cudart=static",
        "-arch=" + os.environ.get("TENSOR0_CUDA_ARCH", "sm_80"),
        "-I", str(native), "-I", jax.ffi.include_dir(),
        str(REPO_ROOT / "tests" / "stride" / "native" / "cuda" / f"{name}_contract.cu"),
        str(native / "layout" / "descriptor.cc"),
        str(native / "layout" / "record.cc"),
        "-o", str(executable),
    ]
    compiled = subprocess.run(command, text=True, capture_output=True, timeout=180)
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    return executable


def run_contract(executable, name):
    completed = subprocess.run([str(executable)], text=True, capture_output=True, timeout=60)
    if completed.returncode == 77:
        pytest.skip(completed.stdout.strip())
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert f"CUDA {name.title()} contract passed" in completed.stdout
