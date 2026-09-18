"""Standalone native contracts, including UBSan and concrete JAX promotion."""

import os
from pathlib import Path
import shutil
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from .test_native_reduction import REDUCTION_DTYPES


SOURCES = Path(__file__).with_name("cpp")
pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="native CPU contracts require Linux")


@pytest.mark.parametrize("source", sorted(path.name for path in SOURCES.glob("*_test.cc")))
def test_cpp_contract(tmp_path, source):
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    if compiler is None:
        pytest.skip("a C++20 compiler is required")
    root = Path(__file__).resolve().parents[2]
    native = root / "crates/tensor0-py/native"
    executable = tmp_path / "contract"
    if source == "ffi_boundary_test.cc":
        unit = (native / "stride_ffi.cc").read_text()
        bindings = (native / "ffi/bindings.inc").read_text().split("XLA_FFI_DEFINE_HANDLER_SYMBOL(", 1)[0]
        (tmp_path / "ffi_under_test.inc").write_text(unit.replace('#include "ffi/bindings.inc"', bindings))
    subprocess.run([
        compiler, "-std=c++20", "-O0" if source == "ffi_boundary_test.cc" else "-O1",
        "-Wall", "-Wextra", "-Wpedantic", "-Werror",
        "-fsanitize=undefined", "-fno-sanitize-recover=undefined", "-pthread",
        "-include", str(native / "kernels/avx2.inc"), "-I", str(native), "-I", str(tmp_path),
        "-isystem", str(root / "crates/tensor0-py/vendor/jaxlib-0.10.1/include"),
        str(SOURCES / source), "-o", str(executable),
    ], check=True, capture_output=True, text=True, timeout=300)
    completed = subprocess.run([str(executable)], capture_output=True, text=True, timeout=120)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    if source == "promotion_test.cc":
        _check_promotion(completed.stdout)


def _check_promotion(output):
    rows = output.splitlines()
    assert len(rows) == len(REDUCTION_DTYPES) ** 2
    with jax.enable_x64(), jax.numpy_dtype_promotion("standard"):
        real = jnp.asarray([-2.5, 0, 1.25, 7.5, 1.0003, 127], dtype=jnp.float64)
        imaginary = jnp.asarray([0.25, -0.5, 1.5, 3, -2, 0], dtype=jnp.float64)

        def sample(name, values):
            if jnp.issubdtype(jnp.dtype(name), jnp.complexfloating):
                component = jnp.float32 if name == "complex64" else jnp.float64
                return jax.lax.complex(values.astype(component), imaginary.astype(component))
            return values.astype(name)

        for row in rows:
            fields = row.split()
            left_index, right_index, result_index = map(int, fields[:3])
            left_name, right_name = REDUCTION_DTYPES[left_index], REDUCTION_DTYPES[right_index]
            expected_dtype = jnp.result_type(jnp.dtype(left_name), jnp.dtype(right_name))
            assert REDUCTION_DTYPES[result_index] == expected_dtype.name
            left, right = sample(left_name, real), sample(right_name, real[::-1])
            actual = np.asarray(fields[3:], dtype=np.float64).reshape(6, 2, 2)
            for operation_index, operation in enumerate((jnp.add, jnp.multiply)):
                expected = operation(left, right)
                for component_index, component in enumerate((jnp.real, jnp.imag)):
                    np.testing.assert_allclose(
                        actual[:, operation_index, component_index],
                        np.asarray(component(expected), dtype=np.float64),
                        rtol=2e-6, atol=1e-12,
                        err_msg=f"{left_name} {operation.__name__} {right_name}",
                    )
