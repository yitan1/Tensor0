"""Standalone native C++ executable contracts and build support."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.stride.support.cpp_cache import compile_cached
from tests.stride.support.data import REDUCTION_DTYPES
from tests.stride.support.paths import REPO_ROOT


pytestmark = pytest.mark.skipif(sys.platform != 'linux', reason='native CPU contracts require Linux')


SOURCES = REPO_ROOT / "tests/stride/cpp"


@pytest.fixture(scope="session")
def native_build(pytestconfig):
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    if compiler is None:
        pytest.skip("a C++20 compiler is required")
    root = REPO_ROOT
    native = root / "crates/tensor0-py/native"
    vendor = root / "crates/tensor0-py/vendor/jaxlib-0.10.1/include"
    output = pytestconfig.cache.mkdir("stride-cpp")
    identity = {
        "compiler": subprocess.run([compiler, "-v"], check=True, capture_output=True,
                                   text=True, timeout=30).stderr,
        "compiler_stat": (Path(compiler).stat().st_size, Path(compiler).stat().st_mtime_ns),
        "environment": {name: os.environ.get(name) for name in (
            "PATH", "CPATH", "CPLUS_INCLUDE_PATH", "C_INCLUDE_PATH", "LIBRARY_PATH",
            "COMPILER_PATH", "GCC_EXEC_PREFIX", "LD_LIBRARY_PATH", "SDKROOT",
            "SOURCE_DATE_EPOCH",
        )},
    }
    # Conservatively invalidate on any native or vendored change. Test translation
    # units are tracked individually; shared test headers invalidate all contracts.
    dependencies = [path for directory in (native, vendor, SOURCES)
                    for path in directory.rglob("*") if path.is_file()
                    and (directory != SOURCES or path.suffix != ".cc")]
    flags = ["-std=c++20", "-Wall", "-Wextra", "-Wpedantic", "-Werror",
             "-fsanitize=undefined", "-fno-sanitize-recover=undefined", "-pthread",
             "-I", str(native), "-isystem", str(vendor)]
    objects = []
    for source in sorted(native.rglob("*.cc")):
        if source.parent.name == "ffi" and source.stem in {"copy", "update", "reduction", "dot"}:
            continue
        obj = output / (str(source.relative_to(native)).replace("/", "_") + ".o")
        compile_cached([compiler, "-O1", *flags, "-c", str(source)], obj,
                       inputs=dependencies, identity=identity, timeout=60)
        objects.append(str(obj))
    return compiler, native, output, flags, objects, dependencies, identity


@pytest.mark.parametrize("source", sorted(path.name for path in SOURCES.glob("*_test.cc")))
def test_cpp_contract(source, native_build):
    native, output = native_build[1:3]
    executable = output / Path(source).stem
    if source == "ffi_boundary_test.cc":
        import fcntl

        units = [(native / "ffi" / f"{name}.cc").read_text()
                 for name in ("copy", "update", "reduction", "dot")]
        header = "\n".join(unit.split("\n#define TENSOR0_STRIDE_DEFINE_", 1)[0] for unit in units)
        # Serialize header generation with this contract's compilation.
        with (output / "ffi-header.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            (output / "ffi_under_test.h").write_text(header)
            _compile_contract(source, native_build, executable)
    else:
        _compile_contract(source, native_build, executable)
    completed = subprocess.run([str(executable)], capture_output=True, text=True, timeout=120)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    if source == "promotion_test.cc":
        _check_promotion(completed.stdout)


def _compile_contract(source, native_build, executable):
    compiler, native, output, flags, objects, dependencies, identity = native_build
    compile_cached([
        compiler, "-O0" if source == "ffi_boundary_test.cc" else "-O1", *flags,
        "-I", str(output), "-iquote", str(native / "ffi"),
        str(SOURCES / source), *objects,
    ], executable, inputs=[*dependencies, SOURCES / source, *objects,
                           Path(__file__), REPO_ROOT / "tests/stride/support/cpp_cache.py"],
        identity=identity, timeout=300)


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
