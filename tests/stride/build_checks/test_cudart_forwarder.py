"""Host-only ABI/ELF diagnostics; fake runtimes do not execute CUDA kernels."""

import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from tests.stride.support.paths import REPO_ROOT


@pytest.fixture(scope="module")
def forwarder_elf(tmp_path_factory):
    compiler = shutil.which("g++")
    home = os.environ.get("CUDA_HOME")
    if sys.platform != "linux" or not compiler or not home:
        pytest.skip("host ABI tests require g++ and CUDA_HOME SDK headers")
    include = Path(home) / "include"
    if not (include / "cuda_runtime_api.h").is_file():
        pytest.skip("CUDA_HOME must contain real CUDA SDK headers")
    directory = tmp_path_factory.mktemp("cudart-forwarder")
    native = REPO_ROOT / "crates/tensor0-cuda12/native"
    fixtures = Path(__file__).with_name("fixtures") / "cudart"

    def compile(*args, success=True):
        result = subprocess.run([compiler, "-std=c++20", "-O2", "-fPIC", "-pthread",
                                 "-I", str(include), *map(str, args)], capture_output=True, text=True)
        assert (result.returncode == 0) == success, result.stderr
        return result

    compile("-c", native / "cudart_forwarder.cc", "-o", directory / "forwarder.o")
    compile("-c", fixtures / "registration.cc", "-o", directory / "registration.o")
    for order in ("forward", "reverse"):
        objects = [directory / "forwarder.o", directory / "registration.o"]
        if order == "reverse":
            objects.reverse()
        compile("-shared", *objects, "-Wl,-z,defs", f"-Wl,--version-script={native / 'cudart_forwarder.map'}",
                "-ldl", "-o", directory / f"{order}.so")
    for variant, flag in (("complete", ""), ("missing-normal", "-DMISSING_NORMAL"),
                          ("missing-registration", "-DMISSING_REGISTRATION")):
        compile("-shared", *([flag] if flag else []), fixtures / "fake_runtime.cc",
                "-Wl,-soname,libcudart.so.12", "-o", directory / f"{variant}.so")
    compile("-shared", fixtures / "interceptors.cc", "-o", directory / "interceptors.so")
    # Production has no device variable import. A used NVCC global must force review,
    # not silently add an ABI to the runtime forwarder.
    compile("-DTEST_DEVICE_VAR", "-c", fixtures / "registration.cc", "-o", directory / "variable.o")
    failure = compile("-shared", directory / "forwarder.o", directory / "variable.o",
                      "-Wl,-z,defs", "-ldl", "-o", directory / "must-not-link.so", success=False)
    assert "__cudaRegisterVar" in failure.stderr
    return directory


@pytest.mark.parametrize("order", ["forward", "reverse"])
@pytest.mark.parametrize("runtime,interposed", [("complete", False), ("complete", True),
                                                ("missing-normal", False),
                                                ("missing-registration", False), (None, False)])
def test_constructor_status_failure_and_local_binding(forwarder_elf, order, runtime, interposed):
    directory = forwarder_elf
    command = [sys.executable, "-I", "-c", '''
import ctypes, os, sys
from concurrent.futures import ThreadPoolExecutor
root, order, runtime, interposed = sys.argv[1:]
retained = []
if interposed == 'True':
    retained.append(ctypes.CDLL(root + '/interceptors.so', mode=os.RTLD_NOW | os.RTLD_GLOBAL))
if runtime != 'None':
    retained.append(ctypes.CDLL(root + '/' + runtime + '.so', mode=os.RTLD_NOW | os.RTLD_GLOBAL))
library = ctypes.CDLL(root + '/' + order + '.so', mode=os.RTLD_NOW | os.RTLD_LOCAL)
status = library.Tensor0CudaForwarderStatusV1
status.restype = ctypes.c_int
bound = library.Tensor0CudaForwarderRuntimeV1
bound.restype = ctypes.c_void_p
error = library.Tensor0CudaForwarderErrorV1
error.restype = ctypes.c_char_p
expected = 0 if runtime == 'complete' else 302
with ThreadPoolExecutor(max_workers=8) as pool:
    assert list(pool.map(lambda _: status(), range(64))) == [expected] * 64
assert library.Tensor0FixtureLastError() == (123 if expected == 0 else 302)
if expected == 0:
    assert bound() == retained[-1]._handle
else:
    assert b'restart required' in error()
    # Constructor failure is sticky even after the complete runtime is loaded.
    retained.append(ctypes.CDLL(root + '/complete.so', mode=os.RTLD_NOW | os.RTLD_GLOBAL))
    assert status() == 302
import _ctypes
_ctypes.dlclose(library._handle)
''', str(directory), order, str(runtime), str(interposed)]
    env = {key: value for key, value in os.environ.items() if key not in ("LD_LIBRARY_PATH", "LD_PRELOAD")}
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
    expected = ("registrations=1 ends=1 functions=1 variables=0 unregistrations=1"
                if runtime == "complete" else "registrations=0 ends=0 functions=0 variables=0 unregistrations=0")
    assert expected in result.stderr


@pytest.mark.parametrize("order", ["forward", "reverse"])
def test_elf_local_wrappers_no_cuda_dependencies(forwarder_elf, order):
    library = forwarder_elf / f"{order}.so"
    dynamic = subprocess.check_output(["readelf", "-dW", str(library)], text=True)
    assert "libcudart" not in dynamic and "libcuda" not in dynamic
    assert "RPATH" not in dynamic and "RUNPATH" not in dynamic
    exports = subprocess.check_output(["nm", "-D", str(library)], text=True)
    assert "Tensor0CudaForwarderStatusV1" in exports
    assert not any("cuda" in line for line in exports.splitlines())
