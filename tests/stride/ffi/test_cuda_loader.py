"""CUDA loading uses only wheel-local companions and the pinned pip runtime."""

import importlib.util
from pathlib import Path
import os
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def loader(monkeypatch, tmp_path):
    source = Path(__file__).resolve().parents[3] / "src/tensor0/_stride/_ffi/_cuda.py"
    spec = importlib.util.spec_from_file_location("_cuda_loader_test", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    package = tmp_path / "tensor0_cuda12"
    companion = package / "_stride_cuda.so"
    companion.parent.mkdir()
    companion.touch()
    runtime = tmp_path / "pip/nvidia/cuda_runtime/lib/libcudart.so.12"
    runtime.parent.mkdir(parents=True)
    runtime.touch()
    monkeypatch.setattr(module, "__file__", str(package / "_stride/_ffi/_cuda.py"))
    calls = []

    def distribution(name):
        calls.append(("distribution", name))
        if name == "tensor0-cuda12":
            return SimpleNamespace(version="0.2.0.dev0", files=["tensor0_cuda12/_stride_cuda.so"],
                                   locate_file=lambda path: tmp_path / path)
        return SimpleNamespace(version="12.9.79", locate_file=locate_file)

    def locate_file(name):
        calls.append(("locate", name))
        return tmp_path / "pip" / name

    def cdll(path, *, mode):
        calls.append(("load", path, mode))
        functions = {name: (lambda: 0) for name in module._REQUIRED_SYMBOLS}
        functions.update(Tensor0CudaForwarderStatusV1=lambda: 0,
                         Tensor0CudaForwarderRuntimeV1=lambda: 1,
                         Tensor0CudaForwarderErrorV1=lambda: b"ok")
        return SimpleNamespace(_handle=1, **functions)

    monkeypatch.setattr(module.importlib.metadata, "distribution", distribution)
    monkeypatch.setattr(module.importlib.metadata, "version", lambda name: "0.2.0.dev0")
    monkeypatch.setattr(module.ctypes, "CDLL", cdll)
    return module, companion, runtime, calls


@pytest.mark.skipif(sys.platform != "linux", reason="CUDA runtime loading is Linux-only")
def test_success_explicit_paths_order_and_cache(loader):
    module, companion, runtime, calls = loader
    libraries = module._load_cuda_library()
    assert module._load_cuda_library() is libraries
    assert calls == [
        ("distribution", "tensor0-cuda12"),
        ("distribution", "nvidia-cuda-runtime-cu12"),
        ("locate", "nvidia/cuda_runtime/lib/libcudart.so.12"),
        ("load", str(runtime), os.RTLD_NOW | os.RTLD_GLOBAL),
        ("load", "libcudart.so.12", os.RTLD_NOW | os.RTLD_NOLOAD),
        ("load", str(companion), os.RTLD_NOW | os.RTLD_LOCAL),
    ]


def test_missing_companion_is_not_missing_extra(loader):
    module, companion, _, calls = loader
    companion.unlink()
    with pytest.raises(RuntimeError, match="companion library is missing"):
        module._load_cuda_library()
    assert not any(call[0] == "load" for call in calls)
    assert module._libraries is None


def test_missing_runtime_distribution(loader, monkeypatch):
    module, _, _, calls = loader

    original = module.importlib.metadata.distribution

    def missing(name):
        if name == "nvidia-cuda-runtime-cu12":
            raise module.importlib.metadata.PackageNotFoundError(name)
        return original(name)

    monkeypatch.setattr(module.importlib.metadata, "distribution", missing)
    with pytest.raises(RuntimeError, match=r"pip install 'tensor0\[cuda12\]'"):
        module._load_cuda_library()
    assert not any(call[0] == "load" for call in calls)
    assert module._libraries is None


@pytest.mark.parametrize("version", ["12.9.78", "12.8", "13", "13.0.0", "12.9.79rc1"])
def test_invalid_runtime_version(loader, monkeypatch, version):
    module, _, _, calls = loader
    original = module.importlib.metadata.distribution
    monkeypatch.setattr(module.importlib.metadata, "distribution", lambda name:
                        SimpleNamespace(version=version) if name == "nvidia-cuda-runtime-cu12" else original(name))
    with pytest.raises(RuntimeError, match="runtime version|requires nvidia"):
        module._load_cuda_library()
    assert not any(call[0] == "load" for call in calls)
    assert module._libraries is None


def test_missing_runtime_file(loader):
    module, _, runtime, calls = loader
    runtime.unlink()
    with pytest.raises(RuntimeError, match="runtime library is missing"):
        module._load_cuda_library()
    assert not any(call[0] == "load" for call in calls)
    assert module._libraries is None


@pytest.mark.parametrize("failed_load", [1, 2, 3])
@pytest.mark.skipif(sys.platform != "linux", reason="CUDA runtime loading is Linux-only")
def test_failed_load_is_retryable(loader, monkeypatch, failed_load):
    module, _, _, _ = loader
    original = module.ctypes.CDLL
    loads = []

    def fail(path, *, mode):
        loads.append(path)
        if len(loads) == failed_load:
            raise OSError("test failure")
        return original(path, mode=mode)

    monkeypatch.setattr(module.ctypes, "CDLL", fail)
    with pytest.raises(RuntimeError, match="Unable to load Tensor0 CUDA libraries: test failure"):
        module._load_cuda_library()
    assert module._libraries is None
    monkeypatch.setattr(module.ctypes, "CDLL", original)
    assert len(module._load_cuda_library()) == 2


def test_normal_import_and_capability_query_do_not_load_cuda():
    subprocess.run([sys.executable, "-c", """
import importlib.metadata
import ctypes
import jax
original = importlib.metadata.distribution

def distribution(name):
    assert name != 'nvidia-cuda-runtime-cu12', 'unexpected CUDA runtime lookup'
    return original(name)

importlib.metadata.distribution = distribution
jax.devices = lambda *a, **kw: (_ for _ in ()).throw(AssertionError('device probe'))
import tensor0
assert isinstance(tensor0._native._stride_cuda_available(), bool)
# The unified loader module may be imported, but no runtime/ELF/device is probed.
"""], check=True)


@pytest.mark.parametrize("version", ["12.9.79", "12.9.80", "12.10.0", "12.10"])
@pytest.mark.skipif(sys.platform != "linux", reason="CUDA runtime loading is Linux-only")
def test_eligible_runtime_version(loader, monkeypatch, version):
    module, _, runtime, _ = loader
    original = module.importlib.metadata.distribution
    monkeypatch.setattr(module.importlib.metadata, "distribution", lambda name:
                        SimpleNamespace(version=version, locate_file=lambda name: runtime)
                        if name == "nvidia-cuda-runtime-cu12" else original(name))
    assert len(module._load_cuda_library()) == 2


@pytest.mark.skipif(sys.platform != "linux", reason="CUDA runtime loading is Linux-only")
def test_runtime_conflict_rejects_before_companion_and_is_retryable(loader, monkeypatch):
    module, companion, runtime, calls = loader
    original = module.ctypes.CDLL

    def conflict(path, *, mode):
        library = original(path, mode=mode)
        if path == "libcudart.so.12":
            return SimpleNamespace(_handle=2)
        return library

    monkeypatch.setattr(module.ctypes, "CDLL", conflict)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="runtime conflict.*Restart in a clean environment"):
            module._load_cuda_library()
        assert module._libraries is None
    assert sum(call[0] == "load" and call[1] == str(runtime) for call in calls) == 2
    assert not any(call[0] == "load" and call[1] == str(companion) for call in calls)
    monkeypatch.setattr(module.ctypes, "CDLL", original)
    assert len(module._load_cuda_library()) == 2


@pytest.fixture
def fake_cuda_elf(tmp_path):
    compiler = shutil.which("g++")
    if sys.platform != "linux" or compiler is None:
        pytest.skip("Linux ELF fixtures require g++")
    good = tmp_path / "pip/nvidia/cuda_runtime/lib/libcudart.so.12"
    rogue = tmp_path / "rogue/libcudart.so.12"
    companion = tmp_path / "tensor0_cuda12/_stride_cuda.so"
    for library, version in [(good, 12090), (rogue, 12080)]:
        library.parent.mkdir(parents=True)
        source = library.parent / "runtime.cpp"
        source.write_text(f'extern "C" int cudaRuntimeGetVersion() {{ return {version}; }}\n' +\
                          ''.join(f'extern "C" void {name}() {{}}\n' for name in (\
                              'cudaGetErrorString', 'cudaGetLastError', 'cudaMemcpyAsync', 'cudaMemsetAsync',\
                              'cudaLaunchKernel', '__cudaPushCallConfiguration', '__cudaPopCallConfiguration',\
                              '__cudaRegisterFatBinary', '__cudaRegisterFatBinaryEnd', '__cudaRegisterFunction',\
                              '__cudaUnregisterFatBinary')))
        subprocess.run([compiler, "-shared", "-fPIC", str(source),
                        "-Wl,-soname,libcudart.so.12", "-o", str(library)], check=True)
    companion.parent.mkdir()
    source = companion.parent / "companion.cpp"
    source.write_text('extern "C" int cudaRuntimeGetVersion();\n'
                      'extern "C" int boundRuntimeVersion() { return cudaRuntimeGetVersion(); }\n'\
                      '#include <dlfcn.h>\n'\
                      'extern "C" int Tensor0CudaForwarderStatusV1() { return 0; }\n'\
                      'extern "C" void* Tensor0CudaForwarderRuntimeV1() { return dlopen("libcudart.so.12", RTLD_NOW|RTLD_NOLOAD); }\n'\
                      'extern "C" const char* Tensor0CudaForwarderErrorV1() { return "ok"; }\n')
    subprocess.run([compiler, "-shared", "-fPIC", str(source), "-L", str(good.parent),
                    "-Wl,--no-as-needed", "-l:libcudart.so.12", "-o", str(companion)], check=True)
    return tmp_path, good, rogue, companion


@pytest.mark.parametrize("scenario", ["unguarded-conflict", "conflict", "clean", "right-preloaded"])
def test_real_elf_runtime_binding(fake_cuda_elf, scenario):
    root, good, rogue, companion = fake_cuda_elf
    source = Path(__file__).resolve().parents[3] / "src/tensor0/_stride/_ffi/_cuda.py"
    subprocess.run([sys.executable, "-c", '''
import ctypes
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace

source, root, good, rogue, companion, scenario = sys.argv[1:]
retained = []
if scenario in ('conflict', 'unguarded-conflict'):
    retained.append(ctypes.CDLL(rogue, mode=os.RTLD_NOW | os.RTLD_GLOBAL))
elif scenario == 'right-preloaded':
    retained.append(ctypes.CDLL(good, mode=os.RTLD_NOW | os.RTLD_GLOBAL))

if scenario == 'unguarded-conflict':
    # Reproduce the original bug: an explicit pip load does not protect binding.
    retained.append(ctypes.CDLL(good, mode=os.RTLD_NOW | os.RTLD_GLOBAL))
    library = ctypes.CDLL(companion, mode=os.RTLD_NOW | os.RTLD_LOCAL)
    assert library.boundRuntimeVersion() == 12080
else:
    spec = importlib.util.spec_from_file_location('_cuda_loader_elf_test', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.__file__ = str(Path(root) / 'tensor0/_stride/_ffi/_cuda.py')
    module.importlib.metadata.version = lambda name: '0.2.0.dev0'
    module.importlib.metadata.distribution = lambda name: (
        SimpleNamespace(version='0.2.0.dev0', files=['tensor0_cuda12/_stride_cuda.so'],
                        locate_file=lambda path: Path(root) / path)
        if name == 'tensor0-cuda12' else SimpleNamespace(
            version='12.9.79', locate_file=lambda path: Path(root) / 'pip' / path))
    if scenario == 'conflict':
        for _ in range(2):
            try:
                module._load_cuda_library()
            except RuntimeError as exc:
                assert 'runtime conflict' in str(exc)
                assert 'Restart in a clean environment' in str(exc)
            else:
                raise AssertionError('conflicting runtime accepted')
            assert module._libraries is None
            # NOLOAD must fail: the companion has never been loaded.
            try:
                ctypes.CDLL(companion, mode=os.RTLD_NOW | os.RTLD_NOLOAD)
            except OSError:
                pass
            else:
                raise AssertionError('companion loaded before conflict rejection')
    else:
        libraries = module._load_cuda_library()
        assert module._load_cuda_library() is libraries
        assert libraries[0].cudaRuntimeGetVersion() == 12090
        assert libraries[1].boundRuntimeVersion() == 12090
''', str(source), str(root), str(good), str(rogue), str(companion), scenario], check=True)


def test_absent_addon_presence_is_false_and_explicit_load_fails(loader, monkeypatch):
    module, _, _, calls = loader
    original = module.importlib.metadata.distribution
    def missing(name):
        if name == "tensor0-cuda12":
            raise module.importlib.metadata.PackageNotFoundError(name)
        return original(name)
    monkeypatch.setattr(module.importlib.metadata, "distribution", missing)
    assert not module._cuda_companion_present()
    with pytest.raises(RuntimeError, match="companion is missing"):
        module._load_cuda_library()
    assert not calls


@pytest.mark.parametrize("version", ["0.1.0", "0.2.0.dev1", "0.2.0"])
def test_mismatched_addon_visible_but_fails_before_any_load(loader, monkeypatch, version):
    module, _, _, calls = loader
    original = module.importlib.metadata.distribution
    monkeypatch.setattr(module.importlib.metadata, "distribution", lambda name:
        SimpleNamespace(version=version) if name == "tensor0-cuda12" else original(name))
    assert module._cuda_companion_present()
    with pytest.raises(RuntimeError, match="version mismatch"):
        module._load_cuda_library()
    assert not calls and module._libraries is None


def test_unrecorded_companion_fails_closed(loader, monkeypatch):
    module, _, _, calls = loader
    monkeypatch.setattr(module.importlib.metadata, "distribution", lambda name:
                        SimpleNamespace(version="0.2.0.dev0", files=[]))
    with pytest.raises(RuntimeError, match="distribution records"):
        module._load_cuda_library()
    assert not calls


@pytest.mark.parametrize("symbol", ["cudaGetLastError", "__cudaRegisterFunction"])
@pytest.mark.skipif(sys.platform != "linux", reason="CUDA runtime loading is Linux-only")
def test_symbol_preflight_before_companion_is_retryable(loader, monkeypatch, symbol):
    module, companion, _, calls = loader
    original = module.ctypes.CDLL
    def missing(path, *, mode):
        library = original(path, mode=mode)
        if path != str(companion):
            delattr(library, symbol)
        return library
    monkeypatch.setattr(module.ctypes, "CDLL", missing)
    with pytest.raises(RuntimeError, match=f"missing required symbol {symbol}"):
        module._load_cuda_library()
    assert module._libraries is None and module._failure is None
    assert not any(call[0] == "load" and call[1] == str(companion) for call in calls)
    monkeypatch.setattr(module.ctypes, "CDLL", original)
    assert len(module._load_cuda_library()) == 2


@pytest.mark.parametrize("failure", ["status", "identity", "abi"])
@pytest.mark.skipif(sys.platform != "linux", reason="CUDA runtime loading is Linux-only")
def test_post_constructor_validation_is_sticky_without_cache(loader, monkeypatch, failure):
    module, companion, _, calls = loader
    original = module.ctypes.CDLL
    def invalid(path, *, mode):
        library = original(path, mode=mode)
        if path == str(companion):
            if failure == "status":
                library.Tensor0CudaForwarderStatusV1 = lambda: 302
            elif failure == "identity":
                library.Tensor0CudaForwarderRuntimeV1 = lambda: 2
            else:
                del library.Tensor0CudaForwarderStatusV1
        return library
    monkeypatch.setattr(module.ctypes, "CDLL", invalid)
    with pytest.raises(RuntimeError, match="companion validation failed.*Restart"):
        module._load_cuda_library()
    assert module._libraries is None and module._failure is not None
    assert len(module._retained) == 3
    before = calls.copy()
    monkeypatch.setattr(module.ctypes, "CDLL", original)
    with pytest.raises(RuntimeError, match="Restart"):
        module._load_cuda_library()
    assert calls == before


def test_cpu_jit_without_cuda_loader_or_rtld_noload():
    subprocess.run([sys.executable, "-I", "-c", """
import ctypes
import os
import jax
import jax.numpy as jnp
import numpy as np
# Initialize JAX before blocking CDLL: its own CPU initialization is legitimate.
jax.devices('cpu')
if hasattr(os, 'RTLD_NOLOAD'):
    del os.RTLD_NOLOAD

def forbidden(*args, **kwargs):
    raise AssertionError('CPU operation attempted CUDA loading')

ctypes.CDLL = forbidden
import tensor0
from tensor0._stride._ffi import _cuda
from tensor0._stride._ffi._calls import execute_copy
from tensor0._stride._ffi._descriptor import encode_layout
from tensor0._stride._layout import AffineRecord
assert tensor0._native._stride_ffi_available()
# Isolate from any locally installed optional companion.
_cuda._cuda_companion_present = lambda: False
_cuda._load_cuda_library = forbidden
assert not tensor0._native._stride_cuda_available()
layout = encode_layout((AffineRecord((4,), (1,), 0, (1,), 0),), source_size=4, output_size=4)
copy = jax.jit(lambda x: execute_copy(x, layout=layout, output_size=4))
x = jnp.arange(4, dtype=jnp.float32)
np.testing.assert_array_equal(copy(x), x)
"""], check=True, env=dict(os.environ, JAX_PLATFORMS="cpu"))
