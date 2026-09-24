"""Independent CUDA registration and fail-closed dtype checks."""

from importlib.metadata import version

import jax
import numpy as np
import pytest

from tensor0 import _native
from tensor0._stride._ffi import _registration


@pytest.mark.parametrize("dtype,suffix", [
    ("float32", "f32"), ("float64", "f64"), ("complex64", "c64"), ("complex128", "c128"),
])
def test_cuda_registration_is_execute_only_and_independent(monkeypatch, dtype, suffix):
    capsule = object()
    calls = []
    monkeypatch.setattr(_registration, "_CUDA_REGISTERED", set())
    monkeypatch.setattr(_registration, "_REGISTERED", set())
    monkeypatch.setattr(_native, "_stride_cuda_available", lambda: True, raising=False)
    monkeypatch.setattr(_native, "_stride_cuda_registration", lambda: {f"copy_{suffix}": capsule}, raising=False)
    monkeypatch.setattr(_native, "_stride_ffi_build_versions", lambda: (jax.__version__, version("jaxlib")))
    monkeypatch.setattr(jax.ffi, "register_ffi_target", lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setattr(jax, "devices", lambda *args: pytest.fail("registration must not select a backend"))
    dtype = np.dtype(dtype)
    target = _registration.cuda_copy_target(dtype, dtype)
    assert _registration.cuda_copy_target(dtype, dtype) == target
    assert calls == [((target, capsule), {"platform": "CUDA", "api_version": 1})]
    assert not _registration._REGISTERED


@pytest.mark.parametrize("source,result", [
    ("int32", "int32"), ("float16", "float16"), ("bool", "bool"),
    ("float32", "float64"), ("complex64", "float32"),
])
def test_cuda_rejects_dtype_before_registration(source, result):
    with pytest.raises(NotImplementedError, match="same-dtype F32/F64/C64/C128"):
        _registration.cuda_copy_target(np.dtype(source), np.dtype(result))


@pytest.mark.parametrize("failure,message", [
    ("missing", "CUDA copy is unavailable"), ("unavailable", "CUDA copy is unavailable"),
    ("versions", "different JAX versions"),
])
def test_cuda_unavailable_without_cpu_fallback(monkeypatch, failure, message):
    monkeypatch.setattr(_registration, "_CUDA_REGISTERED", set())
    monkeypatch.setattr(_native, "_stride_cuda_available", lambda: failure != "unavailable", raising=False)
    monkeypatch.setattr(_native, "_stride_cuda_registration", lambda: pytest.fail("unexpected registration"), raising=False)
    monkeypatch.setattr(_native, "_stride_ffi_build_versions", lambda: ("invalid", "invalid"))
    if failure == "missing":
        monkeypatch.delattr(_native, "_stride_cuda_registration")
    with pytest.raises(RuntimeError, match=message):
        _registration.cuda_copy_target(np.dtype("float32"), np.dtype("float32"))
    assert not _registration._CUDA_REGISTERED


@pytest.mark.parametrize("dtype,suffix,coefficients", [
    ("float32", "f32", ("int32", "float32")),
    ("float64", "f64", ("int32", "float64")),
    ("complex64", "c64", ("int32", "float32", "complex64")),
    ("complex128", "c128", ("int32", "float64", "complex128")),
])
def test_cuda_update_registration_matrix(monkeypatch, dtype, suffix, coefficients):
    capsule = object()
    calls = []
    monkeypatch.setattr(_registration, "_CUDA_REGISTERED", set())
    monkeypatch.setattr(_registration, "_REGISTERED", set())
    monkeypatch.setattr(_native, "_stride_cuda_available", lambda: True, raising=False)
    monkeypatch.setattr(_native, "_stride_cuda_registration", lambda: {f"update_{suffix}": capsule}, raising=False)
    monkeypatch.setattr(_native, "_stride_ffi_build_versions", lambda: (jax.__version__, version("jaxlib")))
    monkeypatch.setattr(jax.ffi, "register_ffi_target", lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setattr(jax, "devices", lambda *args: pytest.fail("must not select a backend"))
    target = f"tensor0_stride_update_{suffix}_cuda_v1"
    for alpha in coefficients:
        for beta in coefficients:
            assert _registration.cuda_update_target(*map(np.dtype, (dtype, dtype, alpha, beta))) == target
    assert calls == [((target, capsule), {"platform": "CUDA", "api_version": 1})]
    assert not _registration._REGISTERED


@pytest.mark.parametrize("storage,coefficient", [
    ("float32", "int64"), ("float64", "int64"), ("complex64", "int64"), ("complex128", "int64"),
    ("float32", "float64"), ("float64", "float32"), ("complex64", "float64"),
    ("complex128", "complex64"), ("float32", "complex64"), ("float32", "bool"),
    ("float32", "float16"), ("complex64", "uint32"),
])
@pytest.mark.parametrize("position", [0, 1])
def test_cuda_update_rejects_coefficients_even_after_registration(monkeypatch, storage, coefficient, position):
    suffix = _registration._STORAGE_SUFFIXES[storage]
    monkeypatch.setattr(_registration, "_CUDA_REGISTERED", {f"tensor0_stride_update_{suffix}_cuda_v1"})
    coefficients = ["int32", "int32"]
    coefficients[position] = coefficient
    with pytest.raises(NotImplementedError, match="coefficient dtype"):
        _registration.cuda_update_target(*map(np.dtype, (storage, storage, *coefficients)))


@pytest.mark.parametrize("source,base", [("float16", "float16"), ("int32", "int32"), ("float32", "float64")])
def test_cuda_update_rejects_storage(source, base):
    with pytest.raises(NotImplementedError, match="same-dtype"):
        _registration.cuda_update_target(*map(np.dtype, (source, base, "int32", "int32")))


@pytest.mark.parametrize("failure", ["missing", "unavailable", "versions"])
def test_cuda_update_unavailable(monkeypatch, failure):
    monkeypatch.setattr(_registration, "_CUDA_REGISTERED", set())
    monkeypatch.setattr(_native, "_stride_cuda_available", lambda: failure != "unavailable", raising=False)
    monkeypatch.setattr(_native, "_stride_cuda_registration", lambda: pytest.fail("unexpected registration"), raising=False)
    monkeypatch.setattr(_native, "_stride_ffi_build_versions", lambda: ("invalid", "invalid"))
    if failure == "missing":
        monkeypatch.delattr(_native, "_stride_cuda_registration")
    message = "different JAX versions" if failure == "versions" else "CUDA update is unavailable"
    with pytest.raises(RuntimeError, match=message):
        _registration.cuda_update_target(*map(np.dtype, ("float32",) * 4))


@pytest.mark.parametrize("storage,coefficients", [
    ("float32", ("int32", "float32")), ("float64", ("int32", "float64")),
    ("complex64", ("int32", "float32", "complex64")),
    ("complex128", ("int32", "float64", "complex128")),
])
def test_accumulation_registration_matrix(monkeypatch, storage, coefficients):
    suffix = _registration._STORAGE_SUFFIXES[storage]
    capsule, calls = object(), []
    monkeypatch.setattr(_registration, "_CUDA_REGISTERED", set())
    monkeypatch.setattr(_native, "_stride_cuda_available", lambda: True, raising=False)
    monkeypatch.setattr(_native, "_stride_cuda_registration", lambda: {f"accumulation_{suffix}": capsule}, raising=False)
    monkeypatch.setattr(_native, "_stride_ffi_build_versions", lambda: (jax.__version__, version("jaxlib")))
    monkeypatch.setattr(jax.ffi, "register_ffi_target", lambda *args, **kwargs: calls.append((args, kwargs)))
    target = _registration.cuda_accumulation_target(*map(np.dtype, (storage, storage, *coefficients)))
    assert _registration.cuda_accumulation_target(*map(np.dtype, (storage, storage))) == target
    assert calls == [((target, capsule), {"platform": "CUDA", "api_version": 1})]
    with pytest.raises(NotImplementedError, match="coefficient dtype"):
        _registration.cuda_accumulation_target(*map(np.dtype, (storage, storage, "int64")))
    with pytest.raises(NotImplementedError, match="same-dtype"):
        _registration.cuda_accumulation_target(np.dtype("float16"), np.dtype("float16"))


@pytest.mark.parametrize("storage", ["float32", "float64", "complex64", "complex128"])
def test_dot_registration_matrix(monkeypatch, storage):
    suffix = _registration._STORAGE_SUFFIXES[storage]
    capsule, calls = object(), []
    monkeypatch.setattr(_registration, "_CUDA_REGISTERED", set())
    monkeypatch.setattr(_native, "_stride_cuda_available", lambda: True, raising=False)
    monkeypatch.setattr(_native, "_stride_cuda_registration", lambda: {f"dot_{suffix}": capsule}, raising=False)
    monkeypatch.setattr(_native, "_stride_ffi_build_versions", lambda: (jax.__version__, version("jaxlib")))
    monkeypatch.setattr(jax.ffi, "register_ffi_target", lambda *args, **kwargs: calls.append((args, kwargs)))
    target = _registration.cuda_dot_target(*map(np.dtype, (storage,) * 3))
    assert _registration.cuda_dot_target(*map(np.dtype, (storage,) * 3)) == target
    assert calls == [((target, capsule), {"platform": "CUDA", "api_version": 1})]
    with pytest.raises(NotImplementedError, match="same-dtype"):
        _registration.cuda_dot_target(*map(np.dtype, (storage, "int32", storage)))
    with pytest.raises(NotImplementedError, match="same-dtype"):
        _registration.cuda_dot_target(*map(np.dtype, ("complex64", "complex64", "float32")))


@pytest.mark.parametrize("failure", ["missing", "unavailable", "versions"])
def test_cuda_dot_unavailable(monkeypatch, failure):
    monkeypatch.setattr(_registration, "_CUDA_REGISTERED", set())
    monkeypatch.setattr(_native, "_stride_cuda_available", lambda: failure != "unavailable", raising=False)
    monkeypatch.setattr(_native, "_stride_cuda_registration", lambda: pytest.fail("unexpected registration"), raising=False)
    monkeypatch.setattr(_native, "_stride_ffi_build_versions", lambda: ("invalid", "invalid"))
    if failure == "missing": monkeypatch.delattr(_native, "_stride_cuda_registration")
    message = "different JAX versions" if failure == "versions" else "CUDA dot is unavailable"
    with pytest.raises(RuntimeError, match=message):
        _registration.cuda_dot_target(*map(np.dtype, ("float32",) * 3))


@pytest.mark.parametrize("storage,coefficients", [
    ("float32", ("int32", "float32")), ("float64", ("int32", "float64")),
    ("complex64", ("int32", "float32", "complex64")),
    ("complex128", ("int32", "float64", "complex128")),
])
def test_reduction_registration_matrix(monkeypatch, storage, coefficients):
    suffix = _registration._STORAGE_SUFFIXES[storage]
    capsule, calls = object(), []
    monkeypatch.setattr(_registration, "_CUDA_REGISTERED", set())
    monkeypatch.setattr(_native, "_stride_cuda_available", lambda: True, raising=False)
    monkeypatch.setattr(_native, "_stride_cuda_registration", lambda: {f"reduction_{suffix}": capsule}, raising=False)
    monkeypatch.setattr(_native, "_stride_ffi_build_versions", lambda: (jax.__version__, version("jaxlib")))
    monkeypatch.setattr(jax.ffi, "register_ffi_target", lambda *args, **kwargs: calls.append((args, kwargs)))
    target = _registration.cuda_reduction_target(*map(np.dtype, (storage, storage, *coefficients)))
    assert _registration.cuda_reduction_target(*map(np.dtype, (storage, storage))) == target
    assert calls == [((target, capsule), {"platform": "CUDA", "api_version": 1})]
    with pytest.raises(NotImplementedError, match="coefficient dtype"):
        _registration.cuda_reduction_target(*map(np.dtype, (storage, storage, "int64")))
    with pytest.raises(NotImplementedError, match="same-dtype"):
        _registration.cuda_reduction_target(np.dtype("float16"), np.dtype("float16"))
