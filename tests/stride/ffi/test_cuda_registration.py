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
def test_cuda_copy_registration_has_independent_prepared_state(monkeypatch, dtype, suffix):
    capsule, instantiate, type_id, type_info = (object() for _ in range(4))
    calls, types = [], []
    monkeypatch.setattr(_registration, "_CUDA_REGISTERED", set())
    monkeypatch.setattr(_registration, "_REGISTERED", set())
    monkeypatch.setattr(_native, "_stride_cuda_available", lambda: True, raising=False)
    monkeypatch.setattr(_native, "_stride_cuda_registration", lambda: {
        f"copy_{suffix}": capsule, "copy_instantiate": instantiate,
        "copy_type_id": type_id, "copy_type_info": type_info,
    }, raising=False)
    monkeypatch.setattr(_native, "_stride_ffi_build_versions", lambda: (jax.__version__, version("jaxlib")))
    monkeypatch.setattr(jax.ffi, "register_ffi_target", lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setattr(jax.ffi, "register_ffi_type", lambda *args, **kwargs: types.append((args, kwargs)))
    monkeypatch.setattr(jax, "devices", lambda *args: pytest.fail("registration must not select a backend"))
    dtype = np.dtype(dtype)
    target = _registration.cuda_copy_target(dtype, dtype)
    assert _registration.cuda_copy_target(dtype, dtype) == target
    assert calls == [((target, {"instantiate": instantiate, "execute": capsule}),
                      {"platform": "CUDA", "api_version": 1})]
    assert types == [(("tensor0_stride_cuda_copy_prepared_v1",
                      {"type_id": type_id, "type_info": type_info}), {"platform": "CUDA"})]
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
    capsule, instantiate, type_id, type_info = (object() for _ in range(4))
    calls, types = [], []
    monkeypatch.setattr(_registration, "_CUDA_REGISTERED", set())
    monkeypatch.setattr(_registration, "_REGISTERED", set())
    monkeypatch.setattr(_native, "_stride_cuda_available", lambda: True, raising=False)
    monkeypatch.setattr(_native, "_stride_cuda_registration", lambda: {
        f"update_{suffix}": capsule, "update_instantiate": instantiate,
        "update_type_id": type_id, "update_type_info": type_info,
    }, raising=False)
    monkeypatch.setattr(_native, "_stride_ffi_build_versions", lambda: (jax.__version__, version("jaxlib")))
    monkeypatch.setattr(jax.ffi, "register_ffi_target", lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setattr(jax, "devices", lambda *args: pytest.fail("must not select a backend"))
    monkeypatch.setattr(jax.ffi, "register_ffi_type", lambda *args, **kwargs: types.append((args, kwargs)))
    target = f"tensor0_stride_update_{suffix}_cuda_v1"
    for alpha in coefficients:
        for beta in coefficients:
            assert _registration.cuda_update_target(*map(np.dtype, (dtype, dtype, alpha, beta))) == target
    assert calls == [((target, {"instantiate": instantiate, "execute": capsule}),
                      {"platform": "CUDA", "api_version": 1})]
    assert types == [(("tensor0_stride_cuda_update_prepared_v1",
                      {"type_id": type_id, "type_info": type_info}), {"platform": "CUDA"})]
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
    capsule, instantiate, type_id, type_info = (object() for _ in range(4))
    calls, types = [], []
    monkeypatch.setattr(_registration, "_CUDA_REGISTERED", set())
    monkeypatch.setattr(_native, "_stride_cuda_available", lambda: True, raising=False)
    monkeypatch.setattr(_native, "_stride_cuda_registration", lambda: {
        f"accumulation_{suffix}": capsule, "accumulation_instantiate": instantiate,
        "accumulation_type_id": type_id, "accumulation_type_info": type_info,
    }, raising=False)
    monkeypatch.setattr(jax.ffi, "register_ffi_type", lambda *args, **kwargs: types.append((args, kwargs)))
    monkeypatch.setattr(jax, "devices", lambda *args: pytest.fail("must not select a backend"))
    monkeypatch.setattr(_native, "_stride_ffi_build_versions", lambda: (jax.__version__, version("jaxlib")))
    monkeypatch.setattr(jax.ffi, "register_ffi_target", lambda *args, **kwargs: calls.append((args, kwargs)))
    target = _registration.cuda_accumulation_target(*map(np.dtype, (storage, storage, *coefficients)))
    assert _registration.cuda_accumulation_target(*map(np.dtype, (storage, storage))) == target
    assert calls == [((target, {"instantiate": instantiate, "execute": capsule}),
                      {"platform": "CUDA", "api_version": 1})]
    assert types == [(("tensor0_stride_cuda_accumulation_prepared_v1",
                      {"type_id": type_id, "type_info": type_info}), {"platform": "CUDA"})]
    with pytest.raises(NotImplementedError, match="coefficient dtype"):
        _registration.cuda_accumulation_target(*map(np.dtype, (storage, storage, "int64")))
    with pytest.raises(NotImplementedError, match="same-dtype"):
        _registration.cuda_accumulation_target(np.dtype("float16"), np.dtype("float16"))


@pytest.mark.parametrize("storage", ["float32", "float64", "complex64", "complex128"])
def test_dot_registration_matrix(monkeypatch, storage):
    suffix = _registration._STORAGE_SUFFIXES[storage]
    capsule, instantiate, type_id, type_info = (object() for _ in range(4))
    calls, types = [], []
    monkeypatch.setattr(_registration, "_CUDA_REGISTERED", set())
    monkeypatch.setattr(_native, "_stride_cuda_available", lambda: True, raising=False)
    monkeypatch.setattr(_native, "_stride_cuda_registration", lambda: {
        f"dot_{suffix}": capsule, "dot_instantiate": instantiate,
        "dot_type_id": type_id, "dot_type_info": type_info,
    }, raising=False)
    monkeypatch.setattr(_native, "_stride_ffi_build_versions", lambda: (jax.__version__, version("jaxlib")))
    monkeypatch.setattr(jax.ffi, "register_ffi_target", lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setattr(jax.ffi, "register_ffi_type", lambda *args, **kwargs: types.append((args, kwargs)))
    target = _registration.cuda_dot_target(*map(np.dtype, (storage,) * 3))
    assert _registration.cuda_dot_target(*map(np.dtype, (storage,) * 3)) == target
    assert calls == [((target, {"instantiate": instantiate, "execute": capsule}),
                      {"platform": "CUDA", "api_version": 1})]
    assert types == [(("tensor0_stride_cuda_dot_prepared_v1",
                      {"type_id": type_id, "type_info": type_info}), {"platform": "CUDA"})]
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
    capsule, instantiate, type_id, type_info = (object() for _ in range(4))
    calls, types = [], []
    monkeypatch.setattr(_registration, "_CUDA_REGISTERED", set())
    monkeypatch.setattr(_native, "_stride_cuda_available", lambda: True, raising=False)
    monkeypatch.setattr(_native, "_stride_cuda_registration", lambda: {
        f"reduction_{suffix}": capsule, "reduction_instantiate": instantiate,
        "reduction_type_id": type_id, "reduction_type_info": type_info,
    }, raising=False)
    monkeypatch.setattr(_native, "_stride_ffi_build_versions", lambda: (jax.__version__, version("jaxlib")))
    monkeypatch.setattr(jax.ffi, "register_ffi_target", lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setattr(jax.ffi, "register_ffi_type", lambda *args, **kwargs: types.append((args, kwargs)))
    target = _registration.cuda_reduction_target(*map(np.dtype, (storage, storage, *coefficients)))
    assert _registration.cuda_reduction_target(*map(np.dtype, (storage, storage))) == target
    assert calls == [((target, {"instantiate": instantiate, "execute": capsule}),
                      {"platform": "CUDA", "api_version": 1})]
    assert types == [(("tensor0_stride_cuda_reduction_prepared_v1",
                      {"type_id": type_id, "type_info": type_info}), {"platform": "CUDA"})]
    with pytest.raises(NotImplementedError, match="coefficient dtype"):
        _registration.cuda_reduction_target(*map(np.dtype, (storage, storage, "int64")))
    with pytest.raises(NotImplementedError, match="same-dtype"):
        _registration.cuda_reduction_target(np.dtype("float16"), np.dtype("float16"))


@pytest.mark.parametrize("operation,other", [
    ("copy", "update"), ("update", "copy"), ("accumulation", "update"), ("dot", "copy"), ("reduction", "dot"),
])
def test_cuda_type_registered_once_across_dtypes_after_other_operation(monkeypatch, operation, other):
    registration = {f"{operation}_{name}": object() for name in (
        "instantiate", "type_id", "type_info", "f32", "f64")}
    types, targets = [], []
    monkeypatch.setattr(_registration, "_CUDA_REGISTERED", {f"tensor0_stride_{other}_f32_cuda_v1"})
    monkeypatch.setattr(_native, "_stride_cuda_available", lambda: True)
    monkeypatch.setattr(_native, "_stride_cuda_registration", lambda: registration)
    monkeypatch.setattr(_native, "_stride_ffi_build_versions", lambda: (jax.__version__, version("jaxlib")))
    monkeypatch.setattr(jax.ffi, "register_ffi_type", lambda *args, **kwargs: types.append((args, kwargs)))
    monkeypatch.setattr(jax.ffi, "register_ffi_target", lambda *args, **kwargs: targets.append((args, kwargs)))
    for dtype in (np.dtype("float32"), np.dtype("float64"), np.dtype("float32")):
        if operation == "copy":
            _registration.cuda_copy_target(dtype, dtype)
        elif operation == "update":
            _registration.cuda_update_target(dtype, dtype, dtype, dtype)
        elif operation == "accumulation":
            _registration.cuda_accumulation_target(dtype, dtype, dtype)
        elif operation == "reduction":
            _registration.cuda_reduction_target(dtype, dtype, dtype)
        else:
            _registration.cuda_dot_target(dtype, dtype, dtype)
    assert len(types) == 1
    assert types[0][0][0] == f"tensor0_stride_cuda_{operation}_prepared_v1"
    assert len(targets) == 2
    assert all(call[0][1]["instantiate"] is registration[f"{operation}_instantiate"] for call in targets)
