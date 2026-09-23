"""Native registration, availability and fail-closed routing."""

from importlib.metadata import version

import jax
import jax.numpy as jnp
import pytest

from tensor0 import _native
from tensor0._stride._ffi import _registration
from tensor0._stride._ffi._calls import execute_copy
from tensor0._stride._ffi._descriptor import encode_layout
from tensor0._stride._layout import AffineRecord

from tests.stride.support.availability import native_available


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
def test_build_versions_match_runtime():
    assert _native._stride_ffi_build_versions() == (jax.__version__, version("jaxlib"))


@pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")
def test_extension_exposes_only_native_stride_registration() -> None:
    assert _native._stride_ffi_build_versions() == (jax.__version__, version("jaxlib"))
    assert {name for name in dir(_native) if "stride" in name} == {
        "_stride_ffi_available", "_stride_ffi_build_versions",
        "_stride_native_registration", "_stride_native_prepared_stats",
        "_stride_native_worker_limit", "_set_stride_native_worker_limit",
    }
    registration = _native._stride_native_registration()
    expected = {f"{operation}_{suffix}"
                for operation in ("copy", "update", "reduction", "dot", "accumulation")
                for suffix in _registration._STORAGE_SUFFIXES.values()}
    expected.update({"instantiate", "type_id", "type_info", "reduction_instantiate",
                     "dot_instantiate", "accumulation_instantiate"})
    assert set(registration) == expected


def test_availability_does_not_query_legacy_abi(monkeypatch) -> None:
    def reject_legacy():
        raise AssertionError("native availability must not query the legacy ABI")

    monkeypatch.setattr(_native, "_stride_ffi_abi_version", reject_legacy, raising=False)
    monkeypatch.setattr(_native, "_stride_ffi_available", lambda: True)
    monkeypatch.setattr(_native, "_stride_ffi_build_versions", lambda: (jax.__version__, version("jaxlib")))
    assert native_available()


@pytest.mark.parametrize("available,versions,error", [
    (False, None, "unavailable"),
    (True, ("invalid", "invalid"), "different JAX versions"),
])
def test_unavailable_or_incompatible_backend_is_not_registered(available, versions, error, monkeypatch) -> None:
    monkeypatch.setattr(_registration, "_REGISTERED", set())
    monkeypatch.setattr(_native, "_stride_ffi_available", lambda: available)
    monkeypatch.setattr(_native, "_stride_ffi_build_versions", lambda: versions)
    assert not native_available()
    with pytest.raises(RuntimeError, match=error):
        _registration.operation_target("copy", jnp.dtype("float32"))
    assert not _registration._REGISTERED


@pytest.mark.parametrize("failure,message", [
    ("missing", "CPU execution is unavailable"),
    ("unavailable", "CPU execution is unavailable"),
    ("versions", "different JAX versions"),
])
def test_unavailable_cpu_call_fails_without_fallback(failure, message, monkeypatch):
    monkeypatch.setattr(_registration, "_REGISTERED", set())
    if failure == "missing":
        monkeypatch.delattr(_native, "_stride_native_registration", raising=False)
    elif failure == "unavailable":
        monkeypatch.setattr(_native, "_stride_ffi_available", lambda: False)
    else:
        monkeypatch.setattr(_native, "_stride_ffi_available", lambda: True)
        monkeypatch.setattr(_native, "_stride_ffi_build_versions", lambda: ("invalid", "invalid"))
    layout = encode_layout((AffineRecord((4,), (1,), 0, (1,), 0),), source_size=4, output_size=4)
    with pytest.raises(RuntimeError, match=message):
        execute_copy(jnp.arange(4, dtype=jnp.float32), layout=layout, output_size=4)
    assert not _registration._REGISTERED
