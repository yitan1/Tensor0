"""Registration checks independent of legacy descriptor and ABI helpers."""

from importlib.metadata import version

import jax
import jax.numpy as jnp
import pytest

from tensor0 import _native
from tensor0._stride._ffi import _registration
from ._support import native_available


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
