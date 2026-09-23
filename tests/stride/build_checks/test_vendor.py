"""Vendored FFI hashes, license and runtime version alignment."""

import hashlib
import tomllib

from tests.stride.support.paths import REPO_ROOT


_REPO_ROOT = REPO_ROOT


_VENDORED_FFI_FILE_HASHES = {
    "include/xla/ffi/api/api.h": "7f76572a80ed2097e5924e6d02d84891c725300172280bd009c0a7c9ac7961eb",
    "include/xla/ffi/api/c_api.h": "85fc385c2d3a6b539a05b9cf4c3535aa24b4b41040f9e111c1f2c11b0e2fa539",
    "include/xla/ffi/api/ffi.h": "4e4a1d8f9825e88e15a2bcbb7c08eb6233f020b952cab5bbbb8510e3017515c5",
    "LICENSE.txt": "e3d8688a2c75d4e33641cc88046a8a1593ee79822411fa45a7bd32e37f847d29",
}


def test_vendored_ffi_headers_and_license_match_frozen_hashes():
    vendor_root = _REPO_ROOT / "crates" / "tensor0-py" / "vendor" / "jaxlib-0.10.1"
    for relative_path, expected in _VENDORED_FFI_FILE_HASHES.items():
        assert hashlib.sha256((vendor_root / relative_path).read_bytes()).hexdigest() == expected
    packaged_license = _REPO_ROOT / "src" / "tensor0" / "_licenses" / "JAXLIB_FFI_LICENSE.txt"
    assert hashlib.sha256(packaged_license.read_bytes()).hexdigest() == _VENDORED_FFI_FILE_HASHES["LICENSE.txt"]


def test_declared_jax_runtime_matches_vendored_ffi_version():
    project = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text())
    dependencies = set(project["project"]["dependencies"])
    build_script = (_REPO_ROOT / "crates" / "tensor0-py" / "build.rs").read_text()
    assert "jax==0.10.1" in dependencies
    assert "jaxlib==0.10.1" in dependencies
    assert 'const VENDORED_JAX_VERSION: &str = "0.10.1";' in build_script
    assert 'const VENDORED_JAXLIB_VERSION: &str = "0.10.1";' in build_script
