"""Vendored FFI hashes, license and runtime version alignment."""

import hashlib
import tomllib

from tests.stride.support.paths import REPO_ROOT


_REPO_ROOT = REPO_ROOT


_VENDORED_FFI_FILE_HASHES = {
    "include/xla/ffi/api/api.h": "1426ea8206a1b9f1824f460572ea731665eb7cb9def1f0fc166b13df4b955ad0",
    "include/xla/ffi/api/c_api.h": "9847993912aa1aadb2dac4af16eeb9d8bae3f470905b519977d1d09b3f37eedf",
    "include/xla/ffi/api/ffi.h": "0878473a1ee35fdd9ff15424d46a44602fb53019f0d0a9f5117abdb386c35129",
    "LICENSE.txt": "e3d8688a2c75d4e33641cc88046a8a1593ee79822411fa45a7bd32e37f847d29",
}


def test_vendored_ffi_headers_and_license_match_frozen_hashes():
    vendor_root = _REPO_ROOT / "crates" / "tensor0-py" / "vendor" / "jaxlib-0.11.2"
    for relative_path, expected in _VENDORED_FFI_FILE_HASHES.items():
        assert hashlib.sha256((vendor_root / relative_path).read_bytes()).hexdigest() == expected
    assert not (_REPO_ROOT / "src/tensor0/_licenses").exists()


def test_declared_license_files_use_standard_metadata_layout():
    for path, vendor_license in (
        ("pyproject.toml", "crates/tensor0-py/vendor/jaxlib-0.11.2/LICENSE.txt"),
        ("crates/tensor0-cuda12/pyproject.toml", "vendor/jaxlib-0.11.2/LICENSE.txt"),
    ):
        project = tomllib.loads((_REPO_ROOT / path).read_text())
        assert project["project"]["license-files"] == ["LICENSE", vendor_license]
        assert project["project"]["license"] == "MIT AND Apache-2.0"
    addon = tomllib.loads((_REPO_ROOT / "crates/tensor0-cuda12/pyproject.toml").read_text())
    assert addon["tool"]["setuptools"]["package-data"]["tensor0_cuda12"] == ["_stride_cuda.so"]


def test_declared_jax_runtime_matches_vendored_ffi_version():
    project = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text())
    dependencies = set(project["project"]["dependencies"])
    build_script = (_REPO_ROOT / "crates" / "tensor0-py" / "build.rs").read_text()
    assert "jax==0.11.2" in dependencies
    assert "jaxlib==0.11.2" in dependencies
    assert 'const VENDORED_JAX_VERSION: &str = "0.11.2";' in build_script
    assert 'const VENDORED_JAXLIB_VERSION: &str = "0.11.2";' in build_script
