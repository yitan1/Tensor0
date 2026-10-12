"""Build the companion from a self-contained out-of-checkout sdist."""

from pathlib import Path
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tomllib
import zipfile

import pytest

from tests.stride.support.paths import REPO_ROOT


def test_distribution_versions_and_exact_pairing():
    core = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    addon = tomllib.loads((REPO_ROOT / "crates/tensor0-cuda12/pyproject.toml").read_text())
    cargo = tomllib.loads((REPO_ROOT / "Cargo.toml").read_text())
    version = core["project"]["version"]
    assert version == addon["project"]["version"] == "0.2.0"
    assert cargo["workspace"]["package"]["version"].replace("-dev.", ".dev") == version
    assert f"tensor0=={version}" in addon["project"]["dependencies"]
    assert f"tensor0-cuda12=={version}" in core["project"]["optional-dependencies"]["cuda12"]
    assert "0.1.0" not in str(addon["project"]["dependencies"])
    assert addon["build-system"]["build-backend"] == "setuptools.build_meta"


@pytest.mark.skipif(sys.platform != "linux" or platform.machine() != "x86_64",
                    reason="CUDA companion packaging fixture requires Linux x86_64")
def test_independent_sdist_and_platform_wheel(tmp_path):
    uv = shutil.which("uv")
    if uv is None or shutil.which("rustc") is None:
        pytest.skip("uv and rustc required for isolated packaging fixture")
    dist = tmp_path / "dist"
    log = tmp_path / "build.log"
    def run(command, **kwargs):
        with log.open("a") as stream:
            process = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                     timeout=120, **kwargs)
        assert process.returncode == 0, log.read_text()
    # Isolated PEP 517 builds resolve declared build dependencies on cold runners.
    run([uv, "build", "--no-sources", "--sdist", str(REPO_ROOT / "crates/tensor0-cuda12"),
         "--out-dir", str(dist)], cwd=tmp_path)
    sdist, = dist.glob("*.tar.gz")
    with tarfile.open(sdist) as archive:
        archive.extractall(tmp_path / "extracted", filter="data")
    source, = (tmp_path / "extracted").iterdir()
    assert not source.is_relative_to(REPO_ROOT)
    assert not (source.parent / "tensor0-py").exists()
    assert not (source / "Cargo.toml").exists()
    assert len(list((source / "native/cuda").glob("*.cu"))) == 5
    canonical = REPO_ROOT / "crates/tensor0-cuda12/native/cuda"
    assert {p.relative_to(canonical): p.read_bytes() for p in canonical.rglob("*") if p.is_file()} == {
        p.relative_to(source / "native/cuda"): p.read_bytes()
        for p in (source / "native/cuda").rglob("*") if p.is_file()
    }
    for directory in ("ffi", "layout"):
        for p in (source / "native" / directory).iterdir():
            assert p.read_bytes() == (REPO_ROOT / "crates/tensor0-py/native" / directory / p.name).read_bytes()
    assert sorted(p.name for p in (source / "native/ffi").iterdir()) == ["buffers.h", "errors.h"]
    assert not (source / "native/numeric").exists()
    assert (source / "build_helpers.rs").is_file()
    for name in ("cudart_forwarder.cc", "cudart_forwarder.map"):
        assert (source / "native" / name).read_bytes() == (REPO_ROOT / "crates/tensor0-cuda12/native" / name).read_bytes()
    for name in ("LICENSE",):
        assert (source / name).read_bytes() == (REPO_ROOT / name).read_bytes()
    vendor_license = "vendor/jaxlib-0.10.1/LICENSE.txt"
    license_bytes = (REPO_ROOT / "crates/tensor0-py" / vendor_license).read_bytes()
    assert (source / vendor_license).read_bytes() == license_bytes
    assert not list(source.rglob("_licenses"))
    for file in (source / "native").rglob("*"):
        if file.is_file():
            for header in re.findall(r'^#include "([^"\n]+)"', file.read_text(), re.MULTILINE):
                assert (file.parent / header).is_file() or (source / "vendor/jaxlib-0.10.1/include" / header).is_file(), (file, header)
    compiler = tmp_path / "compiler"
    compiler.write_text(f"#!{sys.executable}\n" + '''import pathlib, sys
if sys.argv[1:] == ["--version"]:
    print("fake compiler 1")
else:
    def value(flag): return sys.argv[sys.argv.index(flag)+1]
    pathlib.Path(value("-o")).write_bytes(b"fixture ELF")
    if "-c" in sys.argv:
        pathlib.Path(value("-MF")).write_text("tensor0-object: " + value("-c") + "\\n")
''')
    compiler.chmod(0o755)
    toolkit = tmp_path / "toolkit/lib"
    toolkit.mkdir(parents=True)
    (toolkit / "libcudart.so").write_bytes(b"fixture runtime")
    run([uv, "build", "--no-sources", "--wheel", str(source), "--out-dir", str(dist)],
        cwd=tmp_path, env=dict(os.environ, CXX=str(compiler), NVCC=str(compiler),
                              CUDA_HOME=str(toolkit.parent)))
    wheel, = dist.glob("*.whl")
    assert wheel.name.endswith("-py3-none-linux_x86_64.whl")
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        assert "tensor0_cuda12/_stride_cuda.so" in names
        assert not any("_licenses" in Path(name).parts for name in names)
        for name in ("LICENSE",):
            packaged, = (entry for entry in names if entry.endswith(f".dist-info/licenses/{name}"))
            assert archive.read(packaged) == (REPO_ROOT / name).read_bytes()
        packaged_vendor, = (entry for entry in names
                            if entry.endswith(f".dist-info/licenses/{vendor_license}"))
        assert archive.read(packaged_vendor) == license_bytes
        assert not any(name.startswith(("native/", "vendor/", "crates/", "tensor0/", "local/"))
                       for name in names)
        metadata, = (name for name in names if name.endswith(".dist-info/METADATA"))
        text = archive.read(metadata).decode()
        assert "Requires-Dist: tensor0==0.2.0" in text
        assert "License-Expression: MIT AND Apache-2.0" in text
        assert "License-File: LICENSE" in text
        assert f"License-File: {vendor_license}" in text
        wheel_metadata, = (name for name in names if name.endswith(".dist-info/WHEEL"))
        assert "Root-Is-Purelib: false" in archive.read(wheel_metadata).decode()
    # Also build directly from the checkout: CUDA and shared inputs have different owners.
    checkout_dist = tmp_path / "checkout-dist"
    run([uv, "build", "--no-sources", "--wheel",
         str(REPO_ROOT / "crates/tensor0-cuda12"), "--out-dir", str(checkout_dist)],
        cwd=tmp_path, env=dict(os.environ, CXX=str(compiler), NVCC=str(compiler),
                              CUDA_HOME=str(toolkit.parent)))
    checkout_wheel, = checkout_dist.glob("*.whl")
    with zipfile.ZipFile(checkout_wheel) as checkout, zipfile.ZipFile(wheel) as independent:
        assert checkout.namelist() == independent.namelist()
        assert all(checkout.read(name) == independent.read(name) for name in checkout.namelist())
    # Install the fixture without dependencies; importing the package loads no ELF.
    installed = tmp_path / "installed"
    run([uv, "pip", "install", "--offline", "--no-deps", "--target", str(installed), str(wheel)], cwd=tmp_path)
    run([sys.executable, "-I", "-c",
         "import sys; sys.path.insert(0, sys.argv[1]); "
         "import tensor0_cuda12; from importlib.metadata import distribution; "
         "from pathlib import Path; "
         "assert Path(tensor0_cuda12.__file__).is_relative_to(sys.argv[1]); "
         "d = distribution('tensor0-cuda12'); "
         "assert d.read_text('licenses/LICENSE') == "
         "Path(sys.argv[2]).read_text(); "
         "assert not Path(tensor0_cuda12.__file__).with_name('_licenses').exists()",
         str(installed), str(source / "LICENSE")], cwd=tmp_path)
    run([sys.executable, str(REPO_ROOT / ".github/scripts/check_artifacts.py"), "--cuda", str(sdist)], cwd=REPO_ROOT)
    # Fixture metadata only: a real release must run auditwheel repair, never retag.
    repaired = dist / wheel.name.replace("linux_x86_64", "manylinux_2_28_x86_64")
    with zipfile.ZipFile(wheel) as original, zipfile.ZipFile(repaired, "w") as archive:
        for name in original.namelist():
            data = original.read(name)
            if name.endswith(".dist-info/WHEEL"):
                data = data.replace(b"linux_x86_64", b"manylinux_2_28_x86_64")
            archive.writestr(name, data)
    run([sys.executable, str(REPO_ROOT / ".github/scripts/check_artifacts.py"), "--cuda", str(repaired)], cwd=REPO_ROOT)
    assert sorted(p.name for p in (REPO_ROOT / "crates/tensor0-cuda12/native").iterdir()) == ["cuda", "cudart_forwarder.cc", "cudart_forwarder.map"]
    assert not (REPO_ROOT / "crates/tensor0-py/native/cuda").exists()


@pytest.mark.parametrize("name", ["copy", "update", "update_validation", "accumulation", "dot", "reduction"])
def test_standalone_contract_source_tree(name, tmp_path_factory, monkeypatch):
    from tests.stride.support import cuda_contract

    monkeypatch.setattr(cuda_contract.shutil, "which", lambda _: "/fixture/nvcc")

    def compile_contract(command, **kwargs):
        native = Path(command[command.index("-I") + 1])
        assert not native.is_relative_to(REPO_ROOT)
        contract = Path(command[-5])
        included, = re.findall(r'^#include "(cuda/[^"\n]+)"', contract.read_text(), re.MULTILINE)
        assert (native / included).read_bytes() == (
            REPO_ROOT / "crates/tensor0-cuda12/native" / included
        ).read_bytes()
        for file in native.rglob("*"):
            if file.is_file():
                for header in re.findall(r'^#include "([^"\n]+)"', file.read_text(), re.MULTILINE):
                    assert (file.parent / header).is_file() or (
                        REPO_ROOT / "crates/tensor0-py/vendor/jaxlib-0.10.1/include" / header
                    ).is_file(), (file, header)
        assert command[-4:-2] == [str(native / "layout/descriptor.cc"), str(native / "layout/record.cc")]
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(cuda_contract.subprocess, "run", compile_contract)
    executable = cuda_contract.build_contract(name, tmp_path_factory)
    assert executable.name == f"{name}_contract"
