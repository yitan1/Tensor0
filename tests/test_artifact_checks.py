"""Distribution-content checks keep optional CUDA dependencies and tags explicit."""

from pathlib import Path
import subprocess
import sys
import zipfile

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / ".github/scripts/check_artifacts.py"


@pytest.mark.parametrize("cuda,platform,metadata_platform,companion,bundled,accepted", [
    (False, "manylinux_2_28_x86_64", "manylinux_2_28_x86_64", False, None, True),
    (False, "macosx_15_0_arm64", "macosx_15_0_arm64", False, None, True),
    (False, "macosx_15_0_arm64", "macosx_14_0_arm64", False, None, False),
    (False, "macosx_14_0_arm64", "macosx_14_0_arm64", False, None, False),
    (False, "macosx_16_0_arm64", "macosx_16_0_arm64", False, None, False),
    (False, "macosx_15_0_universal2", "macosx_15_0_universal2", False, None, False),
    (False, "macosx_15_0_x86_64", "macosx_15_0_x86_64", False, None, False),
    (False, "manylinux_2_28_aarch64", "manylinux_2_28_aarch64", False, None, False),
    (True, "linux_x86_64", "linux_x86_64", True, None, False),
    (True, "manylinux_2_28_x86_64", "manylinux_2_28_x86_64", True, None, True),
    (True, "linux_x86_64", "manylinux_2_28_x86_64", True, None, False),
    (True, "linux_x86_64", None, True, None, False),
    (True, "linux_x86_64", "linux_x86_64", False, None, False),
    (False, "linux_x86_64", "linux_x86_64", True, None, False),
    (True, "linux_x86_64", "linux_x86_64", True, "libcudart.so.12", False),
    (True, "linux_x86_64", "linux_x86_64", True, "libcudart-abcdef.so.12", False),
    (True, "linux_x86_64", "linux_x86_64", True, "libcuda.so.1", False),
])
def test_wheel_contents(tmp_path, cuda, platform, metadata_platform, companion, bundled, accepted):
    (tmp_path / "pyproject.toml").write_text('[project]\nversion = "0.2.0.dev0"\nlicense = "MIT AND Apache-2.0"\nlicense-files = ["LICENSE"]\n')
    addon = tmp_path / "crates/tensor0-cuda12"
    addon.mkdir(parents=True)
    (addon / "pyproject.toml").write_text(
        '[project]\nlicense = "MIT AND Apache-2.0"\nlicense-files = ["LICENSE"]\n')
    (tmp_path / "LICENSE").write_text("test license")
    package = tmp_path / "src/tensor0"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    wheel = tmp_path / (f"tensor0_cuda12-0.2.0.dev0-py3-none-{platform}.whl" if cuda else
                        f"tensor0-0.2.0.dev0-cp311-cp311-{platform}.whl")
    with zipfile.ZipFile(wheel, "w") as archive:
        info = "tensor0_cuda12-0.2.0.dev0.dist-info" if cuda else "tensor0-0.2.0.dev0.dist-info"
        for name in (("tensor0_cuda12/__init__.py", f"{info}/licenses/LICENSE") if cuda else
                     ("tensor0/__init__.py", "tensor0/_native.cpython-311-x86_64-linux-gnu.so",
                      f"{info}/licenses/LICENSE")):
            archive.writestr(name, "")
        archive.writestr(f"{info}/METADATA",
                         "Name: tensor0-cuda12\nVersion: 0.2.0.dev0\nRequires-Dist: tensor0==0.2.0.dev0\n"
                         "Requires-Dist: nvidia-cuda-runtime-cu12<13,>=12.9.79\n"
                         "License-Expression: MIT AND Apache-2.0\nLicense-File: LICENSE\n" if cuda else
                         "Name: tensor0\nVersion: 0.2.0.dev0\nLicense-Expression: MIT AND Apache-2.0\nLicense-File: LICENSE\n")
        tag = f"Tag: {'py3-none' if cuda else 'cp311-cp311'}-{metadata_platform}\n" if metadata_platform else ""
        archive.writestr(f"{info}/WHEEL", "Root-Is-Purelib: false\n" + tag)
        if companion:
            archive.writestr("tensor0_cuda12/_stride_cuda.so" if cuda else "tensor0/_stride_cuda.so", "")
        if bundled:
            archive.writestr(f"tensor0.libs/{bundled}", "")
    command = [sys.executable, "-I", str(SCRIPT), str(wheel)]
    command.append("--cuda" if cuda else "--release-core")
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert (result.returncode == 0) == accepted, result.stdout + result.stderr


@pytest.mark.parametrize("cuda", [False, True])
@pytest.mark.parametrize("violation", ["missing_license", "missing_vendor_license", "package_licenses", "private_file"])
def test_license_metadata_and_private_file_exclusions(tmp_path, cuda, violation):
    core_vendor = "crates/tensor0-py/vendor/jaxlib-0.10.1/LICENSE.txt"
    addon_vendor = "vendor/jaxlib-0.10.1/LICENSE.txt"
    vendor = addon_vendor if cuda else core_vendor
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nversion = "0.2.0.dev0"\n'
        'license = "MIT AND Apache-2.0"\n'
        f'license-files = ["LICENSE", "{core_vendor}"]\n')
    addon = tmp_path / "crates/tensor0-cuda12"
    addon.mkdir(parents=True)
    (addon / "pyproject.toml").write_text(
        '[project]\nlicense = "MIT AND Apache-2.0"\n'
        f'license-files = ["LICENSE", "{addon_vendor}"]\n')
    for name in ("LICENSE",):
        (tmp_path / name).write_text("fixture notice")
    source = tmp_path / "src/tensor0"
    source.mkdir(parents=True)
    (source / "__init__.py").write_text("")
    package = "tensor0_cuda12" if cuda else "tensor0"
    tag = "py3-none-manylinux_2_28_x86_64" if cuda else "cp311-cp311-linux_x86_64"
    wheel = tmp_path / f"{package}-0.2.0.dev0-{tag}.whl"
    info = f"{package}-0.2.0.dev0.dist-info"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(f"{package}/__init__.py", "")
        archive.writestr(f"{package}/" + ("_stride_cuda.so" if cuda else "_native.fixture.so"), "")
        if violation != "missing_license":
            archive.writestr(f"{info}/licenses/LICENSE", "fixture license")
        if violation != "missing_vendor_license":
            archive.writestr(f"{info}/licenses/{vendor}", "fixture upstream license")
        archive.writestr(f"{info}/METADATA",
                         "Name: tensor0-cuda12\nVersion: 0.2.0.dev0\nRequires-Dist: tensor0==0.2.0.dev0\n"
                         "Requires-Dist: nvidia-cuda-runtime-cu12<13,>=12.9.79\n"
                         f"License-Expression: MIT AND Apache-2.0\nLicense-File: LICENSE\nLicense-File: {vendor}\n" if cuda else
                         f"Name: tensor0\nVersion: 0.2.0.dev0\nLicense-Expression: MIT AND Apache-2.0\nLicense-File: LICENSE\nLicense-File: {vendor}\n")
        archive.writestr(f"{info}/WHEEL", "Root-Is-Purelib: false\nTag: " + tag + "\n")
        if violation == "package_licenses":
            archive.writestr(f"{package}/_licenses/old.txt", "stale license layout")
        if violation == "private_file":
            archive.writestr("local/private.txt", "private fixture")
    command = [sys.executable, "-I", str(SCRIPT), str(wheel)]
    if cuda:
        command.append("--cuda")
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert ("Missing archive files" if violation in {"missing_license", "missing_vendor_license"} else
            "Private files or build artifacts") in result.stderr
