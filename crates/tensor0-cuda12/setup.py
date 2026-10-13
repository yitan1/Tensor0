"""Package the standalone C-ABI backend, sharing canonical checkout sources."""

from pathlib import Path
import shutil
import subprocess

from setuptools import Distribution, setup
from setuptools.command.build_py import build_py
from setuptools.command.sdist import sdist
from setuptools.command.bdist_wheel import bdist_wheel

ROOT = Path(__file__).resolve().parent
# A released sdist is self-contained; only a checkout uses the canonical tree.
SHARED = ROOT if (ROOT / "build_helpers.rs").is_file() else ROOT.parent / "tensor0-py"
# Stage canonical licenses before setuptools collects PEP 639 metadata.
LICENSE_ROOT = ROOT if SHARED == ROOT else ROOT.parents[1]
LICENSE_FILES = ("LICENSE", "vendor/jaxlib-0.11.2/LICENSE.txt")
if LICENSE_ROOT != ROOT:
    for name in LICENSE_FILES:
        source = LICENSE_ROOT / name if name == "LICENSE" else SHARED / name
        (ROOT / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, ROOT / name)


def _sources(destination):
    # Preserve the relative include tree without copying shared code into the checkout.
    native = destination / "native"
    shutil.copytree(ROOT / "native/cuda", native / "cuda", dirs_exist_ok=True)
    for name in ("cudart_forwarder.cc", "cudart_forwarder.map"):
        shutil.copy2(ROOT / "native" / name, native / name)
    ffi = native / "ffi"
    ffi.mkdir(parents=True, exist_ok=True)
    for name in ("errors.h", "buffers.h"):
        shutil.copy2(SHARED / "native/ffi" / name, ffi / name)
    layout = native / "layout"
    layout.mkdir(parents=True, exist_ok=True)
    for name in ("descriptor.cc", "descriptor.h", "record.cc", "record.h", "owner_fiber.h"):
        shutil.copy2(SHARED / "native/layout" / name, layout / name)
    shutil.copytree(SHARED / "vendor", destination / "vendor", dirs_exist_ok=True)
    shutil.copy2(SHARED / "build_helpers.rs", destination / "build_helpers.rs")


class SourceDistribution(sdist):
    def make_release_tree(self, base_dir, files):
        super().make_release_tree(base_dir, files)
        release = Path(base_dir)
        # Only CUDA sources, their shared layout helpers and vendored FFI headers.
        _sources(release)
        for name in LICENSE_FILES:
            shutil.copy2(ROOT / name, release / name)


class BinaryBuild(build_py):
    def run(self):
        super().run()
        output = Path(self.build_lib).resolve() / "tensor0_cuda12"
        output.mkdir(parents=True, exist_ok=True)
        build = Path(self.get_finalized_command("build").build_temp).resolve()
        build.mkdir(parents=True, exist_ok=True)
        # The std-only builder needs no Cargo workspace or downloaded Rust crates.
        builder_source = build / "builder.rs"
        shutil.copy2(ROOT / "builder.rs", builder_source)
        shutil.copy2(SHARED / "build_helpers.rs", build / "build_helpers.rs")
        executable = build / "cuda-builder"
        subprocess.run(["rustc", "--edition=2021", str(builder_source), "-o", str(executable)], check=True)
        sources = build / "sources"
        _sources(sources)
        subprocess.run([str(executable), str(sources), str(build / "native")], check=True)
        shutil.copy2(build / "native/_stride_cuda.so", output / "_stride_cuda.so")


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        # Native package data belongs in platlib, despite having no Python extension.
        return True


class PlatformWheel(bdist_wheel):
    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self):
        # No Python ABI, but this experimental ELF targets only Linux x86_64.
        import platform
        if platform.system() != "Linux" or platform.machine() != "x86_64":
            raise RuntimeError("tensor0-cuda12 supports only Linux x86_64")
        return "py3", "none", "linux_x86_64"


setup(distclass=BinaryDistribution, cmdclass={"sdist": SourceDistribution, "build_py": BinaryBuild, "bdist_wheel": PlatformWheel})
