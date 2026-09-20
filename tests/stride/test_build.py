"""Native build cache invalidation and optimization profiles."""

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest


@pytest.fixture
def build(tmp_path):
    rustc = shutil.which("rustc")
    if rustc is None:
        pytest.skip("rustc is required")
    root = Path(__file__).resolve().parents[2]
    script = root / "crates/tensor0-py/build.rs"
    executable = tmp_path / "build-script"
    subprocess.run([rustc, "--edition=2021", str(script), "-o", str(executable)], check=True)
    manifest = tmp_path / "crate"
    sources = re.findall(r'"(native/[^"\n]+)"', script.read_text())
    for source in sources:
        path = manifest / source
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    for source in ("numeric/scalar.h", "execute/reduction.h", "kernels/avx2.h", "layout/record.h"):
        path = manifest / "native" / source
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    headers = manifest / "vendor/jaxlib-0.10.1/include/xla/ffi/api"
    headers.mkdir(parents=True)
    for header in ("api.h", "c_api.h", "ffi.h"):
        (headers / header).write_text(header)
    output = tmp_path / "out"
    output.mkdir()
    log = tmp_path / "commands.jsonl"
    compiler = tmp_path / "compiler"
    compiler.write_text(f"#!{sys.executable}\n" + '''import json, os, pathlib, sys
if sys.argv[1:] == ["--version"]:
    print(os.environ.get("MOCK_VERSION", "compiler 1"))
    sys.exit(0)
with open(os.environ["MOCK_LOG"], "a") as log:
    log.write(json.dumps(sys.argv[1:]) + "\\n")
pathlib.Path(sys.argv[sys.argv.index("-o") + 1]).write_text("object")
source = sys.argv[sys.argv.index("-c") + 1]
root = pathlib.Path.cwd()
# This mock conservatively shares headers across objects; real depfiles are
# exercised separately with actual compiler invocations.
deps = [source] + [str(p) for p in root.rglob("*") if p.suffix in (".h",)]
def escape(path):
    return path.replace("$", "$$").replace(" ", "\\ ").replace("#", "\\#")
pathlib.Path(sys.argv[sys.argv.index("-MF") + 1]).write_text(
    "tensor0-object: " + " ".join(escape(p) for p in deps) + "\\n")
sys.exit(int(os.environ.get("MOCK_FAIL", "0")))
''')
    compiler.chmod(0o755)
    environment = dict(os.environ, CARGO_MANIFEST_DIR=str(manifest), OUT_DIR=str(output),
                       CARGO_CFG_TARGET_OS="linux", CXX=str(compiler), OPT_LEVEL="0",
                       MOCK_LOG=str(log))
    environment.pop("TENSOR0_CXX_OPT_LEVEL", None)

    def run(**overrides):
        log.write_text("")
        result = subprocess.run([str(executable)], env=environment | overrides,
                                capture_output=True, text=True)
        commands = [json.loads(line) for line in log.read_text().splitlines()]
        return result, commands

    run.unit_count = len(sources)
    return run, manifest, output


def test_backend_invalidation_and_failed_compile(build):
    run, manifest, output = build
    result, commands = run()
    assert result.returncode == 0
    assert len(commands) == run.unit_count
    assert all("-O0" in command for command in commands)
    result, commands = run()
    assert result.returncode == 0
    assert commands == []
    assert result.stdout.count("cargo:rustc-link-arg=") == run.unit_count
    for operation in ("copy", "update", "reduction", "dot"):
        source = manifest / "native/ffi" / f"{operation}.cc"
        source.write_text(source.read_text() + " changed")
        result, commands = run()
        assert result.returncode == 0
        assert len(commands) == 1
        assert f"native/ffi/{operation}.cc" in commands[0]
    for name in ("numeric/scalar.h", "execute/reduction.h", "layout/record.h",
                 "kernels/avx2.h"):
        source = manifest / "native" / name
        source.write_text(source.read_text() + " changed")
        result, commands = run()
        assert result.returncode == 0
        assert len(commands) == run.unit_count
        assert any("native/ffi/update.cc" in command for command in commands)
    header = manifest / "vendor/jaxlib-0.10.1/include/xla/ffi/api/ffi.h"
    header.write_text("changed header")
    assert len(run()[1]) == run.unit_count
    assert len(run(MOCK_VERSION="compiler 2")[1]) == run.unit_count
    assert len(run(MOCK_VERSION="compiler 2")[1]) == 0
    assert len(run(CPATH=str(manifest))[1]) == run.unit_count
    assert len(run(CPATH=str(manifest))[1]) == 0
    assert len(run()[1]) == run.unit_count
    (output / "native_ffi_update.o").unlink()
    assert len(run()[1]) == 1
    source = manifest / "native/ffi/update.cc"
    original = source.read_text()
    source.write_text(original + " invalid")
    result, commands = run(MOCK_FAIL="1")
    assert result.returncode != 0
    assert len(commands) == 1
    source.write_text(original)
    result, commands = run()
    assert result.returncode == 0
    assert len(commands) == 1
    assert commands[0][-4:] == ["-c", "native/ffi/update.cc", "-o", str(output / "native_ffi_update.o")]
    assert run()[1] == []
    compiler = output.parent / "another-compiler"
    shutil.copy2(output.parent / "compiler", compiler)
    assert len(run(CXX=str(compiler))[1]) == run.unit_count
    assert run(CXX=str(compiler))[1] == []


@pytest.mark.parametrize("level,flag", [("1", "-O1"), ("2", "-O2"), ("3", "-O3"),
                                        ("s", "-Os"), ("z", "-Os")])
def test_profile_optimization_invalidates_objects(build, level, flag):
    run, _, _ = build
    assert run()[0].returncode == 0
    result, commands = run(OPT_LEVEL=level)
    assert result.returncode == 0
    assert len(commands) == run.unit_count
    assert all(flag in command and "-O0" not in command for command in commands)
    assert run(OPT_LEVEL=level)[1] == []


@pytest.mark.parametrize("level,flag", [("0", "-O0"), ("1", "-O1"), ("2", "-O2"),
                                        ("3", "-O3"), ("s", "-Os"), ("z", "-Os")])
def test_cpp_optimization_override(build, level, flag):
    run, _, _ = build
    result, commands = run(OPT_LEVEL="3", TENSOR0_CXX_OPT_LEVEL=level)
    assert result.returncode == 0
    assert flag in commands[0]
    assert "cargo:rerun-if-env-changed=TENSOR0_CXX_OPT_LEVEL" in result.stdout
    assert run(OPT_LEVEL="3", TENSOR0_CXX_OPT_LEVEL=level)[1] == []
    result, commands = run(OPT_LEVEL="3")
    assert result.returncode == 0
    assert len(commands) == (0 if level == "3" else run.unit_count)
    if commands:
        assert "-O3" in commands[0]


@pytest.mark.parametrize("level", ["", "4", "fast", "-O2", "2 -ffast-math"])
def test_invalid_cpp_optimization_override(build, level):
    run, _, _ = build
    result, commands = run(TENSOR0_CXX_OPT_LEVEL=level)
    assert result.returncode != 0
    assert "C++ optimization level must be" in result.stderr
    assert commands == []
