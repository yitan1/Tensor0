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
sys.exit(int(os.environ.get("MOCK_FAIL", "0")))
''')
    compiler.chmod(0o755)
    environment = dict(os.environ, CARGO_MANIFEST_DIR=str(manifest), OUT_DIR=str(output),
                       CARGO_CFG_TARGET_OS="linux", CXX=str(compiler), OPT_LEVEL="0",
                       MOCK_LOG=str(log))

    def run(**overrides):
        log.write_text("")
        result = subprocess.run([str(executable)], env=environment | overrides,
                                capture_output=True, text=True)
        commands = [json.loads(line) for line in log.read_text().splitlines()]
        return result, commands

    return run, manifest, output


def test_backend_invalidation_and_failed_compile(build):
    run, manifest, output = build
    result, commands = run()
    assert result.returncode == 0
    assert len(commands) == 1
    assert all("-O0" in command for command in commands)
    result, commands = run()
    assert result.returncode == 0
    assert commands == []
    assert result.stdout.count("cargo:rustc-link-arg=") == 1
    for backend in ("native",):
        source = manifest / backend / "stride_ffi.cc"
        source.write_text(source.read_text() + " changed")
        result, commands = run()
        assert result.returncode == 0
        assert len(commands) == 1
        assert f"{backend}/stride_ffi.cc" in commands[0]
    for name in ("numeric/scalar.inc", "execute/reduction.inc", "layout/record.inc",
                 "kernels/avx2.inc"):
        source = manifest / "native" / name
        source.write_text(source.read_text() + " changed")
        result, commands = run()
        assert result.returncode == 0
        assert len(commands) == 1
        assert "native/stride_ffi.cc" in commands[0]
    header = manifest / "vendor/jaxlib-0.10.1/include/xla/ffi/api/ffi.h"
    header.write_text("changed header")
    assert len(run()[1]) == 1
    assert len(run(MOCK_VERSION="compiler 2")[1]) == 1
    assert len(run(MOCK_VERSION="compiler 2")[1]) == 0
    assert len(run(CPATH=str(manifest))[1]) == 1
    assert len(run(CPATH=str(manifest))[1]) == 0
    assert len(run()[1]) == 1
    (output / "tensor0_stride_ffi.o").unlink()
    assert len(run()[1]) == 1
    source = manifest / "native/stride_ffi.cc"
    original = source.read_text()
    source.write_text(original + " invalid")
    result, commands = run(MOCK_FAIL="1")
    assert result.returncode != 0
    assert len(commands) == 1
    source.write_text(original)
    result, commands = run()
    assert result.returncode == 0
    assert len(commands) == 1
    assert commands[0][-4:] == ["-c", "native/stride_ffi.cc", "-o", str(output / "tensor0_stride_ffi.o")]
    assert run()[1] == []
    compiler = output.parent / "another-compiler"
    shutil.copy2(output.parent / "compiler", compiler)
    assert len(run(CXX=str(compiler))[1]) == 1
    assert run(CXX=str(compiler))[1] == []


@pytest.mark.parametrize("level,flag", [("1", "-O1"), ("2", "-O2"), ("3", "-O3"),
                                        ("s", "-Os"), ("z", "-Os")])
def test_profile_optimization_invalidates_objects(build, level, flag):
    run, _, _ = build
    assert run()[0].returncode == 0
    result, commands = run(OPT_LEVEL=level)
    assert result.returncode == 0
    assert len(commands) == 1
    assert all(flag in command and "-O0" not in command for command in commands)
    assert run(OPT_LEVEL=level)[1] == []
