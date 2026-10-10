"""CPU-only checks of the manual installed-wheel CUDA acceptance gate."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import subprocess
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    "cuda_acceptance", Path(__file__).resolve().parents[1] / ".github/scripts/cuda_acceptance.py")
assert spec is not None and spec.loader is not None
acceptance = importlib.util.module_from_spec(spec)
spec.loader.exec_module(acceptance)


@pytest.fixture
def wheel_info(tmp_path):
    wheel = tmp_path / "tensor0-test.whl"
    wheel.write_bytes(b"a wheel archive")
    build = dict.fromkeys(acceptance.BUILD_FIELDS, "test")
    build.update(wheel=wheel.name, sha256=hashlib.sha256(wheel.read_bytes()).hexdigest())
    direct = {"url": wheel.as_uri(), "archive_info": {"hashes": {"sha256": build["sha256"]}}}
    return wheel, build, direct


def test_wheel_binding(wheel_info):
    wheel, build, direct = wheel_info
    assert acceptance.verify_wheel(wheel, build, direct) == build["sha256"]
    direct["archive_info"] = {"hash": "sha256=" + build["sha256"]}
    assert acceptance.verify_wheel(wheel, build, direct) == build["sha256"]


@pytest.mark.parametrize("change", ["bytes", "basename", "installed_hash", "installed_name", "editable", "field"])
def test_wheel_binding_rejects(wheel_info, change):
    wheel, build, direct = wheel_info
    if change == "bytes":
        wheel.write_bytes(b"changed")
    elif change == "basename":
        build["wheel"] = "other.whl"
    elif change == "installed_hash":
        direct["archive_info"]["hashes"]["sha256"] = "0" * 64
    elif change == "installed_name":
        direct["url"] = "file:///other.whl"
    elif change == "editable":
        direct["dir_info"] = {"editable": True}
    else:
        del build["cuda_arch"]
    with pytest.raises(RuntimeError):
        acceptance.verify_wheel(wheel, build, direct)


@pytest.mark.parametrize("body,code,accepted", [
    ('<testcase name="pass"/>', 0, True),
    ('<testcase name="pass"/>', 1, False),
    ('', 0, False),
    ('<testcase><failure/></testcase>', 0, False),
    ('<testcase><error/></testcase>', 0, False),
    ('<testcase><skipped type="pytest.skip" message="requires CPU-only extension"/></testcase>', 0, False),
    ('<testcase/><testcase><skipped type="pytest.skip" message="requires CPU-only extension"/></testcase>', 0, True),
    ('<testcase/><testcase><skipped type="pytest.skip" message="requires a CPU-only extension"/></testcase>', 0, True),
    ('<testcase/><testcase><skipped type="pytest.skip" message="CUDA device is unavailable"/></testcase>', 0, False),
    ('<testcase/><testcase><skipped type="pytest.xfail" message="requires CPU-only extension"/></testcase>', 0, False),
])
def test_junit_policy(tmp_path, body, code, accepted):
    path = tmp_path / "junit.xml"
    path.write_text(f'<testsuites><testsuite>{body}</testsuite></testsuites>')
    assert acceptance.junit_result(path, code)["accepted"] is accepted


def test_environment(monkeypatch):
    for key in ("PYTHONPATH", "PYTHONHOME", "PYTEST_ADDOPTS", "PYTEST_PLUGINS", "XLA_FLAGS"):
        monkeypatch.setenv(key, "pollution")
    env = acceptance.environment()
    assert not any(k in env for k in ("PYTHONPATH", "PYTHONHOME", "PYTEST_ADDOPTS", "PYTEST_PLUGINS", "XLA_FLAGS"))
    assert env["JAX_PLATFORMS"] == "cuda,cpu"
    assert env["JAX_DISABLE_JIT"] == "0"


@pytest.mark.parametrize("problem", [None, "noenv", "not_isolated", "dirty", "commit", "origin",
                                     "cuda", "versions", "registration", "gpu", "cpu", "driver"])
def test_preflight(monkeypatch, wheel_info, problem):
    wheel, build, direct = wheel_info
    monkeypatch.setattr(acceptance, "sys", SimpleNamespace(
        prefix="/venv", base_prefix="/venv" if problem == "noenv" else "/base",
        flags=SimpleNamespace(isolated=problem != "not_isolated"), version=sys.version))
    monkeypatch.setattr(acceptance.subprocess, "check_output", lambda cmd, **kw:
                        ("dirty" if problem == "dirty" else "") if "status" in cmd
                        else ("wrong" if problem == "commit" else build["source_commit"]))
    monkeypatch.setattr(acceptance.metadata, "distribution", lambda name:
                        SimpleNamespace(read_text=lambda filename: json.dumps(direct)))
    monkeypatch.setattr(acceptance.metadata, "version", lambda name: "test")
    native = SimpleNamespace(
        __file__="/checkout/native.so" if problem == "origin" else "/venv/native.so",
        _stride_cuda_available=lambda: problem != "cuda",
        _stride_ffi_build_versions=lambda: ("wrong", "test") if problem == "versions" else ("test", "test"),
        _stride_cuda_registration=lambda: {} if problem == "registration" else {"target": object()},
    )
    package = SimpleNamespace(__file__="/venv/tensor0/__init__.py", _native=native)
    monkeypatch.setitem(sys.modules, "tensor0", package)
    monkeypatch.setitem(sys.modules, "tensor0._native", native)
    monkeypatch.setitem(sys.modules, "jax", SimpleNamespace(
        __version__="test", devices=lambda backend: [] if problem == backend or
        (backend == "cuda" and problem == "gpu") else ["device"], default_backend=lambda: "gpu"))
    monkeypatch.setattr(acceptance.subprocess, "run", lambda *args, **kw:
                        SimpleNamespace(returncode=int(problem == "driver"), stdout="driver", stderr=""))
    # Restore the environment after preflight sets JAX_PLATFORMS.
    monkeypatch.setattr(acceptance.os, "environ", dict(acceptance.os.environ))
    if problem:
        with pytest.raises(RuntimeError):
            acceptance.preflight(wheel, build, {})
    else:
        report = {}
        acceptance.preflight(wheel, build, report)
        assert report["devices"] == {"cpu": ["device"], "cuda": ["device"]}
        assert report["sha256"] == build["sha256"]


def test_main_failure_report_and_existing_output(tmp_path, monkeypatch):
    build = tmp_path / "build.json"
    build.write_text('{}')
    output = tmp_path / "results"
    monkeypatch.setattr(acceptance, "preflight", lambda *args: (_ for _ in ()).throw(RuntimeError("no GPU")))
    args = ["--wheel", str(tmp_path / "test.whl"), "--build-info", str(build), "--output", str(output)]
    assert acceptance.main(args) == 1
    report = json.loads((output / "report.json").read_text())
    assert not report["accepted"] and "no GPU" in report["error"]
    assert (output / "preflight.log").exists()
    with pytest.raises(FileExistsError):
        acceptance.main(args)


@pytest.mark.parametrize("empty_suite", [False, True])
def test_fixed_suites_all_run_and_each_needs_passes(tmp_path, monkeypatch, empty_suite):
    build = tmp_path / "build.json"
    build.write_text('{}')
    output = tmp_path / "results"
    monkeypatch.setattr(acceptance, "preflight", lambda *args: None)
    calls = []

    def run(command, **kwargs):
        if command[0] == "git":
            assert not output.exists()
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        calls.append(command[4])
        xml = Path(command[-1].split("=", 1)[1])
        xml.write_text('<testsuites><testsuite>' + ('<testcase/>' if not empty_suite or len(calls) != 2 else '') +
                       '</testsuite></testsuites>')
        assert command[:4] == [sys.executable, "-I", "-m", "pytest"]
        assert "xfail_strict=true" in command
        assert kwargs["cwd"] == acceptance.ROOT
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(acceptance.subprocess, "run", run)
    assert acceptance.main(["--wheel", "test.whl", "--build-info", str(build),
                            "--output", str(output)]) == int(empty_suite)
    assert calls == list(acceptance.SUITES)
    report = json.loads((output / "report.json").read_text())
    assert [s["accepted"] for s in report["suites"]] == [True, not empty_suite, True]
    assert report["accepted"] is not empty_suite


def test_real_pytest_junit_skip_format(tmp_path):
    source = tmp_path / "test_sample.py"
    source.write_text('import pytest\n'
                      'def test_pass(): pass\n'
                      'def test_skip(): pytest.skip("requires a CPU-only extension")\n')
    xml = tmp_path / "result.xml"
    process = subprocess.run(
        [sys.executable, "-I", "-m", "pytest", str(source), "-q", f"--junitxml={xml}"],
        cwd=tmp_path, env=acceptance.environment(), capture_output=True, text=True, timeout=30)
    result = acceptance.junit_result(xml, process.returncode)
    assert result["accepted"], process.stdout + process.stderr
    assert result["passed"] == 1 and len(result["skipped"]) == 1
