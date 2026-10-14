"""Production build.rs cache, jobserver, CUDA linkage and optimization checks.

Most cases use a compiled build script with synthetic sources and mock compilers;
selected cases compile/link small real C++ and Cargo fixtures (offline).
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tomllib

import pytest

from tests.stride.support.paths import REPO_ROOT


pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="native build requires Linux")


@pytest.fixture(scope="session")
def build_script(tmp_path_factory):
    cargo = shutil.which("cargo")
    if cargo is None:
        pytest.skip("cargo is required")
    directory = tmp_path_factory.mktemp("native-build-script")
    package = REPO_ROOT / "crates/tensor0-py"
    dependency = tomllib.loads((package / "Cargo.toml").read_text())["build-dependencies"]["jobserver"]
    (directory / "Cargo.toml").write_text(
        '[workspace]\n[package]\nname = "native-build-test"\nversion = "0.0.0"\nedition = "2021"\n'
        '[dependencies]\njobserver = ' + json.dumps(dependency) + '\n'
        '[[bin]]\nname = "build-script"\npath = ' + json.dumps(str(package / "build.rs")) + '\n'
    )
    subprocess.run([cargo, "build", "--offline", "--manifest-path", str(directory / "Cargo.toml")],
                   check=True, capture_output=True, text=True, timeout=90)
    return directory / "target/debug/build-script"


@pytest.fixture(scope="session")
def cuda_build_script(tmp_path_factory):
    directory = tmp_path_factory.mktemp("cuda-builder")
    shutil.copy2(REPO_ROOT / "crates/tensor0-cuda12/builder.rs", directory / "builder.rs")
    shutil.copy2(REPO_ROOT / "crates/tensor0-py/build_helpers.rs", directory / "build_helpers.rs")
    executable = directory / "builder"
    subprocess.run(["rustc", "--edition=2021", str(directory / "builder.rs"), "-o", str(executable)],
                   check=True, capture_output=True, text=True)
    return executable


@pytest.fixture
def build(tmp_path, build_script):
    return mock_build(tmp_path, build_script, False)


@pytest.fixture
def cuda_build(tmp_path, cuda_build_script):
    return mock_build(tmp_path, cuda_build_script, True)


def mock_build(tmp_path, executable, cuda):
    script = REPO_ROOT / ("crates/tensor0-cuda12/builder.rs" if cuda else "crates/tensor0-py/build.rs")
    manifest = tmp_path / ("crates/tensor0-cuda12" if cuda else "crates/tensor0-py")
    sources = list(dict.fromkeys(re.findall(r'"(native/[^"\n]+)"', script.read_text())))
    for source in sources:
        path = manifest / source
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    for source in ("numeric/scalar.h", "execute/reduction.h", "kernels/avx2.h", "layout/record.h"):
        path = manifest / "native" / source
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    headers = manifest / "vendor/jaxlib-0.11.2/include/xla/ffi/api"
    headers.mkdir(parents=True)
    for header in ("api.h", "c_api.h", "ffi.h"):
        (headers / header).write_text(header)
    (tmp_path / "src/tensor0").mkdir(parents=True)
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
if "-shared" in sys.argv:
    if os.environ.get("MOCK_REAL_COMPILER"):
        import subprocess
        subprocess.run([os.environ["MOCK_REAL_COMPILER"], *sys.argv[1:]], check=True)
    sys.exit(int(os.environ.get("MOCK_FAIL", "0")))
source = sys.argv[sys.argv.index("-c") + 1]
root = pathlib.Path.cwd()
if os.environ.get("MOCK_CONCURRENCY"):
    import fcntl, time
    state = pathlib.Path(os.environ["MOCK_CONCURRENCY"])
    def update(delta):
        with state.open("a+") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            stream.seek(0)
            active, peak = json.loads(stream.read() or "[0, 0]")
            active += delta
            peak = max(active, peak)
            stream.seek(0)
            stream.truncate()
            json.dump([active, peak], stream)
            return peak
    update(1)
    deadline = time.monotonic() + 10
    while update(0) < int(os.environ.get("MOCK_EXPECTED_JOBS", os.environ["NUM_JOBS"])):
        if time.monotonic() > deadline:
            update(-1)
            sys.exit(2)
        time.sleep(0.01)
    time.sleep(0.05)
    update(-1)
# This mock conservatively shares headers across objects; real depfiles are
# exercised separately with actual compiler invocations.
deps = [source] + [str(p) for p in root.rglob("*")
                   if p.suffix == ".h" or (source.endswith(".cu") and p.suffix == ".cuh")]
def escape(path):
    return path.replace("$", "$$").replace(" ", "\\\\ ").replace("#", "\\\\#")
pathlib.Path(sys.argv[sys.argv.index("-MF") + 1]).write_text(
    "tensor0-object: " + " ".join(escape(p) for p in deps) + "\\n")
if os.environ.get("MOCK_REAL_COMPILER"):
    import subprocess
    subprocess.run([os.environ["MOCK_REAL_COMPILER"], *sys.argv[1:]], check=True)
sys.exit(int(os.environ.get("MOCK_FAIL", "0")) or
         int(source == os.environ.get("MOCK_FAIL_SOURCE")))
''')
    compiler.chmod(0o755)
    environment = dict(os.environ, CARGO_MANIFEST_DIR=str(manifest), OUT_DIR=str(output),
                       CARGO_CFG_TARGET_OS="linux", CXX=str(compiler), OPT_LEVEL="0",
                       MOCK_LOG=str(log), NUM_JOBS="1")
    for name in ("TENSOR0_CXX_OPT_LEVEL", "TENSOR0_CUDA", "NVCC", "CUDA_HOME",
                 "TENSOR0_CUDA_ARCH", "CARGO_MAKEFLAGS", "MAKEFLAGS", "MFLAGS"):
        environment.pop(name, None)

    def run(*, slots="auto", **overrides):
        log.write_text("")
        env = environment | overrides
        argv = [str(executable)] + ([str(manifest), str(output)] if cuda else [])
        if slots == "auto":
            try:
                slots = max(0, int(env["NUM_JOBS"]) - 1)
            except ValueError:
                slots = 0
        if slots is None:
            result = subprocess.run(argv, env=env,
                                    capture_output=True, text=True, timeout=30)
        else:
            read_fd, write_fd = os.pipe()
            try:
                os.write(write_fd, b"+" * slots)
                env["CARGO_MAKEFLAGS"] = f"--jobserver-auth={read_fd},{write_fd} -j"
                result = subprocess.run(argv, env=env, pass_fds=(read_fd, write_fd),
                                        capture_output=True, text=True, timeout=30)
                # Every borrowed slot must be returned on success and failure.
                os.set_blocking(read_fd, False)
                try:
                    returned = os.read(read_fd, 1024)
                except BlockingIOError:
                    returned = b""
                assert len(returned) == slots
            finally:
                os.close(read_fd)
                os.close(write_fd)
        commands = [json.loads(line) for line in log.read_text().splitlines()]
        run.link_commands = [command for command in commands if "-shared" in command]
        return result, [command for command in commands if "-c" in command]

    run.unit_count = sum(source.endswith(".cc") for source in sources)
    return run, manifest, output


def test_mock_compiler_has_no_syntax_warnings(build):
    _, _, output = build
    result = subprocess.run(
        [sys.executable, "-Werror", str(output.parent / "compiler"), "--version"],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""


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
    for operation in ("copy", "update_a", "update_b", "reduction_a", "reduction_b", "dot_a", "dot_b"):
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
        assert any("native/ffi/update_a.cc" in command for command in commands)
    header = manifest / "vendor/jaxlib-0.11.2/include/xla/ffi/api/ffi.h"
    header.write_text("changed header")
    assert len(run()[1]) == run.unit_count
    assert len(run(MOCK_VERSION="compiler 2")[1]) == run.unit_count
    assert len(run(MOCK_VERSION="compiler 2")[1]) == 0
    assert len(run(CPATH=str(manifest))[1]) == run.unit_count
    assert len(run(CPATH=str(manifest))[1]) == 0
    assert len(run()[1]) == run.unit_count
    (output / "native_ffi_update_a.o").unlink()
    assert len(run()[1]) == 1
    source = manifest / "native/ffi/update_a.cc"
    original = source.read_text()
    source.write_text(original + " invalid")
    result, commands = run(MOCK_FAIL="1")
    assert result.returncode != 0
    assert len(commands) == 1
    source.write_text(original)
    result, commands = run()
    assert result.returncode == 0
    assert len(commands) == 1
    assert commands[0][-4:] == ["-c", "native/ffi/update_a.cc", "-o", str(output / "native_ffi_update_a.o")]
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


@pytest.mark.parametrize("jobs", [1, 2, 3])
def test_parallel_compilation_is_bounded_and_preserves_link_order(build, jobs):
    run, _, output = build
    state = output / "concurrency.json"
    result, commands = run(NUM_JOBS=str(jobs), MOCK_CONCURRENCY=str(state))
    assert result.returncode == 0, result.stderr
    assert len(commands) == run.unit_count
    assert json.loads(state.read_text()) == [0, jobs]
    assert f"checking {run.unit_count} units with up to {jobs} workers" in result.stderr
    assert result.stderr.count("tensor0-native: compiling ") == run.unit_count
    completed = re.findall(r"tensor0-native: compiled native/\S+ in \d+\.\d{2}s", result.stderr)
    if len(completed) != run.unit_count:
        sources = [command[command.index("-c") + 1] for command in commands]
        # Avoid pytest's truncated assertion repr when diagnosing missing log lines.
        pytest.fail(
            f"NUM_JOBS={jobs}: expected {run.unit_count} completion messages, got {len(completed)}\n"
            f"Compiled sources: {sources!r}\nMatched completions: {completed!r}\n"
            f"Full build-script stderr:\n{result.stderr}",
            pytrace=False,
        )
    assert "tensor0-native: native objects ready in " in result.stderr
    links = [line for line in result.stdout.splitlines()
             if line.startswith("cargo:rustc-link-arg=")]
    cached, commands = run(NUM_JOBS="1")
    assert cached.returncode == 0
    assert commands == []
    assert cached.stderr.count("tensor0-native: cached ") == run.unit_count
    assert "tensor0-native: compiling " not in cached.stderr
    assert links == [line for line in cached.stdout.splitlines()
                     if line.startswith("cargo:rustc-link-arg=")]


def test_parallel_failure_waits_for_workers_and_retries_failed_object(build):
    run, _, output = build
    state = output / "concurrency.json"
    # Start every unit before failure, so this checks joining in-flight workers
    # independently of which worker observes cancellation first.
    result, commands = run(NUM_JOBS=str(run.unit_count), MOCK_CONCURRENCY=str(state),
                           MOCK_FAIL_SOURCE="native/ffi/copy.cc")
    assert result.returncode != 0
    assert len(commands) == run.unit_count
    assert json.loads(state.read_text()) == [0, run.unit_count]
    assert "tensor0-native: failed native/ffi/copy.cc in " in result.stderr
    assert "tensor0-native: compiled native/ffi/copy.cc" not in result.stderr
    assert "tensor0-native: native objects ready" not in result.stderr
    assert "cargo:rustc-link-arg=" not in result.stdout
    result, commands = run(NUM_JOBS="3")
    assert result.returncode == 0, result.stderr
    assert len(commands) == 1
    assert "native/ffi/copy.cc" in commands[0]
    assert run(NUM_JOBS="3")[1] == []


def test_failure_stops_pending_units(build):
    run, _, output = build
    result, commands = run(NUM_JOBS="1", MOCK_FAIL_SOURCE="native/execute/scheduling.cc")
    assert result.returncode != 0
    assert len(commands) == 1
    assert "native/execute/scheduling.cc" in commands[0]
    assert "cargo:rustc-link-arg=" not in result.stdout
    assert (output / "native_execute_scheduling.fingerprint").read_text() == ""
    result, commands = run(NUM_JOBS="3")
    assert result.returncode == 0, result.stderr
    assert len(commands) == run.unit_count
    assert run(NUM_JOBS="3")[1] == []


@pytest.mark.parametrize("jobs", ["0", "invalid", "-1"])
def test_invalid_job_count(build, jobs):
    run, _, _ = build
    result, commands = run(NUM_JOBS=jobs)
    assert result.returncode != 0
    assert "NUM_JOBS must be a positive integer" in result.stderr
    assert commands == []


@pytest.mark.parametrize("operation", ["update", "dot", "reduction"])
def test_parallel_real_compiler_objects_link_and_cache(build, operation):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler is required")
    run, manifest, output = build
    sources = sorted((manifest / "native").rglob("*.cc"))
    for index, source in enumerate(sources):
        source.write_text(f'int native_unit_{index}() {{ return {index}; }}\n')
    header = manifest / f"native/ffi/{operation}_impl.h"
    header.write_text("#pragma once\ninline int update_value() { return 1; }\n")
    for name in (f"{operation}_a", f"{operation}_b"):
        (manifest / f"native/ffi/{name}.cc").write_text(
            f'#include "{operation}_impl.h"\n'
            f'int {name}_unit() {{ return update_value(); }}\n'
        )
    result, _ = run(NUM_JOBS="3", CXX=compiler)
    assert result.returncode == 0, result.stderr
    objects = sorted(output.glob("*.o"))
    assert len(objects) == run.unit_count
    before = {path: path.stat().st_mtime_ns for path in objects}
    subprocess.run([compiler, "-shared", *(str(path) for path in objects),
                    "-o", str(output / "native.so")], check=True, timeout=30)
    assert run(NUM_JOBS="1", CXX=compiler)[0].returncode == 0
    assert {path: path.stat().st_mtime_ns for path in objects} == before
    header.write_text(header.read_text().replace("return 1", "return 2"))
    result, _ = run(NUM_JOBS="3", CXX=compiler)
    assert result.returncode == 0, result.stderr
    changed = [path.name for path in objects
               if path.stat().st_mtime_ns != before[path]]
    assert changed == [f"native_ffi_{operation}_a.o", f"native_ffi_{operation}_b.o"]
    before = {path: path.stat().st_mtime_ns for path in objects}
    source = manifest / f"native/ffi/{operation}_a.cc"
    source.write_text(source.read_text() + "// Changed implementation.\n")
    assert run(NUM_JOBS="3", CXX=compiler)[0].returncode == 0
    changed = [path.name for path in objects
               if path.stat().st_mtime_ns != before[path]]
    assert changed == [f"native_ffi_{operation}_a.o"]


@pytest.mark.parametrize("slots", [0, 1, 2])
def test_jobserver_limits_workers_and_cancels_unfulfilled_requests(build, slots):
    run, _, output = build
    state = output / "limited-concurrency.json"
    result, commands = run(slots=slots, NUM_JOBS="11", MOCK_CONCURRENCY=str(state),
                           MOCK_EXPECTED_JOBS=str(slots + 1))
    assert result.returncode == 0, result.stderr
    assert len(commands) == run.unit_count
    assert json.loads(state.read_text()) == [0, slots + 1]


def test_missing_jobserver_defaults_to_serial(build):
    run, _, output = build
    state = output / "serial-concurrency.json"
    result, commands = run(slots=None, NUM_JOBS="11", MOCK_CONCURRENCY=str(state),
                           MOCK_EXPECTED_JOBS="1")
    assert result.returncode == 0, result.stderr
    assert len(commands) == run.unit_count
    assert json.loads(state.read_text()) == [0, 1]
    assert "with up to 1 workers" in result.stderr


def test_failure_cancels_blocked_jobserver_requests(build):
    run, _, _ = build
    result, commands = run(slots=0, NUM_JOBS="11", MOCK_FAIL_SOURCE="native/execute/scheduling.cc")
    assert result.returncode != 0
    assert len(commands) == 1
    assert "failed native/execute/scheduling.cc" in result.stderr
    assert "cargo:rustc-link-arg=" not in result.stdout
    assert run(slots=0, NUM_JOBS="11")[0].returncode == 0


def test_real_cargo_jobserver_and_noop(build):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler is required")
    _, manifest, _ = build
    package = REPO_ROOT / "crates/tensor0-py"
    dependency = tomllib.loads((package / "Cargo.toml").read_text())["build-dependencies"]["jobserver"]
    (manifest / "Cargo.toml").write_text(
        '[workspace]\n[package]\nname = "native-cargo-test"\nversion = "0.0.0"\nedition = "2021"\n'
        'build = ' + json.dumps(str(package / "build.rs")) + '\n'
        '[build-dependencies]\njobserver = ' + json.dumps(dependency) + '\n'
        '[lib]\npath = "lib.rs"\n'
    )
    (manifest / "lib.rs").write_text("pub fn value() -> u32 { 42 }\n")
    for index, source in enumerate(sorted((manifest / "native").rglob("*.cc"))):
        source.write_text(f'int native_unit_{index}() {{ return {index}; }}\n')
    state = manifest.parent / "cargo-concurrency.json"
    env = dict(os.environ, CXX=str(manifest.parents[1] / "compiler"),
               MOCK_REAL_COMPILER=compiler, MOCK_LOG=str(manifest.parent / "cargo-commands.jsonl"),
               MOCK_CONCURRENCY=str(state), MOCK_EXPECTED_JOBS="3", NUM_JOBS="3")
    env.pop("TENSOR0_CXX_OPT_LEVEL", None)
    command = ["cargo", "build", "--offline", "--color=never", "-vv", "-j", "3",
               "--manifest-path", str(manifest / "Cargo.toml")]
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr
    assert "with up to 3 workers" in result.stderr
    assert json.loads(state.read_text()) == [0, 3]
    objects = list((manifest / "target").rglob("native_*.o"))
    assert len(objects) == 13
    before = {path: path.stat().st_mtime_ns for path in objects}
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "Fresh native-cargo-test" in result.stderr
    assert {path: path.stat().st_mtime_ns for path in objects} == before


@pytest.mark.parametrize("jobs,slots,expected", [(3, 8, 3), (20, 19, 13)])
def test_job_count_and_source_count_bound_token_demand(build, jobs, slots, expected):
    run, _, output = build
    state = output / "bounded-concurrency.json"
    result, commands = run(slots=slots, NUM_JOBS=str(jobs), MOCK_CONCURRENCY=str(state),
                           MOCK_EXPECTED_JOBS=str(expected))
    assert result.returncode == 0, result.stderr
    assert len(commands) == run.unit_count
    assert json.loads(state.read_text()) == [0, expected]


def test_core_build_ignores_cuda_environment_and_never_stages(build):
    run, manifest, _ = build
    staged = manifest.parents[1] / "src/tensor0/_stride_cuda.so"
    staged.write_bytes(b"unrelated stale artifact")
    result, commands = run(TENSOR0_CUDA="1", NVCC="/missing/nvcc", CUDA_HOME="/missing")
    assert result.returncode == 0, result.stderr
    assert len(commands) == 13
    assert staged.read_bytes() == b"unrelated stale artifact"
    assert not any("cuda" in command[command.index("-c") + 1] for command in commands)
    assert "cudart" not in result.stdout
    assert "tensor0_stride_cuda" not in result.stdout


def test_cuda_compile_cache_and_linkage(cuda_build):
    run, manifest, output = cuda_build
    toolkit = output.parent / "cuda"
    (toolkit / "lib").mkdir(parents=True)
    (toolkit / "lib/libcudart.so").touch()
    options = dict(TENSOR0_CUDA="1", NVCC=str(output.parent / "compiler"), CUDA_HOME=str(toolkit))
    result, commands = run(**options)
    assert result.returncode == 0, result.stderr
    assert len(commands) == 8
    assert "native/cuda/copy.cu" in commands[-6]
    assert "native/cuda/update.cu" in commands[-5]
    assert "native/cuda/accumulation.cu" in commands[-4]
    assert "native/cuda/dot.cu" in commands[-3]
    assert "native/cuda/reduction.cu" in commands[-2]
    assert "-arch=sm_80" in commands[-2]
    assert "-Xcompiler" in commands[-2] and "-fPIC" in commands[-2]
    assert "-fvisibility=default" in commands[-2]
    assert "cargo:rustc-link-lib=static=cudart_static" not in result.stdout
    assert "cargo:rustc-link-arg=" not in result.stdout
    link, = run.link_commands
    assert "-shared" in link and "-Wl,-z,defs" in link and "-lcudart" not in link
    assert "-ldl" in link and any("--version-script=" in arg for arg in link)
    assert str(output / "native_layout_descriptor.o") in link
    assert str(output / "native_layout_record.o") in link
    assert (output / "_stride_cuda.so").is_file()
    assert not (manifest.parents[1] / "src/tensor0/_stride_cuda.so").exists()
    assert run(**options)[1] == []
    assert run.link_commands == []
    (output / "_stride_cuda.so").unlink()
    assert run(**options)[1] == []
    assert len(run.link_commands) == 1
    (toolkit / "lib/libcudart.so").write_text("changed runtime")
    assert run(**options)[1] == []
    assert run.link_commands == []
    (manifest / "native/cudart_forwarder.map").write_text("changed exports")
    assert run(**options)[1] == []
    assert len(run.link_commands) == 1
    (manifest / "native/cudart_forwarder.cc").write_text("changed forwarder")
    assert len(run(**options)[1]) == 1
    # The mock emits identical object bytes, so no relink is necessary.
    assert run.link_commands == []
    result, commands = run(**options, TENSOR0_CUDA_ARCH="sm_90")
    assert result.returncode == 0, result.stderr
    assert len(commands) == 5
    assert all("-arch=sm_90" in command for command in commands)
    assert run(**options, TENSOR0_CUDA_ARCH="sm_90")[1] == []
    for name in ("copy", "update", "accumulation", "dot", "reduction"):
        source = manifest / f"native/cuda/{name}.cu"
        source.write_text(source.read_text() + " changed")
        result, commands = run(**options, TENSOR0_CUDA_ARCH="sm_90")
        assert result.returncode == 0, result.stderr
        assert len(commands) == 1 and f"native/cuda/{name}.cu" in commands[0]
        (output / f"native_cuda_{name}.o").unlink()
        result, commands = run(**options, TENSOR0_CUDA_ARCH="sm_90",
                               MOCK_FAIL_SOURCE=f"native/cuda/{name}.cu")
        assert result.returncode != 0
        assert (output / f"native_cuda_{name}.fingerprint").read_text() == ""
        assert "cargo:rustc-cfg=tensor0_stride_cuda" not in result.stdout
        assert len(run(**options, TENSOR0_CUDA_ARCH="sm_90")[1]) == 1
        assert run(**options, TENSOR0_CUDA_ARCH="sm_90")[1] == []


@pytest.mark.parametrize("directory", ["lib64", "lib"])
def test_cuda_runtime_is_not_a_link_input(cuda_build, directory):
    run, _, output = cuda_build
    toolkit = output.parent / "cuda"
    runtime_dir = toolkit / directory
    runtime_dir.mkdir(parents=True)
    target = runtime_dir / "libcudart.so.12.9"
    target.write_text("runtime 1")
    runtime = runtime_dir / "libcudart.so"
    runtime.symlink_to(target.name)
    options = dict(TENSOR0_CUDA="1", NVCC=str(output.parent / "compiler"),
                   CUDA_HOME=str(toolkit), MOCK_VERSION="GCC 14 / CUDA 12.9 fixture")
    result, _ = run(**options)
    assert result.returncode == 0, result.stderr
    assert f"cargo:rerun-if-changed={runtime}" not in result.stdout
    assert f"cargo:rerun-if-changed={target}" not in result.stdout
    assert run(**options)[1] == [] and run.link_commands == []
    target.write_text("runtime 2")
    result, commands = run(**options)
    assert result.returncode == 0, result.stderr
    assert commands == [] and run.link_commands == []
    assert run(**options)[1] == [] and run.link_commands == []


@pytest.mark.parametrize("layer", ["execute", "kernels"])
def test_cuda_cache_tracks_nested_headers(cuda_build, layer):
    run, manifest, output = cuda_build
    # Synthetic mock dependency: this tests cache invalidation, not a real include chain.
    header = manifest / f"native/cuda/{layer}/synthetic_header.cuh"
    header.parent.mkdir(parents=True, exist_ok=True)
    header.write_text("original header")
    toolkit = output.parent / "cuda"
    (toolkit / "lib").mkdir(parents=True)
    (toolkit / "lib/libcudart.so").touch()
    options = dict(TENSOR0_CUDA="1", NVCC=str(output.parent / "compiler"), CUDA_HOME=str(toolkit))
    result, _ = run(**options)
    assert result.returncode == 0, result.stderr
    assert f"cargo:rerun-if-changed={header}" in result.stdout
    assert run(**options)[1] == []
    header.write_text("modified header")
    result, commands = run(**options)
    assert result.returncode == 0, result.stderr
    # The mock conservatively shares CUDA headers across all five CUDA units.
    assert len(commands) == 5
    assert all(command[command.index("-c") + 1].endswith(".cu") for command in commands)
    assert run(**options)[1] == []


def test_cuda_compile_uses_implicit_job_slot_after_cpu_workers(cuda_build):
    run, _, output = cuda_build
    state = output / "cuda-concurrency.json"
    (output.parent / "lib").mkdir()
    (output.parent / "lib/libcudart.so").touch()
    result, commands = run(slots=0, NUM_JOBS="9", TENSOR0_CUDA="1",
                           NVCC=str(output.parent / "compiler"), CUDA_HOME=str(output.parent),
                           MOCK_CONCURRENCY=str(state), MOCK_EXPECTED_JOBS="1")
    assert result.returncode == 0, result.stderr
    assert len(commands) == 8
    assert "native/cuda/copy.cu" in commands[-6]
    assert "native/cuda/update.cu" in commands[-5]
    assert "native/cuda/accumulation.cu" in commands[-4]
    assert "native/cuda/dot.cu" in commands[-3]
    assert "native/cuda/reduction.cu" in commands[-2]
    assert json.loads(state.read_text()) == [0, 1]


def test_cuda_cache_tracks_same_path_host_compiler_change(cuda_build):
    run, _, output = cuda_build
    host_compiler = output.parent / "compiler"
    nvcc = output.parent / "nvcc"
    shutil.copy2(host_compiler, nvcc)
    toolkit = output.parent / "cuda"
    (toolkit / "lib").mkdir(parents=True)
    (toolkit / "lib/libcudart.so").touch()
    options = dict(TENSOR0_CUDA="1", NVCC=str(nvcc), CUDA_HOME=str(toolkit),
                   NVCC_CCBIN="/not/the/selected/compiler")
    result, commands = run(**options)
    assert result.returncode == 0, result.stderr
    assert len(commands) == 8
    for cuda_command, source in zip(commands[-6:-1], ("copy", "update", "accumulation", "dot", "reduction")):
        assert f"native/cuda/{source}.cu" in cuda_command
        assert cuda_command[cuda_command.index("-ccbin") + 1] == str(host_compiler)
    assert run(**options)[1] == []
    # Keep NVCC, the host path and --version output unchanged: only host bytes change.
    host_compiler.write_text(host_compiler.read_text() + "\n# Replaced host compiler.\n")
    result, commands = run(**options)
    assert result.returncode == 0, result.stderr
    assert len(commands) == 8
    assert "native/cuda/copy.cu" in commands[-6]
    assert "native/cuda/update.cu" in commands[-5]
    assert "native/cuda/accumulation.cu" in commands[-4]
    assert "native/cuda/dot.cu" in commands[-3]
    assert "native/cuda/reduction.cu" in commands[-2]
    assert run(**options)[1] == []


def test_real_companion_linkage_and_undefined_symbol(cuda_build):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler is required")
    run, manifest, output = cuda_build
    package = REPO_ROOT / "crates/tensor0-py"
    for name in ("descriptor.cc", "descriptor.h", "record.cc", "record.h", "owner_fiber.h"):
        shutil.copy2(package / "native/layout" / name, manifest / "native/layout" / name)
    for source in (manifest / "native/cuda").glob("*.cu"):
        source.write_text('extern "C" int ' + source.stem + '_value() { return 1; }\n')
    # This fixture checks real linker policy, not the SDK ABI (tested separately).
    (manifest / "native/cudart_forwarder.cc").write_text('extern "C" int Tensor0Fixture() { return 1; }\n')
    shutil.copy2(REPO_ROOT / "crates/tensor0-cuda12/native/cudart_forwarder.map",
                 manifest / "native/cudart_forwarder.map")
    nvcc = output.parent / "nvcc"
    nvcc.write_text(f"#!{sys.executable}\n" + 'import subprocess, sys\nif sys.argv[1:] == ["--version"]: print("mock NVCC")\nelse:\n    def val(flag): return sys.argv[sys.argv.index(flag)+1]\n    subprocess.run([val("-ccbin"), "-x", "c++", "-fPIC", "-MD", "-MT", "tensor0-object", "-MF", val("-MF"), "-c", val("-c"), "-o", val("-o")], check=True)\n')
    nvcc.chmod(0o755)
    options = dict(CXX=compiler, NVCC=str(nvcc), CUDA_HOME=str(output.parent),
                   LD_RUN_PATH="/unexpected/toolkit/lib64")
    result, _ = run(**options)
    assert result.returncode == 0, result.stderr
    elf = subprocess.check_output(["readelf", "-d", str(output / "_stride_cuda.so")], text=True)
    assert "libcudart" not in elf and "RPATH" not in elf and "RUNPATH" not in elf
    linked_at = (output / "_stride_cuda.so").stat().st_mtime_ns
    assert run(**{**options, "LD_RUN_PATH": "/another/toolkit/lib64"})[0].returncode == 0
    assert (output / "_stride_cuda.so").stat().st_mtime_ns == linked_at
    assert len(list(output.glob("*.o"))) == 8
    source = manifest / "native/cuda/copy.cu"
    original = source.read_text()
    source.write_text(original + 'extern "C" int cudaUnknownFutureCall(); extern "C" int broken() {return cudaUnknownFutureCall();}\n')
    assert run(**options)[0].returncode != 0
    assert (output / "_stride_cuda.fingerprint").read_text() == ""
    source.write_text(original)
    assert run(**options)[0].returncode == 0


@pytest.mark.parametrize("operation", ["update", "dot", "reduction"])
def test_operation_groups_cover_dtypes_and_sdist(operation):
    package = REPO_ROOT / "crates/tensor0-py"
    ffi = package / "native/ffi"
    groups = {
        f"{operation}_a.cc": ["S32", "F16", "C64", "C128", "U64", "S8", "U16", "Pred"],
        f"{operation}_b.cc": ["F32", "BF16", "F64", "S64", "S16", "U8", "U32"],
    }
    pairs = []
    for name, suffixes in groups.items():
        text = (ffi / name).read_text()
        entries = re.findall(rf"^TENSOR0_STRIDE_DEFINE_{operation.upper()}\((\w+), (\w+)\)$",
                             text, re.MULTILINE)
        assert entries == [(suffix, suffix.upper()) for suffix in suffixes]
        assert text.startswith(f'#include "{operation}_impl.h"\n')
        macros = [operation.upper()]
        if operation == "reduction":
            macros.append("ACCUMULATION")
            accumulation = re.findall(r"^TENSOR0_STRIDE_DEFINE_ACCUMULATION\((\w+), (\w+)\)$",
                                      text, re.MULTILINE)
            assert accumulation == entries
        assert text.endswith("".join(f"#undef TENSOR0_STRIDE_DEFINE_{macro}\n" for macro in macros))
        pairs.extend(entries)
    expected = re.findall(r"Define\((\w+), (\w+)\)", (ffi / "dtype.h").read_text())
    assert len(pairs) == len(set(pairs)) == len(expected) == 15
    assert set(pairs) == set(expected)
    sources = re.findall(r'"(native/[^"\n]+\.cc)"', (package / "build.rs").read_text())
    assert len(sources) == len(set(sources)) == 13
    start = sources.index(f"native/ffi/{operation}_a.cc")
    assert sources[start:start + 2] == [f"native/ffi/{operation}_a.cc", f"native/ffi/{operation}_b.cc"]
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    included = {path for entry in config["tool"]["maturin"]["include"]
                if entry["format"] == "sdist" for path in REPO_ROOT.glob(entry["path"])}
    assert {ffi / name for name in (*groups, f"{operation}_impl.h")} <= included


@pytest.mark.parametrize("name", ["CPATH", "CPLUS_INCLUDE_PATH", "COMPILER_PATH", "GCC_EXEC_PREFIX",
                                 "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS"])
def test_companion_cache_tracks_compiler_environment(cuda_build, name):
    run, _, output = cuda_build
    toolkit = output.parent / "cuda"
    (toolkit / "lib").mkdir(parents=True)
    (toolkit / "lib/libcudart.so").touch()
    options = dict(NVCC=str(output.parent / "compiler"), CUDA_HOME=str(toolkit))
    assert run(**options)[0].returncode == 0
    assert run(**options)[1] == []
    result, commands = run(**options, **{name: "changed"})
    assert result.returncode == 0, result.stderr
    assert len(commands) == (5 if name.startswith("NVCC_") else 8)
    assert run(**options, **{name: "changed"})[1] == []


@pytest.mark.parametrize("target,runtime", [("linux", "stdc++"), ("macos", "c++")])
def test_cpu_target_compiler_and_runtime(build, target, runtime):
    run, _, _ = build
    result, commands = run(CARGO_CFG_TARGET_OS=target)
    assert result.returncode == 0, result.stderr
    assert len(commands) == run.unit_count
    assert all("-std=c++20" in command and "-fPIC" in command for command in commands)
    assert f"cargo:rustc-link-lib=dylib={runtime}\n" in result.stdout
    assert "cargo:rustc-cfg=tensor0_stride_ffi" in result.stdout
    assert "cargo:rustc-link-lib=dylib=" + ("stdc++" if target == "macos" else "c++") + "\n" not in result.stdout


def test_unsupported_cpu_target_remains_disabled(build):
    run, _, _ = build
    result, commands = run(CARGO_CFG_TARGET_OS="windows")
    assert result.returncode == 0, result.stderr
    assert commands == []
    assert "disabled on unsupported target windows" in result.stdout
    assert "cargo:rustc-cfg=tensor0_stride_ffi" not in result.stdout
