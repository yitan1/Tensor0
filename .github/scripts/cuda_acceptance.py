"""Manual installed-wheel acceptance; invoke with a pre-provisioned venv's python -I.

No installation/build/network is performed. Output must not exist; report.json,
preflight.log, and suite-N.{log,xml} are retained even on acceptance failure.
Native standalone source contracts are separate and are not executed here.
"""

import argparse
import hashlib
from importlib import metadata
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import traceback
from urllib.parse import unquote, urlparse
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[2]
SUITES = (
    "tests/stride/jax/cuda",
    "tests/stride/native/cuda/test_boundary.py",
    "tests/tensor/test_cuda_integration.py",
)
ALLOWED_SKIPS = {"requires a CPU-only extension", "requires CPU-only extension"}
BUILD_FIELDS = ("source_commit", "wheel", "sha256", "cuda_arch", "cuda_toolkit",
                "python", "rust", "cxx", "workflow_run_url", "cuda_wheel", "cuda_sha256")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def environment():
    env = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME", "PYTEST_ADDOPTS", "PYTEST_PLUGINS"):
        env.pop(key, None)
    env.pop("XLA_FLAGS", None)
    env.update(JAX_PLATFORMS="cuda,cpu", JAX_DISABLE_JIT="0", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    return env


def verify_wheel(wheel, build, direct, *, cuda=False):
    require(all(isinstance(build.get(k), str) and build[k] for k in BUILD_FIELDS),
            "missing/invalid build-info fields")
    require(wheel.suffix == ".whl" and build["cuda_wheel" if cuda else "wheel"] == wheel.name, "wheel basename mismatch")
    with wheel.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    require(digest == build["cuda_sha256" if cuda else "sha256"], "wheel SHA256 mismatch")
    archive = direct.get("archive_info", {})
    hashes = archive.get("hashes", {})
    installed_hash = hashes.get("sha256", archive.get("hash", "").removeprefix("sha256="))
    url = urlparse(direct.get("url", ""))
    require(not direct.get("dir_info") and url.scheme == "file"
            and Path(unquote(url.path)).name == wheel.name and installed_hash == digest,
            "installed direct_url.json does not bind the same wheel")
    return digest


def wheel_binary_hash(wheel, *, cuda=False):
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        binaries = ([name for name in names if name == "tensor0_cuda12/_stride_cuda.so"] if cuda else
                    [name for name in names if re.fullmatch(r"tensor0/_native[^/]*\.so", name)])
        require(len(binaries) == 1, "wheel must contain exactly one native binary")
        return hashlib.sha256(archive.read(binaries[0])).hexdigest()


def preflight(wheel, cuda_wheel, build, report, cpu_report=None):
    require(sys.flags.isolated and sys.prefix != sys.base_prefix,
            "use python -I from a pre-provisioned virtual environment")
    report["prefix"] = sys.prefix
    report["executable"] = sys.executable
    git = lambda *args: subprocess.check_output(
        ["git", "-C", str(ROOT), *args], env=environment(), text=True).strip()
    report.setdefault("checkout", {"root": str(ROOT), "commit": git("rev-parse", "HEAD")})
    status = report.pop("checkout_status", None)
    require(not (git("status", "--porcelain", "--untracked-files=all") if status is None else status),
            "checkout is not clean")
    require(report["checkout"]["commit"] == build["source_commit"], "source commit mismatch")
    direct = json.loads(metadata.distribution("tensor0").read_text("direct_url.json") or "null")
    require(isinstance(direct, dict), "installed wheel lacks direct_url.json")
    report["sha256"] = verify_wheel(wheel, build, direct)
    report["direct_url"] = direct
    if cuda_wheel is not None:
        direct = json.loads(metadata.distribution("tensor0-cuda12").read_text("direct_url.json") or "null")
        require(isinstance(direct, dict), "installed addon lacks direct_url.json")
        report["cuda_sha256"] = verify_wheel(cuda_wheel, build, direct, cuda=True)
        report["cuda_direct_url"] = direct
        require(metadata.version("tensor0-cuda12") == metadata.version("tensor0"),
                "core/addon distribution versions mismatch")
    env = environment()
    if cuda_wheel is None:
        env["JAX_PLATFORMS"] = "cpu"
    os.environ.clear()
    os.environ.update(env)
    import tensor0
    import tensor0._native as native
    import jax
    report["origins"] = {"python": tensor0.__file__, "native": native.__file__}
    if cuda_wheel is not None:
        report["origins"]["companion"] = str(metadata.distribution("tensor0-cuda12").locate_file(
            "tensor0_cuda12/_stride_cuda.so"))
    require(all(Path(p).resolve().is_relative_to(Path(sys.prefix).resolve())
                for p in report["origins"].values()), "tensor0 origins must be inside sys.prefix")
    report["versions"] = {name: metadata.version(name) for name in
                          ("tensor0", "jax", "jaxlib", "numpy", "pytest")}
    report["versions"]["python"] = sys.version
    report["installed_packages"] = {dist.metadata["Name"]: dist.version
                                    for dist in metadata.distributions()}
    report["versions"]["native_jax"] = native._stride_ffi_build_versions()
    report["core_binary_sha256"] = hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest()
    require(report["core_binary_sha256"] == wheel_binary_hash(wheel),
            "installed core binary differs from wheel payload")
    if cuda_wheel is not None:
        report["cuda_binary_sha256"] = hashlib.sha256(Path(report["origins"]["companion"]).read_bytes()).hexdigest()
        require(report["cuda_binary_sha256"] == wheel_binary_hash(cuda_wheel, cuda=True),
                "installed addon binary differs from wheel payload")
    require(tensor0.__version__ == metadata.version("tensor0") == native.native_version(),
            "Python/native/distribution version mismatch")
    require(native._stride_ffi_build_versions() == (jax.__version__, metadata.version("jaxlib")),
            "native/JAX build versions mismatch")
    if cuda_wheel is None:
        require(bool(jax.devices("cpu")) and jax.default_backend() == "cpu", "CPU backend unavailable")
        require(not native._stride_cuda_available(), "CPU baseline must precede addon installation")
        require(not native._stride_cuda_registration(), "CPU baseline registration must be empty")
        return
    require(cpu_report is not None and cpu_report.get("accepted") is True
            and cpu_report.get("mode") == "cpu"
            and cpu_report.get("sha256") == report["sha256"]
            and cpu_report.get("core_binary_sha256") == report["core_binary_sha256"]
            and cpu_report.get("checkout") == report["checkout"]
            and cpu_report.get("prefix") == report["prefix"]
            and cpu_report.get("executable") == report["executable"]
            and cpu_report.get("origins", {}).get("native") == report["origins"]["native"],
            "missing/mismatched CPU baseline, environment or changed core binary")
    require(native._stride_cuda_available(), "native CUDA unavailable")
    registration = native._stride_cuda_registration()
    require(bool(registration), "native CUDA registration is empty")
    report["cuda_registration"] = sorted(registration)
    report["devices"] = {}
    for backend in ("cpu", "cuda"):
        devices = jax.devices(backend)
        require(bool(devices), f"no {backend} devices")
        report["devices"][backend] = [str(d) for d in devices]
    require(jax.default_backend() == "gpu", "JAX default backend is not GPU")
    driver = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,uuid",
                             "--format=csv,noheader"], capture_output=True, text=True,
                            env=environment())
    report["driver"] = {"returncode": driver.returncode, "stdout": driver.stdout,
                        "stderr": driver.stderr}
    require(driver.returncode == 0, "nvidia-smi failed")


def junit_result(path, returncode):
    cases = list(ET.parse(path).getroot().iter("testcase"))
    result = {"returncode": returncode, "passed": 0, "failed": 0, "skipped": []}
    for case in cases:
        skip = case.find("skipped")
        if skip is not None:
            result["skipped"].append({"case": case.get("classname", "") + "::" + case.get("name", ""),
                                      "reason": skip.get("message", ""),
                                      "type": skip.get("type", ""), "detail": skip.text})
        elif case.find("failure") is not None or case.find("error") is not None:
            result["failed"] += 1
        else:
            result["passed"] += 1
    result["accepted"] = (returncode == 0 and result["passed"] > 0 and result["failed"] == 0
                          and all(s["reason"] in ALLOWED_SKIPS and s["type"] == "pytest.skip"
                                  for s in result["skipped"]))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("wheel", "build-info", "output"):
        parser.add_argument("--" + option, type=Path, required=True)
    parser.add_argument("--cuda-wheel", type=Path)
    parser.add_argument("--cpu-report", type=Path)
    parser.add_argument("--cpu-only", action="store_true", help="Record CPU smoke before installing the addon")
    args = parser.parse_args(argv)
    if args.cpu_only:
        if args.cuda_wheel or args.cpu_report:
            parser.error("CPU baseline cannot include an addon or prior CPU report")
    elif not args.cuda_wheel or not args.cpu_report:
        parser.error("GPU acceptance requires --cuda-wheel and --cpu-report")
    args.output = args.output.resolve()
    # Snapshot cleanliness before creating any acceptance artifacts.
    status = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain",
                             "--untracked-files=all"], capture_output=True, text=True,
                            env=environment())
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"accepted": False, "wheel": str(args.wheel.resolve()),
              "build_info_path": str(args.build_info.resolve()), "suites": [],
              "executable": sys.executable, "prefix": sys.prefix,
              "mode": "cpu" if args.cpu_only else "cuda"}
    try:
        require(status.returncode == 0, "cannot inspect checkout status")
        report["checkout_status"] = status.stdout.strip()
        if args.output.is_relative_to(ROOT):
            ignored = subprocess.run(["git", "-C", str(ROOT), "check-ignore", "-q",
                                      str(args.output)], env=environment())
            require(ignored.returncode == 0, "output must be outside checkout or Git-ignored")
        report["build_info"] = json.loads(args.build_info.read_text())
        cpu_report = json.loads(args.cpu_report.read_text()) if args.cpu_report else None
        preflight(args.wheel.resolve(), args.cuda_wheel.resolve() if args.cuda_wheel else None,
                  report["build_info"], report, cpu_report)
        if args.cuda_wheel:
            report["cuda_wheel"] = str(args.cuda_wheel.resolve())
        (args.output / "preflight.log").write_text("Preflight passed\n")
        if args.cpu_only:
            command = [sys.executable, "-I", str(ROOT / ".github/scripts/wheel_smoke.py")]
            with (args.output / "cpu-smoke.log").open("w") as log:
                result = subprocess.run(command, env=environment() | {"JAX_PLATFORMS": "cpu"},
                                        cwd=args.output, stdout=log, stderr=subprocess.STDOUT)
            require(result.returncode == 0, "CPU smoke failed")
            report["accepted"] = True
            return 0
        for index, suite in enumerate(SUITES):
            xml = args.output / f"suite-{index}.xml"
            command = [sys.executable, "-I", "-m", "pytest", suite, "-ra",
                       "-o", "addopts=", "-o", "xfail_strict=true", f"--junitxml={xml}"]
            entry = {"suite": suite, "command": command}
            report["suites"].append(entry)
            with (args.output / f"suite-{index}.log").open("w") as log:
                process = subprocess.run(command, cwd=ROOT, env=environment(), stdout=log,
                                         stderr=subprocess.STDOUT)
            entry["returncode"] = process.returncode
            try:
                entry.update(junit_result(xml, process.returncode))
            except (OSError, ET.ParseError) as error:
                entry.update(accepted=False, error=str(error))
        report["accepted"] = all(s["accepted"] for s in report["suites"])
    except Exception:
        report["error"] = traceback.format_exc()
        (args.output / "preflight.log").write_text(report["error"])
    finally:
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"CUDA acceptance {'passed' if report['accepted'] else 'FAILED'}: {args.output / 'report.json'}")
    return 0 if report["accepted"] else 1


if __name__ == "__main__":
    sys.exit(main())
