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
from pathlib import Path
import subprocess
import sys
import traceback
from urllib.parse import unquote, urlparse
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
SUITES = (
    "tests/stride/jax/cuda",
    "tests/stride/native/cuda/test_boundary.py",
    "tests/tensor/test_cuda_integration.py",
)
ALLOWED_SKIPS = {"requires a CPU-only extension", "requires CPU-only extension"}
BUILD_FIELDS = ("source_commit", "wheel", "sha256", "cuda_arch", "cuda_toolkit",
                "python", "rust", "cxx", "workflow_run_url")


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


def verify_wheel(wheel, build, direct):
    require(all(isinstance(build.get(k), str) and build[k] for k in BUILD_FIELDS),
            "missing/invalid build-info fields")
    require(wheel.suffix == ".whl" and build["wheel"] == wheel.name, "wheel basename mismatch")
    with wheel.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    require(digest == build["sha256"], "wheel SHA256 mismatch")
    archive = direct.get("archive_info", {})
    hashes = archive.get("hashes", {})
    installed_hash = hashes.get("sha256", archive.get("hash", "").removeprefix("sha256="))
    url = urlparse(direct.get("url", ""))
    require(not direct.get("dir_info") and url.scheme == "file"
            and Path(unquote(url.path)).name == wheel.name and installed_hash == digest,
            "installed direct_url.json does not bind the same wheel")
    return digest


def preflight(wheel, build, report):
    require(sys.flags.isolated and sys.prefix != sys.base_prefix,
            "use python -I from a pre-provisioned virtual environment")
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
    env = environment()
    os.environ.clear()
    os.environ.update(env)
    import tensor0
    import tensor0._native as native
    import jax
    report["origins"] = {"python": tensor0.__file__, "native": native.__file__}
    require(all(Path(p).resolve().is_relative_to(Path(sys.prefix).resolve())
                for p in report["origins"].values()), "tensor0 origins must be inside sys.prefix")
    report["versions"] = {name: metadata.version(name) for name in
                          ("tensor0", "jax", "jaxlib", "numpy", "pytest")}
    report["versions"]["python"] = sys.version
    report["installed_packages"] = {dist.metadata["Name"]: dist.version
                                    for dist in metadata.distributions()}
    report["versions"]["native_jax"] = native._stride_ffi_build_versions()
    require(native._stride_cuda_available(), "native CUDA unavailable")
    require(native._stride_ffi_build_versions() == (jax.__version__, metadata.version("jaxlib")),
            "native/JAX build versions mismatch")
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
    args = parser.parse_args(argv)
    args.output = args.output.resolve()
    # Snapshot cleanliness before creating any acceptance artifacts.
    status = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain",
                             "--untracked-files=all"], capture_output=True, text=True,
                            env=environment())
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"accepted": False, "wheel": str(args.wheel.resolve()),
              "build_info_path": str(args.build_info.resolve()), "suites": [],
              "executable": sys.executable, "prefix": sys.prefix}
    try:
        require(status.returncode == 0, "cannot inspect checkout status")
        report["checkout_status"] = status.stdout.strip()
        if args.output.is_relative_to(ROOT):
            ignored = subprocess.run(["git", "-C", str(ROOT), "check-ignore", "-q",
                                      str(args.output)], env=environment())
            require(ignored.returncode == 0, "output must be outside checkout or Git-ignored")
        report["build_info"] = json.loads(args.build_info.read_text())
        preflight(args.wheel.resolve(), report["build_info"], report)
        (args.output / "preflight.log").write_text("Preflight passed\n")
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
