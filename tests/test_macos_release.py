"""Offline checks for the fixed native macOS release contract; no native build."""

import importlib.util
from pathlib import Path
import sys
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / '.github/scripts/macos_release.py'
spec = importlib.util.spec_from_file_location('macos_release', SCRIPT)
assert spec and spec.loader
macos = importlib.util.module_from_spec(spec)
spec.loader.exec_module(macos)


@pytest.mark.parametrize('platform,arch,accepted', [
    ('macosx_15_0_arm64', 'arm64', True),
    ('macosx_14_0_arm64', 'arm64', False),
    ('macosx_16_0_arm64', 'arm64', False),
    ('macosx_15_0_universal2', 'arm64 x86_64', False),
    ('macosx_15_0_x86_64', 'x86_64', False),
    ('macosx_15_0_arm64', 'x86_64', False),
    ('macosx_15_0_arm64', 'arm64 x86_64', False),
])
def test_native_wheel_inspection(tmp_path, monkeypatch, platform, arch, accepted):
    abi = f'cp{sys.version_info.major}{sys.version_info.minor}'
    wheel = tmp_path / f'tensor0-0.2.0-{abi}-{abi}-{platform}.whl'
    with zipfile.ZipFile(wheel, 'w') as archive:
        archive.writestr('tensor0/_native.cpython-darwin.so', b'mach-o fixture')
    monkeypatch.setattr(macos.subprocess, 'check_output', lambda *a, **kw: arch + '\n')
    monkeypatch.setattr(macos.subprocess, 'run', lambda *a, **kw: None)
    if accepted:
        assert macos.inspect(wheel) == b'mach-o fixture'
    else:
        with pytest.raises(AssertionError):
            macos.inspect(wheel)


def test_cuda_payload_rejected(tmp_path):
    abi = f'cp{sys.version_info.major}{sys.version_info.minor}'
    wheel = tmp_path / f'tensor0-0.2.0-{abi}-{abi}-macosx_15_0_arm64.whl'
    with zipfile.ZipFile(wheel, 'w') as archive:
        archive.writestr('tensor0/_native.darwin.so', b'fixture')
        archive.writestr('tensor0/_stride_cuda.so', b'fixture')
    with pytest.raises(AssertionError):
        macos.inspect(wheel)


@pytest.mark.parametrize('system,machine,version', [
    ('linux', 'arm64', '15.0'), ('darwin', 'x86_64', '15.0'),
    ('darwin', 'arm64', '14.0'), ('darwin', 'arm64', '27.0'),
])
def test_wrong_host_rejected(monkeypatch, system, machine, version):
    monkeypatch.setattr(macos.sys, 'platform', system)
    monkeypatch.setattr(macos.platform, 'machine', lambda: machine)
    monkeypatch.setattr(macos.platform, 'mac_ver', lambda: (version, (), machine))
    with pytest.raises(AssertionError):
        macos.host()


def test_shared_source_and_full_macos_matrix_gate():
    workflow = (ROOT / '.github/workflows/artifacts.yml').read_text()
    mac_job = workflow.split('  macos-wheel:\n')[1].split('  assemble:\n')[0]
    assert 'needs: sdist' in mac_job and 'runs-on: macos-15' in mac_job
    assert 'python: ["3.11", "3.12", "3.13", "3.14"]' in mac_job
    assert 'name: tensor0-sdist-${{ github.sha }}' in mac_job
    assert 'enable-cache: false' in mac_job and 'CARGO_TARGET_DIR=$work/target' in mac_job
    assert 'CXX: /usr/bin/clang++' in mac_job
    assert 'MACOSX_DEPLOYMENT_TARGET: "15.0"' in mac_job
    assert '--no-install-project' in mac_job and 'maturin" build --release --locked' in mac_job
    assert 'maturin" sdist' not in mac_job and 'cuda12' not in mac_job
    assert '-I -m pip install --no-deps dist/*.whl' in mac_job
    assert '-I -m pytest tests -q -ra' in mac_job
    assert 'macos_release.py installed dist/*.whl' in mac_job
    assert 'needs: [sdist, wheel, macos-wheel]' in workflow
    assert 'cuda_candidate.py files dist' in workflow


def test_cuda_never_selects_or_audits_macos_wheel():
    workflow = (ROOT / '.github/workflows/cuda-build.yml').read_text()
    assert 'cp311-cp311-*.whl' not in workflow
    assert '${abi}-${abi}-*.whl' not in workflow
    assert "directory.glob('tensor0-*-cp311-cp311-*linux*_x86_64.whl')" in workflow
    assert 'for wheel in "$RUNNER_TEMP"/cuda-dist/*_x86_64.whl; do auditwheel show' in workflow
    assert './tensor0-*-${abi}-${abi}-manylinux_2_28_x86_64.whl' in workflow


@pytest.mark.parametrize('problem', [None, 'binary', 'origin', 'url', 'hash', 'version', 'ffi', 'cuda'])
def test_installed_original_wheel(tmp_path, monkeypatch, problem):
    import hashlib
    import json
    from types import SimpleNamespace
    prefix = tmp_path / 'venv'
    prefix.mkdir()
    native_path = prefix / '_native.darwin.so'
    native_path.write_bytes(b'changed' if problem == 'binary' else b'mach-o fixture')
    wheel = tmp_path / 'tensor0-0.2.0-cp311-cp311-macosx_15_0_arm64.whl'
    wheel.write_bytes(b'wheel fixture')
    native = SimpleNamespace(__file__=str(native_path), native_version=lambda: '0.2.0',
                             _stride_ffi_available=lambda: problem != 'ffi',
                             _stride_ffi_build_versions=lambda: ('0.10.1', '0.10.1'),
                             _stride_cuda_available=lambda: problem == 'cuda')
    tensor = SimpleNamespace(__file__=str(tmp_path / 'source.py' if problem == 'origin' else prefix / '__init__.py'),
                             __version__='0.1.0' if problem == 'version' else '0.2.0', _native=native)
    direct = {'url': ('https://example.invalid/' if problem == 'url' else 'file:///dist/') + wheel.name,
              'archive_info': {'hashes': {'sha256': '0' * 64 if problem == 'hash' else
                                         hashlib.sha256(wheel.read_bytes()).hexdigest()}}}
    monkeypatch.setitem(sys.modules, 'tensor0', tensor)
    monkeypatch.setitem(sys.modules, 'jax', SimpleNamespace(__version__='0.10.1',
                        devices=lambda: [SimpleNamespace(platform='cpu')]))
    monkeypatch.setitem(sys.modules, 'jaxlib', SimpleNamespace(__version__='0.10.1'))
    monkeypatch.setattr(macos.sys, 'prefix', str(prefix))
    monkeypatch.setattr(macos.sys, 'flags', SimpleNamespace(isolated=1))
    monkeypatch.setattr(macos.importlib.metadata, 'version', lambda name: '0.2.0')
    monkeypatch.setattr(macos.importlib.metadata, 'distribution',
                        lambda name: SimpleNamespace(read_text=lambda name: json.dumps(direct)))
    monkeypatch.setattr(macos, 'inspect', lambda path: b'mach-o fixture')
    if problem:
        with pytest.raises(AssertionError):
            macos.installed(wheel)
    else:
        macos.installed(wheel)
