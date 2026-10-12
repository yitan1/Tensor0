"""Offline checks for the fixed native macOS release contract; no native build."""

import importlib.util
from pathlib import Path
import sys
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[2]
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
