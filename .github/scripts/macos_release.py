"""Native macOS release preflight and exact installed-wheel checks (CPU only)."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from urllib.parse import unquote, urlparse
import zipfile


def host():
    assert sys.platform == 'darwin' and platform.machine() == 'arm64'
    assert platform.mac_ver()[0].split('.')[0] == '15', 'Release validation requires macOS 15'
    assert 'host: aarch64-apple-darwin' in subprocess.check_output(['rustc', '-vV'], text=True)
    import jax
    import jaxlib
    assert jax.__version__ == jaxlib.__version__ == '0.10.1'
    installed = Path(jaxlib.__file__).parent / 'include/xla/ffi/api'
    vendor = Path('crates/tensor0-py/vendor/jaxlib-0.10.1/include/xla/ffi/api')
    for name in ('api.h', 'c_api.h', 'ffi.h'):
        assert (installed / name).read_bytes() == (vendor / name).read_bytes(), name


def inspect(wheel):
    abi = f'cp{sys.version_info.major}{sys.version_info.minor}'
    assert wheel.name.endswith(f'-{abi}-{abi}-macosx_15_0_arm64.whl'), wheel.name
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        assert not any('_stride_cuda' in name or 'tensor0_cuda12' in name for name in names)
        native, = (name for name in names if name.startswith('tensor0/_native') and name.endswith('.so'))
        binary = archive.read(native)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'native.so'
        path.write_bytes(binary)
        assert subprocess.check_output(['lipo', '-archs', str(path)], text=True).strip() == 'arm64'
        subprocess.run(['file', str(path)], check=True)
        subprocess.run(['otool', '-L', str(path)], check=True)
    print(wheel.name, hashlib.sha256(wheel.read_bytes()).hexdigest())
    return binary


def installed(wheel):
    import jax
    import jaxlib
    import tensor0
    assert sys.flags.isolated and sys.prefix != sys.base_prefix
    version = wheel.name.split('-')[1]
    assert tensor0.__version__ == tensor0._native.native_version() == importlib.metadata.version('tensor0') == version
    prefix = Path(sys.prefix).resolve()
    assert Path(tensor0.__file__).resolve().is_relative_to(prefix)
    native = Path(tensor0._native.__file__).resolve()
    assert native.is_relative_to(prefix)
    assert native.read_bytes() == inspect(wheel), 'Installed native binary differs from wheel'
    assert tensor0._native._stride_ffi_available()
    assert tensor0._native._stride_ffi_build_versions() == ('0.10.1', '0.10.1')
    assert jax.__version__ == jaxlib.__version__ == '0.10.1'
    assert not tensor0._native._stride_cuda_available()
    assert all(device.platform == 'cpu' for device in jax.devices())
    direct = json.loads(importlib.metadata.distribution('tensor0').read_text('direct_url.json'))
    assert not direct.get('dir_info')
    url = urlparse(direct['url'])
    assert url.scheme == 'file' and Path(unquote(url.path)).name == wheel.name
    assert direct['archive_info']['hashes']['sha256'] == hashlib.sha256(wheel.read_bytes()).hexdigest()
    print(tensor0.__file__, native, direct)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('host', 'inspect', 'installed'))
    parser.add_argument('wheel', nargs='?', type=Path)
    args = parser.parse_args()
    if args.command == 'host':
        host()
    else:
        if args.wheel is None:
            parser.error('wheel is required')
        (inspect if args.command == 'inspect' else installed)(args.wheel)
