"""Check Tensor0-specific archive contents; run from the source checkout.

Maturin owns package metadata and wheel tags; auditwheel checks binary policy,
while isolated installation and uv pip check validate runtime dependencies.
"""

import argparse
import re
import sys
from email.parser import Parser
from pathlib import Path
import tarfile
import tomllib
import zipfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('artifact', type=Path)
parser.add_argument('--extract', type=Path, help='Extract an sdist for the wheel build')
parser.add_argument('--release-core', action='store_true', help='Require the fixed core release ABI/platform tags')
parser.add_argument('--cuda', action='store_true', help='Check a standalone tensor0-cuda12 artifact')
args = parser.parse_args()
project = tomllib.loads(Path('pyproject.toml').read_text())['project']
license_project = (tomllib.loads(Path('crates/tensor0-cuda12/pyproject.toml').read_text())['project']
                   if args.cuda else project)
licenses = set(license_project['license-files'])
package_files = {str(path) for path in Path('src/tensor0').rglob('*')
                 if path.is_file() and (path.suffix in {'.py', '.pyi'} or path.name == 'py.typed')}
package_files.update(path for path in licenses if path.startswith('src/tensor0/'))


def check_contents(names, required):
    missing = required - names
    assert not missing, f'Missing archive files: {sorted(missing)}'
    forbidden = {'local', '.venv', 'target', '.git', '__pycache__', '_licenses'}
    leaked = sorted(name for name in names if forbidden.intersection(Path(name).parts))
    assert not leaked, f'Private files or build artifacts in archive: {leaked}'


if args.artifact.name.endswith('.tar.gz'):
    with tarfile.open(args.artifact) as archive:
        roots = {name.split('/')[0] for name in archive.getnames()}
        assert len(roots) == 1, roots
        root = roots.pop()
        names = {name.removeprefix(root + '/') for name in archive.getnames()}
        if args.cuda:
            required = {'pyproject.toml', 'setup.py', 'builder.rs', 'build_helpers.rs', 'LICENSE',
                        'tensor0_cuda12/__init__.py', 'native/cudart_forwarder.cc', 'native/cudart_forwarder.map'}
            required.update('native/' + str(path.relative_to('crates/tensor0-cuda12/native'))
                            for path in Path('crates/tensor0-cuda12/native/cuda').rglob('*') if path.is_file())
            required.update('native/layout/' + name for name in
                            ('descriptor.cc', 'descriptor.h', 'record.cc', 'record.h', 'owner_fiber.h'))
            required.update({'native/ffi/errors.h', 'native/ffi/buffers.h'})
            required.update('vendor/' + str(path.relative_to('crates/tensor0-py/vendor'))
                            for path in Path('crates/tensor0-py/vendor').rglob('*') if path.is_file())
            required.update(licenses)
        else:
            required = {'pyproject.toml', 'Cargo.toml', 'Cargo.lock', 'README.md'}
            required.update(package_files | licenses)
            required.update(str(path) for path in Path('crates').rglob('*')
                            if path.is_file() and path.suffix in {'.rs', '.toml', '.cc', '.h'}
                            and 'tensor0-cuda12' not in path.parts and 'cuda' not in path.parts)
            required.update(str(path) for path in Path('crates/tensor0-py/vendor').rglob('*')
                            if path.is_file())
            assert not any('cuda' in Path(name).parts or 'tensor0-cuda12' in Path(name).parts for name in names), \
                'CUDA native sources leaked into core sdist'
        check_contents(names, required)
        assert not any(Path(name).name == '_stride_cuda.so' for name in names), 'Generated CUDA library leaked into sdist'
        if args.extract is not None:
            archive.extractall(args.extract, filter='data')
            (args.extract / 'source').symlink_to(root, target_is_directory=True)
elif args.artifact.suffix == '.whl':
    if args.extract is not None:
        parser.error('--extract is only supported for an sdist')
    with zipfile.ZipFile(args.artifact) as archive:
        names = set(archive.namelist())
        metadata, = (name for name in names if name.endswith('.dist-info/METADATA'))
        dist_info = str(Path(metadata).parent)
        message = Parser().parsestr(archive.read(metadata).decode())
        assert message['License-Expression'] == license_project['license'], 'Wrong distribution license expression'
        assert set(message.get_all('License-File', [])) == licenses, 'Wrong license-file metadata'
        if args.cuda:
            assert message['Name'] == 'tensor0-cuda12', 'Wrong addon distribution name'
            assert message['Version'] == project['version'], 'Wrong addon distribution version'
            dependencies = message.get_all('Requires-Dist', [])
            assert f"tensor0=={project['version']}" in dependencies, 'Addon must pin the matching core'
            assert any(dep.startswith('nvidia-cuda-runtime-cu12') and '<13' in dep and '>=12.9.79' in dep
                       for dep in dependencies), 'Addon must require the supported external runtime'
            required = {'tensor0_cuda12/__init__.py', 'tensor0_cuda12/_stride_cuda.so'}
            required.update(f'{dist_info}/licenses/{name}' for name in licenses)
            check_contents(names, required)
            assert not any(name.startswith(('tensor0/', 'tests/', 'crates/', 'native/', 'vendor/')) for name in names)
            assert args.artifact.stem.endswith('-py3-none-manylinux_2_28_x86_64'), 'Invalid companion wheel tag'
            tags = [line[5:].strip() for line in archive.read(f'{dist_info}/WHEEL').decode().splitlines()
                    if line.startswith('Tag: ')]
            assert tags == ['py3-none-manylinux_2_28_x86_64'], 'Invalid companion WHEEL metadata tag'
            assert 'Root-Is-Purelib: false' in archive.read(f'{dist_info}/WHEEL').decode()
        else:
            assert message['Name'] == 'tensor0' and message['Version'] == project['version'], 'Wrong core name/version'
            if args.release_core:
                sys.path.insert(0, str(Path(__file__).resolve().parent))
                from cuda_candidate import ABIS, CORE_PLATFORMS, core_wheel_pattern
                assert any(re.fullmatch(core_wheel_pattern(project['version'], abi, platform), args.artifact.name)
                           for abi in ABIS for platform in CORE_PLATFORMS), 'Invalid core release wheel tag'
                expected = '-'.join(args.artifact.stem.split('-')[2:])
                wheel_info = Parser().parsestr(archive.read(f'{dist_info}/WHEEL').decode())
                assert wheel_info.get_all('Tag', []) == [expected], 'Invalid core WHEEL metadata tag'
                assert wheel_info['Root-Is-Purelib'] == 'false'
            required = {name.removeprefix('src/') for name in package_files}
            required.update(f'{dist_info}/licenses/{name}' for name in licenses)
            check_contents(names, required)
            assert any(name.startswith('tensor0/_native.') and name.endswith('.so') for name in names)
            assert not any(name.startswith(('tests/', 'crates/', 'tensor0_cuda12/')) for name in names)
            assert 'tensor0/_stride_cuda.so' not in names, 'CUDA library leaked into core wheel'
        assert not any(Path(name).name.startswith(('libcu', 'libnv', 'libnccl', 'libnvidia')) for name in names), \
            'NVIDIA runtime/driver binaries must be supplied separately'
else:
    parser.error('Expected a .tar.gz sdist or .whl wheel')

print(f'Tensor0 archive contents OK: {args.artifact}')
