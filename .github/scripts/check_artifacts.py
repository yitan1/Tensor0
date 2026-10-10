"""Check Tensor0-specific archive contents; run from the source checkout.

Maturin owns package metadata and wheel tags; auditwheel checks binary policy,
while isolated installation and uv pip check validate runtime dependencies.
"""

import argparse
from pathlib import Path
import tarfile
import tomllib
import zipfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('artifact', type=Path)
parser.add_argument('--extract', type=Path, help='Extract an sdist for the wheel build')
args = parser.parse_args()
project = tomllib.loads(Path('pyproject.toml').read_text())['project']
licenses = {str(path) for pattern in project['license-files']
            for path in Path('.').glob(pattern) if path.is_file()}
package_files = {str(path) for path in Path('src/tensor0').rglob('*')
                 if path.is_file() and (path.suffix in {'.py', '.pyi'} or path.name == 'py.typed')}
package_files.update(path for path in licenses if path.startswith('src/tensor0/'))


def check_contents(names, required):
    missing = required - names
    assert not missing, f'Missing archive files: {sorted(missing)}'
    forbidden = {'local', '.venv', 'target', '.git', '__pycache__'}
    leaked = sorted(name for name in names if forbidden.intersection(Path(name).parts))
    assert not leaked, f'Private files or build artifacts in archive: {leaked}'


if args.artifact.name.endswith('.tar.gz'):
    with tarfile.open(args.artifact) as archive:
        roots = {name.split('/')[0] for name in archive.getnames()}
        assert len(roots) == 1, roots
        root = roots.pop()
        names = {name.removeprefix(root + '/') for name in archive.getnames()}
        required = {'pyproject.toml', 'Cargo.toml', 'Cargo.lock', 'README.md'}
        required.update(package_files | licenses)
        required.update(str(path) for path in Path('crates').rglob('*')
                        if path.is_file() and path.suffix in {'.rs', '.toml', '.cc', '.h', '.cu', '.cuh'})
        required.update(str(path) for path in Path('crates/tensor0-py/vendor').rglob('*')
                        if path.is_file())
        check_contents(names, required)
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
        required = {name.removeprefix('src/') for name in package_files}
        required.update(f'{dist_info}/licenses/{name}' for name in licenses)
        check_contents(names, required)
        assert any(name.startswith('tensor0/_native.') and name.endswith('.so') for name in names)
        assert not any(name.startswith(('tests/', 'crates/')) for name in names)
else:
    parser.error('Expected a .tar.gz sdist or .whl wheel')

print(f'Tensor0 archive contents OK: {args.artifact}')
