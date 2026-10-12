"""Verify the CPU producer and exact distributions reused by CUDA candidates."""

import argparse
import hashlib
import json
import re
import tomllib
from pathlib import Path

ABIS = ('cp311', 'cp312', 'cp313', 'cp314')
CORE_PLATFORMS = ('manylinux_2_28_x86_64', 'macosx_15_0_arm64')


def core_wheel_pattern(version, abi, platform):
    return rf'tensor0-{re.escape(version)}-{abi}-{abi}-{platform}\.whl'


def verify_run(run, workflow, repository, sha, run_id):
    if not re.fullmatch(r'[1-9][0-9]*', run_id):
        raise ValueError('Candidate run ID must be a positive integer')
    if not (run['id'] == int(run_id)
            and run['workflow_id'] == workflow['id']
            and workflow['path'] == '.github/workflows/artifacts.yml'
            and run['path'] == workflow['path']
            and run['repository']['full_name'] == repository
            and run['head_repository']['full_name'] == repository
            and run['event'] == 'workflow_dispatch'
            and run['head_branch'] == 'main'
            and run['status'] == 'completed' and run['conclusion'] == 'success'
            and run['head_sha'] == sha):
        raise ValueError('Require successful manual artifacts.yml on own main at this exact commit')


def verify_files(root, version):
    files = {path.name for path in root.iterdir()}
    packages = {f'tensor0-{version}.tar.gz'}
    wheels = {}
    for abi in ABIS:
        for platform in CORE_PLATFORMS:
            matches = {name for name in files
                       if re.fullmatch(core_wheel_pattern(version, abi, platform), name)}
            if len(matches) != 1:
                raise ValueError(f'Expected one {abi} {platform} wheel')
            name = matches.pop()
            packages.add(name)
            # CUDA acceptance and ABI smoke deliberately select Linux, not macOS.
            if platform == 'manylinux_2_28_x86_64':
                wheels[abi] = name
    if files != packages | {'SHA256SUMS'} or any(not p.is_file() or p.is_symlink() for p in root.iterdir()):
        raise ValueError('Expected only core sdist, eight wheels and SHA256SUMS')
    entries = [line.split() for line in (root / 'SHA256SUMS').read_text().splitlines()]
    if (len(entries) != 9 or any(len(entry) != 2 for entry in entries)
            or {entry[1] for entry in entries} != packages):
        raise ValueError('Expected nine unique basename checksums')
    for digest, name in entries:
        if not re.fullmatch('[0-9a-f]{64}', digest) or hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Checksum mismatch: {name}')
    return wheels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('run')
    for name in ('run-json', 'workflow-json', 'repository', 'sha', 'run-id'):
        run.add_argument('--' + name, required=True)
    files = sub.add_parser('files')
    files.add_argument('directory', type=Path)
    files.add_argument('--project', type=Path, default=Path('pyproject.toml'))
    args = parser.parse_args()
    if args.command == 'run':
        verify_run(json.loads(Path(args.run_json).read_text()),
                   json.loads(Path(args.workflow_json).read_text()),
                   args.repository, args.sha, args.run_id)
    else:
        verify_files(args.directory, tomllib.loads(args.project.read_text())['project']['version'])


if __name__ == '__main__':
    main()
