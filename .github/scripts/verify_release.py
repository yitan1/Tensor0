"""Verify an exact dual-package release; never build, install or execute evidence."""

import argparse
from email.parser import Parser
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tarfile
import tomllib
from urllib.parse import unquote, urlparse
import zipfile
import xml.etree.ElementTree as ET

from cuda_candidate import ABIS, CORE_PLATFORMS, verify_files, verify_run
from cuda_acceptance import BUILD_FIELDS, SUITES, junit_result


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def api(path, *, binary=False):
    command = ['gh', 'api', path]
    if binary:
        command += ['-H', 'Accept: application/octet-stream']
    data = subprocess.check_output(command)
    return data if binary else json.loads(data)


def jobs(repository, run_id):
    result = []
    page = 1
    while True:
        batch = api(f'repos/{repository}/actions/runs/{run_id}/jobs?per_page=100&page={page}')['jobs']
        result.extend(batch)
        if len(batch) < 100:
            return result
        page += 1


def require_jobs(records, names):
    for name in names:
        matches = [job for job in records if job['name'] == name]
        require(len(matches) == 1 and matches[0]['conclusion'] == 'success',
                f'Missing successful job: {name}')


def git(*args):
    return subprocess.check_output(['git', *args], text=True).strip()


def verify_provenance(repository, cpu_id, cuda_id, tag):
    require(re.fullmatch(r'v[0-9]+\.[0-9]+\.[0-9]+', tag) and tag != 'v0.0.0',
            'Require a chosen stable vX.Y.Z release tag; development versions remain blocked')
    for run_id in (cpu_id, cuda_id):
        require(re.fullmatch(r'[1-9][0-9]*', run_id), 'Run ID must be a positive integer')
    sha = git('rev-parse', f'refs/tags/{tag}^{{commit}}')
    subprocess.run(['git', 'merge-base', '--is-ancestor', sha, 'origin/main'], check=True)
    version = tag[1:]
    sources = [('pyproject.toml', ('project', 'version')),
               ('Cargo.toml', ('workspace', 'package', 'version')),
               ('crates/tensor0-cuda12/pyproject.toml', ('project', 'version'))]
    for path, keys in sources:
        value = tomllib.loads(git('show', f'{sha}:{path}'))
        for key in keys:
            value = value[key]
        require(value == version, f'{path} version must literally match stable tag')
    cpu = api(f'repos/{repository}/actions/runs/{cpu_id}')
    verify_run(cpu, api(f'repos/{repository}/actions/workflows/artifacts.yml'), repository, sha, cpu_id)
    require_jobs(jobs(repository, cpu_id),
                 {'Build source distribution once', 'Assemble tested distributions (no rebuild)'}
                 | {f'CPython {v} / manylinux_2_28 x86_64' for v in ('3.11', '3.12', '3.13', '3.14')}
                 | {f'CPython {v} / macOS 15 ARM64 CPU' for v in ('3.11', '3.12', '3.13', '3.14')})
    cuda = api(f'repos/{repository}/actions/runs/{cuda_id}')
    workflow = api(f'repos/{repository}/actions/workflows/cuda-build.yml')
    require(cuda['id'] == int(cuda_id) and cuda['workflow_id'] == workflow['id']
            and cuda['path'] == workflow['path'] == '.github/workflows/cuda-build.yml'
            and cuda['repository']['full_name'] == cuda['head_repository']['full_name'] == repository
            and cuda['event'] == 'workflow_dispatch' and cuda['head_branch'] == 'main'
            and cuda['status'] == 'completed' and cuda['conclusion'] == 'success'
            and cuda['head_sha'] == sha, 'Require same-commit successful manual CUDA candidate on own main')
    require_jobs(jobs(repository, cuda_id),
                 {'verify-candidate', 'CUDA 12.9 / sm_80 / CPython 3.11 (compile and CPU smoke only)'}
                 | {f'Exact CPU candidate + same addon / Python {v} (no GPU)'
                    for v in ('3.11', '3.12', '3.13', '3.14')})
    required = {f'Linux x86_64 / Python {v}' for v in ('3.11', '3.14')}
    required.add('Public-package Pyright (Python 3.11)')
    workflow = api(f'repos/{repository}/actions/workflows/ci.yml')
    checks = api(f'repos/{repository}/actions/workflows/ci.yml/runs?branch=main&status=success&head_sha={sha}&per_page=100')
    for check in checks['workflow_runs']:
        if (check['head_sha'] == sha and check['repository']['full_name'] == repository
                and check['head_repository']['full_name'] == repository
                and check['workflow_id'] == workflow['id']
                and check['path'] == workflow['path'] == '.github/workflows/ci.yml'
                and check['event'] in ('push', 'workflow_dispatch') and check['head_branch'] == 'main'
                and check['status'] == 'completed' and check['conclusion'] == 'success'):
            try:
                require_jobs(jobs(repository, check['id']), required)
                break
            except ValueError:
                continue
    else:
        raise ValueError('No exact-commit CPU matrix and blocking public Pyright CI')
    return sha


def safe_names(names):
    require(len(names) == len(set(names)), 'Duplicate archive members')
    for name in names:
        path = PurePosixPath(name)
        require(name and not path.is_absolute() and '\\' not in name
                and all(p not in ('', '.', '..') for p in name.rstrip('/').split('/')),
                f'Unsafe archive member: {name}')


def metadata(path, version):
    addon = path.name.startswith('tensor0_cuda12-')
    name = 'tensor0-cuda12' if addon else 'tensor0'
    if path.suffix == '.whl':
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            safe_names(names)
            selected = [n for n in names if n.endswith('.dist-info/METADATA')]
            require(len(selected) == 1, 'Expected one wheel METADATA')
            info = Parser().parsestr(archive.read(selected[0]).decode())
            wheel = Parser().parsestr(archive.read(selected[0].replace('METADATA', 'WHEEL')).decode())
            tags = wheel.get_all('Tag', [])
            expected = path.name[:-4].split('-')[2:]
            require(set(tags) == {f'{expected[0]}-{expected[1]}-{plat}' for plat in expected[2].split('.')}
                    and wheel['Root-Is-Purelib'] == 'false', 'Wheel metadata tags mismatch')
    else:
        with tarfile.open(path) as archive:
            names = archive.getnames()
            safe_names(names)
            require(len({n.split('/')[0] for n in names}) == 1, 'Expected one sdist root')
            selected = [m for m in archive.getmembers() if m.name.count('/') == 1
                        and m.name.endswith('/PKG-INFO') and m.isfile()]
            require(len(selected) == 1, 'Expected one root sdist PKG-INFO')
            info = Parser().parsestr(archive.extractfile(selected[0]).read().decode())
    require(info['Name'] == name and info['Version'] == version, 'Distribution metadata name/version mismatch')
    require(info['License-Expression'] == 'MIT AND Apache-2.0', 'Distribution license mismatch')
    require(set(info.get_all('License-File', [])) == {'LICENSE', ('vendor/jaxlib-0.10.1/LICENSE.txt' if addon else
                            'crates/tensor0-py/vendor/jaxlib-0.10.1/LICENSE.txt')},
            'Distribution license files mismatch')
    deps = [re.sub(r'\s+', '', d) for d in info.get_all('Requires-Dist', [])]
    require(not any('@' in d for d in deps), 'Direct URL dependencies forbidden')
    if addon:
        require(f'tensor0=={version}' in deps, 'Addon must exactly pin core')
        require(any(d in ('nvidia-cuda-runtime-cu12>=12.9.79,<13',
                                       'nvidia-cuda-runtime-cu12<13,>=12.9.79') for d in deps),
                'Addon runtime bound mismatch')


def verify_distributions(cpu, cuda, version, sha, cpu_id, cuda_id, repository):
    wheels = verify_files(cpu, version)
    core = {p.name for p in cpu.iterdir()} - {'SHA256SUMS'}
    addon = {f'tensor0_cuda12-{version}-py3-none-manylinux_2_28_x86_64.whl',
             f'tensor0_cuda12-{version}.tar.gz'}
    packages = core | addon
    require({p.name for p in cuda.iterdir()} == packages | {'SHA256SUMS', 'build-info.json'}
            and all(p.is_file() and not p.is_symlink() for p in cuda.iterdir()),
            'CUDA candidate must contain exactly eleven distributions, manifest and build-info')
    entries = [line.split() for line in (cuda / 'SHA256SUMS').read_text().splitlines()]
    require(len(entries) == 11 and all(len(e) == 2 for e in entries)
            and {e[1] for e in entries} == packages, 'Expected eleven unique basename checksums')
    for digest, name in entries:
        require(re.fullmatch(r'[0-9a-f]{64}', digest) and sha256(cuda / name) == digest,
                f'Checksum mismatch: {name}')
    for name in core:
        require((cpu / name).read_bytes() == (cuda / name).read_bytes(), f'Duplicate core differs: {name}')
    for name in packages:
        metadata(cuda / name, version)
    build = json.loads((cuda / 'build-info.json').read_text())
    require(all(isinstance(build.get(k), str) and build[k] for k in BUILD_FIELDS), 'Incomplete build-info')
    require(build.get('release_candidate') is True and build['source_commit'] == sha
            and build.get('cpu_candidate_run_id') == cpu_id
            and build.get('cpu_candidate_run_url') == f'https://github.com/{repository}/actions/runs/{cpu_id}'
            and build['workflow_run_url'] == f'https://github.com/{repository}/actions/runs/{cuda_id}',
            'Build-info is not this paired release candidate')
    require(build['wheel'] == wheels['cp311'] and build['sha256'] == sha256(cpu / wheels['cp311'])
            and build['cuda_wheel'] == f'tensor0_cuda12-{version}-py3-none-manylinux_2_28_x86_64.whl'
            and build['cuda_sha256'] == sha256(cuda / build['cuda_wheel'])
            and build.get('sdists') == {n: sha256(cuda / n) for n in packages if n.endswith('.tar.gz')},
            'Build-info distribution hashes mismatch')
    return packages, build


def evidence_asset(repository, tag, asset_id, digest, destination):
    require(re.fullmatch(r'[1-9][0-9]*', asset_id), 'Asset ID must be a positive integer')
    require(re.fullmatch(r'[0-9a-f]{64}', digest), 'Evidence SHA256 must be lowercase hex')
    release = api(f'repos/{repository}/releases/tags/{tag}')
    require(release['tag_name'] == tag, 'Evidence release tag mismatch')
    # Draft is preferred; a published release with the same immutable tag is also valid.
    assets = []
    page = 1
    while True:
        batch = api(f'repos/{repository}/releases/{release["id"]}/assets?per_page=100&page={page}')
        assets.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    matches = [a for a in assets if a['id'] == int(asset_id)]
    require(len(matches) == 1 and matches[0]['name'] == 'cuda-acceptance.tar.gz'
            and matches[0]['state'] == 'uploaded', 'Evidence asset must belong to the selected release')
    data = api(f'repos/{repository}/releases/assets/{asset_id}', binary=True)
    require(hashlib.sha256(data).hexdigest() == digest, 'Evidence archive SHA256 mismatch')
    destination.write_bytes(data)


def binary_hash(wheel, *, cuda=False):
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        safe_names(names)
        binaries = ([name for name in names if name == 'tensor0_cuda12/_stride_cuda.so'] if cuda else
                    [name for name in names if re.fullmatch(r'tensor0/_native[^/]*\.so', name)])
        require(len(binaries) == 1, 'Expected one native binary in wheel')
        return hashlib.sha256(archive.read(binaries[0])).hexdigest()


def verify_evidence(path, build, version, binary_sha256, cuda_binary_sha256):
    with tarfile.open(path) as archive:
        members = archive.getmembers()
        safe_names([m.name for m in members])
        require(all(m.isfile() or m.isdir() for m in members), 'Evidence links/devices forbidden')
        files = {m.name: m for m in members if m.isfile()}
        required = {'cpu/report.json', 'gpu/report.json', 'cpu/preflight.log', 'gpu/preflight.log',
                    'cpu/cpu-smoke.log'} | {f'gpu/suite-{i}.{ext}' for i in range(3) for ext in ('log', 'xml')}
        require(required == files.keys(), 'Expected only CPU/GPU acceptance reports, logs and JUnit')
        require(all(archive.extractfile(files[f'{mode}/preflight.log']).read() == b'Preflight passed\n'
                    for mode in ('cpu', 'gpu')), 'Acceptance preflight did not pass')
        reports = [json.loads(archive.extractfile(files[f'{mode}/report.json']).read()) for mode in ('cpu', 'gpu')]
        for report, mode in zip(reports, ('cpu', 'cuda')):
            require(isinstance(report, dict), 'Acceptance report must be a JSON object')
            require(report.get('accepted') is True and report.get('mode') == mode
                    and not report.get('error') and report.get('checkout', {}).get('commit') == build['source_commit']
                    and report.get('build_info') == build, 'Production acceptance/source/build-info mismatch')
            # Production preflight consumes checkout_status and does not emit source.is_clean.
            # Its clean checkout gate is mandatory before accepted=True can be produced.
            require(report.get('checkout', {}).get('root') and 'checkout_status' not in report,
                    'Malformed production clean-source report')
            require(report.get('sha256') == build['sha256']
                    and Path(report.get('wheel', '')).name == build['wheel'], 'Acceptance core wheel mismatch')
            direct_url(report.get('direct_url', {}), build['wheel'], build['sha256'])
            require(report.get('core_binary_sha256') == binary_sha256,
                    'Acceptance native binary hash differs from verified cp311 wheel')
            versions = report.get('versions', {})
            require(versions.get('tensor0') == version and versions.get('jax') == versions.get('jaxlib') == '0.10.1'
                    and versions.get('native_jax') == [versions['jax'], versions['jaxlib']]
                    and all(versions.get(k) for k in ('numpy', 'pytest', 'python')), 'Malformed acceptance versions')
            prefix = PurePosixPath(report.get('prefix', ''))
            origins = report.get('origins', {})
            require(report.get('executable') and PurePosixPath(report['executable']).is_relative_to(prefix),
                    'Missing installed acceptance interpreter')
            installed = report.get('installed_packages', {})
            require(installed.get('tensor0') == version
                    and (mode == 'cpu' or installed.get('tensor0-cuda12') == version),
                    'Installed distribution version mismatch')
            require(prefix.is_absolute() and set(origins) == ({'python', 'native'} if mode == 'cpu' else {'python', 'native', 'companion'})
                    and all(PurePosixPath(p).is_relative_to(prefix) and '..' not in PurePosixPath(p).parts for p in origins.values()),
                    'Acceptance origins must belong to installed environment')
        cpu, gpu = reports
        require(cpu.get('suites') == [] and cpu['checkout'] == gpu['checkout']
                and cpu['core_binary_sha256'] == gpu['core_binary_sha256']
                and cpu['prefix'] == gpu['prefix'] and cpu['executable'] == gpu['executable']
                and cpu['origins']['native'] == gpu['origins']['native'],
                'CPU baseline/environment/core binary mismatch')
        require(gpu.get('cuda_binary_sha256') == cuda_binary_sha256,
                'Acceptance addon binary hash differs from verified addon wheel')
        require(gpu.get('cuda_sha256') == build['cuda_sha256']
                and Path(gpu.get('cuda_wheel', '')).name == build['cuda_wheel'], 'GPU addon mismatch')
        direct_url(gpu.get('cuda_direct_url', {}), build['cuda_wheel'], build['cuda_sha256'])
        require(gpu.get('driver', {}).get('returncode') == 0
                and all(gpu.get('devices', {}).get(k) for k in ('cpu', 'cuda'))
                and gpu.get('cuda_registration'), 'Missing GPU preflight evidence')
        require([s.get('suite') for s in gpu.get('suites', [])] == list(SUITES), 'Expected exact three GPU suites once')
        import io
        for i, suite in enumerate(gpu['suites']):
            command = suite.get('command', [])
            require(len(command) == 11 and command[:5] == [gpu['executable'], '-I', '-m', 'pytest', SUITES[i]]
                    and command[5:10] == ['-ra', '-o', 'addopts=', '-o', 'xfail_strict=true']
                    and command[10].startswith('--junitxml=')
                    and Path(command[10].removeprefix('--junitxml=')).name == f'suite-{i}.xml',
                    'Unexpected acceptance suite command')
            xml = archive.extractfile(files[f'gpu/suite-{i}.xml']).read()
            tree = ET.fromstring(xml)
            require(not list(tree.iter('failure')) and not list(tree.iter('error')),
                    'GPU JUnit contains failures/errors')
            result = junit_result(io.BytesIO(xml), suite.get('returncode'))
            require(type(suite.get('passed')) is int and type(suite.get('failed')) is int
                    and type(suite.get('returncode')) is int and result['accepted'] and all(suite.get(k) == v for k, v in result.items()),
                    f'Unaccepted or mismatched GPU JUnit: {SUITES[i]}')
    return reports


def direct_url(direct, name, digest):
    url = urlparse(direct.get('url', ''))
    archive = direct.get('archive_info', {})
    require(url.scheme == 'file' and Path(unquote(url.path)).name == name and not direct.get('dir_info')
            and archive.get('hashes', {}).get('sha256', archive.get('hash', '').removeprefix('sha256=')) == digest,
            'Installed direct_url does not bind exact wheel')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('provenance', 'stage'):
        command = sub.add_parser(name)
        for option in ('repository', 'cpu-run-id', 'cuda-run-id', 'release-tag'):
            command.add_argument('--' + option, required=True)
        if name == 'stage':
            for option in ('sha', 'asset-id', 'evidence-sha256'):
                command.add_argument('--' + option, required=True)
            for option in ('cpu', 'cuda', 'dist', 'evidence'):
                command.add_argument('--' + option, required=True, type=Path)
    args = parser.parse_args()
    if args.command == 'provenance':
        sha = verify_provenance(args.repository, args.cpu_run_id, args.cuda_run_id, args.release_tag)
        with open(os.environ['GITHUB_OUTPUT'], 'a') as output:
            output.write(f'sha={sha}\n')
    else:
        version = args.release_tag[1:]
        packages, build = verify_distributions(args.cpu, args.cuda, version, args.sha,
                                               args.cpu_run_id, args.cuda_run_id, args.repository)
        for name in sorted(packages):
            command = [sys.executable, str(Path(__file__).with_name('check_artifacts.py')),
                       str(args.cuda / name)]
            if name.startswith('tensor0_cuda12-'):
                command.append('--cuda')
            elif name.endswith('.whl'):
                command.append('--release-core')
            subprocess.run(command, check=True)
        args.evidence.mkdir(exist_ok=False)
        archive = args.evidence / 'cuda-acceptance.tar.gz'
        evidence_asset(args.repository, args.release_tag, args.asset_id, args.evidence_sha256, archive)
        binary_sha256 = binary_hash(args.cpu / build['wheel'])
        cuda_binary_sha256 = binary_hash(args.cuda / build['cuda_wheel'], cuda=True)
        reports = verify_evidence(archive, build, version, binary_sha256, cuda_binary_sha256)
        (args.evidence / 'verification.json').write_text(json.dumps({
            'source_commit': args.sha, 'release_tag': args.release_tag,
            'cpu_run_id': args.cpu_run_id, 'cuda_run_id': args.cuda_run_id,
            'asset_id': args.asset_id, 'evidence_sha256': args.evidence_sha256,
            'core_binary_sha256': binary_sha256, 'cuda_binary_sha256': cuda_binary_sha256,
            'distributions': {n: sha256(args.cuda / n) for n in sorted(packages)},
            'reports': reports}, indent=2) + '\n')
        args.dist.mkdir(exist_ok=False)
        for name in packages:
            shutil.copyfile((args.cpu if name.startswith('tensor0-') else args.cuda) / name, args.dist / name)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, KeyError, TypeError, OSError, ET.ParseError, tarfile.TarError,
            zipfile.BadZipFile, subprocess.CalledProcessError) as error:
        raise SystemExit(f'Release verification failed: {error}') from error
