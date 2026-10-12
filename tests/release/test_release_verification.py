"""Offline release verification fixtures; no network, builds or publication."""

import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tarfile
import zipfile

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / '.github/scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('verify_release', SCRIPTS / 'verify_release.py')
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)
sys.path.remove(str(SCRIPTS))
VERSION = '0.2.0'
SHA = 'a' * 40
BINARY_SHA = hashlib.sha256(b'\x7fELFfixture').hexdigest()
REPO = 'owner/repo'


def tar(path, entries):
    with tarfile.open(path, 'w:gz') as archive:
        for name, content in entries.items():
            data = content.encode() if isinstance(content, str) else content
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))


def distribution(path, *, version=VERSION, deps=None):
    addon = path.name.startswith('tensor0_cuda12')
    text = f'Name: {"tensor0-cuda12" if addon else "tensor0"}\nVersion: {version}\n'
    if deps is None:
        deps = [f'tensor0=={VERSION}', 'nvidia-cuda-runtime-cu12<13,>=12.9.79'] if addon else ['jax==0.10.1']
    text += 'License-Expression: MIT AND Apache-2.0\nLicense-File: LICENSE\n'
    text += 'License-File: ' + ('vendor/jaxlib-0.10.1/LICENSE.txt' if addon else
                              'crates/tensor0-py/vendor/jaxlib-0.10.1/LICENSE.txt') + '\n'
    text += ''.join(f'Requires-Dist: {d}\n' for d in deps)
    if path.suffix == '.whl':
        info = f'{"tensor0_cuda12" if addon else "tensor0"}-{VERSION}.dist-info'
        tag = path.name[:-4].split('-')[2:]
        with zipfile.ZipFile(path, 'w') as archive:
            archive.writestr(info + '/METADATA', text)
            archive.writestr(info + '/WHEEL', 'Wheel-Version: 1.0\nRoot-Is-Purelib: false\n'
                             + ''.join(f'Tag: {tag[0]}-{tag[1]}-{p}\n' for p in tag[2].split('.')))
            archive.writestr('tensor0_cuda12/_stride_cuda.so' if addon else 'tensor0/_native.so', b'\x7fELFfixture')
    else:
        tar(path, {path.name[:-7] + '/PKG-INFO': text})


def manifest(path):
    (path / 'SHA256SUMS').write_text(''.join(f'{release.sha256(p)}  {p.name}\n'
                                          for p in sorted(path.iterdir()) if p.name.endswith(('.whl', '.tar.gz'))))


@pytest.fixture
def candidate(tmp_path):
    cpu, cuda = tmp_path / 'cpu', tmp_path / 'cuda'
    cpu.mkdir()
    cuda.mkdir()
    for abi in release.ABIS:
        for platform in release.CORE_PLATFORMS:
            distribution(cpu / f'tensor0-{VERSION}-{abi}-{abi}-{platform}.whl')
    distribution(cpu / f'tensor0-{VERSION}.tar.gz')
    manifest(cpu)
    for path in cpu.iterdir():
        if path.name != 'SHA256SUMS':
            (cuda / path.name).write_bytes(path.read_bytes())
    addon = f'tensor0_cuda12-{VERSION}-py3-none-manylinux_2_28_x86_64.whl'
    distribution(cuda / addon)
    distribution(cuda / f'tensor0_cuda12-{VERSION}.tar.gz')
    manifest(cuda)
    wheel = f'tensor0-{VERSION}-cp311-cp311-manylinux_2_28_x86_64.whl'
    build = {k: 'fixture' for k in release.BUILD_FIELDS}
    build.update(source_commit=SHA, release_candidate=True, cpu_candidate_run_id='1',
                 cpu_candidate_run_url=f'https://github.com/{REPO}/actions/runs/1',
                 workflow_run_url=f'https://github.com/{REPO}/actions/runs/2',
                 wheel=wheel, sha256=release.sha256(cpu / wheel), cuda_wheel=addon,
                 cuda_sha256=release.sha256(cuda / addon),
                 sdists={p.name: release.sha256(p) for p in cuda.glob('*.tar.gz')})
    (cuda / 'build-info.json').write_text(json.dumps(build))
    return cpu, cuda, build


def verify(candidate):
    cpu, cuda, _ = candidate
    return release.verify_distributions(cpu, cuda, VERSION, SHA, '1', '2', REPO)


def test_exact_eleven(candidate):
    packages, build = verify(candidate)
    assert len(packages) == 11
    assert build == candidate[2]


@pytest.mark.parametrize('key,value', [('release_candidate', False), ('cpu_candidate_run_id', '9'),
                                      ('source_commit', 'b' * 40), ('sha256', '0' * 64)])
def test_build_binding(candidate, key, value):
    candidate[2][key] = value
    (candidate[1] / 'build-info.json').write_text(json.dumps(candidate[2]))
    with pytest.raises(ValueError):
        verify(candidate)


def test_duplicate_core_must_be_exact(candidate):
    cpu, cuda, build = candidate
    (cuda / build['wheel']).write_bytes(b'different')
    manifest(cuda)
    with pytest.raises(ValueError, match='Duplicate core differs'):
        verify(candidate)


@pytest.mark.parametrize('change', ['extra', 'traversal', 'duplicate', 'hash'])
def test_manifest_failures(candidate, change):
    cpu, cuda, build = candidate
    file = cuda / 'SHA256SUMS'
    if change == 'extra':
        (cuda / 'unreviewed.whl').write_bytes(b'x')
    elif change == 'traversal':
        file.write_text(file.read_text().replace(build['wheel'], '../' + build['wheel']))
    elif change == 'duplicate':
        file.write_text(file.read_text() + file.read_text().splitlines()[0] + '\n')
    else:
        file.write_text(file.read_text().replace(build['sha256'], '0' * 64))
    with pytest.raises(ValueError):
        verify(candidate)


@pytest.mark.parametrize('deps,version', [([f'tensor0=={VERSION}', 'nvidia-cuda-runtime-cu12>=12,<14'], VERSION),
                                        (['tensor0==0.1.0', 'nvidia-cuda-runtime-cu12<13,>=12.9.79'], VERSION),
                                        (['tensor0 @ https://example.invalid/core.whl'], VERSION),
                                        (None, '0.1.0')])
def test_wrong_metadata(candidate, deps, version):
    path = candidate[1] / candidate[2]['cuda_wheel']
    distribution(path, deps=deps, version=version)
    with pytest.raises(ValueError):
        release.metadata(path, VERSION)


def direct(name, digest):
    return {'url': 'file:///wheels/' + name, 'archive_info': {'hashes': {'sha256': digest}}}


def reports(build):
    cpu = dict(accepted=True, mode='cpu', checkout={'commit': SHA, 'root': '/checkout'}, build_info=build,
               wheel='/wheels/' + build['wheel'], sha256=build['sha256'],
               direct_url=direct(build['wheel'], build['sha256']), core_binary_sha256=BINARY_SHA,
               prefix='/venv', executable='/venv/bin/python', installed_packages={'tensor0': VERSION}, origins={'python': '/venv/tensor0/__init__.py', 'native': '/venv/tensor0/_native.so'},
               versions={'tensor0': VERSION, 'jax': '0.10.1', 'jaxlib': '0.10.1', 'native_jax': ['0.10.1', '0.10.1'],
                         'python': '3.11 fixture', 'numpy': '2.0', 'pytest': '8'}, suites=[])
    gpu = copy.deepcopy(cpu)
    gpu.update(mode='cuda', cuda_binary_sha256=BINARY_SHA, cuda_wheel='/wheels/' + build['cuda_wheel'], cuda_sha256=build['cuda_sha256'],
               cuda_direct_url=direct(build['cuda_wheel'], build['cuda_sha256']),
               driver={'returncode': 0}, devices={'cpu': ['cpu'], 'cuda': ['gpu']}, cuda_registration=['copy_f32'])
    gpu['origins']['companion'] = '/venv/tensor0_cuda12/_stride_cuda.so'
    gpu['installed_packages']['tensor0-cuda12'] = VERSION
    gpu['suites'] = [dict(suite=s, returncode=0, passed=1, failed=0, skipped=[], accepted=True,
                          command=['/venv/bin/python', '-I', '-m', 'pytest', s, '-ra', '-o', 'addopts=',
                                   '-o', 'xfail_strict=true', f'--junitxml=/output/suite-{i}.xml'])
                     for i, s in enumerate(release.SUITES)]
    return cpu, gpu


def evidence(tmp_path, cpu, gpu, *, xml=None, remove=None, extra=None):
    entries = {'cpu/report.json': json.dumps(cpu), 'gpu/report.json': json.dumps(gpu),
               'cpu/preflight.log': 'Preflight passed\n', 'gpu/preflight.log': 'Preflight passed\n',
               'cpu/cpu-smoke.log': 'fixture'}
    for i in range(3):
        entries[f'gpu/suite-{i}.log'] = 'fixture'
        entries[f'gpu/suite-{i}.xml'] = xml or '<testsuite><testcase name="passes"/></testsuite>'
    if remove:
        entries.pop(remove)
    if extra:
        entries[extra] = '{}'
    path = tmp_path / 'evidence.tar.gz'
    tar(path, entries)
    return path


def test_production_reports(tmp_path, candidate):
    cpu, gpu = reports(candidate[2])
    assert release.verify_evidence(evidence(tmp_path, cpu, gpu), candidate[2], VERSION, BINARY_SHA, BINARY_SHA) == [cpu, gpu]


@pytest.mark.parametrize('change', ['false', 'experimental', 'commit', 'core', 'url', 'version',
                                   'origins', 'clean', 'suite', 'zero', 'failed', 'missing', 'json', 'traversal', 'both_core', 'addon_binary', 'two_venvs', 'native_origin', 'executable'])
def test_evidence_failures(tmp_path, candidate, change):
    cpu, gpu = reports(candidate[2])
    kwargs = {}
    if change == 'false':
        gpu['accepted'] = False
    elif change == 'experimental':
        gpu.pop('accepted')
        gpu['experimental_accepted'] = True
    elif change == 'commit':
        gpu['checkout']['commit'] = 'd' * 40
    elif change == 'core':
        gpu['core_binary_sha256'] = 'd' * 64
    elif change == 'both_core':
        cpu['core_binary_sha256'] = gpu['core_binary_sha256'] = 'd' * 64
    elif change == 'addon_binary':
        gpu['cuda_binary_sha256'] = 'd' * 64
    elif change == 'two_venvs':
        gpu['prefix'] = '/other-venv'
        gpu['executable'] = '/other-venv/bin/python'
        gpu['origins'] = {k: v.replace('/venv/', '/other-venv/') for k, v in gpu['origins'].items()}
    elif change == 'native_origin':
        gpu['origins']['native'] = '/venv/different-native.so'
    elif change == 'executable':
        gpu['executable'] = '/venv/bin/different-python'
    elif change == 'url':
        gpu['cuda_direct_url']['archive_info']['hashes']['sha256'] = 'd' * 64
    elif change == 'version':
        gpu['versions']['tensor0'] = '0.1.0'
    elif change == 'origins':
        gpu['origins']['native'] = '/checkout/native.so'
    elif change == 'clean':
        gpu['checkout_status'] = 'M modified.py'
    elif change == 'suite':
        gpu['suites'][0]['suite'] = 'tests/unexpected'
    elif change == 'zero':
        kwargs['xml'] = '<testsuite/>'
    elif change == 'failed':
        kwargs['xml'] = '<testsuite><testcase><failure/></testcase></testsuite>'
    elif change == 'missing':
        kwargs['remove'] = 'cpu/cpu-smoke.log'
    elif change == 'traversal':
        kwargs['extra'] = '../gpu/report.json'
    path = evidence(tmp_path, cpu, gpu, **kwargs)
    if change == 'json':
        tar(path, {'cpu/report.json': '{', 'gpu/report.json': '{}', 'cpu/preflight.log': '',
                   'gpu/preflight.log': '', 'cpu/cpu-smoke.log': '',
                   **{f'gpu/suite-{i}.{ext}': '' for i in range(3) for ext in ('log', 'xml')}})
    with pytest.raises((ValueError, json.JSONDecodeError)):
        release.verify_evidence(path, candidate[2], VERSION, BINARY_SHA, BINARY_SHA)


@pytest.mark.parametrize('reason', ['requires a CPU-only extension', 'requires CPU-only extension', 'no GPU'])
def test_skip_policy(tmp_path, candidate, reason):
    cpu, gpu = reports(candidate[2])
    xml = f'<testsuite><testcase name="passes"/><testcase name="skip"><skipped type="pytest.skip" message="{reason}"/></testcase></testsuite>'
    for suite in gpu['suites']:
        suite['skipped'] = [{'case': '::skip', 'reason': reason, 'type': 'pytest.skip', 'detail': None}]
    path = evidence(tmp_path, cpu, gpu, xml=xml)
    if reason == 'no GPU':
        with pytest.raises(ValueError):
            release.verify_evidence(path, candidate[2], VERSION, BINARY_SHA, BINARY_SHA)
    else:
        release.verify_evidence(path, candidate[2], VERSION, BINARY_SHA, BINARY_SHA)


def test_asset_authenticated_and_bound(tmp_path, monkeypatch):
    data = b'archive bytes'
    calls = []
    def api(path, *, binary=False):
        calls.append((path, binary))
        if '/releases/tags/' in path:
            return {'id': 9, 'tag_name': 'v0.2.0', 'draft': True}
        if '/releases/9/assets?' in path:
            return [{'id': 3, 'name': 'cuda-acceptance.tar.gz', 'state': 'uploaded'}]
        assert path == f'repos/{REPO}/releases/assets/3' and binary
        return data
    monkeypatch.setattr(release, 'api', api)
    path = tmp_path / 'download'
    release.evidence_asset(REPO, 'v0.2.0', '3', hashlib.sha256(data).hexdigest(), path)
    assert path.read_bytes() == data
    with pytest.raises(ValueError, match='belong'):
        release.evidence_asset(REPO, 'v0.2.0', '4', '0' * 64, path)
    with pytest.raises(ValueError, match='SHA256'):
        release.evidence_asset(REPO, 'v0.2.0', '3', '0' * 64, path)


@pytest.fixture
def provenance(monkeypatch):
    def run(run_id, workflow, event='workflow_dispatch'):
        return dict(id=run_id, workflow_id={'artifacts.yml': 10, 'cuda-build.yml': 11, 'ci.yml': 12}[workflow],
                    path='.github/workflows/' + workflow, repository={'full_name': REPO},
                    head_repository={'full_name': REPO}, event=event, head_branch='main',
                    status='completed', conclusion='success', head_sha=SHA)
    runs = {'1': run(1, 'artifacts.yml'), '2': run(2, 'cuda-build.yml'), '3': run(3, 'ci.yml', 'push')}
    cuda_jobs = ['verify-candidate', 'CUDA 12.9 / sm_80 / CPython 3.11 (compile and CPU smoke only)']
    cuda_jobs += [f'Exact CPU candidate + same addon / Python {v} (no GPU)' for v in ('3.11', '3.12', '3.13', '3.14')]
    cpu_jobs = ['Build source distribution once', 'Assemble tested distributions (no rebuild)']
    cpu_jobs += [f'CPython {v} / {platform}' for v in ('3.11', '3.12', '3.13', '3.14')
                 for platform in ('manylinux_2_28 x86_64', 'macOS 15 ARM64 CPU')]
    records = {'1': [dict(name=n, conclusion='success') for n in cpu_jobs], '2': [dict(name=n, conclusion='success') for n in cuda_jobs],
               '3': [dict(name=f'Linux x86_64 / Python {v}', conclusion='success') for v in ('3.11', '3.14')]
                    + [dict(name='Public-package Pyright (Python 3.11)', conclusion='success')]}
    def api(path):
        if '/jobs?' in path:
            return {'jobs': records[path.split('/runs/')[1].split('/')[0]]}
        if '/actions/runs/' in path:
            return runs[path.rsplit('/', 1)[1]]
        if 'ci.yml/runs?' in path:
            assert 'event=' not in path  # Both push and manual full CI must be discoverable.
            assert f'head_sha={SHA}' in path and 'branch=main' in path
            return {'workflow_runs': [run for key, run in runs.items() if key not in ('1', '2')]}
        filename = path.rsplit('/', 1)[1]
        return {'id': {'artifacts.yml': 10, 'cuda-build.yml': 11, 'ci.yml': 12}[filename],
                'path': '.github/workflows/' + filename}
    def git(*args):
        if args[0] == 'rev-parse':
            return SHA
        return '[workspace.package]\nversion="0.2.0"' if args[1].endswith(':Cargo.toml') else '[project]\nversion="0.2.0"'
    monkeypatch.setattr(release, 'api', api)
    monkeypatch.setattr(release, 'git', git)
    monkeypatch.setattr(release.subprocess, 'run', lambda *a, **kw: None)
    return runs, records


def test_provenance_positive(provenance):
    assert release.verify_provenance(REPO, '1', '2', 'v0.2.0') == SHA


@pytest.mark.parametrize('change', ['source', 'automatic', 'foreign', 'skipped', 'missing', 'failure', 'typecheck', 'mac-skipped', 'mac-missing', 'dev'])
def test_provenance_rejections(provenance, change):
    runs, jobs = provenance
    tag = 'v0.2.0'
    if change == 'source':
        runs['2']['head_sha'] = 'b' * 40
    elif change == 'automatic':
        runs['2']['event'] = 'pull_request'
    elif change == 'foreign':
        runs['2']['head_repository']['full_name'] = 'fork/repo'
    elif change == 'skipped':
        jobs['2'][-1]['conclusion'] = 'skipped'
    elif change == 'missing':
        jobs['2'].pop()
    elif change == 'failure':
        jobs['2'][1]['conclusion'] = 'failure'
    elif change == 'mac-skipped':
        jobs['1'][-1]['conclusion'] = 'skipped'
    elif change == 'mac-missing':
        jobs['1'].pop()
    elif change == 'typecheck':
        jobs['3'][-1]['conclusion'] = 'failure'
    else:
        tag = 'v0.2.0.dev0'
    with pytest.raises(ValueError):
        release.verify_provenance(REPO, '1', '2', tag)


@pytest.mark.parametrize('version', ['3.11', '3.14'])
@pytest.mark.parametrize('problem', ['missing', 'failure', 'skipped'])
def test_required_ci_endpoints(provenance, version, problem):
    _, jobs = provenance
    name = f'Linux x86_64 / Python {version}'
    if problem == 'missing':
        jobs['3'][:] = [job for job in jobs['3'] if job['name'] != name]
    else:
        next(job for job in jobs['3'] if job['name'] == name)['conclusion'] = problem
    with pytest.raises(ValueError):
        release.verify_provenance(REPO, '1', '2', 'v0.2.0')


@pytest.mark.parametrize('platform', ['manylinux_2_28 x86_64', 'macOS 15 ARM64 CPU'])
@pytest.mark.parametrize('version', ['3.12', '3.13'])
def test_intermediate_wheel_jobs_still_required(provenance, platform, version):
    _, jobs = provenance
    name = f'CPython {version} / {platform}'
    jobs['1'][:] = [job for job in jobs['1'] if job['name'] != name]
    with pytest.raises(ValueError):
        release.verify_provenance(REPO, '1', '2', 'v0.2.0')


def test_core_hash_from_unique_wheel(candidate):
    wheel = candidate[0] / candidate[2]['wheel']
    assert release.binary_hash(wheel) == BINARY_SHA
    with zipfile.ZipFile(wheel, 'a') as archive:
        archive.writestr('tensor0/_native.other.so', b'\x7fELFother')
    with pytest.raises(ValueError, match='one native'):
        release.binary_hash(wheel)


def test_mac_duplicate_must_be_original(candidate):
    name = f'tensor0-{VERSION}-cp314-cp314-macosx_15_0_arm64.whl'
    (candidate[1] / name).write_bytes(b'repacked macOS wheel')
    manifest(candidate[1])
    with pytest.raises(ValueError, match='Duplicate core differs'):
        verify(candidate)


def test_gpu_pair_cannot_select_mac_wheel(candidate):
    cpu, cuda, build = candidate
    build['wheel'] = f'tensor0-{VERSION}-cp311-cp311-macosx_15_0_arm64.whl'
    build['sha256'] = release.sha256(cpu / build['wheel'])
    (cuda / 'build-info.json').write_text(json.dumps(build))
    with pytest.raises(ValueError, match='distribution hashes mismatch'):
        verify(candidate)


@pytest.mark.parametrize('problem', ['tags', 'license', 'license-files'])
def test_mac_metadata_rejected(candidate, problem):
    path = candidate[0] / f'tensor0-{VERSION}-cp311-cp311-macosx_15_0_arm64.whl'
    with zipfile.ZipFile(path) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    name = next(name for name in entries if name.endswith('/WHEEL' if problem == 'tags' else '/METADATA'))
    old, new = {'tags': (b'macosx_15_0_arm64', b'macosx_14_0_arm64'),
                'license': (b'MIT AND Apache-2.0', b'MIT'),
                'license-files': (b'License-File: LICENSE\n', b'')}[problem]
    entries[name] = entries[name].replace(old, new)
    with zipfile.ZipFile(path, 'w') as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    with pytest.raises(ValueError):
        release.metadata(path, VERSION)


def test_manual_full_ci_is_release_evidence(provenance):
    runs, _ = provenance
    runs['3']['event'] = 'workflow_dispatch'
    assert release.verify_provenance(REPO, '1', '2', 'v0.2.0') == SHA


@pytest.mark.parametrize('event', ['push', 'workflow_dispatch'])
def test_light_ci_result_is_not_release_evidence(provenance, event):
    runs, jobs = provenance
    runs['3']['event'] = event
    for job in jobs['3']:
        job['conclusion'] = 'skipped'
    jobs['3'].append(dict(name='CI result', conclusion='success'))
    with pytest.raises(ValueError, match='CPU matrix and blocking public Pyright'):
        release.verify_provenance(REPO, '1', '2', 'v0.2.0')


@pytest.mark.parametrize('field,value', [
    ('event', 'pull_request'), ('event', 'schedule'), ('head_branch', 'feature'),
    ('head_sha', 'b' * 40), ('workflow_id', 99), ('path', '.github/workflows/other.yml'),
    ('repository', {'full_name': 'foreign/repo'}),
    ('head_repository', {'full_name': 'fork/repo'}),
    ('status', 'in_progress'), ('conclusion', 'failure'), ('conclusion', 'cancelled'),
])
def test_manual_ci_identity_and_result_still_required(provenance, field, value):
    runs, _ = provenance
    runs['3']['event'] = 'workflow_dispatch'
    runs['3'][field] = value
    with pytest.raises(ValueError, match='CPU matrix and blocking public Pyright'):
        release.verify_provenance(REPO, '1', '2', 'v0.2.0')


@pytest.mark.parametrize('light_first', [True, False])
def test_light_ci_does_not_hide_same_commit_full_manual_ci(provenance, light_first):
    runs, jobs = provenance
    runs['4'] = copy.deepcopy(runs['3'])
    runs['4'].update(id=4, event='workflow_dispatch')
    jobs['4'] = copy.deepcopy(jobs['3'])
    for job in jobs['3']:
        job['conclusion'] = 'skipped'
    jobs['3'].append(dict(name='CI result', conclusion='success'))
    if not light_first:
        runs['3'] = runs.pop('3')
    assert release.verify_provenance(REPO, '1', '2', 'v0.2.0') == SHA
