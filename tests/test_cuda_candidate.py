"""Offline tests for CUDA reuse of the single CPU release producer."""
import copy
import hashlib
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / '.github/scripts/cuda_candidate.py'
spec = importlib.util.spec_from_file_location('cuda_candidate', SCRIPT)
assert spec and spec.loader
candidate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate)


def run_fixture():
    return {'id': 123, 'workflow_id': 7, 'path': '.github/workflows/artifacts.yml',
            'repository': {'full_name': 'owner/repo'},
            'head_repository': {'full_name': 'owner/repo'}, 'event': 'workflow_dispatch',
            'head_branch': 'main', 'status': 'completed', 'conclusion': 'success',
            'head_sha': 'a' * 40}


def verify(run, run_id='123'):
    candidate.verify_run(run, {'id': 7, 'path': '.github/workflows/artifacts.yml'},
                         'owner/repo', 'a' * 40, run_id)


def test_valid_run():
    verify(run_fixture())


@pytest.mark.parametrize('field,value', [
    ('id', 124), ('workflow_id', 8), ('path', '.github/workflows/cuda-build.yml'),
    ('repository', {'full_name': 'fork/repo'}), ('head_repository', {'full_name': 'fork/repo'}),
    ('event', 'push'), ('head_branch', 'feature'), ('status', 'in_progress'),
    ('conclusion', 'failure'), ('head_sha', 'b' * 40),
])
def test_mismatched_run(field, value):
    run = copy.deepcopy(run_fixture())
    run[field] = value
    with pytest.raises(ValueError):
        verify(run)


@pytest.mark.parametrize('run_id', ['0', '-1', '../123', '123\n', 'abc'])
def test_invalid_run_id(run_id):
    with pytest.raises(ValueError):
        verify(run_fixture(), run_id)


def distributions(root):
    names = ['tensor0-0.2.0.dev0.tar.gz'] + [
        f'tensor0-0.2.0.dev0-{abi}-{abi}-{platform}.whl'
        for abi in candidate.ABIS for platform in candidate.CORE_PLATFORMS]
    for name in names:
        (root / name).write_bytes(name.encode())
    (root / 'SHA256SUMS').write_text(''.join(
        f'{hashlib.sha256((root / name).read_bytes()).hexdigest()}  {name}\n' for name in names))
    return names


def test_exact_distributions(tmp_path):
    distributions(tmp_path)
    wheels = candidate.verify_files(tmp_path, '0.2.0.dev0')
    assert set(wheels) == set(candidate.ABIS)
    assert all('manylinux_2_28_x86_64' in name for name in wheels.values())


@pytest.mark.parametrize('problem', ['digest', 'traversal', 'duplicate', 'extra', 'version', 'missing', 'symlink'])
def test_bad_distributions(tmp_path, problem):
    names = distributions(tmp_path)
    manifest = tmp_path / 'SHA256SUMS'
    text = manifest.read_text()
    if problem == 'digest':
        (tmp_path / names[0]).write_bytes(b'changed')
    elif problem == 'traversal':
        manifest.write_text(text.replace(names[0], '../' + names[0]))
    elif problem == 'duplicate':
        manifest.write_text(text + text.splitlines()[0] + '\n')
    elif problem == 'extra':
        (tmp_path / 'build-info.json').write_text('{}')
    elif problem == 'version':
        (tmp_path / names[0]).rename(tmp_path / 'tensor0-0.1.0.tar.gz')
    elif problem == 'missing':
        (tmp_path / names[1]).unlink()
    else:
        (tmp_path / names[0]).unlink()
        (tmp_path / names[0]).symlink_to(tmp_path / names[1])
    with pytest.raises(ValueError):
        candidate.verify_files(tmp_path, '0.2.0.dev0')


def test_workflow_keeps_compile_route_and_reuses_manual_core():
    text = (SCRIPT.parents[1] / 'workflows/cuda-build.yml').read_text()
    assert 'pull_request:' in text and 'push:' in text
    assert 'candidate_run_id:' in text and 'required: true' in text
    assert "if: github.event_name != 'workflow_dispatch'" in text
    assert 'name: tensor0-dist-${{ github.sha }}' in text
    assert 'run-id: ${{ inputs.candidate_run_id }}' in text
    assert 'cuda_candidate.py files' in text
    assert "'release_candidate': os.environ['GITHUB_EVENT_NAME'] == 'workflow_dispatch'" in text
    assert "python: ['3.11', '3.12', '3.13', '3.14']" in text
    assert '--only-plat' in text and '--exclude' not in text
    assert 'uv pip install --no-deps --python "$python" ./tensor0_cuda12-*.whl' in text


def test_local_source_does_not_change_published_pin():
    import tomllib
    root = SCRIPT.parents[2]
    core = tomllib.loads((root / 'pyproject.toml').read_text())
    addon = tomllib.loads((root / 'crates/tensor0-cuda12/pyproject.toml').read_text())
    assert core['tool']['uv']['sources']['tensor0-cuda12'] == {
        'path': 'crates/tensor0-cuda12', 'editable': False}
    assert f"tensor0-cuda12=={addon['project']['version']}" in core['project']['optional-dependencies']['cuda12']
    assert f"tensor0=={core['project']['version']}" in addon['project']['dependencies']


@pytest.mark.parametrize('platform', ['macosx_14_0_arm64', 'macosx_16_0_arm64',
                                      'macosx_15_0_x86_64', 'macosx_15_0_universal2',
                                      'manylinux_2_28_aarch64'])
def test_wrong_platform_matrix(tmp_path, platform):
    names = distributions(tmp_path)
    original = next(name for name in names if 'cp311-cp311-macosx' in name)
    (tmp_path / original).rename(tmp_path / original.replace('macosx_15_0_arm64', platform))
    with pytest.raises(ValueError):
        candidate.verify_files(tmp_path, '0.2.0.dev0')


@pytest.mark.parametrize('abi', ['cp310-cp310', 'cp315-cp315', 'cp314-cp314t', 'cp311-abi3'])
def test_wrong_abi_matrix(tmp_path, abi):
    names = distributions(tmp_path)
    original = next(name for name in names if 'cp314-cp314-macosx' in name)
    (tmp_path / original).rename(tmp_path / original.replace('cp314-cp314', abi))
    with pytest.raises(ValueError):
        candidate.verify_files(tmp_path, '0.2.0.dev0')
