"""Offline checks for CI test layering; no YAML dependency or native rebuild."""

import os
from pathlib import Path
import re
import subprocess
import textwrap

import pytest


ROOT = Path(__file__).resolve().parents[2]
CPP = 'tests/stride/native/cpp/test_contracts.py'


def test_daily_linux_endpoints_and_single_platform_contracts():
    workflow = (ROOT / '.github/workflows/ci.yml').read_text()
    job = workflow.split('  test:\n')[1].split('  typecheck:\n')[0]
    assert 'python: ["3.11", "3.14"]' in job
    assert 'assert native_available()' in job
    for command in ('cargo test --workspace', f'pytest {CPP}'):
        step, = (step for step in job.split('      - name:') if command in step)
        assert "if: matrix.python == '3.11'" in step
    assert f'pytest tests -q -ra --ignore={CPP}' in job
    assert job.count('--ignore=') == 1  # Build/packaging checks stay on both ABIs.
    typecheck = workflow.split('  typecheck:\n')[1]
    assert 'uv run --no-sync pyright src examples benchmarks' in typecheck
    assert 'matrix:' not in typecheck


def test_release_keeps_eight_wheels_and_identical_bounded_contracts():
    workflow = (ROOT / '.github/workflows/artifacts.yml').read_text()
    selections = []
    for key, end in (('wheel', 'macos-wheel'), ('macos-wheel', 'assemble')):
        job = workflow.split(f'  {key}:\n')[1].split(f'  {end}:\n')[0]
        assert 'python: ["3.11", "3.12", "3.13", "3.14"]' in job
        assert "if: matrix.python == '3.11' || matrix.python == '3.14'" in job
        assert "if: matrix.python == '3.12' || matrix.python == '3.13'" in job
        assert f'--ignore={CPP}' in job
        assert 'wheel_smoke.py" "$GITHUB_WORKSPACE"/dist/*.whl' in job
        nodes = re.findall(r'tests/stride/jax/ad/test_\w+\.py::test_\w+', job)
        assert len(nodes) == 5
        assert {node.split('::')[0].split('/')[-1] for node in nodes} == {
            'test_copy.py', 'test_update.py', 'test_reduction.py',
            'test_dot.py', 'test_accumulation.py'}
        selections.append(nodes)
    assert selections[0] == selections[1]
    assert 'test "$(find dist -name \'*.whl\' | wc -l)" -eq 8' in workflow


def test_release_host_fetch_reuses_shared_sdist_lock_before_offline_fixtures():
    workflow = (ROOT / '.github/workflows/artifacts.yml').read_text()
    job = workflow.split('  wheel:\n')[1].split('  macos-wheel:\n')[0]
    step, = (step for step in job.split('      - name:') if 'cargo fetch --locked' in step)
    assert "if: matrix.python == '3.11' || matrix.python == '3.14'" in step
    copy = 'cp artifact-source/source/Cargo.lock Cargo.lock'
    fetch = 'cargo fetch --locked'
    assert copy in step
    assert fetch in step
    assert step.index(copy) < step.index(fetch)
    extract = '--extract artifact-source'
    full_suite = '-I -m pytest tests -q'
    assert job.index(extract) < job.index(copy) < job.index(fetch) < job.index(full_suite)


@pytest.mark.parametrize('classification,run_cpu,tests,typing,ok', [
    ('success', 'false', 'skipped', 'skipped', True),
    ('success', 'true', 'success', 'success', True),
    ('failure', 'false', 'skipped', 'skipped', False),
    ('cancelled', 'false', 'skipped', 'skipped', False),
    ('success', '', 'skipped', 'skipped', False),
    ('success', 'true', 'skipped', 'success', False),
    ('success', 'true', 'success', 'skipped', False),
    ('success', 'true', 'failure', 'success', False),
    ('success', 'true', 'cancelled', 'success', False),
    ('success', 'true', 'success', 'failure', False),
    ('success', 'false', 'success', 'skipped', False),
])
def test_ci_result_fails_closed(tmp_path, classification, run_cpu, tests, typing, ok):
    workflow = (ROOT / '.github/workflows/ci.yml').read_text()
    gate = workflow.split('  result:\n')[1]
    assert 'if: always()' in gate
    assert 'needs: [changes, test, typecheck]' in gate
    script = textwrap.dedent(gate.split('        run: |\n')[1])
    env = dict(os.environ, CHANGES_RESULT=classification, RUN_CPU=run_cpu,
               TEST_RESULT=tests, TYPECHECK_RESULT=typing,
               GITHUB_STEP_SUMMARY=str(tmp_path / 'summary'))
    result = subprocess.run(['bash', '-e', '-c', script], env=env, capture_output=True)
    assert (result.returncode == 0) is ok


def test_workflow_keeps_required_check_triggers_and_conditional_heavy_jobs():
    workflow = (ROOT / '.github/workflows/ci.yml').read_text()
    triggers = workflow.split('\non:\n')[1].split('\npermissions:\n')[0]
    assert 'paths' not in triggers
    for event in ('pull_request', 'push', 'workflow_dispatch'):
        assert f'  {event}:' in triggers
    assert 'fetch-depth: 0' in workflow
    for name in ('test', 'typecheck'):
        job = re.split(r'\n  \S', workflow.split(f'  {name}:\n')[1])[0]
        assert 'needs: changes' in job
        assert "if: needs.changes.outputs.run_cpu == 'true'" in job


def test_workflow_keeps_compile_route_and_reuses_manual_core():
    text = (ROOT / '.github/workflows/cuda-build.yml').read_text()
    triggers = '\n' + text.split('\non:\n')[1].split('\npermissions:\n')[0]
    assert '\n  pull_request:\n    paths:\n' in triggers
    assert '\n  workflow_dispatch:\n' in triggers
    assert '\n  push:' not in triggers
    assert 'candidate_run_id:' in text and 'required: true' in text
    core_build, = (step for step in text.split('      - name:') if 'maturin build' in step)
    assert "if: github.event_name != 'workflow_dispatch'" in core_build
    assert 'name: tensor0-dist-${{ github.sha }}' in text
    assert 'run-id: ${{ inputs.candidate_run_id }}' in text
    assert 'cuda_candidate.py files' in text
    assert "'release_candidate': os.environ['GITHUB_EVENT_NAME'] == 'workflow_dispatch'" in text
    assert "python: ['3.11', '3.12', '3.13', '3.14']" in text
    repair, = (step for step in text.split('      - name:') if 'auditwheel repair' in step)
    assert '--plat manylinux_2_28_x86_64' in repair and '--only-plat' in repair
    assert '--exclude' not in text


def test_shared_source_and_full_macos_matrix_gate():
    workflow = (ROOT / '.github/workflows/artifacts.yml').read_text()
    mac_job = workflow.split('  macos-wheel:\n')[1].split('  assemble:\n')[0]
    assert 'needs: sdist' in mac_job and 'runs-on: macos-15' in mac_job
    assert 'name: tensor0-sdist-${{ github.sha }}' in mac_job
    assert 'enable-cache: false' in mac_job and 'CARGO_TARGET_DIR=$work/target' in mac_job
    assert 'CXX: /usr/bin/clang++' in mac_job
    assert 'MACOSX_DEPLOYMENT_TARGET: "15.0"' in mac_job
    assert '--no-install-project' in mac_job and 'maturin" build --release --locked' in mac_job
    assert 'maturin" sdist' not in mac_job and 'cuda12' not in mac_job
    assert 'pytest tests/stride/native/cpp/test_contracts.py' in mac_job
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
