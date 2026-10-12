"""Offline checks for CI test layering; no YAML dependency or native rebuild."""

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
CPP = 'tests/stride/native/cpp/test_contracts.py'


def test_cuda_compilation_runs_for_filtered_prs_or_manual_candidates_only():
    workflow = (ROOT / '.github/workflows/cuda-build.yml').read_text()
    triggers = '\n' + workflow.split('\non:\n')[1].split('\npermissions:\n')[0]
    assert '\n  push:' not in triggers
    assert '\n  pull_request:\n    paths:\n' in triggers
    assert '\n  workflow_dispatch:\n' in triggers
    assert 'candidate_run_id:' in triggers


def test_daily_linux_endpoints_and_single_platform_contracts():
    workflow = (ROOT / '.github/workflows/ci.yml').read_text()
    job = workflow.split('  test:\n')[1].split('  typecheck:\n')[0]
    assert 'python: ["3.11", "3.14"]' in job
    assert 'assert native_available()' in job
    for name in ('Rust workspace tests (once on Linux)',
                 'Standalone native C++ contracts (once on Linux)'):
        step = job.split(f'      - name: {name}\n')[1].split('      - name:')[0]
        assert "if: matrix.python == '3.11'" in step
    assert f'pytest tests -q -ra --ignore={CPP}' in job
    assert job.count('--ignore=') == 1  # Build/packaging checks stay on both ABIs.
    typecheck = workflow.split('  typecheck:\n')[1]
    assert 'name: Public-package Pyright (Python 3.11)' in typecheck
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
    name = 'Resolve locked host Cargo dependencies for offline build fixtures'
    step = job.split(f'      - name: {name}\n')[1].split('      - name:')[0]
    assert "if: matrix.python == '3.11' || matrix.python == '3.14'" in step
    copy = 'cp artifact-source/source/Cargo.lock Cargo.lock'
    fetch = 'cargo fetch --locked'
    assert copy in step
    assert fetch in step
    assert step.index(copy) < step.index(fetch)
    extract = '--extract artifact-source'
    full_suite = 'Full installed-wheel Python suite (baseline and highest ABI)'
    assert job.index(extract) < job.index(copy) < job.index(fetch) < job.index(full_suite)
