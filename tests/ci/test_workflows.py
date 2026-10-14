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
    assert 'python: ["3.12", "3.15"]' in job
    assert 'assert native_available()' in job
    assert 'sys.version_info.releaselevel == "final"' in job
    for command in ('cargo test --workspace', f'pytest {CPP}'):
        step, = (step for step in job.split('      - name:') if command in step)
        assert "if: matrix.python == '3.12'" in step
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
        assert 'python: ["3.12", "3.13", "3.14", "3.15"]' in job
        assert "if: matrix.python == '3.12' || matrix.python == '3.15'" in job
        assert "if: matrix.python == '3.13' || matrix.python == '3.14'" in job
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


@pytest.mark.parametrize('filename', ['ci.yml', 'artifacts.yml', 'cuda-build.yml', 'macos.yml', 'docs.yml'])
def test_uv_uses_a_download_index_with_final_python_315(filename):
    workflow = (ROOT / '.github/workflows' / filename).read_text()
    steps = workflow.split('      - uses: astral-sh/setup-uv@v8.3.2\n')[1:]
    assert steps
    for step in steps:
        inputs = step.split('      - ')[0]
        assert 'version: "0.13.0"' in inputs
        if 'matrix.python' in inputs:
            assert "python-version: ${{ matrix.python == '3.15' && '3.15.0' || matrix.python }}" in inputs


def test_wheel_smoke_requires_final_python_and_direct_tensor_map_vmap():
    smoke = (ROOT / '.github/scripts/wheel_smoke.py').read_text()
    assert "sys.implementation.name == 'cpython' and sys.version_info.releaselevel == 'final'" in smoke
    assert 'jax.jit(jax.vmap(lambda a: from_dense(target, a)))' in smoke
    assert 'jax.jit(jax.vmap(lambda a: TensorMap(symmetric_space, a)))' in smoke
    assert 'isinstance(batched, TensorMap)' in smoke
    assert 'isinstance(batched_symmetric, TensorMap)' in smoke
    assert 'jax.jit(jax.vmap(lambda a: a @ a))(batched)' in smoke
    assert 'jax.jit(jax.vmap(lambda a: a @ a))(batched_symmetric)' in smoke


def test_release_maturin_tool_is_pinned():
    artifacts = (ROOT / '.github/workflows/artifacts.yml').read_text()
    cuda = (ROOT / '.github/workflows/cuda-build.yml').read_text()
    assert 'maturin-version: "1.15.0"' in artifacts
    assert '--from maturin==1.15.0 maturin sdist' in artifacts
    assert '--from maturin==1.15.0 maturin sdist' in cuda
    assert '--from maturin==1.15.0 maturin build' in cuda


@pytest.mark.parametrize('filename', ['artifacts.yml', 'macos.yml'])
def test_macos_build_uses_fixed_maturin_after_dependency_sync(filename):
    workflow = (ROOT / '.github/workflows' / filename).read_text()
    job = workflow.split('  macos-wheel:\n')[1].split('  assemble:\n')[0] if filename == 'artifacts.yml' else workflow
    sync = 'uv sync --group dev --group bench --no-install-project'
    pin = 'uv pip install --python .venv/bin/python maturin==1.15.0'
    build = 'maturin" build --release'
    assert job.index(sync) < job.index(pin) < job.index(build)


def test_manylinux_checks_the_build_interpreter_inside_the_container():
    workflow = (ROOT / '.github/workflows/artifacts.yml').read_text()
    job = workflow.split('  wheel:\n')[1].split('  macos-wheel:\n')[0]
    before = job.split('          before-script-linux: |\n')[1].split('          working-directory:')[0]
    assert 'set -e' in before
    assert "python_abi=\"cp$(printf '%s' '${{ matrix.python }}' | tr -d '.')\"" in before
    assert 'export PATH="/opt/python/$python_abi-$python_abi/bin:$PATH"' in before
    assert "python${{ matrix.python }} -c 'import sys, sysconfig; print(sys.version);" in before
    assert 'sys.implementation.name == "cpython"' in before
    assert 'sys.version_info.releaselevel == "final"' in before
    assert 'not sysconfig.get_config_var("Py_GIL_DISABLED")' in before
    assert '--interpreter python${{ matrix.python }}' in job


def test_release_host_fetch_reuses_shared_sdist_lock_before_offline_fixtures():
    workflow = (ROOT / '.github/workflows/artifacts.yml').read_text()
    job = workflow.split('  wheel:\n')[1].split('  macos-wheel:\n')[0]
    step, = (step for step in job.split('      - name:') if 'cargo fetch --locked' in step)
    assert "if: matrix.python == '3.12' || matrix.python == '3.15'" in step
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
@pytest.mark.parametrize('maintenance', ['success', 'failure', 'cancelled', 'skipped'])
def test_ci_result_fails_closed(tmp_path, classification, run_cpu, tests, typing, ok, maintenance):
    workflow = (ROOT / '.github/workflows/ci.yml').read_text()
    gate = workflow.split('  result:\n')[1]
    assert 'if: always()' in gate
    assert 'needs: [changes, maintenance, test, typecheck]' in gate
    script = textwrap.dedent(gate.split('        run: |\n')[1])
    env = dict(os.environ, MAINTENANCE_RESULT=maintenance,
               CHANGES_RESULT=classification, RUN_CPU=run_cpu,
               TEST_RESULT=tests, TYPECHECK_RESULT=typing,
               GITHUB_STEP_SUMMARY=str(tmp_path / 'summary'))
    result = subprocess.run(['bash', '-e', '-c', script], env=env, capture_output=True)
    assert (result.returncode == 0) is (ok and maintenance == 'success')


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
    assert "python: ['3.12', '3.13', '3.14', '3.15']" in text
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
    assert 'cp312-cp312-*.whl' not in workflow
    assert '${abi}-${abi}-*.whl' not in workflow
    assert "directory.glob('tensor0-*-cp312-cp312-*linux*_x86_64.whl')" in workflow
    assert 'for wheel in "$RUNNER_TEMP"/cuda-dist/*_x86_64.whl; do auditwheel show' in workflow
    assert './tensor0-*-${abi}-${abi}-manylinux_2_28_x86_64.whl' in workflow
