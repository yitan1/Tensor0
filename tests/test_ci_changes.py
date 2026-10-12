"""Offline daily CI scope and aggregate-gate tests; no native build."""

import importlib.util
import os
from pathlib import Path
import re
import subprocess
import textwrap

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('ci_changes', ROOT / '.github/scripts/ci_changes.py')
changes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(changes)


@pytest.mark.parametrize('paths', [
    ['docs/cuda.md'], ['docs/assets/diagram.svg', 'mkdocs.yml'],
    ['README.md', 'CHANGELOG.md', 'CONTRIBUTING.md'], ['docs/a\nb.md'],
])
def test_documentation_only(paths):
    assert not changes.requires_cpu(paths)


@pytest.mark.parametrize('path', [
    'src/tensor0/__init__.py', 'crates/tensor0-py/native/ffi/copy.cc',
    'tests/test_ci_changes.py', 'examples/basic_usage.py', 'benchmarks/standard/run.py',
    'pyproject.toml', 'Cargo.toml', 'Cargo.lock', 'LICENSE', 'unknown.md',
    '.github/workflows/ci.yml', '.github/scripts/ci_changes.py',
])
def test_unknown_or_code_paths_require_full_ci(path):
    assert changes.requires_cpu(['docs/cuda.md', path])


def test_empty_diff_requires_full_ci():
    assert changes.requires_cpu([])


@pytest.mark.parametrize('event_name', ['push', 'pull_request'])
def test_exact_event_range_and_nul_paths(monkeypatch, event_name):
    base, head = 'a' * 40, 'b' * 40
    event = ({'before': base, 'after': head} if event_name == 'push' else
             {'pull_request': {'base': {'sha': base}, 'head': {'sha': head}}})
    calls = []

    def diff(command):
        calls.append(command)
        return b'docs/first.md\0docs/second\nfile.md\0'

    monkeypatch.setattr(changes.subprocess, 'check_output', diff)
    assert changes.classify(event_name, event)[0] is False
    assert calls == [['git', 'diff', '--name-only', '-z', '--no-renames', base, head, '--']]


def test_earlier_code_in_multi_commit_push_is_not_ignored(monkeypatch):
    monkeypatch.setattr(changes.subprocess, 'check_output',
                        lambda command: b'src/tensor0/__init__.py\0docs/latest.md\0')
    assert changes.classify('push', {'before': 'a' * 40, 'after': 'c' * 40})[0]


@pytest.mark.parametrize('paths', [
    b'src/old.py\0docs/new.md\0',  # Code renamed into docs.
    b'docs/old.md\0src/new.py\0',  # Docs renamed into code.
    b'src/deleted.py\0',
])
def test_renames_and_deletions_require_cpu(monkeypatch, paths):
    monkeypatch.setattr(changes.subprocess, 'check_output', lambda command: paths)
    assert changes.classify('push', {'before': 'a' * 40, 'after': 'b' * 40})[0]


@pytest.mark.parametrize('event_name,event', [
    ('workflow_dispatch', {}), ('unknown', {}),
    ('push', {'before': '0' * 40, 'after': 'b' * 40}),
])
def test_manual_and_unknown_ranges_force_cpu(monkeypatch, event_name, event):
    def no_diff(command):
        pytest.fail('No diff is needed')
    monkeypatch.setattr(changes.subprocess, 'check_output', no_diff)
    assert changes.classify(event_name, event)[0]


def test_unavailable_diff_forces_cpu(monkeypatch):
    def unavailable(command):
        raise subprocess.CalledProcessError(128, command)
    monkeypatch.setattr(changes.subprocess, 'check_output', unavailable)
    assert changes.classify('push', {'before': 'a' * 40, 'after': 'b' * 40})[0]


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
