"""Classify daily CI changes conservatively, without installing Tensor0."""

import json
import os
from pathlib import Path
import subprocess


DOC_FILES = {'README.md', 'CHANGELOG.md', 'CONTRIBUTING.md', 'CITATION.cff', 'mkdocs.yml'}


def requires_cpu(paths):
    # Unknown paths are build/test inputs until explicitly reviewed otherwise.
    return not paths or any(not (path.startswith('docs/') or path in DOC_FILES)
                            for path in paths)


def classify(event_name, event):
    if event_name == 'workflow_dispatch':
        return True, 'Manual dispatch forces full CPU verification'
    if event_name == 'pull_request':
        base = event['pull_request']['base']['sha']
        head = event['pull_request']['head']['sha']
    elif event_name == 'push':
        base, head = event['before'], event['after']
    else:
        return True, 'Unknown event: require full CPU verification'
    if not base or base == '0' * 40:
        return True, 'No previous commit: require full CPU verification'
    try:
        # Full event range, not HEAD~1; both sides of renames remain inputs.
        data = subprocess.check_output(
            ['git', 'diff', '--name-only', '-z', '--no-renames', base, head, '--'])
    except subprocess.CalledProcessError:
        return True, 'Diff unavailable: require full CPU verification'
    paths = [os.fsdecode(path) for path in data.split(b'\0') if path]
    run_cpu = requires_cpu(paths)
    return run_cpu, ('Code/build/test or empty diff: require full CPU verification'
                     if run_cpu else 'Documentation-only: CPU tests and Pyright not run')


def main():
    event = json.loads(Path(os.environ['GITHUB_EVENT_PATH']).read_text())
    run_cpu, reason = classify(os.environ['GITHUB_EVENT_NAME'], event)
    with open(os.environ['GITHUB_OUTPUT'], 'a') as output:
        output.write(f'run_cpu={str(run_cpu).lower()}\n')
    with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as summary:
        summary.write(f'## Daily CI scope\n\n{reason}.\n')
    print(reason)


if __name__ == '__main__':
    main()
