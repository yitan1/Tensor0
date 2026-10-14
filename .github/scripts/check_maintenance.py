"""Small maintenance gates; no third-party dependencies.

Only static Python imports are checked, not dynamic imports or filesystem access.
"""

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
# Entry points, not general documentation limits. Changes require human review.
DOCUMENT_BUDGETS = {'README.md': 8 * 1024, 'CONTRIBUTING.md': 4 * 1024}
FORBIDDEN_IMPORTS = {'tests', 'examples', 'benchmarks', 'local'}


def check_document(path, limit):
    size = path.stat().st_size
    if size > limit:
        return [f'{path}: DOC001: {size} bytes exceeds the {limit}-byte entry-point '
                'budget. Move detail to an appropriate docs/ page and link it; '
                'do not raise the budget merely to pass CI.']
    return []


def check_imports(path):
    try:
        tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    except SyntaxError as error:
        return [f'{path}:{error.lineno}: ARCH000: {error.msg}. Fix Python syntax.']
    errors = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            modules = [node.module or '']
        else:
            continue
        for module in modules:
            if module.split('.')[0] in FORBIDDEN_IMPORTS:
                errors.append(
                    f'{path}:{node.lineno}: ARCH001: production code imports {module}. '
                    'Keep test/example helpers outside the package; move required '
                    'runtime functionality into the production package.')
    return errors


def check_repository(root):
    errors = []
    for name, limit in DOCUMENT_BUDGETS.items():
        errors.extend(check_document(root / name, limit))
    for path in sorted((root / 'src/tensor0').rglob('*')):
        if path.suffix in {'.py', '.pyi'}:
            errors.extend(check_imports(path))
    return errors


def main():
    errors = check_repository(ROOT)
    for error in errors:
        print(error)
    if not errors:
        print('Maintenance checks passed (entry-point budgets and static Python imports).')
    return bool(errors)


if __name__ == '__main__':
    raise SystemExit(main())
