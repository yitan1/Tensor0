"""Offline maintenance-rule tests runnable with the standard library."""

import importlib.util
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    'check_maintenance', ROOT / '.github/scripts/check_maintenance.py')
maintenance = importlib.util.module_from_spec(spec)
spec.loader.exec_module(maintenance)


class MaintenanceTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def test_document_budget_counts_bytes_and_accepts_exact_limit(self):
        path = self.root / 'README.md'
        path.write_text('éé', encoding='utf-8')
        self.assertEqual(maintenance.check_document(path, 4), [])
        errors = maintenance.check_document(path, 3)
        self.assertEqual(len(errors), 1)
        self.assertIn('DOC001', errors[0])
        self.assertIn('Move detail', errors[0])

    def test_forbidden_static_imports(self):
        path = self.root / 'module.py'
        for statement in (
            'import tests.support as support',
            'import numpy, examples.basic_usage',
            'from benchmarks import helper',
            'from local.reference import helper',
            'if False:\n    import tests',
        ):
            with self.subTest(statement=statement):
                path.write_text(statement, encoding='utf-8')
                errors = maintenance.check_imports(path)
                self.assertEqual(len(errors), 1)
                self.assertIn('ARCH001', errors[0])
                self.assertIn(str(path), errors[0])

    def test_relative_imports_comments_and_similar_names_are_allowed(self):
        path = self.root / 'module.py'
        path.write_text(
            'from . import tensor\nfrom .local import helper\n'
            'import local_helper\nimport tensor0\n'
            '# import tests\ntext = "import tests"\n',
            encoding='utf-8')
        self.assertEqual(maintenance.check_imports(path), [])

    def test_syntax_errors_fail(self):
        path = self.root / 'module.py'
        path.write_text('import (', encoding='utf-8')
        self.assertIn('ARCH000', maintenance.check_imports(path)[0])

    def test_repository_checks_stubs_but_not_test_helpers(self):
        for name in maintenance.DOCUMENT_BUDGETS:
            (self.root / name).write_text('Entry point', encoding='utf-8')
        package = self.root / 'src/tensor0'
        package.mkdir(parents=True)
        (package / 'native.pyi').write_text('import tests', encoding='utf-8')
        tests = self.root / 'tests'
        tests.mkdir()
        (tests / 'helper.py').write_text('import tests', encoding='utf-8')
        errors = maintenance.check_repository(self.root)
        self.assertEqual(len(errors), 1)
        self.assertIn('native.pyi:1:', errors[0])

    def test_gate_runs_without_change_classification(self):
        workflow = (ROOT / '.github/workflows/ci.yml').read_text()
        job = workflow.split('  maintenance:\n')[1].split('  changes:\n')[0]
        self.assertNotIn('needs:', job)
        self.assertNotIn('if:', job)
        self.assertIn('python3 .github/scripts/check_maintenance.py', job)
        self.assertIn('python3 -m unittest discover', job)


if __name__ == '__main__':
    unittest.main()
