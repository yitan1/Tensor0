import ast
from pathlib import Path
import runpy

import tensor0


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "basic_usage.py"
ALLOWED_IMPORTS = {"jax"}
ALLOWED_FROM_IMPORTS = {"__future__", "tensor0"}


def test_basic_usage_example_uses_public_api_only():
    source = EXAMPLE.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(EXAMPLE))
    public_names = set(tensor0.__all__)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                assert root in ALLOWED_IMPORTS
        elif isinstance(node, ast.ImportFrom):
            assert node.module in ALLOWED_FROM_IMPORTS
            if node.module == "__future__":
                assert {alias.name for alias in node.names} == {"annotations"}
            elif node.module == "tensor0":
                imported_names = {alias.name for alias in node.names}
                assert imported_names <= public_names


def test_basic_usage_example_runs_as_script(capsys):
    runpy.run_path(str(EXAMPLE), run_name="__main__")

    captured = capsys.readouterr()
    assert "Tensor0 basic usage OK" in captured.out
