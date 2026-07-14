import ast
from pathlib import Path
import runpy

import pytest
import tensor0


ROOT = Path(__file__).resolve().parents[1]
BASIC_EXAMPLE = ROOT / "examples" / "basic_usage.py"
CONTRACTION_EXAMPLE = ROOT / "examples" / "contractions.py"
EXAMPLE_CASES = (
    (BASIC_EXAMPLE, "Tensor0 basic usage OK"),
    (CONTRACTION_EXAMPLE, "Tensor0 contraction examples OK"),
)


def _assert_uses_public_api_only(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    public_names = set(tensor0.__all__)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("tensor0."):
                    raise AssertionError(
                        f"{path} imports non-public module {alias.name!r}"
                    )
        elif isinstance(node, ast.ImportFrom):
            if node.module == "tensor0":
                imported_names = {alias.name for alias in node.names}
                assert imported_names <= public_names
            elif node.module is not None and node.module.startswith("tensor0."):
                raise AssertionError(
                    f"{path} imports non-public module {node.module!r}"
                )


@pytest.mark.parametrize(
    ("path", "success_message"),
    EXAMPLE_CASES,
    ids=("basic_usage", "contractions"),
)
def test_public_example_uses_public_api_and_runs(
    path: Path,
    success_message: str,
    capsys,
) -> None:
    _assert_uses_public_api_only(path)
    runpy.run_path(str(path), run_name="__main__")

    assert success_message in capsys.readouterr().out
