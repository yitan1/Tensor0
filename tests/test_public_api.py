from pathlib import Path
import re

import pytest
import tensor0
import tensor0.structure as structure


API_DOCUMENT = (Path(__file__).parents[1] / "docs" / "api.md").read_text(
    encoding="utf-8",
)

ADVANCED_STRUCTURE_NAMES = {
    "BlockStructure",
    "DegeneracyStructure",
    "SectorStructure",
    "SubblockStructure",
    "get_blockstructure",
    "get_degeneracystructure",
    "get_sectorstructure",
}


@pytest.mark.parametrize(
    ("module", "marker"),
    [
        (tensor0, "tensor0-root-api"),
        (structure, "tensor0-structure-api"),
    ],
    ids=["root", "structure"],
)
def test_public_api_matches_inventory(module, marker):
    inventory = _documented_inventory(API_DOCUMENT, marker)
    exports = module.__all__

    assert len(exports) == len(set(exports))
    assert set(exports) == inventory
    assert all(hasattr(module, name) for name in inventory)


def test_advanced_structure_names_are_not_root_exports():
    assert not any(hasattr(tensor0, name) for name in ADVANCED_STRUCTURE_NAMES)


def test_public_productspace_and_advanced_layout_return_types():
    factor = tensor0.space(tensor0.U1Irrep, {0: 2})
    hom_space = tensor0.hom((factor,), (factor,))
    degeneracy = structure.get_degeneracystructure(hom_space)
    blocks = structure.get_blockstructure(hom_space)

    assert isinstance(hom_space.codomain, tensor0.ProductSpace)
    assert isinstance(degeneracy, structure.DegeneracyStructure)
    assert isinstance(blocks[0], structure.BlockStructure)
    assert isinstance(degeneracy.subblock_at(0), structure.SubblockStructure)


def _documented_inventory(document: str, marker: str) -> set[str]:
    start = f"<!-- {marker}:start -->"
    end = f"<!-- {marker}:end -->"
    section = document.split(start, maxsplit=1)[1].split(end, maxsplit=1)[0]
    names = re.findall(r"^- `([^`]+)`$", section, flags=re.MULTILINE)
    assert len(names) == len(set(names))
    return set(names)
