from pathlib import Path
import re

import pytest
import tensor0
import tensor0.factorizations as factorizations
import tensor0.operations as operations
import tensor0.structure as structure
import tensor0.tensor as tensor


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


def test_phase_9_public_names_have_one_canonical_surface():
    for name in ("identity", "isomorphism", "unitary", "isometry", "random_normal"):
        assert getattr(tensor0, name) is getattr(tensor, name)
    assert tensor0.is_diagonal is tensor.is_diagonal
    assert tensor0.tensor_product is tensor.tensor_product
    assert not hasattr(operations, "tensor_product")

    assert not hasattr(tensor0, "id")
    assert not hasattr(tensor0, "isdiag")
    assert not hasattr(tensor0, "randn")
    assert not hasattr(tensor0.TensorMap, "random_normal")

    for name in (
        "qr_compact",
        "lq_compact",
        "left_orth",
        "right_orth",
        "eigh_vals",
        "eigh_full",
        "is_hermitian",
    ):
        assert getattr(tensor0, name) is getattr(factorizations, name)


def test_phase_10_public_names_have_one_canonical_surface():
    for name in (
        "inverse",
        "pseudoinverse",
        "left_solve",
        "right_solve",
        "is_isometric",
        "is_unitary",
        "is_positive_definite",
        "random_isometry",
    ):
        assert getattr(tensor0, name) is getattr(tensor, name)
    assert tensor0.eigh_trunc is factorizations.eigh_trunc

    for alias in (
        "inv",
        "pinv",
        "solve",
        "randisometry",
        "isisometric",
        "isunitary",
        "isposdef",
    ):
        assert not hasattr(tensor0, alias)


def _documented_inventory(document: str, marker: str) -> set[str]:
    start = f"<!-- {marker}:start -->"
    end = f"<!-- {marker}:end -->"
    section = document.split(start, maxsplit=1)[1].split(end, maxsplit=1)[0]
    names = re.findall(r"^- `([^`]+)`$", section, flags=re.MULTILINE)
    assert len(names) == len(set(names))
    return set(names)
