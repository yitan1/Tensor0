import importlib

import pytest

import tensor0
from tensor0 import U1Irrep, space


def _sector_dict_type():
    structure = importlib.import_module("tensor0.structure")
    assert hasattr(structure, "SectorDict")
    return structure.SectorDict


def _sector_index_module():
    return importlib.import_module("tensor0.tensor._sector_index")


def test_sector_dict_preserves_input_order_and_normalizes_lookup_keys():
    SectorDict = _sector_dict_type()
    sectors = SectorDict([(1, "one"), (0, "zero"), ((-1,), "minus")])

    assert list(sectors) == [(1,), (0,), (-1,)]
    assert list(sectors.items()) == [
        ((1,), "one"),
        ((0,), "zero"),
        ((-1,), "minus"),
    ]
    assert sectors[1] == "one"
    assert sectors[(0,)] == "zero"
    assert sectors.get(-1) == "minus"
    assert sectors.get(2, "missing") == "missing"
    with pytest.raises(KeyError):
        sectors[2]


def test_sector_dict_accepts_mapping_inputs():
    SectorDict = _sector_dict_type()
    sectors = SectorDict({1: "one", 0: "zero"})

    assert list(sectors.items()) == [
        ((1,), "one"),
        ((0,), "zero"),
    ]


def test_sector_dict_rejects_duplicate_normalized_keys_and_invalid_keys():
    SectorDict = _sector_dict_type()

    with pytest.raises(ValueError, match="sector appears multiple times"):
        SectorDict([(0, "int"), ((0,), "tuple")])

    with pytest.raises(TypeError, match="sector key"):
        SectorDict([("bad", "value")])


def test_sector_dict_is_available_from_structure_but_not_top_level():
    SectorDict = _sector_dict_type()

    assert SectorDict is tensor0.structure.SectorDict
    assert "SectorDict" not in tensor0.__all__
    assert not hasattr(tensor0, "SectorDict")


def test_sector_slices_for_space_use_native_canonical_order():
    SectorDict = _sector_dict_type()
    sector_index = _sector_index_module()
    vector_space = space(U1Irrep, {1: 3, 0: 2})

    sectors = sector_index._sector_slices_for_space(vector_space)

    assert isinstance(sectors, SectorDict)
    assert list(sectors.items()) == [
        ((0,), slice(0, 2)),
        ((1,), slice(2, 5)),
    ]
    assert sectors[0] == slice(0, 2)
    assert sectors[(1,)] == slice(2, 5)


def test_sector_slices_cache_reuses_same_visible_layout_for_dual_spaces():
    sector_index = _sector_index_module()
    sector_index._clear_sector_index_caches_for_tests()
    dual_space = space(U1Irrep, {1: 2}, dual=True)
    nondual_space = space(U1Irrep, {-1: 2})

    assert dual_space.sectors == nondual_space.sectors
    assert sector_index._sector_slices_for_space(dual_space) is (
        sector_index._sector_slices_for_space(nondual_space)
    )
