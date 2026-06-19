import importlib

import pytest

import tensor0


def _sector_dict_type():
    structure = importlib.import_module("tensor0.structure")
    assert hasattr(structure, "SectorDict")
    return structure.SectorDict


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


def test_sector_dict_is_available_from_structure_and_top_level():
    SectorDict = _sector_dict_type()

    assert SectorDict is tensor0.structure.SectorDict
    assert SectorDict is tensor0.SectorDict
    assert "SectorDict" in tensor0.__all__
