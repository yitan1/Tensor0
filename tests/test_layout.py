import pytest

import tensor0
from tensor0 import (
    SU2Irrep,
    U1Irrep,
    U1SU2Irrep,
    _native,
    get_degeneracystructure,
    get_sectorstructure,
    hom,
    space,
)


def block_spans(degeneracystructure):
    return [
        (block.row_dim, block.col_dim, block.start, block.stop)
        for block in degeneracystructure.blockstructure
    ]


def block_span(block):
    return (block.row_dim, block.col_dim, block.start, block.stop)


def test_native_layout_builders_return_sector_and_degeneracy_structures():
    v = space(U1Irrep, {0: 2, 1: 3})
    h = hom((v,), (v,))

    sectorstructure = _native.build_sectorstructure(h)
    degeneracystructure = _native.build_degeneracystructure(h)

    assert sectorstructure.blocksectors == ((0,), (1,))
    assert len(sectorstructure.fusiontree_pairs) == 2
    assert degeneracystructure.total_dim == 13
    assert block_spans(degeneracystructure) == [(2, 2, 0, 4), (3, 3, 4, 13)]
    assert sectorstructure.blocksector_index(0) == 0
    assert sectorstructure.blocksector_index((1,)) == 1
    assert sectorstructure.blocksector_index(2) is None

    with pytest.raises(TypeError, match="sector key"):
        sectorstructure.blocksector_index("bad")

    row, col = sectorstructure.fusiontree_pairs[0]
    assert row.uncoupled == ((0,),)
    assert row.coupled == (0,)
    assert row.is_dual == (False,)
    assert col.static_key == row.static_key


def test_native_su2_layout_exposes_multileg_fusiontree_metadata():
    half = space(SU2Irrep, {1: 1})
    h = hom((half, half, half, half), ())

    sectorstructure = _native.build_sectorstructure(h)
    degeneracystructure = _native.build_degeneracystructure(h)

    assert sectorstructure.blocksectors == ((0,),)
    assert len(sectorstructure.fusiontree_pairs) == 2
    assert degeneracystructure.blockstructure[0].row_dim == 2
    assert degeneracystructure.blockstructure[0].col_dim == 1
    assert degeneracystructure.total_dim == 2

    expected_innerlines = [((0,), (1,)), ((2,), (1,))]
    for (row, col), innerlines in zip(
        sectorstructure.fusiontree_pairs,
        expected_innerlines,
    ):
        assert row.uncoupled == ((1,), (1,), (1,), (1,))
        assert row.coupled == (0,)
        assert row.innerlines == innerlines
        assert row.vertices == (0, 0, 0)
        assert col.uncoupled == ()
        assert col.coupled == (0,)


def test_native_sectorstructure_blocksector_index_supports_product_sector_keys():
    v = space(U1SU2Irrep, {(0, 0): 2, (0, 1): 3})
    h = hom((v,), (v,))
    sectorstructure = _native.build_sectorstructure(h)

    assert sectorstructure.blocksectors == ((0, 0), (0, 1))
    assert sectorstructure.blocksector_index((0, 0)) == 0
    assert sectorstructure.blocksector_index((0, 1)) == 1
    assert sectorstructure.blocksector_index((0, 2)) is None

    with pytest.raises(ValueError, match="width"):
        sectorstructure.blocksector_index((0,))


def test_get_blockstructure_maps_blocksectors_to_degeneracy_blocks():
    v = space(U1Irrep, {1: 3, 0: 2})
    h = hom((v,), (v,))
    degeneracystructure = get_degeneracystructure(h)

    blockstructure = tensor0.structure.get_blockstructure(h)

    assert tuple(blockstructure) == ((0,), (1,))
    assert block_span(blockstructure[0]) == block_span(
        degeneracystructure.blockstructure[0],
    )
    assert block_span(blockstructure[(1,)]) == block_span(
        degeneracystructure.blockstructure[1],
    )
    assert blockstructure.get(2) is None  # pyright: ignore[reportArgumentType]

    items = blockstructure.items()
    values = blockstructure.values()
    item_tuple = tuple(items)
    value_tuple = tuple(values)
    assert tuple((sector, block_span(block)) for sector, block in item_tuple) == (
        ((0,), block_span(degeneracystructure.blockstructure[0])),
        ((1,), block_span(degeneracystructure.blockstructure[1])),
    )
    assert tuple(block_span(block) for block in value_tuple) == tuple(
        block_span(block) for block in degeneracystructure.blockstructure
    )


def test_get_structure_functions_reject_non_hom_space_inputs():
    with pytest.raises(TypeError, match="^get_sectorstructure\\(\\) requires a HomSpace$"):
        get_sectorstructure(object())  # pyright: ignore[reportArgumentType]

    with pytest.raises(TypeError, match="^get_degeneracystructure\\(\\) requires a HomSpace$"):
        get_degeneracystructure(object())  # pyright: ignore[reportArgumentType]
