import jax.numpy as jnp
import pytest

import tensor0
import tensor0.structure.layout as layout_module
from tensor0 import (
    ComplexSpace,
    SU2Irrep,
    TensorMap,
    U1Irrep,
    U1SU2Irrep,
    Z2Irrep,
    _native,
    from_dense,
    hom,
    identity,
    space,
    storage_dim,
    svd_compact,
    to_dense,
)
from tensor0.structure import get_degeneracystructure, get_sectorstructure


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
    assert sectorstructure.fusiontree_pair_count == 2
    assert block_spans(degeneracystructure) == [(2, 2, 0, 4), (3, 3, 4, 13)]
    assert sectorstructure.blocksector_index(0) == 0
    assert sectorstructure.blocksector_index((1,)) == 1
    assert sectorstructure.blocksector_index(2) is None

    with pytest.raises(TypeError, match="sector key"):
        sectorstructure.blocksector_index("bad")

    row, col = sectorstructure.fusiontree_pairs[0]
    same_pair = sectorstructure.fusiontree_pair_at(0)
    assert same_pair is not None
    same_row, same_col = same_pair
    assert row.uncoupled == ((0,),)
    assert row.coupled == (0,)
    assert row.is_dual == (False,)
    assert col.static_key == row.static_key
    assert (same_row, same_col) == (row, col)
    assert {(row, col): "block"}[(same_row, same_col)] == "block"
    assert row != object()
    assert sectorstructure.fusiontree_pair_index(same_row, same_col) == 0
    assert sectorstructure.fusiontree_pair_at(2) is None
    same_subblock = degeneracystructure.subblock_at(0)
    assert same_subblock is not None
    assert same_subblock.static_key == degeneracystructure.subblockstructure[0].static_key
    assert degeneracystructure.subblock_at(2) is None


def test_native_unique_fusiontree_pair_index_uses_visible_domain_sectors():
    v = space(U1Irrep, {0: 2, 1: 3})
    h = hom((v,), (v,))
    sectorstructure = _native.build_sectorstructure(h)

    assert _native.unique_fusiontree_pair_index(h, sectorstructure, (0, 0)) == 0
    assert _native.unique_fusiontree_pair_index(h, sectorstructure, (1, -1)) == 1
    assert _native.unique_fusiontree_pair_index(h, sectorstructure, (1, 0)) is None

    other = hom((space(U1Irrep, {0: 1}),), (space(U1Irrep, {0: 1}),))
    with pytest.raises(ValueError, match="does not match HomSpace"):
        _native.unique_fusiontree_pair_index(
            h,
            _native.build_sectorstructure(other),
            (0, 0),
        )


def test_native_su2_layout_exposes_multileg_fusiontree_metadata():
    half = space(SU2Irrep, {1: 1})
    h = hom((half, half, half, half), ())

    sectorstructure = _native.build_sectorstructure(h)
    degeneracystructure = _native.build_degeneracystructure(h)

    assert sectorstructure.blocksectors == ((0,),)
    assert len(sectorstructure.fusiontree_pairs) == 2
    assert degeneracystructure.blockstructure[0].row_dim == 2
    assert degeneracystructure.blockstructure[0].col_dim == 1

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
    equivalent = hom((v,), (v,))
    degeneracystructure = get_degeneracystructure(h)

    blockstructure = tensor0.structure.get_blockstructure(h)
    repeated = tensor0.structure.get_blockstructure(h)
    equivalent_blockstructure = tensor0.structure.get_blockstructure(equivalent)

    assert degeneracystructure is get_degeneracystructure(equivalent)
    assert not hasattr(layout_module, "_blockstructure_cache")
    expected_items = (
        ((0,), block_span(degeneracystructure.blockstructure[0])),
        ((1,), block_span(degeneracystructure.blockstructure[1])),
    )
    for candidate in (blockstructure, repeated, equivalent_blockstructure):
        assert tuple(
            (sector, block_span(block)) for sector, block in candidate.items()
        ) == expected_items

    assert tuple(blockstructure) == ((0,), (1,))
    assert tuple(block_span(block) for block in blockstructure.values()) == tuple(
        block_span(block) for block in degeneracystructure.blockstructure
    )
    assert blockstructure.get(2) is None  # pyright: ignore[reportArgumentType]
    with pytest.raises(KeyError, match=r"\(2,\)"):
        blockstructure[(2,)]
    with pytest.raises(ValueError, match="width"):
        blockstructure[(0, 1)]

    assert block_span(blockstructure[0]) == block_span(
        degeneracystructure.blockstructure[0],
    )
    assert block_span(blockstructure[(1,)]) == block_span(
        degeneracystructure.blockstructure[1],
    )


def test_get_structure_functions_reject_non_hom_space_inputs():
    with pytest.raises(TypeError, match="^get_sectorstructure\\(\\) requires a HomSpace$"):
        get_sectorstructure(object())  # pyright: ignore[reportArgumentType]

    with pytest.raises(TypeError, match="^get_degeneracystructure\\(\\) requires a HomSpace$"):
        get_degeneracystructure(object())  # pyright: ignore[reportArgumentType]


def test_internal_block_paths_do_not_construct_public_mapping_wrapper(monkeypatch):
    factor = space(U1Irrep, {0: 2, 1: 3})
    target = hom((factor,), (factor,))
    tensor = TensorMap(
        target,
        jnp.arange(storage_dim(target), dtype=jnp.float32),
    )
    blocks = tensor.blocks()

    def reject_wrapper(*_args, **_kwargs):
        raise AssertionError("internal path constructed _IndexedMapping")

    layout_module._clear_layout_caches_for_tests()
    monkeypatch.setattr(layout_module, "_IndexedMapping", reject_wrapper)

    assert tensor.block(0).shape == (2, 2)
    assert len(tensor.blocks()) == 2
    assert TensorMap.from_blocks(target, blocks).space == target
    assert identity(factor).space == target
    assert svd_compact(tensor)[0].space.codomain == target.codomain
    assert (tensor @ tensor).space == target


def test_sectorstructure_key_tracks_visible_layout_not_degeneracy():
    factor = space(U1Irrep, {0: 2, 1: 3})
    same_support = space(U1Irrep, {0: 5, 1: 7})
    source = hom((factor,), ())
    key = layout_module._sectorstructure_key(source)

    assert key == layout_module._sectorstructure_key(hom((same_support,), ()))
    assert key != layout_module._sectorstructure_key(
        hom((space(U1Irrep, {0: 2, 2: 3}),), ())
    )
    assert key != layout_module._sectorstructure_key(hom((factor.dual(),), ()))
    assert key != layout_module._sectorstructure_key(hom((), (factor,)))
    assert key != layout_module._sectorstructure_key(
        hom((space(Z2Irrep, {0: 2, 1: 3}),), ())
    )


def test_sectorstructure_cache_separates_typed_empty_sector_families():
    u1_empty = _native.make_product_space(U1Irrep, ())
    su2_empty = _native.make_product_space(SU2Irrep, ())
    u1_hom = _native.make_hom_products(u1_empty, u1_empty)
    su2_hom = _native.make_hom_products(su2_empty, su2_empty)

    layout_module._clear_layout_caches_for_tests()
    try:
        u1_structure = get_sectorstructure(u1_hom)
        su2_structure = get_sectorstructure(su2_hom)

        assert u1_structure is get_sectorstructure(u1_hom)
        assert su2_structure is get_sectorstructure(su2_hom)
        assert u1_structure is not su2_structure
    finally:
        layout_module._clear_layout_caches_for_tests()


def test_trivial_layout_and_dense_roundtrip_are_contiguous():
    left = ComplexSpace(2)
    middle = ComplexSpace(3)
    right = ComplexSpace(4)
    target = hom((left, middle), (right,))
    dense = jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4)

    tensor = from_dense(target, dense)
    sectorstructure = get_sectorstructure(target)
    degeneracystructure = get_degeneracystructure(target)

    assert sectorstructure.blocksectors == ((),)
    assert sectorstructure.fusiontree_pair_count == 1
    assert degeneracystructure.total_dim == 24
    assert storage_dim(target) == 24
    assert tensor.block(()).shape == (6, 4)
    assert tensor.subblock(sectorstructure.fusiontree_pairs[0]).shape == (2, 3, 4)
    assert jnp.array_equal(to_dense(tensor), dense)
