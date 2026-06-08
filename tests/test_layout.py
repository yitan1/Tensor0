import pytest

import tensor0.structure.layout as layout_module
from tensor0 import (
    SU2Irrep,
    U1Irrep,
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


def test_native_layout_builders_return_sector_and_degeneracy_structures():
    v = space(U1Irrep, {0: 2, 1: 3})
    h = hom((v,), (v,))

    sectorstructure = _native.build_sectorstructure(h)
    degeneracystructure = _native.build_degeneracystructure(h)

    assert sectorstructure.blocksectors == ((0,), (1,))
    assert len(sectorstructure.fusiontree_pairs) == 2
    assert degeneracystructure.total_dim == 13
    assert block_spans(degeneracystructure) == [(2, 2, 0, 4), (3, 3, 4, 13)]

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


def test_native_checked_degeneracy_fast_path_reuses_sectorstructure_across_degeneracies():
    cached_space = hom(
        (space(U1Irrep, {0: 2, 1: 3}),),
        (space(U1Irrep, {0: 2, 1: 3}),),
    )
    target_space = hom(
        (space(U1Irrep, {0: 5, 1: 7}),),
        (space(U1Irrep, {0: 5, 1: 7}),),
    )
    sectorstructure = _native.build_sectorstructure(cached_space)

    degeneracystructure = _native._build_degeneracystructure_from_sectorstructure(
        target_space,
        sectorstructure,
    )

    assert degeneracystructure.total_dim == 74
    assert block_spans(degeneracystructure) == [(5, 5, 0, 25), (7, 7, 25, 74)]


@pytest.mark.parametrize(
    ("target_space", "message"),
    [
        (
            hom(
                (space(U1Irrep, {0: 3, 1: 4}),),
                (space(U1Irrep, {0: 3, 1: 4}),),
            ),
            "sectorstructure does not match HomSpace sector structure",
        ),
        (
            hom((space(SU2Irrep, {0: 3}),), (space(SU2Irrep, {0: 3}),)),
            "sectorstructure must match HomSpace sector family",
        ),
    ],
    ids=["stale-sectorstructure", "cross-family-sectorstructure"],
)
def test_native_checked_degeneracy_fast_path_rejects_incompatible_sectorstructure(
    target_space,
    message,
):
    cached_space = hom((space(U1Irrep, {0: 2}),), (space(U1Irrep, {0: 2}),))
    sectorstructure = _native.build_sectorstructure(cached_space)

    with pytest.raises(ValueError, match=message):
        _native._build_degeneracystructure_from_sectorstructure(
            target_space,
            sectorstructure,
        )


def test_get_structure_functions_return_native_structures():
    v = space(U1Irrep, {0: 2, 1: 3})
    h = hom((v,), (v,))

    sectorstructure = get_sectorstructure(h)
    degeneracystructure = get_degeneracystructure(h)

    assert isinstance(sectorstructure, _native.SectorStructure)
    assert sectorstructure.blocksectors == ((0,), (1,))
    assert len(sectorstructure.fusiontree_pairs) == 2
    row, col = sectorstructure.fusiontree_pairs[0]
    assert isinstance(row, _native.FusionTree)
    assert isinstance(col, _native.FusionTree)
    assert row.uncoupled == ((0,),)
    assert col.static_key == row.static_key

    assert isinstance(degeneracystructure, _native.DegeneracyStructure)
    assert len(degeneracystructure.blockstructure) == 2
    assert all(
        isinstance(block, _native.BlockStructure)
        for block in degeneracystructure.blockstructure
    )
    assert all(
        isinstance(subblock, _native.SubblockStructure)
        for subblock in degeneracystructure.subblockstructure
    )


def test_get_structure_functions_reuse_cached_structures_for_equal_hom_space():
    layout_module._clear_layout_caches_for_tests()
    left = hom((space(U1Irrep, {0: 2, 1: 3}),), (space(U1Irrep, {0: 2}),))
    right = hom((space(U1Irrep, {0: 2, 1: 3}),), (space(U1Irrep, {0: 2}),))

    assert get_sectorstructure(left) is get_sectorstructure(right)
    assert get_degeneracystructure(left) is get_degeneracystructure(right)


def test_sectorstructure_cache_reuses_sector_only_structure_across_different_degeneracies():
    layout_module._clear_layout_caches_for_tests()
    cached_space = hom(
        (space(U1Irrep, {0: 2, 1: 3}),),
        (space(U1Irrep, {0: 2, 1: 3}),),
    )
    target_space = hom(
        (space(U1Irrep, {0: 5, 1: 7}),),
        (space(U1Irrep, {0: 5, 1: 7}),),
    )

    assert get_sectorstructure(cached_space) is get_sectorstructure(target_space)
    assert get_degeneracystructure(cached_space) is not get_degeneracystructure(
        target_space,
    )


def test_sectorstructure_cache_does_not_reuse_across_visible_structure_changes():
    layout_module._clear_layout_caches_for_tests()

    codomain = space(U1Irrep, {0: 2})
    domain = space(U1Irrep, {1: 3})
    first = space(U1Irrep, {1: 2})
    second = space(U1Irrep, {-1: 3})

    cases = [
        (hom((codomain,), (domain,)), hom((domain,), (codomain,))),
        (hom((first, second), ()), hom((second, first), ())),
    ]
    for left, right in cases:
        assert get_sectorstructure(left) is not get_sectorstructure(right)


def test_get_structure_functions_reject_non_hom_space_inputs():
    with pytest.raises(TypeError, match="^get_sectorstructure\\(\\) requires a HomSpace$"):
        get_sectorstructure(object())  # pyright: ignore[reportArgumentType]

    with pytest.raises(TypeError, match="^get_degeneracystructure\\(\\) requires a HomSpace$"):
        get_degeneracystructure(object())  # pyright: ignore[reportArgumentType]
