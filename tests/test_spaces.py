import pytest

import tensor0.structure.spaces as spaces
from tensor0 import (
    ElementarySpace,
    FermionNumber,
    FermionParity,
    FermionParitySU2Irrep,
    FermionParityU1Irrep,
    FermionParityU1SU2Irrep,
    HomSpace,
    ProductSpace,
    SectorType,
    SU2Irrep,
    U1Irrep,
    U1SU2Irrep,
    Vect,
    Z2Irrep,
    Z4Irrep,
    _native,
    hom,
    space,
    zeros,
)


def assert_space_sectors(sector_type, sector_dims, expected_sectors):
    v = space(sector_type, sector_dims)

    assert isinstance(v, ElementarySpace)
    assert v.sector_spec == sector_type
    assert v.sectors == expected_sectors
    assert hom((v,), (v,)).codomain.spaces == (v,)


def test_sector_type_constants_and_products_expose_expected_aliases():
    assert isinstance(U1Irrep, SectorType)
    assert isinstance(FermionParity, SectorType)
    assert isinstance(SU2Irrep, SectorType)

    S = U1Irrep @ FermionParity
    assert isinstance(S, SectorType)
    assert S == FermionNumber
    assert hash(S) == hash(FermionNumber)

    assert FermionParity @ U1Irrep == FermionParityU1Irrep
    assert U1Irrep @ SU2Irrep == U1SU2Irrep
    assert FermionParity @ SU2Irrep == FermionParitySU2Irrep
    assert FermionParity @ U1Irrep @ SU2Irrep == FermionParityU1SU2Irrep


@pytest.mark.parametrize(
    ("sector_type", "sector", "expected"),
    [
        (U1Irrep, 7, 1),
        (Z4Irrep, 3, 1),
        (SU2Irrep, 1, 2),
        (U1SU2Irrep, (4, 2), 3),
        (FermionParityU1SU2Irrep, (1, 4, 2), 3),
    ],
    ids=[
        "u1",
        "z4",
        "su2",
        "u1-su2-product",
        "fermion-u1-su2-product",
    ],
)
def test_sector_type_quantum_dim_uses_typed_sector_rules(
    sector_type,
    sector,
    expected,
):
    assert sector_type.quantum_dim(sector) == expected


@pytest.mark.parametrize(
    "sector",
    [True, (True,), [1]],
    ids=["bool", "bool-tuple", "list"],
)
def test_sector_type_quantum_dim_rejects_invalid_sector_keys(sector):
    with pytest.raises(TypeError, match="sector key"):
        SU2Irrep.quantum_dim(sector)


def test_sector_type_quantum_dim_rejects_invalid_su2_values():
    with pytest.raises(ValueError, match="SU2 spin"):
        SU2Irrep.quantum_dim((-1,))

    with pytest.raises(ValueError, match="width"):
        SU2Irrep.quantum_dim((1, 2))


def test_empty_hom_without_sector_type_is_not_supported():
    with pytest.raises(ValueError, match="requires sector_type"):
        hom((), ())


def test_typed_scalar_hom_preserves_family_and_allocates_one_entry():
    scalar = spaces.hom((), (), sector_type=U1Irrep)
    scalar_su2 = spaces.hom((), (), sector_type=SU2Irrep)

    assert scalar.codomain.sector_spec == U1Irrep
    assert scalar.domain.sector_spec == U1Irrep
    assert scalar.static_key != scalar_su2.static_key
    assert scalar.codomain.static_key != scalar_su2.codomain.static_key
    assert spaces.dim(scalar.codomain) == 1
    assert spaces.fuse(scalar.codomain) == spaces.unit_space(U1Irrep)
    assert spaces.storage_dim(scalar) == 1
    assert zeros(scalar).storage.data.shape == (1,)


def test_product_space_equality_preserves_typed_empty_family():
    empty_u1 = spaces.hom((), (), sector_type=U1Irrep).domain
    empty_su2 = spaces.hom((), (), sector_type=SU2Irrep).domain

    assert empty_u1 != empty_su2
    assert empty_u1 != ()
    assert empty_su2 != ()
    assert hash(empty_u1) != hash(empty_su2)
    assert empty_u1.spaces == ()
    assert empty_su2.spaces == ()


def test_hom_explicit_sector_type_must_match_every_typed_input():
    u1 = space(U1Irrep, {0: 2})
    empty_u1 = hom((u1,), ()).domain

    assert spaces.hom((u1,), (), sector_type=U1Irrep) == hom((u1,), ())
    assert spaces.hom(empty_u1, (), sector_type=U1Irrep).codomain == empty_u1

    with pytest.raises(ValueError, match="sector_type must match"):
        spaces.hom((u1,), (), sector_type=SU2Irrep)
    with pytest.raises(ValueError, match="sector_type must match"):
        spaces.hom(empty_u1, (), sector_type=SU2Irrep)
    with pytest.raises(TypeError, match="sector_type must be a SectorType"):
        spaces.hom((), (), sector_type=object())  # pyright: ignore[reportArgumentType]


def test_hom_rejects_non_productspace_non_iterable_inputs():
    with pytest.raises(TypeError, match="ProductSpace or iterable of ElementarySpace"):
        hom(object(), ())  # pyright: ignore[reportArgumentType]


def test_vect_builds_native_u1_space_from_primitive_sector_values():
    sector_dims = {
        -1: 4,
        0: 5,
        1: 6,
        2: 3,
    }
    v = Vect[U1Irrep](sector_dims)

    assert isinstance(v, ElementarySpace)
    assert v.sector_spec == U1Irrep
    assert v.sectors == (((0,), 5), ((1,), 6), ((-1,), 4), ((2,), 3))
    assert v.is_dual is False
    assert v == space(U1Irrep, sector_dims)


def test_space_facade_normalizes_public_inputs():
    v = spaces.space(U1Irrep, {1: 2, (-1,): 3}, dual=True)

    assert v.sectors == (((-1,), 2), ((1,), 3))
    assert v.is_dual is True


def test_space_facade_rejects_invalid_public_inputs_before_native():
    with pytest.raises(TypeError, match="space\\(\\) requires sector_dims to be a mapping"):
        space(U1Irrep, [(0, 2)])  # pyright: ignore[reportArgumentType]

    with pytest.raises(TypeError, match="sector key must be an int or a tuple of ints"):
        space(U1Irrep, {"bad": 2})

    with pytest.raises(TypeError, match="sector dimension must be an int"):
        space(U1Irrep, {0: 2.5})  # pyright: ignore[reportArgumentType]

    with pytest.raises(ValueError, match="sector dimension must be non-negative"):
        space(U1Irrep, {0: -1})


def test_space_dimension_facade_distinguishes_dimension_concepts():
    su2 = space(SU2Irrep, {0: 2, 1: 3})
    other = space(SU2Irrep, {0: 5})
    product = hom((su2, other), ()).codomain

    assert spaces.dim(su2) == 8
    assert spaces.reduced_dim(su2) == 5
    assert spaces.dim(product) == 40

    with pytest.raises(TypeError, match="dim\\(\\) requires"):
        spaces.dim(object())  # pyright: ignore[reportArgumentType]
    with pytest.raises(TypeError, match="reduced_dim\\(\\) requires"):
        spaces.reduced_dim(product)  # pyright: ignore[reportArgumentType]
    with pytest.raises(TypeError, match="storage_dim\\(\\) requires"):
        spaces.storage_dim(su2)  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize(
    ("sector_type", "unit_sector"),
    [
        (U1Irrep, (0,)),
        (SU2Irrep, (0,)),
        (FermionParity, (0,)),
        (FermionNumber, (0, 0)),
        (U1SU2Irrep, (0, 0)),
        (FermionParityU1SU2Irrep, (0, 0, 0)),
    ],
)
def test_unit_and_zero_space_preserve_family_and_dual_metadata(
    sector_type,
    unit_sector,
):
    unit = spaces.unit_space(sector_type)
    dual_unit = spaces.unit_space(sector_type, dual=True)
    zero = spaces.zero_space(sector_type)
    dual_zero = spaces.zero_space(sector_type, dual=True)

    assert unit.sector_spec == sector_type
    assert unit.sectors == ((unit_sector, 1),)
    assert spaces.dim(unit) == 1
    assert dual_unit.is_dual is True
    assert dual_unit.sectors == unit.sectors
    assert zero.sector_spec == sector_type
    assert zero.sectors == ()
    assert spaces.dim(zero) == 0
    assert dual_zero.is_dual is True
    assert zero != dual_zero


def test_space_algebra_facade_uses_native_family_and_dual_rules():
    left = space(U1Irrep, {0: 2, 1: 3})
    right = space(U1Irrep, {1: 5, -1: 7})
    product = hom((left, right), ()).codomain

    assert spaces.fuse(product).sectors == (
        ((0,), 21),
        ((1,), 10),
        ((-1,), 14),
        ((2,), 15),
    )
    assert spaces.infimum(left, right).sectors == (((1,), 3),)
    assert spaces.supremum(left, right).sectors == (
        ((0,), 2),
        ((1,), 5),
        ((-1,), 7),
    )
    assert spaces.direct_sum(left, right).sectors == (
        ((0,), 2),
        ((1,), 8),
        ((-1,), 7),
    )

    for operation in (spaces.infimum, spaces.supremum, spaces.direct_sum):
        with pytest.raises(ValueError, match="same dual flag"):
            operation(left, right.dual())
        with pytest.raises(ValueError, match="same sector family"):
            operation(left, space(SU2Irrep, {0: 1}))


def test_space_partial_order_facade_uses_visible_sector_dimensions():
    small = space(U1Irrep, {0: 1, 1: 2})
    large = space(U1Irrep, {0: 3, 1: 2, -1: 4})

    assert spaces.is_isomorphic(small, small)
    assert not spaces.is_isomorphic(small, large)
    assert spaces.is_monomorphic(small, large)
    assert spaces.is_epimorphic(large, small)
    assert not spaces.is_isomorphic(small, space(SU2Irrep, {0: 1}))


def test_hom_space_permute_builds_destination_space_from_visible_indices():
    v = space(U1Irrep, {1: 2})
    w = space(U1Irrep, {2: 3})
    x = space(U1Irrep, {3: 5})
    src = hom((v, w), (x,))

    dst = src.permute((2, 1), (0,))

    assert dst.codomain.spaces == (x.dual(), w)
    assert dst.domain.spaces == (v.dual(),)

    with pytest.raises(ValueError, match="visible index"):
        src.permute((0,), (0,))


def test_hom_facade_builds_typed_empty_product_from_iterables():
    v = space(U1Irrep, {0: 2})
    h = spaces.hom((v,), ())

    assert h.codomain.spaces == (v,)
    assert h.domain.spaces == ()
    assert h.domain.sector_spec == U1Irrep


@pytest.mark.parametrize(
    ("sector_type", "sector_dims", "expected_sectors"),
    [
        (
            U1Irrep @ FermionParity,
            {
                (1, 1): 2,
                (0, 1): 3,
                (0, 0): 4,
                (-1, 0): 5,
            },
            (
                ((0, 0), 4),
                ((0, 1), 3),
                ((1, 1), 2),
                ((-1, 0), 5),
            ),
        ),
        (
            FermionParityU1Irrep,
            {(1, 0): 2},
            (((1, 0), 2),),
        ),
        (
            U1SU2Irrep,
            {
                (1, 1): 2,
                (0, 0): 3,
            },
            (((0, 0), 3), ((1, 1), 2)),
        ),
        (
            FermionParitySU2Irrep,
            {
                (0, 0): 2,
                (1, 1): 3,
            },
            (((0, 0), 2), ((1, 1), 3)),
        ),
        (
            FermionParityU1SU2Irrep,
            {
                (0, 0, 0): 2,
                (1, 1, 1): 3,
            },
            (((0, 0, 0), 2), ((1, 1, 1), 3)),
        ),
    ],
    ids=[
        "u1-fermion-ordering",
        "fermion-u1",
        "u1-su2",
        "fermion-su2",
        "fermion-u1-su2",
    ],
)
def test_product_sector_spaces_normalize_supported_sector_values(
    sector_type,
    sector_dims,
    expected_sectors,
):
    assert_space_sectors(sector_type, sector_dims, expected_sectors)


def test_zn_precompiled_sector_specs_are_supported():
    z2 = space(Z2Irrep, {0: 1, 3: 2})
    z4 = space(Z4Irrep, {-1: 5})

    assert z2.sectors == (((0,), 1), ((1,), 2))
    assert z4.sectors == (((3,), 5),)


def test_su2_space_uses_twice_spin_sector_keys():
    v = space(SU2Irrep, {
        2: 5,
        0: 2,
        1: 3,
    })

    assert isinstance(v, ElementarySpace)
    assert v.sector_spec == SU2Irrep
    assert v.sectors == (((0,), 2), ((1,), 3), ((2,), 5))
    assert v.is_dual is False


def test_su2_space_rejects_negative_spin_in_rust():
    with pytest.raises(ValueError, match="SU2 spin must be non-negative"):
        space(SU2Irrep, {-1: 2})


def test_unlisted_su2_product_permutations_remain_unsupported():
    S = SU2Irrep @ U1Irrep

    assert isinstance(S, SectorType)
    assert S != U1SU2Irrep
    with pytest.raises(ValueError, match="unsupported sector spec"):
        space(S, {(1, 1): 2})


def test_duplicate_canonical_sector_values_are_rejected_by_rust():
    with pytest.raises(ValueError, match="sector appears multiple times"):
        space(FermionParity, {-1: 2, 1: 3})


def test_dual_space_uses_native_dual_rules():
    v = space(U1Irrep, {1: 8, -2: 4})
    dual = v.dual()

    assert isinstance(dual, ElementarySpace)
    assert dual.sector_spec == U1Irrep
    assert dual.sectors == (((-1,), 8), ((2,), 4))
    assert dual.is_dual is True
    assert dual.static_key != v.static_key


def test_hom_space_is_native_and_exposes_visible_legs():
    v = space(U1Irrep, {0: 2, 1: 3})
    w = space(U1Irrep, {-1: 5})
    h = hom((v,), (w,))

    assert isinstance(h, HomSpace)
    assert h.numout == 1
    assert h.numin == 1
    assert h.numind == 2
    assert len(h) == 2
    assert h.codomain.spaces == (v,)
    assert h.domain.spaces == (w,)
    assert h.visible_legs == (v, w.dual())
    assert h[0] == v
    assert h[1] == w.dual()
    assert h[-1] == w.dual()
    with pytest.raises(IndexError, match="visible index"):
        h[2]


def test_native_productspace_is_sequence_like_and_fuse_matches_tensorkit_compact_bond_inputs():
    v = space(U1Irrep, {0: 2, 1: 3})
    w = space(U1Irrep, {0: 5, -1: 7})
    x = space(U1Irrep, {0: 4, 1: 9})
    h = hom((v, w), (x,))

    assert isinstance(h.codomain, ProductSpace)
    assert len(h.codomain) == 2
    assert h.codomain[0] == v
    assert h.codomain[-1] == w
    assert tuple(h.codomain) == (v, w)
    assert h.codomain.spaces == (v, w)
    assert hash(h.codomain) == hash(h.codomain)

    roundtrip = hom(h.codomain, h.domain)

    assert roundtrip == h

    fused_codomain = _native.fuse(h.codomain)
    fused_domain = _native.fuse(h.domain)
    bond = _native.infimum_space(fused_codomain, fused_domain)

    assert fused_codomain.sectors == (((0,), 31), ((1,), 15), ((-1,), 14))
    assert fused_domain.sectors == (((0,), 4), ((1,), 9))
    assert bond.sectors == (((0,), 4), ((1,), 9))
    assert bond.is_dual is False


def test_native_productspace_inputs_preserve_typed_empty_hom_context():
    v = space(U1Irrep, {0: 2})
    empty_u1 = hom((v,), ()).domain

    assert isinstance(empty_u1, ProductSpace)
    assert len(empty_u1) == 0
    assert empty_u1.sector_spec == U1Irrep

    left_empty = hom(empty_u1, (v,))

    assert left_empty.codomain.spaces == ()
    assert left_empty.codomain.sector_spec == U1Irrep
    assert left_empty.domain.spaces == (v,)

    both_empty = hom(empty_u1, empty_u1)

    assert len(both_empty.codomain) == 0
    assert len(both_empty.domain) == 0
    assert both_empty.codomain.sector_spec == U1Irrep
    assert both_empty.domain.sector_spec == U1Irrep

    empty_su2 = hom((space(SU2Irrep, {0: 1}),), ()).domain
    assert empty_u1.sector_spec != empty_su2.sector_spec
    assert hash(empty_u1) != hash(empty_su2)

    with pytest.raises(ValueError, match="same sector family"):
        hom(empty_u1, empty_su2)


def test_native_infimum_preserves_dual_flag_and_rejects_mismatch():
    left = space(U1Irrep, {0: 10, 1: 3}).dual()
    right = space(U1Irrep, {0: 4, -1: 9}).dual()

    common = _native.infimum_space(left, right)

    assert common.sectors == (((0,), 4),)
    assert common.is_dual is True

    with pytest.raises(ValueError, match="same dual flag"):
        _native.infimum_space(space(U1Irrep, {0: 1}), space(U1Irrep, {0: 1}).dual())


def test_native_infimum_rejects_mismatched_sector_families():
    with pytest.raises(ValueError, match="same sector family"):
        _native.infimum_space(space(U1Irrep, {0: 1}), space(SU2Irrep, {0: 1}))


def test_hom_rejects_mixed_sector_families():
    u1 = space(U1Irrep, {0: 2})
    z2 = space(Z2Irrep, {0: 2})

    with pytest.raises(ValueError, match="same sector"):
        hom((u1,), (z2,))


def test_native_hom_unit_methods_validate_indices_and_unit_factors():
    unit = space(U1Irrep, {0: 1})
    empty = hom((unit,), ()).domain
    scalar_space = hom(empty, empty)

    inserted = scalar_space.insert_left_unit(0, False)
    assert inserted.remove_unit(0) == scalar_space

    for operation in (
        lambda: scalar_space.insert_left_unit(-1, False),
        lambda: scalar_space.insert_right_unit(-1, False),
        lambda: scalar_space.remove_unit(-1),
    ):
        with pytest.raises(ValueError, match="non-negative"):
            operation()

    nonunit = space(U1Irrep, {0: 2})
    with pytest.raises(ValueError, match="canonical unit space"):
        hom((nonunit,), empty).remove_unit(0)
