from dataclasses import dataclass

import jax
import jax.numpy as jnp
import pytest

from tensor0 import (
    FermionParity,
    SU2Irrep,
    SectorVector,
    TensorMap,
    U1Irrep,
    _native,
    hom,
    notrunc,
    space,
    svd_compact,
    svd_trunc,
    truncerror,
    truncrank,
    truncspace,
    trunctol,
)
from tests.cases import (
    assert_allclose,
    assert_tensormap_blocks_allclose,
    factorization_cases,
    float_data_for,
)


@dataclass(frozen=True)
class TruncationCase:
    name: str
    trunc: object
    expected_sectors: tuple[tuple[tuple[int, ...], int], ...]
    expected_blocks: tuple[tuple[int, tuple[float, ...]], ...]
    expected_error_squared: float
    zero_reconstruction: bool = False


def _rectangular_u1_hom():
    v = space(U1Irrep, {0: 2, 1: 4})
    w = space(U1Irrep, {0: 3, 1: 2})
    return hom((v,), (w,))


def _known_u1_truncation_tensor():
    h = _rectangular_u1_hom()
    sector_0 = jnp.array([[5.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=jnp.float32)
    sector_1 = jnp.array(
        [[4.0, 0.0], [0.0, 3.0], [0.0, 0.0], [0.0, 0.0]], dtype=jnp.float32
    )
    data = jnp.concatenate((sector_0.reshape((-1,)), sector_1.reshape((-1,))))
    return TensorMap(h, data)


def _known_su2_truncation_tensor():
    v = space(SU2Irrep, {0: 1, 2: 1})
    w = space(SU2Irrep, {0: 1, 2: 1})
    h = hom((v,), (w,))
    data = jnp.array([4.0, 5.0], dtype=jnp.float32)
    return TensorMap(h, data)


def _u1_truncation_cases():
    return (
        TruncationCase(
            "notrunc-none",
            None,
            (((0,), 2), ((1,), 2)),
            ((0, (5.0, 1.0)), (1, (4.0, 3.0))),
            0.0,
        ),
        TruncationCase(
            "rank-2",
            truncrank(2),
            (((0,), 1), ((1,), 1)),
            ((0, (5.0,)), (1, (4.0,))),
            10.0,
        ),
        TruncationCase(
            "rank-0",
            truncrank(0),
            (),
            (),
            51.0,
            zero_reconstruction=True,
        ),
        TruncationCase(
            "tol-3.5",
            trunctol(atol=3.5),
            (((0,), 1), ((1,), 1)),
            ((0, (5.0,)), (1, (4.0,))),
            10.0,
        ),
        TruncationCase(
            "error-1.1",
            truncerror(atol=1.1),
            (((0,), 1), ((1,), 2)),
            ((0, (5.0,)), (1, (4.0, 3.0))),
            1.0,
        ),
        TruncationCase(
            "space",
            truncspace(space(U1Irrep, {0: 1, 1: 2})),
            (((0,), 1), ((1,), 2)),
            ((0, (5.0,)), (1, (4.0, 3.0))),
            1.0,
        ),
        TruncationCase(
            "space-rev-false",
            truncspace(space(U1Irrep, {0: 1, 1: 1}), rev=False),
            (((0,), 1), ((1,), 1)),
            ((0, (1.0,)), (1, (3.0,))),
            41.0,
        ),
        TruncationCase(
            "intersection",
            truncrank(3) & trunctol(atol=3.5),
            (((0,), 1), ((1,), 1)),
            ((0, (5.0,)), (1, (4.0,))),
            10.0,
        ),
        TruncationCase(
            "union",
            truncrank(2) | trunctol(atol=2.5),
            (((0,), 1), ((1,), 2)),
            ((0, (5.0,)), (1, (4.0, 3.0))),
            1.0,
        ),
    )


def _assert_u1_truncation_case(case):
    tensor = _known_u1_truncation_tensor()

    u, s, vh, err = svd_trunc(tensor, trunc=case.trunc)

    assert u.space == hom(tensor.space.codomain, (s.space,))
    assert vh.space == hom((s.space,), tensor.space.domain)
    assert s.space.sectors == case.expected_sectors
    for sector, values in case.expected_blocks:
        assert_allclose(s.block(sector), jnp.array(values, dtype=jnp.float32))
    assert float(err) == pytest.approx(
        float(jnp.sqrt(jnp.array(case.expected_error_squared))),
        rel=1e-5,
    )
    if case.zero_reconstruction:
        reconstructed = u @ s.to_diagonal() @ vh
        for coupled, original_block in tensor.blocks():
            assert_allclose(
                reconstructed.block(coupled),
                jnp.zeros_like(original_block),
            )


def test_svd_compact_returns_expected_spaces_and_sector_ranks():
    h = _rectangular_u1_hom()
    tensor = TensorMap(h, float_data_for(h))

    u, s, vh = svd_compact(tensor)

    assert isinstance(u, TensorMap)
    assert isinstance(s, SectorVector)
    assert isinstance(vh, TensorMap)
    assert s.space.sectors == (((0,), 2), ((1,), 2))
    assert u.space == hom(tensor.space.codomain, (s.space,))
    assert vh.space == hom((s.space,), tensor.space.domain)
    assert u.block(0).shape == (2, 2)
    assert u.block(1).shape == (4, 2)
    assert s.block(0).shape == (2,)
    assert s.block(1).shape == (2,)
    assert vh.block(0).shape == (2, 3)
    assert vh.block(1).shape == (2, 2)


@pytest.mark.parametrize("case", factorization_cases(), ids=lambda case: case.name)
def test_svd_compact_reconstructs_factorization_cases(case):
    tensor = TensorMap(case.space, float_data_for(case.space))

    u, s, vh = svd_compact(tensor)
    reconstructed = u @ s.to_diagonal() @ vh
    expected_bond = _native.infimum_space(
        _native.fuse(tensor.space.codomain),
        _native.fuse(tensor.space.domain),
    )

    assert s.space == expected_bond
    assert u.space == hom(tensor.space.codomain, (expected_bond,))
    assert vh.space == hom((expected_bond,), tensor.space.domain)
    assert_tensormap_blocks_allclose(reconstructed, tensor)


@pytest.mark.parametrize("case", factorization_cases(), ids=lambda case: case.name)
def test_svd_trunc_notrunc_matches_compact_factorization_cases(case):
    tensor = TensorMap(case.space, float_data_for(case.space))

    compact_u, compact_s, compact_vh = svd_compact(tensor)
    u, s, vh, err = svd_trunc(tensor, trunc=notrunc())
    reconstructed = u @ s.to_diagonal() @ vh

    assert s.space == compact_s.space
    assert u.space == compact_u.space
    assert vh.space == compact_vh.space
    assert float(err) == pytest.approx(0.0)
    assert_tensormap_blocks_allclose(reconstructed, tensor)


@pytest.mark.parametrize("case", _u1_truncation_cases(), ids=lambda case: case.name)
def test_svd_trunc_applies_u1_truncation_strategies(case):
    _assert_u1_truncation_case(case)


def test_svd_trunc_rejects_invalid_truncation_strategy():
    tensor = _known_u1_truncation_tensor()

    with pytest.raises(TypeError, match="truncation strategy expected"):
        svd_trunc(tensor, trunc=object())


def test_svd_trunc_supports_fermion_parity_without_braiding_transform():
    v = space(FermionParity, {0: 2, 1: 2})
    w = space(FermionParity, {0: 2, 1: 2})
    h = hom((v,), (w,))
    data = jnp.array(
        [
            5.0,
            0.0,
            0.0,
            1.0,
            4.0,
            0.0,
            0.0,
            3.0,
        ],
        dtype=jnp.float32,
    )
    tensor = TensorMap(h, data)

    _u, s, _vh, err = svd_trunc(tensor, trunc=truncrank(2))

    assert s.space.sectors == (((0,), 1), ((1,), 1))
    assert_allclose(s.block(0), jnp.array([5.0], dtype=jnp.float32))
    assert_allclose(s.block(1), jnp.array([4.0], dtype=jnp.float32))
    assert float(err) == pytest.approx(float(jnp.sqrt(jnp.array(10.0))), rel=1e-5)


@pytest.mark.parametrize(
    ("rank", "expected_sectors", "sector", "expected_values", "expected_error_squared"),
    [
        (3, (((2,), 1),), 2, (5.0,), 16.0),
        (2, (((0,), 1),), 0, (4.0,), 75.0),
    ],
    ids=["keeps-weighted-largest", "skips-overweight-value"],
)
def test_svd_trunc_truncrank_uses_su2_quantum_dimensions(
    rank,
    expected_sectors,
    sector,
    expected_values,
    expected_error_squared,
):
    tensor = _known_su2_truncation_tensor()

    _u, s, _vh, err = svd_trunc(tensor, trunc=truncrank(rank))

    assert s.space.sectors == expected_sectors
    assert_allclose(s.block(sector), jnp.array(expected_values, dtype=jnp.float32))
    assert float(err) == pytest.approx(
        float(jnp.sqrt(jnp.array(expected_error_squared))),
        rel=1e-5,
    )


def test_truncation_constructors_reject_non_callable_by():
    with pytest.raises(TypeError, match="by must be callable"):
        truncrank(1, by=object())

    with pytest.raises(TypeError, match="by must be callable"):
        trunctol(atol=1.0, by=object())

    with pytest.raises(TypeError, match="by must be callable"):
        truncspace(space(U1Irrep, {0: 1}), by=object())


def test_svd_compact_supports_typed_empty_hom_space():
    empty = _native.make_product_space(U1Irrep, ())
    h = hom(empty, empty)
    tensor = TensorMap(h, jnp.array([3.0], dtype=jnp.float32))

    u, s, vh = svd_compact(tensor)
    expected_bond = _native.infimum_space(
        _native.fuse(h.codomain),
        _native.fuse(h.domain),
    )
    reconstructed = u @ s.to_diagonal() @ vh

    assert s.space == expected_bond
    assert s.space.sectors == (((0,), 1),)
    assert u.space == hom(tensor.space.codomain, (expected_bond,))
    assert vh.space == hom((expected_bond,), tensor.space.domain)
    assert_allclose(reconstructed.block(0), tensor.block(0))


def test_svd_compact_complex_input_has_real_singular_values_and_reconstructs():
    h = _rectangular_u1_hom()
    data = float_data_for(h) + 1j * jnp.flip(float_data_for(h))
    tensor = TensorMap(h, data)

    u, s, vh = svd_compact(tensor)
    reconstructed = u @ s.to_diagonal() @ vh

    assert not jnp.issubdtype(s.storage.data.dtype, jnp.complexfloating)
    assert u.storage.data.dtype == data.dtype
    assert vh.storage.data.dtype == data.dtype
    assert_tensormap_blocks_allclose(reconstructed, tensor)


def test_svd_compact_jit_returns_sectorvector_output():
    h = _rectangular_u1_hom()
    tensor = TensorMap(h, float_data_for(h))
    jitted = jax.jit(svd_compact)

    actual_u, actual_s, actual_vh = jitted(tensor)
    expected_u, expected_s, expected_vh = svd_compact(tensor)

    assert isinstance(actual_u, TensorMap)
    assert isinstance(actual_s, SectorVector)
    assert isinstance(actual_vh, TensorMap)
    assert actual_s.space == expected_s.space
    assert_allclose(actual_u.storage.data, expected_u.storage.data)
    assert_allclose(actual_s.storage.data, expected_s.storage.data)
    assert_allclose(actual_vh.storage.data, expected_vh.storage.data)
