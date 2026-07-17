from dataclasses import dataclass

import jax
from jax import Array
import jax.numpy as jnp
import pytest

from tensor0 import (
    DiagonalTensorMap,
    FermionParity,
    HomSpace,
    SectorVector,
    SU2Irrep,
    TensorMap,
    U1Irrep,
    U1SU2Irrep,
    _native,
    cond,
    eigh_full,
    eigh_vals,
    from_blocks,
    fuse,
    hom,
    infimum,
    is_hermitian,
    left_orth,
    lq_compact,
    notrunc,
    qr_compact,
    rank,
    right_orth,
    space,
    svd_compact,
    svd_full,
    svd_trunc,
    svd_vals,
    truncerror,
    truncrank,
    truncspace,
    trunctol,
    zero_space,
)
from tensor0.structure import get_blockstructure, get_degeneracystructure
from tensor0.tensor.constructors import identity
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


def _svd_infimum_bond_for(tensor: TensorMap):
    return _native.infimum_space(
        _native.fuse(tensor.space.codomain),
        _native.fuse(tensor.space.domain),
    )


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

    assert u.space == hom(tensor.space.codomain, (s.index_space,))
    assert vh.space == hom((s.index_space,), tensor.space.domain)
    assert s.domain.spaces == (s.index_space,)
    assert s.codomain.spaces == (s.index_space,)
    assert s.index_space.sectors == case.expected_sectors
    for sector, values in case.expected_blocks:
        assert_allclose(s.diag().block(sector), jnp.array(values, dtype=jnp.float32))
    assert float(err) == pytest.approx(
        float(jnp.sqrt(jnp.array(case.expected_error_squared))),
        rel=1e-5,
    )
    reconstructed = u @ s @ vh
    assert reconstructed.space == tensor.space
    if case.zero_reconstruction:
        for coupled, original_block in tensor.blocks():
            assert_allclose(
                reconstructed.block(coupled),
                jnp.zeros_like(original_block),
            )


@pytest.mark.parametrize("case", factorization_cases(), ids=lambda case: case.name)
def test_svd_compact_reconstructs_factorization_cases(case):
    tensor = TensorMap(case.space, float_data_for(case.space))

    u, s, vh = svd_compact(tensor)
    reconstructed = u @ s @ vh
    expected_bond = _svd_infimum_bond_for(tensor)

    assert s.index_space == expected_bond
    assert s.domain.spaces == (expected_bond,)
    assert s.codomain.spaces == (expected_bond,)
    assert u.space == hom(tensor.space.codomain, (expected_bond,))
    assert vh.space == hom((expected_bond,), tensor.space.domain)
    assert_tensormap_blocks_allclose(reconstructed, tensor)


@pytest.mark.parametrize("case", factorization_cases(), ids=lambda case: case.name)
def test_svd_vals_returns_infimum_sector_vector(case):
    tensor = TensorMap(case.space, float_data_for(case.space))
    expected_bond = _svd_infimum_bond_for(tensor)

    values = svd_vals(tensor)

    assert isinstance(values, SectorVector)
    assert values.sectors == expected_bond.sectors
    for sector, _dim in expected_bond.sectors:
        expected_values = jnp.linalg.svd(tensor.block(sector), compute_uv=False)
        assert_allclose(values.block(sector), expected_values)


def test_rank_and_cond_use_blockwise_singular_values():
    tensor = _known_su2_truncation_tensor()

    assert int(rank(tensor)) == 4
    assert int(rank(tensor, atol=4.5)) == 3
    assert_allclose(cond(tensor), jnp.asarray(5.0 / 4.0, dtype=jnp.float32))


def test_rank_and_cond_handle_zero_and_invalid_inputs():
    h = _rectangular_u1_hom()
    tensor = TensorMap(h, jnp.zeros((get_degeneracystructure(h).total_dim,)))

    assert int(rank(tensor)) == 0
    assert bool(jnp.isinf(cond(tensor)))

    with pytest.raises(TypeError, match="rank.*TensorMap"):
        rank(object())  # pyright: ignore[reportArgumentType]

    with pytest.raises(ValueError, match="atol.*non-negative"):
        rank(tensor, atol=-1.0)

    with pytest.raises(NotImplementedError, match="p=2"):
        cond(tensor, p=1)


@pytest.mark.parametrize("case", factorization_cases(), ids=lambda case: case.name)
def test_svd_full_uses_fused_spaces_and_reconstructs(case):
    tensor = TensorMap(case.space, float_data_for(case.space))
    fused_codomain = _native.fuse(tensor.space.codomain)
    fused_domain = _native.fuse(tensor.space.domain)

    u, s, vh = svd_full(tensor)
    reconstructed = u @ s @ vh

    assert isinstance(s, TensorMap)
    assert not isinstance(s, DiagonalTensorMap)
    assert u.space == hom(tensor.space.codomain, (fused_codomain,))
    assert s.space == hom((fused_codomain,), (fused_domain,))
    assert vh.space == hom((fused_domain,), tensor.space.domain)
    assert_tensormap_blocks_allclose(reconstructed, tensor)


def test_svd_trunc_notrunc_matches_compact():
    target = _rectangular_u1_hom()
    tensor = TensorMap(target, float_data_for(target))

    compact_u, compact_s, compact_vh = svd_compact(tensor)
    u, s, vh, err = svd_trunc(tensor, trunc=notrunc())
    reconstructed = u @ s @ vh

    assert s.index_space == compact_s.index_space
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

    assert s.index_space.sectors == (((0,), 1), ((1,), 1))
    assert_allclose(s.diag().block(0), jnp.array([5.0], dtype=jnp.float32))
    assert_allclose(s.diag().block(1), jnp.array([4.0], dtype=jnp.float32))
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

    assert s.index_space.sectors == expected_sectors
    assert_allclose(s.diag().block(sector), jnp.array(expected_values, dtype=jnp.float32))
    assert float(err) == pytest.approx(
        float(jnp.sqrt(jnp.array(expected_error_squared))),
        rel=1e-5,
    )


def test_truncation_constructors_reject_non_callable_by():
    with pytest.raises(TypeError, match="by must be callable"):
        truncrank(1, by=object())  # pyright: ignore[reportArgumentType]

    with pytest.raises(TypeError, match="by must be callable"):
        trunctol(atol=1.0, by=object())  # pyright: ignore[reportArgumentType]

    with pytest.raises(TypeError, match="by must be callable"):
        truncspace(space(U1Irrep, {0: 1}), by=object())  # pyright: ignore[reportArgumentType]


def test_svd_compact_supports_typed_empty_hom_space():
    empty = _native.make_product_space(U1Irrep, ())
    h = hom(empty, empty)
    tensor = TensorMap(h, jnp.array([3.0], dtype=jnp.float32))

    u, s, vh = svd_compact(tensor)
    expected_bond = _svd_infimum_bond_for(tensor)
    reconstructed = u @ s @ vh

    assert s.index_space == expected_bond
    assert s.index_space.sectors == (((0,), 1),)
    assert u.space == hom(tensor.space.codomain, (expected_bond,))
    assert vh.space == hom((expected_bond,), tensor.space.domain)
    assert_allclose(reconstructed.block(0), tensor.block(0))


def test_svd_compact_complex_input_has_real_singular_values_and_reconstructs():
    h = _rectangular_u1_hom()
    data = float_data_for(h) + 1j * jnp.flip(float_data_for(h))
    tensor = TensorMap(h, data)

    u, s, vh = svd_compact(tensor)
    reconstructed = u @ s @ vh

    assert not jnp.issubdtype(s.storage.data.dtype, jnp.complexfloating)
    assert u.storage.data.dtype == data.dtype
    assert vh.storage.data.dtype == data.dtype
    assert_tensormap_blocks_allclose(reconstructed, tensor)


def test_svd_compact_jit_returns_diagonal_tensormap_output():
    h = _rectangular_u1_hom()
    tensor = TensorMap(h, float_data_for(h))
    jitted = jax.jit(svd_compact)

    actual_u, actual_s, actual_vh = jitted(tensor)
    expected_u, expected_s, expected_vh = svd_compact(tensor)

    assert isinstance(actual_u, TensorMap)
    assert isinstance(actual_s, DiagonalTensorMap)
    assert isinstance(actual_vh, TensorMap)
    assert actual_s.index_space == expected_s.index_space
    assert_allclose(actual_u.storage.data, expected_u.storage.data)
    assert_allclose(actual_s.storage.data, expected_s.storage.data)
    assert_allclose(actual_vh.storage.data, expected_vh.storage.data)


def _assert_tensors_allclose(left: TensorMap, right: TensorMap) -> None:
    assert left.space == right.space
    assert_allclose(left.storage.data, right.storage.data)


def _assert_positive_diagonal(block: Array) -> None:
    diagonal = jnp.diag(block)
    assert_allclose(jnp.imag(diagonal), jnp.zeros_like(jnp.imag(diagonal)))
    assert bool(jnp.all(jnp.real(diagonal) >= 0))


def _assert_qr_contract(tensor: TensorMap) -> tuple[TensorMap, TensorMap]:
    q, r = qr_compact(tensor)
    bond = infimum(fuse(tensor.codomain), fuse(tensor.domain))

    assert q.domain.spaces == (bond,)
    assert r.codomain.spaces == (bond,)
    assert not bond.is_dual
    _assert_tensors_allclose(q @ r, tensor)
    _assert_tensors_allclose(q.adjoint() @ q, identity(bond, dtype=tensor.dtype))
    for _coupled, block in r.blocks():
        _assert_positive_diagonal(block)
    return q, r


def _assert_lq_contract(tensor: TensorMap) -> tuple[TensorMap, TensorMap]:
    l, q = lq_compact(tensor)
    bond = infimum(fuse(tensor.codomain), fuse(tensor.domain))

    assert l.domain.spaces == (bond,)
    assert q.codomain.spaces == (bond,)
    assert not bond.is_dual
    _assert_tensors_allclose(l @ q, tensor)
    _assert_tensors_allclose(q @ q.adjoint(), identity(bond, dtype=tensor.dtype))
    for _coupled, block in l.blocks():
        _assert_positive_diagonal(block)
    return l, q


@pytest.mark.parametrize("case", factorization_cases(), ids=lambda case: case.name)
def test_compact_qr_lq_reconstruct_and_orthogonalize_supported_cases(case):
    data = float_data_for(case.space)
    tensor = TensorMap(case.space, data + 1j * jnp.flip(data))

    _assert_qr_contract(tensor)
    _assert_lq_contract(tensor)


def test_left_and_right_orth_default_to_compact_conventions():
    factor = space(U1Irrep, {0: 3, 1: 2})
    target = hom((factor,), (factor,))
    tensor = TensorMap(target, float_data_for(target))

    q, r = qr_compact(tensor)
    left_q, left_r = left_orth(tensor)
    _assert_tensors_allclose(left_q, q)
    _assert_tensors_allclose(left_r, r)

    l, q = lq_compact(tensor)
    right_l, right_q = right_orth(tensor)
    _assert_tensors_allclose(right_l, l)
    _assert_tensors_allclose(right_q, q)


def test_left_and_right_orth_support_svd_conventions():
    tensor = _known_u1_truncation_tensor()
    u, s, vh = svd_compact(tensor)

    left_basis, left_core = left_orth(tensor, alg="svd")
    right_core, right_basis = right_orth(tensor, alg="svd")

    _assert_tensors_allclose(left_basis, u)
    _assert_tensors_allclose(left_core, s @ vh)
    _assert_tensors_allclose(right_core, u @ s)
    _assert_tensors_allclose(right_basis, vh)
    _assert_tensors_allclose(left_basis @ left_core, tensor)
    _assert_tensors_allclose(right_core @ right_basis, tensor)


def test_orth_truncation_selects_svd_and_reduces_connecting_space():
    tensor = _known_u1_truncation_tensor()
    strategy = truncrank(2)
    u, s, vh, _error = svd_trunc(tensor, trunc=strategy)

    left_basis, left_core = left_orth(tensor, trunc=strategy)
    right_core, right_basis = right_orth(tensor, trunc=strategy)

    assert left_basis.domain.spaces == (s.index_space,)
    assert left_core.codomain.spaces == (s.index_space,)
    assert right_core.domain.spaces == (s.index_space,)
    assert right_basis.codomain.spaces == (s.index_space,)
    _assert_tensors_allclose(left_basis, u)
    _assert_tensors_allclose(left_core, s @ vh)
    _assert_tensors_allclose(right_core, u @ s)
    _assert_tensors_allclose(right_basis, vh)


def test_orth_rejects_incompatible_or_unknown_algorithms():
    tensor = _known_u1_truncation_tensor()
    strategy = notrunc()

    with pytest.raises(ValueError, match="truncation.*alg='qr'"):
        left_orth(tensor, alg="qr", trunc=strategy)
    with pytest.raises(ValueError, match="truncation.*alg='lq'"):
        right_orth(tensor, alg="lq", trunc=strategy)
    with pytest.raises(ValueError, match="unknown algorithm.*polar"):
        left_orth(tensor, alg="polar")  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError, match="unknown algorithm.*polar"):
        right_orth(tensor, alg="polar")  # pyright: ignore[reportArgumentType]


def test_orth_rejects_invalid_automatic_truncation_strategy():
    tensor = _known_u1_truncation_tensor()

    with pytest.raises(TypeError, match="truncation strategy expected"):
        left_orth(tensor, trunc=object())


def test_rank_deficient_blocks_keep_structural_compact_shapes_under_jit():
    codomain = space(U1Irrep, {0: 3})
    domain = space(U1Irrep, {0: 2})
    target = hom((codomain,), (domain,))
    block = jnp.asarray(
        [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]],
        dtype=jnp.float32,
    )
    tensor = from_blocks(target, {0: block})

    eager_q, eager_r = _assert_qr_contract(tensor)
    jitted_q, jitted_r = jax.jit(qr_compact)(tensor)
    _assert_tensors_allclose(jitted_q, eager_q)
    _assert_tensors_allclose(jitted_r, eager_r)
    assert eager_q.block(0).shape == (3, 2)
    assert eager_r.block(0).shape == (2, 2)

    adjoint = tensor.adjoint()
    eager_l, eager_q = _assert_lq_contract(adjoint)
    jitted_l, jitted_q = jax.jit(lq_compact)(adjoint)
    _assert_tensors_allclose(jitted_l, eager_l)
    _assert_tensors_allclose(jitted_q, eager_q)
    assert eager_l.block(0).shape == (2, 2)
    assert eager_q.block(0).shape == (2, 3)


def test_zero_block_uses_neutral_phase_and_keeps_static_shapes_under_jit():
    codomain = space(U1Irrep, {0: 3})
    domain = space(U1Irrep, {0: 2})
    target = hom((codomain,), (domain,))
    block = jnp.zeros((3, 2), dtype=jnp.complex64)
    tensor = from_blocks(target, {0: block})

    raw_q, raw_r = jnp.linalg.qr(block, mode="reduced")
    q, r = jax.jit(qr_compact)(tensor)
    assert_allclose(q.block(0), raw_q)
    assert_allclose(r.block(0), raw_r)
    _assert_qr_contract(tensor)

    raw_q_adjoint, raw_r_adjoint = jnp.linalg.qr(jnp.conj(block.T), mode="reduced")
    l, right_q = jax.jit(lq_compact)(tensor)
    assert_allclose(l.block(0), jnp.conj(raw_r_adjoint.T))
    assert_allclose(right_q.block(0), jnp.conj(raw_q_adjoint.T))
    _assert_lq_contract(tensor)


@pytest.mark.parametrize("factorization", [qr_compact, lq_compact])
def test_integer_inputs_follow_jax_inexact_promotion(factorization):
    factor = space(U1Irrep, {0: 2})
    target = hom((factor,), (factor,))
    block = jnp.asarray([[1, 2], [3, 5]], dtype=jnp.int32)
    tensor = from_blocks(target, {0: block})
    expected_dtype = jnp.linalg.qr(block, mode="reduced")[0].dtype

    left, right = jax.jit(factorization)(tensor)

    assert left.dtype == right.dtype == expected_dtype
    _assert_tensors_allclose(left @ right, tensor)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.int32])
def test_empty_compact_factorizations_preserve_zero_storage_and_reconstruct(dtype):
    factor = space(U1Irrep, {0: 2})
    zero = zero_space(U1Irrep)
    qr_target = hom((factor,), (zero,))
    lq_target = hom((zero,), (factor,))
    qr_tensor = TensorMap(qr_target, jnp.zeros((0,), dtype=dtype))
    lq_tensor = TensorMap(lq_target, jnp.zeros((0,), dtype=dtype))
    expected_dtype = jnp.result_type(qr_tensor.storage.data, 0.0)

    q, r = jax.jit(qr_compact)(qr_tensor)
    l, right_q = jax.jit(lq_compact)(lq_tensor)

    assert q.storage.data.shape == r.storage.data.shape == (0,)
    assert l.storage.data.shape == right_q.storage.data.shape == (0,)
    assert q.dtype == r.dtype == l.dtype == right_q.dtype == expected_dtype
    _assert_tensors_allclose(q @ r, qr_tensor)
    _assert_tensors_allclose(l @ right_q, lq_tensor)
    _assert_tensors_allclose(q.adjoint() @ q, identity(zero, dtype=expected_dtype))
    _assert_tensors_allclose(
        right_q @ right_q.adjoint(),
        identity(zero, dtype=expected_dtype),
    )


def test_scalar_compact_factorizations_use_the_unit_connecting_sector():
    target = hom((), (), sector_type=U1Irrep)
    tensor = TensorMap(target, jnp.asarray([3.0 + 4.0j], dtype=jnp.complex64))

    q, r = _assert_qr_contract(tensor)
    l, right_q = _assert_lq_contract(tensor)

    assert q.domain[0].sectors == (((0,), 1),)
    assert r.codomain == q.domain
    assert l.domain == right_q.codomain
    assert_allclose(r.block(0), jnp.asarray([[5.0 + 0.0j]], dtype=jnp.complex64))
    assert_allclose(l.block(0), jnp.asarray([[5.0 + 0.0j]], dtype=jnp.complex64))


@pytest.mark.parametrize("factorization", [qr_compact, lq_compact])
def test_full_rank_nonzero_diagonal_gradients_are_finite_and_reconstruct(
    factorization,
):
    factor = space(U1Irrep, {0: 2})
    target = hom((factor,), (factor,))
    data = jnp.asarray([1.0, 2.0, 3.0, 5.0], dtype=jnp.float32)

    def objective(storage):
        tensor = TensorMap(target, storage)
        left, right = factorization(tensor)
        reconstructed = left @ right
        return jnp.sum(reconstructed.storage.data**2)

    gradient = jax.jit(jax.grad(objective))(data)
    assert_allclose(gradient, 2 * data)
    assert bool(jnp.all(jnp.isfinite(gradient)))


@pytest.mark.parametrize("function", [qr_compact, lq_compact, left_orth, right_orth])
def test_orthogonal_factorizations_reject_non_tensormaps(function):
    with pytest.raises(TypeError, match=f"{function.__name__}.*TensorMap"):
        function(object())


@dataclass(frozen=True)
class HermitianCase:
    name: str
    space: HomSpace


def _hermitian_cases() -> tuple[HermitianCase, ...]:
    u1 = space(U1Irrep, {0: 2, 1: 3})
    parity = space(FermionParity, {0: 2, 1: 3})
    su2 = space(SU2Irrep, {0: 2, 1: 3})
    product = space(U1SU2Irrep, {(0, 0): 2, (1, 1): 2})
    u1_left = space(U1Irrep, {0: 1, 1: 1})
    u1_right = space(U1Irrep, {0: 1, -1: 1})

    return (
        HermitianCase("u1", hom((u1,), (u1,))),
        HermitianCase("fermion-parity", hom((parity,), (parity,))),
        HermitianCase("su2", hom((su2,), (su2,))),
        HermitianCase("u1-su2-product", hom((product,), (product,))),
        HermitianCase(
            "u1-multileg",
            hom((u1_left, u1_right), (u1_left, u1_right)),
        ),
    )


def _hermitian_tensor(target: HomSpace, dtype: jnp.dtype) -> TensorMap:
    complex_dtype = jnp.issubdtype(dtype, jnp.complexfloating)
    blocks: dict[tuple[int, ...], Array] = {}
    for offset, (coupled, block) in enumerate(get_blockstructure(target).items()):
        size = block.row_dim
        row = jnp.arange(size, dtype=jnp.float32)[:, None]
        col = jnp.arange(size, dtype=jnp.float32)[None, :]
        diagonal = jnp.asarray(offset + 2, dtype=jnp.float32) + 2 * row
        real = jnp.where(row == col, diagonal, 0.05 * (row + col + 1))
        if complex_dtype:
            values = real + 0.03j * (row - col)
        else:
            values = real
        blocks[coupled] = jnp.asarray(values, dtype=dtype)
    return from_blocks(target, blocks, dtype=dtype)


@pytest.mark.parametrize("case", _hermitian_cases(), ids=lambda case: case.name)
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.complex64], ids=["real", "complex"])
def test_eigh_outputs_match_blockwise_jax_and_reconstruct(case, dtype):
    tensor = _hermitian_tensor(case.space, dtype)
    bond = fuse(tensor.domain)

    values = eigh_vals(tensor)
    diagonal, vectors = eigh_full(tensor)

    assert isinstance(values, SectorVector)
    assert isinstance(diagonal, DiagonalTensorMap)
    assert isinstance(vectors, TensorMap)
    assert values.sectors == bond.sectors
    assert diagonal.index_space == bond
    assert vectors.space == hom(tensor.codomain, (bond,))
    assert not bond.is_dual
    assert values.storage.data.dtype == jnp.real(tensor.storage.data).dtype
    assert diagonal.storage.data.dtype == jnp.real(tensor.storage.data).dtype
    assert vectors.storage.data.dtype == tensor.storage.data.dtype
    assert bool(is_hermitian(tensor))

    for coupled, block in tensor.blocks():
        expected_values, expected_vectors = jnp.linalg.eigh(
            block,
            UPLO="L",
            symmetrize_input=False,
        )
        assert_allclose(values.block(coupled), expected_values)
        assert_allclose(diagonal.diag().block(coupled), expected_values)
        assert_allclose(vectors.block(coupled), expected_vectors)
        assert bool(jnp.all(values.block(coupled)[:-1] <= values.block(coupled)[1:]))

    _assert_tensors_allclose(vectors @ diagonal @ vectors.adjoint(), tensor)
    _assert_tensors_allclose(
        vectors.adjoint() @ vectors,
        identity(bond, dtype=tensor.dtype),
    )


@pytest.mark.parametrize("scale", [0.0, 3.0], ids=["zero", "degenerate"])
def test_zero_and_degenerate_spectra_are_stable_under_jit(scale):
    factor = space(U1Irrep, {0: 3, 1: 2})
    target = hom((factor,), (factor,))
    blocks = {
        coupled: scale * jnp.eye(block.row_dim, dtype=jnp.float32)
        for coupled, block in get_blockstructure(target).items()
    }
    tensor = from_blocks(target, blocks, dtype=jnp.float32)

    eager_values = eigh_vals(tensor)
    eager_diagonal, eager_vectors = eigh_full(tensor)
    jitted_values = jax.jit(eigh_vals)(tensor)
    jitted_diagonal, jitted_vectors = jax.jit(eigh_full)(tensor)

    assert_allclose(jitted_values.storage.data, eager_values.storage.data)
    assert_allclose(jitted_diagonal.storage.data, eager_diagonal.storage.data)
    assert_allclose(jitted_vectors.storage.data, eager_vectors.storage.data)
    _assert_tensors_allclose(
        jitted_vectors @ jitted_diagonal @ jitted_vectors.adjoint(),
        tensor,
    )
    assert bool(jax.jit(is_hermitian)(tensor))


@pytest.mark.parametrize("dtype", [jnp.complex64, jnp.int32])
def test_typed_zero_space_has_empty_outputs_and_is_hermitian_under_jit(dtype):
    factor = zero_space(U1Irrep)
    target = hom((factor,), (factor,))
    tensor = TensorMap(target, jnp.zeros((0,), dtype=dtype))
    vector_dtype = jnp.result_type(tensor.storage.data, 0.0)
    value_dtype = jnp.real(jnp.zeros((), dtype=vector_dtype)).dtype

    values = jax.jit(eigh_vals)(tensor)
    diagonal, vectors = jax.jit(eigh_full)(tensor)

    assert values.sectors == ()
    assert values.storage.data.shape == (0,)
    assert values.storage.data.dtype == value_dtype
    assert diagonal.index_space == fuse(tensor.domain)
    assert diagonal.storage.data.shape == (0,)
    assert diagonal.storage.data.dtype == value_dtype
    assert vectors.storage.data.shape == (0,)
    assert vectors.storage.data.dtype == vector_dtype
    assert bool(jax.jit(is_hermitian)(tensor))
    _assert_tensors_allclose(vectors @ diagonal @ vectors.adjoint(), tensor)


def test_integer_eigh_outputs_follow_jax_inexact_promotion():
    factor = space(U1Irrep, {0: 2})
    target = hom((factor,), (factor,))
    block = jnp.asarray([[2, 1], [1, 3]], dtype=jnp.int32)
    tensor = from_blocks(target, {0: block})
    expected_values, expected_vectors = jnp.linalg.eigh(
        block,
        UPLO="L",
        symmetrize_input=False,
    )

    values = jax.jit(eigh_vals)(tensor)
    diagonal, vectors = jax.jit(eigh_full)(tensor)

    assert values.storage.data.dtype == expected_values.dtype
    assert diagonal.storage.data.dtype == expected_values.dtype
    assert vectors.dtype == expected_vectors.dtype
    assert_allclose(values.block(0), expected_values)
    assert_allclose(vectors.block(0), expected_vectors)
    _assert_tensors_allclose(vectors @ diagonal @ vectors.adjoint(), tensor)


def test_scalar_eigh_uses_unit_connecting_sector_and_real_values():
    target = hom((), (), sector_type=U1Irrep)
    tensor = TensorMap(target, jnp.asarray([3.0 + 0.0j], dtype=jnp.complex64))

    values = eigh_vals(tensor)
    diagonal, vectors = eigh_full(tensor)

    assert values.sectors == (((0,), 1),)
    assert diagonal.index_space == fuse(tensor.domain)
    assert values.storage.data.dtype == jnp.float32
    assert diagonal.storage.data.dtype == jnp.float32
    assert_allclose(values.storage.data, jnp.asarray([3.0], dtype=jnp.float32))
    _assert_tensors_allclose(vectors @ diagonal @ vectors.adjoint(), tensor)


def test_is_hermitian_has_exact_defaults_and_explicit_tolerances():
    factor = space(U1Irrep, {0: 2})
    target = hom((factor,), (factor,))
    block = jnp.asarray(
        [[2.0, 1.0 + 1.0e-4j], [1.0, 3.0]],
        dtype=jnp.complex64,
    )
    tensor = from_blocks(target, {0: block})

    exact = is_hermitian(tensor)
    tolerant = jax.jit(lambda value: is_hermitian(value, atol=2.0e-4))(tensor)

    assert isinstance(exact, Array)
    assert exact.shape == ()
    assert not bool(exact)
    assert bool(tolerant)


def test_eigh_uses_jax_lower_triangle_without_symmetrizing_or_raising():
    factor = space(U1Irrep, {0: 2})
    target = hom((factor,), (factor,))
    block = jnp.asarray([[1.0, 10.0], [2.0, 4.0]], dtype=jnp.float32)
    tensor = from_blocks(target, {0: block})

    expected_values, expected_vectors = jnp.linalg.eigh(
        block,
        UPLO="L",
        symmetrize_input=False,
    )
    symmetrized_values = jnp.linalg.eigvalsh(
        block,
        UPLO="L",
        symmetrize_input=True,
    )
    values = jax.jit(eigh_vals)(tensor)
    diagonal, vectors = jax.jit(eigh_full)(tensor)

    assert not bool(is_hermitian(tensor))
    assert_allclose(values.block(0), expected_values)
    assert_allclose(diagonal.diag().block(0), expected_values)
    assert_allclose(vectors.block(0), expected_vectors)
    assert not bool(jnp.allclose(values.block(0), symmetrized_values))


@pytest.mark.parametrize("decomposition", ["values", "full"])
def test_nondegenerate_hermitian_gradients_match_direct_reconstruction(decomposition):
    factor = space(U1Irrep, {0: 2})
    target = hom((factor,), (factor,))
    parameters = jnp.asarray([2.0, 0.5, 5.0], dtype=jnp.float32)

    def matrix_from(value):
        return jnp.asarray(
            [[value[0], value[1]], [value[1], value[2]]],
            dtype=jnp.float32,
        )

    def objective(value):
        tensor = from_blocks(target, {0: matrix_from(value)}, dtype=jnp.float32)
        if decomposition == "values":
            eigenvalues = eigh_vals(tensor)
            return jnp.sum(eigenvalues.storage.data**2)
        diagonal, vectors = eigh_full(tensor)
        reconstructed = vectors @ diagonal @ vectors.adjoint()
        return jnp.sum(reconstructed.storage.data**2)

    def direct_objective(value):
        return jnp.sum(matrix_from(value) ** 2)

    actual = jax.jit(jax.grad(objective))(parameters)
    expected = jax.grad(direct_objective)(parameters)
    assert_allclose(actual, expected)
    assert bool(jnp.all(jnp.isfinite(actual)))


@pytest.mark.parametrize("function", [eigh_vals, eigh_full, is_hermitian])
def test_hermitian_factorizations_reject_non_endomorphisms(function):
    codomain = space(U1Irrep, {0: 2})
    domain = space(U1Irrep, {0: 3})
    tensor = TensorMap(
        hom((codomain,), (domain,)),
        jnp.zeros((6,), dtype=jnp.float32),
    )

    with pytest.raises(ValueError, match=f"{function.__name__}.*endomorphism"):
        function(tensor)


@pytest.mark.parametrize("function", [eigh_vals, eigh_full, is_hermitian])
def test_hermitian_factorizations_reject_non_tensormaps(function):
    with pytest.raises(TypeError, match=f"{function.__name__}.*TensorMap"):
        function(object())
