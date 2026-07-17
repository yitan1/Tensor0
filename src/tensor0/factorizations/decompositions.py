from __future__ import annotations

from typing import Literal

from jax import Array
import jax.numpy as jnp

from .. import _native
from ..structure.spaces import fuse, hom, infimum
from ..tensor._blocks import pack_blocks
from ..tensor.diagonal import DiagonalTensorMap
from ..tensor.sector_vector import SectorVector, _packed_sector_values
from ..tensor.tensor_map import TensorMap
from .truncation import _ensure_strategy, _find_truncated_indices, _truncation_error, notrunc


# Orthogonal decompositions


def qr_compact(tensor: TensorMap) -> tuple[TensorMap, TensorMap]:
    """Return the compact blockwise QR factorization of a TensorMap."""
    tensor = _require_tensor_map(tensor, "qr_compact")
    bond = _compact_bond(tensor)
    blocks = tensor.blocks()

    q_blocks: dict[tuple[int, ...], Array] = {}
    r_blocks: dict[tuple[int, ...], Array] = {}
    factor_dtype = jnp.result_type(tensor.storage.data, 0.0)
    for coupled, block in blocks:
        q_block, r_block = _positive_qr(block)
        q_blocks[coupled] = q_block
        r_blocks[coupled] = r_block
        factor_dtype = q_block.dtype

    q_space = hom(tensor.space.codomain, (bond,))
    r_space = hom((bond,), tensor.space.domain)
    return (
        TensorMap(q_space, pack_blocks(q_space, q_blocks, dtype=factor_dtype)),
        TensorMap(r_space, pack_blocks(r_space, r_blocks, dtype=factor_dtype)),
    )


def lq_compact(tensor: TensorMap) -> tuple[TensorMap, TensorMap]:
    """Return the compact blockwise LQ factorization of a TensorMap."""
    tensor = _require_tensor_map(tensor, "lq_compact")
    bond = _compact_bond(tensor)
    blocks = tensor.blocks()

    l_blocks: dict[tuple[int, ...], Array] = {}
    q_blocks: dict[tuple[int, ...], Array] = {}
    factor_dtype = jnp.result_type(tensor.storage.data, 0.0)
    for coupled, block in blocks:
        q_adjoint, r_adjoint = _positive_qr(jnp.conj(block.T))
        l_blocks[coupled] = jnp.conj(r_adjoint.T)
        q_blocks[coupled] = jnp.conj(q_adjoint.T)
        factor_dtype = q_adjoint.dtype

    l_space = hom(tensor.space.codomain, (bond,))
    q_space = hom((bond,), tensor.space.domain)
    return (
        TensorMap(l_space, pack_blocks(l_space, l_blocks, dtype=factor_dtype)),
        TensorMap(q_space, pack_blocks(q_space, q_blocks, dtype=factor_dtype)),
    )


def left_orth(
    tensor: TensorMap,
    *,
    alg: Literal["qr", "svd"] | None = None,
    trunc: object | None = None,
) -> tuple[TensorMap, TensorMap]:
    """Return a left-orthogonal factorization of a TensorMap.

    Compact QR is used by default. Supplying a truncation strategy selects SVD,
    which can also be requested explicitly with ``alg="svd"``.
    """
    tensor = _require_tensor_map(tensor, "left_orth")
    selected = "svd" if alg is None and trunc is not None else alg
    if selected is None:
        selected = "qr"
    if selected == "qr":
        if trunc is not None:
            raise ValueError("left_orth() does not support truncation with alg='qr'")
        return qr_compact(tensor)
    if selected == "svd":
        u, s, vh = _orthogonal_svd(tensor, trunc)
        return u, s @ vh
    raise ValueError(f"left_orth() received unknown algorithm {selected!r}")


def right_orth(
    tensor: TensorMap,
    *,
    alg: Literal["lq", "svd"] | None = None,
    trunc: object | None = None,
) -> tuple[TensorMap, TensorMap]:
    """Return a right-orthogonal factorization of a TensorMap.

    Compact LQ is used by default. Supplying a truncation strategy selects SVD,
    which can also be requested explicitly with ``alg="svd"``.
    """
    tensor = _require_tensor_map(tensor, "right_orth")
    selected = "svd" if alg is None and trunc is not None else alg
    if selected is None:
        selected = "lq"
    if selected == "lq":
        if trunc is not None:
            raise ValueError("right_orth() does not support truncation with alg='lq'")
        return lq_compact(tensor)
    if selected == "svd":
        u, s, vh = _orthogonal_svd(tensor, trunc)
        return u @ s, vh
    raise ValueError(f"right_orth() received unknown algorithm {selected!r}")


def _orthogonal_svd(
    tensor: TensorMap,
    trunc: object | None,
) -> tuple[TensorMap, DiagonalTensorMap, TensorMap]:
    if trunc is None:
        return svd_compact(tensor)
    u, s, vh, _error = svd_trunc(tensor, trunc=trunc)
    return u, s, vh


def _positive_qr(block: Array) -> tuple[Array, Array]:
    q_block, r_block = jnp.linalg.qr(block, mode="reduced")
    diagonal = jnp.diag(r_block)
    magnitude = jnp.abs(diagonal)
    nonzero = magnitude != 0
    safe_magnitude = jnp.where(nonzero, magnitude, jnp.ones_like(magnitude))
    phase = jnp.where(
        nonzero,
        diagonal / safe_magnitude,
        jnp.ones_like(diagonal),
    )
    return q_block * phase[None, :], jnp.conj(phase)[:, None] * r_block


# Hermitian decompositions


def eigh_vals(tensor: TensorMap) -> SectorVector:
    """Return blockwise eigenvalues of a Hermitian TensorMap endomorphism.

    Hermiticity is a numerical precondition and is not checked by this function.
    """
    tensor = _require_endomorphism(tensor, "eigh_vals")
    bond = _eigh_bond(tensor)
    blocks = tensor.blocks()

    value_blocks: dict[tuple[int, ...], Array] = {}
    factor_dtype = jnp.result_type(tensor.storage.data, 0.0)
    value_dtype = jnp.real(jnp.zeros((), dtype=factor_dtype)).dtype
    for coupled, block in blocks:
        values = jnp.linalg.eigvalsh(
            block,
            UPLO="L",
            symmetrize_input=False,
        )
        value_blocks[coupled] = values
        value_dtype = values.dtype

    return SectorVector(
        bond,
        _packed_sector_values(bond, value_blocks, dtype=value_dtype),
    )


def eigh_full(tensor: TensorMap) -> tuple[DiagonalTensorMap, TensorMap]:
    """Return the full blockwise eigendecomposition of a Hermitian endomorphism.

    Hermiticity is a numerical precondition and is not checked by this function.
    """
    tensor = _require_endomorphism(tensor, "eigh_full")
    bond = _eigh_bond(tensor)
    blocks = tensor.blocks()

    value_blocks: dict[tuple[int, ...], Array] = {}
    vector_blocks: dict[tuple[int, ...], Array] = {}
    vector_dtype = jnp.result_type(tensor.storage.data, 0.0)
    value_dtype = jnp.real(jnp.zeros((), dtype=vector_dtype)).dtype
    for coupled, block in blocks:
        values, vectors = jnp.linalg.eigh(
            block,
            UPLO="L",
            symmetrize_input=False,
        )
        value_blocks[coupled] = values
        vector_blocks[coupled] = vectors
        value_dtype = values.dtype
        vector_dtype = vectors.dtype

    vector_space = hom(tensor.space.codomain, (bond,))
    return (
        DiagonalTensorMap(
            bond,
            _packed_sector_values(bond, value_blocks, dtype=value_dtype),
        ),
        TensorMap(
            vector_space,
            pack_blocks(vector_space, vector_blocks, dtype=vector_dtype),
        ),
    )


def is_hermitian(
    tensor: TensorMap,
    *,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> Array:
    """Return whether every block of a TensorMap endomorphism is Hermitian."""
    tensor = _require_endomorphism(tensor, "is_hermitian")

    result = jnp.asarray(True)
    for _coupled, block in tensor.blocks():
        result = jnp.logical_and(
            result,
            jnp.allclose(block, jnp.conj(block.T), atol=atol, rtol=rtol),
        )
    return result


def _eigh_bond(tensor: TensorMap) -> _native.ElementarySpace:
    return fuse(tensor.space.domain)


def _require_endomorphism(tensor: object, function_name: str) -> TensorMap:
    tensor_map = _require_tensor_map(tensor, function_name)
    if tensor_map.space.codomain != tensor_map.space.domain:
        raise ValueError(
            f"{function_name}() requires an endomorphism with equal codomain "
            "and domain spaces",
        )
    return tensor_map


# Singular value decompositions


def svd_vals(tensor: TensorMap) -> SectorVector:
    tensor = _require_tensor_map(tensor, "svd_vals")

    sector_ranks, s_blocks, singular_dtype = _singular_value_blocks(tensor)
    bond = _compact_bond(tensor)
    _check_svd_sector_ranks(bond, sector_ranks, "svd_vals")
    return SectorVector(
        bond,
        _packed_sector_values(bond, s_blocks, dtype=singular_dtype),
    )


def svd_compact(tensor: TensorMap) -> tuple[TensorMap, DiagonalTensorMap, TensorMap]:
    tensor = _require_tensor_map(tensor, "svd_compact")

    sector_ranks, u_blocks, s_blocks, vh_blocks, tensor_dtype, singular_dtype = (
        _compact_svd_blocks(tensor)
    )
    bond = _compact_bond(tensor)
    _check_svd_sector_ranks(bond, sector_ranks, "svd_compact")
    u_space = hom(tensor.space.codomain, (bond,))
    vh_space = hom((bond,), tensor.space.domain)

    return (
        TensorMap(
            u_space,
            pack_blocks(u_space, u_blocks, dtype=tensor_dtype),
        ),
        DiagonalTensorMap(
            bond,
            _packed_sector_values(bond, s_blocks, dtype=singular_dtype),
        ),
        TensorMap(
            vh_space,
            pack_blocks(vh_space, vh_blocks, dtype=tensor_dtype),
        ),
    )


def svd_full(tensor: TensorMap) -> tuple[TensorMap, TensorMap, TensorMap]:
    tensor = _require_tensor_map(tensor, "svd_full")

    u_blocks, s_blocks, vh_blocks, tensor_dtype, singular_dtype = _full_svd_blocks(
        tensor
    )
    fused_codomain = fuse(tensor.space.codomain)
    fused_domain = fuse(tensor.space.domain)
    u_space = hom(tensor.space.codomain, (fused_codomain,))
    s_space = hom((fused_codomain,), (fused_domain,))
    vh_space = hom((fused_domain,), tensor.space.domain)

    return (
        TensorMap(
            u_space,
            pack_blocks(u_space, u_blocks, dtype=tensor_dtype),
        ),
        TensorMap(
            s_space,
            pack_blocks(s_space, s_blocks, dtype=singular_dtype),
        ),
        TensorMap(
            vh_space,
            pack_blocks(vh_space, vh_blocks, dtype=tensor_dtype),
        ),
    )


def rank(
    tensor: TensorMap,
    *,
    atol: float = 0.0,
    rtol: float | None = None,
) -> Array:
    tensor = _require_tensor_map(tensor, "rank")
    atol_value = _nonnegative_float(atol, "atol")
    if atol_value is None:
        raise TypeError("atol must be a non-negative float")
    rtol_value = _nonnegative_float(rtol, "rtol")
    values = svd_vals(tensor)
    cutoff = _rank_cutoff(tensor, values, atol_value, rtol_value)
    total = jnp.asarray(0, dtype=jnp.int32)
    for sector, block in values.blocks():
        weight = values.sector_type.quantum_dim(sector)
        total = total + weight * jnp.sum(block > cutoff).astype(total.dtype)
    return total


def cond(tensor: TensorMap, p: float | None = 2) -> Array:
    tensor = _require_tensor_map(tensor, "cond")
    if not _is_two_norm(p):
        raise NotImplementedError("cond() currently supports only p=2 or p=None")

    values = svd_vals(tensor)
    data = values.storage.data
    if data.size == 0:
        return jnp.asarray(jnp.inf, dtype=jnp.result_type(tensor.storage.data, 0.0))

    largest = jnp.max(data)
    smallest = jnp.min(data)
    return jnp.where(
        smallest == 0,
        jnp.asarray(jnp.inf, dtype=data.dtype),
        largest / smallest,
    )


def svd_trunc(
    tensor: TensorMap,
    *,
    trunc: object | None = None,
) -> tuple[TensorMap, DiagonalTensorMap, TensorMap, Array]:
    trunc = notrunc() if trunc is None else _ensure_strategy(trunc)

    u, s, vh = svd_compact(tensor)
    values = s.diag()
    keep = _find_truncated_indices(values, trunc)
    error = _truncation_error(values, keep)

    sector_dims = tuple(
        (sector, len(indices))
        for sector, _dim in values.sectors
        if (indices := keep.get(sector, ()))
    )
    bond = _native.make_space(values.sector_type, sector_dims, False)
    u_space = hom(tensor.space.codomain, (bond,))
    vh_space = hom((bond,), tensor.space.domain)

    u_blocks: dict[tuple[int, ...], Array] = {}
    s_blocks: dict[tuple[int, ...], Array] = {}
    vh_blocks: dict[tuple[int, ...], Array] = {}
    for sector, _dim in bond.sectors:
        indices = jnp.asarray(keep[sector], dtype=jnp.int32)
        u_blocks[sector] = jnp.take(u.block(sector), indices, axis=1)
        s_blocks[sector] = jnp.take(values.block(sector), indices, axis=0)
        vh_blocks[sector] = jnp.take(vh.block(sector), indices, axis=0)

    tensor_dtype = jnp.asarray(tensor.storage.data).dtype
    singular_dtype = jnp.asarray(s.storage.data).dtype
    return (
        TensorMap(
            u_space,
            pack_blocks(u_space, u_blocks, dtype=tensor_dtype),
        ),
        DiagonalTensorMap(
            bond,
            _packed_sector_values(bond, s_blocks, dtype=singular_dtype),
        ),
        TensorMap(
            vh_space,
            pack_blocks(vh_space, vh_blocks, dtype=tensor_dtype),
        ),
        error,
    )


def _require_tensor_map(tensor: object, function_name: str) -> TensorMap:
    if not isinstance(tensor, TensorMap):
        raise TypeError(f"{function_name}() requires a TensorMap")
    return tensor


def _compact_bond(tensor: TensorMap) -> _native.ElementarySpace:
    return infimum(
        fuse(tensor.space.codomain),
        fuse(tensor.space.domain),
    )


def _singular_value_blocks(
    tensor: TensorMap,
) -> tuple[
    list[tuple[tuple[int, ...], int]],
    dict[tuple[int, ...], Array],
    jnp.dtype,
]:
    sector_ranks: list[tuple[tuple[int, ...], int]] = []
    s_blocks: dict[tuple[int, ...], Array] = {}
    singular_dtype = jnp.real(jnp.asarray(tensor.storage.data)).dtype

    for coupled, block in tensor.blocks():
        s_block = jnp.linalg.svd(block, compute_uv=False)
        rank = int(s_block.shape[0])
        singular_dtype = s_block.dtype
        if rank > 0:
            sector_ranks.append((coupled, rank))
            s_blocks[coupled] = s_block

    return sector_ranks, s_blocks, singular_dtype


def _compact_svd_blocks(
    tensor: TensorMap,
) -> tuple[
    list[tuple[tuple[int, ...], int]],
    dict[tuple[int, ...], Array],
    dict[tuple[int, ...], Array],
    dict[tuple[int, ...], Array],
    jnp.dtype,
    jnp.dtype,
]:
    sector_ranks: list[tuple[tuple[int, ...], int]] = []
    u_blocks: dict[tuple[int, ...], Array] = {}
    s_blocks: dict[tuple[int, ...], Array] = {}
    vh_blocks: dict[tuple[int, ...], Array] = {}
    tensor_dtype = jnp.asarray(tensor.storage.data).dtype
    singular_dtype = jnp.real(jnp.asarray(tensor.storage.data)).dtype

    for coupled, block in tensor.blocks():
        u_block, s_block, vh_block = jnp.linalg.svd(block, full_matrices=False)
        rank = int(s_block.shape[0])
        singular_dtype = s_block.dtype
        if rank > 0:
            sector_ranks.append((coupled, rank))
            u_blocks[coupled] = u_block
            s_blocks[coupled] = s_block
            vh_blocks[coupled] = vh_block

    return sector_ranks, u_blocks, s_blocks, vh_blocks, tensor_dtype, singular_dtype


def _full_svd_blocks(
    tensor: TensorMap,
) -> tuple[
    dict[tuple[int, ...], Array],
    dict[tuple[int, ...], Array],
    dict[tuple[int, ...], Array],
    jnp.dtype,
    jnp.dtype,
]:
    u_blocks: dict[tuple[int, ...], Array] = {}
    s_blocks: dict[tuple[int, ...], Array] = {}
    vh_blocks: dict[tuple[int, ...], Array] = {}
    tensor_dtype = jnp.asarray(tensor.storage.data).dtype
    singular_dtype = jnp.real(jnp.asarray(tensor.storage.data)).dtype

    for coupled, block in tensor.blocks():
        u_block, s_values, vh_block = jnp.linalg.svd(block, full_matrices=True)
        singular_dtype = s_values.dtype
        u_blocks[coupled] = u_block
        s_blocks[coupled] = _diagonal_block(block.shape[0], block.shape[1], s_values)
        vh_blocks[coupled] = vh_block

    return u_blocks, s_blocks, vh_blocks, tensor_dtype, singular_dtype


def _diagonal_block(row_dim: int, col_dim: int, values: Array) -> Array:
    block = jnp.zeros((row_dim, col_dim), dtype=values.dtype)
    indices = jnp.arange(values.shape[0])
    return block.at[indices, indices].set(values)


def _rank_cutoff(
    tensor: TensorMap,
    values: SectorVector,
    atol: float,
    rtol: float | None,
) -> Array:
    data = values.storage.data
    if data.size == 0:
        return jnp.asarray(atol, dtype=jnp.result_type(tensor.storage.data, 0.0))

    singular_dtype = data.dtype
    default_rtol = _largest_block_dim(tensor) * jnp.finfo(singular_dtype).eps
    rtol_value = default_rtol if rtol is None else rtol
    return jnp.maximum(
        jnp.asarray(atol, dtype=singular_dtype),
        jnp.asarray(rtol_value, dtype=singular_dtype) * jnp.max(data),
    )


def _largest_block_dim(tensor: TensorMap) -> int:
    return max((max(block.shape) for _coupled, block in tensor.blocks()), default=0)


def _nonnegative_float(value: float | None, argument_name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError(f"{argument_name} must be a non-negative float")
    result = float(value)
    if result < 0:
        raise ValueError(f"{argument_name} must be non-negative")
    return result


def _is_two_norm(p: float | None) -> bool:
    if p is None:
        return True
    if isinstance(p, bool):
        return False
    try:
        return float(p) == 2.0
    except (TypeError, ValueError):
        return False


def _check_svd_sector_ranks(
    bond: _native.ElementarySpace,
    sector_ranks: list[tuple[tuple[int, ...], int]],
    function_name: str,
) -> None:
    actual = tuple(sector_ranks)
    expected = bond.sectors
    if actual != expected:
        raise ValueError(
            f"{function_name}() block ranks do not match fused infimum bond space: "
            f"expected {expected}, got {actual}",
        )
