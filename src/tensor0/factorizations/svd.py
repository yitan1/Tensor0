from __future__ import annotations

from jax import Array
import jax.numpy as jnp

from .. import _native
from ..structure.spaces import hom
from ..tensor.sector_vector import SectorVector, _packed_sector_values
from ..tensor.tensor_map import TensorMap, _packed_vector_from_blocks
from .truncation import _ensure_strategy, _find_truncated_indices, _truncation_error, notrunc


def svd_compact(tensor: TensorMap) -> tuple[TensorMap, SectorVector, TensorMap]:
    if not isinstance(tensor, TensorMap):
        raise TypeError("svd_compact() requires a TensorMap")

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

    bond = _native.infimum_space(
        _native.fuse(tensor.space.codomain),
        _native.fuse(tensor.space.domain),
    )
    _check_svd_sector_ranks(bond, sector_ranks)
    u_space = hom(tensor.space.codomain, (bond,))
    vh_space = hom((bond,), tensor.space.domain)

    return (
        TensorMap(
            u_space,
            _packed_vector_from_blocks(u_space, u_blocks, dtype=tensor_dtype),
        ),
        SectorVector(
            bond,
            _packed_sector_values(bond, s_blocks, dtype=singular_dtype),
        ),
        TensorMap(
            vh_space,
            _packed_vector_from_blocks(vh_space, vh_blocks, dtype=tensor_dtype),
        ),
    )


def svd_trunc(
    tensor: TensorMap,
    *,
    trunc: object | None = None,
) -> tuple[TensorMap, SectorVector, TensorMap, Array]:
    trunc = notrunc() if trunc is None else _ensure_strategy(trunc)

    u, s, vh = svd_compact(tensor)
    keep = _find_truncated_indices(s, trunc)
    error = _truncation_error(s, keep)

    sector_dims = tuple(
        (sector, len(indices))
        for sector, _dim in s.space.sectors
        if (indices := keep.get(sector, ()))
    )
    bond = _native.make_space(s.space.sector_spec, sector_dims, False)
    u_space = hom(tensor.space.codomain, (bond,))
    vh_space = hom((bond,), tensor.space.domain)

    u_blocks: dict[tuple[int, ...], Array] = {}
    s_blocks: dict[tuple[int, ...], Array] = {}
    vh_blocks: dict[tuple[int, ...], Array] = {}
    for sector, _dim in bond.sectors:
        indices = jnp.asarray(keep[sector], dtype=jnp.int32)
        u_blocks[sector] = jnp.take(u.block(sector), indices, axis=1)
        s_blocks[sector] = jnp.take(s.block(sector), indices, axis=0)
        vh_blocks[sector] = jnp.take(vh.block(sector), indices, axis=0)

    tensor_dtype = jnp.asarray(tensor.storage.data).dtype
    singular_dtype = jnp.asarray(s.storage.data).dtype
    return (
        TensorMap(
            u_space,
            _packed_vector_from_blocks(u_space, u_blocks, dtype=tensor_dtype),
        ),
        SectorVector(
            bond,
            _packed_sector_values(bond, s_blocks, dtype=singular_dtype),
        ),
        TensorMap(
            vh_space,
            _packed_vector_from_blocks(vh_space, vh_blocks, dtype=tensor_dtype),
        ),
        error,
    )


def _check_svd_sector_ranks(
    bond: _native.ElementarySpace,
    sector_ranks: list[tuple[tuple[int, ...], int]],
) -> None:
    actual = tuple(sector_ranks)
    expected = bond.sectors
    if actual != expected:
        raise ValueError(
            "svd_compact() block ranks do not match fused infimum bond space: "
            f"expected {expected}, got {actual}",
        )
