from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from math import prod
import sys
from typing import TypeAlias

from jax import Array
import jax.numpy as jnp

from .. import _native
from .._stride import (
    StridedCopyRecord,
    strided_copy,
)
from .._stride._plan import _contiguous_strides
from ..structure.layout import (
    _sectorstructure_key,
    get_degeneracystructure,
    get_sectorstructure,
)
from ..tensor.dense import _trivial_dense_array
from ..tensor.storage import _require_jax_storage_data
from ..tensor.tensor_map import TensorMap

Permutation: TypeAlias = tuple[tuple[int, ...], tuple[int, ...]]
_TransformerCache: TypeAlias = OrderedDict[object, _native.TreeTransformer]

_TRANSFORMER_CACHE_MAXSIZE = 10_000
_USIZE_MAX = 2 * sys.maxsize + 1
_TREE_BRAIDER_CACHE: _TransformerCache = OrderedDict()
_TREE_TRANSPOSER_CACHE: _TransformerCache = OrderedDict()


def permute(tensor: TensorMap, p: Permutation) -> TensorMap:
    if not isinstance(tensor, TensorMap):
        raise TypeError("permute() requires a TensorMap")

    p_codomain, p_domain = _normalize_p(tensor.space, p, "permute")
    if _is_identity_permutation(tensor.space, p_codomain, p_domain):
        return tensor

    dst_space = tensor.space.permute(p_codomain, p_domain)
    if tensor.space.sector_spec == _native.Trivial:
        return _apply_trivial_index_transform(
            tensor,
            dst_space,
            p_codomain,
            p_domain,
        )
    transformer = _treepermuter(
        tensor.space,
        dst_space,
        p_codomain,
        p_domain,
    )
    return _apply_tree_transform(
        tensor,
        dst_space,
        p_codomain,
        p_domain,
        transformer,
    )


def braid(tensor: TensorMap, p: Permutation, levels: tuple[int, ...]) -> TensorMap:
    if not isinstance(tensor, TensorMap):
        raise TypeError("braid() requires a TensorMap")

    p_codomain, p_domain = _normalize_p(tensor.space, p, "braid")
    levels = _normalize_levels(tensor.space, levels)
    if _is_identity_permutation(tensor.space, p_codomain, p_domain):
        return tensor

    dst_space = tensor.space.permute(p_codomain, p_domain)
    if tensor.space.sector_spec == _native.Trivial:
        return _apply_trivial_index_transform(
            tensor,
            dst_space,
            p_codomain,
            p_domain,
        )
    transformer = _treebraider(tensor.space, dst_space, p_codomain, p_domain, levels)
    return _apply_tree_transform(
        tensor,
        dst_space,
        p_codomain,
        p_domain,
        transformer,
    )


def transpose(tensor: TensorMap, p: Permutation | None = None) -> TensorMap:
    if not isinstance(tensor, TensorMap):
        raise TypeError("transpose() requires a TensorMap")

    p_codomain, p_domain = _normalize_transpose_p(tensor.space, p)
    return _transpose_normalized(tensor, p_codomain, p_domain)


def _transpose_normalized(
    tensor: TensorMap,
    p_codomain: tuple[int, ...],
    p_domain: tuple[int, ...],
) -> TensorMap:
    if _is_identity_permutation(tensor.space, p_codomain, p_domain):
        return tensor

    dst_space = tensor.space.permute(p_codomain, p_domain)
    if tensor.space.sector_spec == _native.Trivial:
        _validate_cyclic_transpose_permutation(
            tensor.space,
            p_codomain,
            p_domain,
        )
        return _apply_trivial_index_transform(
            tensor,
            dst_space,
            p_codomain,
            p_domain,
        )
    transformer = _treetransposer(tensor.space, dst_space, p_codomain, p_domain)
    return _apply_tree_transform(
        tensor,
        dst_space,
        p_codomain,
        p_domain,
        transformer,
    )


def _apply_trivial_index_transform(
    tensor: TensorMap,
    dst_space: _native.HomSpace,
    p_codomain: tuple[int, ...],
    p_domain: tuple[int, ...],
) -> TensorMap:
    data = jnp.transpose(
        _trivial_dense_array(tensor),
        p_codomain + p_domain,
    ).reshape(-1)
    return TensorMap(dst_space, data)


def _validate_cyclic_transpose_permutation(
    space: _native.HomSpace,
    p_codomain: tuple[int, ...],
    p_domain: tuple[int, ...],
) -> None:
    numout = space.numout
    rank = space.numind
    permutation = tuple(
        index if index < numout else rank - 1 - (index - numout)
        for index in p_codomain + tuple(reversed(p_domain))
    )
    if permutation and any(
        permutation[(index + 1) % rank] != (value + 1) % rank
        for index, value in enumerate(permutation)
    ):
        raise ValueError("fusion tree transpose requires a cyclic planar permutation")


def _apply_tree_transform(
    tensor: TensorMap,
    dst_space: _native.HomSpace,
    p_codomain: tuple[int, ...],
    p_domain: tuple[int, ...],
    transformer: _native.TreeTransformer,
) -> TensorMap:
    source = _require_jax_storage_data(
        tensor.storage.data,
        "tree transform",
    )
    result_dtype = _transform_result_dtype(source, transformer)
    src_degeneracy = get_degeneracystructure(tensor.space)
    dst_degeneracy = get_degeneracystructure(dst_space)
    p = p_codomain + p_domain
    kind = transformer.kind
    if kind == "abelian":
        records = _build_abelian_stride_records(
            src_degeneracy,
            dst_degeneracy,
            p,
            transformer.abelian_data,
        )
        dst_data = strided_copy(
            source,
            records=records,
            output_size=dst_degeneracy.total_dim,
            result_dtype=result_dtype,
        )
    elif kind == "generic":
        dst_data = _apply_generic_transform(
            source,
            src_degeneracy,
            dst_degeneracy,
            result_dtype,
            p,
            transformer.generic_data,
        )
    else:
        raise ValueError(f"unsupported tree transformer kind {kind!r}")
    return TensorMap(dst_space, dst_data)


def repartition(
    tensor: TensorMap,
    nout: int,
    nin: int | None = None,
) -> TensorMap:
    if not isinstance(tensor, TensorMap):
        raise TypeError("repartition() requires a TensorMap")

    p_codomain, p_domain = _repartition_p(tensor.space, nout, nin)
    return _transpose_normalized(tensor, p_codomain, p_domain)


def twist(
    tensor: TensorMap,
    indices: int | tuple[int, ...],
    inv: bool = False,
) -> TensorMap:
    if not isinstance(tensor, TensorMap):
        raise TypeError("twist() requires a TensorMap")
    normalized = _normalize_visible_indices(indices, tensor.space.numind, "twist")
    if not isinstance(inv, bool):
        raise TypeError("twist() requires inv to be a bool")
    if not normalized:
        return tensor

    if _native.twist_is_trivial(tensor.space, normalized):
        return tensor

    sectorstructure = get_sectorstructure(tensor.space)
    factors = _native.twist_subblock_factors(
        tensor.space,
        sectorstructure,
        normalized,
        inv,
    )
    if factors is None:
        return tensor

    degeneracystructure = get_degeneracystructure(tensor.space)
    subblocks = degeneracystructure.subblockstructure
    if len(factors) != len(subblocks):
        raise RuntimeError("twist factors and degeneracy structure are inconsistent")
    data = _require_jax_storage_data(tensor.storage.data, "twist()")
    records = tuple(
        StridedCopyRecord(
            logical_shape=tuple(subblock.sizes),
            source_strides=tuple(subblock.strides),
            source_offset=subblock.offset,
            destination_strides=tuple(subblock.strides),
            destination_offset=subblock.offset,
            scale=factor,
        )
        for subblock, factor in zip(subblocks, factors, strict=True)
    )
    return TensorMap(
        tensor.space,
        strided_copy(
            data,
            records=records,
            output_size=degeneracystructure.total_dim,
            result_dtype=data.dtype,
        ),
    )


def flip(
    tensor: TensorMap,
    indices: int | tuple[int, ...],
    inv: bool = False,
) -> TensorMap:
    if not isinstance(tensor, TensorMap):
        raise TypeError("flip() requires a TensorMap")
    normalized = _normalize_visible_indices(indices, tensor.space.numind, "flip")
    if not isinstance(inv, bool):
        raise TypeError("flip() requires inv to be a bool")
    if not normalized:
        return tensor

    dst_space = tensor.space.flip(normalized)
    entries = _native.flip_entries(
        tensor.space,
        dst_space,
        get_sectorstructure(tensor.space),
        get_sectorstructure(dst_space),
        normalized,
        inv,
    )
    source = _require_jax_storage_data(tensor.storage.data, "flip()")
    result_dtype = (
        source.dtype
        if all(coeff == 1.0 or coeff == -1.0 for _, coeff in entries)
        else jnp.result_type(source, jnp.float32)
    )
    src_subblocks = get_degeneracystructure(tensor.space).subblockstructure
    dst_degeneracy = get_degeneracystructure(dst_space)
    dst_subblocks = dst_degeneracy.subblockstructure
    # Toggling a fixed set of duality flags is bijective, so each destination
    # subblock is written exactly once.
    records: list[StridedCopyRecord] = []
    for src_index, (dst_index, coeff) in enumerate(entries):
        src_subblock = src_subblocks[src_index]
        dst_subblock = dst_subblocks[dst_index]
        logical_shape = tuple(src_subblock.sizes)
        if logical_shape != tuple(dst_subblock.sizes):
            raise ValueError("flip source and destination subblock shapes differ")
        records.append(
            StridedCopyRecord(
                logical_shape=logical_shape,
                source_strides=tuple(src_subblock.strides),
                source_offset=src_subblock.offset,
                destination_strides=tuple(dst_subblock.strides),
                destination_offset=dst_subblock.offset,
                scale=coeff,
            )
        )
    return TensorMap(
        dst_space,
        strided_copy(
            source,
            records=tuple(records),
            output_size=dst_degeneracy.total_dim,
            result_dtype=result_dtype,
        ),
    )


def insertleftunit(
    tensor: TensorMap,
    position: int | None = None,
    *,
    dual: bool = False,
) -> TensorMap:
    position = _normalize_unit_insertion_call(
        tensor,
        position,
        dual,
        "insertleftunit",
    )
    dst_space = tensor.space.insert_left_unit(position, dual)
    return TensorMap(dst_space, tensor.storage)


def insertrightunit(
    tensor: TensorMap,
    position: int | None = None,
    *,
    dual: bool = False,
) -> TensorMap:
    position = _normalize_unit_insertion_call(
        tensor,
        position,
        dual,
        "insertrightunit",
    )
    dst_space = tensor.space.insert_right_unit(position, dual)
    return TensorMap(dst_space, tensor.storage)


def removeunit(tensor: TensorMap, index: int) -> TensorMap:
    if not isinstance(tensor, TensorMap):
        raise TypeError("removeunit() requires a TensorMap")
    if isinstance(index, bool) or not isinstance(index, int):
        raise TypeError("removeunit() requires index to be an int")
    if index < 0 or index >= tensor.numind:
        raise ValueError(f"removeunit index is out of range for rank {tensor.numind}")

    dst_space = tensor.space.remove_unit(index)
    return TensorMap(dst_space, tensor.storage)


def _treebraider(
    src_space: _native.HomSpace,
    dst_space: _native.HomSpace,
    p_codomain: tuple[int, ...],
    p_domain: tuple[int, ...],
    levels: tuple[int, ...],
) -> _native.TreeTransformer:
    levels_codomain = tuple(levels[: src_space.numout])
    levels_domain = tuple(levels[src_space.numout :])
    key = (
        _sectorstructure_key(src_space),
        _sectorstructure_key(dst_space),
        p_codomain,
        p_domain,
        levels_codomain,
        levels_domain,
    )
    return _cached_transformer(
        _TREE_BRAIDER_CACHE,
        key,
        lambda: _native.tree_braider(
            src_space,
            dst_space,
            get_sectorstructure(src_space),
            get_sectorstructure(dst_space),
            p_codomain,
            p_domain,
            levels_codomain,
            levels_domain,
        ),
    )


def _treepermuter(
    src_space: _native.HomSpace,
    dst_space: _native.HomSpace,
    p_codomain: tuple[int, ...],
    p_domain: tuple[int, ...],
) -> _native.TreeTransformer:
    return _treebraider(
        src_space,
        dst_space,
        p_codomain,
        p_domain,
        _identity_levels(src_space),
    )


def _treetransposer(
    src_space: _native.HomSpace,
    dst_space: _native.HomSpace,
    p_codomain: tuple[int, ...],
    p_domain: tuple[int, ...],
) -> _native.TreeTransformer:
    key = (
        _sectorstructure_key(src_space),
        _sectorstructure_key(dst_space),
        p_codomain,
        p_domain,
    )
    return _cached_transformer(
        _TREE_TRANSPOSER_CACHE,
        key,
        lambda: _native.tree_transposer(
            src_space,
            dst_space,
            get_sectorstructure(src_space),
            get_sectorstructure(dst_space),
            p_codomain,
            p_domain,
        ),
    )


def _cached_transformer(
    cache: _TransformerCache,
    key: object,
    factory: Callable[[], _native.TreeTransformer],
) -> _native.TreeTransformer:
    transformer = cache.get(key)
    if transformer is not None:
        cache.move_to_end(key)
        return transformer

    transformer = factory()
    cache[key] = transformer
    cache.move_to_end(key)
    while len(cache) > _TRANSFORMER_CACHE_MAXSIZE:
        cache.popitem(last=False)
    return transformer


def _transform_result_dtype(
    src_data: Array,
    transformer: _native.TreeTransformer,
) -> jnp.dtype:
    if transformer.kind == "abelian" and transformer.has_only_unit_coefficients:
        return src_data.dtype
    return jnp.result_type(src_data, jnp.float32)


def _build_abelian_stride_records(
    src_layout: _native.DegeneracyStructure,
    dst_layout: _native.DegeneracyStructure,
    permutation: tuple[int, ...],
    data: tuple[_native.AbelianTransformData, ...],
) -> tuple[StridedCopyRecord, ...]:
    records: list[StridedCopyRecord] = []
    for entry in data:
        src_subblock = src_layout.subblockstructure[entry.src]
        dst_subblock = dst_layout.subblockstructure[entry.dst]
        logical_shape = tuple(src_subblock.sizes[index] for index in permutation)
        if logical_shape != tuple(dst_subblock.sizes):
            raise ValueError("Abelian tree transform subblock shapes are inconsistent")
        records.append(
            StridedCopyRecord(
                logical_shape=logical_shape,
                source_strides=tuple(
                    src_subblock.strides[index] for index in permutation
                ),
                source_offset=src_subblock.offset,
                destination_strides=tuple(dst_subblock.strides),
                destination_offset=dst_subblock.offset,
                scale=entry.coeff,
            )
        )
    return tuple(records)


def _apply_generic_transform(
    source: Array,
    src_layout: _native.DegeneracyStructure,
    dst_layout: _native.DegeneracyStructure,
    result_dtype: jnp.dtype,
    permutation: tuple[int, ...],
    data: tuple[_native.GenericTransformData, ...],
) -> Array:
    source_subblocks = src_layout.subblockstructure
    destination_subblocks = dst_layout.subblockstructure
    pack_records: list[StridedCopyRecord] = []
    unpack_records: list[StridedCopyRecord] = []
    groups: list[tuple[int, int, Array]] = []
    pack_offset = 0
    unpack_offset = 0
    for entry in data:
        src_indices = tuple(entry.src_indices)
        dst_indices = tuple(entry.dst_indices)
        src_sizes = tuple(source_subblocks[src_indices[0]].sizes)
        block_size = prod(src_sizes)
        group_pack_offset = pack_offset
        contiguous_strides = _contiguous_strides(src_sizes)
        for src_index in src_indices:
            subblock = source_subblocks[src_index]
            if tuple(subblock.sizes) != src_sizes:
                raise ValueError(
                    "generic tree transform source subblock shapes are inconsistent"
                )
            pack_records.append(
                StridedCopyRecord(
                    logical_shape=src_sizes,
                    source_strides=tuple(subblock.strides),
                    source_offset=subblock.offset,
                    destination_strides=contiguous_strides,
                    destination_offset=pack_offset,
                )
            )
            pack_offset += block_size

        transform = jnp.asarray(entry.transform, dtype=result_dtype)
        logical_shape = tuple(src_sizes[index] for index in permutation)
        permuted_strides = tuple(contiguous_strides[index] for index in permutation)
        for dst_index in dst_indices:
            subblock = destination_subblocks[dst_index]
            if tuple(subblock.sizes) != logical_shape:
                raise ValueError(
                    "generic tree transform destination subblock shapes are inconsistent"
                )
            unpack_records.append(
                StridedCopyRecord(
                    logical_shape=logical_shape,
                    source_strides=permuted_strides,
                    source_offset=unpack_offset,
                    destination_strides=tuple(subblock.strides),
                    destination_offset=subblock.offset,
                )
            )
            unpack_offset += block_size
        groups.append((group_pack_offset, block_size, transform))

    packed = strided_copy(
        source,
        records=tuple(pack_records),
        output_size=pack_offset,
        result_dtype=result_dtype,
    )
    batch_shape = packed.shape[:-1]
    pieces: list[Array] = []
    for source_offset, block_size, transform in groups:
        destination_row_count, source_row_count = transform.shape
        start = source_offset
        stop = start + source_row_count * block_size
        source_rows = packed[..., start:stop].reshape(
            (*batch_shape, source_row_count, block_size)
        )
        if source_row_count == 1 and destination_row_count == 1:
            destination_rows = transform.reshape(()) * source_rows
        else:
            destination_rows = transform @ source_rows
        pieces.append(destination_rows.reshape((*batch_shape, -1)))
    arena = (
        jnp.concatenate(pieces, axis=-1)
        if pieces
        else jnp.zeros((*batch_shape, 0), dtype=result_dtype)
    )
    return strided_copy(
        arena,
        records=tuple(unpack_records),
        output_size=dst_layout.total_dim,
        result_dtype=result_dtype,
    )


def _clear_tree_transformer_caches_for_tests() -> None:
    _TREE_BRAIDER_CACHE.clear()
    _TREE_TRANSPOSER_CACHE.clear()


def _normalize_p(space: _native.HomSpace, p: object, op_name: str) -> Permutation:
    if not isinstance(p, tuple) or len(p) != 2:
        raise TypeError(f"{op_name}() requires p to be a pair of integer tuples")

    p_codomain = _normalize_axis_tuple(p[0], op_name)
    p_domain = _normalize_axis_tuple(p[1], op_name)
    visible_indices = p_codomain + p_domain
    expected = set(range(space.numind))

    if len(visible_indices) != space.numind or set(visible_indices) != expected:
        raise ValueError(
            "visible index permutation must include each visible index exactly once"
        )

    return p_codomain, p_domain


def _normalize_transpose_p(space: _native.HomSpace, p: object | None) -> Permutation:
    if p is None:
        return _default_transpose_p(space)
    return _normalize_p(space, p, "transpose")


def _identity_levels(space: _native.HomSpace) -> tuple[int, ...]:
    return tuple(range(space.numind))


def _default_transpose_p(space: _native.HomSpace) -> Permutation:
    return (
        tuple(reversed(range(space.numout, space.numind))),
        tuple(reversed(range(space.numout))),
    )


def _repartition_p(
    space: _native.HomSpace,
    nout: int,
    nin: int | None,
) -> Permutation:
    nout = _normalize_count("nout", nout)
    total = space.numind
    if nin is None:
        if nout > total:
            raise ValueError("repartition nout must not exceed tensor rank")
        nin = total - nout
    else:
        nin = _normalize_count("nin", nin)

    if nout + nin != total:
        raise ValueError("repartition visible index counts must sum to tensor rank")

    codomain_indices = tuple(range(space.numout))
    domain_indices = tuple(range(space.numout, total))
    all_indices = codomain_indices + tuple(reversed(domain_indices))
    return all_indices[:nout], tuple(reversed(all_indices[nout : nout + nin]))


def _normalize_axis_tuple(value: object, op_name: str) -> tuple[int, ...]:
    if not isinstance(value, tuple):
        raise TypeError(f"{op_name}() requires p to be a pair of integer tuples")
    indices: list[int] = []
    for index in value:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError(
                f"{op_name}() requires p to contain integer visible indices"
            )
        indices.append(index)
    return tuple(indices)


def _normalize_visible_indices(
    indices: object,
    rank: int,
    operation: str,
) -> tuple[int, ...]:
    if isinstance(indices, bool):
        raise TypeError(f"{operation}() requires integer visible indices")
    if isinstance(indices, int):
        indices = (indices,)
    elif not isinstance(indices, tuple):
        raise TypeError(f"{operation}() requires one integer or a tuple of integers")

    if any(isinstance(index, bool) or not isinstance(index, int) for index in indices):
        raise TypeError(f"{operation}() requires integer visible indices")
    if any(index < 0 for index in indices):
        raise ValueError(f"{operation} visible indices must be non-negative")
    if any(index >= rank for index in indices):
        raise ValueError(
            f"{operation} visible indices are out of range for rank {rank}"
        )
    if len(set(indices)) != len(indices):
        raise ValueError(f"{operation} visible indices must be unique")
    return indices


def _normalize_unit_insertion_call(
    tensor: object,
    position: object,
    dual: object,
    operation: str,
) -> int:
    if not isinstance(tensor, TensorMap):
        raise TypeError(f"{operation}() requires a TensorMap")
    if position is None:
        position = tensor.numind
    elif isinstance(position, bool) or not isinstance(position, int):
        raise TypeError(f"{operation}() requires position to be an int or None")
    if not isinstance(dual, bool):
        raise TypeError(f"{operation}() requires dual to be a bool")
    if position < 0 or position > tensor.numind:
        raise ValueError(
            f"{operation} position is out of range for rank {tensor.numind}"
        )
    return position


def _normalize_levels(space: _native.HomSpace, levels: object) -> tuple[int, ...]:
    if not isinstance(levels, tuple):
        raise TypeError("braid() requires levels to be a tuple of integers")
    if len(levels) != space.numind:
        raise ValueError("braid levels must match tensor rank")

    normalized: list[int] = []
    for level in levels:
        if isinstance(level, bool) or not isinstance(level, int):
            raise TypeError("braid() requires levels to contain integers")
        if level < 0:
            raise ValueError("braid levels must be non-negative")
        if level > _USIZE_MAX:
            raise OverflowError(
                "braid level is too large for a platform unsigned integer"
            )
        normalized.append(int(level))
    return tuple(normalized)


def _normalize_count(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    if value < 0:
        raise ValueError("repartition visible index counts must be non-negative")
    return value


def _is_identity_permutation(
    space: _native.HomSpace,
    p_codomain: tuple[int, ...],
    p_domain: tuple[int, ...],
) -> bool:
    return p_codomain == tuple(range(space.numout)) and p_domain == tuple(
        range(space.numout, space.numind),
    )
