from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
import sys
from typing import TypeAlias

from jax import Array
import jax.numpy as jnp

from .. import _native
from ..structure.layout import (
    _sectorstructure_key,
    get_degeneracystructure,
    get_sectorstructure,
)
from ..tensor._blocks import (
    add_to_subblock as _add_to_subblock,
    get_subblock as _get_subblock,
    scale_subblock as _scale_subblock,
    set_subblock as _set_subblock,
)
from ..tensor.dense import _trivial_dense_array
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
        raise ValueError(
            "fusion tree transpose requires a cyclic planar permutation"
        )


def _apply_tree_transform(
    tensor: TensorMap,
    dst_space: _native.HomSpace,
    p_codomain: tuple[int, ...],
    p_domain: tuple[int, ...],
    transformer: _native.TreeTransformer,
) -> TensorMap:
    source = jnp.asarray(tensor.storage.data)
    result_dtype = _transform_result_dtype(source, transformer)
    src_degeneracy = get_degeneracystructure(tensor.space)
    dst_degeneracy = get_degeneracystructure(dst_space)
    dst_data = jnp.zeros(
        (dst_degeneracy.total_dim,),
        dtype=result_dtype,
    )
    p = p_codomain + p_domain
    src_subblocks = src_degeneracy.subblockstructure
    dst_subblocks = dst_degeneracy.subblockstructure
    kind = transformer.kind
    if kind == "abelian":
        dst_data = _add_abelian_transform(
            dst_data,
            source,
            p,
            transformer.abelian_data,
            src_subblocks,
            dst_subblocks,
        )
    elif kind == "generic":
        dst_data = _add_generic_transform(
            dst_data,
            source,
            p,
            transformer.generic_data,
            src_subblocks,
            dst_subblocks,
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
    data = tensor.storage.data
    for index, factor in enumerate(factors):
        if factor == 1:
            continue
        subblock = degeneracystructure.subblock_at(index)
        if subblock is None:
            raise RuntimeError("twist factors and degeneracy structure are inconsistent")
        coefficient = jnp.asarray(factor, dtype=data.dtype)
        data = _scale_subblock(data, subblock, coefficient)
    return TensorMap(tensor.space, data)


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
    source = jnp.asarray(tensor.storage.data)
    result_dtype = (
        source.dtype
        if all(coeff == 1.0 or coeff == -1.0 for _, coeff in entries)
        else jnp.result_type(source, jnp.float32)
    )
    src_subblocks = get_degeneracystructure(tensor.space).subblockstructure
    dst_degeneracy = get_degeneracystructure(dst_space)
    dst_data = jnp.zeros((dst_degeneracy.total_dim,), dtype=result_dtype)
    dst_subblocks = dst_degeneracy.subblockstructure
    # Toggling a fixed set of duality flags is bijective, so each destination
    # subblock is written exactly once.
    for src_index, (dst_index, coeff) in enumerate(entries):
        block = _get_subblock(source, src_subblocks[src_index], result_dtype)
        coefficient = jnp.asarray(coeff, dtype=result_dtype)
        dst_data = _set_subblock(
            dst_data,
            dst_subblocks[dst_index],
            coefficient * block,
        )
    return TensorMap(dst_space, dst_data)


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
        raise ValueError(
            f"removeunit index is out of range for rank {tensor.numind}"
        )

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
    if (
        transformer.kind == "abelian"
        and transformer.has_only_unit_coefficients
    ):
        return src_data.dtype
    return jnp.result_type(src_data, jnp.float32)


def _add_abelian_transform(
    result: Array,
    source: Array,
    p: tuple[int, ...],
    data: tuple[_native.AbelianTransformData, ...],
    src_subblocks: tuple[_native.SubblockStructure, ...],
    dst_subblocks: tuple[_native.SubblockStructure, ...],
) -> Array:
    for entry in data:
        block = _get_subblock(source, src_subblocks[entry.src], result.dtype)
        block = _transpose_block(block, p)
        coeff = jnp.asarray(entry.coeff, dtype=result.dtype)
        result = _add_to_subblock(
            result,
            dst_subblocks[entry.dst],
            coeff * block,
        )
    return result


def _add_generic_transform(
    result: Array,
    source: Array,
    p: tuple[int, ...],
    data: tuple[_native.GenericTransformData, ...],
    src_subblocks: tuple[_native.SubblockStructure, ...],
    dst_subblocks: tuple[_native.SubblockStructure, ...],
) -> Array:
    for entry in data:
        transform = jnp.asarray(entry.transform, dtype=result.dtype)
        src_indices = entry.src_indices
        dst_indices = entry.dst_indices
        if (
            transform.size == 1
            and len(src_indices) == 1
            and len(dst_indices) == 1
        ):
            block = _get_subblock(
                source,
                src_subblocks[src_indices[0]],
                result.dtype,
            )
            block = _transpose_block(block, p)
            result = _add_to_subblock(
                result,
                dst_subblocks[dst_indices[0]],
                transform.reshape(()) * block,
            )
            continue

        src_sizes = tuple(src_subblocks[src_indices[0]].sizes)
        src_rows = tuple(
            _get_subblock(source, src_subblocks[index], result.dtype).reshape(-1)
            for index in src_indices
        )
        buffer_src = jnp.stack(src_rows, axis=0)
        buffer_dst = transform @ buffer_src

        for row, destination_index in enumerate(dst_indices):
            block = buffer_dst[row, :].reshape(src_sizes)
            block = _transpose_block(block, p)
            result = _add_to_subblock(
                result,
                dst_subblocks[destination_index],
                block,
            )
    return result


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
        raise ValueError("visible index permutation must include each visible index exactly once")

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
            raise TypeError(f"{op_name}() requires p to contain integer visible indices")
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
        raise TypeError(
            f"{operation}() requires one integer or a tuple of integers"
        )

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
            raise OverflowError("braid level is too large for a platform unsigned integer")
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


def _transpose_block(block: Array, p: tuple[int, ...]) -> Array:
    if not p:
        return block
    return jnp.transpose(block, p)
