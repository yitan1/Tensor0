from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from typing import TypeAlias

from jax import Array
import jax.numpy as jnp

from . import _native
from .structure.layout import get_degeneracystructure
from .tensor._subblocks import (
    gather_strided as _gather_strided,
    gather_subblock as _gather_subblock,
    scatter_add_strided as _scatter_add_strided,
    scatter_add_subblock as _scatter_add_subblock,
)
from .tensor.tensor_map import TensorMap

Permutation: TypeAlias = tuple[tuple[int, ...], tuple[int, ...]]
_TransformerCache: TypeAlias = OrderedDict[object, _native.TreeTransformer]

_TRANSFORMER_CACHE_MAXSIZE = 10_000
_TREE_BRAIDER_CACHE: _TransformerCache = OrderedDict()
_TREE_TRANSPOSER_CACHE: _TransformerCache = OrderedDict()


def permute(tensor: TensorMap, p: Permutation) -> TensorMap:
    if not isinstance(tensor, TensorMap):
        raise TypeError("permute() requires a TensorMap")

    p_codomain, p_domain = _normalize_p(tensor.space, p, "permute")
    if _is_identity_permutation(tensor.space, p_codomain, p_domain):
        return tensor

    dst_space = tensor.space.permute(p_codomain, p_domain)
    transformer = _treebraider(
        tensor.space,
        dst_space,
        p_codomain,
        p_domain,
        _identity_levels(tensor.space),
    )
    return _transform_new(tensor, dst_space, p_codomain, p_domain, transformer)


def braid(tensor: TensorMap, p: Permutation, levels: tuple[int, ...]) -> TensorMap:
    if not isinstance(tensor, TensorMap):
        raise TypeError("braid() requires a TensorMap")

    p_codomain, p_domain = _normalize_p(tensor.space, p, "braid")
    levels = _normalize_levels(tensor.space, levels)
    if _is_identity_permutation(tensor.space, p_codomain, p_domain):
        return tensor

    dst_space = tensor.space.permute(p_codomain, p_domain)
    transformer = _treebraider(tensor.space, dst_space, p_codomain, p_domain, levels)
    return _transform_new(tensor, dst_space, p_codomain, p_domain, transformer)


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
    transformer = _treetransposer(tensor.space, dst_space, p_codomain, p_domain)
    return _transform_new(tensor, dst_space, p_codomain, p_domain, transformer)


def _transform_new(
    tensor: TensorMap,
    dst_space: _native.HomSpace,
    p_codomain: tuple[int, ...],
    p_domain: tuple[int, ...],
    transformer: _native.TreeTransformer,
) -> TensorMap:
    result_dtype = _transform_result_dtype(tensor.storage.data, transformer)
    dst_data = jnp.zeros(
        (get_degeneracystructure(dst_space).total_dim,),
        dtype=result_dtype,
    )
    dst_data = _add_transform(
        dst_data,
        tensor.storage.data,
        p_codomain + p_domain,
        transformer,
        alpha=1.0,
    )
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
        src_space.static_key,
        dst_space.static_key,
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
            p_codomain,
            p_domain,
            levels_codomain,
            levels_domain,
        ),
    )


def _treetransposer(
    src_space: _native.HomSpace,
    dst_space: _native.HomSpace,
    p_codomain: tuple[int, ...],
    p_domain: tuple[int, ...],
) -> _native.TreeTransformer:
    key = (src_space.static_key, dst_space.static_key, p_codomain, p_domain)
    return _cached_transformer(
        _TREE_TRANSPOSER_CACHE,
        key,
        lambda: _native.tree_transposer(src_space, dst_space, p_codomain, p_domain),
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


def _add_transform(
    dst_data: Array,
    src_data: Array,
    p: tuple[int, ...],
    transformer: _native.TreeTransformer,
    *,
    alpha: object = 1.0,
) -> Array:
    result = jnp.asarray(dst_data)
    source = jnp.asarray(src_data)
    alpha_value = jnp.asarray(alpha, dtype=result.dtype)

    if transformer.kind == "abelian":
        return _add_abelian_transform(
            result,
            source,
            p,
            transformer.abelian_data,
            alpha_value,
        )
    if transformer.kind == "generic":
        return _add_generic_transform(
            result,
            source,
            p,
            transformer.generic_data,
            alpha_value,
        )
    raise ValueError(f"unsupported tree transformer kind {transformer.kind!r}")


def _transform_result_dtype(
    src_data: Array,
    transformer: _native.TreeTransformer,
) -> jnp.dtype:
    if transformer.kind == "abelian" and all(
        entry.coeff == 1.0 or entry.coeff == -1.0
        for entry in transformer.abelian_data
    ):
        return jnp.asarray(src_data).dtype
    return jnp.result_type(src_data, jnp.float32)


def _add_abelian_transform(
    result: Array,
    source: Array,
    p: tuple[int, ...],
    data: tuple[_native.AbelianTransformData, ...],
    alpha: Array,
) -> Array:
    for entry in data:
        block = _gather_subblock(source, entry.src, result.dtype)
        block = _transpose_block(block, p)
        coeff = jnp.asarray(entry.coeff, dtype=result.dtype)
        result = _scatter_add_subblock(result, entry.dst, alpha * coeff * block)
    return result


def _add_generic_transform(
    result: Array,
    source: Array,
    p: tuple[int, ...],
    data: tuple[_native.GenericTransformData, ...],
    alpha: Array,
) -> Array:
    for entry in data:
        basis_transform = jnp.asarray(entry.basis_transform, dtype=result.dtype)
        if (
            basis_transform.size == 1
            and len(entry.src.strides_offsets) == 1
            and len(entry.dst.strides_offsets) == 1
        ):
            result = _add_single_generic_transform(
                result,
                source,
                p,
                entry,
                basis_transform.reshape(()),
                alpha,
            )
            continue

        src_sizes = tuple(entry.src.sizes)
        src_rows = tuple(
            _gather_strided(source, src_sizes, strides, offset, result.dtype).reshape(-1)
            for strides, offset in entry.src.strides_offsets
        )
        buffer_src = jnp.stack(src_rows, axis=0)
        buffer_dst = basis_transform @ buffer_src

        for row, (strides, offset) in enumerate(entry.dst.strides_offsets):
            block = buffer_dst[row, :].reshape(src_sizes)
            block = _transpose_block(block, p)
            result = _scatter_add_strided(
                result,
                tuple(entry.dst.sizes),
                strides,
                offset,
                alpha * block,
            )
    return result


def _add_single_generic_transform(
    result: Array,
    source: Array,
    p: tuple[int, ...],
    entry: _native.GenericTransformData,
    coeff: Array,
    alpha: Array,
) -> Array:
    src_strides, src_offset = entry.src.strides_offsets[0]
    dst_strides, dst_offset = entry.dst.strides_offsets[0]
    block = _gather_strided(
        source,
        tuple(entry.src.sizes),
        src_strides,
        src_offset,
        result.dtype,
    )
    block = _transpose_block(block, p)
    return _scatter_add_strided(
        result,
        tuple(entry.dst.sizes),
        dst_strides,
        dst_offset,
        alpha * coeff * block,
    )


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


def _normalize_levels(space: _native.HomSpace, levels: object) -> tuple[int, ...]:
    if not isinstance(levels, tuple):
        raise TypeError("braid() requires levels to be a tuple of integers")
    if len(levels) != space.numind:
        raise ValueError("braid levels must match tensor rank")

    normalized: list[int] = []
    for level in levels:
        if isinstance(level, bool) or not isinstance(level, int):
            raise TypeError("braid() requires levels to contain integers")
        normalized.append(level)
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
