"""Private strided execution adapters for Tensor0 operations."""

from __future__ import annotations

from collections.abc import Iterable
from math import prod
from typing import TypeAlias

from jax import Array
import jax.numpy as jnp

from .. import _native
from .._stride._plan import (
    CompleteMode,
    AffinePlan,
    AffineRecord,
    StridedReductionKind,
    StridedWriteKind,
    build_affine_plan,
    contiguous_strides,
)
from .._stride._map import _execute_map
from .._stride._ops._reduction import _execute_reduction

_StridedCoefficient: TypeAlias = int | float | complex
_StridedEntry: TypeAlias = tuple[int, int, _StridedCoefficient]
_TraceCoefficient: TypeAlias = _StridedCoefficient
_TraceEntry: TypeAlias = _StridedEntry


def _strided_affine_transform(
    source: Array,
    *,
    source_subblocks: tuple[_native.SubblockStructure, ...],
    destination_subblocks: tuple[_native.SubblockStructure, ...],
    entries: Iterable[_StridedEntry],
    permutation: tuple[int, ...],
    output_size: int,
    result_dtype: jnp.dtype,
    shape_error: str,
) -> Array:
    """Execute one complete affine transform over packed subblocks."""

    records: list[AffineRecord] = []
    for source_index, destination_index, coefficient in entries:
        source_subblock = source_subblocks[source_index]
        destination_subblock = destination_subblocks[destination_index]
        if permutation:
            logical_shape = tuple(
                source_subblock.sizes[axis] for axis in permutation
            )
            source_strides = tuple(
                source_subblock.strides[axis] for axis in permutation
            )
        else:
            logical_shape = tuple(source_subblock.sizes)
            source_strides = tuple(source_subblock.strides)
        if logical_shape != tuple(destination_subblock.sizes):
            raise ValueError(shape_error)
        records.append(
            AffineRecord(
                logical_shape=logical_shape,
                source_strides=source_strides,
                source_offset=source_subblock.offset,
                destination_strides=tuple(destination_subblock.strides),
                destination_offset=destination_subblock.offset,
                scale=coefficient,
            )
        )
    plan = build_affine_plan(
        records=tuple(records),
        output_size=output_size,
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=source.shape[-1],
        source_dtype=source.dtype,
        result_dtype=result_dtype,
    )
    return _execute_map(source, plan=plan)


def _strided_tree_transform(
    source: Array,
    *,
    source_layout: _native.DegeneracyStructure,
    destination_layout: _native.DegeneracyStructure,
    result_dtype: jnp.dtype,
    permutation: tuple[int, ...],
    transformer: _native.TreeTransformer,
) -> Array:
    """Execute one symmetry tree transform through strided adapters."""

    if transformer.kind == "abelian":
        return _strided_affine_transform(
            source,
            source_subblocks=source_layout.subblockstructure,
            destination_subblocks=destination_layout.subblockstructure,
            entries=(
                (entry.src, entry.dst, entry.coeff)
                for entry in transformer.abelian_data
            ),
            permutation=permutation,
            output_size=destination_layout.total_dim,
            result_dtype=result_dtype,
            shape_error="Abelian tree transform subblock shapes are inconsistent",
        )
    if transformer.kind == "generic":
        return _strided_grouped_transform(
            source,
            source_layout,
            destination_layout,
            result_dtype,
            permutation,
            transformer.generic_data,
        )
    raise ValueError(f"unsupported tree transformer kind {transformer.kind!r}")


def _strided_grouped_transform(
    source: Array,
    source_layout: _native.DegeneracyStructure,
    destination_layout: _native.DegeneracyStructure,
    result_dtype: jnp.dtype,
    permutation: tuple[int, ...],
    data: tuple[_native.GenericTransformData, ...],
) -> Array:
    source_subblocks = source_layout.subblockstructure
    destination_subblocks = destination_layout.subblockstructure
    pack_records: list[AffineRecord] = []
    unpack_records: list[AffineRecord] = []
    groups: list[tuple[int, int, Array]] = []
    pack_offset = 0
    unpack_offset = 0
    for entry in data:
        source_indices = tuple(entry.src_indices)
        destination_indices = tuple(entry.dst_indices)
        source_sizes = tuple(source_subblocks[source_indices[0]].sizes)
        block_size = prod(source_sizes)
        group_pack_offset = pack_offset
        source_contiguous_strides = contiguous_strides(source_sizes)
        for source_index in source_indices:
            subblock = source_subblocks[source_index]
            if tuple(subblock.sizes) != source_sizes:
                raise ValueError(
                    "generic tree transform source subblock shapes are inconsistent"
                )
            pack_records.append(
                AffineRecord(
                    logical_shape=source_sizes,
                    source_strides=tuple(subblock.strides),
                    source_offset=subblock.offset,
                    destination_strides=source_contiguous_strides,
                    destination_offset=pack_offset,
                )
            )
            pack_offset += block_size

        transform = jnp.asarray(entry.transform, dtype=result_dtype)
        logical_shape = tuple(source_sizes[index] for index in permutation)
        permuted_strides = tuple(
            source_contiguous_strides[index] for index in permutation
        )
        for destination_index in destination_indices:
            subblock = destination_subblocks[destination_index]
            if tuple(subblock.sizes) != logical_shape:
                raise ValueError(
                    "generic tree transform destination subblock shapes "
                    "are inconsistent"
                )
            unpack_records.append(
                AffineRecord(
                    logical_shape=logical_shape,
                    source_strides=permuted_strides,
                    source_offset=unpack_offset,
                    destination_strides=tuple(subblock.strides),
                    destination_offset=subblock.offset,
                )
            )
            unpack_offset += block_size
        groups.append((group_pack_offset, block_size, transform))

    pack_plan = build_affine_plan(
        records=tuple(pack_records),
        output_size=pack_offset,
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=source.shape[-1],
        source_dtype=source.dtype,
        result_dtype=result_dtype,
    )
    packed = _execute_map(source, plan=pack_plan)
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
    unpack_plan = build_affine_plan(
        records=tuple(unpack_records),
        output_size=destination_layout.total_dim,
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=arena.shape[-1],
        source_dtype=arena.dtype,
        result_dtype=result_dtype,
    )
    return _execute_map(arena, plan=unpack_plan)


def _strided_tensortrace(
    source: Array,
    *,
    destination_size: int,
    source_subblocks: tuple[_native.SubblockStructure, ...],
    destination_subblocks: tuple[_native.SubblockStructure, ...],
    entries: Iterable[_TraceEntry],
    result_dtype: jnp.dtype,
    permutation: tuple[int, ...],
    num_open_out: int,
    num_open_in: int,
    trace_count: int,
) -> Array:
    """Execute fused tensor-trace terms through one strided reduction."""

    plan = _build_strided_trace_plan(
        source,
        destination_size,
        source_subblocks,
        destination_subblocks,
        entries,
        result_dtype,
        permutation,
        num_open_out,
        num_open_in,
        trace_count,
    )
    if plan is None:
        return jnp.zeros((destination_size,), dtype=result_dtype)
    return _execute_reduction(source, plan=plan)


def _build_strided_trace_plan(
    source: Array,
    destination_size: int,
    source_subblocks: tuple[_native.SubblockStructure, ...],
    destination_subblocks: tuple[_native.SubblockStructure, ...],
    entries: Iterable[_TraceEntry],
    result_dtype: jnp.dtype,
    permutation: tuple[int, ...],
    num_open_out: int,
    num_open_in: int,
    trace_count: int,
) -> AffinePlan | None:
    records: list[AffineRecord] = []
    for source_index, destination_index, coefficient in entries:
        source_subblock = source_subblocks[source_index]
        destination_subblock = destination_subblocks[destination_index]
        source_sizes = tuple(source_subblock.sizes)
        source_strides = tuple(source_subblock.strides)
        if permutation:
            source_sizes = tuple(source_sizes[axis] for axis in permutation)
            source_strides = tuple(source_strides[axis] for axis in permutation)

        open_rank = num_open_out + num_open_in
        trace_input_start = num_open_out
        domain_open_start = trace_input_start + trace_count
        trace_output_start = domain_open_start + num_open_in
        open_axes = (
            *range(num_open_out),
            *range(domain_open_start, domain_open_start + num_open_in),
        )
        logical_shape = tuple(source_sizes[axis] for axis in open_axes)
        logical_source_strides = tuple(source_strides[axis] for axis in open_axes)
        for trace_index in range(trace_count):
            left_axis = trace_input_start + trace_index
            right_axis = trace_output_start + trace_index
            if source_sizes[left_axis] != source_sizes[right_axis]:
                raise ValueError("trace source subblock axes have inconsistent sizes")
            logical_shape += (source_sizes[left_axis],)
            logical_source_strides += (
                source_strides[left_axis] + source_strides[right_axis],
            )
        destination_sizes = tuple(destination_subblock.sizes)
        if logical_shape[:open_rank] != destination_sizes:
            raise ValueError("trace destination subblock shape is inconsistent")
        reduction_axes = tuple(range(open_rank, open_rank + trace_count))
        records.append(
            AffineRecord(
                logical_shape=logical_shape,
                source_strides=logical_source_strides,
                source_offset=source_subblock.offset,
                destination_strides=(
                    *tuple(destination_subblock.strides),
                    *(0 for _ in reduction_axes),
                ),
                destination_offset=destination_subblock.offset,
                scale=coefficient,
                reduction_axes=reduction_axes,
            )
        )
    if not records:
        return None
    return build_affine_plan(
        records=tuple(records),
        output_size=destination_size,
        coverage=CompleteMode.PARTIAL_UNIQUE_ZERO_FILL,
        source_size=source.shape[-1],
        source_dtype=source.dtype,
        result_dtype=result_dtype,
        write_kind=(
            StridedWriteKind.ACCUMULATE
            if len(records) > 1
            else StridedWriteKind.ASSIGN
        ),
        reduction_kind=StridedReductionKind.SUM,
    )
