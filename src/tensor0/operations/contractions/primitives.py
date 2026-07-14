from __future__ import annotations

from typing import TypeAlias

import jax.numpy as jnp

from ... import _native
from ...structure.layout import get_degeneracystructure, get_sectorstructure
from ...structure.spaces import hom
from ...tensor._blocks import (
    add_to_subblock as _add_to_subblock,
    read_subblock as _read_subblock,
)
from ...tensor.tensor_map import TensorMap
from ..transforms import _treepermuter, permute, twist

AxisRef: TypeAlias = tuple[int, int]
OutputRefs: TypeAlias = tuple[tuple[AxisRef, ...], tuple[AxisRef, ...]]
TraceAxes: TypeAlias = tuple[tuple[int, ...], tuple[int, ...]]
TraceOutput: TypeAlias = tuple[tuple[int, ...], tuple[int, ...]]


def _normalize_axis_tuple(
    value: object,
    rank: int,
    tuple_label: str,
    axis_label: str,
) -> tuple[int, ...]:
    if not isinstance(value, tuple):
        raise TypeError(f"{tuple_label} must be a tuple")

    normalized: list[int] = []
    for axis in value:
        if isinstance(axis, bool) or not isinstance(axis, int):
            raise TypeError(f"{axis_label} must be an int")
        if axis < 0 or axis >= rank:
            raise ValueError(f"{axis_label} is out of range")
        normalized.append(axis)
    return tuple(normalized)


def _normalize_axes(
    value: object,
    left_rank: int,
    right_rank: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError("axes must be a tuple of length 2")

    left_axes = _normalize_axis_tuple(
        value[0],
        left_rank,
        "axes for operand 0",
        "axis for operand 0",
    )
    right_axes = _normalize_axis_tuple(
        value[1],
        right_rank,
        "axes for operand 1",
        "axis for operand 1",
    )
    if len(left_axes) != len(right_axes):
        raise ValueError("axes must contract the same number of left and right axes")
    return left_axes, right_axes


def _normalize_conjugate(value: object) -> tuple[bool, bool]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError("conjugate must be a tuple of length 2")
    if not all(isinstance(flag, bool) for flag in value):
        raise TypeError("conjugate entries must be bool")
    return value


def _normalize_output(value: object, ranks: tuple[int, int]) -> OutputRefs:
    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError("output must be a tuple of length 2")

    groups: list[tuple[AxisRef, ...]] = []
    for group in value:
        if not isinstance(group, tuple):
            raise TypeError("output partitions must be tuples")

        refs: list[AxisRef] = []
        for ref in group:
            if not isinstance(ref, tuple) or len(ref) != 2:
                raise TypeError("each output reference must be a tuple of length 2")
            operand, axis = ref
            if isinstance(operand, bool) or not isinstance(operand, int):
                raise TypeError("output operand must be an int")
            if operand not in (0, 1):
                raise ValueError("output operand must be 0 or 1")
            if isinstance(axis, bool) or not isinstance(axis, int):
                raise TypeError("output axis must be an int")
            if axis < 0 or axis >= ranks[operand]:
                raise ValueError("output axis is out of range")
            refs.append((operand, axis))
        groups.append(tuple(refs))

    return groups[0], groups[1]


def _validate_coverage(
    axes: tuple[tuple[int, ...], tuple[int, ...]],
    output: OutputRefs,
    ranks: tuple[int, int],
) -> None:
    for operand, rank in enumerate(ranks):
        partition = axes[operand] + tuple(
            axis
            for group in output
            for ref_operand, axis in group
            if ref_operand == operand
        )
        if len(partition) != rank or len(set(partition)) != rank:
            raise ValueError(
                f"each original axis of operand {operand} must appear exactly once"
            )


def _adjoint_axis(space: _native.HomSpace, axis: int) -> int:
    if axis < space.numout:
        return space.numin + axis
    return axis - space.numout


def _normalize_trace_axes(value: object, rank: int) -> TraceAxes:
    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError("axes must be a tuple of length 2")

    left_axes = _normalize_axis_tuple(
        value[0],
        rank,
        "trace partitions",
        "trace axis",
    )
    right_axes = _normalize_axis_tuple(
        value[1],
        rank,
        "trace partitions",
        "trace axis",
    )
    if len(left_axes) != len(right_axes):
        raise ValueError("axes must trace the same number of left and right axes")
    return left_axes, right_axes


def _normalize_trace_output(value: object, rank: int) -> TraceOutput:
    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError("output must be a tuple of length 2")

    return (
        _normalize_axis_tuple(
            value[0],
            rank,
            "output partitions",
            "output axis",
        ),
        _normalize_axis_tuple(
            value[1],
            rank,
            "output partitions",
            "output axis",
        ),
    )


def _validate_trace_coverage(
    axes: TraceAxes,
    output: TraceOutput,
    rank: int,
) -> None:
    partition = axes[0] + axes[1] + output[0] + output[1]
    if len(partition) != rank or len(set(partition)) != rank:
        raise ValueError("each original axis must appear exactly once")


def tensorcontract(
    left: TensorMap,
    right: TensorMap,
    *,
    axes: tuple[tuple[int, ...], tuple[int, ...]],
    output: OutputRefs,
    conjugate: tuple[bool, bool] = (False, False),
) -> TensorMap:
    if not isinstance(left, TensorMap) or not isinstance(right, TensorMap):
        raise TypeError("left and right must be TensorMap instances")

    ranks = (left.numind, right.numind)
    left_axes, right_axes = _normalize_axes(axes, *ranks)
    output_refs = _normalize_output(output, ranks)
    conjugate_flags = _normalize_conjugate(conjugate)
    _validate_coverage((left_axes, right_axes), output_refs, ranks)

    if left.space.codomain.sector_spec != right.space.codomain.sector_spec:
        raise ValueError("left and right must use the same sector family")

    left_space = (
        hom(left.space.domain, left.space.codomain)
        if conjugate_flags[0]
        else left.space
    )
    right_space = (
        hom(right.space.domain, right.space.codomain)
        if conjugate_flags[1]
        else right.space
    )

    mapped_left_axes = tuple(
        _adjoint_axis(left.space, axis) if conjugate_flags[0] else axis
        for axis in left_axes
    )
    mapped_right_axes = tuple(
        _adjoint_axis(right.space, axis) if conjugate_flags[1] else axis
        for axis in right_axes
    )
    for left_axis, right_axis in zip(
        mapped_left_axes,
        mapped_right_axes,
    ):
        if left_space[left_axis].dual() != right_space[right_axis]:
            raise ValueError("contracted axes must be dual-compatible")

    left_contracted = set(left_axes)
    right_contracted = set(right_axes)
    open_left = tuple(axis for axis in range(ranks[0]) if axis not in left_contracted)
    open_right = tuple(
        axis for axis in range(ranks[1]) if axis not in right_contracted
    )
    mapped_open_left = tuple(
        _adjoint_axis(left.space, axis) if conjugate_flags[0] else axis
        for axis in open_left
    )
    mapped_open_right = tuple(
        _adjoint_axis(right.space, axis) if conjugate_flags[1] else axis
        for axis in open_right
    )

    left_permutation = (mapped_open_left, mapped_left_axes)
    right_permutation = (mapped_right_axes, mapped_open_right)

    canonical_refs = tuple((0, axis) for axis in open_left) + tuple(
        (1, axis) for axis in open_right
    )
    canonical_positions = {
        ref: position for position, ref in enumerate(canonical_refs)
    }
    output_permutation = (
        tuple(canonical_positions[ref] for ref in output_refs[0]),
        tuple(canonical_positions[ref] for ref in output_refs[1]),
    )

    right_canonical_space = right_space.permute(*right_permutation)
    right_twist_indices = tuple(
        axis
        for axis in range(len(right_axes))
        if right_canonical_space[axis].is_dual
    )

    left_value = left.adjoint() if conjugate_flags[0] else left
    right_value = right.adjoint() if conjugate_flags[1] else right
    left_canonical = permute(left_value, left_permutation)
    right_canonical = permute(right_value, right_permutation)
    right_canonical = twist(right_canonical, right_twist_indices)
    canonical_result = left_canonical @ right_canonical
    return permute(canonical_result, output_permutation)


def tensortrace(
    tensor: TensorMap,
    *,
    axes: TraceAxes,
    output: TraceOutput,
    conjugate: bool = False,
) -> TensorMap:
    if not isinstance(tensor, TensorMap):
        raise TypeError("tensor must be a TensorMap instance")

    rank = tensor.numind
    trace_axes = _normalize_trace_axes(axes, rank)
    output_axes = _normalize_trace_output(output, rank)
    if not isinstance(conjugate, bool):
        raise TypeError("conjugate must be a bool")
    _validate_trace_coverage(trace_axes, output_axes, rank)

    source_space = (
        hom(tensor.space.domain, tensor.space.codomain)
        if conjugate
        else tensor.space
    )
    mapped_trace_axes: TraceAxes = (
        (
            tuple(_adjoint_axis(tensor.space, axis) for axis in trace_axes[0]),
            tuple(_adjoint_axis(tensor.space, axis) for axis in trace_axes[1]),
        )
        if conjugate
        else trace_axes
    )
    mapped_output_axes: TraceOutput = (
        (
            tuple(_adjoint_axis(tensor.space, axis) for axis in output_axes[0]),
            tuple(_adjoint_axis(tensor.space, axis) for axis in output_axes[1]),
        )
        if conjugate
        else output_axes
    )

    for left_axis, right_axis in zip(
        mapped_trace_axes[0],
        mapped_trace_axes[1],
    ):
        if source_space[left_axis].dual() != source_space[right_axis]:
            raise ValueError("paired trace axes must be dual-compatible")

    canonical_permutation = (
        mapped_output_axes[0] + mapped_trace_axes[0],
        mapped_output_axes[1] + mapped_trace_axes[1],
    )
    source_value = tensor.adjoint() if conjugate else tensor
    if not trace_axes[0]:
        return permute(source_value, canonical_permutation)

    canonical_space = source_space.permute(*canonical_permutation)
    num_open_out = len(mapped_output_axes[0])
    num_open_in = len(mapped_output_axes[1])

    sector_spec = source_space.codomain.sector_spec
    destination_codomain = _native.make_product_space(
        sector_spec,
        canonical_space.codomain.spaces[:num_open_out],
    )
    destination_domain = _native.make_product_space(
        sector_spec,
        canonical_space.domain.spaces[:num_open_in],
    )
    destination_space = _native.make_hom_products(
        destination_codomain,
        destination_domain,
    )

    basis_transformer = _treepermuter(
        source_space,
        canonical_space,
        *canonical_permutation,
    )
    transformer = _native.trace_transformer(
        canonical_space,
        destination_space,
        get_sectorstructure(canonical_space),
        get_sectorstructure(destination_space),
        basis_transformer,
    )
    source_degeneracystructure = get_degeneracystructure(source_space)
    destination_degeneracystructure = get_degeneracystructure(destination_space)

    source_data = jnp.asarray(source_value.storage.data)
    result_dtype = _trace_result_dtype(source_data, transformer)
    destination_data = jnp.zeros(
        (destination_degeneracystructure.total_dim,),
        dtype=result_dtype,
    )

    source_subblocks = source_degeneracystructure.subblockstructure
    destination_subblocks = destination_degeneracystructure.subblockstructure
    trace_count = len(trace_axes[0])
    flat_permutation = canonical_permutation[0] + canonical_permutation[1]
    if flat_permutation == tuple(range(rank)):
        flat_permutation = ()
    pending: dict[int, jnp.ndarray] = {}

    if transformer.kind == "abelian":
        for entry in transformer.abelian_data:
            value = _trace_source_subblock(
                source_data,
                source_subblocks[entry.src],
                result_dtype,
                flat_permutation,
                num_open_out,
                trace_count,
            )
            value = jnp.asarray(entry.coeff, dtype=result_dtype) * value
            pending[entry.dst] = (
                pending[entry.dst] + value
                if entry.dst in pending
                else value
            )
    elif transformer.kind == "generic":
        for group in transformer.generic_data:
            group_transform = group.transform
            if group_transform.size == 1:
                value = _trace_source_subblock(
                    source_data,
                    source_subblocks[group.src_indices[0]],
                    result_dtype,
                    flat_permutation,
                    num_open_out,
                    trace_count,
                )
                value = jnp.asarray(
                    group_transform.reshape(()),
                    dtype=result_dtype,
                ) * value
                destination_index = group.dst_indices[0]
                pending[destination_index] = (
                    pending[destination_index] + value
                    if destination_index in pending
                    else value
                )
                continue

            traced_rows = tuple(
                _trace_source_subblock(
                    source_data,
                    source_subblocks[source_index],
                    result_dtype,
                    flat_permutation,
                    num_open_out,
                    trace_count,
                ).reshape(-1)
                for source_index in group.src_indices
            )
            source_matrix = jnp.stack(traced_rows, axis=0)
            destination_matrix = (
                jnp.asarray(group_transform, dtype=result_dtype) @ source_matrix
            )
            for row, destination_index in enumerate(group.dst_indices):
                destination_subblock = destination_subblocks[destination_index]
                value = destination_matrix[row].reshape(
                    tuple(destination_subblock.sizes)
                )
                pending[destination_index] = (
                    pending[destination_index] + value
                    if destination_index in pending
                    else value
                )
    else:
        raise ValueError(f"unsupported trace transformer kind {transformer.kind!r}")

    for destination_index in sorted(pending):
        destination_data = _add_to_subblock(
            destination_data,
            destination_subblocks[destination_index],
            pending[destination_index],
        )

    return TensorMap(destination_space, destination_data)


def _trace_result_dtype(
    source_data: object,
    transformer: _native.TreeTransformer,
) -> jnp.dtype:
    dtype = jnp.asarray(source_data).dtype
    return (
        dtype
        if transformer.has_only_unit_coefficients
        else jnp.result_type(dtype, jnp.float32)
    )


def _trace_source_subblock(
    source_data: object,
    source_subblock: _native.SubblockStructure,
    result_dtype: jnp.dtype,
    permutation: tuple[int, ...],
    num_open_out: int,
    trace_count: int,
) -> jnp.ndarray:
    value = _read_subblock(source_data, source_subblock, result_dtype)
    if permutation:
        value = jnp.transpose(value, permutation)
    for trace_index in reversed(range(trace_count)):
        value = jnp.trace(
            value,
            axis1=num_open_out + trace_index,
            axis2=value.ndim - 1,
        )
    return value
