from __future__ import annotations

from typing import TypeAlias

from . import _native
from .structure.spaces import hom
from .tensor.tensor_map import TensorMap
from .transforms import permute, twist

AxisRef: TypeAlias = tuple[int, int]
OutputRefs: TypeAlias = tuple[tuple[AxisRef, ...], tuple[AxisRef, ...]]


def _normalize_axis_tuple(
    value: object,
    rank: int,
    operand: int,
) -> tuple[int, ...]:
    if not isinstance(value, tuple):
        raise TypeError(f"axes for operand {operand} must be a tuple")

    normalized: list[int] = []
    for axis in value:
        if isinstance(axis, bool) or not isinstance(axis, int):
            raise TypeError(f"axis for operand {operand} must be an int")
        if axis < 0 or axis >= rank:
            raise ValueError(f"axis for operand {operand} is out of range")
        normalized.append(axis)
    return tuple(normalized)


def _normalize_axes(
    value: object,
    left_rank: int,
    right_rank: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError("axes must be a tuple of length 2")

    left_axes = _normalize_axis_tuple(value[0], left_rank, 0)
    right_axes = _normalize_axis_tuple(value[1], right_rank, 1)
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
        strict=True,
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
