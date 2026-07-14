from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TypeAlias

from ... import _native
from ...tensor.tensor_map import TensorMap
from .primitives import tensorcontract, tensortrace

_LabelGroups: TypeAlias = tuple[tuple[str, ...], tuple[str, ...]]
_Occurrence: TypeAlias = tuple[int, int, _native.ElementarySpace]


@dataclass(frozen=True)
class _IndexedTensor:
    tensor: TensorMap
    labels: tuple[str, ...]
    conjugate: bool


_OperandInput: TypeAlias = (
    _IndexedTensor
    | tuple[TensorMap, str]
    | tuple[TensorMap, str, bool]
)


def idx(
    tensor: TensorMap,
    labels: str,
    *,
    conjugate: bool = False,
) -> _IndexedTensor:
    """Bind comma-separated visible-index labels to a contraction operand."""
    if not isinstance(tensor, TensorMap):
        raise TypeError("tensor must be a TensorMap")
    if not isinstance(labels, str):
        raise TypeError("labels must be a string")
    if not isinstance(conjugate, bool):
        raise TypeError("conjugate must be a bool")
    return _IndexedTensor(
        tensor,
        _parse_labels(labels, "labels"),
        conjugate,
    )


def contract(
    *operands: _OperandInput,
    output: str | tuple[str, str],
) -> TensorMap:
    """Contract tensors connected by repeated visible-index labels."""
    if not operands:
        raise ValueError("contract() requires at least one operand")
    normalized_operands = tuple(
        _normalize_operand(operand, position)
        for position, operand in enumerate(operands)
    )
    output_labels = _parse_output(output)
    input_labels = _validate_network(
        normalized_operands,
        output_labels,
    )
    tensors = tuple(operand.tensor for operand in normalized_operands)
    conjugate_flags = tuple(
        operand.conjugate for operand in normalized_operands
    )

    values: list[TensorMap] = []
    value_labels: list[_LabelGroups] = []
    for tensor, labels, flag in zip(tensors, input_labels, conjugate_flags):
        flattened = labels[0] + labels[1]
        local_positions: dict[str, list[int]] = defaultdict(list)
        for axis, label in enumerate(flattened):
            local_positions[label].append(axis)
        pairs = tuple(
            tuple(positions)
            for positions in local_positions.values()
            if len(positions) == 2
        )
        trace_axes = (
            tuple(pair[0] for pair in pairs),
            tuple(pair[1] for pair in pairs),
        )
        traced_axes = {axis for pair in pairs for axis in pair}
        effective_axes = tuple(
            sorted(
                (
                    _effective_axis(tensor.space, axis, flag)
                    for axis in range(tensor.numind)
                    if axis not in traced_axes
                ),
            )
        )
        original_by_effective = {
            _effective_axis(tensor.space, axis, flag): axis
            for axis in range(tensor.numind)
            if axis not in traced_axes
        }
        output_axes = (
            tuple(
                original_by_effective[axis]
                for axis in effective_axes
                if axis < (tensor.numin if flag else tensor.numout)
            ),
            tuple(
                original_by_effective[axis]
                for axis in effective_axes
                if axis >= (tensor.numin if flag else tensor.numout)
            ),
        )
        current_labels = (
            tuple(flattened[axis] for axis in output_axes[0]),
            tuple(flattened[axis] for axis in output_axes[1]),
        )
        values.append(
            tensortrace(
                tensor,
                axes=trace_axes,
                output=output_axes,
                conjugate=flag,
            )
        )
        value_labels.append(current_labels)

    result = values[0]
    result_labels = value_labels[0]
    for right, right_labels in zip(values[1:], value_labels[1:]):
        left_flat = result_labels[0] + result_labels[1]
        right_flat = right_labels[0] + right_labels[1]
        right_positions = {label: axis for axis, label in enumerate(right_flat)}
        shared = tuple(label for label in left_flat if label in right_positions)
        shared_set = set(shared)
        left_axes = tuple(left_flat.index(label) for label in shared)
        right_axes = tuple(right_positions[label] for label in shared)

        left_open: dict[str, tuple[int, int]] = {
            label: (0, axis)
            for axis, label in enumerate(left_flat)
            if label not in shared_set
        }
        right_open: dict[str, tuple[int, int]] = {
            label: (1, axis)
            for axis, label in enumerate(right_flat)
            if label not in shared_set
        }
        next_labels = (
            tuple(label for label in result_labels[0] if label not in shared_set)
            + tuple(label for label in right_labels[0] if label not in shared_set),
            tuple(label for label in result_labels[1] if label not in shared_set)
            + tuple(label for label in right_labels[1] if label not in shared_set),
        )
        open_positions = left_open | right_open
        output_refs = (
            tuple(open_positions[label] for label in next_labels[0]),
            tuple(open_positions[label] for label in next_labels[1]),
        )
        result = tensorcontract(
            result,
            right,
            axes=(left_axes, right_axes),
            output=output_refs,
        )
        result_labels = next_labels

    flattened_result = result_labels[0] + result_labels[1]
    result_positions = {
        label: axis for axis, label in enumerate(flattened_result)
    }
    return tensortrace(
        result,
        axes=((), ()),
        output=(
            tuple(result_positions[label] for label in output_labels[0]),
            tuple(result_positions[label] for label in output_labels[1]),
        ),
    )


def _normalize_operand(operand: _OperandInput, position: int) -> _IndexedTensor:
    if isinstance(operand, _IndexedTensor):
        return operand
    if not isinstance(operand, tuple) or len(operand) not in (2, 3):
        raise TypeError(
            f"operand {position} must be created by idx() or be a tuple "
            "of (TensorMap, labels[, conjugate])"
        )

    if len(operand) == 2:
        tensor, labels = operand
        return idx(tensor, labels)

    tensor, labels, conjugate = operand
    return idx(tensor, labels, conjugate=conjugate)


def _parse_output(output: object) -> _LabelGroups:
    if isinstance(output, str):
        if output.count(";") != 1:
            raise ValueError("output string must contain exactly one ';'")
        codomain, domain = output.split(";")
    elif isinstance(output, tuple) and len(output) == 2:
        codomain, domain = output
        if not isinstance(codomain, str) or not isinstance(domain, str):
            raise TypeError("output tuple entries must be strings")
    else:
        raise TypeError("output must be a string or a tuple of two strings")
    return (
        _parse_labels(codomain, "output codomain"),
        _parse_labels(domain, "output domain"),
    )


def _parse_labels(text: str, clause_name: str) -> tuple[str, ...]:
    if not text.strip():
        return ()
    labels = tuple(label.strip() for label in text.split(","))
    for label in labels:
        if not label.isidentifier():
            raise ValueError(
                f"{clause_name} contains invalid label {label!r}"
            )
    return labels


def _validate_network(
    operands: tuple[_IndexedTensor, ...],
    output_labels: _LabelGroups,
) -> tuple[_LabelGroups, ...]:
    sector_spec = operands[0].tensor.space.codomain.sector_spec
    input_labels: list[_LabelGroups] = []
    occurrences: dict[str, list[_Occurrence]] = defaultdict(list)
    for tensor_index, operand in enumerate(operands):
        tensor = operand.tensor
        if len(operand.labels) != tensor.numind:
            raise ValueError(
                f"operand {tensor_index} must have one label per tensor index"
            )
        labels = (
            operand.labels[: tensor.numout],
            operand.labels[tensor.numout :],
        )
        input_labels.append(labels)
        if tensor.space.codomain.sector_spec != sector_spec:
            raise ValueError("all tensors must use the same sector family")

        for axis, label in enumerate(labels[0] + labels[1]):
            effective_space = (
                tensor.space[axis].dual()
                if operand.conjugate
                else tensor.space[axis]
            )
            occurrences[label].append(
                (tensor_index, axis, effective_space)
            )

    output_flat = output_labels[0] + output_labels[1]
    if len(output_flat) != len(set(output_flat)):
        raise ValueError("each output label must appear exactly once")
    output_set = set(output_flat)
    for label in output_flat:
        count = len(occurrences.get(label, ()))
        if count != 1:
            raise ValueError(
                f"output label {label!r} must occur exactly once in the inputs"
            )

    for label, entries in occurrences.items():
        if label in output_set:
            continue
        if len(entries) != 2:
            raise ValueError(
                f"contracted label {label!r} must occur exactly twice"
            )
        if entries[0][2].dual() != entries[1][2]:
            raise ValueError(
                f"contracted label {label!r} must join dual-compatible spaces"
            )

    return tuple(input_labels)


def _effective_axis(
    space: _native.HomSpace,
    axis: int,
    conjugate: bool,
) -> int:
    if not conjugate:
        return axis
    if axis < space.numout:
        return space.numin + axis
    return axis - space.numout
