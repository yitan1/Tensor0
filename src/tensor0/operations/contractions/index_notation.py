from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import TypeAlias, cast

from ... import _native
from ...tensor.tensor_map import TensorMap
from .primitives import tensorcontract, tensortrace

_Label: TypeAlias = str | int
_LabelGroups: TypeAlias = tuple[tuple[_Label, ...], tuple[_Label, ...]]
_StringLabelGroups: TypeAlias = tuple[tuple[str, ...], tuple[str, ...]]
_Occurrence: TypeAlias = tuple[int, int, _native.ElementarySpace]
_LeafIds: TypeAlias = tuple[int, ...]
_TraceAxes: TypeAlias = tuple[tuple[int, ...], tuple[int, ...]]
_MetadataNode: TypeAlias = tuple[_LeafIds, _LabelGroups]
_NodeValue: TypeAlias = tuple[TensorMap, _LabelGroups]
_ContractionStep: TypeAlias = tuple[_LeafIds, _LeafIds]
_StringSequence: TypeAlias = tuple[str, ...] | list[str]
_IntSequence: TypeAlias = tuple[int, ...] | list[int]
_NconLabels: TypeAlias = tuple[_IntSequence, ...] | list[_IntSequence]
_NconOutput: TypeAlias = tuple[_IntSequence, _IntSequence] | list[_IntSequence]


@dataclass(frozen=True)
class _IndexedTensor:
    tensor: TensorMap
    labels: tuple[_Label, ...]
    conjugate: bool


@dataclass(frozen=True)
class _TracePlan:
    trace_axes: _TraceAxes
    output_axes: _TraceAxes
    open_labels: _LabelGroups


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
    order: _StringSequence | None = None,
) -> TensorMap:
    """Contract tensors connected by repeated visible-index labels.

    Explicit ``order`` gives every contracted label once; otherwise operands
    are folded from left to right.
    """
    if not operands:
        raise ValueError("contract() requires at least one operand")
    indexed_operands = tuple(
        _normalize_operand(operand, position)
        for position, operand in enumerate(operands)
    )
    output_labels = _parse_output(output)
    return _contract_network(
        indexed_operands,
        output_labels,
        order=_normalize_contract_order(
            order,
            indexed_operands,
            output_labels,
        ),
    )


def ncon(
    tensors: tuple[TensorMap, ...] | list[TensorMap],
    labels: _NconLabels,
    *,
    conjugate: tuple[bool, ...] | list[bool] | None = None,
    order: _IntSequence | None = None,
    output: _NconOutput | None = None,
) -> TensorMap:
    """Contract a network labeled by signed integers.

    Positive labels identify contracted pairs and must occur twice. Negative
    labels identify open indices and must occur once. By default, positive
    labels are contracted in increasing order and open labels are ordered as
    ``-1, -2, ...`` within their effective codomain and domain.
    """
    indexed_operands = _normalize_ncon_operands(
        tensors,
        labels,
        conjugate,
    )
    label_counts: Counter[int] = Counter(
        label
        for operand in indexed_operands
        for label in cast(tuple[int, ...], operand.labels)
    )
    _validate_ncon_occurrences(label_counts)
    positive_labels = tuple(sorted(label for label in label_counts if label > 0))
    negative_labels = tuple(sorted(label for label in label_counts if label < 0))
    output_labels = _normalize_ncon_output(
        output,
        indexed_operands,
        negative_labels,
    )
    contraction_order = _normalize_ncon_order(order, positive_labels)
    return _contract_network(
        indexed_operands,
        output_labels,
        order=contraction_order,
    )


def _contract_network(
    operands: tuple[_IndexedTensor, ...],
    output_labels: _LabelGroups,
    *,
    order: tuple[_Label, ...] | None,
) -> TensorMap:
    operand_label_groups = _validate_network(
        operands,
        output_labels,
    )
    trace_plans = tuple(
        _build_trace_plan(operand.tensor, labels, operand.conjugate)
        for operand, labels in zip(
            operands,
            operand_label_groups,
            strict=True,
        )
    )
    planning_nodes: tuple[_MetadataNode, ...] = tuple(
        ((position,), plan.open_labels)
        for position, plan in enumerate(trace_plans)
    )
    steps = _contraction_steps(
        planning_nodes,
        order,
    )

    active_values: dict[_LeafIds, _NodeValue] = {}
    for position, (operand, plan) in enumerate(
        zip(operands, trace_plans, strict=True)
    ):
        active_values[(position,)] = (
            tensortrace(
                operand.tensor,
                axes=plan.trace_axes,
                output=plan.output_axes,
                conjugate=operand.conjugate,
            ),
            plan.open_labels,
        )

    for left_ids, right_ids in steps:
        left = active_values.pop(left_ids)
        right = active_values.pop(right_ids)
        merged_ids = _merge_leaf_ids(left_ids, right_ids)
        active_values[merged_ids] = _contract_values(left, right)

    result, result_labels = next(iter(active_values.values()))
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


def _build_trace_plan(
    tensor: TensorMap,
    labels: _LabelGroups,
    conjugate: bool,
) -> _TracePlan:
    """Plan local self-traces and the effective order of open labels."""
    flattened = labels[0] + labels[1]
    local_positions: dict[_Label, list[int]] = defaultdict(list)
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
            _effective_axis(tensor.space, axis, conjugate)
            for axis in range(tensor.numind)
            if axis not in traced_axes
        )
    )
    original_by_effective = {
        _effective_axis(tensor.space, axis, conjugate): axis
        for axis in range(tensor.numind)
        if axis not in traced_axes
    }
    effective_numout = tensor.numin if conjugate else tensor.numout
    output_axes = (
        tuple(
            original_by_effective[axis]
            for axis in effective_axes
            if axis < effective_numout
        ),
        tuple(
            original_by_effective[axis]
            for axis in effective_axes
            if axis >= effective_numout
        ),
    )
    open_labels = (
        tuple(flattened[axis] for axis in output_axes[0]),
        tuple(flattened[axis] for axis in output_axes[1]),
    )
    return _TracePlan(trace_axes, output_axes, open_labels)


def _contraction_steps(
    nodes: tuple[_MetadataNode, ...],
    order: tuple[_Label, ...] | None,
) -> tuple[_ContractionStep, ...]:
    if order is None:
        return _left_to_right_steps(tuple(node[0] for node in nodes))
    return _ordered_steps(nodes, order)


def _left_to_right_steps(
    component_ids: tuple[_LeafIds, ...],
) -> tuple[_ContractionStep, ...]:
    steps: list[_ContractionStep] = []
    left_ids = component_ids[0]
    for right_ids in component_ids[1:]:
        steps.append((left_ids, right_ids))
        left_ids = _merge_leaf_ids(left_ids, right_ids)
    return tuple(steps)


def _ordered_steps(
    nodes: tuple[_MetadataNode, ...],
    order: tuple[_Label, ...],
) -> tuple[_ContractionStep, ...]:
    """Plan contractions, consuming every label shared by each chosen pair."""
    state = {leaf_ids: labels for leaf_ids, labels in nodes}
    active_labels = {
        label
        for _leaf_ids, labels in nodes
        for label in _flatten_labels(labels)
    }
    remaining = [label for label in order if label in active_labels]
    steps: list[_ContractionStep] = []

    while remaining:
        label = remaining[0]
        matching = tuple(
            leaf_ids for leaf_ids, labels in sorted(state.items())
            if label in _flatten_labels(labels)
        )
        left_ids, right_ids = matching
        left_labels = state.pop(left_ids)
        right_labels = state.pop(right_ids)
        shared, next_labels = _merge_labels(left_labels, right_labels)
        steps.append((left_ids, right_ids))
        merged_ids = _merge_leaf_ids(left_ids, right_ids)
        state[merged_ids] = next_labels
        remaining = [candidate for candidate in remaining if candidate not in shared]

    _append_product_steps(tuple(state), steps)
    return tuple(steps)


def _append_product_steps(
    component_ids: tuple[_LeafIds, ...],
    steps: list[_ContractionStep],
) -> None:
    ordered_ids = tuple(sorted(component_ids))
    left_ids = ordered_ids[0]
    for right_ids in ordered_ids[1:]:
        steps.append((left_ids, right_ids))
        left_ids = _merge_leaf_ids(left_ids, right_ids)


def _merge_leaf_ids(left: _LeafIds, right: _LeafIds) -> _LeafIds:
    return tuple(sorted(left + right))


def _merge_labels(
    left: _LabelGroups,
    right: _LabelGroups,
) -> tuple[tuple[_Label, ...], _LabelGroups]:
    shared = _shared_labels(left, right)
    shared_set = set(shared)
    next_labels = (
        tuple(label for label in left[0] if label not in shared_set)
        + tuple(label for label in right[0] if label not in shared_set),
        tuple(label for label in left[1] if label not in shared_set)
        + tuple(label for label in right[1] if label not in shared_set),
    )
    return shared, next_labels


def _contract_values(left: _NodeValue, right: _NodeValue) -> _NodeValue:
    left_tensor, left_labels = left
    right_tensor, right_labels = right
    left_flat = _flatten_labels(left_labels)
    right_flat = _flatten_labels(right_labels)
    shared, next_labels = _merge_labels(left_labels, right_labels)
    shared_set = set(shared)
    right_positions = {
        label: axis for axis, label in enumerate(right_flat)
    }
    left_axes = tuple(left_flat.index(label) for label in shared)
    right_axes = tuple(right_positions[label] for label in shared)
    open_positions: dict[_Label, tuple[int, int]] = {
        label: (0, axis)
        for axis, label in enumerate(left_flat)
        if label not in shared_set
    }
    open_positions.update(
        {
            label: (1, axis)
            for axis, label in enumerate(right_flat)
            if label not in shared_set
        }
    )
    output_refs = (
        tuple(open_positions[label] for label in next_labels[0]),
        tuple(open_positions[label] for label in next_labels[1]),
    )
    return (
        tensorcontract(
            left_tensor,
            right_tensor,
            axes=(left_axes, right_axes),
            output=output_refs,
        ),
        next_labels,
    )


def _flatten_labels(labels: _LabelGroups) -> tuple[_Label, ...]:
    return labels[0] + labels[1]


def _shared_labels(
    left: _LabelGroups,
    right: _LabelGroups,
) -> tuple[_Label, ...]:
    right_set = set(_flatten_labels(right))
    return tuple(
        label for label in _flatten_labels(left) if label in right_set
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


def _parse_output(output: object) -> _StringLabelGroups:
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
    operand_label_groups: list[_LabelGroups] = []
    occurrences: dict[_Label, list[_Occurrence]] = defaultdict(list)
    for tensor_index, operand in enumerate(operands):
        if len(operand.labels) != operand.tensor.numind:
            raise ValueError(
                f"operand {tensor_index} must have one label per tensor index"
            )
        labels = (
            operand.labels[: operand.tensor.numout],
            operand.labels[operand.tensor.numout :],
        )
        operand_label_groups.append(labels)
        if operand.tensor.space.codomain.sector_spec != sector_spec:
            raise ValueError("all tensors must use the same sector family")

        for axis, label in enumerate(_flatten_labels(labels)):
            effective_space = (
                operand.tensor.space[axis].dual()
                if operand.conjugate
                else operand.tensor.space[axis]
            )
            occurrences[label].append(
                (tensor_index, axis, effective_space)
            )

    output_flat = _flatten_labels(output_labels)
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

    return tuple(operand_label_groups)


def _normalize_sequence(value: object, name: str) -> tuple[object, ...]:
    if not isinstance(value, (tuple, list)):
        raise TypeError(f"{name} must be a tuple or list")
    return tuple(value)


def _normalize_ncon_operands(
    tensors: tuple[TensorMap, ...] | list[TensorMap],
    labels: _NconLabels,
    conjugate: tuple[bool, ...] | list[bool] | None,
) -> tuple[_IndexedTensor, ...]:
    tensor_values = _normalize_sequence(tensors, "tensors")
    label_values = _normalize_sequence(labels, "labels")
    if not tensor_values:
        raise ValueError("ncon() requires at least one tensor")
    if len(label_values) != len(tensor_values):
        raise ValueError("tensors and labels must have the same length")
    if not all(isinstance(tensor, TensorMap) for tensor in tensor_values):
        raise TypeError("tensors must contain only TensorMap instances")
    tensor_maps = cast(tuple[TensorMap, ...], tensor_values)

    normalized_labels: list[tuple[int, ...]] = []
    for position, (tensor, tensor_labels) in enumerate(
        zip(tensor_maps, label_values, strict=True)
    ):
        group = _normalize_sequence(
            tensor_labels,
            f"labels for tensor {position}",
        )
        normalized_group = tuple(
            _normalize_integer_label(label, "tensor labels")
            for label in group
        )
        if len(normalized_group) != tensor.numind:
            raise ValueError(
                f"labels for tensor {position} must match its visible rank"
            )
        normalized_labels.append(normalized_group)

    conjugate_flags = _normalize_ncon_conjugate(
        conjugate,
        len(tensor_maps),
    )
    return tuple(
        _IndexedTensor(tensor, tensor_labels, conjugate_flag)
        for tensor, tensor_labels, conjugate_flag in zip(
            tensor_maps,
            normalized_labels,
            conjugate_flags,
            strict=True,
        )
    )


def _normalize_contract_order(
    order: object,
    operands: tuple[_IndexedTensor, ...],
    output_labels: _StringLabelGroups,
) -> tuple[str, ...] | None:
    if order is None:
        return None
    values = _normalize_sequence(order, "order")
    if not all(isinstance(value, str) for value in values):
        raise TypeError("order must contain only strings")
    normalized = cast(tuple[str, ...], values)
    if any(not label.isidentifier() for label in normalized):
        raise ValueError("order contains an invalid label")

    output_set = set(_flatten_labels(output_labels))
    contracted_labels = {
        label
        for operand in operands
        for label in cast(tuple[str, ...], operand.labels)
        if label not in output_set
    }
    if (
        len(normalized) != len(set(normalized))
        or set(normalized) != contracted_labels
    ):
        raise ValueError(
            "order must contain every contracted label exactly once"
        )
    return normalized


def _normalize_integer_label(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must contain only integers")
    if value == 0:
        raise ValueError("ncon label 0 is invalid")
    return value


def _normalize_ncon_conjugate(
    conjugate: object,
    tensor_count: int,
) -> tuple[bool, ...]:
    if conjugate is None:
        return (False,) * tensor_count
    values = _normalize_sequence(conjugate, "conjugate")
    if len(values) != tensor_count:
        raise ValueError("conjugate must match the number of tensors")
    if not all(isinstance(value, bool) for value in values):
        raise TypeError("conjugate must contain only bool values")
    return cast(tuple[bool, ...], values)


def _validate_ncon_occurrences(label_counts: Counter[int]) -> None:
    for label, count in label_counts.items():
        if label > 0 and count != 2:
            raise ValueError(
                f"positive label {label} must occur exactly twice"
            )
        if label < 0 and count != 1:
            raise ValueError(
                f"negative label {label} must occur exactly once"
            )


def _normalize_ncon_output(
    output: object,
    operands: tuple[_IndexedTensor, ...],
    negative_labels: tuple[int, ...],
) -> _LabelGroups:
    if output is None:
        groups: tuple[list[int], list[int]] = ([], [])
        for operand in operands:
            tensor_labels = cast(tuple[int, ...], operand.labels)
            effective_numout = (
                operand.tensor.numin
                if operand.conjugate
                else operand.tensor.numout
            )
            for axis, label in enumerate(tensor_labels):
                if label > 0:
                    continue
                effective_axis = _effective_axis(
                    operand.tensor.space,
                    axis,
                    operand.conjugate,
                )
                groups[0 if effective_axis < effective_numout else 1].append(label)
        return (
            tuple(sorted(groups[0], reverse=True)),
            tuple(sorted(groups[1], reverse=True)),
        )

    partitions = _normalize_sequence(output, "output")
    if len(partitions) != 2:
        raise ValueError("output must contain codomain and domain partitions")
    normalized: list[tuple[int, ...]] = []
    for partition in partitions:
        values = _normalize_sequence(partition, "output partitions")
        group = tuple(
            _normalize_integer_label(value, "output") for value in values
        )
        if any(label > 0 for label in group):
            raise ValueError("output may contain only negative labels")
        normalized.append(group)
    flattened = normalized[0] + normalized[1]
    if len(flattened) != len(set(flattened)):
        raise ValueError("each output label must appear exactly once")
    if set(flattened) != set(negative_labels):
        raise ValueError("output must contain every negative label exactly once")
    return normalized[0], normalized[1]


def _normalize_ncon_order(
    order: object,
    positive_labels: tuple[int, ...],
) -> tuple[int, ...]:
    if order is None:
        return positive_labels
    values = _normalize_sequence(order, "order")
    normalized = tuple(
        _normalize_integer_label(value, "order") for value in values
    )
    if any(label < 0 for label in normalized):
        raise ValueError("order may contain only positive labels")
    if (
        len(normalized) != len(set(normalized))
        or set(normalized) != set(positive_labels)
    ):
        raise ValueError("order must contain every positive label exactly once")
    return normalized


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
