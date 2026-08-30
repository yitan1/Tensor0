"""Bounded pure-JAX oracles for stride differential tests."""

from __future__ import annotations

from itertools import product
from math import prod
from typing import Any

import jax
from jax import Array
from jax.core import Tracer
import jax.numpy as jnp
import numpy as np

from tensor0._stride._errors import raise_no_eligible_route
from tensor0._stride._plan import (
    StridedCopyPlan,
    StridedCopyRecord,
    StridedWriteKind,
)


REFERENCE_PYTHON_ADDRESS_BYTE_CAP = 64 * 1024 * 1024
REFERENCE_LIVE_BYTE_CAP = 256 * 1024 * 1024
REFERENCE_RECORD_CAP = 1_024


def assert_bitwise_equal(actual: object, expected: object) -> None:
    actual_array = np.asarray(actual)
    expected_array = np.asarray(expected)
    assert actual_array.dtype == expected_array.dtype
    assert actual_array.shape == expected_array.shape
    np.testing.assert_array_equal(
        actual_array.reshape(-1).view(np.uint8),
        expected_array.reshape(-1).view(np.uint8),
    )


def enumerate_addresses(
    record: StridedCopyRecord,
    side: str,
) -> tuple[int, ...]:
    """Exhaustively enumerate one small affine record."""

    if side == "source":
        strides = record.source_strides
        offset = record.source_offset
    elif side == "destination":
        strides = record.destination_strides
        offset = record.destination_offset
    else:
        raise AssertionError(f"unknown view side {side!r}")
    if 0 in record.logical_shape:
        return ()
    coordinates = product(*(range(extent) for extent in record.logical_shape))
    return tuple(
        offset
        + sum(index * stride for index, stride in zip(q, strides, strict=True))
        for q in coordinates
    )


def legacy_strided_indices(
    sizes: tuple[int, ...],
    strides: tuple[int, ...],
    offset: int,
) -> Array:
    """Materialize affine addresses for small differential-test cases."""

    index_dtype = jnp.result_type(offset)
    indices = jnp.zeros(sizes, dtype=index_dtype)
    for axis, (size, stride) in enumerate(zip(sizes, strides, strict=True)):
        shape = (1,) * axis + (size,) + (1,) * (len(sizes) - axis - 1)
        indices = indices + jnp.arange(size, dtype=index_dtype).reshape(shape) * stride
    return (indices + offset).reshape(-1)


def legacy_materialize(
    storage: object,
    sizes: tuple[int, ...],
    strides: tuple[int, ...],
    offset: int,
    dtype: jnp.dtype | None = None,
) -> Array:
    """Read through explicit addresses as a bounded correctness oracle."""

    block = jnp.asarray(storage)[legacy_strided_indices(sizes, strides, offset)]
    if dtype is not None:
        block = jnp.asarray(block, dtype=dtype)
    return block.reshape(sizes)


def legacy_strided_assign(
    storage: Array,
    sizes: tuple[int, ...],
    strides: tuple[int, ...],
    offset: int,
    value: Array,
) -> Array:
    indices = legacy_strided_indices(sizes, strides, offset)
    return storage.at[indices].set(value.reshape(-1))


def legacy_strided_accumulate(
    storage: Array,
    sizes: tuple[int, ...],
    strides: tuple[int, ...],
    offset: int,
    value: Array,
) -> Array:
    indices = legacy_strided_indices(sizes, strides, offset)
    return storage.at[indices].add(value.reshape(-1))


def legacy_strided_scale(
    storage: Array,
    sizes: tuple[int, ...],
    strides: tuple[int, ...],
    offset: int,
    factor: Array,
) -> Array:
    indices = legacy_strided_indices(sizes, strides, offset)
    return storage.at[indices].multiply(factor)


def reference_ineligibility_reasons(
    bound: StridedCopyPlan,
    *,
    source_shape: tuple[int, ...] | None = None,
) -> tuple[str, ...]:
    """Return bounded-resource reasons that forbid materialized indices."""

    reasons: list[str] = []
    copied_elements = bound.copied_elements
    index_width = 8 if bool(jax.config.read("jax_enable_x64")) else 4
    logical_index_bytes = 2 * copied_elements * index_width
    if 2 * copied_elements * 36 > REFERENCE_PYTHON_ADDRESS_BYTE_CAP:
        reasons.append("reference_python_address_byte_cap")
    if len(bound.records) > REFERENCE_RECORD_CAP:
        reasons.append("reference_record_cap")
    if source_shape is not None:
        batch_count = prod(source_shape[:-1])
        source_bytes = (
            batch_count * bound.source_size * jnp.dtype(bound.source_dtype).itemsize
        )
        result_bytes = (
            batch_count * bound.output_size * jnp.dtype(bound.result_dtype).itemsize
        )
        update_bytes = (
            batch_count * copied_elements * jnp.dtype(bound.result_dtype).itemsize
        )
        live_bytes = (
            source_bytes + 2 * result_bytes + update_bytes + logical_index_bytes
        )
        if live_bytes > REFERENCE_LIVE_BYTE_CAP:
            reasons.append("reference_live_byte_cap")
    if index_width == 4:
        maximum_address = max(
            (
                max(
                    record.source_offset
                    + sum(
                        max(0, (extent - 1) * stride)
                        for extent, stride in zip(
                            record.logical_shape,
                            record.source_strides,
                            strict=True,
                        )
                    ),
                    record.destination_offset
                    + sum(
                        max(0, (extent - 1) * stride)
                        for extent, stride in zip(
                            record.logical_shape,
                            record.destination_strides,
                            strict=True,
                        )
                    ),
                )
                for record in bound.records
            ),
            default=0,
        )
        if maximum_address > np.iinfo(np.int32).max:
            reasons.append("reference_index_width")
    return tuple(reasons)


def _check_eligibility(
    bound: StridedCopyPlan,
    source_shape: tuple[int, ...],
) -> None:
    reasons = reference_ineligibility_reasons(bound, source_shape=source_shape)
    if reasons:
        raise_no_eligible_route(reasons)


def _index_array(
    addresses: tuple[int, ...],
    *,
    device: Any | None = None,
) -> Array:
    x64_enabled = bool(jax.config.read("jax_enable_x64"))
    dtype = np.int64 if x64_enabled else np.int32
    if not x64_enabled and addresses and max(addresses) > np.iinfo(np.int32).max:
        raise ValueError("pure-JAX stride oracle needs x64 for addresses above int32")
    host = np.asarray(addresses, dtype=dtype)
    return jnp.asarray(host) if device is None else jax.device_put(host, device)


def _single_projection_device(data: Array) -> Any | None:
    if isinstance(data, Tracer):
        return None
    devices = tuple(data.devices())
    return devices[0] if len(devices) == 1 else None


def _reference_index_arrays(
    bound: StridedCopyPlan,
    side: str,
    data: Array,
) -> tuple[Array, ...]:
    """Build per-record indices on the operand device when one is known."""

    device = _single_projection_device(data)
    return tuple(
        _index_array(enumerate_addresses(record, side), device=device)
        for record in bound.records
    )


def _is_partial_dense_destination(bound: StridedCopyPlan) -> bool:
    if len(bound.records) != 1 or bound.copied_elements == bound.source_size:
        return False
    record = bound.records[0]
    if record.destination_offset != 0 or bound.copied_elements != bound.output_size:
        return False
    expected_stride = 1
    for extent, stride in zip(
        reversed(record.logical_shape),
        reversed(record.destination_strides),
        strict=True,
    ):
        if extent > 1 and stride != expected_stride:
            return False
        expected_stride *= max(extent, 1)
    return True


def execute_reference(source: object, bound: StridedCopyPlan) -> Array:
    """Execute a validated plan using bounded explicit address arrays."""

    data = jnp.asarray(source)
    if data.ndim < 1 or data.shape[-1] != bound.source_size:
        raise ValueError(
            f"expected source shape (*batch, {bound.source_size}), got {data.shape}"
        )
    expected_dtype = jnp.dtype(bound.source_dtype)
    if data.dtype != expected_dtype:
        raise TypeError(
            f"expected source dtype {expected_dtype.name}, got {data.dtype.name}"
        )
    _check_eligibility(bound, tuple(data.shape))

    result_dtype = jnp.dtype(bound.result_dtype)
    converted = jnp.asarray(data, dtype=result_dtype)
    if _is_partial_dense_destination(bound):
        record = bound.records[0]
        source_indices = _reference_index_arrays(bound, "source", data)[0]
        if source_indices.size == 0:
            return jnp.zeros((*data.shape[:-1], bound.output_size), dtype=result_dtype)
        mapped = converted[..., source_indices]
        scale_value = np.asarray(record.scale, dtype=result_dtype)
        if bool(scale_value == np.asarray(1, dtype=result_dtype)):
            return mapped
        return jnp.asarray(record.scale, dtype=result_dtype) * mapped

    result = jnp.zeros((*data.shape[:-1], bound.output_size), dtype=result_dtype)
    source_indices_by_record = _reference_index_arrays(bound, "source", data)
    destination_indices_by_record = _reference_index_arrays(
        bound,
        "destination",
        result,
    )
    for record, source_indices, destination_indices in zip(
        bound.records,
        source_indices_by_record,
        destination_indices_by_record,
        strict=True,
    ):
        if source_indices.size == 0:
            continue
        scale = jnp.asarray(record.scale, dtype=result.dtype)
        values = scale * converted[..., source_indices]
        result = result.at[..., destination_indices].set(
            values,
            unique_indices=True,
        )
    return result


def _validate_base_update_operands(
    base: Array,
    source: Array,
    plan: StridedCopyPlan,
) -> None:
    bound = plan
    if not base.shape or base.shape[-1] != bound.output_size:
        raise ValueError(
            f"expected base shape (*batch, {bound.output_size}), got {base.shape}"
        )
    if not source.shape or source.shape[-1] != bound.source_size:
        raise ValueError(
            f"expected source shape (*batch, {bound.source_size}), got {source.shape}"
        )
    if base.shape[:-1] != source.shape[:-1]:
        raise ValueError("base and source batch shapes must match")
    if base.dtype != jnp.dtype(bound.result_dtype):
        raise TypeError(f"expected base dtype {bound.result_dtype}, got {base.dtype.name}")
    if source.dtype != jnp.dtype(bound.source_dtype):
        raise TypeError(
            f"expected source dtype {bound.source_dtype}, got {source.dtype.name}"
        )


def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    expected = 1
    strides = [0] * len(shape)
    for axis in reversed(range(len(shape))):
        strides[axis] = expected
        expected *= shape[axis]
    return tuple(strides)


def _execute_base_update_reference(
    base: object,
    source: object,
    plan: StridedCopyPlan,
) -> Array:
    base_data = jnp.asarray(base)
    source_data = jnp.asarray(source)
    _validate_base_update_operands(base_data, source_data, plan)
    _check_eligibility(plan, tuple(source_data.shape))

    result = base_data
    if jnp.issubdtype(source_data.dtype, jnp.complexfloating) and not jnp.issubdtype(
        base_data.dtype,
        jnp.complexfloating,
    ):
        converted = jnp.real(source_data)
    elif plan.write_kind is StridedWriteKind.ASSIGN:
        converted = jnp.asarray(source_data, dtype=base_data.dtype)
    else:
        converted = source_data
    destination_indices_by_record = _reference_index_arrays(
        plan,
        "destination",
        base_data,
    )
    compact_source = False
    if len(plan.records) == 1:
        record = plan.records[0]
        compact_source = (
            record.source_offset == 0
            and record.source_strides == _contiguous_strides(record.logical_shape)
            and prod(record.logical_shape) == plan.source_size
        )
    source_indices_by_record = (
        ()
        if compact_source
        else _reference_index_arrays(plan, "source", source_data)
    )
    for record_index, (record, destination_indices) in enumerate(
        zip(plan.records, destination_indices_by_record, strict=True)
    ):
        source_indices = (
            None if compact_source else source_indices_by_record[record_index]
        )
        if source_indices is not None and source_indices.size == 0:
            continue
        if destination_indices.size == 0:
            continue
        scale = jnp.asarray(record.scale, dtype=base_data.dtype)
        mapped = converted if source_indices is None else converted[..., source_indices]
        scale_value = np.asarray(record.scale, dtype=base_data.dtype)
        values = (
            mapped
            if bool(scale_value == np.asarray(1, dtype=base_data.dtype))
            else scale * mapped
        )
        if plan.write_kind is StridedWriteKind.ASSIGN:
            result = result.at[..., destination_indices].set(
                values,
                unique_indices=True,
            )
        else:
            result = result.at[..., destination_indices].add(
                values,
                unique_indices=True,
            )
    return result


def execute_base_assign_reference(
    base: object,
    source: object,
    plan: StridedCopyPlan,
) -> Array:
    return _execute_base_update_reference(base, source, plan)


def execute_base_accumulate_reference(
    base: object,
    source: object,
    plan: StridedCopyPlan,
) -> Array:
    return _execute_base_update_reference(base, source, plan)


def _validate_selected_scale_operands(
    base: Array,
    factor: Array,
    plan: StridedCopyPlan,
) -> None:
    bound = plan
    if not base.shape or base.shape[-1] != bound.output_size:
        raise ValueError(
            f"expected base shape (*batch, {bound.output_size}), got {base.shape}"
        )
    if base.dtype != jnp.dtype(bound.result_dtype):
        raise TypeError(f"expected base dtype {bound.result_dtype}, got {base.dtype.name}")
    if factor.dtype != base.dtype:
        raise TypeError("selected scale factor must use the base dtype")
    if factor.shape not in ((), base.shape[:-1]):
        raise ValueError(
            "selected scale factor must be scalar or match the base batch shape"
        )


def execute_selected_scale_reference(
    base: object,
    factor: object,
    plan: StridedCopyPlan,
) -> Array:
    base_data = jnp.asarray(base)
    factor_data = jnp.asarray(factor, dtype=base_data.dtype)
    _validate_selected_scale_operands(base_data, factor_data, plan)
    _check_eligibility(plan, tuple(base_data.shape))

    result = base_data
    expanded_factor = factor_data if factor_data.ndim == 0 else factor_data[..., None]
    indices_by_record = _reference_index_arrays(
        plan,
        "destination",
        base_data,
    )
    for indices in indices_by_record:
        if indices.size == 0:
            continue
        values = expanded_factor * base_data[..., indices]
        result = result.at[..., indices].set(values, unique_indices=True)
    return result
