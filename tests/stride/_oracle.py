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
    AffinePlan,
    AffineRecord,
    StridedWriteKind,
    StridedScalarKind,
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
    record: AffineRecord,
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
    bound: AffinePlan,
    *,
    source_shape: tuple[int, ...] | None = None,
) -> tuple[str, ...]:
    """Return bounded-resource reasons that forbid materialized indices."""

    reasons: list[str] = []
    mapped_elements = bound.mapped_elements
    index_width = 8 if bool(jax.config.read("jax_enable_x64")) else 4
    logical_index_bytes = 2 * mapped_elements * index_width
    if 2 * mapped_elements * 36 > REFERENCE_PYTHON_ADDRESS_BYTE_CAP:
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
            batch_count * mapped_elements * jnp.dtype(bound.result_dtype).itemsize
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
    bound: AffinePlan,
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
    bound: AffinePlan,
    side: str,
    data: Array,
) -> tuple[Array, ...]:
    """Build per-record indices on the operand device when one is known."""

    device = _single_projection_device(data)
    return tuple(
        _index_array(enumerate_addresses(record, side), device=device)
        for record in bound.records
    )


def _is_partial_dense_destination(bound: AffinePlan) -> bool:
    if len(bound.records) != 1 or bound.mapped_elements == bound.source_size:
        return False
    record = bound.records[0]
    if record.destination_offset != 0 or bound.mapped_elements != bound.output_size:
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


def _mapped_reference(values: Array, record: AffineRecord, bound: AffinePlan) -> Array:
    def forward(source):
        mapped = source if record.scale is None else jnp.multiply(record.scale, source)
        dtype = (bound.source_dtype if bound.scalar_kind is StridedScalarKind.JAX_TRANSPOSE
                 else bound.result_dtype)
        return jnp.asarray(mapped, dtype=dtype)

    if bound.scalar_kind is StridedScalarKind.JAX_TRANSPOSE:
        return jax.linear_transpose(forward, jnp.zeros(values.shape, dtype=bound.result_dtype))(values)[0]
    return forward(values)


def execute_reference(source: object, bound: AffinePlan) -> Array:
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
    if _is_partial_dense_destination(bound):
        record = bound.records[0]
        source_indices = _reference_index_arrays(bound, "source", data)[0]
        if source_indices.size == 0:
            return jnp.zeros((*data.shape[:-1], bound.output_size), dtype=result_dtype)
        return _mapped_reference(data[..., source_indices], record, bound)

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
        values = _mapped_reference(data[..., source_indices], record, bound)
        result = result.at[..., destination_indices].set(
            values,
            unique_indices=True,
        )
    return result


def _validate_base_update_operands(
    base: Array,
    source: Array,
    plan: AffinePlan,
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


def _execute_base_update_reference(
    base: object,
    source: object,
    plan: AffinePlan,
) -> Array:
    base_data = jnp.asarray(base)
    source_data = jnp.asarray(source)
    _validate_base_update_operands(base_data, source_data, plan)
    _check_eligibility(plan, tuple(source_data.shape))
    return execute_update_reference(
        base_data, source_data, 1,
        0 if plan.write_kind is StridedWriteKind.ASSIGN else 1, plan,
    )


def execute_update_reference(base, source, source_factor, base_factor, plan):
    """Select actual expressions after each branch's final storage conversion."""
    result = jnp.asarray(base)
    source = jnp.asarray(source)
    source_factor, base_factor = jnp.asarray(source_factor), jnp.asarray(base_factor)
    source_indices = _reference_index_arrays(plan, "source", source)
    destination_indices = _reference_index_arrays(plan, "destination", result)
    for record, selected_source, selected_base in zip(
        plan.records, source_indices, destination_indices, strict=True,
    ):
        coefficient = (source_factor if record.scale is None else
                       jnp.multiply(source_factor, record.scale))
        coefficient = coefficient if coefficient.ndim == 0 else coefficient[..., None]
        beta = base_factor if base_factor.ndim == 0 else base_factor[..., None]
        values, previous = source[..., selected_source], result[..., selected_base]
        source_product = coefficient * values
        base_product = beta * previous
        source_only = jnp.where(coefficient == 1, values.astype(result.dtype),
                                source_product.astype(result.dtype))
        base_only = jnp.where(beta == 1, previous, base_product.astype(result.dtype))
        combined = jnp.where(
            coefficient == 1,
            jnp.where(beta == 1, (values + previous).astype(result.dtype),
                       (values + base_product).astype(result.dtype)),
            jnp.where(beta == 1, (source_product + previous).astype(result.dtype),
                       (source_product + base_product).astype(result.dtype)),
        )
        combined = jnp.where(
            coefficient == 0,
            jnp.where(beta == 0, jnp.zeros_like(previous), base_only),
            jnp.where(beta == 0, source_only, combined),
        )
        result = result.at[..., selected_base].set(combined, unique_indices=True)
    return result


def execute_base_assign_reference(
    base: object,
    source: object,
    plan: AffinePlan,
) -> Array:
    return _execute_base_update_reference(base, source, plan)


def execute_base_accumulate_reference(
    base: object,
    source: object,
    plan: AffinePlan,
) -> Array:
    return _execute_base_update_reference(base, source, plan)


def _validate_selected_scale_operands(
    base: Array,
    factor: Array,
    plan: AffinePlan,
) -> None:
    bound = plan
    if not base.shape or base.shape[-1] != bound.output_size:
        raise ValueError(
            f"expected base shape (*batch, {bound.output_size}), got {base.shape}"
        )
    if base.dtype != jnp.dtype(bound.result_dtype):
        raise TypeError(f"expected base dtype {bound.result_dtype}, got {base.dtype.name}")
    if factor.shape not in ((), base.shape[:-1]):
        raise ValueError(
            "selected scale factor must be scalar or match the base batch shape"
        )


@jax.custom_jvp
def _short_scale_reference(values: Array, factor: Array) -> Array:
    dtype = jnp.result_type(values, factor)
    values = jnp.asarray(values, dtype=dtype)
    factor = jnp.asarray(factor, dtype=dtype)
    return jnp.where(factor == 0, jnp.zeros_like(values),
                     jnp.where(factor == 1, values, factor * values))


@_short_scale_reference.defjvp
def _short_scale_reference_jvp(primals, tangents):
    values, factor = primals
    values_tangent, factor_tangent = tangents
    return (_short_scale_reference(values, factor),
            factor * values_tangent + factor_tangent * values)


def execute_selected_scale_reference(
    base: object,
    factor: object,
    plan: AffinePlan,
) -> Array:
    base_data = jnp.asarray(base)
    factor_data = jnp.asarray(factor)
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
        values = _short_scale_reference(base_data[..., indices], expanded_factor)
        result = result.at[..., indices].set(values.astype(base_data.dtype), unique_indices=True)
    return result
