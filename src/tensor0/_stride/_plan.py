"""Validated metadata and compact descriptors for private strided copies."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import struct
import sys
from typing import Never, TypeAlias

import jax.numpy as jnp
from jax.typing import DTypeLike
import numpy as np

ADDRESS_PLAN_KEY_VERSION = 4
MAXIMUM_RANK = 8
UINT64_MAX = (1 << 64) - 1
INT64_MIN = -(1 << 63)
INT64_MAX = (1 << 63) - 1

_COVERAGE_COMPLETE_UNIQUE = 1
_COVERAGE_PARTIAL_UNIQUE_ZERO_FILL = 2
_DTYPE_F16 = 2
_DTYPE_BF16 = 3
_DTYPE_F32 = 1
_DTYPE_C64 = 4
_DTYPE_S32 = 5
_DTYPE_F64 = 6
_DTYPE_C128 = 7
_DTYPE_PRED = 8
_DTYPE_S8 = 9
_DTYPE_S16 = 10
_DTYPE_S64 = 11
_DTYPE_U8 = 12
_DTYPE_U16 = 13
_DTYPE_U32 = 14
_DTYPE_U64 = 15

Scalar: TypeAlias = int | float | complex


def _stable_key_bytes(domain: str, *values: object) -> bytes:
    """Serialize internal key fields without process-randomized hashing."""

    def frame(payload: bytes) -> bytes:
        return struct.pack("<Q", len(payload)) + payload

    def encode(value: object) -> bytes:
        if value is None:
            return b"n"
        if isinstance(value, bool):
            return b"b\x01" if value else b"b\x00"
        if isinstance(value, int):
            negative = value < 0
            magnitude = -value if negative else value
            raw = magnitude.to_bytes(
                max(1, (magnitude.bit_length() + 7) // 8),
                byteorder="little",
            )
            return b"i" + (b"\x01" if negative else b"\x00") + frame(raw)
        if isinstance(value, str):
            return b"s" + frame(value.encode("utf-8"))
        if isinstance(value, bytes):
            return b"y" + frame(value)
        if isinstance(value, tuple):
            return (
                b"t"
                + struct.pack("<Q", len(value))
                + b"".join(encode(item) for item in value)
            )
        raise TypeError(f"unsupported stable-key field {type(value)!r}")

    return encode((domain, *values))


class CompleteMode(str, Enum):
    """Destination initialization and coverage contract."""

    COMPLETE_UNIQUE = "complete_unique"
    PARTIAL_UNIQUE_ZERO_FILL = "partial_unique_zero_fill"


class StridedOutputInit(str, Enum):
    """How an operation obtains destination values before affine writes."""

    UNINITIALIZED = "uninitialized"
    ZERO = "zero"
    PRESERVE_BASE = "preserve_base"


class StridedWriteKind(str, Enum):
    """How one unique mapped value updates its destination."""

    ASSIGN = "assign"
    ACCUMULATE = "accumulate"


class StridedScalarKind(str, Enum):
    """Finite scalar stages supported by retained operation signatures."""

    STATIC_SCALE_CAST = "static_scale_cast"
    DYNAMIC_SCALE = "dynamic_scale"
    JAX_TRANSPOSE = "jax_transpose"


class StridedReductionKind(str, Enum):
    """Structured reduction applied after scalar mapping."""

    NONE = "none"
    SUM = "sum"


class PlanValidationError(ValueError):
    """A deterministic address-plan validation failure."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True, slots=True)
class StridedCopyRecord:
    """One affine copy record in destination coordinate order."""

    logical_shape: tuple[int, ...]
    source_strides: tuple[int, ...]
    source_offset: int
    destination_strides: tuple[int, ...]
    destination_offset: int
    scale: Scalar = 1
    source_broadcast_axes: tuple[int, ...] = ()
    reduction_axes: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True, eq=False)
class StridedCopyPlan:
    """Per-record-validated affine metadata bound to array contracts."""

    records: tuple[StridedCopyRecord, ...]
    output_size: int
    coverage: CompleteMode
    required_source_size: int
    copied_elements: int
    source_size: int
    source_dtype: str
    result_dtype: str
    promoted_scale_dtype: str
    output_init: StridedOutputInit
    write_kind: StridedWriteKind
    scalar_kind: StridedScalarKind
    reduction_kind: StridedReductionKind
    semantic_key: bytes = field(repr=False)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, StridedCopyPlan):
            return NotImplemented
        return self.semantic_key == other.semantic_key

    def __hash__(self) -> int:
        return hash(self.semantic_key)

    @property
    def has_source_broadcast(self) -> bool:
        return any(record.source_broadcast_axes for record in self.records)


def _fail(code: str, message: str) -> Never:
    raise PlanValidationError(code, message)


def _u64(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        _fail("invalid_integer", f"{field} must be an integer")
    if value < 0:
        _fail("negative_value", f"{field} must be non-negative")
    if value > UINT64_MAX:
        _fail("integer_overflow", f"{field} exceeds uint64")
    return value


def _i64(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        _fail("invalid_integer", f"{field} must be an integer")
    if value < INT64_MIN or value > INT64_MAX:
        _fail("integer_overflow", f"{field} exceeds int64")
    return value


def _checked_add(left: int, right: int, field: str) -> int:
    if left > UINT64_MAX - right:
        _fail("address_overflow", f"addition overflow while computing {field}")
    return left + right


def _checked_mul(left: int, right: int, field: str) -> int:
    if left != 0 and right > UINT64_MAX // left:
        _fail("address_overflow", f"multiplication overflow while computing {field}")
    return left * right


def _checked_product(values: tuple[int, ...], field: str) -> int:
    result = 1
    for value in values:
        result = _checked_mul(result, value, field)
    return result


def _record_count(record: StridedCopyRecord, record_index: int) -> int:
    return _checked_product(
        record.logical_shape,
        f"record[{record_index}] logical element count",
    )


def _view(
    record: StridedCopyRecord,
    side: str,
) -> tuple[tuple[int, ...], tuple[int, ...], int]:
    if side == "source":
        return record.logical_shape, record.source_strides, record.source_offset
    if side == "destination":
        return (
            record.logical_shape,
            record.destination_strides,
            record.destination_offset,
        )
    raise AssertionError(f"unknown view side {side!r}")


def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Return row-major strides while preserving empty-axis metadata."""

    stride = 1
    result = [0] * len(shape)
    for axis in range(len(shape) - 1, -1, -1):
        result[axis] = stride
        stride *= max(shape[axis], 1)
    return tuple(result)


def _address_bounds(
    record: StridedCopyRecord,
    record_index: int,
    side: str,
) -> tuple[int, int] | None:
    shape, strides, offset = _view(record, side)
    if _record_count(record, record_index) == 0:
        return None
    low = offset
    high = offset
    for axis, (extent, stride) in enumerate(zip(shape, strides, strict=True)):
        span = extent - 1
        magnitude = abs(stride)
        if span and magnitude > UINT64_MAX // span:
            _fail(
                "address_overflow",
                f"record[{record_index}] {side} axis[{axis}] address overflows",
            )
        delta = span * stride
        low += min(0, delta)
        high += max(0, delta)
        if high > UINT64_MAX:
            _fail(
                "address_overflow",
                f"record[{record_index}] {side} address span overflows",
            )
    return low, high


def _validate_injective_view(
    record: StridedCopyRecord,
    record_index: int,
    side: str,
    *,
    axes: tuple[int, ...] | None = None,
) -> None:
    shape, strides, _ = _view(record, side)
    if 0 in shape:
        return

    covered_span = 0
    broadcast_axes = (
        frozenset(record.source_broadcast_axes) if side == "source" else frozenset()
    )
    selected_axes = range(len(shape)) if axes is None else axes
    for axis in sorted(
        selected_axes,
        key=lambda index: (abs(strides[index]), index),
    ):
        extent = shape[axis]
        stride = abs(strides[axis])
        if extent <= 1 or axis in broadcast_axes:
            continue
        if stride <= covered_span:
            _fail(
                "noninjective_view",
                f"record[{record_index}] {side} axis[{axis}] stride does not "
                "prove mixed-radix injectivity",
            )
        axis_span = _checked_mul(
            extent - 1,
            stride,
            f"record[{record_index}] {side} injectivity span",
        )
        covered_span = _checked_add(
            covered_span,
            axis_span,
            f"record[{record_index}] {side} injectivity span",
        )


def _validate_address_spec(
    records: tuple[StridedCopyRecord, ...],
    output_size: int,
    coverage: CompleteMode,
    write_kind: StridedWriteKind,
    reduction_kind: StridedReductionKind,
) -> tuple[int, int]:
    """Validate affine metadata and return its derived storage metadata."""

    output_size = _u64(output_size, "output_size")
    if not isinstance(coverage, CompleteMode):
        _fail("coverage_mode", "coverage must be a CompleteMode")

    required_source_size = 0
    copied_elements = 0
    mapped_output_elements = 0
    has_reduction_axis = False
    for record_index, record in enumerate(records):
        rank = len(record.logical_shape)
        if (
            len(record.source_strides) != rank
            or len(record.destination_strides) != rank
        ):
            _fail(
                "rank_mismatch",
                f"record[{record_index}] shape and stride ranks differ",
            )
        for axis, extent in enumerate(record.logical_shape):
            _u64(extent, f"record[{record_index}].logical_shape[{axis}]")
        broadcast_axes = tuple(
            _u64(axis, f"record[{record_index}].source_broadcast_axes[{position}]")
            for position, axis in enumerate(record.source_broadcast_axes)
        )
        expected_broadcast_axes = tuple(
            axis
            for axis, (extent, stride) in enumerate(
                zip(
                    record.logical_shape,
                    record.source_strides,
                    strict=True,
                )
            )
            if extent > 1 and stride == 0
        )
        if broadcast_axes != expected_broadcast_axes:
            _fail(
                "source_broadcast_axes",
                f"record[{record_index}] broadcast metadata must exactly match "
                "its nontrivial zero source strides",
            )
        reduction_axes = tuple(
            _u64(axis, f"record[{record_index}].reduction_axes[{position}]")
            for position, axis in enumerate(record.reduction_axes)
        )
        if reduction_axes != tuple(sorted(set(reduction_axes))):
            _fail(
                "reduction_axes",
                f"record[{record_index}] reduction axes must be sorted and unique",
            )
        if any(axis >= rank for axis in reduction_axes):
            _fail(
                "reduction_axes",
                f"record[{record_index}] reduction axis exceeds record rank",
            )
        if reduction_kind is StridedReductionKind.NONE:
            if reduction_axes:
                _fail(
                    "reduction_axes",
                    f"record[{record_index}] map record cannot carry reduction axes",
                )
        else:
            if any(axis in reduction_axes for axis in broadcast_axes):
                _fail(
                    "reduction_source_broadcast",
                    "structured reduction axes cannot also broadcast the input",
                )
            has_reduction_axis = has_reduction_axis or bool(reduction_axes)
        for side, strides, offset in (
            ("source", record.source_strides, record.source_offset),
            ("destination", record.destination_strides, record.destination_offset),
        ):
            _u64(offset, f"record[{record_index}].{side}_offset")
            for axis, stride in enumerate(strides):
                _i64(stride, f"record[{record_index}].{side}_strides[{axis}]")
                if record.logical_shape[axis] <= 1 or side != "destination":
                    continue
                if reduction_kind is StridedReductionKind.NONE and stride == 0:
                    _fail(
                        "zero_stride",
                        f"record[{record_index}] destination axis[{axis}] aliases",
                    )
                if reduction_kind is StridedReductionKind.SUM:
                    if axis in reduction_axes and stride != 0:
                        _fail(
                            "reduction_output_dependence",
                            f"record[{record_index}] destination depends on "
                            f"reduction axis[{axis}]",
                        )
                    if axis not in reduction_axes and stride == 0:
                        _fail(
                            "zero_stride",
                            f"record[{record_index}] map axis[{axis}] aliases",
                        )
            injective_axes = (
                tuple(axis for axis in range(rank) if axis not in reduction_axes)
                if side == "destination"
                and reduction_kind is StridedReductionKind.SUM
                else None
            )
            _validate_injective_view(
                record,
                record_index,
                side,
                axes=injective_axes,
            )

        logical_count = _record_count(record, record_index)
        copied_elements = _checked_add(
            copied_elements,
            logical_count,
            "plan copied element count",
        )
        record_output_elements = (
            _checked_product(
                tuple(
                    extent
                    for axis, extent in enumerate(record.logical_shape)
                    if axis not in reduction_axes
                ),
                f"record[{record_index}] mapped output count",
            )
            if reduction_kind is StridedReductionKind.SUM
            else logical_count
        )
        mapped_output_elements = _checked_add(
            mapped_output_elements,
            record_output_elements,
            "plan mapped output count",
        )
        source_bounds = _address_bounds(record, record_index, "source")
        destination_bounds = _address_bounds(record, record_index, "destination")
        if source_bounds is not None:
            source_minimum, source_maximum = source_bounds
            if source_minimum < 0:
                _fail(
                    "source_bounds",
                    f"record[{record_index}] source precedes storage",
                )
            required_source_size = max(required_source_size, source_maximum + 1)
        if destination_bounds is not None:
            destination_minimum, destination_maximum = destination_bounds
            if destination_minimum < 0 or destination_maximum >= output_size:
                _fail(
                    "destination_bounds",
                    f"record[{record_index}] destination exceeds output_size",
                )
        if logical_count == 0:
            required_source_size = max(required_source_size, record.source_offset)
            if record.destination_offset > output_size:
                _fail(
                    "destination_bounds",
                    f"record[{record_index}] empty destination offset exceeds output_size",
                )

    grouped_sum = (
        reduction_kind is StridedReductionKind.SUM
        and len(records) > 1
        and write_kind is StridedWriteKind.ACCUMULATE
    )
    if reduction_kind is StridedReductionKind.SUM and not (
        has_reduction_axis or grouped_sum
    ):
        _fail(
            "no_reduction_axis",
            "structured sum requires a reduction axis or grouped records",
        )
    if reduction_kind is StridedReductionKind.SUM and len(records) > 1:
        if write_kind is not StridedWriteKind.ACCUMULATE:
            _fail(
                "reduction_write_kind",
                "multi-record structured reduction requires accumulate writes",
            )
        return required_source_size, copied_elements
    if coverage is CompleteMode.COMPLETE_UNIQUE and mapped_output_elements != output_size:
        _fail(
            "incomplete_destination",
            "CompleteUnique copied element count does not match output_size",
        )
    if (
        coverage is CompleteMode.PARTIAL_UNIQUE_ZERO_FILL
        and mapped_output_elements > output_size
    ):
        _fail(
            "destination_coverage",
            "PartialUnique copied element count exceeds output_size",
        )
    return required_source_size, copied_elements


def _validate_host_storage(size: int, dtype: jnp.dtype, field: str) -> None:
    itemsize = dtype.itemsize
    if size > sys.maxsize // itemsize:
        _fail(
            "host_address_overflow",
            f"{field} byte range exceeds the host pointer-difference domain",
        )


def _exact_scalar_bytes(value: object, dtype: jnp.dtype) -> bytes:
    """Return one scalar's exact result-dtype representation in little endian."""

    scalar = np.asarray(value, dtype=dtype)
    if scalar.shape != ():
        _fail("scale_shape", "record scale must be scalar")
    little_endian_dtype = dtype.newbyteorder("<")
    return scalar.astype(little_endian_dtype, copy=False).tobytes()


def build_strided_copy_plan(
    *,
    records: tuple[StridedCopyRecord, ...],
    output_size: int,
    coverage: CompleteMode,
    source_size: int,
    source_dtype: DTypeLike,
    result_dtype: DTypeLike,
    output_init: StridedOutputInit | None = None,
    write_kind: StridedWriteKind = StridedWriteKind.ASSIGN,
    scalar_kind: StridedScalarKind = StridedScalarKind.STATIC_SCALE_CAST,
    reduction_kind: StridedReductionKind = StridedReductionKind.NONE,
) -> StridedCopyPlan:
    """Validate each record and bind metadata to flat array contracts."""

    if output_init is None:
        if reduction_kind is StridedReductionKind.SUM:
            output_init = StridedOutputInit.ZERO
        else:
            output_init = (
                StridedOutputInit.UNINITIALIZED
                if coverage is CompleteMode.COMPLETE_UNIQUE
                else StridedOutputInit.ZERO
            )
    if not isinstance(output_init, StridedOutputInit):
        _fail("output_init", "output_init must be a StridedOutputInit")
    if not isinstance(write_kind, StridedWriteKind):
        _fail("write_kind", "write_kind must be a StridedWriteKind")
    if not isinstance(scalar_kind, StridedScalarKind):
        _fail("scalar_kind", "scalar_kind must be a StridedScalarKind")
    if not isinstance(reduction_kind, StridedReductionKind):
        _fail("reduction_kind", "reduction_kind must be a StridedReductionKind")
    if (
        output_init is StridedOutputInit.UNINITIALIZED
        and coverage is not CompleteMode.COMPLETE_UNIQUE
    ):
        _fail("output_init", "uninitialized output requires complete coverage")
    if reduction_kind is StridedReductionKind.SUM:
        if output_init is not StridedOutputInit.ZERO:
            _fail(
                "reduction_output_init",
                "structured sum currently requires zero initialization",
            )
        if scalar_kind is StridedScalarKind.DYNAMIC_SCALE:
            _fail(
                "reduction_scalar_kind",
                "structured sum does not support dynamic scale",
            )

    required_source_size, copied_elements = _validate_address_spec(
        records,
        output_size,
        coverage,
        write_kind,
        reduction_kind,
    )

    actual_source_size = _u64(source_size, "source_size")
    if actual_source_size < required_source_size:
        _fail(
            "source_bounds",
            "actual source is smaller than required_source_size",
        )
    source_dtype_value = jnp.dtype(source_dtype)
    result_dtype_value = jnp.dtype(result_dtype)
    _validate_host_storage(actual_source_size, source_dtype_value, "source")
    _validate_host_storage(
        output_size,
        result_dtype_value,
        "destination",
    )

    scale_dtype = (
        source_dtype_value
        if scalar_kind is StridedScalarKind.JAX_TRANSPOSE
        else result_dtype_value
    )
    scale_dtypes = tuple(jnp.asarray(record.scale).dtype for record in records)
    promoted = jnp.result_type(source_dtype_value, *scale_dtypes)
    for record_index, record in enumerate(records):
        try:
            scalar = jnp.asarray(record.scale, dtype=scale_dtype)
        except (TypeError, ValueError, OverflowError) as exc:
            _fail(
                "scale_dtype",
                f"record[{record_index}] scale cannot be represented: {exc}",
            )
        if scalar.shape != ():
            _fail("scale_shape", f"record[{record_index}] scale must be scalar")
    promoted_name = jnp.dtype(promoted).name
    return StridedCopyPlan(
        records=records,
        output_size=output_size,
        coverage=coverage,
        required_source_size=required_source_size,
        copied_elements=copied_elements,
        source_size=actual_source_size,
        source_dtype=source_dtype_value.name,
        result_dtype=result_dtype_value.name,
        promoted_scale_dtype=promoted_name,
        output_init=output_init,
        write_kind=write_kind,
        scalar_kind=scalar_kind,
        reduction_kind=reduction_kind,
        semantic_key=_bound_static_key(
            records=records,
            output_size=output_size,
            coverage=coverage,
            required_source_size=required_source_size,
            copied_elements=copied_elements,
            source_size=actual_source_size,
            source_dtype=source_dtype_value,
            result_dtype=result_dtype_value,
            scale_dtype=scale_dtype,
            promoted_scale_dtype=promoted_name,
            output_init=output_init,
            write_kind=write_kind,
            scalar_kind=scalar_kind,
            reduction_kind=reduction_kind,
        ),
    )


def transpose_plan(bound: StridedCopyPlan) -> StridedCopyPlan:
    """Build the JAX linear transpose of one static affine map or sum."""

    source_dtype = jnp.dtype(bound.source_dtype)
    result_dtype = jnp.dtype(bound.result_dtype)
    if not (
        jnp.issubdtype(source_dtype, jnp.inexact)
        and jnp.issubdtype(result_dtype, jnp.inexact)
    ):
        _fail(
            "transpose_dtype_fallback",
            "integer and boolean plans do not have a JAX linear transpose",
        )
    if bound.scalar_kind is StridedScalarKind.STATIC_SCALE_CAST:
        scalar_kind = StridedScalarKind.JAX_TRANSPOSE
    elif bound.scalar_kind is StridedScalarKind.JAX_TRANSPOSE:
        scalar_kind = StridedScalarKind.STATIC_SCALE_CAST
    else:
        _fail(
            "transpose_scalar_kind",
            "dynamic-scale plans require a dedicated multi-operand transpose",
        )

    if bound.reduction_kind is StridedReductionKind.SUM:
        records = tuple(
            StridedCopyRecord(
                logical_shape=record.logical_shape,
                source_strides=record.destination_strides,
                source_offset=record.destination_offset,
                destination_strides=record.source_strides,
                destination_offset=record.source_offset,
                scale=record.scale,
                source_broadcast_axes=tuple(
                    axis
                    for axis in record.reduction_axes
                    if record.logical_shape[axis] > 1
                ),
                reduction_axes=record.source_broadcast_axes,
            )
            for record in bound.records
        )
        reduction_kind = (
            StridedReductionKind.SUM
            if len(records) > 1 or any(record.reduction_axes for record in records)
            else StridedReductionKind.NONE
        )
        destination_complete = bound.copied_elements == bound.source_size
        return build_strided_copy_plan(
            records=records,
            output_size=bound.source_size,
            coverage=(
                CompleteMode.COMPLETE_UNIQUE
                if reduction_kind is StridedReductionKind.NONE
                and destination_complete
                else CompleteMode.PARTIAL_UNIQUE_ZERO_FILL
            ),
            source_size=bound.output_size,
            source_dtype=bound.result_dtype,
            result_dtype=bound.source_dtype,
            write_kind=(
                StridedWriteKind.ACCUMULATE
                if reduction_kind is StridedReductionKind.SUM
                and len(records) > 1
                else StridedWriteKind.ASSIGN
            ),
            scalar_kind=scalar_kind,
            reduction_kind=reduction_kind,
        )

    if bound.has_source_broadcast:
        records = tuple(
            StridedCopyRecord(
                logical_shape=record.logical_shape,
                source_strides=record.destination_strides,
                source_offset=record.destination_offset,
                destination_strides=record.source_strides,
                destination_offset=record.source_offset,
                scale=record.scale,
                reduction_axes=record.source_broadcast_axes,
            )
            for record in bound.records
        )
        return build_strided_copy_plan(
            records=records,
            output_size=bound.source_size,
            coverage=CompleteMode.PARTIAL_UNIQUE_ZERO_FILL,
            source_size=bound.output_size,
            source_dtype=bound.result_dtype,
            result_dtype=bound.source_dtype,
            write_kind=(
                StridedWriteKind.ACCUMULATE
                if len(records) > 1
                else StridedWriteKind.ASSIGN
            ),
            scalar_kind=scalar_kind,
            reduction_kind=StridedReductionKind.SUM,
        )

    records = tuple(
        StridedCopyRecord(
            logical_shape=record.logical_shape,
            source_strides=record.destination_strides,
            source_offset=record.destination_offset,
            destination_strides=record.source_strides,
            destination_offset=record.source_offset,
            scale=record.scale,
        )
        for record in bound.records
    )
    destination_complete = bound.copied_elements == bound.source_size
    return build_strided_copy_plan(
        records=records,
        output_size=bound.source_size,
        coverage=(
            CompleteMode.COMPLETE_UNIQUE
            if destination_complete
            else CompleteMode.PARTIAL_UNIQUE_ZERO_FILL
        ),
        source_size=bound.output_size,
        source_dtype=bound.result_dtype,
        result_dtype=bound.source_dtype,
        scalar_kind=scalar_kind,
    )


def transpose_same_dtype_plan(
    bound: StridedCopyPlan,
) -> StridedCopyPlan:
    """Build one same-dtype transpose for retained internal callers."""

    if bound.source_dtype != bound.result_dtype:
        _fail(
            "transpose_dtype_fallback",
            "mixed-dtype transpose requires typed native scalar dispatch",
        )
    return transpose_plan(bound)


def _bound_static_key(
    *,
    records: tuple[StridedCopyRecord, ...],
    output_size: int,
    coverage: CompleteMode,
    required_source_size: int,
    copied_elements: int,
    source_size: int,
    source_dtype: jnp.dtype,
    result_dtype: jnp.dtype,
    scale_dtype: jnp.dtype,
    promoted_scale_dtype: str,
    output_init: StridedOutputInit,
    write_kind: StridedWriteKind,
    scalar_kind: StridedScalarKind,
    reduction_kind: StridedReductionKind,
) -> bytes:
    """Build the raw bound-plan identity used by Python/JAX static arguments."""

    record_keys = tuple(
        (
            record.logical_shape,
            record.source_strides,
            record.source_offset,
            record.destination_strides,
            record.destination_offset,
            _exact_scalar_bytes(record.scale, scale_dtype),
            record.source_broadcast_axes,
            record.reduction_axes,
        )
        for record in records
    )
    return _stable_key_bytes(
        "tensor0-stride-bound-static-v5",
        ADDRESS_PLAN_KEY_VERSION,
        source_size,
        required_source_size,
        copied_elements,
        output_size,
        coverage.value,
        source_dtype.name,
        result_dtype.name,
        promoted_scale_dtype,
        output_init.value,
        write_kind.value,
        scalar_kind.value,
        reduction_kind.value,
        record_keys,
    )
