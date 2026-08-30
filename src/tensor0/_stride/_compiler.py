"""Static canonical and CPU execution planning for private stride plans."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from math import prod
import struct
from typing import TypeAlias

import jax.numpy as jnp

from ._plan import (
    MAXIMUM_RANK,
    UINT64_MAX,
    _DTYPE_BF16,
    _DTYPE_C128,
    _DTYPE_C64,
    _DTYPE_F16,
    _DTYPE_F32,
    _DTYPE_F64,
    _DTYPE_PRED,
    _DTYPE_S16,
    _DTYPE_S32,
    _DTYPE_S64,
    _DTYPE_S8,
    _DTYPE_U16,
    _DTYPE_U32,
    _DTYPE_U64,
    _DTYPE_U8,
    StridedCopyPlan,
    CompleteMode,
    StridedCopyRecord,
    StridedReductionKind,
    StridedScalarKind,
    _exact_scalar_bytes,
    _stable_key_bytes,
)


COMPILER_POLICY_VERSION = 3
CPU_POLICY_VERSION = 7
COMPILED_DESCRIPTOR_MAGIC = 0x3150434952543054
COMPILED_DESCRIPTOR_VERSION = 7

_RAW_DESCRIPTOR_WITNESS = 1
_PARALLEL_CHUNK_BYTES = 256 * 1024
_PARALLEL_GENERAL_MINIMUM_BYTES = 512 * 1024
_PARALLEL_COMPACT_MINIMUM_BYTES = 4 * 1024 * 1024
_PARALLEL_RANK4_MINIMUM_BYTES = 512 * 1024
_NATIVE_CPU_DTYPES = frozenset(
    {
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float16",
        "bfloat16",
        "float32",
        "float64",
        "complex64",
        "complex128",
    }
)

_DTYPE_CODES = {
    jnp.dtype(jnp.bool_): _DTYPE_PRED,
    jnp.dtype(jnp.int8): _DTYPE_S8,
    jnp.dtype(jnp.int16): _DTYPE_S16,
    jnp.dtype(jnp.int32): _DTYPE_S32,
    jnp.dtype(jnp.int64): _DTYPE_S64,
    jnp.dtype(jnp.uint8): _DTYPE_U8,
    jnp.dtype(jnp.uint16): _DTYPE_U16,
    jnp.dtype(jnp.uint32): _DTYPE_U32,
    jnp.dtype(jnp.uint64): _DTYPE_U64,
    jnp.dtype(jnp.float16): _DTYPE_F16,
    jnp.dtype(jnp.bfloat16): _DTYPE_BF16,
    jnp.dtype(jnp.float32): _DTYPE_F32,
    jnp.dtype(jnp.float64): _DTYPE_F64,
    jnp.dtype(jnp.complex64): _DTYPE_C64,
    jnp.dtype(jnp.complex128): _DTYPE_C128,
}

PlanKey: TypeAlias = bytes


def _dtype_code(dtype: jnp.dtype) -> int:
    code = _DTYPE_CODES.get(dtype)
    if code is None:
        raise ValueError(f"native stride descriptor does not support {dtype.name}")
    return code


class LayoutKind(str, Enum):
    """Backend-independent normalized affine structure."""

    EMPTY = "empty"
    COMPACT = "compact"
    PERMUTATION_LIKE = "permutation_like"
    POSITIVE_AFFINE = "positive_affine"
    SIGNED_AFFINE = "signed_affine"
    BROADCAST_READ = "broadcast_read"
    STRUCTURED_REDUCTION = "structured_reduction"
    GENERIC_AFFINE = "generic_affine"


class ScaleKind(str, Enum):
    """Exact promoted-result-dtype scale class."""

    UNIT = "unit"
    NEGATIVE_UNIT = "negative_unit"
    GENERAL = "general"


class CpuKernelKind(str, Enum):
    """The finite native CPU work-kind projection."""

    EMPTY = "empty"
    COMPACT = "compact"
    GENERIC = "generic"
    RANK2_FORWARD = "rank2_forward"
    RANK2_REVERSE = "rank2_reverse"
    RANK4_AXIS0 = "rank4_axis0"
    RANK4_TWO_PAIR = "rank4_two_pair"
    RANK2_SIGNED_PERMUTATION = "rank2_signed_permutation"
    STRUCTURED_REDUCTION = "structured_reduction"


_LAYOUT_CODES = {
    LayoutKind.EMPTY: 0,
    LayoutKind.COMPACT: 1,
    LayoutKind.PERMUTATION_LIKE: 2,
    LayoutKind.POSITIVE_AFFINE: 3,
    LayoutKind.SIGNED_AFFINE: 4,
    LayoutKind.GENERIC_AFFINE: 5,
    LayoutKind.BROADCAST_READ: 6,
    LayoutKind.STRUCTURED_REDUCTION: 7,
}
_SCALE_CODES = {
    ScaleKind.UNIT: 1,
    ScaleKind.NEGATIVE_UNIT: 2,
    ScaleKind.GENERAL: 3,
}
_CPU_KERNEL_CODES = {
    CpuKernelKind.EMPTY: 0,
    CpuKernelKind.COMPACT: 1,
    CpuKernelKind.GENERIC: 2,
    CpuKernelKind.RANK2_FORWARD: 3,
    CpuKernelKind.RANK2_REVERSE: 4,
    CpuKernelKind.RANK4_AXIS0: 5,
    CpuKernelKind.RANK4_TWO_PAIR: 6,
    CpuKernelKind.RANK2_SIGNED_PERMUTATION: 7,
    CpuKernelKind.STRUCTURED_REDUCTION: 8,
}


@dataclass(frozen=True, slots=True)
class CompiledRecord:
    """One canonical normalized record plus factor-level provenance."""

    logical_shape: tuple[int, ...]
    source_strides: tuple[int, ...]
    source_offset: int
    destination_strides: tuple[int, ...]
    destination_offset: int
    scale_bytes: bytes
    source_broadcast_axes: tuple[int, ...]
    reduction_axes: tuple[int, ...]
    logical_elements: int
    layout_kind: LayoutKind
    scale_kind: ScaleKind
    axis_provenance_fastest_first: tuple[tuple[int, ...], ...]
    map_loop_order: tuple[int, ...]
    reduction_loop_order: tuple[int, ...]
    output_count: int
    reduction_count: int
    accumulator_dtype: str

    @property
    def rank(self) -> int:
        return len(self.logical_shape)

    @property
    def execution_key(self) -> PlanKey:
        return _stable_key_bytes(
            "tensor0-stride-compiled-record-v1",
            self.logical_shape,
            self.source_strides,
            self.source_offset,
            self.destination_strides,
            self.destination_offset,
            self.scale_bytes,
            self.source_broadcast_axes,
            self.reduction_axes,
            self.map_loop_order,
            self.reduction_loop_order,
            self.output_count,
            self.reduction_count,
            self.accumulator_dtype,
            self.layout_kind.value,
            self.scale_kind.value,
        )


@dataclass(frozen=True, slots=True)
class CompiledStridePlan:
    """Backend-independent canonical plan retained by its explicit owner."""

    bound: StridedCopyPlan = field(compare=False, hash=False, repr=False)
    records: tuple[CompiledRecord, ...]
    canonical_execution_key: PlanKey
    capability_key: PlanKey
    policy_version: int = COMPILER_POLICY_VERSION

@dataclass(frozen=True, slots=True)
class CpuExecutionRecord:
    """One raw-descriptor-equivalent CPU work projection."""

    kernel_kind: CpuKernelKind
    loop_axes_fastest_first: tuple[int, ...]
    elements_per_unit: int
    unit_count: int
    units_per_chunk: int
    chunk_count: int


def _try_mul_u64(left: int, right: int) -> int | None:
    if left != 0 and right > UINT64_MAX // left:
        return None
    return left * right


def _try_mul_i64(left: int, right: int) -> int | None:
    result = left * right
    if result < -(1 << 63) or result > (1 << 63) - 1:
        return None
    return result


def _is_dense_view(
    shape: tuple[int, ...],
    strides: tuple[int, ...],
) -> bool:
    axes = sorted(range(len(shape)), key=lambda axis: abs(strides[axis]))
    expected = 1
    for axis in axes:
        if shape[axis] <= 1:
            continue
        if abs(strides[axis]) != expected:
            return False
        multiplied = _try_mul_u64(expected, shape[axis])
        if multiplied is None:
            return False
        expected = multiplied
    return True


def _is_compact_same_mapping(record: CompiledRecord) -> bool:
    return (
        record.source_strides == record.destination_strides
        and all(stride > 0 for stride in record.source_strides)
        and _is_dense_view(record.logical_shape, record.source_strides)
    )


def _scale_kind(
    dtype: jnp.dtype,
    scale_bytes: bytes,
) -> ScaleKind:
    try:
        unit_bytes = _exact_scalar_bytes(1, dtype)
    except (TypeError, ValueError, OverflowError):
        return ScaleKind.GENERAL
    if scale_bytes == unit_bytes:
        return ScaleKind.UNIT
    try:
        negative_unit_bytes = _exact_scalar_bytes(-1, dtype)
    except (TypeError, ValueError, OverflowError):
        negative_unit_bytes = None
    if scale_bytes == negative_unit_bytes:
        return ScaleKind.NEGATIVE_UNIT
    return ScaleKind.GENERAL


def _layout_kind(
    *,
    logical_elements: int,
    shape: tuple[int, ...],
    source_strides: tuple[int, ...],
    destination_strides: tuple[int, ...],
    source_broadcast_axes: tuple[int, ...],
    reduction_axes: tuple[int, ...],
) -> LayoutKind:
    if logical_elements == 0:
        return LayoutKind.EMPTY
    if source_broadcast_axes:
        return LayoutKind.BROADCAST_READ
    if reduction_axes:
        return LayoutKind.STRUCTURED_REDUCTION
    if (
        source_strides == destination_strides
        and all(stride > 0 for stride in source_strides)
        and _is_dense_view(shape, source_strides)
    ):
        return LayoutKind.COMPACT
    if _is_dense_view(shape, source_strides) and _is_dense_view(
        shape,
        destination_strides,
    ):
        return LayoutKind.PERMUTATION_LIKE
    if all(
        source_stride > 0 and destination_stride > 0
        for extent, source_stride, destination_stride in zip(
            shape,
            source_strides,
            destination_strides,
            strict=True,
        )
        if extent > 1
    ):
        return LayoutKind.POSITIVE_AFFINE
    if all(
        source_stride != 0 and destination_stride != 0
        for extent, source_stride, destination_stride in zip(
            shape,
            source_strides,
            destination_strides,
            strict=True,
        )
        if extent > 1
    ):
        return LayoutKind.SIGNED_AFFINE
    # This is deliberately unreachable under the current semantic contract
    # after singleton removal. It reserves a deterministic future catch-all
    # without making a CPU specialization part of backend classification.
    return LayoutKind.GENERIC_AFFINE


def _normalize_record(
    record: StridedCopyRecord,
    dtype: jnp.dtype,
) -> CompiledRecord:
    shape = list(record.logical_shape)
    source_strides = list(record.source_strides)
    destination_strides = list(record.destination_strides)
    source_broadcast = [
        axis in record.source_broadcast_axes for axis in range(len(shape))
    ]
    reduction = [axis in record.reduction_axes for axis in range(len(shape))]
    provenance = [[axis] for axis in range(len(shape))]
    logical_elements = prod(shape)

    if logical_elements != 0 and shape:
        retained = [
            axis
            for axis, extent in enumerate(shape)
            if extent != 1 or reduction[axis]
        ]
        shape = [shape[axis] for axis in retained]
        source_strides = [source_strides[axis] for axis in retained]
        destination_strides = [destination_strides[axis] for axis in retained]
        source_broadcast = [source_broadcast[axis] for axis in retained]
        reduction = [reduction[axis] for axis in retained]
        provenance = [provenance[axis] for axis in retained]

        axis = 0
        while axis + 1 < len(shape):
            left_source_span = _try_mul_i64(
                source_strides[axis],
                shape[axis],
            )
            left_destination_span = _try_mul_i64(
                destination_strides[axis],
                shape[axis],
            )
            right_source_span = _try_mul_i64(
                source_strides[axis + 1],
                shape[axis + 1],
            )
            right_destination_span = _try_mul_i64(
                destination_strides[axis + 1],
                shape[axis + 1],
            )
            left_fastest = (
                left_source_span == source_strides[axis + 1]
                and left_destination_span == destination_strides[axis + 1]
            )
            right_fastest = (
                right_source_span == source_strides[axis]
                and right_destination_span == destination_strides[axis]
            )
            if not left_fastest and not right_fastest:
                axis += 1
                continue
            if reduction[axis] != reduction[axis + 1]:
                axis += 1
                continue
            merged_shape = _try_mul_u64(shape[axis], shape[axis + 1])
            if merged_shape is None:
                axis += 1
                continue

            shape[axis] = merged_shape
            if right_fastest:
                source_strides[axis] = source_strides[axis + 1]
                destination_strides[axis] = destination_strides[axis + 1]
                provenance[axis] = provenance[axis + 1] + provenance[axis]
            else:
                provenance[axis] = provenance[axis] + provenance[axis + 1]
            source_broadcast[axis] = (
                source_broadcast[axis] or source_broadcast[axis + 1]
            )
            del shape[axis + 1]
            del source_strides[axis + 1]
            del destination_strides[axis + 1]
            del source_broadcast[axis + 1]
            del reduction[axis + 1]
            del provenance[axis + 1]
            if axis:
                axis -= 1

    frozen_shape = tuple(shape)
    frozen_source_strides = tuple(source_strides)
    frozen_destination_strides = tuple(destination_strides)
    frozen_source_broadcast_axes = tuple(
        axis for axis, broadcast in enumerate(source_broadcast) if broadcast
    )
    frozen_reduction_axes = tuple(
        axis for axis, is_reduction in enumerate(reduction) if is_reduction
    )
    scale_bytes = _exact_scalar_bytes(record.scale, dtype)
    return CompiledRecord(
        logical_shape=frozen_shape,
        source_strides=frozen_source_strides,
        source_offset=record.source_offset,
        destination_strides=frozen_destination_strides,
        destination_offset=record.destination_offset,
        scale_bytes=scale_bytes,
        source_broadcast_axes=frozen_source_broadcast_axes,
        reduction_axes=frozen_reduction_axes,
        logical_elements=logical_elements,
        layout_kind=_layout_kind(
            logical_elements=logical_elements,
            shape=frozen_shape,
            source_strides=frozen_source_strides,
            destination_strides=frozen_destination_strides,
            source_broadcast_axes=frozen_source_broadcast_axes,
            reduction_axes=frozen_reduction_axes,
        ),
        scale_kind=_scale_kind(
            dtype,
            scale_bytes,
        ),
        axis_provenance_fastest_first=tuple(tuple(group) for group in provenance),
        map_loop_order=tuple(
            sorted(
                range(len(frozen_shape)),
                key=lambda axis: (
                    abs(frozen_destination_strides[axis]),
                    abs(frozen_source_strides[axis]),
                ),
            )
        ),
        reduction_loop_order=(),
        output_count=logical_elements,
        reduction_count=1,
        accumulator_dtype=dtype.name,
    )


def _compile_reduction_record(
    bound: StridedCopyPlan,
    record: CompiledRecord,
) -> CompiledRecord:
    reduction_axis_set = frozenset(record.reduction_axes)
    map_axes = tuple(
        axis
        for axis in range(record.rank)
        if axis not in reduction_axis_set
    )
    return replace(
        record,
        map_loop_order=tuple(
            sorted(
                map_axes,
                key=lambda axis: (
                    abs(record.destination_strides[axis]),
                    abs(record.source_strides[axis]),
                ),
            )
        ),
        reduction_loop_order=tuple(
            sorted(
                record.reduction_axes,
                key=lambda axis: abs(record.source_strides[axis]),
            )
        ),
        output_count=prod(record.logical_shape[axis] for axis in map_axes),
        reduction_count=prod(
            record.logical_shape[axis] for axis in record.reduction_axes
        ),
        accumulator_dtype=bound.result_dtype,
    )


def compile_plan(bound: StridedCopyPlan) -> CompiledStridePlan:
    """Compile canonical metadata without changing execution routing."""

    scale_dtype = jnp.dtype(
        bound.source_dtype
        if bound.scalar_kind is StridedScalarKind.JAX_TRANSPOSE
        else bound.result_dtype
    )
    normalized_records = tuple(
        _normalize_record(record, scale_dtype) for record in bound.records
    )
    records = (
        tuple(
            _compile_reduction_record(bound, record)
            for record in normalized_records
        )
        if bound.reduction_kind is StridedReductionKind.SUM
        else normalized_records
    )
    common_contract = (
        bound.source_size,
        bound.required_source_size,
        bound.output_size,
        bound.coverage.value,
        bound.source_dtype,
        bound.result_dtype,
        bound.promoted_scale_dtype,
    )
    canonical_execution_key = _stable_key_bytes(
        "tensor0-stride-canonical-execution-v1",
        COMPILER_POLICY_VERSION,
        common_contract,
        tuple(record.execution_key for record in records),
    )
    capability_key = _stable_key_bytes(
        "tensor0-stride-capability-v2",
        COMPILER_POLICY_VERSION,
        tuple(
            (
                record.logical_shape,
                record.source_strides,
                record.destination_strides,
                record.source_broadcast_axes,
                record.reduction_axes,
                record.layout_kind.value,
            )
            for record in records
        ),
    )
    return CompiledStridePlan(
        bound=bound,
        records=records,
        canonical_execution_key=canonical_execution_key,
        capability_key=capability_key,
    )


def _is_rank4_tiled_candidate(record: CompiledRecord) -> bool:
    if record.rank != 4:
        return False
    return (record.source_strides[1] == 1 and record.destination_strides[3] == 1) or (
        record.source_strides[3] == 1 and record.destination_strides[1] == 1
    )


def _is_rank4_two_pair_candidate(record: CompiledRecord) -> bool:
    if record.rank != 4:
        return False
    n0, n1, n2, n3 = record.logical_shape
    inner = n2 * n3
    return (
        n2 % 8 == 0
        and n3 % 8 == 0
        and record.source_strides == (inner, n0 * inner, 1, n2)
        and record.destination_strides[3] == 1
        and record.destination_strides[2] == n3
        and record.destination_strides[1] >= inner
        and record.destination_strides[0] == n1 * record.destination_strides[1]
    )


def _is_rank2_signed_permutation_candidate(record: CompiledRecord) -> bool:
    if record.rank != 2:
        return False
    rows, columns = record.logical_shape
    return (
        rows > 1
        and columns > 1
        and record.source_strides == (-columns, 1)
        and record.source_offset == (rows - 1) * columns
        and record.destination_strides == (1, rows)
        and record.destination_offset == 0
    )


def _cpu_record(
    record: CompiledRecord,
    *,
    dtype: jnp.dtype,
    item_size: int,
) -> CpuExecutionRecord:
    if record.logical_elements == 0:
        return CpuExecutionRecord(
            kernel_kind=CpuKernelKind.EMPTY,
            loop_axes_fastest_first=(),
            elements_per_unit=0,
            unit_count=0,
            units_per_chunk=0,
            chunk_count=0,
        )

    loop_order = tuple(
        sorted(
            range(record.rank),
            key=lambda axis: (
                abs(record.destination_strides[axis]),
                abs(record.source_strides[axis]),
            ),
        )
    )
    elements_per_unit = 1
    if _is_compact_same_mapping(record):
        kernel_kind = CpuKernelKind.COMPACT
        unit_count = record.logical_elements
    elif (
        dtype == jnp.dtype(jnp.float32)
        and _is_rank2_signed_permutation_candidate(record)
    ):
        kernel_kind = CpuKernelKind.RANK2_SIGNED_PERMUTATION
        if record.logical_shape[0] >= record.logical_shape[1]:
            unit_count = record.logical_shape[1]
            elements_per_unit = record.logical_shape[0]
        else:
            unit_count = record.logical_shape[0]
            elements_per_unit = record.logical_shape[1]
    elif (
        dtype in (jnp.dtype(jnp.float32), jnp.dtype(jnp.complex64))
        and record.rank == 2
        and record.source_strides[0] == 1
        and record.source_strides[1] == record.logical_shape[0]
        and record.destination_strides[1] == 1
        and record.destination_strides[0] == record.logical_shape[1]
    ):
        kernel_kind = CpuKernelKind.RANK2_FORWARD
        unit_count = record.logical_shape[1]
        elements_per_unit = record.logical_shape[0]
    elif (
        dtype in (jnp.dtype(jnp.float32), jnp.dtype(jnp.complex64))
        and record.rank == 2
        and record.source_strides[1] == 1
        and record.source_strides[0] == record.logical_shape[1]
        and record.destination_strides[0] == 1
        and record.destination_strides[1] == record.logical_shape[0]
    ):
        kernel_kind = CpuKernelKind.RANK2_REVERSE
        unit_count = record.logical_shape[0]
        elements_per_unit = record.logical_shape[1]
    elif (
        dtype == jnp.dtype(jnp.float32)
        and _is_rank4_tiled_candidate(record)
        and record.logical_shape[0] != 0
    ):
        kernel_kind = CpuKernelKind.RANK4_AXIS0
        unit_count = record.logical_shape[0]
        elements_per_unit = record.logical_elements // unit_count
    elif (
        dtype == jnp.dtype(jnp.float32)
        and _is_rank4_two_pair_candidate(record)
        and record.logical_shape[0] != 0
    ):
        kernel_kind = CpuKernelKind.RANK4_TWO_PAIR
        unit_count = record.logical_shape[0]
        elements_per_unit = record.logical_elements // unit_count
    elif record.rank == 0:
        kernel_kind = CpuKernelKind.GENERIC
        unit_count = record.logical_elements
    else:
        kernel_kind = CpuKernelKind.GENERIC
        elements_per_unit = record.logical_shape[loop_order[0]]
        unit_count = record.logical_elements // elements_per_unit

    target_elements = max(1, _PARALLEL_CHUNK_BYTES // item_size)
    units_per_chunk = max(1, target_elements // elements_per_unit)
    chunk_count = 1 + (unit_count - 1) // units_per_chunk
    return CpuExecutionRecord(
        kernel_kind=kernel_kind,
        loop_axes_fastest_first=loop_order,
        elements_per_unit=elements_per_unit,
        unit_count=unit_count,
        units_per_chunk=units_per_chunk,
        chunk_count=chunk_count,
    )


def _cpu_reduction_record(
    record: CompiledRecord,
    *,
    input_item_size: int,
) -> CpuExecutionRecord:
    fiber_bytes = record.reduction_count * input_item_size
    units_per_chunk = max(
        1,
        record.output_count
        if fiber_bytes == 0
        else _PARALLEL_CHUNK_BYTES // fiber_bytes,
    )
    chunk_count = (
        0
        if record.output_count == 0
        else 1 + (record.output_count - 1) // units_per_chunk
    )
    return CpuExecutionRecord(
        kernel_kind=CpuKernelKind.STRUCTURED_REDUCTION,
        loop_axes_fastest_first=(
            *record.map_loop_order,
            *record.reduction_loop_order,
        ),
        elements_per_unit=record.reduction_count,
        unit_count=record.output_count,
        units_per_chunk=units_per_chunk,
        chunk_count=chunk_count,
    )


def _project_cpu_execution(
    compiled: CompiledStridePlan,
) -> tuple[tuple[CpuExecutionRecord, ...], int, int, int] | None:
    """Project canonical metadata into transient native CPU decisions."""

    bound = compiled.bound
    if bound.reduction_kind is StridedReductionKind.SUM:
        if (
            bound.source_dtype not in _NATIVE_CPU_DTYPES
            or bound.result_dtype not in _NATIVE_CPU_DTYPES
            or any(
                record.rank > MAXIMUM_RANK
                for record in compiled.records
            )
        ):
            return None
        input_item_size = jnp.dtype(bound.source_dtype).itemsize
        records = tuple(
            _cpu_reduction_record(
                record,
                input_item_size=input_item_size,
            )
            for record in compiled.records
        )
        chunks_per_batch = sum(record.chunk_count for record in records)
        parallel_minimum_bytes = _PARALLEL_CHUNK_BYTES
        return (
            records,
            _PARALLEL_CHUNK_BYTES,
            parallel_minimum_bytes,
            chunks_per_batch,
        )
    if (
        bound.source_dtype not in _NATIVE_CPU_DTYPES
        or bound.result_dtype not in _NATIVE_CPU_DTYPES
        or any(len(record.logical_shape) > MAXIMUM_RANK for record in bound.records)
    ):
        return None
    dtype = jnp.dtype(
        bound.source_dtype
        if bound.scalar_kind is StridedScalarKind.JAX_TRANSPOSE
        else bound.result_dtype
    )
    item_size = dtype.itemsize
    records = tuple(
        _cpu_record(record, dtype=dtype, item_size=item_size)
        for record in compiled.records
    )
    all_compact = all(_is_compact_same_mapping(record) for record in compiled.records)
    has_rank4_tiled = any(
        not _is_compact_same_mapping(record) and _is_rank4_tiled_candidate(record)
        for record in compiled.records
    )
    scaled_compact_c64 = (
        dtype == jnp.dtype(jnp.complex64)
        and all_compact
        and any(record.scale_kind is not ScaleKind.UNIT for record in compiled.records)
    )
    if scaled_compact_c64:
        parallel_minimum_bytes = UINT64_MAX
    elif all_compact:
        parallel_minimum_bytes = _PARALLEL_COMPACT_MINIMUM_BYTES
    elif has_rank4_tiled and len(compiled.records) == 1:
        parallel_minimum_bytes = _PARALLEL_RANK4_MINIMUM_BYTES
    else:
        parallel_minimum_bytes = _PARALLEL_GENERAL_MINIMUM_BYTES
    chunks_per_batch = sum(record.chunk_count for record in records)
    return (
        records,
        _PARALLEL_CHUNK_BYTES,
        parallel_minimum_bytes,
        chunks_per_batch,
    )


def _descriptor_scale_words(
    scale_bytes: bytes,
    dtype: jnp.dtype,
) -> tuple[int, int]:
    """Project exact scalar bytes into the two raw descriptor scale words."""

    if dtype in (jnp.dtype(jnp.complex64), jnp.dtype(jnp.complex128)):
        component_size = dtype.itemsize // 2
        if len(scale_bytes) != dtype.itemsize:
            raise ValueError(
                f"{dtype.name} scale must contain {dtype.itemsize} bytes"
            )
        return (
            int.from_bytes(scale_bytes[:component_size], byteorder="little"),
            int.from_bytes(scale_bytes[component_size:], byteorder="little"),
        )
    expected_size = {
        jnp.dtype(jnp.bool_): 1,
        jnp.dtype(jnp.int8): 1,
        jnp.dtype(jnp.uint8): 1,
        jnp.dtype(jnp.float16): 2,
        jnp.dtype(jnp.bfloat16): 2,
        jnp.dtype(jnp.int16): 2,
        jnp.dtype(jnp.uint16): 2,
        jnp.dtype(jnp.float32): 4,
        jnp.dtype(jnp.int32): 4,
        jnp.dtype(jnp.uint32): 4,
        jnp.dtype(jnp.float64): 8,
        jnp.dtype(jnp.int64): 8,
        jnp.dtype(jnp.uint64): 8,
    }.get(dtype)
    if expected_size is None:
        raise ValueError(f"native stride ABI v7 does not support {dtype.name}")
    if len(scale_bytes) != expected_size:
        raise ValueError(f"{dtype.name} scale must contain {expected_size} bytes")
    return int.from_bytes(scale_bytes, byteorder="little"), 0


def lower_compiled_descriptor(
    *,
    raw_descriptor_words: tuple[int, ...],
    compiled: CompiledStridePlan,
    cpu_records: tuple[CpuExecutionRecord, ...],
    preferred_chunk_bytes: int,
    parallel_minimum_bytes: int,
    chunks_per_batch: int,
) -> bytes:
    """Encode one mixed-capable ABI-v7 compiled descriptor."""

    if len(compiled.records) != len(cpu_records):
        raise ValueError("compiled and CPU record counts differ")
    bound = compiled.bound
    scalar_dtype = jnp.dtype(
        bound.source_dtype
        if bound.scalar_kind is StridedScalarKind.JAX_TRANSPOSE
        else bound.result_dtype
    )
    words: list[int] = [
        COMPILED_DESCRIPTOR_MAGIC,
        COMPILED_DESCRIPTOR_VERSION,
        0,
        compiled.policy_version,
        CPU_POLICY_VERSION,
        len(raw_descriptor_words),
        len(compiled.records),
        preferred_chunk_bytes,
        parallel_minimum_bytes,
        chunks_per_batch,
        _RAW_DESCRIPTOR_WITNESS,
        _dtype_code(jnp.dtype(bound.source_dtype)),
        _dtype_code(jnp.dtype(bound.result_dtype)),
        (
            2
            if bound.scalar_kind is StridedScalarKind.JAX_TRANSPOSE
            else 1
        ),
        *raw_descriptor_words,
    ]
    for record, cpu_record in zip(
        compiled.records,
        cpu_records,
        strict=True,
    ):
        scale_real_bits, scale_imaginary_bits = _descriptor_scale_words(
            record.scale_bytes,
            scalar_dtype,
        )
        words.extend(
            (
                record.rank,
                _LAYOUT_CODES[record.layout_kind],
                _SCALE_CODES[record.scale_kind],
                _CPU_KERNEL_CODES[cpu_record.kernel_kind],
                record.logical_elements,
                record.source_offset,
                record.destination_offset,
                scale_real_bits,
                scale_imaginary_bits,
                sum(1 << axis for axis in record.source_broadcast_axes),
                cpu_record.elements_per_unit,
                cpu_record.unit_count,
                cpu_record.units_per_chunk,
                cpu_record.chunk_count,
                len(cpu_record.loop_axes_fastest_first),
                len(record.axis_provenance_fastest_first),
                *record.logical_shape,
                *(stride & UINT64_MAX for stride in record.source_strides),
                *(stride & UINT64_MAX for stride in record.destination_strides),
                *cpu_record.loop_axes_fastest_first,
            )
        )
        for group in record.axis_provenance_fastest_first:
            words.extend((len(group), *group))
    words[2] = len(words)
    for index, word in enumerate(words):
        if word < 0 or word > UINT64_MAX:
            raise ValueError(f"compiled descriptor word[{index}] exceeds uint64")
    return b"".join(struct.pack("<Q", word) for word in words)


__all__ = [
    "COMPILED_DESCRIPTOR_MAGIC",
    "COMPILED_DESCRIPTOR_VERSION",
    "COMPILER_POLICY_VERSION",
    "CPU_POLICY_VERSION",
    "CompiledRecord",
    "CompiledStridePlan",
    "CpuExecutionRecord",
    "CpuKernelKind",
    "LayoutKind",
    "ScaleKind",
    "compile_plan",
    "lower_compiled_descriptor",
]
