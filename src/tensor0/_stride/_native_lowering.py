"""Native descriptor lowering for validated strided-copy plans."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import struct

import jax.numpy as jnp

from ._compiler import (
    CompiledStridePlan,
    CpuExecutionRecord,
    _descriptor_scale_words,
    _dtype_code,
    _project_cpu_execution,
    compile_plan,
    lower_compiled_descriptor,
)
from ._plan import (
    MAXIMUM_RANK,
    _COVERAGE_COMPLETE_UNIQUE,
    _COVERAGE_PARTIAL_UNIQUE_ZERO_FILL,
    CompleteMode,
    StridedCopyPlan,
    StridedReductionKind,
    StridedScalarKind,
    _exact_scalar_bytes,
    _u64,
)


_SEMANTIC_DESCRIPTOR_VERSION = 5
_SEMANTIC_DESCRIPTOR_MAGIC = 0x3150444952543054


def _descriptor_words(descriptor: bytes) -> tuple[int, ...]:
    if len(descriptor) % 8 != 0:
        raise AssertionError("native descriptor size is not a multiple of uint64")
    return tuple(word[0] for word in struct.iter_unpack("<Q", descriptor))


class NativeExecutionKind(str, Enum):
    """The prepared native ABI family owned by one execution artifact."""

    AFFINE_MAP = "affine_map"
    STRUCTURED_REDUCTION = "structured_reduction"


@dataclass(frozen=True, slots=True)
class NativeExecutionPlan:
    """One immutable native descriptor plus its retained route metadata."""

    kind: NativeExecutionKind
    descriptor: bytes
    compiled: CompiledStridePlan = field(
        compare=False,
        hash=False,
        repr=False,
    )
    records: tuple[CpuExecutionRecord, ...] = field(
        compare=False,
        hash=False,
        repr=False,
    )
    preferred_chunk_bytes: int = field(compare=False, hash=False, repr=False)
    parallel_minimum_bytes: int = field(compare=False, hash=False, repr=False)
    chunks_per_batch: int = field(compare=False, hash=False, repr=False)

    @property
    def bound(self) -> StridedCopyPlan:
        return self.compiled.bound

    @property
    def words(self) -> tuple[int, ...]:
        """Decode current native descriptor words for diagnostics."""

        return _descriptor_words(self.descriptor)


def lower_compiled_plan(compiled: CompiledStridePlan) -> NativeExecutionPlan:
    """Lower one canonical affine plan to a mixed-capable native artifact."""

    bound = compiled.bound
    if bound.reduction_kind is not StridedReductionKind.NONE:
        raise ValueError("ordinary affine descriptor does not encode reduction")
    scalar_dtype = jnp.dtype(
        bound.source_dtype
        if bound.scalar_kind is StridedScalarKind.JAX_TRANSPOSE
        else bound.result_dtype
    )
    dtype_code = _dtype_code(scalar_dtype)
    for index, record in enumerate(bound.records):
        if len(record.logical_shape) > MAXIMUM_RANK:
            raise ValueError(
                f"record[{index}] rank exceeds native limit {MAXIMUM_RANK}"
            )

    words: list[int] = [
        _SEMANTIC_DESCRIPTOR_MAGIC,
        _SEMANTIC_DESCRIPTOR_VERSION,
        0,
        bound.source_size,
        bound.required_source_size,
        bound.output_size,
        len(bound.records),
        (
            _COVERAGE_COMPLETE_UNIQUE
            if bound.coverage is CompleteMode.COMPLETE_UNIQUE
            else _COVERAGE_PARTIAL_UNIQUE_ZERO_FILL
        ),
        dtype_code,
    ]
    for record in bound.records:
        scale_real, scale_imaginary = _descriptor_scale_words(
            _exact_scalar_bytes(record.scale, scalar_dtype),
            scalar_dtype,
        )
        words.extend(
            (
                len(record.logical_shape),
                record.source_offset,
                record.destination_offset,
                scale_real,
                scale_imaginary,
                sum(1 << axis for axis in record.source_broadcast_axes),
                *record.logical_shape,
                *(stride & ((1 << 64) - 1) for stride in record.source_strides),
                *(
                    stride & ((1 << 64) - 1)
                    for stride in record.destination_strides
                ),
            )
        )
    words[2] = len(words)
    for index, word in enumerate(words):
        _u64(word, f"descriptor[{index}]")
    frozen = tuple(words)
    projection = _project_cpu_execution(compiled)
    if projection is None:
        raise ValueError("affine plan has no native CPU projection")
    (
        cpu_records,
        preferred_chunk_bytes,
        parallel_minimum_bytes,
        chunks_per_batch,
    ) = projection
    descriptor = lower_compiled_descriptor(
        raw_descriptor_words=frozen,
        compiled=compiled,
        cpu_records=cpu_records,
        preferred_chunk_bytes=preferred_chunk_bytes,
        parallel_minimum_bytes=parallel_minimum_bytes,
        chunks_per_batch=chunks_per_batch,
    )
    return NativeExecutionPlan(
        kind=NativeExecutionKind.AFFINE_MAP,
        descriptor=descriptor,
        compiled=compiled,
        records=cpu_records,
        preferred_chunk_bytes=preferred_chunk_bytes,
        parallel_minimum_bytes=parallel_minimum_bytes,
        chunks_per_batch=chunks_per_batch,
    )


def lower_plan(bound: StridedCopyPlan) -> NativeExecutionPlan:
    """Compile and lower one semantic affine plan."""

    return lower_compiled_plan(compile_plan(bound))
