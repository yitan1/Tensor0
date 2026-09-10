"""Semantic native descriptor lowering for validated affine plans."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import struct

import jax.numpy as jnp

from ._scalar import mapping_dtype
from ._plan import (
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
    _COVERAGE_COMPLETE_UNIQUE,
    _COVERAGE_PARTIAL_UNIQUE_ZERO_FILL,
    CompleteMode,
    AffinePlan,
    StridedReductionKind,
    StridedScalarKind,
    _exact_scalar_bytes,
    _u64,
)


AFFINE_DESCRIPTOR_VERSION = 9
AFFINE_DESCRIPTOR_MAGIC = 0x3150444952543054


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


def _dtype_code(dtype: jnp.dtype) -> int:
    code = _DTYPE_CODES.get(dtype)
    if code is None:
        raise ValueError(f"native stride descriptor does not support {dtype.name}")
    return code


def _descriptor_scale_words(
    scale_bytes: bytes,
    dtype: jnp.dtype,
) -> tuple[int, int]:
    """Project exact scalar bytes into the two raw descriptor scale words."""

    _dtype_code(dtype)
    if len(scale_bytes) != dtype.itemsize:
        raise ValueError(f"{dtype.name} scale must contain {dtype.itemsize} bytes")
    if dtype in (jnp.dtype(jnp.complex64), jnp.dtype(jnp.complex128)):
        component_size = dtype.itemsize // 2
        return (
            int.from_bytes(scale_bytes[:component_size], byteorder="little"),
            int.from_bytes(scale_bytes[component_size:], byteorder="little"),
        )
    return int.from_bytes(scale_bytes, byteorder="little"), 0


def _descriptor_words(descriptor: bytes) -> tuple[int, ...]:
    if len(descriptor) % 8 != 0:
        raise AssertionError("native descriptor size is not a multiple of uint64")
    return tuple(word[0] for word in struct.iter_unpack("<Q", descriptor))


class NativeCallKind(str, Enum):
    """The native ABI family selected for one encoded call."""

    AFFINE_MAP = "affine_map"
    STRUCTURED_REDUCTION = "structured_reduction"


@dataclass(frozen=True, slots=True)
class NativeCallSpec:
    """One immutable native descriptor bound to its semantic affine plan."""

    kind: NativeCallKind
    descriptor: bytes
    semantic: AffinePlan = field(
        compare=False,
        hash=False,
        repr=False,
    )

    @property
    def words(self) -> tuple[int, ...]:
        """Decode current native descriptor words for diagnostics."""

        return _descriptor_words(self.descriptor)




def lower_plan(
    plan: AffinePlan,
    *,
    dynamic_dtype: str | None = None,
    coefficient_dtypes: tuple[str, ...] | None = None,
) -> NativeCallSpec:
    """Encode raw affine semantics for native compile-at-instantiate."""
    if plan.reduction_kind is not StridedReductionKind.NONE:
        raise ValueError("ordinary affine descriptor does not encode reduction")
    mapping_source_dtype = jnp.dtype(
        plan.result_dtype if plan.scalar_kind is StridedScalarKind.JAX_TRANSPOSE
        else plan.source_dtype
    )
    scalar_policy = (
        2
        if plan.scalar_kind is StridedScalarKind.JAX_TRANSPOSE
        else 1
    )
    words: list[int] = [
        AFFINE_DESCRIPTOR_MAGIC,
        AFFINE_DESCRIPTOR_VERSION,
        0,
        plan.source_size,
        plan.required_source_size,
        plan.output_size,
        len(plan.records),
        (
            _COVERAGE_COMPLETE_UNIQUE
            if plan.coverage is CompleteMode.COMPLETE_UNIQUE
            else _COVERAGE_PARTIAL_UNIQUE_ZERO_FILL
        ),
        _dtype_code(jnp.dtype(plan.source_dtype)),
        _dtype_code(jnp.dtype(plan.result_dtype)),
        scalar_policy,
        0,
    ]
    for index, record in enumerate(plan.records):
        if coefficient_dtypes is not None:
            scalar_dtype = jnp.dtype(coefficient_dtypes[index])
        elif dynamic_dtype is not None:
            scalar_dtype = jnp.dtype(dynamic_dtype)
        else:
            scalar_dtype = mapping_dtype(mapping_source_dtype, record.scale)
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
                int(record.scale is None),
                _dtype_code(scalar_dtype),
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
    descriptor = b"".join(struct.pack("<Q", word) for word in words)
    return NativeCallSpec(
        kind=NativeCallKind.AFFINE_MAP,
        descriptor=descriptor,
        semantic=plan,
    )
