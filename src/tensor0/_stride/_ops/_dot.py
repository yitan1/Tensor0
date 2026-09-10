"""Native two-input affine dot execution."""

from __future__ import annotations

from functools import lru_cache, partial
from math import prod
import struct

import jax
from jax import Array
import jax.numpy as jnp

from ._materialize import materialize
from .._native import _dot_ffi_call
from .._native_descriptor import _dtype_code
from .._plan import (
    AffinePlan,
    AffineRecord,
    CompleteMode,
    UINT64_MAX,
    build_affine_plan,
    contiguous_strides,
)
from .._map import _execute_map
from .._view import StridedView


_DOT_DESCRIPTOR_MAGIC = 0x31544F4449523054
_DOT_DESCRIPTOR_VERSION = 1


def _append_record_words(
    words: list[int],
    record: AffineRecord,
    *,
    right_strides: tuple[int, ...],
    right_offset: int,
) -> None:
    words.extend(
        (
            len(record.logical_shape),
            record.source_offset,
            right_offset,
            prod(record.logical_shape),
            *record.logical_shape,
            *(stride & UINT64_MAX for stride in record.source_strides),
            *(stride & UINT64_MAX for stride in right_strides),
        )
    )


@lru_cache(maxsize=1_024)
def _dot_descriptor(
    *,
    sizes: tuple[int, ...],
    left_strides: tuple[int, ...],
    left_offset: int,
    left_size: int,
    right_strides: tuple[int, ...],
    right_offset: int,
    right_size: int,
    dtype_name: str,
    conjugate_left: bool,
) -> bytes:
    record = AffineRecord(
        logical_shape=sizes,
        source_strides=left_strides,
        source_offset=left_offset,
        destination_strides=right_strides,
        destination_offset=right_offset,
    )
    words = [
        _DOT_DESCRIPTOR_MAGIC,
        _DOT_DESCRIPTOR_VERSION,
        0,
        left_size,
        right_size,
        1,
        _dtype_code(jnp.dtype(dtype_name)),
        int(conjugate_left),
    ]
    _append_record_words(
        words,
        record,
        right_strides=right_strides,
        right_offset=right_offset,
    )
    words[2] = len(words)
    return b"".join(struct.pack("<Q", word) for word in words)


@lru_cache(maxsize=1_024)
def _projection_dot_descriptor(plan: AffinePlan) -> bytes:
    words = [
        _DOT_DESCRIPTOR_MAGIC,
        _DOT_DESCRIPTOR_VERSION,
        0,
        plan.source_size,
        plan.source_size,
        len(plan.records),
        _dtype_code(jnp.dtype(plan.source_dtype)),
        0,
    ]
    for record in plan.records:
        _append_record_words(
            words,
            record,
            right_strides=record.source_strides,
            right_offset=record.source_offset,
        )
    words[2] = len(words)
    return b"".join(struct.pack("<Q", word) for word in words)


@lru_cache(maxsize=1_024)
def _compact_projection_plan(plan: AffinePlan) -> AffinePlan:
    destination_offset = 0
    records: list[AffineRecord] = []
    for record in plan.records:
        records.append(
            AffineRecord(
                logical_shape=record.logical_shape,
                source_strides=record.source_strides,
                source_offset=record.source_offset,
                destination_strides=contiguous_strides(record.logical_shape),
                destination_offset=destination_offset,
            )
        )
        destination_offset += prod(record.logical_shape)
    return build_affine_plan(
        records=tuple(records),
        output_size=destination_offset,
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=plan.source_size,
        source_dtype=plan.source_dtype,
        result_dtype=plan.result_dtype,
    )


def _execute_native_dot(
    left: StridedView,
    right: StridedView,
    conjugate_left: bool,
) -> Array:
    descriptor = _dot_descriptor(
        sizes=left.sizes,
        left_strides=left.strides,
        left_offset=left.offset,
        left_size=left.storage_size,
        right_strides=right.strides,
        right_offset=right.offset,
        right_size=right.storage_size,
        dtype_name=left.data.dtype.name,
        conjugate_left=conjugate_left,
    )
    return _dot_ffi_call(
        left.data,
        right.data,
        descriptor=descriptor,
        conjugate_left=conjugate_left,
    )


def _reference_dot(
    left: StridedView,
    right: StridedView,
    conjugate_left: bool,
) -> Array:
    left_value = materialize(left)
    right_value = materialize(right)
    if conjugate_left:
        left_value = jnp.conj(left_value)
    logical_axes = tuple(range(left_value.ndim - left.rank, left_value.ndim))
    return jnp.sum(left_value * right_value, axis=logical_axes)


def _reference_projection_dot(
    left: Array,
    right: Array,
    plan: AffinePlan,
) -> Array:
    compact = _compact_projection_plan(plan)
    selected_left = _execute_map(left, plan=compact)
    selected_right = _execute_map(right, plan=compact)
    return jnp.sum(selected_left * selected_right, axis=-1)


@partial(jax.custom_jvp, nondiff_argnums=(2,))
def _execute_dot(
    left: StridedView,
    right: StridedView,
    conjugate_left: bool,
) -> Array:
    """Execute one same-dtype affine DOTU or DOTC."""

    return _execute_native_dot(left, right, conjugate_left)


@_execute_dot.defjvp
def _native_dot_jvp(
    conjugate_left: bool,
    primals: tuple[StridedView, StridedView],
    tangents: tuple[StridedView, StridedView],
) -> tuple[Array, Array]:
    left, right = primals
    left_tangent, right_tangent = tangents
    primal = _execute_dot(left, right, conjugate_left)
    tangent = _reference_dot(left_tangent, right, conjugate_left)
    tangent += _reference_dot(left, right_tangent, conjugate_left)
    return primal, tangent


@partial(jax.custom_jvp, nondiff_argnums=(2,))
def native_projection_dotu(
    left: Array,
    right: Array,
    plan: AffinePlan,
) -> Array:
    """Reduce matching addresses from every record with one native DOTU."""

    return _dot_ffi_call(
        left,
        right,
        descriptor=_projection_dot_descriptor(plan),
        conjugate_left=False,
    )


@native_projection_dotu.defjvp
def _native_projection_dotu_jvp(
    plan: AffinePlan,
    primals: tuple[Array, Array],
    tangents: tuple[Array, Array],
) -> tuple[Array, Array]:
    left, right = primals
    left_tangent, right_tangent = tangents
    primal = native_projection_dotu(left, right, plan)
    tangent = _reference_projection_dot(left_tangent, right, plan)
    tangent += _reference_projection_dot(left, right_tangent, plan)
    return primal, tangent


__all__ = ["_execute_dot", "native_projection_dotu"]
