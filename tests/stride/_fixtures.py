from __future__ import annotations

import jax.numpy as jnp
from jax.typing import DTypeLike

from tensor0._stride import (
    StridedCopyPlan,
    CompleteMode,
    StridedCopyRecord,
    build_strided_copy_plan,
)


def _bind(
    records: tuple[StridedCopyRecord, ...],
    *,
    size: int,
) -> StridedCopyPlan:
    return build_strided_copy_plan(
        records=records,
        output_size=size,
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=size,
        source_dtype=jnp.float32,
        result_dtype=jnp.float32,
    )


def two_record_noncompact_plan() -> StridedCopyPlan:
    """Two disjoint column partitions, swapped at the destination."""

    records = (
        StridedCopyRecord(
            logical_shape=(4, 2),
            source_strides=(4, 1),
            source_offset=0,
            destination_strides=(4, 1),
            destination_offset=2,
            scale=1.25,
        ),
        StridedCopyRecord(
            logical_shape=(4, 2),
            source_strides=(4, 1),
            source_offset=2,
            destination_strides=(4, 1),
            destination_offset=0,
            scale=-0.5,
        ),
    )
    return _bind(records, size=16)


def heterogeneous_grid_fusion_plan() -> StridedCopyPlan:
    """A layout-backed grid where only one block fuses across both axes."""

    records = (
        StridedCopyRecord((2, 3), (3, 1), 0, (3, 1), 0),
        StridedCopyRecord((2, 2), (4, 1), 6, (4, 1), 6),
        StridedCopyRecord((2, 2), (4, 1), 8, (4, 1), 8),
    )
    return _bind(records, size=14)


def rank_zero_plan() -> StridedCopyPlan:
    record = StridedCopyRecord((), (), 0, (), 0, 2.5)
    return build_strided_copy_plan(
        records=(record,),
        output_size=1,
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=1,
        source_dtype=jnp.float32,
        result_dtype=jnp.float32,
    )


def empty_plan() -> StridedCopyPlan:
    return build_strided_copy_plan(
        records=(),
        output_size=0,
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=0,
        source_dtype=jnp.float32,
        result_dtype=jnp.float32,
    )


def rank2_transpose_plan(
    *,
    rows: int = 8,
    columns: int = 12,
) -> StridedCopyPlan:
    record = StridedCopyRecord(
        (rows, columns),
        (1, rows),
        0,
        (columns, 1),
        0,
        1.25,
    )
    size = rows * columns
    return build_strided_copy_plan(
        records=(record,),
        output_size=size,
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=size,
        source_dtype=jnp.float32,
        result_dtype=jnp.float32,
    )


def dtype_transpose_plan(
    dtype: DTypeLike,
    scale: int | float | complex,
    *,
    rows: int = 5,
    columns: int = 7,
) -> StridedCopyPlan:
    record = StridedCopyRecord(
        (rows, columns),
        (1, rows),
        0,
        (columns, 1),
        0,
        scale,
    )
    size = rows * columns
    return build_strided_copy_plan(
        records=(record,),
        output_size=size,
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=size,
        source_dtype=dtype,
        result_dtype=dtype,
    )


def contiguous_dtype_plan(
    dtype: DTypeLike,
    scale: int | float | complex,
    *,
    size: int,
) -> StridedCopyPlan:
    record = StridedCopyRecord(
        (size,),
        (1,),
        0,
        (1,),
        0,
        scale,
    )
    return build_strided_copy_plan(
        records=(record,),
        output_size=size,
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=size,
        source_dtype=dtype,
        result_dtype=dtype,
    )


def many_tiny_balanced_plan() -> StridedCopyPlan:
    record_count = 1_024
    record_size = 8
    records = tuple(
        StridedCopyRecord(
            (record_size,),
            (1,),
            record * record_size,
            (1,),
            record * record_size,
            1.0,
        )
        for record in range(record_count)
    )
    size = record_count * record_size
    return build_strided_copy_plan(
        records=records,
        output_size=size,
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=size,
        source_dtype=jnp.float32,
        result_dtype=jnp.float32,
    )


def composite_pack_plan() -> StridedCopyPlan:
    """One compact and one generic partition at the retained pack threshold."""

    compact_size = 4_096
    generic_shape = (7, 64, 64)
    records = (
        StridedCopyRecord(
            (compact_size,),
            (1,),
            0,
            (1,),
            0,
        ),
        StridedCopyRecord(
            generic_shape,
            (4_096, 64, 1),
            compact_size,
            (1, 7, 448),
            compact_size,
        ),
    )
    return _bind(records, size=32_768)


def rank4_tiled_plan() -> StridedCopyPlan:
    record = StridedCopyRecord(
        (2, 4, 3, 8),
        (32, 1, 64, 4),
        0,
        (96, 24, 8, 1),
        0,
        -0.75,
    )
    return build_strided_copy_plan(
        records=(record,),
        output_size=192,
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=192,
        source_dtype=jnp.float32,
        result_dtype=jnp.float32,
    )


def rank4_avx2_shape_plan(
    shape: tuple[int, int, int, int] = (2, 8, 3, 12),
) -> StridedCopyPlan:
    axis0, axis1, axis2, axis3 = shape
    size = axis0 * axis1 * axis2 * axis3
    record = StridedCopyRecord(
        shape,
        (axis1 * axis3, 1, axis0 * axis1 * axis3, axis1),
        0,
        (axis1 * axis2 * axis3, axis2 * axis3, axis3, 1),
        0,
        1.25,
    )
    return build_strided_copy_plan(
        records=(record,),
        output_size=size,
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=size,
        source_dtype=jnp.float32,
        result_dtype=jnp.float32,
    )


def rank4_two_pair_plan(
    shape: tuple[int, int, int, int] = (8, 8, 8, 8),
    *,
    scale: float = 1.0,
) -> StridedCopyPlan:
    axis0, axis1, axis2, axis3 = shape
    size = axis0 * axis1 * axis2 * axis3
    inner = axis2 * axis3
    record = StridedCopyRecord(
        shape,
        (inner, axis0 * inner, 1, axis2),
        0,
        (axis1 * inner, inner, axis3, 1),
        0,
        scale,
    )
    return build_strided_copy_plan(
        records=(record,),
        output_size=size,
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=size,
        source_dtype=jnp.float32,
        result_dtype=jnp.float32,
    )


def large_contiguous_plan() -> StridedCopyPlan:
    size = 131_072
    record = StridedCopyRecord(
        (size,),
        (1,),
        0,
        (1,),
        0,
        1.25,
    )
    return build_strided_copy_plan(
        records=(record,),
        output_size=size,
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=size,
        source_dtype=jnp.float32,
        result_dtype=jnp.float32,
    )


def gapped_single_source_plan() -> StridedCopyPlan:
    record = StridedCopyRecord(
        (3,),
        (2,),
        0,
        (1,),
        0,
        -2.0,
    )
    return build_strided_copy_plan(
        records=(record,),
        output_size=3,
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=5,
        source_dtype=jnp.float32,
        result_dtype=jnp.float32,
    )


def gapped_interval_source_plan() -> StridedCopyPlan:
    records = (
        StridedCopyRecord((2,), (1,), 0, (1,), 0, 1.5),
        StridedCopyRecord((2,), (1,), 3, (1,), 2, -0.5),
    )
    return build_strided_copy_plan(
        records=records,
        output_size=4,
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=5,
        source_dtype=jnp.float32,
        result_dtype=jnp.float32,
    )


def partial_mixed_plan() -> StridedCopyPlan:
    record = StridedCopyRecord(
        (16,),
        (2,),
        1,
        (3,),
        2,
        0.5,
    )
    return build_strided_copy_plan(
        records=(record,),
        output_size=50,
        coverage=CompleteMode.PARTIAL_UNIQUE_ZERO_FILL,
        source_size=32,
        source_dtype=jnp.float32,
        result_dtype=jnp.float32,
    )


def noncompact_identity_plan(
    logical_size: int,
    dtype: DTypeLike = jnp.float32,
) -> StridedCopyPlan:
    """One positive-affine identity map with a hole after every element."""

    record = StridedCopyRecord(
        logical_shape=(logical_size,),
        source_strides=(2,),
        source_offset=0,
        destination_strides=(2,),
        destination_offset=0,
    )
    storage_size = 0 if logical_size == 0 else 2 * logical_size - 1
    return build_strided_copy_plan(
        records=(record,),
        output_size=storage_size,
        coverage=CompleteMode.PARTIAL_UNIQUE_ZERO_FILL,
        source_size=storage_size,
        source_dtype=dtype,
        result_dtype=dtype,
    )


def selected_scale_plan(
    dtype: DTypeLike = jnp.float32,
) -> StridedCopyPlan:
    """A partial identity-address selection with no static scaling."""

    record = StridedCopyRecord(
        (16,),
        (3,),
        2,
        (3,),
        2,
        1,
    )
    return build_strided_copy_plan(
        records=(record,),
        output_size=50,
        coverage=CompleteMode.PARTIAL_UNIQUE_ZERO_FILL,
        source_size=50,
        source_dtype=dtype,
        result_dtype=dtype,
    )


def large_partial_transpose_plan() -> StridedCopyPlan:
    rows = 1024
    columns = 512
    padding = 1024
    size = rows * columns
    record = StridedCopyRecord(
        (rows, columns),
        (1, rows),
        0,
        (columns, 1),
        padding,
        -0.75,
    )
    return build_strided_copy_plan(
        records=(record,),
        output_size=size + 2 * padding,
        coverage=CompleteMode.PARTIAL_UNIQUE_ZERO_FILL,
        source_size=size,
        source_dtype=jnp.float32,
        result_dtype=jnp.float32,
    )


def empty_partial_plan() -> StridedCopyPlan:
    return build_strided_copy_plan(
        records=(),
        output_size=8,
        coverage=CompleteMode.PARTIAL_UNIQUE_ZERO_FILL,
        source_size=0,
        source_dtype=jnp.float32,
        result_dtype=jnp.float32,
    )


def u1_two_record_plan() -> StridedCopyPlan:
    """Frozen address metadata from the real Phase 1 U1 small fixture."""

    records = (
        StridedCopyRecord((1, 1, 1), (1, 1, 1), 0, (1, 1, 1), 0),
        StridedCopyRecord((2, 2, 1), (1, 2, 1), 1, (2, 1, 1), 1),
    )
    return _bind(records, size=5)


def u1_three_record_plan() -> StridedCopyPlan:
    """Frozen address metadata from the real Phase 1 U1 large fixture."""

    records = (
        StridedCopyRecord((12, 16, 8), (8, 96, 1), 0, (256, 8, 1), 0),
        StridedCopyRecord((12, 16, 8), (8, 96, 1), 1536, (256, 8, 1), 128),
        StridedCopyRecord((12, 16, 8), (8, 96, 1), 3072, (128, 8, 1), 3072),
    )
    return _bind(records, size=4608)
