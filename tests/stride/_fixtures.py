from __future__ import annotations

from jax import Array
import jax.numpy as jnp
from jax.typing import DTypeLike

from tensor0._stride import StridedView
from tensor0._stride._plan import (
    AffinePlan,
    CompleteMode,
    AffineRecord,
    build_affine_plan,
)


def _bind(
    records: tuple[AffineRecord, ...],
    *,
    size: int,
) -> AffinePlan:
    return build_affine_plan(
        records=records,
        output_size=size,
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=size,
        source_dtype=jnp.float32,
        result_dtype=jnp.float32,
    )


def two_record_noncompact_plan() -> AffinePlan:
    """Two disjoint column partitions, swapped at the destination."""

    records = (
        AffineRecord(
            logical_shape=(4, 2),
            source_strides=(4, 1),
            source_offset=0,
            destination_strides=(4, 1),
            destination_offset=2,
            scale=1.25,
        ),
        AffineRecord(
            logical_shape=(4, 2),
            source_strides=(4, 1),
            source_offset=2,
            destination_strides=(4, 1),
            destination_offset=0,
            scale=-0.5,
        ),
    )
    return _bind(records, size=16)


def heterogeneous_grid_fusion_plan() -> AffinePlan:
    """A layout-backed grid where only one block fuses across both axes."""

    records = (
        AffineRecord((2, 3), (3, 1), 0, (3, 1), 0),
        AffineRecord((2, 2), (4, 1), 6, (4, 1), 6),
        AffineRecord((2, 2), (4, 1), 8, (4, 1), 8),
    )
    return _bind(records, size=14)


def rank_zero_plan() -> AffinePlan:
    record = AffineRecord((), (), 0, (), 0, 2.5)
    return build_affine_plan(
        records=(record,),
        output_size=1,
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=1,
        source_dtype=jnp.float32,
        result_dtype=jnp.float32,
    )


def empty_plan() -> AffinePlan:
    return build_affine_plan(
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
) -> AffinePlan:
    record = AffineRecord(
        (rows, columns),
        (1, rows),
        0,
        (columns, 1),
        0,
        1.25,
    )
    size = rows * columns
    return build_affine_plan(
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
) -> AffinePlan:
    record = AffineRecord(
        (rows, columns),
        (1, rows),
        0,
        (columns, 1),
        0,
        scale,
    )
    size = rows * columns
    return build_affine_plan(
        records=(record,),
        output_size=size,
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=size,
        source_dtype=dtype,
        result_dtype=dtype,
    )


def contiguous_dtype_plan(
    dtype: DTypeLike,
    scale: int | float | complex | None,
    *,
    size: int,
) -> AffinePlan:
    record = AffineRecord(
        (size,),
        (1,),
        0,
        (1,),
        0,
        scale,
    )
    return build_affine_plan(
        records=(record,),
        output_size=size,
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=size,
        source_dtype=dtype,
        result_dtype=dtype,
    )


def many_tiny_balanced_plan() -> AffinePlan:
    record_count = 1_024
    record_size = 8
    records = tuple(
        AffineRecord(
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
    return build_affine_plan(
        records=records,
        output_size=size,
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=size,
        source_dtype=jnp.float32,
        result_dtype=jnp.float32,
    )


def rank4_tiled_plan() -> AffinePlan:
    record = AffineRecord(
        (2, 4, 3, 8),
        (32, 1, 64, 4),
        0,
        (96, 24, 8, 1),
        0,
        -0.75,
    )
    return build_affine_plan(
        records=(record,),
        output_size=192,
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=192,
        source_dtype=jnp.float32,
        result_dtype=jnp.float32,
    )


def rank4_avx2_shape_plan(
    shape: tuple[int, int, int, int] = (2, 8, 3, 12),
) -> AffinePlan:
    axis0, axis1, axis2, axis3 = shape
    size = axis0 * axis1 * axis2 * axis3
    record = AffineRecord(
        shape,
        (axis1 * axis3, 1, axis0 * axis1 * axis3, axis1),
        0,
        (axis1 * axis2 * axis3, axis2 * axis3, axis3, 1),
        0,
        1.25,
    )
    return build_affine_plan(
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
) -> AffinePlan:
    axis0, axis1, axis2, axis3 = shape
    size = axis0 * axis1 * axis2 * axis3
    inner = axis2 * axis3
    record = AffineRecord(
        shape,
        (inner, axis0 * inner, 1, axis2),
        0,
        (axis1 * inner, inner, axis3, 1),
        0,
        scale,
    )
    return build_affine_plan(
        records=(record,),
        output_size=size,
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=size,
        source_dtype=jnp.float32,
        result_dtype=jnp.float32,
    )


def large_contiguous_plan() -> AffinePlan:
    size = 131_072
    record = AffineRecord(
        (size,),
        (1,),
        0,
        (1,),
        0,
        1.25,
    )
    return build_affine_plan(
        records=(record,),
        output_size=size,
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=size,
        source_dtype=jnp.float32,
        result_dtype=jnp.float32,
    )


def gapped_single_source_plan() -> AffinePlan:
    record = AffineRecord(
        (3,),
        (2,),
        0,
        (1,),
        0,
        -2.0,
    )
    return build_affine_plan(
        records=(record,),
        output_size=3,
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=5,
        source_dtype=jnp.float32,
        result_dtype=jnp.float32,
    )


def gapped_interval_source_plan() -> AffinePlan:
    records = (
        AffineRecord((2,), (1,), 0, (1,), 0, 1.5),
        AffineRecord((2,), (1,), 3, (1,), 2, -0.5),
    )
    return build_affine_plan(
        records=records,
        output_size=4,
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=5,
        source_dtype=jnp.float32,
        result_dtype=jnp.float32,
    )


def partial_mixed_plan() -> AffinePlan:
    record = AffineRecord(
        (16,),
        (2,),
        1,
        (3,),
        2,
        0.5,
    )
    return build_affine_plan(
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
) -> AffinePlan:
    """One positive-affine identity map with a hole after every element."""

    record = AffineRecord(
        logical_shape=(logical_size,),
        source_strides=(2,),
        source_offset=0,
        destination_strides=(2,),
        destination_offset=0,
    )
    storage_size = 0 if logical_size == 0 else 2 * logical_size - 1
    return build_affine_plan(
        records=(record,),
        output_size=storage_size,
        coverage=CompleteMode.PARTIAL_UNIQUE_ZERO_FILL,
        source_size=storage_size,
        source_dtype=dtype,
        result_dtype=dtype,
    )


def selected_scale_plan(
    dtype: DTypeLike = jnp.float32,
) -> AffinePlan:
    """A partial identity-address selection with no static scaling."""

    record = AffineRecord(
        (16,),
        (3,),
        2,
        (3,),
        2,
        None,
    )
    return build_affine_plan(
        records=(record,),
        output_size=50,
        coverage=CompleteMode.PARTIAL_UNIQUE_ZERO_FILL,
        source_size=50,
        source_dtype=dtype,
        result_dtype=dtype,
    )


def large_partial_transpose_plan() -> AffinePlan:
    rows = 1024
    columns = 512
    padding = 1024
    size = rows * columns
    record = AffineRecord(
        (rows, columns),
        (1, rows),
        0,
        (columns, 1),
        padding,
        -0.75,
    )
    return build_affine_plan(
        records=(record,),
        output_size=size + 2 * padding,
        coverage=CompleteMode.PARTIAL_UNIQUE_ZERO_FILL,
        source_size=size,
        source_dtype=jnp.float32,
        result_dtype=jnp.float32,
    )


def empty_partial_plan() -> AffinePlan:
    return build_affine_plan(
        records=(),
        output_size=8,
        coverage=CompleteMode.PARTIAL_UNIQUE_ZERO_FILL,
        source_size=0,
        source_dtype=jnp.float32,
        result_dtype=jnp.float32,
    )


def u1_two_record_plan() -> AffinePlan:
    """Frozen address metadata from the real Phase 1 U1 small fixture."""

    records = (
        AffineRecord((1, 1, 1), (1, 1, 1), 0, (1, 1, 1), 0),
        AffineRecord((2, 2, 1), (1, 2, 1), 1, (2, 1, 1), 1),
    )
    return _bind(records, size=5)


def u1_three_record_plan() -> AffinePlan:
    """Frozen address metadata from the real Phase 1 U1 large fixture."""

    records = (
        AffineRecord((12, 16, 8), (8, 96, 1), 0, (256, 8, 1), 0),
        AffineRecord((12, 16, 8), (8, 96, 1), 1536, (256, 8, 1), 128),
        AffineRecord((12, 16, 8), (8, 96, 1), 3072, (128, 8, 1), 3072),
    )
    return _bind(records, size=4608)


def execute_update_assign(base, source, *, plan):
    from tensor0._stride._ops._update import _execute_update

    return _execute_update(base, source, source_factor=1, base_factor=0, plan=plan)


def execute_update_accumulate(base, source, *, plan):
    from tensor0._stride._ops._update import _execute_update

    return _execute_update(base, source, source_factor=1, base_factor=1, plan=plan)


def execute_update_scale(base, factor, *, plan):
    from tensor0._stride._ops._update import _execute_update

    return _execute_update(base, base, source_factor=factor, base_factor=0, plan=plan)


def execute_view_assign(destination: StridedView, source: StridedView) -> Array:
    from tensor0._stride._ops._update_support import _prepare_strided_update

    base, values, plan = _prepare_strided_update(destination, source)
    return execute_update_assign(base, values, plan=plan)


def execute_view_accumulate(destination: StridedView, source: StridedView) -> Array:
    from tensor0._stride._ops._update_support import _prepare_strided_update

    base, values, plan = _prepare_strided_update(destination, source)
    return execute_update_accumulate(base, values, plan=plan)
