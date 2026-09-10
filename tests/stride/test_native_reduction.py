from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from itertools import permutations, product
from math import prod
import warnings

import jax
import jax.numpy as jnp
from jax.typing import DTypeLike
import numpy as np
import pytest

from tensor0._stride._plan import (
    CompleteMode,
    AffinePlan,
    AffineRecord,
    StridedReductionKind,
    StridedWriteKind,
)
from tensor0._stride._map import _execute_map
from tensor0._stride._native import _structured_reduction_ffi_call
from tensor0._stride._testing import (
    _native_call_count_for_tests,
    _native_grouped_output_owner_for_tests,
    _native_reduction_fiber_chunks_for_tests,
    _native_worker_counts_for_tests,
    _reset_native_call_count_for_tests,
    _set_native_reduction_fiber_parallel_mode_for_tests,
    _set_native_worker_limit_for_tests,
    native_available,
)
from tensor0._stride._ops._reduction import (
    _bind_grouped_native_structured_reduction,
    _bind_sequential_native_structured_reduction,
    bind_native_structured_reduction,
    lower_broadcast_transpose,
    lower_structured_reduction,
)
from tensor0._stride._plan import (
    PlanValidationError,
    build_affine_plan,
)

from ._oracle import assert_bitwise_equal, execute_reference


@pytest.mark.parametrize("mode", ["ordered", "grouped", "fiber"])
@pytest.mark.parametrize("source_dtype", [jnp.float16, jnp.float32])
def test_record_mapping_binding_is_shared_across_reduction_schedules(mode, source_dtype):
    with jax.enable_x64():
        output_count, reduction_count = (1, 65_536) if mode == "fiber" else (9, 17)
        record_size = output_count * reduction_count
        plan = build_affine_plan(
            records=tuple(AffineRecord(
                logical_shape=(output_count, reduction_count),
                source_strides=(reduction_count, 1), source_offset=index * record_size,
                destination_strides=(1, 0), destination_offset=0,
                scale=factor, reduction_axes=(1,),
            ) for index, factor in enumerate((None, np.float32(.75), np.float64(-.25)))),
            source_size=3 * record_size, output_size=output_count,
            source_dtype=source_dtype, result_dtype=jnp.float32,
            coverage=CompleteMode.PARTIAL_UNIQUE_ZERO_FILL,
            write_kind=StridedWriteKind.ACCUMULATE,
            reduction_kind=StridedReductionKind.SUM,
        )
        source = jnp.stack((jnp.ones(plan.source_size, dtype=source_dtype),
                            jnp.full(plan.source_size, 2, dtype=source_dtype)))
        binding = (_bind_grouped_native_structured_reduction if mode == "grouped" else
                   _bind_sequential_native_structured_reduction if mode == "ordered" else
                   bind_native_structured_reduction)
        execute = jax.jit(lambda values: binding(values, plan))
        _set_native_worker_limit_for_tests(4)
        _set_native_reduction_fiber_parallel_mode_for_tests(1 if mode == "fiber" else 0)
        try:
            _reset_native_call_count_for_tests()
            actual = execute(source)
            actual.block_until_ready()
            assert _native_call_count_for_tests() == 1
            chunks = _native_reduction_fiber_chunks_for_tests()
            workers, available = _native_worker_counts_for_tests()
        finally:
            _set_native_reduction_fiber_parallel_mode_for_tests(None)
            _set_native_worker_limit_for_tests(None)
        expected = np.broadcast_to(
            np.asarray([[1.5], [3.]], dtype=np.float32) * reduction_count,
            (2, output_count),
        )
        assert_bitwise_equal(actual, expected)
        if mode == "fiber" and source_dtype == jnp.float32 and available >= 2:
            assert chunks >= 2
            assert workers >= 2


def _row_major_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    strides = [0] * len(shape)
    expected = 1
    for axis in reversed(range(len(shape))):
        strides[axis] = expected
        expected *= shape[axis]
    return tuple(strides)


def _broadcast_plan(
    *,
    logical_shape: tuple[int, ...] = (2, 3),
    broadcast_axes: tuple[int, ...] = (0,),
    source_signs: tuple[int, ...] | None = None,
    destination_fastest: tuple[int, ...] = (1, 0),
    destination_signs: tuple[int, ...] = (1, 1),
    source_dtype: DTypeLike = jnp.float32,
    result_dtype: DTypeLike = jnp.float32,
    scale: int | float | complex = -0.75,
) -> AffinePlan:
    map_axes = tuple(
        axis
        for axis, extent in enumerate(logical_shape)
        if extent > 1 and axis not in broadcast_axes
    )
    if source_signs is None:
        source_signs = (1,) * len(map_axes)
    source_shape = tuple(logical_shape[axis] for axis in map_axes) or (1,)
    source_physical_strides = _row_major_strides(source_shape)
    source_strides = [0] * len(logical_shape)
    source_offset = 0
    for position, (axis, sign) in enumerate(
        zip(map_axes, source_signs, strict=True)
    ):
        stride = source_physical_strides[position]
        source_strides[axis] = sign * stride
        if sign < 0:
            source_offset += (logical_shape[axis] - 1) * stride

    destination_absolute = [0] * len(logical_shape)
    expected = 1
    for axis in destination_fastest:
        destination_absolute[axis] = expected
        expected *= logical_shape[axis]
    destination_strides = tuple(
        sign * stride
        for sign, stride in zip(
            destination_signs,
            destination_absolute,
            strict=True,
        )
    )
    destination_offset = sum(
        (extent - 1) * absolute
        for extent, absolute, stride in zip(
            logical_shape,
            destination_absolute,
            destination_strides,
            strict=True,
        )
        if stride < 0
    )
    record = AffineRecord(
        logical_shape=logical_shape,
        source_strides=tuple(source_strides),
        source_offset=source_offset,
        destination_strides=destination_strides,
        destination_offset=destination_offset,
        scale=scale,
        source_broadcast_axes=broadcast_axes,
    )
    return build_affine_plan(
        records=(record,),
        output_size=prod(logical_shape),
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=prod(source_shape),
        source_dtype=source_dtype,
        result_dtype=result_dtype,
    )


def _values(dtype: DTypeLike, size: int) -> jax.Array:
    resolved = jnp.dtype(dtype)
    values = jnp.linspace(-2, 3, size, dtype=jnp.float32)
    if jnp.issubdtype(resolved, jnp.complexfloating):
        return jnp.asarray(values + 1j * values[::-1], dtype=resolved)
    return jnp.asarray(values, dtype=resolved)


def _native_forward(source: jax.Array, plan: AffinePlan) -> jax.Array:
    return _execute_map(source, plan=plan)


def _forward_sum_plan(
    source_dtype: DTypeLike = jnp.float32,
    result_dtype: DTypeLike = jnp.float32,
    scale: int | float | complex = -0.75,
) -> AffinePlan:
    return build_affine_plan(
        records=(
            AffineRecord(
                logical_shape=(4, 3),
                source_strides=(3, 1),
                source_offset=0,
                destination_strides=(0, 1),
                destination_offset=0,
                scale=scale,
                reduction_axes=(0,),
            ),
        ),
        output_size=3,
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=12,
        source_dtype=source_dtype,
        result_dtype=result_dtype,
        reduction_kind=StridedReductionKind.SUM,
    )


@pytest.mark.parametrize(
    ("source_strides", "source_size", "broadcast_axes"),
    (
        ((1, 1), 5, ()),
        ((1, 0), 3, (1,)),
    ),
)
@pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")
def test_structured_reduction_allows_repeated_source_reads(
    source_strides: tuple[int, int],
    source_size: int,
    broadcast_axes: tuple[int, ...],
) -> None:
    plan = build_affine_plan(
        records=(
            AffineRecord(
                logical_shape=(3, 3),
                source_strides=source_strides,
                source_offset=0,
                destination_strides=(1, 0),
                destination_offset=0,
                source_broadcast_axes=broadcast_axes,
                reduction_axes=(1,),
            ),
        ),
        output_size=3,
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=source_size,
        source_dtype=jnp.float32,
        result_dtype=jnp.float32,
        reduction_kind=StridedReductionKind.SUM,
    )
    source = jnp.arange(source_size, dtype=jnp.float32)

    def expected_for(values: jax.Array) -> np.ndarray:
        source_values = np.asarray(values)
        return np.asarray(
            [
                sum(
                    source_values[
                        output * source_strides[0]
                        + reduction * source_strides[1]
                    ]
                    for reduction in range(3)
                )
                for output in range(3)
            ],
            dtype=np.float32,
        )

    execute = lambda value: bind_native_structured_reduction(value, plan)
    tangent = source * 0.25 + 0.5
    actual, actual_tangent = jax.jvp(execute, (source,), (tangent,))

    np.testing.assert_array_equal(actual, expected_for(source))
    np.testing.assert_array_equal(actual_tangent, expected_for(tangent))
    pullback = jax.vjp(execute, source)[1]
    cotangent = jnp.ones((3,), dtype=jnp.float32)
    if broadcast_axes:
        np.testing.assert_array_equal(
            pullback(cotangent)[0],
            jnp.full((source_size,), 3, dtype=jnp.float32),
        )
    else:
        with pytest.raises(PlanValidationError, match="noninjective_view"):
            pullback(cotangent)


def _grouped_sum_plan() -> AffinePlan:
    return build_affine_plan(
        records=(
            AffineRecord((1,), (1,), 0, (1,), 0, 2),
            AffineRecord((1,), (1,), 1, (1,), 0, -3),
        ),
        output_size=1,
        coverage=CompleteMode.PARTIAL_UNIQUE_ZERO_FILL,
        source_size=2,
        source_dtype=jnp.float32,
        result_dtype=jnp.float32,
        write_kind=StridedWriteKind.ACCUMULATE,
        reduction_kind=StridedReductionKind.SUM,
    )


def _exact_output_group_plan(
    *,
    output_count: int = 5,
    reduction_count: int = 3,
    dtype: DTypeLike = jnp.float32,
    result_dtype: DTypeLike | None = None,
) -> AffinePlan:
    if result_dtype is None:
        result_dtype = dtype
    records = []
    source_offset = 0
    for output_group in range(2):
        for scale in (1.25, -0.5):
            records.append(
                AffineRecord(
                    logical_shape=(output_count, reduction_count),
                    source_strides=(reduction_count, 1),
                    source_offset=source_offset,
                    destination_strides=(1, 0),
                    destination_offset=output_group * output_count,
                    scale=scale,
                    reduction_axes=(1,),
                )
            )
            source_offset += output_count * reduction_count
    return build_affine_plan(
        records=tuple(records),
        output_size=2 * output_count,
        coverage=CompleteMode.PARTIAL_UNIQUE_ZERO_FILL,
        source_size=source_offset,
        source_dtype=dtype,
        result_dtype=result_dtype,
        write_kind=StridedWriteKind.ACCUMULATE,
        reduction_kind=StridedReductionKind.SUM,
    )


def _many_small_exact_output_groups_plan(
    group_count: int,
    reduction_count: int,
) -> AffinePlan:
    records = []
    source_offset = 0
    for group in range(group_count):
        for scale in (1.0, -0.5):
            records.append(
                AffineRecord(
                    logical_shape=(1, reduction_count),
                    source_strides=(reduction_count, 1),
                    source_offset=source_offset,
                    destination_strides=(1, 0),
                    destination_offset=group,
                    scale=scale,
                    reduction_axes=(1,),
                )
            )
            source_offset += reduction_count
    return build_affine_plan(
        records=tuple(records),
        output_size=group_count,
        coverage=CompleteMode.PARTIAL_UNIQUE_ZERO_FILL,
        source_size=source_offset,
        source_dtype=jnp.float32,
        result_dtype=jnp.float32,
        write_kind=StridedWriteKind.ACCUMULATE,
        reduction_kind=StridedReductionKind.SUM,
    )


def _long_fiber_plan(
    *,
    output_count: int = 1,
    reduction_count: int = 1 << 18,
    dtype: DTypeLike = jnp.float32,
    scale: int | float | complex = 1,
) -> AffinePlan:
    return build_affine_plan(
        records=(
            AffineRecord(
                logical_shape=(output_count, reduction_count),
                source_strides=(reduction_count, 1),
                source_offset=0,
                destination_strides=(1, 0),
                destination_offset=0,
                scale=scale,
                reduction_axes=(1,),
            ),
        ),
        output_size=output_count,
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=output_count * reduction_count,
        source_dtype=dtype,
        result_dtype=dtype,
        reduction_kind=StridedReductionKind.SUM,
    )


def _multidimensional_long_fiber_plan(
    *,
    reduction_shape: tuple[int, ...] = (257, 263),
    dtype: DTypeLike = jnp.float32,
    scale: int | float | complex = 1,
) -> AffinePlan:
    source_strides: list[int] = []
    stride = 2
    for extent in reversed(reduction_shape):
        source_strides.append(stride)
        stride = stride * extent + 1
    source_strides.reverse()
    source_size = 1 + sum(
        (extent - 1) * axis_stride
        for extent, axis_stride in zip(
            reduction_shape, source_strides, strict=True
        )
    )
    return build_affine_plan(
        records=(
            AffineRecord(
                logical_shape=reduction_shape,
                source_strides=tuple(source_strides),
                source_offset=0,
                destination_strides=(0,) * len(reduction_shape),
                destination_offset=0,
                scale=scale,
                reduction_axes=tuple(range(len(reduction_shape))),
            ),
        ),
        output_size=1,
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=source_size,
        source_dtype=dtype,
        result_dtype=dtype,
        reduction_kind=StridedReductionKind.SUM,
    )


def _execute_serial_native_reduction(
    source: jax.Array,
    plan: AffinePlan,
) -> jax.Array:
    _set_native_reduction_fiber_parallel_mode_for_tests(0)
    try:
        result = _bind_sequential_native_structured_reduction(source, plan)
        result.block_until_ready()
        return result
    finally:
        _set_native_reduction_fiber_parallel_mode_for_tests(None)


def _signed_variable_fiber_group_plan() -> AffinePlan:
    return build_affine_plan(
        records=(
            AffineRecord(
                logical_shape=(4, 2),
                source_strides=(2, 1),
                source_offset=0,
                destination_strides=(-1, 0),
                destination_offset=3,
                scale=1.25,
                reduction_axes=(1,),
            ),
            AffineRecord(
                logical_shape=(4, 3),
                source_strides=(3, 1),
                source_offset=8,
                destination_strides=(-1, 0),
                destination_offset=3,
                scale=-0.5,
                reduction_axes=(1,),
            ),
        ),
        output_size=4,
        coverage=CompleteMode.PARTIAL_UNIQUE_ZERO_FILL,
        source_size=20,
        source_dtype=jnp.float32,
        result_dtype=jnp.float32,
        write_kind=StridedWriteKind.ACCUMULATE,
        reduction_kind=StridedReductionKind.SUM,
    )


def _disjoint_output_record_plan() -> AffinePlan:
    return build_affine_plan(
        records=(
            AffineRecord((1,), (1,), 0, (1,), 0),
            AffineRecord((1,), (1,), 1, (1,), 1),
        ),
        output_size=2,
        coverage=CompleteMode.PARTIAL_UNIQUE_ZERO_FILL,
        source_size=2,
        source_dtype=jnp.float32,
        result_dtype=jnp.float32,
        write_kind=StridedWriteKind.ACCUMULATE,
        reduction_kind=StridedReductionKind.SUM,
    )


def _forward_sum_reference(source: jax.Array, plan: AffinePlan) -> jax.Array:
    factor = plan.records[0].scale
    mapped = source if factor is None else jnp.multiply(factor, source)
    return jnp.sum(mapped.reshape(4, 3), axis=0, dtype=jnp.dtype(plan.result_dtype))


def _reference_forward(source: jax.Array, plan: AffinePlan) -> jax.Array:
    return execute_reference(source, plan)


@pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")
def test_single_record_empty_fibers_write_reduction_identity() -> None:
    plan = build_affine_plan(
        records=(
            AffineRecord(
                logical_shape=(3, 0),
                source_strides=(0, 1),
                source_offset=0,
                destination_strides=(1, 0),
                destination_offset=0,
                source_broadcast_axes=(0,),
                reduction_axes=(1,),
            ),
        ),
        output_size=3,
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=0,
        source_dtype=jnp.float32,
        result_dtype=jnp.float32,
        reduction_kind=StridedReductionKind.SUM,
    )

    actual = bind_native_structured_reduction(
        jnp.asarray([], dtype=jnp.float32),
        plan,
    )

    assert_bitwise_equal(actual, jnp.zeros((3,), dtype=jnp.float32))


@pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")
def test_single_record_partial_reduction_zero_fills_only_uncovered_outputs() -> None:
    plan = build_affine_plan(
        records=(
            AffineRecord(
                logical_shape=(3, 2),
                source_strides=(2, 1),
                source_offset=0,
                destination_strides=(1, 0),
                destination_offset=1,
                scale=-0.75,
                reduction_axes=(1,),
            ),
        ),
        output_size=5,
        coverage=CompleteMode.PARTIAL_UNIQUE_ZERO_FILL,
        source_size=6,
        source_dtype=jnp.float32,
        result_dtype=jnp.float32,
        reduction_kind=StridedReductionKind.SUM,
    )
    source = jnp.asarray([1.0, 2.0, -3.0, 4.0, 5.0, -6.0], dtype=jnp.float32)

    actual = bind_native_structured_reduction(source, plan)
    expected = jnp.asarray(
        [0.0, -2.25, -0.75, 0.75, 0.0],
        dtype=jnp.float32,
    )

    assert_bitwise_equal(actual, expected)


@pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")
@pytest.mark.parametrize(
    ("source_dtype", "result_dtype", "scale"),
    [
        (jnp.float16, jnp.float32, 1.25),
        (jnp.float32, jnp.complex64, 1.25 - 0.5j),
        (jnp.complex64, jnp.float32, -0.75),
        (jnp.float32, jnp.float32, -0.75),
        (jnp.complex64, jnp.complex64, 0.5 + 0.25j),
    ],
)
def test_forward_structured_sum_uses_the_same_plan_for_primal_jvp_and_vjp(
    source_dtype: DTypeLike,
    result_dtype: DTypeLike,
    scale: int | float | complex,
) -> None:
    plan = _forward_sum_plan(source_dtype, result_dtype, scale)
    source = _values(source_dtype, plan.source_size)
    tangent = _values(source_dtype, plan.source_size) * 0.25
    cotangent = _values(result_dtype, plan.output_size)
    native = lambda value: bind_native_structured_reduction(value, plan)
    reference = lambda value: _forward_sum_reference(value, plan)

    actual = jax.jit(native)(source)
    expected = reference(source)
    actual_primal, actual_tangent = jax.jvp(native, (source,), (tangent,))
    expected_primal, expected_tangent = jax.jvp(
        reference,
        (source,),
        (tangent,),
    )
    actual_vjp = jax.vjp(native, source)[1](cotangent)[0]
    expected_vjp = jax.vjp(reference, source)[1](cotangent)[0]

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-3)
    np.testing.assert_allclose(
        np.asarray(actual_primal),
        np.asarray(expected_primal),
        rtol=1e-3,
    )
    np.testing.assert_allclose(
        np.asarray(actual_tangent),
        np.asarray(expected_tangent),
        rtol=1e-3,
    )
    np.testing.assert_allclose(
        np.asarray(actual_vjp),
        np.asarray(expected_vjp),
        rtol=1e-3,
    )


@pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")
def test_grouped_sum_without_record_local_axes_is_closed_under_autodiff() -> None:
    plan = _grouped_sum_plan()
    source = jnp.asarray([1.25, -0.5], dtype=jnp.float32)
    tangent = jnp.asarray([-2.0, 4.0], dtype=jnp.float32)
    cotangent = jnp.asarray([3.0], dtype=jnp.float32)
    run = lambda value: bind_native_structured_reduction(value, plan)
    reference = lambda value: jnp.asarray(
        [2 * value[0] - 3 * value[1]],
        dtype=jnp.float32,
    )

    actual_primal, actual_tangent = jax.jvp(run, (source,), (tangent,))
    expected_primal, expected_tangent = jax.jvp(
        reference,
        (source,),
        (tangent,),
    )
    actual_pullback = jax.vjp(run, source)[1]
    expected_pullback = jax.vjp(reference, source)[1]
    actual_nested = jax.linear_transpose(
        actual_pullback,
        jnp.zeros_like(cotangent),
    )((source,))[0]
    expected_nested = jax.linear_transpose(
        expected_pullback,
        jnp.zeros_like(cotangent),
    )((source,))[0]

    np.testing.assert_array_equal(actual_primal, expected_primal)
    np.testing.assert_array_equal(actual_tangent, expected_tangent)
    np.testing.assert_array_equal(
        actual_pullback(cotangent)[0],
        expected_pullback(cotangent)[0],
    )
    np.testing.assert_array_equal(actual_nested, expected_nested)


@pytest.mark.parametrize(
    ("dtype", "scale"),
    [
        (jnp.bool_, True),
        (jnp.int8, -3),
        (jnp.int16, -3),
        (jnp.int32, -3),
        (jnp.uint8, 3),
        (jnp.uint16, 3),
        (jnp.uint32, 3),
    ],
)
@pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")
def test_forward_integer_structured_sum_matches_casted_jax_sum(
    dtype: DTypeLike,
    scale: int | bool,
) -> None:
    plan = _forward_sum_plan(dtype, dtype, scale)
    source = (
        jnp.arange(plan.source_size, dtype=jnp.int32) % 2 == 0
        if dtype == jnp.bool_
        else jnp.arange(plan.source_size, dtype=dtype)
    )
    expected = jnp.asarray(
        jnp.sum(
            jnp.asarray(scale, dtype=dtype) * source.reshape(4, 3),
            axis=0,
        ),
        dtype=dtype,
    )

    actual = jax.jit(
        lambda value: bind_native_structured_reduction(value, plan)
    )(source)

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")
def test_grouped_sum_supports_concurrent_executable_calls() -> None:
    plan = _grouped_sum_plan()
    executable = jax.jit(
        lambda value: bind_native_structured_reduction(value, plan)
    ).lower(jax.ShapeDtypeStruct((2,), jnp.float32)).compile()
    sources = tuple(
        jnp.asarray([shift + 0.25, 2 * shift - 0.5], dtype=jnp.float32)
        for shift in range(16)
    )

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = tuple(pool.map(executable, sources))
    for source, result in zip(sources, results, strict=True):
        result.block_until_ready()
        np.testing.assert_array_equal(
            result,
            jnp.asarray([2 * source[0] - 3 * source[1]], dtype=jnp.float32),
        )


@pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")
@pytest.mark.parametrize(
    ("source_dtype", "result_dtype"),
    (
        (jnp.float32, jnp.float32),
        (jnp.complex64, jnp.complex64),
        (jnp.float16, jnp.float32),
        (jnp.float32, jnp.complex64),
        (jnp.complex64, jnp.float32),
    ),
)
def test_exact_output_group_candidate_matches_sequential_and_transforms(
    source_dtype: DTypeLike,
    result_dtype: DTypeLike,
) -> None:
    plan = _exact_output_group_plan(
        dtype=source_dtype,
        result_dtype=result_dtype,
    )
    source = _values(source_dtype, plan.source_size)
    tangent = _values(source_dtype, plan.source_size) * 0.25
    grouped = lambda value: _bind_grouped_native_structured_reduction(
        value,
        plan,
    )
    sequential = lambda value: bind_native_structured_reduction(value, plan)

    actual = jax.jit(grouped)(source)
    expected = jax.jit(sequential)(source)
    actual_primal, actual_tangent = jax.jvp(grouped, (source,), (tangent,))
    expected_primal, expected_tangent = jax.jvp(
        sequential,
        (source,),
        (tangent,),
    )
    cotangent = _values(result_dtype, plan.output_size)
    actual_vjp = jax.vjp(grouped, source)[1](cotangent)[0]
    expected_vjp = jax.vjp(sequential, source)[1](cotangent)[0]
    actual_nested = jax.linear_transpose(
        jax.vjp(grouped, source)[1],
        jnp.zeros_like(cotangent),
    )((source,))[0]
    expected_nested = jax.linear_transpose(
        jax.vjp(sequential, source)[1],
        jnp.zeros_like(cotangent),
    )((source,))[0]

    assert_bitwise_equal(actual, expected)
    assert_bitwise_equal(actual_primal, expected_primal)
    assert_bitwise_equal(actual_tangent, expected_tangent)
    assert_bitwise_equal(actual_vjp, expected_vjp)
    assert_bitwise_equal(actual_nested, expected_nested)


@pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")
def test_exact_output_group_candidate_batches_and_runs_concurrently() -> None:
    plan = _exact_output_group_plan()
    sources = jnp.stack(
        tuple(
            _values(jnp.float32, plan.source_size) + shift
            for shift in range(4)
        )
    )
    run = jax.jit(
        jax.vmap(
            lambda value: _bind_grouped_native_structured_reduction(
                value,
                plan,
            )
        )
    )
    expected = jax.vmap(
        lambda value: bind_native_structured_reduction(value, plan)
    )(sources)

    executable = run.lower(sources).compile()
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = tuple(pool.map(executable, (sources,) * 8))
    for result in results:
        assert_bitwise_equal(result, expected)


@pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")
def test_exact_output_group_candidate_supports_signed_maps_and_varied_fibers() -> None:
    plan = _signed_variable_fiber_group_plan()
    source = _values(jnp.float32, plan.source_size)
    actual = _bind_grouped_native_structured_reduction(source, plan)
    expected = _execute_serial_native_reduction(source, plan)

    assert_bitwise_equal(actual, expected)


@pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")
@pytest.mark.parametrize("dtype", (jnp.float32, jnp.complex64))
def test_exact_output_group_candidate_preserves_finite_value_accuracy(
    dtype: DTypeLike,
) -> None:
    plan = _exact_output_group_plan(dtype=dtype)
    if dtype == jnp.float32:
        pattern = jnp.asarray(
            [0.0, -0.0, 1.25, -1.25, .125, 1.0],
            dtype=dtype,
        )
    else:
        pattern = jnp.asarray(
            [
                0 + 0j,
                complex(-0.0, 0.0),
                complex(1.25, 1.0),
                complex(-1.25, -2.0),
                complex(.125, 3.0),
                1.25 - 0.75j,
            ],
            dtype=dtype,
        )
    source = jnp.resize(pattern, (plan.source_size,))

    actual = _bind_grouped_native_structured_reduction(source, plan)
    expected = _execute_serial_native_reduction(source, plan)

    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-6)


@pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")
def test_exact_output_group_candidate_uses_multiple_output_owner_workers() -> None:
    plan = _exact_output_group_plan(output_count=4096, reduction_count=64)
    source = jnp.ones(plan.source_size, dtype=jnp.float32)

    _set_native_worker_limit_for_tests(4)
    try:
        executable = jax.jit(
            lambda value: _bind_grouped_native_structured_reduction(
                value,
                plan,
            )
        ).lower(source).compile()
        memory = executable.memory_analysis()
        actual = executable(source)
        actual.block_until_ready()
        workers, available = _native_worker_counts_for_tests()
    finally:
        _set_native_worker_limit_for_tests(None)

    assert memory is not None
    assert memory.temp_size_in_bytes == 0
    np.testing.assert_array_equal(
        actual,
        jnp.full(plan.output_size, 48, dtype=jnp.float32),
    )
    if available >= 2:
        assert workers >= 2


@pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")
def test_grouped_target_rejects_a_descriptor_without_overlap_proof() -> None:
    plan = _disjoint_output_record_plan()
    compiled = lower_structured_reduction(plan)
    source = jnp.ones(plan.source_size, dtype=jnp.float32)
    execute = jax.jit(
        lambda value: _structured_reduction_ffi_call(
            value,
            descriptor=compiled.descriptor,
            output_size=plan.output_size,
            output_dtype=jnp.dtype(plan.result_dtype),
            scalar_kind=plan.scalar_kind,
            grouped_output_owner=True,
        )
    )

    with pytest.raises(
        Exception,
        match="requires an exact overlapping output-map group",
    ):
        execute(source).block_until_ready()


@pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")
def test_grouped_target_rejects_noninterval_output_maps() -> None:
    plan = build_affine_plan(
        records=(
            AffineRecord(
                logical_shape=(2,),
                source_strides=(1,),
                source_offset=0,
                destination_strides=(2,),
                destination_offset=0,
            ),
            AffineRecord(
                logical_shape=(2,),
                source_strides=(1,),
                source_offset=2,
                destination_strides=(2,),
                destination_offset=0,
            ),
        ),
        output_size=3,
        coverage=CompleteMode.PARTIAL_UNIQUE_ZERO_FILL,
        source_size=4,
        source_dtype=jnp.float32,
        result_dtype=jnp.float32,
        write_kind=StridedWriteKind.ACCUMULATE,
        reduction_kind=StridedReductionKind.SUM,
    )
    compiled = lower_structured_reduction(plan)
    source = jnp.ones(plan.source_size, dtype=jnp.float32)
    execute = jax.jit(
        lambda value: _structured_reduction_ffi_call(
            value,
            descriptor=compiled.descriptor,
            output_size=plan.output_size,
            output_dtype=jnp.dtype(plan.result_dtype),
            scalar_kind=plan.scalar_kind,
            grouped_output_owner=True,
        )
    )

    with pytest.raises(Exception, match="not one physical interval"):
        execute(source).block_until_ready()


@pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")
def test_production_grouped_route_uses_only_static_work_features() -> None:
    small_plan = _exact_output_group_plan()
    large_plan = _exact_output_group_plan(
        output_count=4096,
        reduction_count=64,
    )
    small_source = jnp.ones(small_plan.source_size, dtype=jnp.float32)
    large_source = jnp.ones(large_plan.source_size, dtype=jnp.float32)

    bind_native_structured_reduction(small_source, small_plan).block_until_ready()
    assert not _native_grouped_output_owner_for_tests()
    bind_native_structured_reduction(large_source, large_plan).block_until_ready()
    assert _native_grouped_output_owner_for_tests()


@pytest.mark.parametrize(
    ("group_count", "reduction_count"),
    ((100, 512), (1_000, 64)),
)
def test_many_small_groups_do_not_enter_the_parallel_route(
    group_count: int,
    reduction_count: int,
) -> None:
    plan = _many_small_exact_output_groups_plan(
        group_count,
        reduction_count,
    )
    source = jnp.ones(plan.source_size, dtype=jnp.float32)
    bind_native_structured_reduction(source, plan).block_until_ready()
    assert not _native_grouped_output_owner_for_tests()


@pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")
def test_grouped_target_handles_one_thousand_small_groups() -> None:
    group_count = 1_000
    reduction_count = 2
    plan = _many_small_exact_output_groups_plan(
        group_count,
        reduction_count,
    )
    source = jnp.linspace(
        -1,
        1,
        plan.source_size,
        dtype=jnp.float32,
    )
    grouped = jax.jit(
        lambda value: _bind_grouped_native_structured_reduction(value, plan)
    )
    sequential = jax.jit(
        lambda value: _bind_sequential_native_structured_reduction(value, plan)
    )

    actual = grouped(source)
    expected = sequential(source)

    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    bind_native_structured_reduction(source, plan).block_until_ready()
    assert not _native_grouped_output_owner_for_tests()


@pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")
def test_vmap_rechecks_grouped_work_after_adding_the_batch_axis() -> None:
    plan = _exact_output_group_plan(
        output_count=2048,
        reduction_count=4,
    )
    source = jax.ShapeDtypeStruct((4, plan.source_size), jnp.float32)
    executable = jax.jit(
        jax.vmap(lambda value: bind_native_structured_reduction(value, plan))
    ).lower(source).compile()
    executable(jnp.ones(source.shape, dtype=source.dtype)).block_until_ready()
    assert _native_grouped_output_owner_for_tests()


def test_forward_reduction_descriptor_carries_explicit_scalar_policy() -> None:
    compiled = lower_structured_reduction(_forward_sum_plan())
    words = np.frombuffer(compiled.descriptor, dtype="<u8")

    assert words[1] == 4
    assert words[8] == 1


@pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")
def test_native_reduction_rank_limit_applies_after_normalization() -> None:
    logical_shape = (1,) * 8 + (2,)
    plan = build_affine_plan(
        records=(
            AffineRecord(
                logical_shape=logical_shape,
                source_strides=(2,) * 8 + (1,),
                source_offset=0,
                destination_strides=(0,) * len(logical_shape),
                destination_offset=0,
                reduction_axes=tuple(range(len(logical_shape))),
            ),
        ),
        output_size=1,
        coverage=CompleteMode.PARTIAL_UNIQUE_ZERO_FILL,
        source_size=2,
        source_dtype=jnp.float32,
        result_dtype=jnp.float32,
        reduction_kind=StridedReductionKind.SUM,
    )

    execution = lower_structured_reduction(plan)
    actual = bind_native_structured_reduction(
        jnp.asarray([1.25, -0.5], dtype=jnp.float32),
        plan,
    )

    assert execution.semantic is plan
    assert np.frombuffer(execution.descriptor, dtype="<u8")[9] == 9
    np.testing.assert_array_equal(actual, jnp.asarray([0.75], dtype=jnp.float32))


def test_native_reduction_rejects_rank_above_limit_after_normalization() -> None:
    logical_shape = (2,) * 9
    source_strides = tuple(3**axis for axis in range(len(logical_shape)))
    plan = build_affine_plan(
        records=(
            AffineRecord(
                logical_shape=logical_shape,
                source_strides=source_strides,
                source_offset=0,
                destination_strides=(0,) * len(logical_shape),
                destination_offset=0,
                reduction_axes=tuple(range(len(logical_shape))),
            ),
        ),
        output_size=1,
        coverage=CompleteMode.PARTIAL_UNIQUE_ZERO_FILL,
        source_size=sum(source_strides) + 1,
        source_dtype=jnp.float32,
        result_dtype=jnp.float32,
        reduction_kind=StridedReductionKind.SUM,
    )

    source = jnp.zeros(plan.source_size, dtype=jnp.float32)
    with pytest.raises(Exception, match="effective reduction rank exceeds native limit 8"):
        jax.jit(lambda value: bind_native_structured_reduction(value, plan))(
            source
        ).block_until_ready()


@pytest.mark.parametrize(
    ("broadcast_axes", "source_signs", "destination_fastest", "destination_signs"),
    (
        ((0,), (1,), (1, 0), (1, 1)),
        ((0,), (-1,), (0, 1), (-1, 1)),
        ((1,), (-1,), (1, 0), (1, -1)),
        ((0, 1), (), (0, 1), (-1, -1)),
    ),
)
@pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")
def test_native_structured_reduction_covers_signed_permuted_fibers(
    broadcast_axes: tuple[int, ...],
    source_signs: tuple[int, ...],
    destination_fastest: tuple[int, ...],
    destination_signs: tuple[int, ...],
) -> None:
    plan = _broadcast_plan(
        broadcast_axes=broadcast_axes,
        source_signs=source_signs,
        destination_fastest=destination_fastest,
        destination_signs=destination_signs,
    )
    source = _values(plan.source_dtype, plan.source_size)
    cotangent = _values(plan.result_dtype, plan.output_size)
    native_pullback = jax.vjp(lambda value: _native_forward(value, plan), source)[1]
    reference_pullback = jax.vjp(
        lambda value: _reference_forward(value, plan), source
    )[1]

    _reset_native_call_count_for_tests()
    actual = jax.jit(native_pullback)(cotangent)[0]
    actual.block_until_ready()
    expected = reference_pullback(cotangent)[0]

    assert _native_call_count_for_tests() == 1
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    hlo = str(
        jax.jit(native_pullback).lower(cotangent).compiler_ir("stablehlo")
    ).lower()
    assert "structured_reduction" in hlo
    assert "gather" not in hlo
    assert "scatter" not in hlo


@pytest.mark.parametrize(
    ("source_dtype", "result_dtype", "scale"),
    (
        (jnp.float16, jnp.float16, -1.25),
        (jnp.bfloat16, jnp.bfloat16, -1.25),
        (jnp.float32, jnp.float32, -1.25),
        (jnp.complex64, jnp.complex64, 1.25 - 0.75j),
        (jnp.float16, jnp.float32, -1.25),
        (jnp.float32, jnp.complex64, 0.5 + 0.75j),
        (jnp.complex64, jnp.float32, -0.75),
    ),
)
@pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")
def test_native_structured_reduction_covers_transpose_scalar_policies(
    source_dtype: DTypeLike,
    result_dtype: DTypeLike,
    scale: float | complex,
) -> None:
    plan = _broadcast_plan(
        source_signs=(-1,),
        destination_signs=(-1, 1),
        source_dtype=source_dtype,
        result_dtype=result_dtype,
        scale=scale,
    )
    source = _values(source_dtype, plan.source_size)
    cotangent = _values(result_dtype, plan.output_size)
    native_pullback = jax.vjp(lambda value: _native_forward(value, plan), source)[1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", np.exceptions.ComplexWarning)
        reference_pullback = jax.vjp(
            lambda value: _reference_forward(value, plan), source
        )[1]

    actual = jax.jit(native_pullback)(cotangent)[0]
    expected = reference_pullback(cotangent)[0]

    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))

    tangent = _values(result_dtype, plan.output_size) + jnp.asarray(
        0.25,
        dtype=result_dtype,
    )
    actual_jvp = jax.jvp(native_pullback, (cotangent,), (tangent,))
    expected_jvp = jax.jvp(reference_pullback, (cotangent,), (tangent,))
    for actual_part, expected_part in zip(
        actual_jvp,
        expected_jvp,
        strict=True,
    ):
        np.testing.assert_array_equal(actual_part, expected_part)

    source_cotangent = _values(source_dtype, plan.source_size)
    actual_nested = jax.linear_transpose(
        native_pullback,
        jnp.zeros(plan.output_size, dtype=result_dtype),
    )((source_cotangent,))[0]
    expected_nested = jax.linear_transpose(
        reference_pullback,
        jnp.zeros(plan.output_size, dtype=result_dtype),
    )((source_cotangent,))[0]
    np.testing.assert_array_equal(actual_nested, expected_nested)


@pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")
def test_native_structured_reduction_batch_vmap_jvp_and_nested_transpose() -> None:
    plan = _broadcast_plan(
        logical_shape=(3, 4),
        broadcast_axes=(0,),
        source_signs=(-1,),
        destination_fastest=(0, 1),
        destination_signs=(-1, 1),
        scale=-0.75,
    )
    source = _values(plan.source_dtype, plan.source_size)
    native_pullback = lambda cotangent: jax.vjp(
        lambda value: _native_forward(value, plan), source
    )[1](cotangent)[0]
    reference_pullback = lambda cotangent: jax.vjp(
        lambda value: _reference_forward(value, plan), source
    )[1](cotangent)[0]
    cotangents = jnp.stack(
        tuple(
            _values(plan.result_dtype, plan.output_size) + shift
            for shift in (-2, 0, 3)
        )
    )
    tangents = 0.25 * cotangents + 1

    actual_batch = jax.jit(jax.vmap(native_pullback))(cotangents)
    expected_batch = jax.jit(jax.vmap(reference_pullback))(cotangents)
    np.testing.assert_array_equal(actual_batch, expected_batch)

    actual_jvp = jax.jvp(native_pullback, (cotangents[0],), (tangents[0],))
    expected_jvp = jax.jvp(reference_pullback, (cotangents[0],), (tangents[0],))
    for actual, expected in zip(actual_jvp, expected_jvp, strict=True):
        np.testing.assert_array_equal(actual, expected)

    source_cotangent = jnp.linspace(
        -1,
        2,
        plan.source_size,
        dtype=jnp.float32,
    )
    actual_nested = jax.linear_transpose(
        native_pullback,
        jnp.zeros(plan.output_size, dtype=jnp.float32),
    )(source_cotangent)[0]
    expected_nested = jax.linear_transpose(
        reference_pullback,
        jnp.zeros(plan.output_size, dtype=jnp.float32),
    )(source_cotangent)[0]
    np.testing.assert_array_equal(actual_nested, expected_nested)


@pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")
def test_multi_record_reduction_accumulates_overlaps_in_record_order() -> None:
    records = (
        AffineRecord((2,), (0,), 0, (1,), 0, 1, (0,)),
        AffineRecord((2,), (0,), 0, (1,), 2, -0.5, (0,)),
        AffineRecord((1,), (1,), 1, (1,), 4, 2),
    )
    plan = build_affine_plan(
        records=records,
        output_size=5,
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=2,
        source_dtype=jnp.float32,
        result_dtype=jnp.float32,
    )
    source = jnp.asarray([2.0, -3.0], dtype=jnp.float32)
    cotangent = jnp.asarray([1.0, -2.0, 3.0, 4.0, -5.0], dtype=jnp.float32)

    actual = jax.vjp(lambda value: _native_forward(value, plan), source)[1](
        cotangent
    )[0]
    expected = jax.vjp(lambda value: _reference_forward(value, plan), source)[1](
        cotangent
    )[0]

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual, jnp.asarray([-4.5, -10.0]))


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.complex64])
@pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")
def test_native_structured_reduction_finite_values_match_reference(
    dtype: DTypeLike,
) -> None:
    scale = 3.5 if dtype == jnp.float32 else complex(3.5, 3.5)
    plan = _broadcast_plan(
        logical_shape=(4, 3),
        broadcast_axes=(0,),
        destination_fastest=(1, 0),
        destination_signs=(-1, 1),
        source_dtype=dtype,
        result_dtype=dtype,
        scale=scale,
    )
    source = _values(dtype, plan.source_size)
    if dtype == jnp.float32:
        pattern = jnp.asarray(
            [0.0, -0.0, 1.25, -1.25, .125, 1.0],
            dtype=dtype,
        )
    else:
        pattern = jnp.asarray(
            [
                0 + 0j,
                complex(-0.0, 0.0),
                complex(1.25, 1.0),
                complex(-1.25, -2.0),
                complex(.125, 3.0),
                1.25 - 0.75j,
            ],
            dtype=dtype,
        )
    cotangent = jnp.resize(pattern, (plan.output_size,))

    actual = jax.vjp(lambda value: _native_forward(value, plan), source)[1](
        cotangent
    )[0]
    expected = jax.vjp(lambda value: _reference_forward(value, plan), source)[1](
        cotangent
    )[0]

    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-6)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.complex64])
@pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")
def test_rank_one_output_block_preserves_order_and_handles_tail(
    dtype: DTypeLike,
) -> None:
    output_count = 11
    reduction_count = 3
    scale = -0.75 if dtype == jnp.float32 else 1.25 - 0.5j
    plan = build_affine_plan(
        records=(
            AffineRecord(
                logical_shape=(output_count, reduction_count),
                source_strides=(-reduction_count, 1),
                source_offset=(output_count - 1) * reduction_count,
                destination_strides=(-1, 0),
                destination_offset=output_count - 1,
                scale=scale,
                reduction_axes=(1,),
            ),
        ),
        output_size=output_count,
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=output_count * reduction_count,
        source_dtype=dtype,
        result_dtype=dtype,
        reduction_kind=StridedReductionKind.SUM,
    )
    if dtype == jnp.float32:
        pattern = jnp.asarray(
            [0.0, -0.0, jnp.inf, -jnp.inf, jnp.nan, 1.0],
            dtype=dtype,
        )
    else:
        pattern = jnp.asarray(
            [
                0 + 0j,
                complex(-0.0, 0.0),
                complex(jnp.inf, 1.0),
                complex(-jnp.inf, -2.0),
                complex(jnp.nan, 3.0),
                1.25 - 0.75j,
            ],
            dtype=dtype,
        )
    source = jnp.resize(pattern, (plan.source_size,))
    scalar_plan = build_affine_plan(
        records=tuple(
            AffineRecord(
                logical_shape=(1, reduction_count),
                source_strides=(0, 1),
                source_offset=(output_count - 1 - output) * reduction_count,
                destination_strides=(0, 0),
                destination_offset=output_count - 1 - output,
                scale=scale,
                reduction_axes=(1,),
            )
            for output in range(output_count)
        ),
        output_size=output_count,
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=output_count * reduction_count,
        source_dtype=dtype,
        result_dtype=dtype,
        write_kind=StridedWriteKind.ACCUMULATE,
        reduction_kind=StridedReductionKind.SUM,
    )

    actual = bind_native_structured_reduction(source, plan)
    expected = bind_native_structured_reduction(source, scalar_plan)

    assert_bitwise_equal(actual, expected)


@pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")
def test_large_native_structured_reduction_uses_output_owner_workers() -> None:
    plan = _broadcast_plan(
        logical_shape=(4096, 512),
        broadcast_axes=(1,),
        source_signs=(1,),
        destination_fastest=(1, 0),
        scale=1,
    )
    source = jnp.zeros(plan.source_size, dtype=jnp.float32)
    cotangent = jnp.ones(plan.output_size, dtype=jnp.float32)
    pullback = jax.vjp(lambda value: _native_forward(value, plan), source)[1]

    _set_native_worker_limit_for_tests(4)
    try:
        executable = jax.jit(pullback).lower(cotangent).compile()
        memory = executable.memory_analysis()
        actual = executable(cotangent)[0]
        actual.block_until_ready()
        workers, available = _native_worker_counts_for_tests()
    finally:
        _set_native_worker_limit_for_tests(None)

    assert memory is not None
    assert memory.argument_size_in_bytes == cotangent.nbytes
    assert memory.output_size_in_bytes == source.nbytes
    assert memory.temp_size_in_bytes == 0
    np.testing.assert_array_equal(
        np.asarray(actual),
        np.full(plan.source_size, 512, dtype=np.float32),
    )
    if available >= 2:
        assert workers >= 2


@pytest.mark.parametrize(
    ("dtype", "scale"),
    ((jnp.float32, -0.75), (jnp.complex64, 1.25 - 0.5j)),
)
@pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")
def test_forced_long_fiber_reduction_uses_deterministic_chunk_partials(
    dtype: DTypeLike,
    scale: float | complex,
) -> None:
    plan = _long_fiber_plan(dtype=dtype, scale=scale)
    source = _values(dtype, plan.source_size)
    execute = jax.jit(
        lambda value: bind_native_structured_reduction(value, plan)
    )

    _set_native_worker_limit_for_tests(4)
    _set_native_reduction_fiber_parallel_mode_for_tests(1)
    try:
        results = tuple(execute(source) for _ in range(3))
        jax.block_until_ready(results)
        workers, available = _native_worker_counts_for_tests()
        chunks = _native_reduction_fiber_chunks_for_tests()
    finally:
        _set_native_reduction_fiber_parallel_mode_for_tests(None)
        _set_native_worker_limit_for_tests(None)

    expected = _execute_serial_native_reduction(source, plan)
    for result in results[1:]:
        assert_bitwise_equal(result, results[0])
    np.testing.assert_allclose(
        np.asarray(results[0]),
        np.asarray(expected),
        rtol=2e-5,
        atol=2e-5,
        equal_nan=True,
    )
    if available >= 2:
        assert workers >= 2
        assert chunks >= 2


@pytest.mark.parametrize(
    ("dtype", "scale"),
    ((jnp.float32, -0.75), (jnp.complex64, 1.25 - 0.5j)),
)
@pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")
def test_forced_multidimensional_fiber_uses_shared_range_walker(
    dtype: DTypeLike,
    scale: float | complex,
) -> None:
    plan = _multidimensional_long_fiber_plan(dtype=dtype, scale=scale)
    source = _values(dtype, plan.source_size)
    execute = jax.jit(
        lambda value: bind_native_structured_reduction(value, plan)
    )

    _set_native_worker_limit_for_tests(4)
    _set_native_reduction_fiber_parallel_mode_for_tests(1)
    try:
        results = tuple(execute(source) for _ in range(3))
        jax.block_until_ready(results)
        _, available = _native_worker_counts_for_tests()
        chunks = _native_reduction_fiber_chunks_for_tests()
    finally:
        _set_native_reduction_fiber_parallel_mode_for_tests(None)
        _set_native_worker_limit_for_tests(None)

    expected = _execute_serial_native_reduction(source, plan)
    for result in results[1:]:
        assert_bitwise_equal(result, results[0])
    np.testing.assert_allclose(
        np.asarray(results[0]),
        np.asarray(expected),
        rtol=2e-5,
        atol=2e-5,
        equal_nan=True,
    )
    if available >= 2:
        assert chunks >= 2


@pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")
def test_forced_multidimensional_fiber_preserves_jvp_and_vjp_semantics() -> None:
    plan = _multidimensional_long_fiber_plan(scale=-0.75)
    source = _values(jnp.float32, plan.source_size)
    tangent = source * 0.25 - 1
    cotangent = jnp.asarray([1.25], dtype=jnp.float32)
    execute = lambda value: bind_native_structured_reduction(value, plan)

    _set_native_worker_limit_for_tests(4)
    _set_native_reduction_fiber_parallel_mode_for_tests(1)
    try:
        primal, actual_tangent = jax.jit(
            lambda value, direction: jax.jvp(
                execute, (value,), (direction,)
            )
        )(source, tangent)
        actual_vjp = jax.jit(
            lambda value, ct: jax.vjp(execute, value)[1](ct)[0]
        )(source, cotangent)
        jax.block_until_ready((primal, actual_tangent, actual_vjp))
        chunks = _native_reduction_fiber_chunks_for_tests()
        _set_native_worker_limit_for_tests(1)
        _set_native_reduction_fiber_parallel_mode_for_tests(0)
        reference = lambda value: _bind_sequential_native_structured_reduction(
            value, plan
        )
        expected_primal, expected_tangent = jax.jvp(
            reference, (source,), (tangent,)
        )
        expected_vjp = jax.vjp(reference, source)[1](cotangent)[0]
        jax.block_until_ready(
            (expected_primal, expected_tangent, expected_vjp)
        )
    finally:
        _set_native_reduction_fiber_parallel_mode_for_tests(None)
        _set_native_worker_limit_for_tests(None)

    assert chunks >= 2
    np.testing.assert_allclose(primal, expected_primal, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(
        actual_tangent, expected_tangent, rtol=2e-5, atol=2e-5
    )
    assert_bitwise_equal(actual_vjp, expected_vjp)


@pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")
def test_production_multidimensional_fiber_uses_recursive_partials() -> None:
    plan = _multidimensional_long_fiber_plan(reduction_shape=(512, 512))
    source = _values(jnp.float32, plan.source_size)

    _set_native_worker_limit_for_tests(4)
    _set_native_reduction_fiber_parallel_mode_for_tests(None)
    try:
        actual = jax.jit(
            lambda value: bind_native_structured_reduction(value, plan)
        )(source)
        actual.block_until_ready()
        chunks = _native_reduction_fiber_chunks_for_tests()
    finally:
        _set_native_reduction_fiber_parallel_mode_for_tests(None)
        _set_native_worker_limit_for_tests(None)

    expected = _execute_serial_native_reduction(source, plan)
    assert chunks >= 2
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


@pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")
def test_production_multirecord_scalar_reduction_preserves_record_order() -> None:
    reduction_count = 65_536
    plan = build_affine_plan(
        records=(
            AffineRecord(
                logical_shape=(reduction_count,),
                source_strides=(1,),
                source_offset=0,
                destination_strides=(0,),
                destination_offset=0,
                scale=0.75,
                reduction_axes=(0,),
            ),
            AffineRecord(
                logical_shape=(reduction_count,),
                source_strides=(-1,),
                source_offset=2 * reduction_count - 1,
                destination_strides=(0,),
                destination_offset=0,
                scale=-0.25,
                reduction_axes=(0,),
            ),
        ),
        output_size=1,
        coverage=CompleteMode.PARTIAL_UNIQUE_ZERO_FILL,
        source_size=2 * reduction_count,
        source_dtype=jnp.float32,
        result_dtype=jnp.float32,
        write_kind=StridedWriteKind.ACCUMULATE,
        reduction_kind=StridedReductionKind.SUM,
    )
    source = _values(jnp.float32, plan.source_size)

    _set_native_worker_limit_for_tests(4)
    _set_native_reduction_fiber_parallel_mode_for_tests(None)
    try:
        actual = jax.jit(
            lambda value: bind_native_structured_reduction(value, plan)
        )(source)
        actual.block_until_ready()
        chunks = _native_reduction_fiber_chunks_for_tests()
    finally:
        _set_native_reduction_fiber_parallel_mode_for_tests(None)
        _set_native_worker_limit_for_tests(None)

    expected = _execute_serial_native_reduction(source, plan)
    assert chunks >= 2
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


@pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")
def test_long_fiber_reduction_keeps_serial_behavior_with_one_worker() -> None:
    plan = _long_fiber_plan(reduction_count=65_536)
    source = _values(jnp.float32, plan.source_size)

    _set_native_worker_limit_for_tests(1)
    _set_native_reduction_fiber_parallel_mode_for_tests(1)
    try:
        actual = bind_native_structured_reduction(source, plan)
        actual.block_until_ready()
        chunks = _native_reduction_fiber_chunks_for_tests()
        _set_native_reduction_fiber_parallel_mode_for_tests(0)
        expected = _bind_sequential_native_structured_reduction(source, plan)
        expected.block_until_ready()
    finally:
        _set_native_reduction_fiber_parallel_mode_for_tests(None)
        _set_native_worker_limit_for_tests(None)

    assert chunks == 0
    assert_bitwise_equal(actual, expected)


@pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")
def test_batched_outputs_keep_output_owner_parallelism() -> None:
    plan = _long_fiber_plan(reduction_count=65_536)
    source = jnp.stack(
        tuple(
            _values(jnp.float32, plan.source_size) + batch
            for batch in range(4)
        )
    )
    execute = jax.jit(
        jax.vmap(lambda value: bind_native_structured_reduction(value, plan))
    )

    _set_native_worker_limit_for_tests(4)
    _set_native_reduction_fiber_parallel_mode_for_tests(1)
    try:
        actual = execute(source)
        actual.block_until_ready()
        workers, available = _native_worker_counts_for_tests()
        chunks = _native_reduction_fiber_chunks_for_tests()
    finally:
        _set_native_reduction_fiber_parallel_mode_for_tests(None)
        _set_native_worker_limit_for_tests(None)

    _set_native_reduction_fiber_parallel_mode_for_tests(0)
    try:
        expected = jax.vmap(
            lambda value: _bind_sequential_native_structured_reduction(
                value, plan
            )
        )(source)
        expected.block_until_ready()
    finally:
        _set_native_reduction_fiber_parallel_mode_for_tests(None)
    assert chunks == 0
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
    if available >= 2:
        assert workers >= 2


@pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")
def test_forced_long_fiber_reduction_preserves_jvp_and_vjp_semantics() -> None:
    plan = _long_fiber_plan(reduction_count=65_536, scale=-0.75)
    source = _values(jnp.float32, plan.source_size)
    tangent = source * 0.25 - 1
    cotangent = jnp.asarray([1.25], dtype=jnp.float32)
    execute = lambda value: bind_native_structured_reduction(value, plan)

    _set_native_worker_limit_for_tests(4)
    _set_native_reduction_fiber_parallel_mode_for_tests(1)
    try:
        primal, actual_tangent = jax.jit(
            lambda value, direction: jax.jvp(
                execute, (value,), (direction,)
            )
        )(source, tangent)
        actual_vjp = jax.jit(
            lambda value, ct: jax.vjp(execute, value)[1](ct)[0]
        )(source, cotangent)
        jax.block_until_ready((primal, actual_tangent, actual_vjp))
        _set_native_worker_limit_for_tests(1)
        _set_native_reduction_fiber_parallel_mode_for_tests(0)
        reference = lambda value: _bind_sequential_native_structured_reduction(
            value, plan
        )
        expected_primal, expected_tangent = jax.jvp(
            reference, (source,), (tangent,)
        )
        expected_vjp = jax.vjp(reference, source)[1](cotangent)[0]
        jax.block_until_ready(
            (expected_primal, expected_tangent, expected_vjp)
        )
    finally:
        _set_native_reduction_fiber_parallel_mode_for_tests(None)
        _set_native_worker_limit_for_tests(None)

    np.testing.assert_allclose(primal, expected_primal, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(
        actual_tangent, expected_tangent, rtol=2e-5, atol=2e-5
    )
    assert_bitwise_equal(actual_vjp, expected_vjp)


@pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")
def test_production_long_fiber_threshold_selects_only_underused_workers() -> None:
    long_plan = _long_fiber_plan(reduction_count=65_536)
    small_plan = _long_fiber_plan(reduction_count=4_096)
    two_outputs_plan = _long_fiber_plan(
        output_count=2,
        reduction_count=65_536,
    )
    three_outputs_plan = _long_fiber_plan(
        output_count=3,
        reduction_count=65_536,
    )
    enough_outputs_plan = _long_fiber_plan(
        output_count=4,
        reduction_count=65_536,
    )
    long_source = _values(jnp.float32, long_plan.source_size)
    small_source = _values(jnp.float32, small_plan.source_size)
    two_outputs_source = _values(
        jnp.float32, two_outputs_plan.source_size
    )
    three_outputs_source = _values(
        jnp.float32, three_outputs_plan.source_size
    )
    enough_outputs_source = _values(
        jnp.float32, enough_outputs_plan.source_size
    )

    _set_native_worker_limit_for_tests(4)
    _set_native_reduction_fiber_parallel_mode_for_tests(None)
    try:
        long_result = jax.jit(
            lambda value: bind_native_structured_reduction(value, long_plan)
        )(long_source)
        long_result.block_until_ready()
        long_chunks = _native_reduction_fiber_chunks_for_tests()
        small_result = jax.jit(
            lambda value: bind_native_structured_reduction(value, small_plan)
        )(small_source)
        small_result.block_until_ready()
        small_chunks = _native_reduction_fiber_chunks_for_tests()
        two_outputs_result = jax.jit(
            lambda value: bind_native_structured_reduction(
                value, two_outputs_plan
            )
        )(two_outputs_source)
        two_outputs_result.block_until_ready()
        two_outputs_chunks = _native_reduction_fiber_chunks_for_tests()
        three_outputs_result = jax.jit(
            lambda value: bind_native_structured_reduction(
                value, three_outputs_plan
            )
        )(three_outputs_source)
        three_outputs_result.block_until_ready()
        three_outputs_chunks = _native_reduction_fiber_chunks_for_tests()
        enough_outputs_result = jax.jit(
            lambda value: bind_native_structured_reduction(
                value, enough_outputs_plan
            )
        )(enough_outputs_source)
        enough_outputs_result.block_until_ready()
        enough_outputs_chunks = _native_reduction_fiber_chunks_for_tests()
    finally:
        _set_native_reduction_fiber_parallel_mode_for_tests(None)
        _set_native_worker_limit_for_tests(None)

    assert long_chunks >= 2
    assert small_chunks == 0
    assert two_outputs_chunks == 0
    assert three_outputs_chunks == 0
    assert enough_outputs_chunks == 0
    np.testing.assert_allclose(
        long_result,
        _execute_serial_native_reduction(long_source, long_plan),
        rtol=2e-5,
        atol=2e-5,
    )
    assert_bitwise_equal(
        small_result,
        _execute_serial_native_reduction(small_source, small_plan),
    )
    np.testing.assert_allclose(
        two_outputs_result,
        _execute_serial_native_reduction(
            two_outputs_source, two_outputs_plan
        ),
        rtol=2e-5,
        atol=2e-5,
    )
    np.testing.assert_allclose(
        three_outputs_result,
        _execute_serial_native_reduction(
            three_outputs_source, three_outputs_plan
        ),
        rtol=2e-5,
        atol=2e-5,
    )


@pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")
def test_integer_long_fiber_reduction_retains_exact_serial_order() -> None:
    plan = _long_fiber_plan(reduction_count=65_536, dtype=jnp.int32)
    source = jnp.arange(plan.source_size, dtype=jnp.int32) % 13

    _set_native_worker_limit_for_tests(4)
    _set_native_reduction_fiber_parallel_mode_for_tests(None)
    try:
        actual = jax.jit(
            lambda value: bind_native_structured_reduction(value, plan)
        )(source)
        actual.block_until_ready()
        chunks = _native_reduction_fiber_chunks_for_tests()
    finally:
        _set_native_reduction_fiber_parallel_mode_for_tests(None)
        _set_native_worker_limit_for_tests(None)

    expected = _execute_serial_native_reduction(source, plan)
    assert chunks == 0
    assert_bitwise_equal(actual, expected)


@pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")
def test_forced_long_fiber_reduction_zero_fills_uncovered_output() -> None:
    reduction_count = 65_536
    plan = build_affine_plan(
        records=(
            AffineRecord(
                logical_shape=(1, reduction_count),
                source_strides=(reduction_count, 1),
                source_offset=0,
                destination_strides=(1, 0),
                destination_offset=1,
                reduction_axes=(1,),
            ),
        ),
        output_size=2,
        coverage=CompleteMode.PARTIAL_UNIQUE_ZERO_FILL,
        source_size=reduction_count,
        source_dtype=jnp.float32,
        result_dtype=jnp.float32,
        reduction_kind=StridedReductionKind.SUM,
    )
    source = jnp.ones(plan.source_size, dtype=jnp.float32)

    _set_native_worker_limit_for_tests(4)
    _set_native_reduction_fiber_parallel_mode_for_tests(1)
    try:
        actual = bind_native_structured_reduction(source, plan)
        actual.block_until_ready()
    finally:
        _set_native_reduction_fiber_parallel_mode_for_tests(None)
        _set_native_worker_limit_for_tests(None)

    np.testing.assert_array_equal(
        np.asarray(actual),
        np.asarray([0, reduction_count], dtype=np.float32),
    )


@pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")
def test_forced_long_fiber_reduction_propagates_nonfinite_values() -> None:
    plan = _long_fiber_plan(reduction_count=65_536)
    finite = jnp.ones(plan.source_size, dtype=jnp.float32)
    with_nan = finite.at[17].set(jnp.nan)
    with_inf = finite.at[31].set(jnp.inf)

    _set_native_worker_limit_for_tests(4)
    _set_native_reduction_fiber_parallel_mode_for_tests(1)
    try:
        nan_result = bind_native_structured_reduction(with_nan, plan)
        inf_result = bind_native_structured_reduction(with_inf, plan)
        jax.block_until_ready((nan_result, inf_result))
    finally:
        _set_native_reduction_fiber_parallel_mode_for_tests(None)
        _set_native_worker_limit_for_tests(None)

    assert bool(jnp.isnan(nan_result[0]))
    assert bool(jnp.isposinf(inf_result[0]))


@pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")
def test_forced_long_fiber_nested_transpose_is_numerically_correct() -> None:
    plan = _long_fiber_plan(reduction_count=65_536, scale=-0.75)
    source = _values(jnp.float32, plan.source_size)
    cotangent = jnp.zeros((1,), dtype=jnp.float32)
    tangent = jnp.ones_like(source)
    execute = lambda value: bind_native_structured_reduction(value, plan)
    pullback = lambda ct: jax.vjp(execute, source)[1](ct)[0]

    _set_native_worker_limit_for_tests(4)
    _set_native_reduction_fiber_parallel_mode_for_tests(1)
    try:
        actual = jax.jit(
            lambda ct, direction: jax.linear_transpose(
                pullback, ct
            )(direction)[0]
        )(cotangent, tangent)
        actual.block_until_ready()
    finally:
        _set_native_reduction_fiber_parallel_mode_for_tests(None)
        _set_native_worker_limit_for_tests(None)

    expected = _execute_serial_native_reduction(tangent, plan)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


@pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")
def test_forced_long_fiber_reduction_uses_per_call_scratch() -> None:
    plan = _long_fiber_plan(reduction_count=65_536)
    source = _values(jnp.float32, plan.source_size)
    execute = jax.jit(
        lambda value: bind_native_structured_reduction(value, plan)
    )

    _set_native_worker_limit_for_tests(4)
    _set_native_reduction_fiber_parallel_mode_for_tests(1)
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = tuple(
                pool.map(
                    lambda shift: execute(source + shift),
                    range(8),
                )
            )
        jax.block_until_ready(results)
    finally:
        _set_native_reduction_fiber_parallel_mode_for_tests(None)
        _set_native_worker_limit_for_tests(None)

    for shift, result in enumerate(results):
        expected = _execute_serial_native_reduction(
            source + shift, plan
        )
        np.testing.assert_allclose(result, expected, rtol=2e-5, atol=2e-5)


def test_structured_reduction_non_cpu_lowering_fails_closed() -> None:
    plan = _broadcast_plan()
    source = jax.ShapeDtypeStruct((plan.source_size,), jnp.float32)
    cotangent = jax.ShapeDtypeStruct((plan.output_size,), jnp.float32)
    run = jax.jit(
        lambda value, ct: jax.vjp(
            lambda item: _native_forward(item, plan),
            value,
        )[1](ct)[0]
    )

    with pytest.raises(RuntimeError, match="native_structured_reduction_non_cpu"):
        run.trace(source, cotangent).lower(lowering_platforms=("tpu",))


@pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")
def test_native_reduction_rejects_mutated_descriptor() -> None:
    plan = _broadcast_plan()
    compiled = lower_broadcast_transpose(plan)
    descriptor = bytearray(compiled.descriptor)
    descriptor[0] ^= 0xFF
    cotangent = jnp.ones(plan.output_size, dtype=jnp.float32)
    execute = jax.jit(
        lambda value: _structured_reduction_ffi_call(
            value,
            descriptor=bytes(descriptor),
            output_size=plan.source_size,
            output_dtype=jnp.dtype(plan.source_dtype),
        )
    )

    with pytest.raises(Exception, match="reduction descriptor header mismatch"):
        execute(cotangent).block_until_ready()


@pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")
def test_native_reduction_rejects_typed_target_mismatch() -> None:
    plan = _broadcast_plan()
    compiled = lower_broadcast_transpose(plan)
    cotangent = jnp.ones(plan.output_size, dtype=jnp.float32)
    execute = jax.jit(
        lambda value: _structured_reduction_ffi_call(
            value,
            descriptor=compiled.descriptor,
            output_size=plan.source_size,
            output_dtype=jnp.dtype(jnp.complex64),
        )
    )

    with pytest.raises(Exception, match="dtype does not match typed target"):
        execute(cotangent).block_until_ready()


def test_all_rank_two_native_reduction_descriptors_compile() -> None:
    for broadcast_axes in ((0,), (1,), (0, 1)):
        map_count = 2 - len(broadcast_axes)
        for source_signs in product((-1, 1), repeat=map_count):
            for destination_fastest in permutations(range(2)):
                for destination_signs in product((-1, 1), repeat=2):
                    plan = _broadcast_plan(
                        broadcast_axes=broadcast_axes,
                        source_signs=source_signs,
                        destination_fastest=destination_fastest,
                        destination_signs=destination_signs,
                    )
                    compiled = lower_broadcast_transpose(plan)
                    assert compiled.semantic.records[0].reduction_axes == broadcast_axes
