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

from tensor0._stride import (
    CompleteMode,
    StridedCopyPlan,
    StridedCopyRecord,
    StridedReductionKind,
    StridedWriteKind,
    strided_copy,
)
from tensor0._stride._ffi import (
    _native_call_count_for_tests,
    _native_worker_counts_for_tests,
    _reset_native_call_count_for_tests,
    _set_native_worker_limit_for_tests,
    native_available,
    _structured_reduction_ffi_call_v2,
)
from tensor0._stride._native_reduction import (
    bind_native_structured_reduction,
    compile_native_broadcast_transpose,
    compile_native_structured_reduction,
)
from tensor0._stride._plan import (
    build_strided_copy_plan,
)

from ._oracle import assert_bitwise_equal, execute_reference


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
) -> StridedCopyPlan:
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
    record = StridedCopyRecord(
        logical_shape=logical_shape,
        source_strides=tuple(source_strides),
        source_offset=source_offset,
        destination_strides=destination_strides,
        destination_offset=destination_offset,
        scale=scale,
        source_broadcast_axes=broadcast_axes,
    )
    return build_strided_copy_plan(
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


def _native_forward(source: jax.Array, plan: StridedCopyPlan) -> jax.Array:
    return strided_copy(source, plan=plan, native_required=True)


def _forward_sum_plan(
    source_dtype: DTypeLike = jnp.float32,
    result_dtype: DTypeLike = jnp.float32,
    scale: int | float | complex = -0.75,
) -> StridedCopyPlan:
    return build_strided_copy_plan(
        records=(
            StridedCopyRecord(
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


def _grouped_sum_plan() -> StridedCopyPlan:
    return build_strided_copy_plan(
        records=(
            StridedCopyRecord((1,), (1,), 0, (1,), 0, 2),
            StridedCopyRecord((1,), (1,), 1, (1,), 0, -3),
        ),
        output_size=1,
        coverage=CompleteMode.PARTIAL_UNIQUE_ZERO_FILL,
        source_size=2,
        source_dtype=jnp.float32,
        result_dtype=jnp.float32,
        write_kind=StridedWriteKind.ACCUMULATE,
        reduction_kind=StridedReductionKind.SUM,
    )


def _forward_sum_reference(source: jax.Array, plan: StridedCopyPlan) -> jax.Array:
    result_dtype = jnp.dtype(plan.result_dtype)
    converted = jnp.asarray(
        jnp.real(source)
        if jnp.issubdtype(source.dtype, jnp.complexfloating)
        and not jnp.issubdtype(result_dtype, jnp.complexfloating)
        else source,
        dtype=result_dtype,
    )
    scale = jnp.asarray(plan.records[0].scale, dtype=converted.dtype)
    return jnp.sum(scale * converted.reshape(4, 3), axis=0)


def _reference_forward(source: jax.Array, plan: StridedCopyPlan) -> jax.Array:
    return execute_reference(source, plan)


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


def test_forward_reduction_descriptor_carries_explicit_scalar_policy() -> None:
    compiled = compile_native_structured_reduction(_forward_sum_plan())
    words = np.frombuffer(compiled.descriptor, dtype="<u8")

    assert words[1] == 2
    assert words[3] == 3
    assert words[10] == 1


@pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")
def test_native_reduction_rank_limit_applies_after_normalization() -> None:
    logical_shape = (1,) * 8 + (2,)
    plan = build_strided_copy_plan(
        records=(
            StridedCopyRecord(
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

    execution = compile_native_structured_reduction(plan)
    actual = bind_native_structured_reduction(
        jnp.asarray([1.25, -0.5], dtype=jnp.float32),
        plan,
    )

    assert execution.compiled.records[0].rank == 1
    assert np.frombuffer(execution.descriptor, dtype="<u8")[13] == 1
    np.testing.assert_array_equal(actual, jnp.asarray([0.75], dtype=jnp.float32))


def test_native_reduction_rejects_rank_above_limit_after_normalization() -> None:
    logical_shape = (2,) * 9
    source_strides = tuple(3**axis for axis in range(len(logical_shape)))
    plan = build_strided_copy_plan(
        records=(
            StridedCopyRecord(
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

    with pytest.raises(ValueError, match="rank exceeds native reduction limit 8"):
        compile_native_structured_reduction(plan)


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
        StridedCopyRecord((2,), (0,), 0, (1,), 0, 1, (0,)),
        StridedCopyRecord((2,), (0,), 0, (1,), 2, -0.5, (0,)),
        StridedCopyRecord((1,), (1,), 1, (1,), 4, 2),
    )
    plan = build_strided_copy_plan(
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
def test_native_structured_reduction_special_values_match_reference_bits(
    dtype: DTypeLike,
) -> None:
    scale = 3e38 if dtype == jnp.float32 else complex(3e38, 3e38)
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
    cotangent = jnp.resize(pattern, (plan.output_size,))

    actual = jax.vjp(lambda value: _native_forward(value, plan), source)[1](
        cotangent
    )[0]
    expected = jax.vjp(lambda value: _reference_forward(value, plan), source)[1](
        cotangent
    )[0]

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
    compiled = compile_native_broadcast_transpose(plan)
    descriptor = bytearray(compiled.descriptor)
    descriptor[0] ^= 0xFF
    cotangent = jnp.ones(plan.output_size, dtype=jnp.float32)
    execute = jax.jit(
        lambda value: _structured_reduction_ffi_call_v2(
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
    compiled = compile_native_broadcast_transpose(plan)
    cotangent = jnp.ones(plan.output_size, dtype=jnp.float32)
    execute = jax.jit(
        lambda value: _structured_reduction_ffi_call_v2(
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
                    compiled = compile_native_broadcast_transpose(plan)
                    record = compiled.compiled.records[0]
                    original_reduction_axes = tuple(
                        sorted(
                            axis
                            for compiled_axis in record.reduction_axes
                            for axis in record.axis_provenance_fastest_first[
                                compiled_axis
                            ]
                        )
                    )
                    assert original_reduction_axes == broadcast_axes
