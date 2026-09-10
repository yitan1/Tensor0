from __future__ import annotations

from math import prod

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensor0._stride._plan import (
    CompleteMode,
    AffinePlan,
    AffineRecord,
)
from tensor0._stride._map import _execute_map
from tensor0._stride._testing import (
    _native_call_count_for_tests,
    _native_worker_counts_for_tests,
    _prepared_native_call_for_tests,
    _reset_native_call_count_for_tests,
    _set_native_worker_limit_for_tests,
    native_available,
)
from tensor0._stride._native_descriptor import AFFINE_DESCRIPTOR_VERSION, lower_plan
from tensor0._stride._plan import (
    PlanValidationError,
    build_affine_plan,
)

from ._oracle import assert_bitwise_equal, execute_reference


def _row_major_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    expected = 1
    strides = [0] * len(shape)
    for axis in reversed(range(len(shape))):
        strides[axis] = expected
        expected *= shape[axis]
    return tuple(strides)


def _destination_layout(
    shape: tuple[int, ...],
    fastest_first: tuple[int, ...],
    signs: tuple[int, ...],
) -> tuple[tuple[int, ...], int]:
    absolute = [0] * len(shape)
    expected = 1
    for axis in fastest_first:
        absolute[axis] = expected
        expected *= shape[axis]
    strides = tuple(
        sign * stride for sign, stride in zip(signs, absolute, strict=True)
    )
    offset = sum(
        (extent - 1) * stride
        for extent, stride, sign in zip(shape, absolute, signs, strict=True)
        if sign < 0
    )
    return strides, offset


def _signed_plan(
    *,
    physical_shape: tuple[int, ...] = (2, 3),
    logical_to_physical: tuple[int, ...] = (0, 1),
    source_signs: tuple[int, ...] = (-1, 1),
    destination_fastest: tuple[int, ...] = (1, 0),
    destination_signs: tuple[int, ...] = (1, 1),
    source_dtype: str = "float32",
    result_dtype: str = "float32",
    scale: float | complex = -0.75,
) -> AffinePlan:
    physical_strides = _row_major_strides(physical_shape)
    logical_shape = tuple(physical_shape[axis] for axis in logical_to_physical)
    source_strides = tuple(
        source_signs[logical_axis] * physical_strides[physical_axis]
        for logical_axis, physical_axis in enumerate(logical_to_physical)
    )
    source_offset = sum(
        (physical_shape[physical_axis] - 1) * physical_strides[physical_axis]
        for logical_axis, physical_axis in enumerate(logical_to_physical)
        if source_signs[logical_axis] < 0
    )
    destination_strides, destination_offset = _destination_layout(
        logical_shape,
        destination_fastest,
        destination_signs,
    )
    return build_affine_plan(
        records=(
            AffineRecord(
                logical_shape,
                source_strides,
                source_offset,
                destination_strides,
                destination_offset,
                scale,
            ),
        ),
        output_size=prod(logical_shape),
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=prod(physical_shape),
        source_dtype=source_dtype,
        result_dtype=result_dtype,
    )


def _broadcast_plan(
    *,
    logical_shape: tuple[int, ...] = (2, 3),
    broadcast_axes: tuple[int, ...] = (0,),
    source_signs: tuple[int, ...] = (-1,),
    destination_fastest: tuple[int, ...] = (1, 0),
    destination_signs: tuple[int, ...] = (1, 1),
    source_dtype: str = "float32",
    result_dtype: str = "float32",
    scale: float | complex = -0.75,
) -> AffinePlan:
    map_axes = tuple(
        axis
        for axis, extent in enumerate(logical_shape)
        if extent > 1 and axis not in broadcast_axes
    )
    source_shape = tuple(logical_shape[axis] for axis in map_axes) or (1,)
    physical_strides = _row_major_strides(source_shape)
    source_strides = [0] * len(logical_shape)
    source_offset = 0
    for position, (axis, sign) in enumerate(
        zip(map_axes, source_signs, strict=True)
    ):
        stride = physical_strides[position]
        source_strides[axis] = sign * stride
        if sign < 0:
            source_offset += (logical_shape[axis] - 1) * stride
    destination_strides, destination_offset = _destination_layout(
        logical_shape,
        destination_fastest,
        destination_signs,
    )
    return build_affine_plan(
        records=(
            AffineRecord(
                logical_shape,
                tuple(source_strides),
                source_offset,
                destination_strides,
                destination_offset,
                scale,
                source_broadcast_axes=broadcast_axes,
            ),
        ),
        output_size=prod(logical_shape),
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=prod(source_shape),
        source_dtype=source_dtype,
        result_dtype=result_dtype,
    )


def _values(dtype: str, size: int) -> jax.Array:
    real = np.linspace(-2, 3, size, dtype=np.float32)
    if dtype.startswith("complex"):
        return jnp.asarray(real + 1j * real[::-1], dtype=dtype)
    return jnp.asarray(real, dtype=dtype)


def _native(source: jax.Array, plan: AffinePlan) -> jax.Array:
    return _execute_map(source, plan=plan)


def test_signed_affine_uses_the_common_record_and_descriptors() -> None:
    plan = _signed_plan(
        physical_shape=(4, 3),
        source_signs=(-1, 1),
        destination_fastest=(0, 1),
    )
    lowered = lower_plan(plan)

    assert plan.records[0].source_strides == (-3, 1)
    assert plan.records[0].destination_strides == (1, 4)
    assert AFFINE_DESCRIPTOR_VERSION == 9
    assert lowered.words[1] == AFFINE_DESCRIPTOR_VERSION
    assert lowered.words[21:25] == (-3 & ((1 << 64) - 1), 1, 1, 4)


@pytest.mark.parametrize(
    (
        "logical_to_physical",
        "source_signs",
        "destination_fastest",
        "destination_signs",
    ),
    (
        ((0, 1), (-1, 1), (1, 0), (-1, 1)),
        ((0, 1), (1, -1), (0, 1), (1, -1)),
        ((1, 0), (-1, -1), (1, 0), (1, -1)),
        ((1, 0), (1, -1), (0, 1), (-1, 1)),
    ),
)
@pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")
def test_native_executes_signed_permuted_layouts(
    logical_to_physical: tuple[int, ...],
    source_signs: tuple[int, ...],
    destination_fastest: tuple[int, ...],
    destination_signs: tuple[int, ...],
) -> None:
    plan = _signed_plan(
        logical_to_physical=logical_to_physical,
        source_signs=source_signs,
        destination_fastest=destination_fastest,
        destination_signs=destination_signs,
        scale=1.25,
    )
    source = jnp.linspace(-2, 3, plan.source_size, dtype=jnp.float32)
    compiled = jax.jit(lambda value: _native(value, plan))

    np.testing.assert_array_equal(
        np.asarray(compiled(source)),
        np.asarray(execute_reference(source, plan)),
    )
    hlo = str(compiled.lower(source).compiler_ir("stablehlo")).lower()
    assert hlo.count("custom_call") == 1
    assert "gather" not in hlo
    assert "scatter" not in hlo


@pytest.mark.parametrize(
    ("source_dtype", "result_dtype", "scale"),
    (
        ("float16", "float32", -1.25),
        ("float32", "complex64", 0.5 + 0.75j),
        ("complex64", "float32", -0.75),
    ),
)
@pytest.mark.filterwarnings(
    "ignore:Casting complex values to real discards the imaginary part"
)
@pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")
def test_signed_native_jit_vmap_jvp_and_vjp_match_oracle(
    source_dtype: str,
    result_dtype: str,
    scale: float | complex,
) -> None:
    plan = _signed_plan(
        physical_shape=(4, 3),
        destination_fastest=(0, 1),
        destination_signs=(-1, 1),
        source_dtype=source_dtype,
        result_dtype=result_dtype,
        scale=scale,
    )
    source = _values(source_dtype, plan.source_size)
    tangent = jnp.asarray(source * 0.25 + 0.5, dtype=source.dtype)
    batch = jnp.stack((source, -source, 2 * source))
    execute = lambda value: _native(value, plan)
    oracle = lambda value: execute_reference(value, plan)

    np.testing.assert_array_equal(jax.jit(execute)(source), oracle(source))
    np.testing.assert_array_equal(
        jax.jit(jax.vmap(execute))(batch),
        jax.vmap(oracle)(batch),
    )
    actual_jvp = jax.jvp(execute, (source,), (tangent,))[1]
    expected_jvp = jax.jvp(oracle, (source,), (tangent,))[1]
    np.testing.assert_array_equal(actual_jvp, expected_jvp)
    cotangent = _values(result_dtype, plan.output_size)
    actual_vjp = jax.vjp(execute, source)[1](cotangent)[0]
    expected_vjp = jax.vjp(oracle, source)[1](cotangent)[0]
    np.testing.assert_array_equal(actual_vjp, expected_vjp)


@pytest.mark.parametrize(
    ("source_dtype", "result_dtype", "scale"),
    (
        ("float16", "float32", -1.25),
        ("float32", "complex64", 0.5 + 0.75j),
        ("complex64", "float32", -0.75),
    ),
)
@pytest.mark.filterwarnings(
    "ignore:Casting complex values to real discards the imaginary part"
)
@pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")
def test_large_mixed_rank2_leaf_matches_reference_sequential_and_parallel(
    source_dtype: str,
    result_dtype: str,
    scale: float | complex,
) -> None:
    rows = columns = 512
    size = rows * columns
    plan = build_affine_plan(
        records=(
            AffineRecord(
                (rows, columns),
                (1, rows),
                0,
                (columns, 1),
                0,
                scale,
            ),
        ),
        output_size=size,
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=size,
        source_dtype=source_dtype,
        result_dtype=result_dtype,
    )
    source = _values(source_dtype, size)
    expected = execute_reference(source, plan)
    compiled = jax.jit(lambda value: _native(value, plan))

    worker_counts: list[tuple[int, int]] = []
    for worker_limit in (1, 4):
        _set_native_worker_limit_for_tests(worker_limit)
        try:
            _reset_native_call_count_for_tests()
            actual = compiled(source)
            actual.block_until_ready()
            worker_counts.append(_native_worker_counts_for_tests())
        finally:
            _set_native_worker_limit_for_tests(None)

        assert _native_call_count_for_tests() == 1
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))

    assert worker_counts[0][0] == 1
    parallel_workers, available_workers = worker_counts[1]
    if available_workers >= 2:
        assert parallel_workers >= 2

    cotangent = _values(result_dtype, size)
    pullback = jax.jit(jax.vjp(lambda value: _native(value, plan), source)[1])
    expected_vjp = jax.vjp(
        lambda value: execute_reference(value, plan),
        source,
    )[1](cotangent)[0]
    reverse_worker_counts: list[tuple[int, int]] = []
    for worker_limit in (1, 4):
        _set_native_worker_limit_for_tests(worker_limit)
        try:
            _reset_native_call_count_for_tests()
            actual_vjp = pullback(cotangent)[0]
            actual_vjp.block_until_ready()
            reverse_worker_counts.append(_native_worker_counts_for_tests())
        finally:
            _set_native_worker_limit_for_tests(None)

        assert _native_call_count_for_tests() == 1
        np.testing.assert_array_equal(
            np.asarray(actual_vjp),
            np.asarray(expected_vjp),
        )

    assert reverse_worker_counts[0][0] == 1
    reverse_parallel_workers, reverse_available_workers = reverse_worker_counts[1]
    if reverse_available_workers >= 2:
        assert reverse_parallel_workers >= 2


@pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")
def test_prepared_native_rejects_int64_min_signed_stride() -> None:
    plan = _signed_plan(physical_shape=(4, 3), destination_fastest=(0, 1))
    words = list(lower_plan(plan).words)
    words[21] = 1 << 63
    descriptor = b"".join(word.to_bytes(8, "little") for word in words)
    source = jnp.arange(plan.source_size, dtype=jnp.float32)

    with pytest.raises(Exception, match="stride magnitude exceeds int64"):
        _prepared_native_call_for_tests(
            source,
            descriptor=descriptor,
            output_size=plan.output_size,
        ).block_until_ready()


def test_broadcast_read_uses_the_common_affine_record() -> None:
    plan = _broadcast_plan(destination_signs=(1, -1))
    record = plan.records[0]
    lowered = lower_plan(plan)

    assert record.source_broadcast_axes == (0,)
    assert plan.has_source_broadcast
    assert len(lowered.words) == 12 + 7 + 3 * len(record.logical_shape)
    assert lowered.words[21:23] == tuple(
        stride & ((1 << 64) - 1) for stride in record.source_strides
    )
    assert lowered.words[1] == AFFINE_DESCRIPTOR_VERSION


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
def test_native_executes_signed_broadcast_layouts(
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
    source = jnp.linspace(-2, 3, plan.source_size, dtype=jnp.float32)
    compiled = jax.jit(lambda value: _native(value, plan))

    _reset_native_call_count_for_tests()
    actual = compiled(source)
    actual.block_until_ready()
    assert _native_call_count_for_tests() == 1
    np.testing.assert_array_equal(actual, execute_reference(source, plan))


@pytest.mark.parametrize("broadcast_extent", (2, 3, 4, 7, 16, 64))
@pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")
def test_native_broadcast_contiguous_row_extents_match_reference(
    broadcast_extent: int,
) -> None:
    plan = _broadcast_plan(
        logical_shape=(257, broadcast_extent),
        broadcast_axes=(1,),
        source_signs=(1,),
        scale=1.25,
    )
    source = _values("float32", plan.source_size)
    actual = jax.jit(lambda value: _native(value, plan))(source)

    assert_bitwise_equal(actual, execute_reference(source, plan))


@pytest.mark.parametrize(
    ("source_dtype", "result_dtype", "scale"),
    (
        ("float16", "float32", -1.25),
        ("float32", "complex64", 0.5 + 0.75j),
        ("complex64", "float32", -0.75),
    ),
)
@pytest.mark.filterwarnings(
    "ignore:Casting complex values to real discards the imaginary part"
)
@pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")
def test_broadcast_native_batch_vmap_and_jvp_match_oracle(
    source_dtype: str,
    result_dtype: str,
    scale: float | complex,
) -> None:
    plan = _broadcast_plan(
        broadcast_axes=(1,),
        destination_signs=(-1, 1),
        source_dtype=source_dtype,
        result_dtype=result_dtype,
        scale=scale,
    )
    source = _values(source_dtype, plan.source_size)
    tangent = jnp.asarray(source * 0.25 + 0.5, dtype=source.dtype)
    batch = jnp.stack((source, -source, 2 * source))
    execute = lambda value: _native(value, plan)
    oracle = lambda value: execute_reference(value, plan)

    np.testing.assert_array_equal(jax.jit(execute)(batch), oracle(batch))
    np.testing.assert_array_equal(
        jax.jit(jax.vmap(execute))(batch),
        jax.vmap(oracle)(batch),
    )
    actual_jvp = jax.jvp(execute, (source,), (tangent,))[1]
    expected_jvp = jax.jvp(oracle, (source,), (tangent,))[1]
    np.testing.assert_array_equal(actual_jvp, expected_jvp)
    cotangent = _values(result_dtype, plan.output_size)
    actual_vjp = jax.vjp(execute, source)[1](cotangent)[0]
    expected_vjp = jax.vjp(oracle, source)[1](cotangent)[0]
    np.testing.assert_array_equal(actual_vjp, expected_vjp)


@pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")
def test_mixed_broadcast_contiguous_row_matches_promoted_complex_oracle() -> None:
    plan = _broadcast_plan(
        logical_shape=(8, 7),
        broadcast_axes=(1,),
        source_signs=(1,),
        source_dtype="float32",
        result_dtype="complex64",
        scale=complex(np.nan, 1.0),
    )
    source = jnp.asarray(
        [
            0.0,
            -0.0,
            np.inf,
            -np.inf,
            np.nan,
            np.finfo(np.float32).max,
            -np.finfo(np.float32).max,
            np.finfo(np.float32).tiny,
        ],
        dtype=jnp.float32,
    )
    actual = jax.jit(lambda value: _native(value, plan))(source)

    assert_bitwise_equal(actual, execute_reference(source, plan))


@pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")
def test_real_to_complex_map_keeps_full_complex_cross_terms() -> None:
    plan = _broadcast_plan(
        logical_shape=(1, 2),
        broadcast_axes=(1,),
        source_signs=(),
        source_dtype="float32",
        result_dtype="complex64",
        scale=complex(np.nan, 1.0),
    )
    source = jnp.asarray([-0.0], dtype=jnp.float32)

    actual = jax.jit(lambda value: _native(value, plan))(source)
    expected = execute_reference(source, plan)

    assert_bitwise_equal(actual, expected)
    components = np.asarray(actual).view(np.float32)
    assert np.isnan(components).all()


@pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")
def test_large_native_broadcast_uses_parallel_ranges() -> None:
    plan = _broadcast_plan(
        logical_shape=(4096, 512),
        broadcast_axes=(1,),
        source_signs=(1,),
        scale=1,
    )
    source = jnp.arange(plan.source_size, dtype=jnp.float32)
    _set_native_worker_limit_for_tests(4)
    try:
        actual = jax.jit(lambda value: _native(value, plan))(source)
        actual.block_until_ready()
        workers, available = _native_worker_counts_for_tests()
    finally:
        _set_native_worker_limit_for_tests(None)

    np.testing.assert_array_equal(
        np.asarray(actual).reshape(4096, 512)[:, 0],
        np.asarray(source),
    )
    if available >= 2:
        assert workers >= 2


@pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")
def test_large_repeated_source_map_executes_but_unstructured_vjp_is_rejected() -> (
    None
):
    rows, columns = 1024, 512
    plan = build_affine_plan(
        records=(
            AffineRecord(
                (rows, columns),
                (1, 1),
                0,
                (columns, 1),
                0,
                1.25,
            ),
        ),
        output_size=rows * columns,
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=rows + columns - 1,
        source_dtype=jnp.float32,
        result_dtype=jnp.float32,
    )
    source = jnp.linspace(-2, 3, plan.source_size, dtype=jnp.float32)
    execute = jax.jit(lambda value: _native(value, plan))

    _set_native_worker_limit_for_tests(4)
    try:
        actual = execute(source)
        actual.block_until_ready()
        workers, available = _native_worker_counts_for_tests()
    finally:
        _set_native_worker_limit_for_tests(None)

    np.testing.assert_array_equal(actual, execute_reference(source, plan))
    if available >= 2:
        assert workers >= 2

    with pytest.raises(NotImplementedError, match="unstructured repeated source"):
        jax.vjp(lambda value: _native(value, plan), source)[1](
            jnp.ones(plan.output_size, dtype=jnp.float32)
        )


def test_broadcast_axes_must_name_every_nontrivial_zero_stride() -> None:
    record = AffineRecord((2, 3), (0, 1), 0, (3, 1), 0)
    with pytest.raises(PlanValidationError, match="source_broadcast_axes"):
        build_affine_plan(
            records=(record,),
            output_size=6,
            coverage=CompleteMode.COMPLETE_UNIQUE,
            source_size=3,
            source_dtype=jnp.float32,
            result_dtype=jnp.float32,
        )
