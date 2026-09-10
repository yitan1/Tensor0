from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Iterator
from math import prod

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensor0._stride._plan import (
    CompleteMode,
    AffinePlan,
    AffineRecord,
    StridedReductionKind,
    StridedScalarKind,
    StridedWriteKind,
    build_affine_plan,
)
from tensor0._stride._map import _execute_map
from tensor0._stride._testing import (
    _native_call_count_for_tests,
    _native_worker_counts_for_tests,
    _reset_native_call_count_for_tests,
    _set_native_disable_f16_f16_contiguous_simd_for_tests,
    _set_native_disable_f16_f32_contiguous_simd_for_tests,
    _set_native_force_generic_for_tests,
    _set_native_worker_limit_for_tests,
    native_available,
)
from tensor0._stride._ops._reduction import (
    _bind_grouped_native_structured_reduction,
    _bind_sequential_native_structured_reduction,
    bind_native_structured_reduction,
)
from tensor0._stride._ops._selected_scale import _selected_scale_alias, build_selected_scale_plan
from tests.stride._fixtures import execute_update_scale
from tests.stride._fixtures import execute_update_accumulate, execute_update_assign
from tensor0._stride._ops._update_support import build_base_accumulate_plan, build_base_assign_plan

from ._fixtures import (
    contiguous_dtype_plan,
    rank2_transpose_plan,
    rank4_tiled_plan,
    rank4_two_pair_plan,
    rank_zero_plan,
    two_record_noncompact_plan,
)
from ._oracle import (
    assert_bitwise_equal,
    execute_base_accumulate_reference,
    execute_base_assign_reference,
    execute_reference,
    execute_selected_scale_reference,
)


pytestmark = pytest.mark.skipif(
    not native_available(),
    reason="native CPU stride unavailable",
)


@contextmanager
def _generic_only() -> Iterator[None]:
    """Compile new executables with every layout specialization disabled."""

    _set_native_force_generic_for_tests(True)
    jax.clear_caches()
    try:
        yield
    finally:
        _set_native_force_generic_for_tests(False)
        jax.clear_caches()


@contextmanager
def _strided_generated() -> Iterator[None]:
    """Compile a fresh executable with the production generated policy."""

    jax.clear_caches()
    try:
        yield
    finally:
        jax.clear_caches()


@contextmanager
def _without_f16_f32_contiguous_simd() -> Iterator[None]:
    """Disable the production continuous half-to-float SIMD policy."""

    _set_native_disable_f16_f32_contiguous_simd_for_tests(True)
    try:
        yield
    finally:
        _set_native_disable_f16_f32_contiguous_simd_for_tests(False)


@contextmanager
def _without_f16_f16_contiguous_simd() -> Iterator[None]:
    """Disable the production continuous half-to-half SIMD policy."""

    _set_native_disable_f16_f16_contiguous_simd_for_tests(True)
    jax.clear_caches()
    try:
        yield
    finally:
        _set_native_disable_f16_f16_contiguous_simd_for_tests(False)
        jax.clear_caches()


def _affine_plan(
    shape: tuple[int, ...],
    source_strides: tuple[int, ...],
    destination_strides: tuple[int, ...],
) -> AffinePlan:
    source_size = 1 + sum(
        (extent - 1) * abs(stride)
        for extent, stride in zip(shape, source_strides, strict=True)
    )
    return build_affine_plan(
        records=(
            AffineRecord(
                logical_shape=shape,
                source_strides=source_strides,
                source_offset=0,
                destination_strides=destination_strides,
                destination_offset=0,
                scale=0.75,
            ),
        ),
        output_size=prod(shape),
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=source_size,
        source_dtype=jnp.float32,
        result_dtype=jnp.float32,
    )


def _high_rank_blocked_plan(shape: tuple[int, ...]) -> AffinePlan:
    return _high_rank_blocked_dtype_plan(
        shape, jnp.float32, jnp.float32, -1.25
    )


def _high_rank_blocked_dtype_plan(
    shape: tuple[int, ...], source_dtype, result_dtype, scale
) -> AffinePlan:
    source_strides = tuple(
        prod(shape[axis + 1 :]) for axis in range(len(shape))
    )
    destination_strides = tuple(
        prod(shape[:axis]) for axis in range(len(shape))
    )
    source_offset = (shape[0] - 1) * source_strides[0]
    return build_affine_plan(
        records=(
            AffineRecord(
                logical_shape=shape,
                source_strides=(-source_strides[0], *source_strides[1:]),
                source_offset=source_offset,
                destination_strides=destination_strides,
                destination_offset=0,
                scale=scale,
            ),
        ),
        output_size=prod(shape),
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=prod(shape),
        source_dtype=source_dtype,
        result_dtype=result_dtype,
    )


def _rank_plan(rank: int) -> AffinePlan:
    if rank == 0:
        return rank_zero_plan()
    shape = (2,) * rank
    size = prod(shape)
    source_strides = tuple(1 << axis for axis in reversed(range(rank)))
    source_offset = 0
    if rank > 1:
        source_strides = (-source_strides[0], *source_strides[1:])
        source_offset = size // 2
    destination_strides = tuple(1 << axis for axis in range(rank))
    return build_affine_plan(
        records=(
            AffineRecord(
                logical_shape=shape,
                source_strides=source_strides,
                source_offset=source_offset,
                destination_strides=destination_strides,
                destination_offset=0,
                scale=1.25,
            ),
        ),
        output_size=size,
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=size,
        source_dtype=jnp.float32,
        result_dtype=jnp.float32,
    )


def _rank_zero_unit_plan() -> AffinePlan:
    return build_affine_plan(
        records=(AffineRecord((), (), 0, (), 0),),
        output_size=1,
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=1,
        source_dtype=jnp.float32,
        result_dtype=jnp.float32,
    )


def _mixed_broadcast_plan() -> AffinePlan:
    return build_affine_plan(
        records=(
            AffineRecord(
                logical_shape=(2, 3),
                source_strides=(0, -1),
                source_offset=2,
                destination_strides=(3, 1),
                destination_offset=0,
                scale=1.25 - 0.5j,
                source_broadcast_axes=(0,),
            ),
        ),
        output_size=6,
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=3,
        source_dtype=jnp.float32,
        result_dtype=jnp.complex64,
    )


def _dtype_values(dtype, size: int) -> jax.Array:
    resolved = jnp.dtype(dtype)
    if resolved == jnp.dtype(jnp.bool_):
        return jnp.asarray(np.arange(size) % 3 != 0, dtype=resolved)
    values = jnp.arange(size, dtype=resolved)
    if jnp.issubdtype(resolved, jnp.complexfloating):
        values = values + 1j * values[::-1]
    return jnp.asarray(values, dtype=resolved)


def _complex_finite_values(size: int) -> jax.Array:
    components = np.asarray(
        [
            0.0,
            -0.0,
            1.25,
            -1.25,
            .125,
            np.float32(1e10),
            -np.float32(1e10),
            np.float32(1e-10),
        ],
        dtype=np.float32,
    )
    values = np.asarray(
        [
            complex(real, imaginary)
            for real in components
            for imaginary in components
        ],
        dtype=np.complex64,
    )
    return jnp.asarray(np.resize(values, size))


@pytest.mark.parametrize("rank", range(9))
def test_generic_only_executes_effective_ranks_zero_through_eight(rank: int) -> None:
    plan = _rank_plan(rank)
    source = jnp.arange(plan.source_size, dtype=jnp.float32) - 3

    with _generic_only():
        actual = jax.jit(
            lambda value: _execute_map(
                value,
                plan=plan,
            )
        )(source)
        actual.block_until_ready()

    assert_bitwise_equal(actual, execute_reference(source, plan))


def test_strided_generated_recursive_subdomains_match_special_values() -> None:
    plan = _affine_plan((65_536, 4), (8, 2), (4, 1))
    values = np.resize(
        np.asarray(
            [0.0, -0.0, np.inf, -np.inf, np.nan, np.finfo(np.float32).max],
            dtype=np.float32,
        ),
        plan.source_size,
    )
    source = jnp.asarray(values)

    _set_native_worker_limit_for_tests(4)
    try:
        with _strided_generated():
            actual = jax.jit(
                lambda value: _execute_map(
                    value,
                    plan=plan,
                )
            )(source)
            actual.block_until_ready()
            workers, _ = _native_worker_counts_for_tests()
    finally:
        _set_native_worker_limit_for_tests(None)

    assert workers == 4
    assert_bitwise_equal(actual, execute_reference(source, plan))


def test_strided_generated_walker_executes_multiple_records() -> None:
    plan = two_record_noncompact_plan()
    source = jnp.arange(plan.source_size, dtype=jnp.float32) - 3

    _set_native_worker_limit_for_tests(4)
    try:
        with _strided_generated():
            actual = jax.jit(
                lambda value: _execute_map(
                    value,
                    plan=plan,
                )
            )(source)
            actual.block_until_ready()
            workers, _ = _native_worker_counts_for_tests()
    finally:
        _set_native_worker_limit_for_tests(None)

    assert workers == 1
    assert_bitwise_equal(actual, execute_reference(source, plan))


@pytest.mark.parametrize("rank", (3, 8))
def test_strided_generated_walker_executes_vmap_batches(rank: int) -> None:
    plan = _rank_plan(rank)
    source = jnp.arange(plan.source_size, dtype=jnp.float32) - 3
    batch = jnp.stack(tuple(source + index for index in range(4)))
    execute = jax.vmap(
        lambda value: _execute_map(
            value,
            plan=plan,
        )
    )

    _set_native_worker_limit_for_tests(4)
    try:
        with _strided_generated():
            actual = jax.jit(execute)(batch)
            actual.block_until_ready()
            workers, _ = _native_worker_counts_for_tests()
    finally:
        _set_native_worker_limit_for_tests(None)

    assert workers == 1
    assert_bitwise_equal(actual, execute_reference(batch, plan))


def test_strided_generated_walker_executes_batched_record_sequences() -> None:
    plan = two_record_noncompact_plan()
    source = jnp.arange(plan.source_size, dtype=jnp.float32) - 3
    batch = jnp.stack(tuple(source + index for index in range(4)))

    _set_native_worker_limit_for_tests(4)
    try:
        with _strided_generated():
            actual = jax.jit(
                jax.vmap(
                    lambda value: _execute_map(
                        value,
                        plan=plan,
                    )
                )
            )(batch)
            actual.block_until_ready()
            workers, _ = _native_worker_counts_for_tests()
    finally:
        _set_native_worker_limit_for_tests(None)

    assert workers == 1
    assert_bitwise_equal(actual, execute_reference(batch, plan))


def test_strided_generated_batched_mixed_policy_matches_vjp_oracle() -> None:
    plan = _high_rank_blocked_dtype_plan(
        (8, 7, 4),
        jnp.float32,
        jnp.complex64,
        1.25 - 0.75j,
    )
    source = jnp.arange(plan.source_size, dtype=jnp.float32) - 3
    batch = jnp.stack(tuple(source + index for index in range(4)))
    cotangent = (
        jnp.arange(4 * plan.output_size, dtype=jnp.float32).reshape(
            4, plan.output_size
        )
        * (0.25 + 0.5j)
    ).astype(jnp.complex64)
    native = jax.vmap(
        lambda value: _execute_map(
            value,
            plan=plan,
        )
    )
    reference = jax.vmap(lambda value: execute_reference(value, plan))

    _set_native_worker_limit_for_tests(4)
    try:
        with _strided_generated():
            actual, pullback = jax.jit(
                lambda value: jax.vjp(native, value)
            )(batch)
            actual_gradient = jax.jit(pullback)(cotangent)[0]
            jax.block_until_ready((actual, actual_gradient))
            workers, _ = _native_worker_counts_for_tests()
    finally:
        _set_native_worker_limit_for_tests(None)

    expected, reference_pullback = jax.vjp(reference, batch)
    expected_gradient = reference_pullback(cotangent)[0]
    assert workers == 1
    assert_bitwise_equal(actual, expected)
    assert_bitwise_equal(actual_gradient, expected_gradient)


@pytest.mark.parametrize("rank", range(4, 9))
def test_strided_generated_fixed_loops_cover_effective_rank_eight(rank: int) -> None:
    plan = _rank_plan(rank)
    source = jnp.arange(plan.source_size, dtype=jnp.float32) - 3

    with _strided_generated():
        actual = jax.jit(
            lambda value: _execute_map(
                value,
                plan=plan,
            )
        )(source)
        actual.block_until_ready()

    assert_bitwise_equal(actual, execute_reference(source, plan))


@pytest.mark.parametrize(
    "shape",
    (
        (16, 16, 16, 64),
        (8, 8, 8, 8, 64),
        (8, 8, 8, 8, 8, 8),
        (4, 4, 4, 4, 4, 4, 64),
        (4, 4, 4, 4, 4, 4, 4, 16),
    ),
)
def test_strided_generated_high_rank_blocking_and_threading_match_oracle(
    shape: tuple[int, ...],
) -> None:
    plan = _high_rank_blocked_plan(shape)
    source = jnp.arange(plan.source_size, dtype=jnp.float32) - 3

    _set_native_worker_limit_for_tests(4)
    try:
        with _strided_generated():
            actual = jax.jit(
                lambda value: _execute_map(
                    value,
                    plan=plan,
                )
            )(source)
            actual.block_until_ready()
            workers, _ = _native_worker_counts_for_tests()
    finally:
        _set_native_worker_limit_for_tests(None)

    assert workers == 4
    assert_bitwise_equal(actual, execute_reference(source, plan))


def test_strided_generated_mixed_policy_matches_primal_and_vjp_oracle() -> None:
    plan = _high_rank_blocked_dtype_plan(
        (1024, 64, 4),
        jnp.float32,
        jnp.complex64,
        1.25 - 0.75j,
    )
    source = _dtype_values(jnp.float32, plan.source_size) - 3
    cotangent = (
        _dtype_values(jnp.complex64, plan.output_size) * (0.5 - 0.25j)
    )
    native_function = lambda value: _execute_map(  # noqa: E731
        value, plan=plan
    )
    reference_function = lambda value: execute_reference(  # noqa: E731
        value, plan
    )

    _set_native_worker_limit_for_tests(4)
    try:
        with _strided_generated():
            actual, native_pullback = jax.jit(
                lambda value: jax.vjp(native_function, value)
            )(source)
            actual_gradient = jax.jit(native_pullback)(cotangent)[0]
            jax.block_until_ready((actual, actual_gradient))
    finally:
        _set_native_worker_limit_for_tests(None)

    expected, reference_pullback = jax.vjp(reference_function, source)
    expected_gradient = reference_pullback(cotangent)[0]
    assert_bitwise_equal(actual, expected)
    assert_bitwise_equal(actual_gradient, expected_gradient)


def test_strided_generated_mixed_policy_matches_finite_values() -> None:
    mixed_plan = build_affine_plan(
        records=(
            AffineRecord(
                logical_shape=(7,),
                source_strides=(1,),
                source_offset=0,
                destination_strides=(1,),
                destination_offset=0,
                scale=1.25 - 0.75j,
            ),
        ),
        output_size=7,
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=7,
        source_dtype=jnp.float32,
        result_dtype=jnp.complex64,
    )
    mixed_source = jnp.asarray(
        [0.0, -0.0, 1.25, -2.5, .125, 2.0, -2.0],
        dtype=jnp.float32,
    )
    cotangent = jnp.asarray(
        [
            0.0 - 0.0j,
            -0.0 + 0.0j,
            1.25 + 0.0j,
            0.0 + 2.5j,
            .125 + 1.0j,
            3.0 + 3.0j,
            -3.0 - 3.0j,
        ],
        dtype=jnp.complex64,
    )

    with _strided_generated():
        actual_mixed, pullback = jax.jit(
            lambda value: jax.vjp(
                lambda item: _execute_map(
                    item, plan=mixed_plan
                ),
                value,
            )
        )(mixed_source)
        actual_gradient = jax.jit(pullback)(cotangent)[0]
        jax.block_until_ready((actual_mixed, actual_gradient))

    expected_mixed, reference_pullback = jax.vjp(
        lambda value: execute_reference(value, mixed_plan), mixed_source
    )
    expected_gradient = reference_pullback(cotangent)[0]
    np.testing.assert_allclose(actual_mixed, expected_mixed, rtol=2e-6, atol=1e-6)
    np.testing.assert_allclose(actual_gradient, expected_gradient, rtol=2e-6, atol=1e-6)


@pytest.mark.parametrize("operation", ("assign", "accumulate"))
def test_strided_generated_float32_update_policies_match_oracle(
    operation: str,
) -> None:
    bound = _high_rank_blocked_plan((1024, 64, 4))
    source = _dtype_values(jnp.float32, bound.source_size) - 3
    base = _dtype_values(jnp.float32, bound.output_size) + 11
    if operation == "assign":
        plan = build_base_assign_plan(bound)
        execute = lambda old, value: execute_update_assign(  # noqa: E731
            old, value, plan=plan
        )
        reference = execute_base_assign_reference
    else:
        plan = build_base_accumulate_plan(bound)
        execute = lambda old, value: execute_update_accumulate(  # noqa: E731
            old, value, plan=plan
        )
        reference = execute_base_accumulate_reference

    _set_native_worker_limit_for_tests(4)
    try:
        with _strided_generated():
            actual = jax.jit(execute)(base, source)
            actual.block_until_ready()
            workers, _ = _native_worker_counts_for_tests()
    finally:
        _set_native_worker_limit_for_tests(None)

    assert workers == 4
    assert_bitwise_equal(actual, reference(base, source, plan))


@pytest.mark.parametrize(
    ("dtype", "scale"),
    (
        (jnp.int32, -3),
        (jnp.float16, -1.25),
        (jnp.bfloat16, -1.25),
        (jnp.complex64, 1.25 - 0.75j),
        (jnp.float64, -1.25),
        (jnp.complex128, 1.25 - 0.75j),
    ),
)
def test_production_generated_range_covers_same_dtype_policies(
    dtype, scale
) -> None:
    with jax.enable_x64():
        plan = _high_rank_blocked_dtype_plan(
            (8, 7, 4), dtype, dtype, scale
        )
        source = _dtype_values(dtype, plan.source_size)
        jax.clear_caches()
        actual = jax.jit(
            lambda value: _execute_map(
                value, plan=plan
            )
        )(source)
        actual.block_until_ready()
        expected = execute_reference(source, plan)

    assert_bitwise_equal(actual, expected)


@pytest.mark.parametrize(
    ("source_dtype", "result_dtype", "scale"),
    (
        (jnp.float16, jnp.float32, -1.25),
        (jnp.float32, jnp.complex64, 1.25 - 0.75j),
        (jnp.complex64, jnp.float32, -0.75),
        (jnp.float64, jnp.complex128, 0.5 + 0.75j),
        (jnp.complex128, jnp.float64, -0.75),
    ),
)
@pytest.mark.filterwarnings("ignore:Casting complex values to real")
def test_production_generated_range_covers_mixed_primal_and_vjp(
    source_dtype, result_dtype, scale
) -> None:
    with jax.enable_x64():
        plan = _high_rank_blocked_dtype_plan(
            (8, 7, 4), source_dtype, result_dtype, scale
        )
        source = _dtype_values(source_dtype, plan.source_size)
        cotangent = _dtype_values(result_dtype, plan.output_size)
        native = lambda value: _execute_map(  # noqa: E731
            value, plan=plan
        )
        reference = lambda value: execute_reference(value, plan)  # noqa: E731
        jax.clear_caches()
        actual, actual_vjp = jax.jit(
            lambda value, ct: (
                native(value),
                jax.vjp(native, value)[1](ct)[0],
            )
        )(source, cotangent)
        jax.block_until_ready((actual, actual_vjp))
        expected = reference(source)
        expected_vjp = jax.vjp(reference, source)[1](cotangent)[0]

    np.testing.assert_allclose(actual, expected)
    np.testing.assert_allclose(actual_vjp, expected_vjp)


@pytest.mark.parametrize(
    ("dtype", "scale"),
    (
        (jnp.bool_, True),
        (jnp.int8, -3),
        (jnp.int16, -3),
        (jnp.int32, -3),
        (jnp.int64, -3),
        (jnp.uint8, 3),
        (jnp.uint16, 3),
        (jnp.uint32, 3),
        (jnp.uint64, 3),
        (jnp.float16, -1.25),
        (jnp.bfloat16, -1.25),
        (jnp.float32, -1.25),
        (jnp.float64, -1.25),
        (jnp.complex64, 1.25 - 0.75j),
        (jnp.complex128, 1.25 - 0.75j),
    ),
)
def test_generic_only_covers_every_same_dtype_scalar_family(dtype, scale) -> None:
    with jax.enable_x64(), _generic_only():
        plan = contiguous_dtype_plan(dtype, scale, size=17)
        source = _dtype_values(dtype, plan.source_size)
        actual = jax.jit(
            lambda value: _execute_map(
                value,
                plan=plan,
            )
        )(source)
        actual.block_until_ready()
        expected = execute_reference(source, plan)

    assert_bitwise_equal(actual, expected)


@pytest.mark.parametrize(
    ("source_dtype", "result_dtype", "scale"),
    (
        (jnp.float16, jnp.float32, -1.25),
        (jnp.float32, jnp.complex64, 1.25 - 0.75j),
        (jnp.complex64, jnp.float32, -0.75),
        (jnp.float64, jnp.complex128, 0.5 + 0.75j),
        (jnp.complex128, jnp.float64, -0.75),
    ),
)
@pytest.mark.filterwarnings("ignore:Casting complex values to real")
def test_generic_only_covers_mixed_forward_and_transpose_scalar_families(
    source_dtype,
    result_dtype,
    scale,
) -> None:
    template = rank2_transpose_plan(rows=5, columns=7)
    with jax.enable_x64(), _generic_only():
        plan = build_affine_plan(
            records=tuple(
                AffineRecord(
                    logical_shape=record.logical_shape,
                    source_strides=record.source_strides,
                    source_offset=record.source_offset,
                    destination_strides=record.destination_strides,
                    destination_offset=record.destination_offset,
                    scale=scale,
                )
                for record in template.records
            ),
            output_size=template.output_size,
            coverage=template.coverage,
            source_size=template.source_size,
            source_dtype=source_dtype,
            result_dtype=result_dtype,
        )
        source = _dtype_values(source_dtype, plan.source_size)
        cotangent = _dtype_values(result_dtype, plan.output_size)
        native = lambda value: _execute_map(
            value,
            plan=plan,
        )
        reference = lambda value: execute_reference(value, plan)
        actual = jax.jit(native)(source)
        actual_vjp = jax.jit(
            lambda value, ct: jax.vjp(native, value)[1](ct)[0]
        )(source, cotangent)
        jax.block_until_ready((actual, actual_vjp))
        expected = reference(source)
        expected_vjp = jax.vjp(reference, source)[1](cotangent)[0]

    np.testing.assert_allclose(actual, expected)
    np.testing.assert_allclose(actual_vjp, expected_vjp)


def test_contiguous_map_core_covers_mixed_fresh_and_base_updates() -> None:
    size = 257
    bound = build_affine_plan(
        records=(
            AffineRecord(
                (size,),
                (1,),
                0,
                (1,),
                0,
                0.5 + 0.75j,
            ),
        ),
        output_size=size,
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=size,
        source_dtype=jnp.float32,
        result_dtype=jnp.complex64,
    )
    assign_plan = build_base_assign_plan(bound)
    accumulate_plan = build_base_accumulate_plan(bound)
    source = jnp.linspace(-2, 3, size, dtype=jnp.float32)
    base_real = jnp.linspace(3, -2, size, dtype=jnp.float32)
    base = (base_real + 1j * base_real[::-1]).astype(jnp.complex64)

    fresh = jax.jit(
        lambda value: _execute_map(
            value,
            plan=bound,
        )
    ).lower(source).compile()
    assign = jax.jit(
        lambda old, value: execute_update_assign(old, value, plan=assign_plan)
    ).lower(base, source).compile()
    accumulate = jax.jit(
        lambda old, value: execute_update_accumulate(
            old,
            value,
            plan=accumulate_plan,
        )
    ).lower(base, source).compile()

    _reset_native_call_count_for_tests()
    actual_fresh = fresh(source)
    actual_assign = assign(base, source)
    actual_accumulate = accumulate(base, source)
    jax.block_until_ready((actual_fresh, actual_assign, actual_accumulate))

    assert _native_call_count_for_tests() == 3
    assert_bitwise_equal(actual_fresh, execute_reference(source, bound))
    assert_bitwise_equal(
        actual_assign,
        execute_base_assign_reference(base, source, assign_plan),
    )
    assert_bitwise_equal(
        actual_accumulate,
        execute_base_accumulate_reference(base, source, accumulate_plan),
    )


@pytest.mark.parametrize("scale", (1.0, -1.25))
def test_f16_f16_contiguous_simd_matches_scalar_finite_values(scale: float) -> None:
    bits = np.resize(
        np.asarray(
            (
                0x0000,
                0x8000,
                0x3C00,
                0xBC00,
                0x4100,
                0xC100,
                0x3000,
                0x0001,
                0x03FF,
                0x7800,
                0x3555,
                0xB555,
                0x4000,
                0xC000,
                0x0400,
                0x8400,
                0x3800,
            ),
            dtype=np.uint16,
        ),
        257,
    )
    source = jnp.asarray(bits.view(np.float16))
    plan = build_affine_plan(
        records=(
            AffineRecord(
                (source.size,),
                (1,),
                0,
                (1,),
                0,
                scale,
            ),
        ),
        output_size=source.size,
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=source.size,
        source_dtype=jnp.float16,
        result_dtype=jnp.float16,
    )
    executable = jax.jit(
        lambda value: _execute_map(
            value,
            plan=plan,
        )
    ).lower(source).compile()

    with _without_f16_f16_contiguous_simd():
        expected = executable(source)
        expected.block_until_ready()
    actual = executable(source)
    actual.block_until_ready()

    np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=1e-7)


def test_f16_f16_contiguous_simd_preserves_jvp_vjp_and_batching() -> None:
    size = 257
    plan = build_affine_plan(
        records=(
            AffineRecord(
                (size,),
                (1,),
                0,
                (1,),
                0,
                -1.25,
            ),
        ),
        output_size=size,
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=size,
        source_dtype=jnp.float16,
        result_dtype=jnp.float16,
    )
    source = jnp.asarray(np.linspace(-1, 1, size, dtype=np.float16))
    tangent = jnp.linspace(1, -1, size, dtype=jnp.float16)
    cotangent = jnp.linspace(-2, 3, size, dtype=jnp.float16)
    batch = jnp.stack((source, source * jnp.float16(0.5)))
    run = lambda value: _execute_map(
        value,
        plan=plan,
    )
    transformed = jax.jit(
        lambda value, direction, ct, batched: (
            jax.jvp(run, (value,), (direction,)),
            jax.vjp(run, value)[1](ct)[0],
            jax.vmap(run)(batched),
        )
    ).lower(source, tangent, cotangent, batch).compile()

    with _without_f16_f16_contiguous_simd():
        expected = transformed(source, tangent, cotangent, batch)
        jax.block_until_ready(expected)
    actual = transformed(source, tangent, cotangent, batch)
    jax.block_until_ready(actual)

    assert_bitwise_equal(actual[0][0], expected[0][0])
    assert_bitwise_equal(actual[0][1], expected[0][1])
    assert_bitwise_equal(actual[1], expected[1])
    assert_bitwise_equal(actual[2], expected[2])


def test_large_f16_f16_vjp_uses_parallel_contiguous_leaf() -> None:
    size = 1 << 20
    plan = build_affine_plan(
        records=(AffineRecord((size,), (1,), 0, (1,), 0, -1.25),),
        output_size=size,
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=size,
        source_dtype=jnp.float16,
        result_dtype=jnp.float16,
    )
    source = jnp.asarray(np.linspace(-1, 1, size, dtype=np.float16))
    cotangent = jnp.ones((size,), dtype=jnp.float16)
    run = lambda value: _execute_map(
        value,
        plan=plan,
    )
    executable = jax.jit(
        lambda value: jax.vjp(run, source)[1](value)[0]
    ).lower(cotangent).compile()

    try:
        _set_native_worker_limit_for_tests(1)
        serial = executable(cotangent)
        serial.block_until_ready()
        serial_workers, available_workers = _native_worker_counts_for_tests()

        _set_native_worker_limit_for_tests(8)
        parallel = executable(cotangent)
        parallel.block_until_ready()
        parallel_workers, parallel_available = _native_worker_counts_for_tests()
    finally:
        _set_native_worker_limit_for_tests(None)

    assert serial_workers == 1
    assert parallel_available == available_workers
    if parallel_available > 1:
        assert parallel_workers > 1
    assert_bitwise_equal(parallel, serial)


def test_f16_f16_contiguous_inner_rows_simd_matches_scalar() -> None:
    rows = 128
    columns = 257
    source_pitch = 512
    bits = np.resize(
        np.asarray(
            (
                0x0000,
                0x8000,
                0x3C00,
                0xBC00,
                0x7C00,
                0xFC00,
                0x7E00,
                0x0001,
                0x03FF,
                0x7BFF,
            ),
            dtype=np.uint16,
        ),
        rows * source_pitch,
    )
    source = jnp.asarray(bits.view(np.float16))
    plan = build_affine_plan(
        records=(
            AffineRecord(
                (rows, columns),
                (source_pitch, 1),
                0,
                (columns, 1),
                0,
                -1.25,
            ),
        ),
        output_size=rows * columns,
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=source.size,
        source_dtype=jnp.float16,
        result_dtype=jnp.float16,
    )
    executable = jax.jit(
        lambda value: _execute_map(
            value,
            plan=plan,
        )
    ).lower(source).compile()

    with _without_f16_f16_contiguous_simd():
        expected = executable(source)
        expected.block_until_ready()
    actual = executable(source)
    actual.block_until_ready()

    assert_bitwise_equal(actual, expected)
    assert_bitwise_equal(actual, execute_reference(source, plan))


@pytest.mark.parametrize(
    ("shape", "source_strides", "destination_strides"),
    [
        ((65, 129), (1, 65), (129, 1)),
        ((65, 129), (129, 1), (1, 65)),
        ((512, 512), (1, 512), (512, 1)),
        ((512, 512), (512, 1), (1, 512)),
    ],
    ids=("forward-tail", "reverse-tail", "forward-parallel", "reverse-parallel"),
)
def test_f16_rank2_permutation_leaf_matches_generic(
    shape: tuple[int, int],
    source_strides: tuple[int, int],
    destination_strides: tuple[int, int],
) -> None:
    size = prod(shape)
    bits = np.resize(
        np.asarray(
            (
                0x0000,
                0x8000,
                0x3C00,
                0xBC00,
                0x7C00,
                0xFC00,
                0x7E00,
                0x0001,
                0x03FF,
                0x7BFF,
            ),
            dtype=np.uint16,
        ),
        size,
    )
    source = jnp.asarray(bits.view(np.float16))
    plan = build_affine_plan(
        records=(
            AffineRecord(
                shape,
                source_strides,
                0,
                destination_strides,
                0,
                -1.25,
            ),
        ),
        output_size=size,
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=size,
        source_dtype=jnp.float16,
        result_dtype=jnp.float16,
    )
    specialized = jax.jit(
        lambda value: _execute_map(
            value,
            plan=plan,
        )
    ).lower(source).compile()
    with _without_f16_f16_contiguous_simd():
        expected = specialized(source)
        expected.block_until_ready()
    actual = specialized(source)
    jax.block_until_ready((expected, actual))
    assert_bitwise_equal(actual, expected)
    assert_bitwise_equal(actual, execute_reference(source, plan))


@pytest.mark.parametrize(
    ("source_strides", "destination_strides"),
    [
        ((1, 65), (129, 1)),
        ((129, 1), (1, 65)),
    ],
    ids=("forward", "reverse"),
)
def test_f16_rank2_without_f16c_uses_generated_fallback(
    source_strides: tuple[int, int],
    destination_strides: tuple[int, int],
) -> None:
    shape = (65, 129)
    size = prod(shape)
    source = jnp.asarray(np.linspace(-2, 3, size, dtype=np.float16))
    plan = build_affine_plan(
        records=(
            AffineRecord(
                shape,
                source_strides,
                0,
                destination_strides,
                0,
                -1.25,
            ),
        ),
        output_size=size,
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=size,
        source_dtype=jnp.float16,
        result_dtype=jnp.float16,
    )

    with _without_f16_f16_contiguous_simd():
        executable = jax.jit(
            lambda value: _execute_map(
                value,
                plan=plan,
            )
        ).lower(source).compile()
        actual = executable(source)
        actual.block_until_ready()

    assert_bitwise_equal(actual, execute_reference(source, plan))


def test_f16_rank2_permutation_simd_preserves_jvp_vjp_and_batching() -> None:
    rows = 64
    columns = 128
    size = rows * columns
    plan = build_affine_plan(
        records=(
            AffineRecord(
                (rows, columns),
                (1, rows),
                0,
                (columns, 1),
                0,
                -1.25,
            ),
        ),
        output_size=size,
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=size,
        source_dtype=jnp.float16,
        result_dtype=jnp.float16,
    )
    source = jnp.asarray(np.linspace(-1, 1, size, dtype=np.float16))
    tangent = source[::-1]
    cotangent = jnp.asarray(np.linspace(2, -3, size, dtype=np.float16))
    batch = jnp.stack((source, source * jnp.float16(0.5)))
    run = lambda value: _execute_map(
        value,
        plan=plan,
    )
    transformed = jax.jit(
        lambda value, direction, ct, batched: (
            jax.jvp(run, (value,), (direction,)),
            jax.vjp(run, value)[1](ct)[0],
            jax.vmap(run)(batched),
        )
    ).lower(source, tangent, cotangent, batch).compile()

    with _without_f16_f16_contiguous_simd():
        expected = transformed(source, tangent, cotangent, batch)
        jax.block_until_ready(expected)
    actual = transformed(source, tangent, cotangent, batch)
    jax.block_until_ready(actual)

    assert_bitwise_equal(actual[0][0], expected[0][0])
    assert_bitwise_equal(actual[0][1], expected[0][1])
    assert_bitwise_equal(actual[1], expected[1])
    assert_bitwise_equal(actual[2], expected[2])


@pytest.mark.parametrize("scale", (1.0, -1.25))
def test_f16_f32_contiguous_simd_matches_scalar_finite_values(scale: float) -> None:
    bits = np.asarray(
        (
            0x0000,
            0x8000,
            0x3C00,
            0xBC00,
            0x4100,
            0xC100,
            0x3000,
            0x0001,
            0x03FF,
            0x7800,
            0x3555,
            0xB555,
            0x4000,
            0xC000,
            0x0400,
            0x8400,
            0x3800,
        ),
        dtype=np.uint16,
    )
    source = jnp.asarray(bits.view(np.float16))
    plan = build_affine_plan(
        records=(
            AffineRecord(
                (source.size,),
                (1,),
                0,
                (1,),
                0,
                scale,
            ),
        ),
        output_size=source.size,
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=source.size,
        source_dtype=jnp.float16,
        result_dtype=jnp.float32,
    )
    executable = jax.jit(
        lambda value: _execute_map(
            value,
            plan=plan,
        )
    ).lower(source).compile()

    with _without_f16_f32_contiguous_simd():
        expected = executable(source)
        expected.block_until_ready()
    actual = executable(source)
    actual.block_until_ready()

    np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=1e-7)


def test_f16_f32_contiguous_simd_preserves_jvp_vjp_and_batching() -> None:
    size = 257
    plan = build_affine_plan(
        records=(
            AffineRecord(
                (size,),
                (1,),
                0,
                (1,),
                0,
                -1.25,
            ),
        ),
        output_size=size,
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=size,
        source_dtype=jnp.float16,
        result_dtype=jnp.float32,
    )
    source = jnp.linspace(-1, 1, size, dtype=jnp.float16)
    tangent = jnp.linspace(1, -1, size, dtype=jnp.float16)
    cotangent = jnp.linspace(-2, 3, size, dtype=jnp.float32)
    batch = jnp.stack((source, source * jnp.float16(0.5)))
    run = lambda value: _execute_map(
        value,
        plan=plan,
    )
    transformed = jax.jit(
        lambda value, direction, ct, batched: (
            jax.jvp(run, (value,), (direction,)),
            jax.vjp(run, value)[1](ct)[0],
            jax.vmap(run)(batched),
        )
    ).lower(source, tangent, cotangent, batch).compile()

    with _without_f16_f32_contiguous_simd():
        expected = transformed(source, tangent, cotangent, batch)
        jax.block_until_ready(expected)
    actual = transformed(source, tangent, cotangent, batch)
    jax.block_until_ready(actual)

    assert_bitwise_equal(actual[0][0], expected[0][0])
    assert_bitwise_equal(actual[0][1], expected[0][1])
    assert_bitwise_equal(actual[1], expected[1])
    assert_bitwise_equal(actual[2], expected[2])


@pytest.mark.parametrize(
    "plan_factory",
    (
        lambda: contiguous_dtype_plan(jnp.float32, -0.75, size=31),
        rank2_transpose_plan,
        rank4_tiled_plan,
        rank4_two_pair_plan,
        two_record_noncompact_plan,
        _mixed_broadcast_plan,
    ),
)
def test_generic_only_covers_retained_map_layout_and_dtype_families(
    plan_factory,
) -> None:
    plan = plan_factory()
    source = jnp.arange(plan.source_size, dtype=jnp.float32) - 5
    source = jnp.asarray(source, dtype=jnp.dtype(plan.source_dtype))

    with _generic_only():
        actual = jax.jit(
            lambda value: _execute_map(
                value,
                plan=plan,
            )
        )(source)
        actual.block_until_ready()

    assert_bitwise_equal(actual, execute_reference(source, plan))


def test_generic_only_covers_assign_accumulate_and_dynamic_scale() -> None:
    affine = rank2_transpose_plan(rows=5, columns=7)
    assign_plan = build_base_assign_plan(affine)
    accumulate_plan = build_base_accumulate_plan(affine)
    source = jnp.arange(affine.source_size, dtype=jnp.float32) - 4
    base = jnp.linspace(-3, 2, affine.output_size, dtype=jnp.float32)
    scale_plan = build_selected_scale_plan(
        contiguous_dtype_plan(jnp.complex64, None, size=19)
    )
    complex_base = (
        jnp.arange(19, dtype=jnp.float32)
        + 1j * jnp.arange(18, -1, -1, dtype=jnp.float32)
    ).astype(jnp.complex64)
    factor = jnp.asarray(1.25 - 0.75j, dtype=jnp.complex64)

    with _generic_only():
        actual_assign = jax.jit(
            lambda old, value: execute_update_assign(old, value, plan=assign_plan)
        )(base, source)
        actual_accumulate = jax.jit(
            lambda old, value: execute_update_accumulate(
                old,
                value,
                plan=accumulate_plan,
            )
        )(base, source)
        actual_scale = jax.jit(
            lambda old, value: execute_update_scale(old, value, plan=scale_plan)
        )(complex_base, factor)
        actual_alias = jax.jit(
            lambda old, value: _selected_scale_alias(
                old,
                value,
                plan=scale_plan,
            )
        )(complex_base, factor)
        jax.block_until_ready(
            (actual_assign, actual_accumulate, actual_scale, actual_alias)
        )

    assert_bitwise_equal(
        actual_assign,
        execute_base_assign_reference(base, source, assign_plan),
    )
    assert_bitwise_equal(
        actual_accumulate,
        execute_base_accumulate_reference(base, source, accumulate_plan),
    )
    assert_bitwise_equal(
        actual_scale,
        execute_selected_scale_reference(complex_base, factor, scale_plan),
    )
    assert_bitwise_equal(
        actual_alias,
        execute_selected_scale_reference(complex_base, factor, scale_plan),
    )


def test_generic_only_executes_rank_zero_base_updates() -> None:
    bound = _rank_zero_unit_plan()
    assign_plan = build_base_assign_plan(bound)
    accumulate_plan = build_base_accumulate_plan(bound)
    base = jnp.asarray([7], dtype=jnp.float32)
    source = jnp.asarray([3], dtype=jnp.float32)

    with _generic_only():
        actual_assign = jax.jit(
            lambda old, value: execute_update_assign(old, value, plan=assign_plan)
        )(base, source)
        actual_accumulate = jax.jit(
            lambda old, value: execute_update_accumulate(
                old,
                value,
                plan=accumulate_plan,
            )
        )(base, source)
        jax.block_until_ready((actual_assign, actual_accumulate))

    assert_bitwise_equal(
        actual_assign,
        execute_base_assign_reference(base, source, assign_plan),
    )
    assert_bitwise_equal(
        actual_accumulate,
        execute_base_accumulate_reference(base, source, accumulate_plan),
    )


def test_generic_only_executes_rank_zero_dynamic_scale_and_alias() -> None:
    plan = build_selected_scale_plan(_rank_zero_unit_plan())
    base = jnp.asarray([7], dtype=jnp.float32)
    alias_base = jnp.asarray([7], dtype=jnp.float32)
    factor = jnp.asarray(3, dtype=jnp.float32)

    with _generic_only():
        actual = jax.jit(
            lambda old, value: execute_update_scale(old, value, plan=plan)
        )(base, factor)
        actual_alias = jax.jit(
            lambda old, value: _selected_scale_alias(
                old,
                value,
                plan=plan,
            ),
            donate_argnums=(0,),
        )(alias_base, factor)
        jax.block_until_ready((actual, actual_alias))

    expected = execute_selected_scale_reference(base, factor, plan)
    assert_bitwise_equal(actual, expected)
    assert_bitwise_equal(actual_alias, expected)


def test_generic_only_preserves_jvp_and_vjp_programs() -> None:
    plan = _rank_plan(3)
    source = jnp.arange(plan.source_size, dtype=jnp.float32) - 3
    tangent = jnp.linspace(-1, 2, plan.source_size, dtype=jnp.float32)
    cotangent = jnp.linspace(2, -1, plan.output_size, dtype=jnp.float32)
    native = lambda value: _execute_map(
        value,
        plan=plan,
    )
    reference = lambda value: execute_reference(value, plan)

    with _generic_only():
        actual_jvp = jax.jit(
            lambda value, direction: jax.jvp(
                native,
                (value,),
                (direction,),
            )
        )(source, tangent)
        actual_vjp = jax.jit(
            lambda value, ct: jax.vjp(native, value)[1](ct)[0]
        )(source, cotangent)
        jax.block_until_ready((actual_jvp, actual_vjp))

    expected_jvp = jax.jvp(reference, (source,), (tangent,))
    expected_vjp = jax.vjp(reference, source)[1](cotangent)[0]
    assert_bitwise_equal(actual_jvp[0], expected_jvp[0])
    assert_bitwise_equal(actual_jvp[1], expected_jvp[1])
    assert_bitwise_equal(actual_vjp, expected_vjp)


def test_generic_only_preserves_broadcast_transpose_reduction() -> None:
    mixed = _mixed_broadcast_plan()
    plan = build_affine_plan(
        records=tuple(
            AffineRecord(
                logical_shape=record.logical_shape,
                source_strides=record.source_strides,
                source_offset=record.source_offset,
                destination_strides=record.destination_strides,
                destination_offset=record.destination_offset,
                scale=1.25,
                source_broadcast_axes=record.source_broadcast_axes,
            )
            for record in mixed.records
        ),
        output_size=mixed.output_size,
        coverage=mixed.coverage,
        source_size=mixed.source_size,
        source_dtype=jnp.float32,
        result_dtype=jnp.float32,
    )
    source = jnp.asarray([1.0, -2.0, 3.0], dtype=jnp.float32)
    cotangent = jnp.linspace(-2, 3, plan.output_size, dtype=jnp.float32)
    native = lambda value: _execute_map(
        value,
        plan=plan,
    )
    reference = lambda value: execute_reference(value, plan)

    with _generic_only():
        actual = jax.jit(
            lambda value, ct: jax.vjp(native, value)[1](ct)[0]
        )(source, cotangent)
        actual.block_until_ready()

    expected = jax.vjp(reference, source)[1](cotangent)[0]
    assert_bitwise_equal(actual, expected)


def test_generic_only_keeps_grouped_reduction_output_ownership() -> None:
    output_count = 32_768
    reduction_count = 4
    elements_per_record = output_count * reduction_count
    plan = build_affine_plan(
        records=(
            AffineRecord(
                logical_shape=(output_count, reduction_count),
                source_strides=(reduction_count, 1),
                source_offset=0,
                destination_strides=(1, 0),
                destination_offset=0,
                scale=2,
                reduction_axes=(1,),
            ),
            AffineRecord(
                logical_shape=(output_count, reduction_count),
                source_strides=(reduction_count, 1),
                source_offset=elements_per_record,
                destination_strides=(1, 0),
                destination_offset=0,
                scale=-3,
                reduction_axes=(1,),
            ),
        ),
        output_size=output_count,
        coverage=CompleteMode.PARTIAL_UNIQUE_ZERO_FILL,
        source_size=2 * elements_per_record,
        source_dtype=jnp.float32,
        result_dtype=jnp.float32,
        write_kind=StridedWriteKind.ACCUMULATE,
        reduction_kind=StridedReductionKind.SUM,
    )
    source = jnp.arange(plan.source_size, dtype=jnp.float32) - 5

    _set_native_worker_limit_for_tests(4)
    try:
        with _generic_only():
            actual = jax.jit(
                lambda value: _bind_grouped_native_structured_reduction(
                    value,
                    plan,
                )
            )(source)
            actual.block_until_ready()
            workers, available = _native_worker_counts_for_tests()
    finally:
        _set_native_worker_limit_for_tests(None)

    expected = (
        2 * source[:elements_per_record].reshape(output_count, -1).sum(axis=1)
        - 3 * source[elements_per_record:].reshape(output_count, -1).sum(axis=1)
    )
    assert available >= workers > 1
    assert_bitwise_equal(actual, expected)


def test_mixed_repeated_source_reduction_ad_matches_reference() -> None:
    output_count = 1_024
    reduction_count = 16
    plan = build_affine_plan(
        records=(
            AffineRecord(
                logical_shape=(output_count, reduction_count),
                source_strides=(1, 0),
                source_offset=0,
                destination_strides=(1, 0),
                destination_offset=0,
                scale=1.25 - 0.5j,
                source_broadcast_axes=(1,),
                reduction_axes=(1,),
            ),
        ),
        output_size=output_count,
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=output_count,
        source_dtype=jnp.float32,
        result_dtype=jnp.complex64,
        reduction_kind=StridedReductionKind.SUM,
    )
    source = jnp.linspace(-2, 3, output_count, dtype=jnp.float32)
    tangent = source * -0.25 + 1
    cotangent = (
        jnp.linspace(1, -1, output_count, dtype=jnp.float32)
        + 1j * jnp.linspace(-2, 0.5, output_count, dtype=jnp.float32)
    ).astype(jnp.complex64)
    execute = lambda value: bind_native_structured_reduction(value, plan)
    reference = lambda value: _bind_sequential_native_structured_reduction(
        value,
        plan,
    )

    expected_primal, expected_tangent = jax.jvp(
        reference,
        (source,),
        (tangent,),
    )
    expected_vjp = jax.vjp(reference, source)[1](cotangent)[0]
    actual_primal, actual_tangent = jax.jit(
        lambda value, direction: jax.jvp(
            execute,
            (value,),
            (direction,),
        )
    )(source, tangent)
    actual_vjp = jax.jit(
        lambda value, ct: jax.vjp(execute, value)[1](ct)[0]
    )(source, cotangent)

    assert_bitwise_equal(actual_primal, expected_primal)
    assert_bitwise_equal(actual_tangent, expected_tangent)
    assert_bitwise_equal(actual_vjp, expected_vjp)


def test_generic_only_parallelizes_large_special_layout_as_generic() -> None:
    rows, columns = 1_024, 512
    plan = rank2_transpose_plan(rows=rows, columns=columns)
    source = jnp.arange(plan.source_size, dtype=jnp.float32) * 0.25

    _set_native_worker_limit_for_tests(4)
    try:
        with _generic_only():
            _reset_native_call_count_for_tests()
            actual = jax.jit(
                lambda value: _execute_map(
                    value,
                    plan=plan,
                )
            )(source)
            actual.block_until_ready()
            workers, available = _native_worker_counts_for_tests()
    finally:
        _set_native_worker_limit_for_tests(None)

    expected = 1.25 * source.reshape(columns, rows).T.reshape(-1)
    assert _native_call_count_for_tests() == 1
    assert available >= workers > 1
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def test_generic_only_parallelizes_one_long_affine_row() -> None:
    plan = contiguous_dtype_plan(
        jnp.complex64,
        0.75 - 0.5j,
        size=(1 << 19) + 3,
    )
    source = _dtype_values(jnp.complex64, plan.source_size)

    _set_native_worker_limit_for_tests(4)
    try:
        with _generic_only():
            _reset_native_call_count_for_tests()
            actual = jax.jit(
                lambda value: _execute_map(
                    value,
                    plan=plan,
                )
            )(source)
            actual.block_until_ready()
            workers, available = _native_worker_counts_for_tests()
    finally:
        _set_native_worker_limit_for_tests(None)

    expected = execute_reference(source, plan)
    assert _native_call_count_for_tests() == 1
    assert available >= workers > 1
    assert_bitwise_equal(actual, expected)


@pytest.mark.parametrize(
    "scale",
    (
        complex(0.0, -0.0),
        complex(.125, 1.0),
        complex(2.5, 3.5),
        complex(-1.25, 0.75),
    ),
)
def test_generic_only_complex_contiguous_rows_match_finite_values(
    scale: complex,
) -> None:
    plan = build_affine_plan(
        records=(
            AffineRecord(
                logical_shape=(3, 9),
                source_strides=(-11, 1),
                source_offset=22,
                destination_strides=(9, 1),
                destination_offset=0,
                scale=scale,
            ),
        ),
        output_size=27,
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=31,
        source_dtype=jnp.complex64,
        result_dtype=jnp.complex64,
    )
    source = _complex_finite_values(plan.source_size)

    with _generic_only():
        actual = jax.jit(
            lambda value: _execute_map(
                value,
                plan=plan,
            )
        )(source)
        actual.block_until_ready()

    np.testing.assert_allclose(actual, execute_reference(source, plan))


@pytest.mark.parametrize(
    "factor",
    (
        complex(0.0, -0.0),
        complex(.125, 1.0),
        complex(2.5, 3.5),
        complex(-1.25, 0.75),
    ),
)
def test_generic_only_dynamic_complex_rows_match_finite_values(
    factor: complex,
) -> None:
    affine = build_affine_plan(
        records=(
            AffineRecord(
                logical_shape=(3, 9),
                source_strides=(-11, 1),
                source_offset=22,
                destination_strides=(-11, 1),
                destination_offset=22,
            ),
        ),
        output_size=31,
        coverage=CompleteMode.PARTIAL_UNIQUE_ZERO_FILL,
        source_size=31,
        source_dtype=jnp.complex64,
        result_dtype=jnp.complex64,
    )
    plan = build_selected_scale_plan(affine)
    base = _complex_finite_values(affine.source_size)
    factor_array = jnp.asarray(factor, dtype=jnp.complex64)

    with _generic_only():
        actual = jax.jit(
            lambda old, value: execute_update_scale(old, value, plan=plan)
        )(base, factor_array)
        actual.block_until_ready()

    np.testing.assert_allclose(
        actual,
        execute_selected_scale_reference(base, factor_array, plan),
    )


def test_contiguous_inner_complex_rows_reuse_vector_operation() -> None:
    rows = 128
    columns = 257
    source_pitch = 512
    plan = build_affine_plan(
        records=(
            AffineRecord(
                logical_shape=(rows, columns),
                source_strides=(source_pitch, 1),
                source_offset=0,
                destination_strides=(columns, 1),
                destination_offset=0,
                scale=-1.25 + 0.75j,
            ),
        ),
        output_size=rows * columns,
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=rows * source_pitch,
        source_dtype=jnp.complex64,
        result_dtype=jnp.complex64,
    )
    source = _complex_finite_values(plan.source_size)
    actual = jax.jit(
        lambda value: _execute_map(
            value,
            plan=plan,
        )
    )(source)

    np.testing.assert_allclose(actual, execute_reference(source, plan))


def test_contiguous_c64_f32_transpose_simd_matches_generic_finite_values() -> None:
    size = 1025
    plan = build_affine_plan(
        records=(
            AffineRecord(
                logical_shape=(size,),
                source_strides=(1,),
                source_offset=0,
                destination_strides=(1,),
                destination_offset=0,
                scale=-1.25 + 0.75j,
            ),
        ),
        output_size=size,
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=size,
        source_dtype=jnp.complex64,
        result_dtype=jnp.float32,
        scalar_kind=StridedScalarKind.JAX_TRANSPOSE,
    )
    source = _complex_finite_values(size)
    run = lambda value: _execute_map(
        value,
        plan=plan,
    )

    with _generic_only():
        expected = jax.jit(run)(source)
        expected.block_until_ready()
    with _strided_generated():
        actual = jax.jit(run)(source)
        actual.block_until_ready()

    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize(
    ("source_dtype", "result_dtype", "scalar_kind", "scale"),
    [
        (
            jnp.complex64,
            jnp.float32,
            StridedScalarKind.STATIC_SCALE_CAST,
            -0.75,
        ),
        (
            jnp.float32,
            jnp.complex64,
            StridedScalarKind.JAX_TRANSPOSE,
            -0.75,
        ),
    ],
)
def test_remaining_mixed_contiguous_simd_matches_generic_finite_values(
    source_dtype,
    result_dtype,
    scalar_kind: StridedScalarKind,
    scale: float,
) -> None:
    size = 1025
    plan = build_affine_plan(
        records=(AffineRecord((size,), (1,), 0, (1,), 0, scale),),
        output_size=size,
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=size,
        source_dtype=source_dtype,
        result_dtype=result_dtype,
        scalar_kind=scalar_kind,
    )
    components = np.asarray(
        [
            0.0,
            -0.0,
            1.25,
            -1.25,
            .125,
            np.float32(1e10),
            -np.float32(1e10),
            np.float32(1e-10),
        ],
        dtype=np.float32,
    )
    source = (
        _complex_finite_values(size)
        if source_dtype == jnp.complex64
        else jnp.asarray(np.resize(components, size))
    )
    run = lambda value: _execute_map(
        value,
        plan=plan,
    )

    with _generic_only():
        expected = jax.jit(run)(source)
        expected.block_until_ready()
    with _strided_generated():
        actual = jax.jit(run)(source)
        actual.block_until_ready()

    np.testing.assert_allclose(actual, expected)


def test_contiguous_inner_dynamic_complex_rows_support_fresh_and_alias() -> None:
    rows = 128
    columns = 257
    pitch = 512
    affine = build_affine_plan(
        records=(
            AffineRecord(
                logical_shape=(rows, columns),
                source_strides=(pitch, 1),
                source_offset=0,
                destination_strides=(pitch, 1),
                destination_offset=0,
            ),
        ),
        output_size=rows * pitch,
        coverage=CompleteMode.PARTIAL_UNIQUE_ZERO_FILL,
        source_size=rows * pitch,
        source_dtype=jnp.complex64,
        result_dtype=jnp.complex64,
    )
    plan = build_selected_scale_plan(affine)
    base = _complex_finite_values(affine.source_size)
    factor = jnp.asarray(-1.25 + 0.75j, dtype=jnp.complex64)
    expected = execute_selected_scale_reference(base, factor, plan)
    actual = jax.jit(
        lambda old, value: execute_update_scale(old, value, plan=plan)
    )(base, factor)
    alias = jax.jit(
        lambda old, value: _selected_scale_alias(old, value, plan=plan),
        donate_argnums=(0,),
    )(jnp.array(base), factor)

    np.testing.assert_allclose(actual, expected)
    np.testing.assert_allclose(alias, expected)


def test_f16_f32_vjp_leaf_matches_generic_finite_values() -> None:
    size = 17
    plan = build_affine_plan(
        records=(
            AffineRecord((size,), (1,), 0, (1,), 0, -1.25),
        ),
        output_size=size,
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=size,
        source_dtype=jnp.float16,
        result_dtype=jnp.float32,
    )
    source = jnp.ones((size,), dtype=jnp.float16)
    cotangent = jnp.asarray(
        [
            0.0,
            -0.0,
            np.nextafter(np.float32(0), np.float32(1)),
            -np.nextafter(np.float32(0), np.float32(1)),
            1.0,
            -1.0,
            1.25,
            -1.25,
            .125,
        ]
        * 2,
        dtype=jnp.float32,
    )[:size]
    run = lambda value: _execute_map(
        value, plan=plan
    )
    pullback = lambda cot: jax.vjp(run, source)[1](cot)[0]

    with _generic_only():
        generic = jax.jit(pullback).lower(cotangent).compile()
    with _strided_generated():
        generated = jax.jit(pullback).lower(cotangent).compile()

    expected = generic(cotangent)
    actual = generated(cotangent)
    np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=1e-7)
    hlo = str(generated.as_text()).lower()
    assert hlo.count("custom-call") == 1
    assert "compare" not in hlo
    assert "select" not in hlo


def test_c64_vjp_leaf_matches_generic_finite_values() -> None:
    size = 17
    plan = contiguous_dtype_plan(jnp.complex64, 1.0, size=size)
    source = jnp.ones((size,), dtype=jnp.complex64)
    real = np.asarray(
        [0.0, -0.0, 1.0, -1.0, 1.25, -1.25, .125],
        dtype=np.float32,
    )
    imaginary = np.asarray(
        [-0.0, 0.0, -1.0, 1.0, -1.25, 1.25, .125],
        dtype=np.float32,
    )
    complex_values = real.astype(np.complex64)
    complex_values.imag = imaginary
    cotangent = jnp.asarray(np.resize(complex_values, size))
    run = lambda value: _execute_map(
        value, plan=plan
    )
    pullback = lambda cot: jax.vjp(run, source)[1](cot)[0]

    with _generic_only():
        generic = jax.jit(pullback).lower(cotangent).compile()
    with _strided_generated():
        generated = jax.jit(pullback).lower(cotangent).compile()

    expected = generic(cotangent)
    actual = generated(cotangent)
    np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=1e-7)
