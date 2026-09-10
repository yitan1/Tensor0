from __future__ import annotations

from itertools import product

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
    _reset_native_call_count_for_tests,
    _set_native_disable_f32_c64_contiguous_simd_for_tests,
    _set_native_worker_limit_for_tests,
    native_available,
)
from tensor0._stride._ops._selected_scale import build_selected_scale_plan
from tests.stride._fixtures import execute_update_scale
from tensor0._stride._plan import build_affine_plan

from ._oracle import execute_reference, execute_selected_scale_reference


def _plan(
    shape: tuple[int, ...],
    source_strides: tuple[int, ...],
    destination_strides: tuple[int, ...],
    scale: complex,
) -> AffinePlan:
    size = int(np.prod(shape, dtype=np.int64))
    return build_affine_plan(
        records=(
            AffineRecord(
                shape,
                source_strides,
                0,
                destination_strides,
                0,
                scale,
            ),
        ),
        output_size=size,
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=size,
        source_dtype=jnp.complex64,
        result_dtype=jnp.complex64,
    )


def _compact(size: int, scale: complex) -> AffinePlan:
    return _plan((size,), (1,), (1,), scale)


def _rank2(
    rows: int,
    columns: int,
    scale: complex,
    *,
    forward: bool,
) -> AffinePlan:
    if forward:
        return _plan(
            (rows, columns),
            (1, rows),
            (columns, 1),
            scale,
        )
    return _plan(
        (rows, columns),
        (columns, 1),
        (1, rows),
        scale,
    )


def _complete_selected_scale(size: int):
    return build_selected_scale_plan(
        build_affine_plan(
            records=(
                AffineRecord((size,), (1,), 0, (1,), 0),
            ),
            output_size=size,
            coverage=CompleteMode.COMPLETE_UNIQUE,
            source_size=size,
            source_dtype=jnp.complex64,
            result_dtype=jnp.complex64,
        )
    )


def _values(size: int) -> jax.Array:
    values = jnp.arange(size, dtype=jnp.float32)
    real = (values % 4093) / 1024 - 2
    imaginary = 0.5 - (values % 4079) / 2048
    return jnp.asarray(real + 1j * imaginary, dtype=jnp.complex64)


def _assert_close(actual: object, expected: object) -> None:
    actual_array = np.asarray(actual, dtype=np.complex64)
    expected_array = np.asarray(expected, dtype=np.complex64)
    np.testing.assert_allclose(actual_array, expected_array, rtol=2e-6, atol=1e-6)


@pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")
def test_n6_mixed_compact_execution_uses_parallel_generic_range() -> None:
    size = (1 << 19) + 3
    plan = build_affine_plan(
        records=(
            AffineRecord(
                (size,),
                (1,),
                0,
                (1,),
                0,
                0.75 - 0.5j,
            ),
        ),
        output_size=size,
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=size,
        source_dtype=jnp.float32,
        result_dtype=jnp.complex64,
    )
    source = jnp.arange(size, dtype=jnp.float32) / 1024 - 2

    _set_native_worker_limit_for_tests(4)
    try:
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

    assert available >= workers > 1
    _assert_close(actual, execute_reference(source, plan))


@pytest.mark.parametrize(
    "scale",
    (
        complex(0.0, -0.0),
        complex(.125, 1.0),
        complex(2.5, 3.5),
        complex(-1.25, 0.75),
    ),
)
@pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")
def test_n6_promoted_complex_contiguous_simd_matches_scalar_accuracy(
    scale: complex,
) -> None:
    components = np.asarray(
        [
            0.0,
            -0.0,
            1.25,
            -2.5,
            .125,
            np.float32(1e10),
            -np.float32(1e10),
            np.float32(1e-10),
            np.float32(1e-20),
        ],
        dtype=np.float32,
    )
    source = jnp.asarray(np.resize(components, 73))
    plan = build_affine_plan(
        records=(
            AffineRecord((source.size,), (1,), 0, (1,), 0, scale),
        ),
        output_size=source.size,
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=source.size,
        source_dtype=jnp.float32,
        result_dtype=jnp.complex64,
    )
    executable = jax.jit(
        lambda value: _execute_map(value, plan=plan)
    ).lower(source).compile()

    _set_native_disable_f32_c64_contiguous_simd_for_tests(True)
    try:
        expected = executable(source)
        expected.block_until_ready()
    finally:
        _set_native_disable_f32_c64_contiguous_simd_for_tests(False)
    actual = executable(source)
    actual.block_until_ready()

    _assert_close(actual, expected)
    _assert_close(actual, execute_reference(source, plan))


@pytest.mark.parametrize(
    "plan",
    (
        _compact(131_075, 1.25 - 0.75j),
        _rank2(256, 256, 1.25 - 0.75j, forward=True),
        _rank2(517, 263, 1.25 - 0.75j, forward=False),
    ),
)
@pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")
def test_n6_complex_kernels_match_reference_for_finite_values(
    plan: AffinePlan,
) -> None:
    source = _values(plan.source_size)
    function = jax.jit(
        lambda value: _execute_map(value, plan=plan)
    )

    _reset_native_call_count_for_tests()
    actual = function(source)
    actual.block_until_ready()
    expected = execute_reference(source, plan)

    assert _native_call_count_for_tests() == 1
    _assert_close(actual, expected)


@pytest.mark.parametrize("forward", (False, True))
@pytest.mark.parametrize(
    "scale",
    (
        complex(0.0, -0.0),
        complex(.125, 1.0),
        complex(2.5, 3.5),
        complex(-1.25, 0.75),
    ),
)
@pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")
def test_n6_rank_two_simd_matches_finite_values_and_tails(
    forward: bool,
    scale: complex,
) -> None:
    plan = _rank2(17, 13, scale, forward=forward)
    components = np.asarray(
        [
            0.0,
            -0.0,
            1.25,
            -2.5,
            .125,
            np.float32(1e10),
            -np.float32(1e10),
            np.float32(1e-10),
        ],
        dtype=np.float32,
    )
    source = np.asarray(
        [complex(real, imaginary) for real, imaginary in product(components, repeat=2)],
        dtype=np.complex64,
    )
    source = jnp.asarray(np.resize(source, plan.source_size))

    actual = _execute_map(source, plan=plan)
    expected = execute_reference(source, plan)

    _assert_close(actual, expected)


@pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")
def test_n6_compact_simd_batch_jvp_and_vjp_match_reference() -> None:
    plan = _compact(65_539, 0.75 - 0.5j)
    source = jnp.stack((_values(plan.source_size), _values(plan.source_size) + 1j))
    tangent = source * jnp.complex64(0.25 + 0.125j)
    cotangent = source * jnp.complex64(-0.5 + 0.25j)
    native = lambda value: _execute_map(
        value,
        plan=plan,
    )
    reference = lambda value: execute_reference(value, plan)

    actual_jvp = jax.jit(lambda x, dx: jax.jvp(native, (x,), (dx,)))(
        source,
        tangent,
    )
    expected_jvp = jax.jvp(reference, (source,), (tangent,))
    for actual, expected in zip(actual_jvp, expected_jvp, strict=True):
        _assert_close(actual, expected)

    actual_vjp = jax.jit(lambda ct: jax.vjp(native, source)[1](ct)[0])(
        cotangent
    )
    expected_vjp = jax.vjp(reference, source)[1](cotangent)[0]
    _assert_close(actual_vjp, expected_vjp)


@pytest.mark.parametrize(
    "factor",
    (
        complex(0.0, -0.0),
        complex(.125, 1.0),
        complex(2.5, 3.5),
        complex(-1.25, 0.75),
    ),
)
@pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")
def test_n6_selected_scale_compact_simd_matches_finite_values_and_tails(
    factor: complex,
) -> None:
    plan = _complete_selected_scale(67)
    components = np.asarray(
        [
            0.0,
            -0.0,
            1.25,
            -2.5,
            .125,
            np.float32(1e10),
            -np.float32(1e10),
            np.float32(1e-10),
        ],
        dtype=np.float32,
    )
    values = np.asarray(
        [complex(real, imaginary) for real, imaginary in product(components, repeat=2)],
        dtype=np.complex64,
    )
    base = jnp.asarray(np.resize(values, 67))
    factor_array = jnp.asarray(factor, dtype=jnp.complex64)
    function = jax.jit(lambda old, value: execute_update_scale(old, value, plan=plan))

    _reset_native_call_count_for_tests()
    actual = function(base, factor_array)
    actual.block_until_ready()
    expected = execute_selected_scale_reference(base, factor_array, plan)

    assert _native_call_count_for_tests() == 1
    _assert_close(actual, expected)


@pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")
def test_n6_selected_scale_batched_factors_match_serial_and_parallel() -> None:
    size = 262_147
    plan = _complete_selected_scale(size)
    base = jnp.stack(tuple(_values(size) + index for index in range(3)))
    factor = jnp.asarray(
        [0.75 - 0.5j, -1.25 + 0.25j, 2 + 0.125j],
        dtype=jnp.complex64,
    )
    function = jax.jit(lambda old, value: execute_update_scale(old, value, plan=plan))

    _set_native_worker_limit_for_tests(1)
    try:
        sequential = function(base, factor)
        sequential.block_until_ready()
    finally:
        _set_native_worker_limit_for_tests(None)
    _set_native_worker_limit_for_tests(8)
    try:
        parallel = function(base, factor)
        parallel.block_until_ready()
        workers, available = _native_worker_counts_for_tests()
    finally:
        _set_native_worker_limit_for_tests(None)

    if available > 1:
        assert workers > 1
    _assert_close(parallel, sequential)
