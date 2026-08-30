from __future__ import annotations

from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensor0._stride import (
    CompleteMode,
    StridedCopyPlan,
    StridedCopyRecord,
    strided_copy,
)
from tensor0._stride._compiler import (
    CPU_POLICY_VERSION,
    CpuKernelKind,
    compile_plan,
)
from tensor0._stride._ffi import (
    _native_call_count_for_tests,
    _native_worker_counts_for_tests,
    _reset_native_call_count_for_tests,
    _set_native_worker_limit_for_tests,
    native_available,
)
from tensor0._stride._selected_scale import (
    compile_selected_scale_plan,
    selected_scale,
)
from tensor0._stride._plan import (
    UINT64_MAX,
    build_strided_copy_plan,
)
from tensor0._stride._native_lowering import lower_compiled_plan

from ._oracle import execute_reference, execute_selected_scale_reference


def _plan(
    shape: tuple[int, ...],
    source_strides: tuple[int, ...],
    destination_strides: tuple[int, ...],
    scale: complex,
) -> StridedCopyPlan:
    size = int(np.prod(shape, dtype=np.int64))
    return build_strided_copy_plan(
        records=(
            StridedCopyRecord(
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


def _compact(size: int, scale: complex) -> StridedCopyPlan:
    return _plan((size,), (1,), (1,), scale)


def _rank2(
    rows: int,
    columns: int,
    scale: complex,
    *,
    forward: bool,
) -> StridedCopyPlan:
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
    return compile_selected_scale_plan(
        build_strided_copy_plan(
            records=(
                StridedCopyRecord((size,), (1,), 0, (1,), 0, 1),
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


def _assert_bits_equal(actual: object, expected: object) -> None:
    actual_array = np.asarray(actual, dtype=np.complex64)
    expected_array = np.asarray(expected, dtype=np.complex64)
    np.testing.assert_array_equal(
        actual_array.view(np.uint32),
        expected_array.view(np.uint32),
    )


def test_n6_cpu_policy_classifies_complex_rank_two_kernels() -> None:
    assert CPU_POLICY_VERSION == 7
    forward = lower_compiled_plan(
        compile_plan(_rank2(17, 13, 0.75 - 0.5j, forward=True))
    )
    reverse = lower_compiled_plan(
        compile_plan(_rank2(17, 13, 0.75 - 0.5j, forward=False))
    )
    assert forward.records[0].kernel_kind is CpuKernelKind.RANK2_FORWARD
    assert reverse.records[0].kernel_kind is CpuKernelKind.RANK2_REVERSE


def test_n6_cpu_policy_keeps_scaled_compact_complex_serial() -> None:
    execution = lower_compiled_plan(
        compile_plan(_compact(131_075, 0.75 - 0.5j))
    )
    assert execution.parallel_minimum_bytes == UINT64_MAX


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
    plan: StridedCopyPlan,
) -> None:
    source = _values(plan.source_size)
    function = jax.jit(
        lambda value: strided_copy(value, plan=plan, native_required=True)
    )

    _reset_native_call_count_for_tests()
    actual = function(source)
    actual.block_until_ready()
    expected = execute_reference(source, plan)

    assert _native_call_count_for_tests() == 1
    _assert_bits_equal(actual, expected)


@pytest.mark.parametrize("forward", (False, True))
@pytest.mark.parametrize(
    "scale",
    (
        complex(0.0, -0.0),
        complex(np.nan, 1.0),
        complex(3e38, 3e38),
        complex(-1.25, 0.75),
    ),
)
@pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")
def test_n6_rank_two_simd_matches_special_value_bits(
    forward: bool,
    scale: complex,
) -> None:
    plan = _rank2(17, 13, scale, forward=forward)
    components = np.asarray(
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
        dtype=np.float32,
    )
    source = np.asarray(
        [complex(real, imaginary) for real, imaginary in product(components, repeat=2)],
        dtype=np.complex64,
    )
    source = jnp.asarray(np.resize(source, plan.source_size))

    actual = strided_copy(source, plan=plan, native_required=True)
    expected = execute_reference(source, plan)

    _assert_bits_equal(actual, expected)


@pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")
def test_n6_compact_simd_batch_jvp_and_vjp_match_reference() -> None:
    plan = _compact(65_539, 0.75 - 0.5j)
    source = jnp.stack((_values(plan.source_size), _values(plan.source_size) + 1j))
    tangent = source * jnp.complex64(0.25 + 0.125j)
    cotangent = source * jnp.complex64(-0.5 + 0.25j)
    native = lambda value: strided_copy(
        value,
        plan=plan,
        native_required=True,
    )
    reference = lambda value: execute_reference(value, plan)

    actual_jvp = jax.jit(lambda x, dx: jax.jvp(native, (x,), (dx,)))(
        source,
        tangent,
    )
    expected_jvp = jax.jvp(reference, (source,), (tangent,))
    for actual, expected in zip(actual_jvp, expected_jvp, strict=True):
        _assert_bits_equal(actual, expected)

    actual_vjp = jax.jit(lambda ct: jax.vjp(native, source)[1](ct)[0])(
        cotangent
    )
    expected_vjp = jax.vjp(reference, source)[1](cotangent)[0]
    _assert_bits_equal(actual_vjp, expected_vjp)


@pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")
def test_n6_scaled_compact_worker_sweep_retains_serial_execution() -> None:
    plan = _compact(131_075, 0.75 - 0.5j)
    source = jnp.stack(tuple(_values(plan.source_size) + index for index in range(8)))
    function = jax.jit(
        lambda value: strided_copy(value, plan=plan, native_required=True)
    )

    _set_native_worker_limit_for_tests(1)
    try:
        sequential = function(source)
        sequential.block_until_ready()
    finally:
        _set_native_worker_limit_for_tests(None)
    _set_native_worker_limit_for_tests(4)
    try:
        parallel = function(source)
        parallel.block_until_ready()
        workers, _ = _native_worker_counts_for_tests()
    finally:
        _set_native_worker_limit_for_tests(None)

    assert workers == 1
    _assert_bits_equal(parallel, sequential)


@pytest.mark.parametrize(
    "factor",
    (
        complex(0.0, -0.0),
        complex(np.nan, 1.0),
        complex(3e38, 3e38),
        complex(-1.25, 0.75),
    ),
)
@pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")
def test_n6_selected_scale_compact_simd_matches_special_value_bits(
    factor: complex,
) -> None:
    plan = _complete_selected_scale(67)
    components = np.asarray(
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
        dtype=np.float32,
    )
    values = np.asarray(
        [complex(real, imaginary) for real, imaginary in product(components, repeat=2)],
        dtype=np.complex64,
    )
    base = jnp.asarray(np.resize(values, 67))
    factor_array = jnp.asarray(factor, dtype=jnp.complex64)
    function = jax.jit(lambda old, value: selected_scale(old, value, plan=plan))

    _reset_native_call_count_for_tests()
    actual = function(base, factor_array)
    actual.block_until_ready()
    expected = execute_selected_scale_reference(base, factor_array, plan)

    assert _native_call_count_for_tests() == 1
    _assert_bits_equal(actual, expected)


@pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")
def test_n6_selected_scale_batched_factors_retain_serial_simd() -> None:
    size = 262_147
    plan = _complete_selected_scale(size)
    base = jnp.stack(tuple(_values(size) + index for index in range(3)))
    factor = jnp.asarray(
        [0.75 - 0.5j, -1.25 + 0.25j, 2 + 0.125j],
        dtype=jnp.complex64,
    )
    function = jax.jit(lambda old, value: selected_scale(old, value, plan=plan))

    _set_native_worker_limit_for_tests(1)
    try:
        sequential = function(base, factor)
        sequential.block_until_ready()
    finally:
        _set_native_worker_limit_for_tests(None)
    _set_native_worker_limit_for_tests(4)
    try:
        parallel = function(base, factor)
        parallel.block_until_ready()
        workers, _ = _native_worker_counts_for_tests()
    finally:
        _set_native_worker_limit_for_tests(None)

    assert workers == 1
    _assert_bits_equal(parallel, sequential)


@pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")
def test_n6_rank_two_worker_sweep_retains_parallel_ranges() -> None:
    plan = _rank2(517, 263, 0.75 - 0.5j, forward=False)
    source = _values(plan.source_size)
    function = jax.jit(
        lambda value: strided_copy(value, plan=plan, native_required=True)
    )

    _set_native_worker_limit_for_tests(1)
    try:
        sequential = function(source)
        sequential.block_until_ready()
    finally:
        _set_native_worker_limit_for_tests(None)
    _set_native_worker_limit_for_tests(8)
    try:
        parallel = function(source)
        parallel.block_until_ready()
        workers, available = _native_worker_counts_for_tests()
    finally:
        _set_native_worker_limit_for_tests(None)

    assert workers == min(available, 5)
    _assert_bits_equal(parallel, sequential)
