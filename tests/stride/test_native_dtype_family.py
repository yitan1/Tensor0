from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax.typing import DTypeLike
import numpy as np
import pytest

from tensor0._stride import (
    CompleteMode,
    StridedCopyRecord,
    strided_copy,
)
from tensor0._stride._ffi import (
    _native_call_count_for_tests,
    _reset_native_call_count_for_tests,
    native_available,
)
from tensor0._stride._selected_scale import (
    compile_selected_scale_plan,
    selected_scale,
)
from tensor0._stride._update import (
    base_accumulate,
    base_assign,
    compile_base_accumulate_plan,
    compile_base_assign_plan,
)
from tensor0._stride._plan import (
    build_strided_copy_plan,
)

from ._fixtures import (
    contiguous_dtype_plan,
    dtype_transpose_plan,
    selected_scale_plan,
)
from ._oracle import (
    assert_bitwise_equal,
    execute_base_accumulate_reference,
    execute_base_assign_reference,
    execute_reference,
    execute_selected_scale_reference,
)


_NEW_SAME_DTYPE_CASES = (
    (jnp.bool_, False),
    (jnp.int8, -3),
    (jnp.int16, -3),
    (jnp.int64, -3),
    (jnp.uint8, 3),
    (jnp.uint16, 3),
    (jnp.uint32, 3),
    (jnp.uint64, 3),
    (jnp.float64, -1.25),
    (jnp.complex128, 1.25 - 0.75j),
)


def _values(dtype: DTypeLike, size: int, *, shift: int = 0) -> jax.Array:
    resolved = jnp.dtype(dtype)
    indices = np.arange(size, dtype=np.int64) + shift
    if resolved == jnp.dtype(jnp.bool_):
        host = indices % 3 != 0
    elif jnp.issubdtype(resolved, jnp.unsignedinteger):
        host = (indices * 71 + 200).astype(np.dtype(resolved))
    elif jnp.issubdtype(resolved, jnp.signedinteger):
        host = (indices * 37 - 120).astype(np.dtype(resolved))
    elif jnp.issubdtype(resolved, jnp.complexfloating):
        real = indices.astype(np.float64) / 3 - 2
        host = (real + 1j * real[::-1]).astype(np.dtype(resolved))
    else:
        host = (indices.astype(np.float64) / 3 - 2).astype(np.dtype(resolved))
    return jnp.asarray(host, dtype=resolved)


@pytest.mark.parametrize(("dtype", "factor"), _NEW_SAME_DTYPE_CASES)
@pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")
def test_new_same_dtype_family_executes_map_update_and_selected_scale(
    dtype: DTypeLike,
    factor: bool | int | float | complex,
) -> None:
    with jax.enable_x64():
        resolved = jnp.dtype(dtype)
        static_scale = True if resolved == jnp.dtype(jnp.bool_) else factor
        source = _values(resolved, 35)

        map_plan = dtype_transpose_plan(resolved, static_scale)
        _reset_native_call_count_for_tests()
        mapped = jax.jit(
            lambda value: strided_copy(
                value,
                plan=map_plan,
                native_required=True,
            )
        )(source)
        mapped.block_until_ready()
        assert_bitwise_equal(mapped, execute_reference(source, map_plan))
        assert _native_call_count_for_tests() == 1

        update_bound = contiguous_dtype_plan(resolved, static_scale, size=35)
        base = _values(resolved, 35, shift=11)
        assign_plan = compile_base_assign_plan(update_bound)
        _reset_native_call_count_for_tests()
        assigned = jax.jit(
            lambda old, value: base_assign(old, value, plan=assign_plan)
        )(
            base,
            source,
        )
        assigned.block_until_ready()
        assert_bitwise_equal(
            assigned,
            execute_base_assign_reference(base, source, assign_plan),
        )
        assert _native_call_count_for_tests() == 1

        if resolved != jnp.dtype(jnp.bool_):
            accumulate_plan = compile_base_accumulate_plan(update_bound)
            _reset_native_call_count_for_tests()
            accumulated = jax.jit(
                lambda old, value: base_accumulate(
                    old,
                    value,
                    plan=accumulate_plan,
                )
            )(
                base,
                source,
            )
            accumulated.block_until_ready()
            assert_bitwise_equal(
                accumulated,
                execute_base_accumulate_reference(base, source, accumulate_plan),
            )
            assert _native_call_count_for_tests() == 1

        scale_plan = compile_selected_scale_plan(selected_scale_plan(resolved))
        selected_base = _values(resolved, 50, shift=11)
        factor_array = jnp.asarray(factor, dtype=resolved)
        _reset_native_call_count_for_tests()
        scaled = jax.jit(
            lambda old, value: selected_scale(old, value, plan=scale_plan)
        )(selected_base, factor_array)
        scaled.block_until_ready()
        assert_bitwise_equal(
            scaled,
            execute_selected_scale_reference(
                selected_base,
                factor_array,
                scale_plan,
            ),
        )
        assert _native_call_count_for_tests() == 1


@pytest.mark.parametrize(
    ("source_dtype", "result_dtype"),
    (
        (jnp.int32, jnp.bool_),
        (jnp.int32, jnp.int8),
        (jnp.int32, jnp.int16),
        (jnp.uint32, jnp.uint8),
    ),
)
@pytest.mark.parametrize(
    ("compile_plan", "operation", "oracle"),
    (
        (compile_base_assign_plan, base_assign, execute_base_assign_reference),
        (
            compile_base_accumulate_plan,
            base_accumulate,
            execute_base_accumulate_reference,
        ),
    ),
)
@pytest.mark.filterwarnings("ignore:scatter inputs have incompatible types")
@pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")
def test_integer_narrowing_base_updates_remain_native(
    source_dtype: DTypeLike,
    result_dtype: DTypeLike,
    compile_plan: Callable,
    operation: Callable,
    oracle: Callable,
) -> None:
    record = StridedCopyRecord((17,), (1,), 0, (1,), 0, 1)
    bound = build_strided_copy_plan(
        records=(record,),
        output_size=17,
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=17,
        source_dtype=source_dtype,
        result_dtype=result_dtype,
    )
    plan = compile_plan(bound)
    source = _values(source_dtype, 17)
    base = _values(result_dtype, 17, shift=7)

    _reset_native_call_count_for_tests()
    actual = jax.jit(lambda old, value: operation(old, value, plan=plan))(
        base,
        source,
    )
    actual.block_until_ready()

    assert_bitwise_equal(actual, oracle(base, source, plan))
    assert _native_call_count_for_tests() == 1


@pytest.mark.parametrize(
    ("source_dtype", "result_dtype", "scale"),
    (
        (jnp.float64, jnp.float64, -1.25),
        (jnp.complex128, jnp.complex128, 1.25 - 0.75j),
        (jnp.float64, jnp.complex128, 0.5 + 0.75j),
        (jnp.complex128, jnp.float64, -0.75),
    ),
)
@pytest.mark.filterwarnings("ignore:Casting complex values to real")
@pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")
def test_wide_inexact_broadcast_vjp_uses_native_structured_reduction(
    source_dtype: DTypeLike,
    result_dtype: DTypeLike,
    scale: float | complex,
) -> None:
    with jax.enable_x64():
        record = StridedCopyRecord(
            (4, 3),
            (0, 1),
            0,
            (3, 1),
            0,
            scale,
            (0,),
        )
        plan = build_strided_copy_plan(
            records=(record,),
            output_size=12,
            coverage=CompleteMode.COMPLETE_UNIQUE,
            source_size=3,
            source_dtype=source_dtype,
            result_dtype=result_dtype,
        )
        source = _values(source_dtype, 3)
        cotangent = _values(result_dtype, 12, shift=5)
        native_pullback = jax.vjp(
            lambda value: strided_copy(value, plan=plan, native_required=True),
            source,
        )[1]
        reference_pullback = jax.vjp(
            lambda value: execute_reference(value, plan),
            source,
        )[1]

        _reset_native_call_count_for_tests()
        actual = jax.jit(native_pullback)(cotangent)[0]
        actual.block_until_ready()
        expected = reference_pullback(cotangent)[0]

        assert_bitwise_equal(actual, expected)
        assert _native_call_count_for_tests() == 1
        hlo = str(jax.jit(native_pullback).lower(cotangent).compiler_ir()).lower()
        assert "structured_reduction" in hlo
        assert "gather" not in hlo
        assert "scatter" not in hlo


def _assert_wide_float_bits(actual: object, expected: object) -> None:
    actual_array = np.asarray(actual)
    expected_array = np.asarray(expected)
    assert actual_array.dtype == expected_array.dtype
    if actual_array.dtype.kind == "c":
        actual_components = actual_array.view(np.float64)
        expected_components = expected_array.view(np.float64)
    else:
        actual_components = actual_array
        expected_components = expected_array
    actual_bits = actual_components.view(np.uint64).copy()
    expected_bits = expected_components.view(np.uint64).copy()
    actual_bits[np.isnan(actual_components)] = 0
    expected_bits[np.isnan(expected_components)] = 0
    np.testing.assert_array_equal(actual_bits, expected_bits)


@pytest.mark.parametrize(
    "scale",
    (
        complex(1.25, -0.75),
        complex(np.nan, 1.0),
        complex(np.inf, -0.0),
        complex(1.0e308, 1.0e308),
    ),
)
@pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")
def test_complex128_special_values_match_jax_scalar_oracle(scale: complex) -> None:
    with jax.enable_x64():
        values = jnp.asarray(
            [0.0, -0.0, np.inf, -np.inf, np.nan, 2.0, -2.0],
            dtype=jnp.float64,
        )
        cotangent = jnp.asarray(
            [
                0.0 - 0.0j,
                -0.0 + 0.0j,
                np.inf + 0.0j,
                0.0 + np.inf * 1j,
                np.nan + 1.0j,
                1.0e308 + 1.0e308j,
                -1.0e308 - 1.0e308j,
            ],
            dtype=jnp.complex128,
        )
        template = contiguous_dtype_plan(jnp.complex128, scale, size=values.size)
        mixed = build_strided_copy_plan(
            records=template.records,
            output_size=template.output_size,
            coverage=template.coverage,
            source_size=template.source_size,
            source_dtype=jnp.float64,
            result_dtype=jnp.complex128,
        )
        native = lambda value: strided_copy(
            value,
            plan=mixed,
            native_required=True,
        )
        reference = lambda value: execute_reference(value, mixed)

        actual_forward, actual_reverse = jax.jit(
            lambda value, cot: (
                native(value),
                jax.vjp(native, value)[1](cot)[0],
            )
        )(values, cotangent)
        expected_forward = reference(values)
        expected_reverse = jax.vjp(reference, values)[1](cotangent)[0]

        _assert_wide_float_bits(actual_forward, expected_forward)
        _assert_wide_float_bits(actual_reverse, expected_reverse)

        same_dtype = contiguous_dtype_plan(
            jnp.complex128,
            scale,
            size=cotangent.size,
        )
        actual_same = jax.jit(
            lambda value: strided_copy(
                value,
                plan=same_dtype,
                native_required=True,
            )
        )(cotangent)
        expected_same = execute_reference(cotangent, same_dtype)
        _assert_wide_float_bits(actual_same, expected_same)
