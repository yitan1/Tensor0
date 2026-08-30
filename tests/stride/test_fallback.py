from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensor0._stride import (
    CompleteMode,
    StridedCopyRecord,
    strided_copy,
)
import tensor0._stride._ffi as ffi_module
from tensor0._stride._errors import NO_ROUTE_PREFIX
from tensor0._stride._ffi import (
    _native_call_count_for_tests,
    _reset_native_call_count_for_tests,
    native_available,
)
from tensor0._stride._plan import (
    build_strided_copy_plan,
)

from ._fixtures import (
    contiguous_dtype_plan,
    dtype_transpose_plan,
    large_contiguous_plan,
    many_tiny_balanced_plan,
    two_record_noncompact_plan,
)
from ._oracle import execute_reference


def test_records_interface_builds_and_executes_the_complete_unique_plan() -> None:
    plan = two_record_noncompact_plan()
    source = jnp.arange(plan.source_size, dtype=jnp.float32)
    function = jax.jit(
        lambda value: strided_copy(
            value,
            records=plan.records,
            output_size=plan.output_size,
            result_dtype=jnp.float32,
        )
    )

    actual = function(source)

    np.testing.assert_array_equal(
        np.asarray(actual),
        np.asarray(execute_reference(source, plan)),
    )


def test_records_interface_rejects_an_explicit_plan_at_the_same_time() -> None:
    plan = two_record_noncompact_plan()
    source = jnp.arange(plan.source_size, dtype=jnp.float32)

    with pytest.raises(TypeError, match="either plan or records"):
        strided_copy(
            source,
            plan=plan,
            records=plan.records,
            output_size=plan.output_size,
        )


@pytest.mark.skipif(not native_available(), reason="native CPU stride is unavailable")
def test_auto_selection_uses_native_for_general_maps_and_measured_large_class() -> (
    None
):
    small = two_record_noncompact_plan()
    small_source = jnp.arange(small.source_size, dtype=jnp.float32)
    small_compiled = jax.jit(lambda value: strided_copy(value, plan=small))

    _reset_native_call_count_for_tests()
    small_actual = small_compiled(small_source)
    small_actual.block_until_ready()

    np.testing.assert_allclose(
        np.asarray(small_actual),
        np.asarray(execute_reference(small_source, small)),
    )
    assert _native_call_count_for_tests() == 1
    small_hlo = str(small_compiled.lower(small_source).compiler_ir()).lower()
    assert small_hlo.count("custom_call") == 1
    assert "gather" not in small_hlo
    assert "scatter" not in small_hlo

    large = large_contiguous_plan()
    large_source = jnp.arange(large.source_size, dtype=jnp.float32)
    large_compiled = jax.jit(lambda value: strided_copy(value, plan=large))

    _reset_native_call_count_for_tests()
    large_actual = large_compiled(large_source)
    large_actual.block_until_ready()

    np.testing.assert_allclose(
        np.asarray(large_actual),
        np.asarray(execute_reference(large_source, large)),
    )
    assert _native_call_count_for_tests() == 1
    assert (
        str(large_compiled.lower(large_source).compiler_ir())
        .lower()
        .count("custom_call")
        == 1
    )


@pytest.mark.skipif(not native_available(), reason="native CPU stride is unavailable")
def test_auto_selection_accounts_for_direct_and_vmap_batch_work() -> None:
    bound = contiguous_dtype_plan(jnp.float32, 1.25, size=4_096)
    source = jnp.arange(256 * bound.source_size, dtype=jnp.float32).reshape(
        256,
        bound.source_size,
    )
    functions = (
        jax.jit(lambda value: strided_copy(value, plan=bound)),
        jax.jit(jax.vmap(lambda value: strided_copy(value, plan=bound))),
    )

    for function in functions:
        hlo = str(function.lower(source).compiler_ir()).lower()
        assert hlo.count("custom_call") == 1
        assert "gather" not in hlo
        assert "scatter" not in hlo

        _reset_native_call_count_for_tests()
        actual = function(source)
        actual.block_until_ready()
        assert _native_call_count_for_tests() == 1
        np.testing.assert_allclose(
            np.asarray(actual),
            np.asarray(source * jnp.float32(1.25)),
        )


@pytest.mark.skipif(not native_available(), reason="native CPU stride is unavailable")
def test_auto_selection_protects_batched_complex64_regression() -> None:
    bound = contiguous_dtype_plan(jnp.complex64, 1, size=131_072)
    real = jnp.arange(8 * bound.source_size, dtype=jnp.float32).reshape(
        8,
        bound.source_size,
    )
    source = (real + 1j * (1 - real)).astype(jnp.complex64)
    expected = execute_reference(source, bound)
    automatic = jax.jit(jax.vmap(lambda value: strided_copy(value, plan=bound)))

    _reset_native_call_count_for_tests()
    actual = automatic(source)
    actual.block_until_ready()

    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    assert _native_call_count_for_tests() == 0
    assert "custom_call" not in str(automatic.lower(source).compiler_ir()).lower()

    direct = jax.jit(lambda value: strided_copy(value, plan=bound))
    _reset_native_call_count_for_tests()
    direct_actual = direct(source)
    direct_actual.block_until_ready()
    np.testing.assert_array_equal(
        np.asarray(direct_actual),
        np.asarray(expected),
    )
    assert _native_call_count_for_tests() == 0

    forced = jax.jit(
        jax.vmap(
            lambda value: strided_copy(
                value,
                plan=bound,
                native_required=True,
            )
        )
    )
    _reset_native_call_count_for_tests()
    forced_actual = forced(source)
    forced_actual.block_until_ready()
    np.testing.assert_array_equal(
        np.asarray(forced_actual),
        np.asarray(expected),
    )
    assert _native_call_count_for_tests() == 1
    assert str(forced.lower(source).compiler_ir()).lower().count("custom_call") == 1


@pytest.mark.skipif(not native_available(), reason="native CPU stride is unavailable")
def test_auto_selection_uses_measured_many_record_class() -> None:
    bound = many_tiny_balanced_plan()
    source = jnp.arange(bound.source_size, dtype=jnp.float32)
    compiled = jax.jit(lambda value: strided_copy(value, plan=bound))

    _reset_native_call_count_for_tests()
    actual = compiled(source)
    actual.block_until_ready()

    np.testing.assert_array_equal(
        np.asarray(actual),
        np.asarray(execute_reference(source, bound)),
    )
    assert _native_call_count_for_tests() == 1
    hlo = str(compiled.lower(source).compiler_ir()).lower()
    assert hlo.count("custom_call") == 1
    assert "gather" not in hlo
    assert "scatter" not in hlo


def test_excessive_native_rank_uses_portable_stablehlo() -> None:
    shape = (2,) * 9
    strides = tuple(2 ** (8 - axis) for axis in range(9))
    record = StridedCopyRecord(shape, strides, 0, strides, 0, 1.0)
    bound = build_strided_copy_plan(
        records=(record,),
        output_size=512,
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=512,
        source_dtype=jnp.float32,
        result_dtype=jnp.float32,
    )
    source = jnp.arange(bound.source_size, dtype=jnp.float32)

    _reset_native_call_count_for_tests()
    compiled = jax.jit(lambda value: strided_copy(value, plan=bound))
    stablehlo = str(compiled.lower(source).compiler_ir()).lower()
    actual = compiled(source)

    np.testing.assert_array_equal(
        np.asarray(actual),
        np.asarray(execute_reference(source, bound)),
    )
    assert "custom_call" not in stablehlo
    assert "gather" not in stablehlo
    assert "scatter" not in stablehlo
    assert _native_call_count_for_tests() == 0


@pytest.mark.parametrize(
    ("source_dtype", "result_dtype"),
    [
        (jnp.float16, jnp.float32),
        (jnp.float32, jnp.complex64),
        (jnp.complex64, jnp.float32),
    ],
)
@pytest.mark.filterwarnings("ignore:Casting complex values to real")
def test_mixed_inexact_dtype_primal_jvp_and_vjp_match_reference(
    source_dtype: jnp.dtype,
    result_dtype: jnp.dtype,
) -> None:
    template = two_record_noncompact_plan()
    bound = build_strided_copy_plan(
        records=template.records,
        output_size=template.output_size,
        coverage=template.coverage,
        source_size=template.source_size,
        source_dtype=source_dtype,
        result_dtype=result_dtype,
    )
    base = jnp.arange(3 * bound.source_size, dtype=jnp.float32).reshape(
        3,
        bound.source_size,
    )
    source = base.astype(source_dtype)
    if source_dtype == jnp.complex64:
        source = source * jnp.complex64(1 + 0.25j)
    tangent = source / 8
    function = lambda value: strided_copy(value, plan=bound)
    reference = lambda value: execute_reference(value, bound)
    actual_primal, actual_tangent = jax.jvp(
        function,
        (source,),
        (tangent,),
    )
    expected_primal, expected_tangent = jax.jvp(
        reference,
        (source,),
        (tangent,),
    )
    np.testing.assert_allclose(
        np.asarray(actual_primal),
        np.asarray(expected_primal),
    )
    np.testing.assert_allclose(
        np.asarray(actual_tangent),
        np.asarray(expected_tangent),
    )

    cotangent = jnp.ones(actual_primal.shape, dtype=result_dtype)
    actual_vjp = jax.vjp(function, source)[1](cotangent)[0]
    expected_vjp = jax.vjp(reference, source)[1](cotangent)[0]
    np.testing.assert_allclose(
        np.asarray(actual_vjp),
        np.asarray(expected_vjp),
    )
    hlo = str(jax.jit(function).lower(source).compiler_ir()).lower()
    assert hlo.count("custom_call") == 1
    assert "gather" not in hlo
    assert "scatter" not in hlo


def test_large_unique_mixed_dtype_vjp_uses_native_without_addresses() -> None:
    source_size = 100_000_000
    copied_elements = 4
    record = StridedCopyRecord(
        (copied_elements,),
        (1,),
        0,
        (1,),
        0,
        1.0,
    )
    bound = build_strided_copy_plan(
        records=(record,),
        output_size=copied_elements,
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=source_size,
        source_dtype=jnp.float16,
        result_dtype=jnp.float32,
    )
    source_shape = (source_size,)

    source = jax.ShapeDtypeStruct(source_shape, jnp.float16)
    cotangent = jax.ShapeDtypeStruct((copied_elements,), jnp.float32)
    vjp = jax.jit(
        lambda value, cot: jax.vjp(
            lambda item: strided_copy(item, plan=bound),
            value,
        )[1](cot)[0]
    )
    hlo = str(vjp.lower(source, cotangent).compiler_ir()).lower()
    assert hlo.count("custom_call") == 1
    assert "gather" not in hlo
    assert "scatter" not in hlo


def test_rank9_mixed_plan_uses_portable_stablehlo_automatically() -> None:
    shape = (2,) * 9
    strides = tuple(2 ** (8 - axis) for axis in range(9))
    record = StridedCopyRecord(shape, strides, 0, strides, 0, 1.0 + 0.5j)
    bound = build_strided_copy_plan(
        records=(record,),
        output_size=512,
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=512,
        source_dtype=jnp.float32,
        result_dtype=jnp.complex64,
    )
    source = jnp.arange(bound.source_size, dtype=jnp.float32)
    function = jax.jit(lambda value: strided_copy(value, plan=bound))

    _reset_native_call_count_for_tests()
    stablehlo = str(function.lower(source).compiler_ir()).lower()
    actual = function(source)

    np.testing.assert_array_equal(
        np.asarray(actual),
        np.asarray(execute_reference(source, bound)),
    )
    assert "custom_call" not in stablehlo
    assert "gather" not in stablehlo
    assert "scatter" not in stablehlo
    assert _native_call_count_for_tests() == 0
    with pytest.raises(ValueError, match="rank exceeds native limit"):
        strided_copy(source, plan=bound, native_required=True)


def test_excessive_rank_direct_and_vmap_share_portable_lowering() -> None:
    shape = (2,) * 9
    strides = tuple(2 ** (8 - axis) for axis in range(9))
    record = StridedCopyRecord(shape, strides, 0, strides, 0, 1.0)
    bound = build_strided_copy_plan(
        records=(record,),
        output_size=512,
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=512,
        source_dtype=jnp.float32,
        result_dtype=jnp.float32,
    )
    source = jax.ShapeDtypeStruct((6_000, bound.source_size), jnp.float32)
    functions = (
        jax.jit(lambda value: strided_copy(value, plan=bound)),
        jax.jit(jax.vmap(lambda value: strided_copy(value, plan=bound))),
    )

    for function in functions:
        stablehlo = str(function.lower(source).compiler_ir()).lower()
        assert "custom_call" not in stablehlo
        assert "gather" not in stablehlo
        assert "scatter" not in stablehlo


@pytest.mark.skipif(not native_available(), reason="native CPU stride is unavailable")
def test_auto_selection_routes_many_single_element_records_to_native() -> None:
    record_count = 1_024
    records = tuple(
        StridedCopyRecord((1,), (1,), index, (1,), index, 1.0)
        for index in range(record_count)
    )
    bound = build_strided_copy_plan(
        records=records,
        output_size=record_count,
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=record_count,
        source_dtype=jnp.float32,
        result_dtype=jnp.float32,
    )
    source = jnp.arange(record_count, dtype=jnp.float32)
    compiled = jax.jit(lambda value: strided_copy(value, plan=bound))
    hlo = str(compiled.lower(source).compiler_ir()).lower()

    assert hlo.count("custom_call") == 1
    assert "gather" not in hlo
    assert "scatter" not in hlo
    np.testing.assert_array_equal(
        np.asarray(compiled(source)),
        np.asarray(source),
    )


def test_missing_transformation_primitive_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bound = two_record_noncompact_plan()
    source = jnp.arange(bound.source_size, dtype=jnp.float32)
    monkeypatch.setattr(
        ffi_module,
        "_STRIDE_PRIMITIVE",
        None,
    )

    with pytest.raises(
        RuntimeError,
        match=f"{NO_ROUTE_PREFIX} native_primitive_unavailable",
    ):
        strided_copy(source, plan=bound)
    with pytest.raises(RuntimeError, match="transformation-safe"):
        strided_copy(source, plan=bound, native_required=True)


@pytest.mark.skipif(not native_available(), reason="native CPU stride is unavailable")
def test_forced_native_rejects_single_non_cpu_lowering() -> None:
    bound = contiguous_dtype_plan(jnp.float32, 1, size=4_096)
    source = jnp.zeros(bound.source_size, dtype=jnp.float32)
    traced = jax.jit(
        lambda value: strided_copy(
            value,
            plan=bound,
            native_required=True,
        )
    ).trace(source)

    with pytest.raises(RuntimeError, match="only supports the CPU platform"):
        traced.lower(lowering_platforms=("tpu",))


def test_native_build_runtime_mismatch_disables_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ffi_module._native,
        "_stride_ffi_build_versions",
        lambda: ("mismatch", "mismatch"),
    )
    bound = two_record_noncompact_plan()
    source = jnp.arange(bound.source_size, dtype=jnp.float32)

    assert not ffi_module.native_available()
    with pytest.raises(
        RuntimeError,
        match=f"{NO_ROUTE_PREFIX} native_cpu_executor_unavailable",
    ):
        strided_copy(source, plan=bound)
    with pytest.raises(RuntimeError, match="native CPU path"):
        strided_copy(source, plan=bound, native_required=True)


@pytest.mark.skipif(not native_available(), reason="native CPU stride is unavailable")
def test_large_batched_noncompact_complex64_uses_native_without_addresses() -> (
    None
):
    bound = dtype_transpose_plan(
        jnp.complex64,
        0.75 - 0.5j,
        rows=1_025,
        columns=1_024,
    )
    source = jnp.zeros((2, bound.source_size), dtype=jnp.complex64)
    function = jax.jit(lambda value: strided_copy(value, plan=bound))
    hlo = str(function.lower(source).compiler_ir()).lower()

    assert hlo.count("custom_call") == 1
    assert "gather" not in hlo
    assert "scatter" not in hlo
    _reset_native_call_count_for_tests()
    actual = function(source)
    actual.block_until_ready()
    assert _native_call_count_for_tests() == 1
    np.testing.assert_array_equal(np.asarray(actual), np.zeros_like(actual))


def test_native_unavailable_positive_affine_uses_portable_stablehlo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ffi_module._native,
        "_stride_ffi_build_versions",
        lambda: ("mismatch", "mismatch"),
    )
    bound = dtype_transpose_plan(
        jnp.float32,
        1,
        rows=1_025,
        columns=1_024,
    )
    source = jnp.zeros(bound.source_size, dtype=jnp.float32)
    function = jax.jit(lambda value: strided_copy(value, plan=bound))
    stablehlo = str(function.lower(source).compiler_ir()).lower()

    assert "custom_call" not in stablehlo
    assert "gather" not in stablehlo
    assert "scatter" not in stablehlo
