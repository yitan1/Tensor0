from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensor0._stride._plan import CompleteMode, AffineRecord
from tensor0._stride._map import _execute_map
import tensor0._stride._map as primitive_module
import tensor0._stride._native as native_module
from tensor0._stride._errors import NO_ROUTE_PREFIX
from tensor0._stride._testing import (
    _native_call_count_for_tests,
    _reset_native_call_count_for_tests,
    native_available,
)
from tensor0._stride._plan import (
    build_affine_plan,
)

from ._fixtures import (
    contiguous_dtype_plan,
    dtype_transpose_plan,
    large_contiguous_plan,
    many_tiny_balanced_plan,
    two_record_noncompact_plan,
)
from ._oracle import execute_reference


@pytest.mark.skipif(not native_available(), reason="native CPU stride is unavailable")
def test_auto_selection_uses_native_for_general_maps_and_measured_large_class() -> (
    None
):
    small = two_record_noncompact_plan()
    small_source = jnp.arange(small.source_size, dtype=jnp.float32)
    small_compiled = jax.jit(lambda value: _execute_map(value, plan=small))

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
    large_compiled = jax.jit(lambda value: _execute_map(value, plan=large))

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
        jax.jit(lambda value: _execute_map(value, plan=bound)),
        jax.jit(jax.vmap(lambda value: _execute_map(value, plan=bound))),
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
def test_auto_selection_uses_native_for_batched_complex64() -> None:
    bound = contiguous_dtype_plan(jnp.complex64, 1, size=131_072)
    real = jnp.arange(8 * bound.source_size, dtype=jnp.float32).reshape(
        8,
        bound.source_size,
    )
    source = (real + 1j * (1 - real)).astype(jnp.complex64)
    expected = execute_reference(source, bound)
    compiled = jax.jit(jax.vmap(lambda value: _execute_map(value, plan=bound)))

    _reset_native_call_count_for_tests()
    actual = compiled(source)
    actual.block_until_ready()

    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    assert _native_call_count_for_tests() == 1
    assert (
        str(compiled.lower(source).compiler_ir()).lower().count("custom_call")
        == 1
    )

    direct = jax.jit(lambda value: _execute_map(value, plan=bound))
    _reset_native_call_count_for_tests()
    direct_actual = direct(source)
    direct_actual.block_until_ready()
    np.testing.assert_array_equal(
        np.asarray(direct_actual),
        np.asarray(expected),
    )
    assert _native_call_count_for_tests() == 1

    forced = jax.jit(
        jax.vmap(
            lambda value: _execute_map(
                value,
                plan=bound,
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
    compiled = jax.jit(lambda value: _execute_map(value, plan=bound))

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


def test_high_semantic_rank_normalizes_before_native_execution() -> None:
    shape = (2,) * 9
    strides = tuple(2 ** (8 - axis) for axis in range(9))
    record = AffineRecord(shape, strides, 0, strides, 0, 1.0)
    bound = build_affine_plan(
        records=(record,),
        output_size=512,
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=512,
        source_dtype=jnp.float32,
        result_dtype=jnp.float32,
    )
    source = jnp.arange(bound.source_size, dtype=jnp.float32)

    _reset_native_call_count_for_tests()
    compiled = jax.jit(lambda value: _execute_map(value, plan=bound))
    stablehlo = str(compiled.lower(source).compiler_ir()).lower()
    actual = compiled(source)

    np.testing.assert_array_equal(
        np.asarray(actual),
        np.asarray(execute_reference(source, bound)),
    )
    assert stablehlo.count("custom_call") == 1
    assert "gather" not in stablehlo
    assert "scatter" not in stablehlo
    assert _native_call_count_for_tests() == 1


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
    bound = build_affine_plan(
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
    function = lambda value: _execute_map(value, plan=bound)
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
    mapped_elements = 4
    record = AffineRecord(
        (mapped_elements,),
        (1,),
        0,
        (1,),
        0,
        1.0,
    )
    bound = build_affine_plan(
        records=(record,),
        output_size=mapped_elements,
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=source_size,
        source_dtype=jnp.float16,
        result_dtype=jnp.float32,
    )
    source_shape = (source_size,)

    source = jax.ShapeDtypeStruct(source_shape, jnp.float16)
    cotangent = jax.ShapeDtypeStruct((mapped_elements,), jnp.float32)
    vjp = jax.jit(
        lambda value, cot: jax.vjp(
            lambda item: _execute_map(item, plan=bound),
            value,
        )[1](cot)[0]
    )
    hlo = str(vjp.lower(source, cotangent).compiler_ir()).lower()
    assert hlo.count("custom_call") == 1
    assert "gather" not in hlo
    assert "scatter" not in hlo


def test_reducible_rank9_mixed_plan_supports_native_after_normalization() -> None:
    shape = (2,) * 9
    strides = tuple(2 ** (8 - axis) for axis in range(9))
    record = AffineRecord(shape, strides, 0, strides, 0, 1.0 + 0.5j)
    bound = build_affine_plan(
        records=(record,),
        output_size=512,
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=512,
        source_dtype=jnp.float32,
        result_dtype=jnp.complex64,
    )
    source = jnp.arange(bound.source_size, dtype=jnp.float32)
    function = jax.jit(lambda value: _execute_map(value, plan=bound))

    _reset_native_call_count_for_tests()
    stablehlo = str(function.lower(source).compiler_ir()).lower()
    actual = function(source)

    np.testing.assert_array_equal(
        np.asarray(actual),
        np.asarray(execute_reference(source, bound)),
    )
    assert "gather" not in stablehlo
    assert "scatter" not in stablehlo

    forced = jax.jit(
        lambda value: _execute_map(value, plan=bound)
    )
    _reset_native_call_count_for_tests()
    forced_hlo = str(forced.lower(source).compiler_ir()).lower()
    forced_actual = forced(source)
    forced_actual.block_until_ready()

    np.testing.assert_array_equal(np.asarray(forced_actual), np.asarray(actual))
    assert forced_hlo.count("custom_call") == 1
    assert _native_call_count_for_tests() == 1


def test_irreducible_rank9_direct_and_vmap_fail_closed() -> None:
    shape = (2,) * 9
    source_strides = tuple(3 ** (8 - axis) for axis in range(9))
    destination_strides = tuple(2 ** (8 - axis) for axis in range(9))
    record = AffineRecord(
        shape,
        source_strides,
        0,
        destination_strides,
        0,
        1.0,
    )
    bound = build_affine_plan(
        records=(record,),
        output_size=512,
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=sum(source_strides) + 1,
        source_dtype=jnp.float32,
        result_dtype=jnp.float32,
    )
    source = jax.ShapeDtypeStruct((3, bound.source_size), jnp.float32)
    functions = (
        jax.jit(lambda value: _execute_map(value, plan=bound)),
        jax.jit(jax.vmap(lambda value: _execute_map(value, plan=bound))),
    )

    for function in functions:
        with pytest.raises(
            Exception,
            match="effective logical rank exceeds native limit 8",
        ):
            function.lower(source).compile()


@pytest.mark.skipif(not native_available(), reason="native CPU stride is unavailable")
def test_auto_selection_routes_many_single_element_records_to_native() -> None:
    record_count = 1_024
    records = tuple(
        AffineRecord((1,), (1,), index, (1,), index, 1.0)
        for index in range(record_count)
    )
    bound = build_affine_plan(
        records=records,
        output_size=record_count,
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=record_count,
        source_dtype=jnp.float32,
        result_dtype=jnp.float32,
    )
    source = jnp.arange(record_count, dtype=jnp.float32)
    compiled = jax.jit(lambda value: _execute_map(value, plan=bound))
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
        primitive_module,
        "_STRIDE_PRIMITIVE",
        None,
    )

    with pytest.raises(
        RuntimeError,
        match=f"{NO_ROUTE_PREFIX} native_primitive_unavailable",
    ):
        _execute_map(source, plan=bound)


@pytest.mark.skipif(not native_available(), reason="native CPU stride is unavailable")
def test_non_cpu_lowering_fails_closed() -> None:
    bound = contiguous_dtype_plan(jnp.float32, 1, size=4_096)
    source = jnp.zeros(bound.source_size, dtype=jnp.float32)
    traced = jax.jit(
        lambda value: _execute_map(
            value,
            plan=bound,
        )
    ).trace(source)

    with pytest.raises(
        RuntimeError,
        match=f"{NO_ROUTE_PREFIX} native_device_executor_unavailable",
    ):
        traced.lower(lowering_platforms=("tpu",))


def test_native_build_runtime_mismatch_disables_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        native_module._native,
        "_stride_ffi_build_versions",
        lambda: ("mismatch", "mismatch"),
    )
    bound = two_record_noncompact_plan()
    source = jnp.arange(bound.source_size, dtype=jnp.float32)

    assert not native_module.native_available()
    with pytest.raises(
        RuntimeError,
        match=f"{NO_ROUTE_PREFIX} native_cpu_executor_unavailable",
    ):
        _execute_map(source, plan=bound)


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
    function = jax.jit(lambda value: _execute_map(value, plan=bound))
    hlo = str(function.lower(source).compiler_ir()).lower()

    assert hlo.count("custom_call") == 1
    assert "gather" not in hlo
    assert "scatter" not in hlo
    _reset_native_call_count_for_tests()
    actual = function(source)
    actual.block_until_ready()
    assert _native_call_count_for_tests() == 1
    np.testing.assert_array_equal(np.asarray(actual), np.zeros_like(actual))


def test_native_unavailable_positive_affine_fails_closed_on_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        native_module._native,
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
    function = jax.jit(lambda value: _execute_map(value, plan=bound))
    with pytest.raises(RuntimeError, match="native_cpu_executor_unavailable"):
        function.lower(source)
