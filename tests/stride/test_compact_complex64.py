from __future__ import annotations

from math import prod

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensor0._stride._plan import CompleteMode, AffineRecord
from tensor0._stride._map import _execute_map
from tensor0._stride._testing import (
    _native_call_count_for_tests,
    _reset_native_call_count_for_tests,
    native_available,
)
from tensor0._stride._plan import (
    build_affine_plan,
)
from ._oracle import execute_reference


def _row_major_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    expected = 1
    strides = [0] * len(shape)
    for axis in reversed(range(len(shape))):
        strides[axis] = expected
        expected *= shape[axis]
    return tuple(strides)


def _compact_plan(
    shape: tuple[int, ...] = (128, 64),
    *,
    source_offset: int = 0,
    source_padding: int = 0,
    scale: complex = 0.75 - 0.5j,
    dtype: jnp.dtype = jnp.complex64,
):
    elements = prod(shape)
    strides = _row_major_strides(shape)
    record = AffineRecord(
        shape,
        strides,
        source_offset,
        strides,
        0,
        scale,
    )
    return build_affine_plan(
        records=(record,),
        output_size=elements,
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=source_offset + elements + source_padding,
        source_dtype=dtype,
        result_dtype=dtype,
    )


def _complex_values(shape: tuple[int, ...]) -> jax.Array:
    size = prod(shape)
    values = jnp.arange(size, dtype=jnp.float32).reshape(shape)
    return (values / 32.0 + 1j * (values / 64.0 - 1.0)).astype(jnp.complex64)


def _stablehlo(function, argument: jax.Array) -> str:
    return str(
        jax.jit(function).lower(argument).compiler_ir(dialect="stablehlo")
    ).lower()


def _permutation_plan():
    shape = (8, 16)
    record = AffineRecord(
        shape,
        (1, shape[0]),
        0,
        _row_major_strides(shape),
        0,
        0.75 - 0.5j,
    )
    return build_affine_plan(
        records=(record,),
        output_size=prod(shape),
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=prod(shape),
        source_dtype=jnp.complex64,
        result_dtype=jnp.complex64,
    )


@pytest.mark.parametrize(
    ("dtype", "scale", "elements"),
    [
        (jnp.bfloat16, 0.75, 131_072),
        (jnp.float32, 0.75, 65_536),
        (jnp.complex64, 0.75 - 0.5j, 32_768),
        (jnp.int32, 3, 65_536),
    ],
)
def test_rank_one_compact_native_route_uses_native_without_address_arrays(
    dtype: jnp.dtype,
    scale: int | float | complex,
    elements: int,
) -> None:
    bound = _compact_plan(
        (elements,),
        source_offset=3,
        source_padding=5,
        scale=scale,
        dtype=dtype,
    )
    real = jnp.arange(bound.source_size, dtype=jnp.float32)
    source = (
        (real / 32 + 1j * (real / 64 - 1)).astype(dtype)
        if jnp.issubdtype(dtype, jnp.complexfloating)
        else real.astype(dtype)
    )
    function = lambda value: _execute_map(value, plan=bound)
    stablehlo = _stablehlo(function, source)

    assert stablehlo.count("custom_call") == 1
    assert "gather" not in stablehlo
    assert "scatter" not in stablehlo
    assert "iota" not in stablehlo
    np.testing.assert_array_equal(
        np.asarray(jax.jit(function)(source)),
        np.asarray(execute_reference(source, bound)),
    )


def test_medium_float16_compact_native_route_uses_native_simd() -> None:
    bound = _compact_plan(
        (131_072,),
        source_offset=3,
        source_padding=5,
        scale=0.75,
        dtype=jnp.float16,
    )
    source = jnp.asarray(
        np.linspace(-1, 1, bound.source_size, dtype=np.float16)
    )
    function = lambda value: _execute_map(value, plan=bound)
    stablehlo = _stablehlo(function, source)

    assert "tensor0_stride_affine_f16" in stablehlo.lower()
    assert "gather" not in stablehlo
    assert "scatter" not in stablehlo
    np.testing.assert_array_equal(
        np.asarray(jax.jit(function)(source)),
        np.asarray(execute_reference(source, bound)),
    )


def test_small_compact_vmap_and_float32_ad_use_native() -> None:
    bound = _compact_plan(
        (4_096,),
        source_offset=3,
        source_padding=5,
        scale=0.75,
        dtype=jnp.float32,
    )
    source = jnp.arange(4 * bound.source_size, dtype=jnp.float32).reshape(
        4,
        bound.source_size,
    )
    function = jax.vmap(lambda value: _execute_map(value, plan=bound))
    reference = jax.vmap(lambda value: execute_reference(value, bound))
    stablehlo = _stablehlo(function, source)

    assert stablehlo.count("custom_call") == 1
    assert "gather" not in stablehlo
    assert "scatter" not in stablehlo
    primal, tangent = jax.jvp(function, (source,), (source / 8,))
    expected_primal, expected_tangent = jax.jvp(
        reference,
        (source,),
        (source / 8,),
    )
    np.testing.assert_array_equal(np.asarray(primal), np.asarray(expected_primal))
    np.testing.assert_array_equal(
        np.asarray(tangent),
        np.asarray(expected_tangent),
    )
    cotangent = jnp.ones_like(primal)
    actual_vjp = jax.vjp(function, source)[1](cotangent)[0]
    expected_vjp = jax.vjp(reference, source)[1](cotangent)[0]
    np.testing.assert_array_equal(
        np.asarray(actual_vjp),
        np.asarray(expected_vjp),
    )
    vjp_function = jax.jit(lambda value: jax.vjp(function, source)[1](value)[0])
    vjp_hlo = str(vjp_function.lower(cotangent).compiler_ir()).lower()
    assert vjp_hlo.count("custom_call") == 1
    assert "tensor0_stride_affine_f32_cpu_v8" in vjp_hlo
    assert "gather" not in vjp_hlo
    assert "scatter" not in vjp_hlo


def test_rank_one_compact_native_composes_with_adjacent_ops() -> None:
    bound = _compact_plan(
        (65_536,),
        source_offset=3,
        source_padding=5,
        scale=0.75,
        dtype=jnp.float32,
    )
    source = jnp.arange(bound.source_size, dtype=jnp.float32) / 64
    function = jax.jit(lambda value: jnp.tanh(_execute_map(jnp.sin(value), plan=bound)))
    compiled = function.lower(source).compile()
    assert compiled is not None
    compiled_text = compiled.as_text()
    assert compiled_text is not None
    hlo = compiled_text.lower()

    assert "tensor0_stride_" in hlo
    assert "gather" not in hlo
    assert "scatter" not in hlo
    assert "fusion" in hlo
    np.testing.assert_allclose(
        np.asarray(compiled(source)),
        np.asarray(jnp.tanh(execute_reference(jnp.sin(source), bound))),
        rtol=1e-6,
        atol=1e-6,
    )


def test_batched_native_route_uses_one_native_operation() -> None:
    bound = _compact_plan(source_offset=3, source_padding=5)
    source = _complex_values((3, bound.source_size))
    function = lambda value: _execute_map(value, plan=bound)
    stablehlo = _stablehlo(function, source)

    assert stablehlo.count("custom_call") == 1
    assert "gather" not in stablehlo
    assert "scatter" not in stablehlo
    assert "iota" not in stablehlo

    _reset_native_call_count_for_tests()
    actual = jax.jit(function)(source)
    actual.block_until_ready()
    assert _native_call_count_for_tests() == 1
    np.testing.assert_array_equal(
        np.asarray(actual),
        np.asarray(execute_reference(source, bound)),
    )


def test_compact_recipe_jvp_vjp_and_nonleading_vmap_match_reference() -> None:
    bound = _compact_plan(source_offset=3, source_padding=5)
    source = _complex_values((bound.source_size, 3))
    tangent = (source * jnp.complex64(0.25 + 0.125j)).astype(jnp.complex64)
    function = lambda value: jax.vmap(
        lambda item: _execute_map(item, plan=bound),
        in_axes=1,
        out_axes=1,
    )(value)
    reference = lambda value: jax.vmap(
        lambda item: execute_reference(item, bound),
        in_axes=1,
        out_axes=1,
    )(value)

    stablehlo = _stablehlo(function, source)
    assert stablehlo.count("custom_call") == 1
    assert "tensor0_stride_affine_c64_cpu_v8" in stablehlo
    assert "stablehlo.transpose" not in stablehlo
    assert "gather" not in stablehlo
    assert "scatter" not in stablehlo
    assert "iota" not in stablehlo

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
    np.testing.assert_array_equal(
        np.asarray(actual_primal),
        np.asarray(expected_primal),
    )
    np.testing.assert_array_equal(
        np.asarray(actual_tangent),
        np.asarray(expected_tangent),
    )

    cotangent = _complex_values(actual_primal.shape)
    actual_vjp = jax.vjp(function, source)[1](cotangent)[0]
    expected_vjp = jax.vjp(reference, source)[1](cotangent)[0]
    np.testing.assert_array_equal(
        np.asarray(actual_vjp),
        np.asarray(expected_vjp),
    )
    vjp_stablehlo = _stablehlo(
        lambda value: jax.vjp(function, source)[1](value)[0],
        cotangent,
    )
    assert vjp_stablehlo.count("custom_call") == 1
    assert "tensor0_stride_affine_c64_cpu_v8" in vjp_stablehlo
    assert "stablehlo.transpose" not in vjp_stablehlo
    assert "gather" not in vjp_stablehlo
    assert "scatter" not in vjp_stablehlo


def test_compact_native_hlo_size_is_independent_of_element_count() -> None:
    operation_counts: list[int] = []
    for elements in (64, 16_384):
        bound = _compact_plan((elements,))
        source = _complex_values((2, elements))
        stablehlo = _stablehlo(
            lambda value, plan=bound: _execute_map(value, plan=plan),
            source,
        )
        operation_counts.append(stablehlo.count("stablehlo."))
        assert stablehlo.count("custom_call") == 1
        assert "gather" not in stablehlo
        assert "scatter" not in stablehlo
        assert "iota" not in stablehlo

    assert operation_counts[0] == operation_counts[1]


def test_compact_and_positive_affine_native_routes_are_address_free() -> None:
    small = _compact_plan((64,))
    small_source = _complex_values((small.source_size,))
    small_hlo = _stablehlo(
        lambda value: _execute_map(value, plan=small),
        small_source,
    )
    assert small_hlo.count("custom_call") == 1
    assert "gather" not in small_hlo
    assert "scatter" not in small_hlo

    permutation = _permutation_plan()
    batched_source = _complex_values((2, permutation.source_size))
    permutation_hlo = _stablehlo(
        lambda value: _execute_map(value, plan=permutation),
        batched_source,
    )
    assert permutation_hlo.count("custom_call") == 1
    assert "gather" not in permutation_hlo
    assert "scatter" not in permutation_hlo
    assert "stablehlo.transpose" not in permutation_hlo

    if native_available():
        large = _compact_plan((65_536,))
        large_source = _complex_values((large.source_size,))
        large_hlo = _stablehlo(
            lambda value: _execute_map(value, plan=large),
            large_source,
        )
        assert "tensor0_stride_affine_c64_cpu_v8" in large_hlo
        assert "gather" not in large_hlo
        assert "scatter" not in large_hlo
