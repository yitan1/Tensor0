from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from tensor0._stride import (
    CompleteMode,
    StridedCopyRecord,
    build_strided_copy_plan,
    strided_copy,
)
from tensor0._stride._compiler import compile_plan
from tensor0._stride._plan import transpose_same_dtype_plan
from tensor0._stride._stablehlo import (
    compile_affine_single_record_stablehlo_recipe,
    compile_affine_transpose_stablehlo_recipe,
    execute_affine_single_record_stablehlo_recipe,
    execute_affine_transpose_stablehlo_recipe,
)

from ._oracle import execute_reference


def _transpose_plan(rows: int, columns: int):
    size = rows * columns
    record = StridedCopyRecord(
        logical_shape=(rows, columns),
        source_strides=(1, rows),
        source_offset=0,
        destination_strides=(columns, 1),
        destination_offset=0,
        scale=1.25,
    )
    return build_strided_copy_plan(
        records=(record,),
        output_size=size,
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=size,
        source_dtype=jnp.float32,
        result_dtype=jnp.float32,
    )


def _positive_slice_plan():
    record = StridedCopyRecord(
        logical_shape=(3, 4),
        source_strides=(8, 1),
        source_offset=1,
        destination_strides=(4, 1),
        destination_offset=0,
        scale=-0.75,
    )
    return build_strided_copy_plan(
        records=(record,),
        output_size=12,
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=25,
        source_dtype=jnp.float32,
        result_dtype=jnp.float32,
    )


def test_positive_affine_recipes_match_forward_and_transpose() -> None:
    bound = _positive_slice_plan()
    reverse_bound = transpose_same_dtype_plan(bound)
    forward = compile_affine_single_record_stablehlo_recipe(compile_plan(bound))
    reverse = compile_affine_transpose_stablehlo_recipe(compile_plan(reverse_bound))
    source = jnp.arange(bound.source_size, dtype=jnp.float32)
    cotangent = jnp.linspace(-2, 3, bound.output_size, dtype=jnp.float32)

    np.testing.assert_array_equal(
        execute_affine_single_record_stablehlo_recipe(source, forward),
        execute_reference(source, bound),
    )
    np.testing.assert_array_equal(
        execute_affine_transpose_stablehlo_recipe(cotangent, reverse),
        execute_reference(cotangent, reverse_bound),
    )


def test_small_positive_affine_automatic_route_is_address_free() -> None:
    bound = _positive_slice_plan()
    source = jnp.arange(bound.source_size, dtype=jnp.float32)
    function = jax.jit(lambda value: strided_copy(value, plan=bound))

    np.testing.assert_array_equal(
        function(source),
        execute_reference(source, bound),
    )
    stablehlo = str(function.lower(source).compiler_ir(dialect="stablehlo")).lower()
    assert "custom_call" not in stablehlo
    assert "gather" not in stablehlo
    assert "scatter" not in stablehlo
    assert "stablehlo.slice" in stablehlo


def test_large_positive_affine_plan_prefers_native() -> None:
    bound = _transpose_plan(512, 512)
    source = jnp.arange(bound.source_size, dtype=jnp.float32)

    stablehlo = str(
        jax.jit(lambda value: strided_copy(value, plan=bound))
        .lower(source)
        .compiler_ir(dialect="stablehlo")
    ).lower()

    assert stablehlo.count("custom_call") == 1
    assert "tensor0_stride_r2_prepared_f32_cpu_v7" in stablehlo
