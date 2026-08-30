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
from tensor0._stride._ffi import (
    _native_call_count_for_tests,
    _reset_native_call_count_for_tests,
    native_available,
)
from tensor0._stride._plan import (
    build_strided_copy_plan,
    transpose_same_dtype_plan,
)

from ._fixtures import (
    contiguous_dtype_plan,
    dtype_transpose_plan,
    partial_mixed_plan,
    rank4_avx2_shape_plan,
    two_record_noncompact_plan,
)
from ._oracle import execute_reference


pytestmark = pytest.mark.skipif(
    not native_available(),
    reason="the Tensor0 extension was built without JAX FFI headers",
)


def _assert_same_float_bits(actual: object, expected: object) -> None:
    actual_array = np.asarray(actual)
    expected_array = np.asarray(expected)
    assert actual_array.dtype == expected_array.dtype
    if actual_array.dtype.kind == "c":
        actual_components = actual_array.view(np.float32)
        expected_components = expected_array.view(np.float32)
    else:
        actual_components = actual_array
        expected_components = expected_array
    unsigned_dtype = {
        2: np.uint16,
        4: np.uint32,
        8: np.uint64,
    }[actual_components.dtype.itemsize]
    actual_bits = actual_components.view(unsigned_dtype).copy()
    expected_bits = expected_components.view(unsigned_dtype).copy()
    actual_bits[np.isnan(actual_components)] = 0
    expected_bits[np.isnan(expected_components)] = 0
    np.testing.assert_array_equal(
        actual_bits,
        expected_bits,
    )


@pytest.mark.parametrize(
    "scale",
    [
        complex(1.25, -0.75),
        complex(np.nan, 1.0),
        complex(np.inf, -0.0),
        complex(3.0e38, 3.0e38),
    ],
)
def test_complex_native_special_values_match_jax_scalar_oracle(
    scale: complex,
) -> None:
    values = jnp.asarray(
        [0.0, -0.0, np.inf, -np.inf, np.nan, 2.0, -2.0],
        dtype=jnp.float32,
    )
    cotangent = jnp.asarray(
        [
            0.0 - 0.0j,
            -0.0 + 0.0j,
            np.inf + 0.0j,
            0.0 + np.inf * 1j,
            np.nan + 1.0j,
            3.0e38 + 3.0e38j,
            -3.0e38 - 3.0e38j,
        ],
        dtype=jnp.complex64,
    )
    template = contiguous_dtype_plan(jnp.complex64, scale, size=values.size)
    mixed = build_strided_copy_plan(
        records=template.records,
        output_size=template.output_size,
        coverage=template.coverage,
        source_size=template.source_size,
        source_dtype=jnp.float32,
        result_dtype=jnp.complex64,
    )
    native = lambda value: strided_copy(value, plan=mixed, native_required=True)
    reference = lambda value: execute_reference(value, mixed)

    actual_forward, actual_reverse = jax.jit(
        lambda value, cot: (
            native(value),
            jax.vjp(native, value)[1](cot)[0],
        )
    )(values, cotangent)
    expected_forward = reference(values)
    expected_reverse = jax.vjp(reference, values)[1](cotangent)[0]

    _assert_same_float_bits(actual_forward, expected_forward)
    _assert_same_float_bits(actual_reverse, expected_reverse)

    same_dtype = contiguous_dtype_plan(jnp.complex64, scale, size=cotangent.size)
    actual_same = jax.jit(
        lambda value: strided_copy(value, plan=same_dtype, native_required=True)
    )(cotangent)
    expected_same = execute_reference(cotangent, same_dtype)
    _assert_same_float_bits(actual_same, expected_same)


def test_complex_native_finite_values_match_jax_bits() -> None:
    rng = np.random.default_rng(20260825)
    values = (
        rng.standard_normal(131_072).astype(np.float32)
        + 1j * rng.standard_normal(131_072).astype(np.float32)
    ).astype(np.complex64)
    bound = contiguous_dtype_plan(
        jnp.complex64,
        1.25 - 0.75j,
        size=values.size,
    )
    source = jnp.asarray(values)

    actual = jax.jit(
        lambda value: strided_copy(
            value,
            plan=bound,
            native_required=True,
        )
    )(source)
    expected = execute_reference(source, bound)

    _assert_same_float_bits(actual, expected)


def test_complex_native_max_float_overflow_matches_jax() -> None:
    maximum = np.finfo(np.float32).max
    source = jnp.asarray([complex(maximum, -maximum)], dtype=jnp.complex64)
    bound = contiguous_dtype_plan(
        jnp.complex64,
        complex(3.0e38, 3.0e38),
        size=1,
    )

    actual = jax.jit(
        lambda value: strided_copy(
            value,
            plan=bound,
            native_required=True,
        )
    )(source)
    expected = execute_reference(source, bound)

    _assert_same_float_bits(actual, expected)


@pytest.mark.parametrize(
    ("source_dtype", "result_dtype"),
    [(jnp.float32, jnp.float32), (jnp.float16, jnp.float32)],
)
def test_native_vjp_matches_reference_signed_zero(
    source_dtype: jnp.dtype,
    result_dtype: jnp.dtype,
) -> None:
    template = contiguous_dtype_plan(result_dtype, 1.0, size=1)
    bound = build_strided_copy_plan(
        records=template.records,
        output_size=template.output_size,
        coverage=template.coverage,
        source_size=1,
        source_dtype=source_dtype,
        result_dtype=result_dtype,
    )
    source = jnp.ones((1,), dtype=source_dtype)
    cotangent = jnp.asarray([-0.0], dtype=result_dtype)
    native = lambda value: strided_copy(value, plan=bound, native_required=True)
    reference = lambda value: execute_reference(value, bound)

    actual = jax.jit(lambda cot: jax.vjp(native, source)[1](cot)[0])(cotangent)
    expected = jax.vjp(reference, source)[1](cotangent)[0]

    _assert_same_float_bits(actual, expected)


def test_complex_native_vjp_normalizes_signed_zero_per_component() -> None:
    bound = contiguous_dtype_plan(jnp.complex64, 1.0, size=3)
    source = jnp.ones((3,), dtype=jnp.complex64)
    cotangent = jnp.asarray(
        [complex(-0.0, 1.0), complex(1.0, -0.0), complex(-0.0, -0.0)],
        dtype=jnp.complex64,
    )
    native = lambda value: strided_copy(value, plan=bound, native_required=True)
    reference = lambda value: execute_reference(value, bound)

    actual = jax.jit(lambda cot: jax.vjp(native, source)[1](cot)[0])(cotangent)
    expected = jax.vjp(reference, source)[1](cotangent)[0]

    _assert_same_float_bits(actual, expected)


def test_signed_zero_normalization_has_identity_jvp() -> None:
    bound = contiguous_dtype_plan(jnp.float32, 1.0, size=1)
    source = jnp.ones((1,), dtype=jnp.float32)
    cotangent = jnp.zeros((1,), dtype=jnp.float32)
    tangent = jnp.ones((1,), dtype=jnp.float32)
    native = lambda value: strided_copy(value, plan=bound, native_required=True)
    reference = lambda value: execute_reference(value, bound)

    actual = jax.jit(
        lambda cot, dot: jax.jvp(
            lambda item: jax.vjp(native, source)[1](item)[0],
            (cot,),
            (dot,),
        )
    )(cotangent, tangent)
    expected = jax.jvp(
        lambda item: jax.vjp(reference, source)[1](item)[0],
        (cotangent,),
        (tangent,),
    )

    _assert_same_float_bits(actual[0], expected[0])
    _assert_same_float_bits(actual[1], expected[1])


def test_signed_zero_normalization_has_identity_linear_transpose() -> None:
    bound = contiguous_dtype_plan(jnp.float32, 1.0, size=1)
    source = jnp.ones((1,), dtype=jnp.float32)
    cotangent = jnp.zeros((1,), dtype=jnp.float32)
    transpose_cotangent = jnp.ones((1,), dtype=jnp.float32)
    native = lambda value: strided_copy(value, plan=bound, native_required=True)
    reference = lambda value: execute_reference(value, bound)
    native_pullback = lambda cot: jax.vjp(native, source)[1](cot)[0]
    reference_pullback = lambda cot: jax.vjp(reference, source)[1](cot)[0]

    actual = jax.linear_transpose(native_pullback, cotangent)(transpose_cotangent)[0]
    actual_jit = jax.jit(
        lambda ct: jax.linear_transpose(native_pullback, cotangent)(ct)[0]
    )(transpose_cotangent)
    expected = jax.linear_transpose(reference_pullback, cotangent)(transpose_cotangent)[
        0
    ]

    _assert_same_float_bits(actual, expected)
    _assert_same_float_bits(actual_jit, expected)


@pytest.mark.parametrize(
    ("source_dtype", "result_dtype"),
    [
        (jnp.float16, jnp.float32),
        (jnp.float32, jnp.complex64),
        (jnp.complex64, jnp.float32),
    ],
)
@pytest.mark.filterwarnings("ignore:Casting complex values to real")
def test_mixed_affine_native_forward_and_transpose_match_jax_oracle(
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
    source = jnp.linspace(-3.0, 4.0, bound.source_size, dtype=jnp.float32)
    source = source.astype(source_dtype)
    if source_dtype == jnp.complex64:
        source = source * jnp.complex64(1.0 + 0.375j)
    cotangent = jnp.linspace(
        -2.0,
        3.0,
        bound.output_size,
        dtype=jnp.float32,
    ).astype(result_dtype)
    if result_dtype == jnp.complex64:
        cotangent = cotangent * jnp.complex64(0.75 + 1.25j)

    native = lambda value: strided_copy(
        value,
        plan=bound,
        native_required=True,
    )
    reference = lambda value: execute_reference(value, bound)
    compiled = jax.jit(
        lambda value, cot: (
            native(value),
            jax.vjp(native, value)[1](cot)[0],
        )
    )
    actual_forward, actual_reverse = compiled(source, cotangent)
    expected_forward = reference(source)
    expected_reverse = jax.vjp(reference, source)[1](cotangent)[0]
    np.testing.assert_allclose(
        np.asarray(actual_forward),
        np.asarray(expected_forward),
        rtol=2e-3 if source_dtype == jnp.float16 else 1e-6,
        atol=2e-3 if source_dtype == jnp.float16 else 1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(actual_reverse),
        np.asarray(expected_reverse),
        rtol=2e-3 if source_dtype == jnp.float16 else 1e-6,
        atol=2e-3 if source_dtype == jnp.float16 else 1e-6,
    )
    hlo = str(compiled.lower(source, cotangent).compiler_ir()).lower()
    assert hlo.count("custom_call") == 2
    assert "gather" not in hlo
    assert "scatter" not in hlo


def test_mixed_affine_native_partial_reverse_zero_fills_source_gradient() -> None:
    template = partial_mixed_plan()
    bound = build_strided_copy_plan(
        records=template.records,
        output_size=template.output_size,
        coverage=template.coverage,
        source_size=template.source_size,
        source_dtype=jnp.float32,
        result_dtype=jnp.complex64,
    )
    source = jnp.linspace(-1.0, 2.0, bound.source_size, dtype=jnp.float32)
    cotangent = jnp.linspace(
        -2.0, 3.0, bound.output_size, dtype=jnp.float32
    ) * jnp.complex64(0.5 + 1.5j)
    native = lambda value: strided_copy(
        value,
        plan=bound,
        native_required=True,
    )
    reference = lambda value: execute_reference(value, bound)
    actual = jax.jit(lambda value: jax.vjp(native, source)[1](value)[0])(cotangent)
    expected = jax.vjp(reference, source)[1](cotangent)[0]
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected))
    covered = np.zeros(bound.source_size, dtype=bool)
    covered[1 : 1 + 2 * 16 : 2] = True
    np.testing.assert_array_equal(np.asarray(actual)[~covered], 0)


def test_f32_c64_native_transpose_uses_jax_bilinear_complex_scale() -> None:
    bound = build_strided_copy_plan(
        records=(StridedCopyRecord((1,), (1,), 0, (1,), 0, 2.0 + 3j),),
        output_size=1,
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=1,
        source_dtype=jnp.float32,
        result_dtype=jnp.complex64,
    )
    source = jnp.asarray([1.25], dtype=jnp.float32)
    cotangent = jnp.asarray([5.0 + 7.0j], dtype=jnp.complex64)
    native = lambda value: strided_copy(
        value,
        plan=bound,
        native_required=True,
    )
    forward, reverse = jax.jit(
        lambda value, cot: (
            native(value),
            jax.vjp(native, value)[1](cot)[0],
        )
    )(source, cotangent)
    np.testing.assert_array_equal(
        np.asarray(forward),
        np.asarray([2.5 + 3.75j], dtype=np.complex64),
    )
    np.testing.assert_array_equal(
        np.asarray(reverse),
        np.asarray([-11.0], dtype=np.float32),
    )


def test_jitted_jvp_uses_the_same_native_plan_for_the_tangent() -> None:
    bound = two_record_noncompact_plan()
    source = jnp.arange(bound.source_size, dtype=jnp.float32)
    tangent = jnp.linspace(1.0, 2.0, bound.source_size, dtype=jnp.float32)
    apply = lambda value: strided_copy(
        value,
        plan=bound,
        native_required=True,
    )
    compiled = jax.jit(
        lambda primal, direction: jax.jvp(
            apply,
            (primal,),
            (direction,),
        )
    )

    _reset_native_call_count_for_tests()
    primal, tangent_result = compiled(source, tangent)
    primal.block_until_ready()
    tangent_result.block_until_ready()

    np.testing.assert_allclose(
        np.asarray(primal),
        np.asarray(execute_reference(source, bound)),
    )
    np.testing.assert_allclose(
        np.asarray(tangent_result),
        np.asarray(execute_reference(tangent, bound)),
    )
    assert _native_call_count_for_tests() == 2
    hlo = str(compiled.lower(source, tangent).compiler_ir()).lower()
    assert hlo.count("custom_call") == 2
    assert "gather" not in hlo
    assert "scatter" not in hlo


def test_partial_plan_jvp_zero_fills_primal_and_tangent() -> None:
    bound = partial_mixed_plan()
    source = jnp.arange(bound.source_size, dtype=jnp.float32)
    tangent = jnp.linspace(-2.0, 2.0, bound.source_size, dtype=jnp.float32)
    apply = lambda value: strided_copy(
        value,
        plan=bound,
        native_required=True,
    )
    compiled = jax.jit(
        lambda primal, direction: jax.jvp(
            apply,
            (primal,),
            (direction,),
        )
    )

    _reset_native_call_count_for_tests()
    primal, tangent_result = compiled(source, tangent)
    primal.block_until_ready()
    tangent_result.block_until_ready()

    np.testing.assert_array_equal(
        np.asarray(primal),
        np.asarray(execute_reference(source, bound)),
    )
    np.testing.assert_array_equal(
        np.asarray(tangent_result),
        np.asarray(execute_reference(tangent, bound)),
    )
    assert _native_call_count_for_tests() == 2


def test_jitted_vjp_and_linear_transpose_use_the_reversed_native_plan() -> None:
    bound = two_record_noncompact_plan()
    reverse = transpose_same_dtype_plan(bound)
    source = jnp.arange(bound.source_size, dtype=jnp.float32)
    cotangent = jnp.linspace(-2.0, 3.0, bound.output_size, dtype=jnp.float32)
    apply = lambda value: strided_copy(
        value,
        plan=bound,
        native_required=True,
    )

    def value_and_pullback(
        value: jax.Array,
        output_cotangent: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        primal, pullback = jax.vjp(apply, value)
        return primal, pullback(output_cotangent)[0]

    compiled = jax.jit(value_and_pullback)
    _reset_native_call_count_for_tests()
    primal, source_cotangent = compiled(source, cotangent)
    primal.block_until_ready()
    source_cotangent.block_until_ready()

    np.testing.assert_allclose(
        np.asarray(primal),
        np.asarray(execute_reference(source, bound)),
    )
    expected_cotangent = execute_reference(cotangent, reverse)
    np.testing.assert_allclose(
        np.asarray(source_cotangent),
        np.asarray(expected_cotangent),
    )
    assert _native_call_count_for_tests() == 2
    hlo = str(compiled.lower(source, cotangent).compiler_ir()).lower()
    assert hlo.count("custom_call") == 2
    assert "gather" not in hlo
    assert "scatter" not in hlo

    transposed = jax.linear_transpose(apply, source)(cotangent)[0]
    np.testing.assert_allclose(
        np.asarray(transposed),
        np.asarray(expected_cotangent),
    )


def test_rank4_tiled_vjp_exercises_the_reversed_kernel() -> None:
    bound = rank4_avx2_shape_plan()
    reverse = transpose_same_dtype_plan(bound)
    source = jnp.arange(bound.source_size, dtype=jnp.float32)
    cotangent = jnp.linspace(-1.0, 1.0, bound.output_size, dtype=jnp.float32)
    apply = lambda value: strided_copy(
        value,
        plan=bound,
        native_required=True,
    )

    def pullback(
        value: jax.Array,
        output_cotangent: jax.Array,
    ) -> jax.Array:
        _, apply_pullback = jax.vjp(apply, value)
        return apply_pullback(output_cotangent)[0]

    _reset_native_call_count_for_tests()
    actual = jax.jit(pullback)(source, cotangent)
    actual.block_until_ready()

    np.testing.assert_allclose(
        np.asarray(actual),
        np.asarray(execute_reference(cotangent, reverse)),
    )
    assert _native_call_count_for_tests() == 1


def test_complex_vjp_uses_the_nonconjugated_reversed_plan() -> None:
    bound = dtype_transpose_plan(jnp.complex64, 0.75 - 0.5j)
    reverse = transpose_same_dtype_plan(bound)
    source = (
        jnp.linspace(-2.0, 2.0, bound.source_size, dtype=jnp.float32)
        + 1j * jnp.linspace(3.0, -1.0, bound.source_size, dtype=jnp.float32)
    ).astype(jnp.complex64)
    cotangent = (
        jnp.linspace(1.0, -2.0, bound.output_size, dtype=jnp.float32)
        + 1j * jnp.linspace(-4.0, 2.0, bound.output_size, dtype=jnp.float32)
    ).astype(jnp.complex64)
    apply = lambda value: strided_copy(
        value,
        plan=bound,
        native_required=True,
    )

    def value_and_pullback(
        value: jax.Array,
        output_cotangent: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        primal, pullback = jax.vjp(apply, value)
        return primal, pullback(output_cotangent)[0]

    _reset_native_call_count_for_tests()
    primal, source_cotangent = jax.jit(value_and_pullback)(
        source,
        cotangent,
    )
    primal.block_until_ready()
    source_cotangent.block_until_ready()

    np.testing.assert_allclose(
        np.asarray(primal),
        np.asarray(execute_reference(source, bound)),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(source_cotangent),
        np.asarray(execute_reference(cotangent, reverse)),
        rtol=1e-6,
        atol=1e-6,
    )
    assert _native_call_count_for_tests() == 2


def test_vmap_of_jvp_batches_metadata_instead_of_calls() -> None:
    bound = two_record_noncompact_plan()
    source = jnp.arange(3 * bound.source_size, dtype=jnp.float32).reshape(
        3, bound.source_size
    )
    tangent = jnp.full_like(source, 0.5)
    apply = lambda value: strided_copy(
        value,
        plan=bound,
        native_required=True,
    )
    compiled = jax.jit(
        jax.vmap(
            lambda primal, direction: jax.jvp(
                apply,
                (primal,),
                (direction,),
            )
        )
    )

    _reset_native_call_count_for_tests()
    primal, tangent_result = compiled(source, tangent)
    primal.block_until_ready()
    tangent_result.block_until_ready()

    expected_primal = jax.vmap(lambda value: execute_reference(value, bound))(source)
    expected_tangent = jax.vmap(lambda value: execute_reference(value, bound))(tangent)
    np.testing.assert_allclose(np.asarray(primal), np.asarray(expected_primal))
    np.testing.assert_allclose(
        np.asarray(tangent_result),
        np.asarray(expected_tangent),
    )
    assert _native_call_count_for_tests() == 2
    hlo = str(compiled.lower(source, tangent).compiler_ir()).lower()
    assert hlo.count("custom_call") == 2


def test_vmap_of_vjp_batches_the_reversed_plan_once() -> None:
    bound = two_record_noncompact_plan()
    reverse = transpose_same_dtype_plan(bound)
    source = jnp.arange(3 * bound.source_size, dtype=jnp.float32).reshape(
        3,
        bound.source_size,
    )
    cotangent = jnp.linspace(
        -2.0,
        3.0,
        3 * bound.output_size,
        dtype=jnp.float32,
    ).reshape(3, bound.output_size)
    apply = lambda value: strided_copy(
        value,
        plan=bound,
        native_required=True,
    )
    compiled = jax.jit(
        jax.vmap(
            lambda value, output_cotangent: jax.vjp(
                apply,
                value,
            )[1](output_cotangent)[0]
        )
    )

    _reset_native_call_count_for_tests()
    actual = compiled(source, cotangent)
    actual.block_until_ready()

    expected = jax.vmap(lambda value: execute_reference(value, reverse))(cotangent)
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected))
    assert _native_call_count_for_tests() == 1
    hlo = str(compiled.lower(source, cotangent).compiler_ir()).lower()
    assert hlo.count("custom_call") == 1


def test_zero_and_integer_tangents_have_explicit_jax_behavior() -> None:
    bound = two_record_noncompact_plan()
    source = jnp.arange(bound.source_size, dtype=jnp.float32)
    apply = lambda value: strided_copy(
        value,
        plan=bound,
        native_required=True,
    )
    zero_tangent = jax.jit(
        lambda value: jax.jvp(
            apply,
            (value,),
            (jnp.zeros_like(value),),
        )[1]
    )(source)
    np.testing.assert_array_equal(np.asarray(zero_tangent), 0)

    integer = dtype_transpose_plan(jnp.int32, -1)
    integer_source = jnp.arange(integer.source_size, dtype=jnp.int32)
    integer_tangent = jnp.zeros(
        integer_source.shape,
        dtype=jax.dtypes.float0,
    )
    integer_apply = lambda value: strided_copy(
        value,
        plan=integer,
        native_required=True,
    )
    integer_primal, result_tangent = jax.jit(
        lambda value, tangent: jax.jvp(
            integer_apply,
            (value,),
            (tangent,),
        )
    )(integer_source, integer_tangent)

    np.testing.assert_array_equal(
        np.asarray(integer_primal),
        np.asarray(execute_reference(integer_source, integer)),
    )
    assert result_tangent.dtype == jax.dtypes.float0
    assert result_tangent.shape == (integer.output_size,)


def test_reverse_plan_with_uncovered_source_tail_uses_native_zero_fill() -> None:
    complete = two_record_noncompact_plan()
    bound = build_strided_copy_plan(
        records=complete.records,
        output_size=complete.output_size,
        coverage=complete.coverage,
        source_size=complete.source_size + 4,
        source_dtype=jnp.float32,
        result_dtype=jnp.float32,
    )
    source = jnp.arange(bound.source_size, dtype=jnp.float32)
    cotangent = jnp.ones((bound.output_size,), dtype=jnp.float32)
    apply = lambda value: strided_copy(
        value,
        plan=bound,
        native_required=True,
    )

    def value_and_pullback(
        value: jax.Array,
        output_cotangent: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        primal, pullback = jax.vjp(apply, value)
        return primal, pullback(output_cotangent)[0]

    compiled = jax.jit(value_and_pullback)
    _reset_native_call_count_for_tests()
    _, source_cotangent = compiled(source, cotangent)
    source_cotangent.block_until_ready()

    reverse = transpose_same_dtype_plan(bound)
    np.testing.assert_allclose(
        np.asarray(source_cotangent),
        np.asarray(execute_reference(cotangent, reverse)),
    )
    np.testing.assert_array_equal(np.asarray(source_cotangent[-4:]), 0)
    assert _native_call_count_for_tests() == 2
    hlo = str(compiled.lower(source, cotangent).compiler_ir()).lower()
    assert hlo.count("custom_call") == 2
    assert "gather" not in hlo
    assert "scatter" not in hlo
