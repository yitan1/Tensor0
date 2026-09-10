"""View algebra contracts and their native execution boundaries."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.typing import DTypeLike
import numpy as np
import pytest

import tensor0
import tensor0._stride as stride_api
from tensor0._stride._plan import CompleteMode, AffineRecord
from tensor0._stride import (
    StridedView,
    add,
    dotc,
    dotu,
    materialize,
    reduce_sum,
    scale,
)
from tensor0._stride._testing import (
    _native_call_count_for_tests,
    _native_reduction_fiber_chunks_for_tests,
    _native_worker_counts_for_tests,
    _reset_native_call_count_for_tests,
    _set_native_worker_limit_for_tests,
    native_available,
)
from tensor0._stride._ops._dot import (
    _dot_descriptor,
    native_projection_dotu,
)
from tensor0._stride._native import _dot_ffi_call
from tensor0._stride._plan import build_affine_plan


def _pitched(data: jax.Array) -> StridedView:
    return StridedView(data, (2, 3), (1, 4), 2)


def _dense(data: jax.Array) -> StridedView:
    return StridedView.from_dense(data, (2, 3))


def test_stride_facade_exports_only_view_algebra_and_threads() -> None:
    assert set(stride_api.__all__) == {
        "StridedView",
        "materialize",
        "scale",
        "add",
        "dotu",
        "dotc",
        "reduce_sum",
        "get_num_threads",
        "set_num_threads",
        "enable_threads",
        "disable_threads",
    }
    assert {
        name for name in vars(stride_api) if not name.startswith("_")
    } == set(stride_api.__all__)
    assert not hasattr(StridedView, "with_data")


def test_add_and_scale_follow_tensor_map_coefficient_order() -> None:
    factor = tensor0.space(tensor0.U1Irrep, {0: 2, 1: 3})
    target = tensor0.hom((factor,), (factor,))
    left_data = jnp.arange(13, dtype=jnp.float32)
    right_data = left_data + 10
    left_tensor = tensor0.TensorMap(target, left_data)
    right_tensor = tensor0.TensorMap(target, right_data)
    left = StridedView.from_dense(left_data, (13,))
    right = StridedView.from_dense(right_data, (13,))
    np.testing.assert_array_equal(
        add(left, right, alpha=2, beta=-3).data,
        tensor0.add(left_tensor, right_tensor, 2, -3).storage.data,
    )
    np.testing.assert_array_equal(
        scale(left, 2).data,
        tensor0.scale(left_tensor, 2).storage.data,
    )
    with pytest.raises(TypeError):
        getattr(stride_api, "add")(left, right, 2, -3)


@pytest.mark.parametrize("alpha,beta", [(0, 1), (0, 0), (1, 1), (1, 0), (2, 1)])
@pytest.mark.parametrize(
    "source_dtype,result_dtype",
    [
        (jnp.float32, jnp.float32),
        (jnp.complex64, jnp.complex64),
        (jnp.float16, jnp.float32),
    ],
)
def test_add_static_and_dynamic_coefficients_short_circuit_special_values(
    alpha: int,
    beta: int,
    source_dtype: DTypeLike,
    result_dtype: DTypeLike,
) -> None:
    left_data = jnp.asarray(
        [-0.0, jnp.nan, 7, jnp.inf, 9, 2, -0.0], dtype=result_dtype,
    )
    right_data = jnp.asarray([3, 4, jnp.nan], dtype=source_dtype)
    left = StridedView(left_data, (3,), (2,), 1)
    right = StridedView.from_dense(right_data, (3,))
    indices = jnp.asarray([1, 3, 5])
    terms = []
    for factor, values in ((beta, right_data.astype(result_dtype)),
                           (alpha, left_data[indices])):
        if factor != 0:
            terms.append(values if factor == 1 else factor * values)
    selected = (jnp.zeros((3,), dtype=result_dtype) if not terms
                else terms[0] if len(terms) == 1 else terms[0] + terms[1])
    expected = left_data.at[indices].set(selected)
    dynamic = jax.jit(
        lambda old, new, lhs, rhs: add(
            old, new, alpha=lhs, beta=rhs,
        )
    )
    outputs = (
        add(left, right, alpha=alpha, beta=beta),
        dynamic(left, right, jnp.asarray(alpha), jnp.asarray(beta)),
    )
    outside = np.asarray([0, 2, 4, 6])
    for output in outputs:
        np.testing.assert_allclose(output.data, expected, equal_nan=True)
        np.testing.assert_array_equal(
            np.asarray(output.data)[outside].view(np.uint8),
            np.asarray(left_data)[outside].view(np.uint8),
        )
    np.testing.assert_array_equal(left.data, left_data)
    np.testing.assert_array_equal(right.data, right_data)


@pytest.mark.parametrize(
    "source_dtype,result_dtype",
    [
        (jnp.float32, jnp.float32),
        (jnp.complex64, jnp.complex64),
        (jnp.float16, jnp.float32),
    ],
)
def test_add_batched_zero_one_coefficients_jvp_vjp_and_mixed_derivatives(
    source_dtype: DTypeLike,
    result_dtype: DTypeLike,
) -> None:
    left = jnp.arange(36, dtype=jnp.float32).reshape(3, 12).astype(result_dtype)
    right = jnp.arange(18, dtype=jnp.float32).reshape(3, 2, 3).astype(source_dtype)
    if jnp.issubdtype(result_dtype, jnp.complexfloating):
        left = left * (1 - 0.5j)
        right = right * (1 + 0.25j)
    alpha = jnp.asarray([0, 1, -0.5], dtype=jnp.float32)
    beta = jnp.asarray([1, 0, 2], dtype=jnp.float32)
    indices = jnp.asarray([2, 6, 10, 3, 7, 11])

    def operation(old, new, lhs, rhs):
        return add(_pitched(old), _dense(new), alpha=lhs, beta=rhs).data

    def oracle(old, new, lhs, rhs):
        mapped = new.astype(old.dtype).reshape(3, 6)
        return old.at[:, indices].set(
            rhs.astype(old.dtype)[:, None] * mapped
            + lhs.astype(old.dtype)[:, None] * old[:, indices]
        )

    primals = (left, right, alpha, beta)
    tangents = tuple(jnp.ones_like(value) for value in primals)
    np.testing.assert_allclose(jax.vmap(operation)(*primals), oracle(*primals))
    for actual, expected in zip(
        jax.jvp(operation, primals, tangents),
        jax.jvp(oracle, primals, tangents),
        strict=True,
    ):
        np.testing.assert_allclose(actual, expected, rtol=5e-3, atol=5e-3)
    _, actual_pullback = jax.vjp(operation, *primals)
    _, expected_pullback = jax.vjp(oracle, *primals)
    cotangent = jnp.ones_like(left)
    for actual, expected in zip(
        actual_pullback(cotangent), expected_pullback(cotangent), strict=True,
    ):
        np.testing.assert_allclose(actual, expected, rtol=5e-3, atol=5e-3)
    actual_gradient = jax.grad(
        lambda *arguments: jnp.real(jnp.sum(operation(*arguments))), argnums=2,
    )
    expected_gradient = jax.grad(
        lambda *arguments: jnp.real(jnp.sum(oracle(*arguments))), argnums=2,
    )
    for actual, expected in zip(
        jax.jvp(actual_gradient, primals, tangents),
        jax.jvp(expected_gradient, primals, tangents),
        strict=True,
    ):
        np.testing.assert_allclose(actual, expected, rtol=5e-3, atol=5e-3)


def test_algebra_updates_preserve_inputs_layout_and_unselected_storage() -> None:
    destination_data = jnp.arange(12, dtype=jnp.float32)
    destination = _pitched(destination_data)
    source = _dense(jnp.arange(6, dtype=jnp.float32).reshape(2, 3) + 10)

    scaled = scale(destination, jnp.asarray(-2, dtype=jnp.float32))
    added = add(destination, source, beta=jnp.asarray(0.5, dtype=jnp.float32))
    combined = add(
        destination,
        source,
        alpha=jnp.asarray(-2, dtype=jnp.float32),
        beta=jnp.asarray(0.5, dtype=jnp.float32),
    )

    outside = np.asarray([0, 1, 4, 5, 8, 9])
    for result in (scaled, added, combined):
        assert result.sizes == destination.sizes
        assert result.strides == destination.strides
        assert result.offset == destination.offset
        assert result.data is not destination_data
        np.testing.assert_array_equal(result.data[outside], destination_data[outside])
    np.testing.assert_array_equal(materialize(scaled), -2 * materialize(destination))
    np.testing.assert_array_equal(
        materialize(added),
        0.5 * materialize(source) + materialize(destination),
    )
    np.testing.assert_array_equal(
        materialize(combined),
        0.5 * materialize(source) - 2 * materialize(destination),
    )
    np.testing.assert_array_equal(destination.data, destination_data)


def test_algebra_dotu_dotc_and_reduce_sum_preserve_batch_axes() -> None:
    left_dense = jnp.asarray(
        [
            [[1 + 2j, 3 - 1j, -2 + 0j], [4 + 1j, 2 + 2j, 1 - 3j]],
            [[-1 + 1j, 2 + 0j, 3 + 4j], [1 - 2j, -3 + 1j, 2 + 2j]],
        ],
        dtype=jnp.complex64,
    )
    right_dense = jnp.asarray(
        [
            [[2 - 1j, 1 + 3j, 4 + 0j], [-1 + 2j, 3 - 1j, 2 + 1j]],
            [[1 + 0j, 4 - 2j, -2 + 1j], [3 + 3j, 2 + 0j, -1 - 1j]],
        ],
        dtype=jnp.complex64,
    )
    left = StridedView.from_dense(left_dense, (2, 3))
    right = StridedView.from_dense(right_dense, (2, 3))

    np.testing.assert_allclose(dotu(left, right), jnp.sum(left_dense * right_dense, axis=(1, 2)))
    np.testing.assert_allclose(
        dotc(left, right),
        jnp.sum(jnp.conj(left_dense) * right_dense, axis=(1, 2)),
    )
    np.testing.assert_allclose(
        reduce_sum(left, (1,)),
        jnp.sum(left_dense, axis=2),
    )
    np.testing.assert_allclose(reduce_sum(left), jnp.sum(left_dense, axis=(1, 2)))


def test_reduce_sum_uses_native_reduction_for_supported_dtype() -> None:
    source = jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4)

    @jax.jit
    def operation(value: jax.Array) -> jax.Array:
        return reduce_sum(StridedView.from_dense(value, (3, 4)), (1,))

    _reset_native_call_count_for_tests()
    actual = operation(source)
    actual.block_until_ready()

    np.testing.assert_array_equal(actual, jnp.sum(source, axis=2))
    assert _native_call_count_for_tests() == 1


def test_reduce_sum_preserves_jax_integer_promotion() -> None:
    source = jnp.arange(12, dtype=jnp.int8).reshape(3, 4)

    actual = reduce_sum(StridedView.from_dense(source, (3, 4)))
    expected = jnp.sum(source, axis=(0, 1))

    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)


def test_native_dot_uses_one_custom_call() -> None:
    left = jnp.arange(6, dtype=jnp.float32).reshape(2, 3) + 1
    right = jnp.linspace(-2, 2, 12, dtype=jnp.float32)

    @jax.jit
    def operation(left_data: jax.Array, right_data: jax.Array) -> jax.Array:
        return dotu(_dense(left_data), _pitched(right_data))

    _reset_native_call_count_for_tests()
    actual = operation(left, right)
    actual.block_until_ready()
    assert _native_call_count_for_tests() == 1
    np.testing.assert_allclose(
        actual,
        jnp.sum(left * materialize(_pitched(right))),
    )

    stablehlo = str(
        operation.lower(left, right).compiler_ir("stablehlo")
    ).lower()
    assert stablehlo.count("stablehlo.custom_call") == 1
    assert "tensor0_stride_dotu_float32_cpu_v1" in stablehlo


def test_native_dot_supports_negative_stride_and_batched_rank_zero() -> None:
    left_data = jnp.arange(6, dtype=jnp.float32)
    right_data = jnp.linspace(-2, 3, 6, dtype=jnp.float32)
    left = StridedView(left_data, (6,), (-1,), 5)
    right = StridedView(right_data, (6,), (1,), 0)
    np.testing.assert_allclose(
        dotu(left, right),
        jnp.sum(left_data[::-1] * right_data),
    )

    left_scalars = jnp.asarray([1.0, -2.0, 3.0], dtype=jnp.float32)
    right_scalars = jnp.asarray([4.0, 5.0, -6.0], dtype=jnp.float32)
    np.testing.assert_array_equal(
        dotu(
            StridedView.from_dense(left_scalars, ()),
            StridedView.from_dense(right_scalars, ()),
        ),
        left_scalars * right_scalars,
    )


@pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")
def test_native_dot_recursively_splits_multidimensional_domain() -> None:
    shape = (512, 512)
    element_count = shape[0] * shape[1]
    left_data = jnp.linspace(-1, 2, element_count, dtype=jnp.float32)
    right_data = jnp.linspace(3, -2, element_count, dtype=jnp.float32)
    left = StridedView(left_data, shape, (1, shape[0]), 0)
    right = StridedView(right_data, shape, (shape[1], 1), 0)

    _set_native_worker_limit_for_tests(4)
    try:
        actual = jax.jit(dotu)(left, right)
        actual.block_until_ready()
        workers, available = _native_worker_counts_for_tests()
        chunks = _native_reduction_fiber_chunks_for_tests()
    finally:
        _set_native_worker_limit_for_tests(None)

    expected = jnp.sum(materialize(left) * materialize(right))
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
    if available >= 2:
        assert workers >= 2
        assert chunks >= 2


@pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")
def test_native_multirecord_dot_uses_parallel_subdomains() -> None:
    record_size = 65_536
    plan = build_affine_plan(
        records=(
            AffineRecord((record_size,), (1,), 0, (1,), 0),
            AffineRecord(
                (record_size,),
                (-1,),
                2 * record_size - 1,
                (-1,),
                2 * record_size - 1,
            ),
        ),
        output_size=2 * record_size,
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=2 * record_size,
        source_dtype=jnp.float32,
        result_dtype=jnp.float32,
    )
    left = jnp.linspace(-1, 2, plan.source_size, dtype=jnp.float32)
    right = jnp.linspace(3, -2, plan.source_size, dtype=jnp.float32)

    _set_native_worker_limit_for_tests(4)
    try:
        actual = jax.jit(
            lambda lhs, rhs: native_projection_dotu(lhs, rhs, plan)
        )(left, right)
        actual.block_until_ready()
        _, available = _native_worker_counts_for_tests()
        chunks = _native_reduction_fiber_chunks_for_tests()
    finally:
        _set_native_worker_limit_for_tests(None)

    expected = jnp.sum(left * right)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
    if available >= 2:
        assert chunks >= 2


def test_native_dot_rejects_malformed_descriptor() -> None:
    left = jnp.arange(4, dtype=jnp.float32)
    right = jnp.arange(4, dtype=jnp.float32)
    descriptor = bytearray(
        _dot_descriptor(
            sizes=(4,),
            left_strides=(1,),
            left_offset=0,
            left_size=4,
            right_strides=(1,),
            right_offset=0,
            right_size=4,
            dtype_name="float32",
            conjugate_left=False,
        )
    )
    descriptor[0] ^= 0xFF

    with pytest.raises(Exception, match="magic mismatch"):
        _dot_ffi_call(
            left,
            right,
            descriptor=bytes(descriptor),
            conjugate_left=False,
        ).block_until_ready()


@pytest.mark.parametrize("conjugate_left", [False, True])
def test_native_complex_dot_jvp_vjp_and_higher_order(
    conjugate_left: bool,
) -> None:
    left = jnp.asarray(
        [[1 + 2j, -3 + 1j, 2 - 4j], [5 + 0j, -1 - 2j, 3 + 3j]],
        dtype=jnp.complex64,
    )
    right = jnp.asarray(
        [complex(real / 2, -real / 3) for real in range(12)],
        dtype=jnp.complex64,
    )
    left_tangent = jnp.asarray(left * (0.25 - 0.5j), dtype=jnp.complex64)
    right_tangent = jnp.asarray(right * (-0.75 + 0.125j), dtype=jnp.complex64)

    def operation(left_data: jax.Array, right_data: jax.Array) -> jax.Array:
        left_view = _dense(left_data)
        right_view = _pitched(right_data)
        return (
            dotc(left_view, right_view)
            if conjugate_left
            else dotu(left_view, right_view)
        )

    def oracle(left_data: jax.Array, right_data: jax.Array) -> jax.Array:
        left_value = jnp.conj(left_data) if conjugate_left else left_data
        return jnp.sum(left_value * materialize(_pitched(right_data)))

    actual_primal, actual_tangent = jax.jvp(
        operation,
        (left, right),
        (left_tangent, right_tangent),
    )
    expected_primal, expected_tangent = jax.jvp(
        oracle,
        (left, right),
        (left_tangent, right_tangent),
    )
    np.testing.assert_allclose(actual_primal, expected_primal)
    np.testing.assert_allclose(actual_tangent, expected_tangent, rtol=1e-6)

    cotangent = jnp.asarray(0.75 - 1.25j, dtype=jnp.complex64)
    _, actual_pullback = jax.vjp(operation, left, right)
    _, expected_pullback = jax.vjp(oracle, left, right)
    for actual, expected in zip(
        actual_pullback(cotangent),
        expected_pullback(cotangent),
        strict=True,
    ):
        np.testing.assert_allclose(actual, expected, rtol=1e-6)

    def actual_first_tangent(
        left_data: jax.Array,
        right_data: jax.Array,
    ) -> jax.Array:
        return jax.jvp(
            operation,
            (left_data, right_data),
            (left_tangent, right_tangent),
        )[1]

    def expected_first_tangent(
        left_data: jax.Array,
        right_data: jax.Array,
    ) -> jax.Array:
        return jax.jvp(
            oracle,
            (left_data, right_data),
            (left_tangent, right_tangent),
        )[1]

    actual_second = jax.jvp(
        actual_first_tangent,
        (left, right),
        (left_tangent, right_tangent),
    )[1]
    expected_second = jax.jvp(
        expected_first_tangent,
        (left, right),
        (left_tangent, right_tangent),
    )[1]
    np.testing.assert_allclose(actual_second, expected_second, rtol=1e-6)


def test_algebra_jit_and_vmap_chain_views() -> None:
    source = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)
    destination = jnp.arange(12, dtype=jnp.float32)

    @jax.jit
    def operation(x: jax.Array, y: jax.Array, alpha: jax.Array) -> jax.Array:
        result = add(_pitched(y), _dense(x), beta=alpha)
        return materialize(result)

    actual = operation(source, destination, jnp.asarray(2, dtype=jnp.float32))
    expected = 2 * source + materialize(_pitched(destination))
    np.testing.assert_array_equal(actual, expected)

    batched = jax.vmap(operation, in_axes=(0, 0, 0))(
        jnp.stack((source, source + 1)),
        jnp.stack((destination, destination + 1)),
        jnp.asarray([2, -1], dtype=jnp.float32),
    )
    np.testing.assert_array_equal(
        batched[0],
        2 * source + materialize(_pitched(destination)),
    )
    np.testing.assert_array_equal(
        batched[1],
        -(source + 1) + materialize(_pitched(destination + 1)),
    )


def test_add_rejects_shape_batch_mismatch_and_casts_at_write() -> None:
    destination = _pitched(jnp.arange(12, dtype=jnp.float32))
    wrong_shape = StridedView.from_dense(jnp.arange(4, dtype=jnp.float32), (2, 2))
    wrong_batch = StridedView.from_dense(jnp.arange(12, dtype=jnp.float32).reshape(2, 2, 3), (2, 3))
    complex_source = _dense(jnp.arange(6, dtype=jnp.float32).astype(jnp.complex64))

    with pytest.raises(ValueError, match="logical shapes"):
        add(destination, wrong_shape)
    with pytest.raises(ValueError, match="batch shapes"):
        add(destination, wrong_batch)
    np.testing.assert_array_equal(
        add(destination, complex_source).data,
        add(destination, _dense(complex_source.data.real)).data,
    )


def test_add_promotes_source_and_coefficient_to_destination_dtype() -> None:
    source = _dense(jnp.asarray([[1, 2, 3], [4, 5, 6]], dtype=jnp.float16))
    destination = _pitched(jnp.linspace(-1, 1, 12, dtype=jnp.float32))
    alpha = jnp.asarray(1.0003, dtype=jnp.float32)

    actual = materialize(add(destination, source, beta=alpha))
    expected = (
        alpha * jnp.asarray(materialize(source), dtype=jnp.float32)
        + materialize(destination)
    )

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    ("dtype", "alpha_value", "beta_value"),
    (
        (jnp.float16, 0.5, -2),
        (jnp.bfloat16, 0.5, -2),
        (jnp.float32, 0.5, -2),
        (jnp.float64, 0.5, -2),
        (jnp.complex64, 0.5 + 0.25j, -2 + 0.5j),
        (jnp.complex128, 0.5 + 0.25j, -2 + 0.5j),
    ),
)
def test_native_add_uses_one_custom_call(
    dtype: DTypeLike,
    alpha_value: float | complex,
    beta_value: float | complex,
) -> None:
    with jax.enable_x64():
        resolved_dtype = jnp.dtype(dtype)
        source = jnp.arange(6, dtype=jnp.float32).astype(resolved_dtype)
        source = jnp.reshape(source, (2, 3))
        destination = jnp.arange(12, dtype=jnp.float32).astype(resolved_dtype)
        alpha = jnp.asarray(alpha_value, dtype=resolved_dtype)
        beta = jnp.asarray(beta_value, dtype=resolved_dtype)

        @jax.jit
        def operation(
            coefficient: jax.Array,
            source_data: jax.Array,
            destination_coefficient: jax.Array,
            destination_data: jax.Array,
        ) -> jax.Array:
            return add(
                _pitched(destination_data),
                _dense(source_data),
                alpha=destination_coefficient,
                beta=coefficient,
            ).data

        _reset_native_call_count_for_tests()
        actual = operation(alpha, source, beta, destination)
        actual.block_until_ready()
        assert _native_call_count_for_tests() == 1
        destination_indices = jnp.asarray([2, 6, 10, 3, 7, 11])
        expected = destination.at[destination_indices].set(
            jnp.reshape(alpha * source, (-1,))
            + beta * destination[destination_indices]
        )
        np.testing.assert_allclose(actual, expected, rtol=5e-3, atol=5e-3)

        stablehlo = str(
            operation.lower(alpha, source, beta, destination).compiler_ir(
                "stablehlo"
            )
        ).lower()
        assert stablehlo.count("stablehlo.custom_call") == 1
        assert (
            f"tensor0_stride_update_{resolved_dtype.name}_cpu_v1" in stablehlo
        )


def test_native_add_jvp_and_vjp_match_dense_oracle() -> None:
    source = jnp.arange(6, dtype=jnp.float32).reshape(2, 3) + 1
    destination = jnp.arange(12, dtype=jnp.float32)
    alpha = jnp.asarray(0.5, dtype=jnp.float32)
    beta = jnp.asarray(-2, dtype=jnp.float32)
    source_tangent = jnp.linspace(-1, 1, 6, dtype=jnp.float32).reshape(2, 3)
    destination_tangent = jnp.linspace(1, -1, 12, dtype=jnp.float32)
    alpha_tangent = jnp.asarray(0.25, dtype=jnp.float32)
    beta_tangent = jnp.asarray(-0.125, dtype=jnp.float32)

    def operation(
        coefficient: jax.Array,
        source_data: jax.Array,
        destination_coefficient: jax.Array,
        destination_data: jax.Array,
    ) -> jax.Array:
        return materialize(
            add(
                _pitched(destination_data),
                _dense(source_data),
                alpha=destination_coefficient,
                beta=coefficient,
            )
        )

    def oracle(
        coefficient: jax.Array,
        source_data: jax.Array,
        destination_coefficient: jax.Array,
        destination_data: jax.Array,
    ) -> jax.Array:
        return (
            coefficient * source_data
            + destination_coefficient * materialize(_pitched(destination_data))
        )

    primals = (alpha, source, beta, destination)
    tangents = (
        alpha_tangent,
        source_tangent,
        beta_tangent,
        destination_tangent,
    )
    actual_primal, actual_tangent = jax.jvp(operation, primals, tangents)
    expected_primal, expected_tangent = jax.jvp(oracle, primals, tangents)
    np.testing.assert_allclose(actual_primal, expected_primal)
    np.testing.assert_allclose(actual_tangent, expected_tangent, rtol=1e-6)

    cotangent = jnp.linspace(-2, 2, 6, dtype=jnp.float32).reshape(2, 3)
    _, actual_pullback = jax.vjp(operation, *primals)
    _, expected_pullback = jax.vjp(oracle, *primals)
    for actual, expected in zip(
        actual_pullback(cotangent),
        expected_pullback(cotangent),
        strict=True,
    ):
        np.testing.assert_allclose(actual, expected)

    actual_alpha_gradient = jax.grad(
        lambda *arguments: jnp.sum(operation(*arguments) ** 2),
        argnums=0,
    )
    expected_alpha_gradient = jax.grad(
        lambda *arguments: jnp.sum(oracle(*arguments) ** 2),
        argnums=0,
    )
    actual_gradient, actual_gradient_tangent = jax.jvp(
        actual_alpha_gradient,
        primals,
        tangents,
    )
    expected_gradient, expected_gradient_tangent = jax.jvp(
        expected_alpha_gradient,
        primals,
        tangents,
    )
    np.testing.assert_allclose(actual_gradient, expected_gradient)
    np.testing.assert_allclose(
        actual_gradient_tangent,
        expected_gradient_tangent,
        rtol=1e-6,
    )


def test_native_add_complex64_matches_dense_oracle() -> None:
    source_data = jnp.asarray(
        [[1 + 2j, -3 + 1j, 2 - 4j], [5 + 0j, -1 - 2j, 3 + 3j]],
        dtype=jnp.complex64,
    )
    destination_data = jnp.asarray(
        [complex(real, -real / 3) for real in range(12)],
        dtype=jnp.complex64,
    )
    alpha = jnp.asarray(0.75 - 1.25j, dtype=jnp.complex64)
    beta = jnp.asarray(-0.5 + 0.25j, dtype=jnp.complex64)

    actual = materialize(
        add(_pitched(destination_data), _dense(source_data), alpha=beta, beta=alpha)
    )
    expected = (
        alpha * source_data + beta * materialize(_pitched(destination_data))
    )
    np.testing.assert_allclose(actual, expected)


def test_add_elides_complex_unit_multiplication() -> None:
    source_data = jnp.asarray([1 + 0j, 1 + 0j, 1 + 0j], dtype=jnp.complex64)
    destination_data = jnp.asarray(
        [complex(jnp.inf, 0), complex(0, jnp.inf), complex(jnp.nan, 1)],
        dtype=jnp.complex64,
    )
    source = StridedView.from_dense(source_data, (3,))
    destination = StridedView.from_dense(destination_data, (3,))
    alpha = jnp.asarray(2 + 0j, dtype=jnp.complex64)

    actual = materialize(add(destination, source, beta=alpha))
    expected = alpha * source_data + destination_data
    np.testing.assert_allclose(actual, expected, equal_nan=True)
    assert np.asarray(actual)[0].imag == 0
