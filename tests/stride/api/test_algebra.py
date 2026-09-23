"""Public view algebra, coefficient order and input preservation."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.typing import DTypeLike

import tensor0
import tensor0._stride as stride_api
from tensor0._stride import StridedView, add, dotc, dotu, materialize, reduce_sum, scale

from tests.stride.support.views import _dense, _pitched


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
    for factor, values in ((beta, right_data),
                           (alpha, left_data[indices])):
        if factor != 0:
            if factor == 1:
                terms.append(values)
            elif jnp.issubdtype(values.dtype, jnp.complexfloating):
                terms.append(jax.lax.complex(factor * values.real, factor * values.imag))
            else:
                terms.append(factor * values)
    selected = (jnp.zeros((3,), dtype=result_dtype) if not terms
                else terms[0] if len(terms) == 1 else terms[0] + terms[1])
    expected = left_data.at[indices].set(selected.astype(result_dtype))
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


def test_reduce_sum_preserves_jax_integer_promotion() -> None:
    source = jnp.arange(12, dtype=jnp.int8).reshape(3, 4)

    actual = reduce_sum(StridedView.from_dense(source, (3, 4)))
    expected = jnp.sum(source, axis=(0, 1))

    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)


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


def test_add_uses_mixed_coefficient_product_before_storage_conversion() -> None:
    source = _dense(jnp.asarray([[1, 2, 3], [4, 5, 6]], dtype=jnp.float16))
    destination = _pitched(jnp.linspace(-1, 1, 12, dtype=jnp.float32))
    alpha = jnp.asarray(1.0003, dtype=jnp.float32)

    actual = materialize(add(destination, source, beta=alpha))
    expected = (
        alpha * jnp.asarray(materialize(source), dtype=jnp.float32)
        + materialize(destination)
    )

    np.testing.assert_array_equal(actual, expected)


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
