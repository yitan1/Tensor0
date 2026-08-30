from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensor0._stride import StridedView, materialize


def _addresses(view: StridedView) -> list[int]:
    if view.element_count == 0:
        return []
    result: list[int] = []
    for coordinate in np.ndindex(view.sizes):
        result.append(
            view.offset
            + sum(
                index * stride
                for index, stride in zip(coordinate, view.strides, strict=True)
            )
        )
    return result


def test_strided_view_binds_jax_data_and_affine_metadata() -> None:
    data = jnp.arange(24, dtype=jnp.float32).reshape(2, 12)
    view = StridedView(data, (2, 3), (1, 4), 2)

    assert view.data is data
    assert view.sizes == (2, 3)
    assert view.strides == (1, 4)
    assert view.offset == 2
    assert view.rank == 2
    assert view.element_count == 6
    assert view.storage_size == 12
    assert view.batch_shape == (2,)
    with pytest.raises(TypeError, match="unhashable"):
        hash(view)


def test_strided_view_from_dense_preserves_batch_and_logical_order() -> None:
    data = jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4)
    view = StridedView.from_dense(data, (3, 4))

    assert view.data.shape == (2, 12)
    assert view.sizes == (3, 4)
    assert view.strides == (4, 1)
    np.testing.assert_array_equal(view.data, data.reshape(2, 12))


def test_strided_view_from_dense_roundtrips_batched_rank_zero() -> None:
    dense = jnp.asarray([1.5, -2.0, 3.25], dtype=jnp.float32)
    view = StridedView.from_dense(dense, ())

    assert view.data.shape == (3, 1)
    assert view.sizes == ()
    assert view.strides == ()
    np.testing.assert_array_equal(materialize(view), dense)


def test_strided_view_rejects_non_jax_and_out_of_bounds_data() -> None:
    with pytest.raises(TypeError, match="data must be a JAX Array"):
        StridedView(np.arange(8, dtype=np.float32), (2,), (1,), 0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="addresses exceed storage bounds"):
        StridedView(jnp.arange(8), (3,), (2,), 4)
    with pytest.raises(ValueError, match="offset exceeds storage size"):
        StridedView(jnp.arange(8), (0,), (1,), 9)


def test_strided_view_is_a_pytree_with_only_data_as_dynamic_child() -> None:
    data = jnp.arange(12, dtype=jnp.float32)
    view = StridedView(data, (2, 3), (1, 4), 2)
    leaves, structure = jax.tree.flatten(view)
    rebuilt = jax.tree.unflatten(structure, leaves)

    assert len(leaves) == 1
    assert leaves[0] is data
    assert isinstance(rebuilt, StridedView)
    assert rebuilt.data is data
    assert rebuilt.sizes == view.sizes
    assert rebuilt.strides == view.strides
    assert rebuilt.offset == view.offset


def test_strided_view_permute_only_changes_affine_metadata() -> None:
    data = jnp.arange(20, dtype=jnp.float32)
    view = StridedView(data, (2, 3), (1, 4), 2)
    permuted = view.permute((1, 0))

    assert permuted.data is data
    assert permuted.sizes == (3, 2)
    assert permuted.strides == (4, 1)
    assert permuted.offset == 2
    np.testing.assert_array_equal(
        np.asarray(_addresses(permuted)).reshape(permuted.sizes),
        np.asarray(_addresses(view)).reshape(view.sizes).T,
    )


def test_strided_view_subview_supports_integer_and_signed_slice_axes() -> None:
    data = jnp.arange(30, dtype=jnp.float32)
    view = StridedView(data, (3, 4), (4, 1), 2)
    sliced = view.subview(slice(None, None, -1), slice(1, 4, 2))
    row = view.subview(1, slice(None, None, -1))

    assert sliced.data is data
    assert sliced.sizes == (3, 2)
    assert sliced.strides == (-4, 2)
    assert sliced.offset == 11
    assert _addresses(sliced) == [11, 13, 7, 9, 3, 5]
    assert row.sizes == (4,)
    assert row.strides == (-1,)
    assert row.offset == 9
    assert _addresses(row) == [9, 8, 7, 6]


def test_strided_view_reshape_accepts_only_address_preserving_order() -> None:
    data = jnp.arange(24, dtype=jnp.float32)
    compact = StridedView(data, (2, 3), (3, 1), 4)
    signed = StridedView(data, (6,), (-1,), 9)
    permuted = StridedView(data, (2, 3), (1, 2), 0)

    reshaped = compact.reshape((3, 2))
    signed_reshaped = signed.reshape((2, 3))

    assert reshaped.sizes == (3, 2)
    assert reshaped.strides == (2, 1)
    assert _addresses(reshaped) == _addresses(compact)
    assert signed_reshaped.strides == (-3, -1)
    assert _addresses(signed_reshaped) == _addresses(signed)
    with pytest.raises(ValueError, match="not affine-compatible"):
        permuted.reshape((6,))


def test_empty_strided_view_reshape_remains_metadata_only() -> None:
    data = jnp.arange(4, dtype=jnp.float32)
    empty = StridedView(data, (2, 0, 3), (3, 3, 1), 2)
    reshaped = empty.reshape((0, 7))

    assert reshaped.data is data
    assert reshaped.sizes == (0, 7)
    assert reshaped.strides == (7, 1)
    assert reshaped.offset == 2


def test_strided_view_can_flow_as_a_jitted_pytree_argument() -> None:
    data = jnp.arange(12, dtype=jnp.float32)
    view = StridedView(data, (2, 3), (3, 1), 2)

    actual = jax.jit(lambda value: value.data + value.offset)(view)

    np.testing.assert_array_equal(actual, data + 2)
