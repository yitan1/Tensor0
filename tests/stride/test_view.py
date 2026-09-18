from __future__ import annotations

from dataclasses import FrozenInstanceError
from math import prod

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensor0._stride import StridedView, materialize
from tensor0._stride._layout import INT64_MAX, INT64_MIN, UINT64_MAX, contiguous_strides


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
    with pytest.raises(FrozenInstanceError):
        view.offset = 0


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
    with pytest.raises(ValueError, match="at least one storage axis"):
        StridedView(jnp.asarray(1), (), (), 0)
    with pytest.raises(ValueError, match="dense data shape"):
        StridedView.from_dense(jnp.arange(4), (3,))


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


def test_strided_view_rebinding_preserves_metadata_and_requires_storage_shape() -> None:
    data = jnp.arange(12, dtype=jnp.float32)
    view = StridedView(data, (2, 3), (1, 4), 2)
    replacement = jnp.linspace(-1, 1, 12, dtype=jnp.float32)

    rebound = view._with_data(replacement)

    assert rebound.data is replacement
    assert rebound.sizes == view.sizes
    assert rebound.strides == view.strides
    assert rebound.offset == view.offset
    with pytest.raises(ValueError, match="replacement storage shape"):
        view._with_data(jnp.arange(13, dtype=jnp.float32))
    with pytest.raises(TypeError, match="data must be a JAX Array"):
        view._with_data(np.arange(12, dtype=np.float32))  # type: ignore[arg-type]


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


@pytest.mark.parametrize("leaf", [object(), None, jax.ShapeDtypeStruct((12,), jnp.float32)])
def test_pytree_rebuild_accepts_metadata_without_relaxing_constructor(leaf) -> None:
    view = StridedView(jnp.arange(12, dtype=jnp.float32), (2, 3), (1, 4), 2)
    _, tree = jax.tree.flatten(view)
    rebuilt = jax.tree.unflatten(tree, [leaf])
    assert isinstance(rebuilt, StridedView)
    assert rebuilt.data is leaf
    assert (rebuilt.sizes, rebuilt.strides, rebuilt.offset) == ((2, 3), (1, 4), 2)
    with pytest.raises(FrozenInstanceError):
        rebuilt.offset = 0
    with pytest.raises(TypeError, match="data must be a JAX Array"):
        StridedView(leaf, view.sizes, view.strides, view.offset)


def test_view_lower_compile_and_eval_shape() -> None:
    view = StridedView(jnp.arange(12, dtype=jnp.float32), (2, 3), (1, 4), 2)

    def transform(value):
        return value._with_data(value.data + 1)

    actual = jax.jit(transform).lower(view).compile()(view)
    np.testing.assert_array_equal(actual.data, view.data + 1)
    assert (actual.sizes, actual.strides, actual.offset) == ((2, 3), (1, 4), 2)
    abstract = jax.eval_shape(transform, view)
    assert isinstance(abstract, StridedView)
    assert isinstance(abstract.data, jax.ShapeDtypeStruct)
    assert abstract.data.shape == view.data.shape
    assert abstract.data.dtype == view.data.dtype
    assert (abstract.sizes, abstract.strides, abstract.offset) == (view.sizes, view.strides, view.offset)


@pytest.mark.parametrize("shape,expected", [
    ((), ()), ((4,), (1,)), ((2, 3), (3, 1)),
    ((2, 0, 3), (3, 3, 1)), ((0, 0), (1, 1)), ((1,) * 12, (1,) * 12),
])
def test_contiguous_strides(shape, expected) -> None:
    assert contiguous_strides(shape) == expected


@pytest.mark.parametrize("storage_shape,sizes,strides,offset", [
    ((12,), (3,), (-2,), 8),
    ((1,), (3, 4), (0, 0), 0),
    ((1,), (), (), 0),
    ((0,), (0,), (INT64_MIN,), 0),
    ((3,), (0,), (INT64_MAX,), 3),
    ((1,), (1,) * 12, (INT64_MAX,) * 12, 0),
    ((1,), (UINT64_MAX,), (0,), 0),
])
def test_view_metadata_boundaries(storage_shape, sizes, strides, offset) -> None:
    data = jnp.arange(prod(storage_shape), dtype=jnp.float32).reshape(storage_shape)
    view = StridedView(data, sizes, strides, offset)
    assert view.data is data
    assert (view.sizes, view.strides, view.offset) == (sizes, strides, offset)
    assert view.rank == len(sizes)
    assert view.element_count == prod(sizes)
    assert view.storage_size == storage_shape[-1]
    assert view.batch_shape == storage_shape[:-1]


@pytest.mark.parametrize("sizes,strides,offset,error", [
    ([2], (1,), 0, TypeError),
    ((True,), (1,), 0, TypeError),
    ((-1,), (1,), 0, ValueError),
    ((UINT64_MAX + 1,), (0,), 0, ValueError),
    ((2,), [1], 0, TypeError),
    ((2,), (False,), 0, TypeError),
    ((1,), (INT64_MIN - 1,), 0, ValueError),
    ((1,), (INT64_MAX + 1,), 0, ValueError),
    ((2,), (), 0, ValueError),
    ((1,), (1,), True, TypeError),
    ((1,), (1,), -1, ValueError),
    ((0,), (1,), UINT64_MAX + 1, ValueError),
    ((2,), (-1,), 0, ValueError),
])
def test_invalid_view_metadata(sizes, strides, offset, error) -> None:
    with pytest.raises(error):
        StridedView(jnp.arange(8), sizes, strides, offset)


@pytest.mark.parametrize("shape,sizes,storage_shape", [
    ((2, 12), (3, 4), (2, 12)),
    ((), (), (1,)),
    ((2, 0, 3), (0, 3), (2, 0)),
])
def test_from_dense_flat_scalar_and_empty_layouts(shape, sizes, storage_shape) -> None:
    data = jnp.arange(prod(shape), dtype=jnp.float32).reshape(shape)
    view = StridedView.from_dense(data, sizes)
    assert view.data.shape == storage_shape
    assert (view.sizes, view.strides, view.offset) == (sizes, contiguous_strides(sizes), 0)
    np.testing.assert_array_equal(view.data, np.asarray(data).reshape(storage_shape))


@pytest.mark.parametrize("indices", [(-1, -2), (slice(0, 0), slice(None))])
def test_negative_index_and_empty_subview(indices) -> None:
    view = StridedView(jnp.arange(30), (3, 4), (4, 1), 2)
    actual = view.subview(*indices)
    expected = np.asarray(_addresses(view)).reshape(view.sizes)[indices]
    assert actual.data is view.data
    assert actual.sizes == expected.shape
    np.testing.assert_array_equal(np.asarray(_addresses(actual)).reshape(actual.sizes), expected)


@pytest.mark.parametrize("sizes,strides,new_sizes", [
    ((1, 1), (INT64_MAX, INT64_MIN), ()),
    ((2, 3), (0, 0), (6,)),
])
def test_singleton_and_broadcast_reshape(sizes, strides, new_sizes) -> None:
    view = StridedView(jnp.arange(24), sizes, strides, 0)
    actual = view.reshape(new_sizes)
    assert actual.data is view.data
    assert actual.sizes == new_sizes
    assert _addresses(actual) == _addresses(view)


@pytest.mark.parametrize("method,args,error", [
    ("permute", ([1, 0],), TypeError),
    ("permute", ((0, 0),), ValueError),
    ("subview", (slice(None),), IndexError),
    ("subview", (True, slice(None)), TypeError),
    ("subview", (2, slice(None)), IndexError),
    ("subview", (slice(None), 0.5), TypeError),
    ("subview", (slice(None, None, 0), slice(None)), ValueError),
    ("reshape", ((7,),), ValueError),
])
def test_invalid_view_transforms(method, args, error) -> None:
    view = StridedView(jnp.arange(6), (2, 3), (1, 2), 0)
    with pytest.raises(error):
        getattr(view, method)(*args)


def test_singleton_slice_stride_overflow() -> None:
    view = StridedView(jnp.ones(1), (1,), (INT64_MAX,), 0)
    with pytest.raises(ValueError, match="subview stride exceeds int64"):
        view.subview(slice(None, None, 2))


def test_jitted_view_permute_and_rebind() -> None:
    view = StridedView(jnp.arange(12, dtype=jnp.float32), (2, 3), (1, 4), 2)
    actual = jax.jit(lambda value: value.permute((1, 0))._with_data(value.data + 1))(view)
    assert isinstance(actual, StridedView)
    assert (actual.sizes, actual.strides, actual.offset) == ((3, 2), (4, 1), 2)
    np.testing.assert_array_equal(actual.data, view.data + 1)
