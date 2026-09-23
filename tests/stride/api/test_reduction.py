"""Public reduction axes, dtype and layout contracts."""

from math import prod

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensor0._stride import StridedView, _jax, reduce_sum

from tests.stride.support.availability import native_available


pytestmark = pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')


@pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3), (0,), (2, 0)])
@pytest.mark.parametrize("axes", [None, (), (0,), (1,), (-1,), (1, 0)])
@pytest.mark.parametrize("shape,strides,offset,storage_size", [
    ((2, 3), (3, 1), 0, 6),
    ((2, 3), (1, 2), 1, 8),
    ((2, 3), (-3, -1), 6, 8),
    ((2, 3), (0, 1), 1, 4),
    ((0, 3), (3, 1), 0, 0),
    ((2, 0), (1, 1), 0, 0),
])
def test_public_sum_layout_axes_and_batches(batch_shape, axes, shape, strides, offset, storage_size):
    source = jnp.arange(prod(batch_shape) * storage_size, dtype=jnp.float32).reshape(
        (*batch_shape, storage_size))
    view = StridedView(source, shape, strides, offset)
    indices = np.asarray([
        offset + sum(coordinate * stride for coordinate, stride in zip(index, strides))
        for index in np.ndindex(shape)
    ], dtype=np.int64).reshape(shape)
    selected = np.asarray(source)[..., indices]
    logical_axes = tuple(range(len(shape))) if axes is None else tuple(axis % len(shape) for axis in axes)
    expected = selected.sum(axis=tuple(len(batch_shape) + axis for axis in logical_axes))
    for execute in (lambda bound: reduce_sum(bound, axes), jax.jit(lambda bound: reduce_sum(bound, axes))):
        actual = execute(view)
        assert actual.shape == expected.shape
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("x64", [False, True])
@pytest.mark.parametrize("dtype", [
    "bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
    "float16", "bfloat16", "float32", "float64", "complex64", "complex128",
])
def test_public_sum_default_dtype_resolution(dtype, x64):
    with jax.enable_x64(x64):
        concrete = jax.dtypes.canonicalize_dtype(jnp.dtype(dtype))
        source = jnp.asarray([0, 1, 2, 3], dtype=concrete)
        actual = reduce_sum(StridedView.from_dense(source, (2, 2)))
        expected = jnp.sum(source)
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("axes", [None, ()])
def test_public_sum_scalar_view(axes):
    source = jnp.asarray([[5, 7, 9], [11, 13, 15]], dtype=jnp.int8)
    actual = reduce_sum(StridedView(source, (), (), 1), axes)
    np.testing.assert_array_equal(actual, [7, 13])
    assert actual.dtype == jnp.sum(source[0, 1]).dtype


def test_public_sum_real_output_layout(monkeypatch):
    encode = _jax.encode_reduction_layout
    captured = []

    def capture(records, **fields):
        captured.append(dict(records=records, **fields))
        return encode(records, **fields)

    monkeypatch.setattr(_jax, "encode_reduction_layout", capture)
    view = StridedView.from_dense(jnp.arange(24, dtype=jnp.float32), (2, 3, 4))
    actual = reduce_sum(view, (1,))
    np.testing.assert_array_equal(actual, np.arange(24).reshape(2, 3, 4).sum(axis=1))
    assert len(captured) == 1
    assert captured[0]["output_shapes"] == ((2, 1, 4),)
    assert captured[0]["records"][0].destination_strides == (4, 4, 1)
    assert captured[0]["reduction_axes"] == ((False, True, False),)


def test_public_sum_dtype_does_not_preconvert_source():
    view = StridedView.from_dense(jnp.asarray([1.75, 1.75]), (2,))
    np.testing.assert_array_equal(reduce_sum(view, dtype=jnp.int32), 3)
    np.testing.assert_array_equal(jnp.sum(view.data, dtype=jnp.int32), 2)


def test_public_sum_low_precision_uses_native_accumulation():
    view = StridedView.from_dense(jnp.asarray([2048, 1, -2048], dtype=jnp.float16), (3,))
    np.testing.assert_array_equal(reduce_sum(view), 0)
    np.testing.assert_array_equal(jnp.sum(view.data), 1)
    np.testing.assert_array_equal(reduce_sum(view, dtype=jnp.float32), 1)


def test_public_sum_empty_axes_do_not_add_zero():
    view = StridedView.from_dense(jnp.asarray([-0.0], dtype=jnp.float32), (1,))
    assert np.signbit(np.asarray(reduce_sum(view, ())))[0]


def test_public_sum_explicit_dtype_respects_x64():
    with jax.enable_x64(False):
        view = StridedView.from_dense(jnp.asarray([1, 2], dtype=jnp.float32), (2,))
        assert reduce_sum(view, dtype=jnp.float64).dtype == jnp.float32


def test_public_sum_high_rank_singleton_layout():
    view = StridedView(jnp.asarray([7], dtype=jnp.int32), (1,) * 12, ((1 << 63) - 1,) * 12, 0)
    np.testing.assert_array_equal(reduce_sum(view), 7)


def test_public_sum_vmap_and_native_lowering():
    execute = lambda source: reduce_sum(StridedView.from_dense(source, (2, 3)), (1,))
    source = jnp.arange(24, dtype=jnp.float32).reshape(4, 2, 3)
    actual = jax.jit(jax.vmap(execute))(source)
    np.testing.assert_array_equal(actual, np.asarray(source).sum(axis=2))
    lowered = str(jax.jit(execute).lower(source[0]).compiler_ir())
    assert "tensor0_stride_reduction_f32_cpu_v1" in lowered


@pytest.mark.parametrize("axes,error", [((True,), TypeError), ((0.0,), TypeError),
                                       ((2,), ValueError), ((-3,), ValueError),
                                       ((1, -1), ValueError)])
def test_public_sum_invalid_axes(axes, error):
    view = StridedView.from_dense(jnp.ones((2, 3)), (2, 3))
    with pytest.raises(error):
        reduce_sum(view, axes)


def test_public_sum_unsupported_calls_fail_explicitly():
    with pytest.raises(TypeError, match="StridedView"):
        reduce_sum(jnp.ones(3))
