from __future__ import annotations

from math import prod

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensor0._stride import enable_threads, get_num_threads, set_num_threads
from tensor0._stride._ffi._descriptor import encode_layout
from tensor0._stride._ffi._registration import operation_target
from tensor0._stride._jax import update_p
from tensor0._stride._layout import AffineRecord, contiguous_strides

from ._support import native_available


pytestmark = pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")
MIXED = [("float16", "float32", -1.25), ("float32", "complex64", .5 + .75j),
         ("complex64", "float32", -.75)]


def destination_layout(shape, fastest_first, signs):
    absolute = [0] * len(shape)
    stride = 1
    for axis in fastest_first:
        absolute[axis] = stride
        stride *= shape[axis]
    strides = tuple(sign * stride for sign, stride in zip(signs, absolute, strict=True))
    offset = sum((extent - 1) * stride for extent, stride, sign in zip(shape, absolute, signs, strict=True)
                 if sign < 0)
    return strides, offset


def signed_record(physical_shape=(2, 3), logical_to_physical=(0, 1), source_signs=(-1, 1),
                  destination_fastest=(1, 0), destination_signs=(1, 1)):
    physical_strides = contiguous_strides(physical_shape)
    shape = tuple(physical_shape[axis] for axis in logical_to_physical)
    source_strides = tuple(sign * physical_strides[axis]
                           for sign, axis in zip(source_signs, logical_to_physical, strict=True))
    source_offset = sum((physical_shape[axis] - 1) * physical_strides[axis]
                        for sign, axis in zip(source_signs, logical_to_physical, strict=True) if sign < 0)
    strides, offset = destination_layout(shape, destination_fastest, destination_signs)
    return AffineRecord(shape, source_strides, source_offset, strides, offset)


def broadcast_record(shape=(2, 3), broadcast_axes=(0,), source_signs=(-1,),
                     destination_fastest=(1, 0), destination_signs=(1, 1)):
    axes = tuple(axis for axis, extent in enumerate(shape) if extent > 1 and axis not in broadcast_axes)
    source_shape = tuple(shape[axis] for axis in axes)
    source_strides = [0] * len(shape)
    source_offset = 0
    for axis, sign, stride in zip(axes, source_signs, contiguous_strides(source_shape), strict=True):
        source_strides[axis] = sign * stride
        if sign < 0:
            source_offset += (shape[axis] - 1) * stride
    strides, offset = destination_layout(shape, destination_fastest, destination_signs)
    return AffineRecord(shape, tuple(source_strides), source_offset, strides, offset), prod(source_shape)


def values(dtype, size):
    real = np.linspace(-2, 3, size, dtype=np.float32)
    return jnp.asarray(real + 1j * real[::-1] if dtype.startswith("complex") else real, dtype=dtype)


def indices(record):
    coordinates = np.indices(record.logical_shape, dtype=np.int64)
    source = np.full(record.logical_shape, record.source_offset, dtype=np.int64)
    destination = np.full(record.logical_shape, record.destination_offset, dtype=np.int64)
    for axis, coordinate in enumerate(coordinates):
        source += coordinate * record.source_strides[axis]
        destination += coordinate * record.destination_strides[axis]
    return source.ravel(), destination.ravel()


def native(source, record, factor, dtype):
    output = jnp.zeros((*source.shape[:-1], prod(record.logical_shape)), dtype=dtype)
    coefficient = jnp.asarray(factor, dtype=jnp.complex64 if isinstance(factor, complex) else jnp.float32)
    return update_p.bind(source, output, coefficient, jnp.float32(0), records=(record,))


def reference(source, record, factor, dtype):
    source_indices, destination_indices = indices(record)
    selected = source[..., source_indices] * jnp.asarray(factor)
    if not jnp.issubdtype(jnp.dtype(dtype), jnp.complexfloating):
        selected = jnp.real(selected)
    return jnp.zeros((*source.shape[:-1], prod(record.logical_shape)), dtype=dtype).at[
        ..., destination_indices].set(selected.astype(dtype))


def assert_close(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    for component in (jnp.real, jnp.imag):
        np.testing.assert_allclose(component(actual), component(expected), rtol=2e-3, atol=1e-6, equal_nan=True)


def restore_threads(previous):
    if previous is None:
        enable_threads()
    else:
        set_num_threads(previous)


def test_signed_affine_uses_address_only_descriptor():
    record = signed_record(physical_shape=(4, 3), destination_fastest=(0, 1))
    layout = encode_layout((record,), source_size=12, output_size=12)
    np.testing.assert_array_equal(layout, [1, 12, 12, 1, 2, 9, 0, 4, 3, -3, 1, 1, 4])


@pytest.mark.parametrize("permutation,source_signs,destination_fastest,destination_signs", [
    ((0, 1), (-1, 1), (1, 0), (-1, 1)),
    ((0, 1), (1, -1), (0, 1), (1, -1)),
    ((1, 0), (-1, -1), (1, 0), (1, -1)),
    ((1, 0), (1, -1), (0, 1), (-1, 1)),
])
def test_native_executes_signed_permuted_layouts(permutation, source_signs, destination_fastest, destination_signs):
    record = signed_record(logical_to_physical=permutation, source_signs=source_signs,
                           destination_fastest=destination_fastest, destination_signs=destination_signs)
    source = values("float32", 6)
    compiled = jax.jit(lambda value: native(value, record, 1.25, "float32"))
    assert_close(compiled(source), reference(source, record, 1.25, "float32"))
    text = compiled.lower(source).as_text().lower()
    assert text.count("custom_call") == 1
    assert "tensor0_stride_update_" in text
    assert "gather" not in text and "scatter" not in text


@pytest.mark.parametrize("source_dtype,result_dtype,factor", MIXED)
@pytest.mark.parametrize("broadcast", [False, True])
def test_signed_and_broadcast_batch_vmap_jvp_vjp(source_dtype, result_dtype, factor, broadcast):
    if broadcast:
        record, size = broadcast_record(broadcast_axes=(1,), destination_signs=(-1, 1))
    else:
        record = signed_record(physical_shape=(4, 3), destination_fastest=(0, 1), destination_signs=(-1, 1))
        size = 12
    source = values(source_dtype, size)
    tangent = (source * .25 + .5).astype(source.dtype)
    batch = jnp.stack((source, -source, 2 * source))
    execute = lambda value: native(value, record, factor, result_dtype)
    oracle = lambda value: reference(value, record, factor, result_dtype)
    assert_close(jax.jit(execute)(batch), oracle(batch))
    assert_close(jax.jit(jax.vmap(execute))(batch), oracle(batch))
    assert_close(jax.jvp(execute, (source,), (tangent,))[1], jax.jvp(oracle, (source,), (tangent,))[1])
    cotangent = values(result_dtype, prod(record.logical_shape))
    assert_close(jax.vjp(execute, source)[1](cotangent)[0], jax.vjp(oracle, source)[1](cotangent)[0])


@pytest.mark.parametrize("source_dtype,result_dtype,factor", MIXED)
def test_large_mixed_rank2_forward_and_reverse_under_thread_limits(source_dtype, result_dtype, factor):
    rows = columns = 512
    record = AffineRecord((rows, columns), (1, rows), 0, (columns, 1), 0)
    source = values(source_dtype, rows * columns)
    execute = lambda value: native(value, record, factor, result_dtype)
    oracle = lambda value: reference(value, record, factor, result_dtype)
    compiled = jax.jit(execute)
    expected = oracle(source)
    cotangent = values(result_dtype, rows * columns)
    pullback = jax.jit(jax.vjp(execute, source)[1])
    expected_vjp = jax.vjp(oracle, source)[1](cotangent)[0]
    previous = get_num_threads()
    try:
        for limit in (1, 4):
            set_num_threads(limit)
            assert_close(compiled(source), expected)
            assert_close(pullback(cotangent)[0], expected_vjp)
    finally:
        restore_threads(previous)


def test_native_rejects_signed_stride_address_overflow():
    record = signed_record(physical_shape=(4, 3), destination_fastest=(0, 1))
    layout = encode_layout((record,), source_size=12, output_size=12)
    layout[9] = -(1 << 63)
    call = jax.ffi.ffi_call(operation_target("copy", np.dtype(jnp.float32)),
                            jax.ShapeDtypeStruct((1, 12), jnp.float32))
    with pytest.raises(Exception, match="source address arithmetic overflow"):
        call(jnp.arange(12, dtype=jnp.float32).reshape(1, 12), layout=layout).block_until_ready()


def test_broadcast_requires_no_separate_axis_metadata():
    record = AffineRecord((2, 3), (0, 1), 0, (3, 1), 0)
    layout = encode_layout((record,), source_size=3, output_size=6)
    np.testing.assert_array_equal(layout, [1, 3, 6, 1, 2, 0, 0, 2, 3, 0, 1, 3, 1])
    source = jnp.arange(3, dtype=jnp.float32)
    np.testing.assert_array_equal(native(source, record, 1, "float32"), jnp.tile(source, 2))


@pytest.mark.parametrize("axes,signs,fastest,destination_signs", [
    ((0,), (1,), (1, 0), (1, 1)), ((0,), (-1,), (0, 1), (-1, 1)),
    ((1,), (-1,), (1, 0), (1, -1)), ((0, 1), (), (0, 1), (-1, -1)),
])
def test_native_executes_signed_broadcast_layouts(axes, signs, fastest, destination_signs):
    record, size = broadcast_record(broadcast_axes=axes, source_signs=signs,
                                    destination_fastest=fastest, destination_signs=destination_signs)
    source = values("float32", size)
    assert_close(jax.jit(lambda value: native(value, record, -.75, "float32"))(source),
                 reference(source, record, -.75, "float32"))


@pytest.mark.parametrize("extent", [2, 3, 4, 7, 16, 64])
def test_broadcast_contiguous_row_extents(extent):
    record, size = broadcast_record(shape=(257, extent), broadcast_axes=(1,), source_signs=(1,))
    source = values("float32", size)
    assert_close(jax.jit(lambda value: native(value, record, 1.25, "float32"))(source),
                 reference(source, record, 1.25, "float32"))


def test_broadcast_real_complex_multiply_uses_real_operand_form():
    record, size = broadcast_record(shape=(8, 7), broadcast_axes=(1,), source_signs=(1,))
    source = jnp.asarray([0., -0., np.inf, -np.inf, np.nan, np.finfo(np.float32).max,
                          -np.finfo(np.float32).max, np.finfo(np.float32).tiny], dtype=jnp.float32)
    assert size == source.size
    actual = jax.jit(lambda value: native(value, record, complex(np.nan, 1), "complex64"))(source)
    assert np.isnan(np.asarray(actual.real)).all()
    np.testing.assert_array_equal(actual.imag.reshape(8, 7), jnp.broadcast_to(source[:, None], (8, 7)))
    assert np.signbit(np.asarray(actual.imag).reshape(8, 7)[1]).all()


def test_large_broadcast_under_thread_limits():
    record, size = broadcast_record(shape=(4096, 512), broadcast_axes=(1,), source_signs=(1,))
    source = jnp.arange(size, dtype=jnp.float32)
    compiled = jax.jit(lambda value: native(value, record, 1, "float32"))
    previous = get_num_threads()
    try:
        for limit in (1, 4):
            set_num_threads(limit)
            np.testing.assert_array_equal(compiled(source).reshape(4096, 512),
                                           jnp.broadcast_to(source[:, None], (4096, 512)))
    finally:
        restore_threads(previous)


def test_large_repeated_source_forward_and_vjp_accumulate():
    rows, columns = 1024, 512
    record = AffineRecord((rows, columns), (1, 1), 0, (columns, 1), 0)
    source = values("float32", rows + columns - 1)
    addresses = np.arange(rows)[:, None] + np.arange(columns)[None, :]
    expected = (source[addresses] * jnp.float32(1.25)).reshape(-1)
    counts = np.bincount(addresses.ravel(), minlength=source.size)
    execute = lambda value: native(value, record, 1.25, "float32")
    compiled = jax.jit(execute)
    pullback = jax.jit(jax.vjp(execute, source)[1])
    previous = get_num_threads()
    try:
        for limit in (1, 4):
            set_num_threads(limit)
            assert_close(compiled(source), expected)
            assert_close(pullback(jnp.ones(rows * columns, dtype=jnp.float32))[0],
                         jnp.asarray(counts * 1.25, dtype=source.dtype))
    finally:
        restore_threads(previous)
