from __future__ import annotations

from math import prod

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensor0._stride._jax import accumulation_p
from tensor0._stride._layout import AffineRecord, contiguous_strides

from ._support import native_available


pytestmark = pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")


def compact_record(shape, source_offset=0):
    strides = contiguous_strides(shape)
    return AffineRecord(shape, strides, source_offset, strides, 0)


def complex_values(shape):
    values = jnp.arange(prod(shape), dtype=jnp.float32).reshape(shape)
    return (values / 32 + 1j * (values / 64 - 1)).astype(jnp.complex64)


def mapped(source, record, factor):
    return accumulation_p.bind(source, jnp.asarray(factor, dtype=source.dtype), records=(record,),
                               coefficient_records=(0,), output_size=prod(record.logical_shape), dtype=source.dtype)


def reference(source, record, factor):
    elements = prod(record.logical_shape)
    return (source[..., record.source_offset:record.source_offset + elements]
            * jnp.asarray(factor, dtype=source.dtype)).astype(source.dtype)


def assert_native(function, argument):
    text = jax.jit(function).lower(argument).as_text().lower()
    assert text.count("custom_call") == 1
    assert "tensor0_stride_" in text
    assert "gather" not in text and "scatter" not in text and "iota" not in text
    return text


def assert_close(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    if jnp.issubdtype(actual.dtype, jnp.integer):
        np.testing.assert_array_equal(actual, expected)
    else:
        tolerance = 2e-3 if actual.dtype in (jnp.float16, jnp.bfloat16) else 2e-6
        for component in (jnp.real, jnp.imag):
            np.testing.assert_allclose(np.asarray(component(actual)).astype(np.float64),
                                       np.asarray(component(expected)).astype(np.float64),
                                       rtol=tolerance, atol=1e-6)


@pytest.mark.parametrize("dtype,factor,elements", [
    (jnp.bfloat16, .75, 131072), (jnp.float32, .75, 65536),
    (jnp.complex64, .75 - .5j, 32768), (jnp.int32, 3, 65536),
    (jnp.float16, .75, 131072),
])
def test_compact_native_route_with_source_offset_and_padding(dtype, factor, elements):
    record = compact_record((elements,), source_offset=3)
    size = elements + 8
    if jnp.issubdtype(dtype, jnp.complexfloating):
        source = complex_values((size,))
    elif dtype == jnp.float16:
        source = jnp.asarray(np.linspace(-1, 1, size, dtype=np.float16))
    else:
        source = jnp.arange(size, dtype=jnp.float32).astype(dtype)
    function = lambda value: mapped(value, record, factor)
    assert_native(function, source)
    assert_close(jax.jit(function)(source), reference(source, record, factor))


def test_compact_vmap_and_float32_derivatives_use_native():
    record = compact_record((4096,), source_offset=3)
    source = jnp.arange(4 * 4104, dtype=jnp.float32).reshape(4, 4104)
    function = jax.vmap(lambda value: mapped(value, record, .75))
    oracle = lambda value: reference(value, record, .75)
    assert_native(function, source)
    actual = jax.jvp(function, (source,), (source / 8,))
    expected = jax.jvp(oracle, (source,), (source / 8,))
    for result, wanted in zip(actual, expected, strict=True):
        assert_close(result, wanted)
    cotangent = jnp.ones_like(actual[0])
    pullback = lambda value: jax.vjp(function, source)[1](value)[0]
    gradient = jax.jit(pullback)(cotangent)
    assert_close(gradient, jax.vjp(oracle, source)[1](cotangent)[0])
    np.testing.assert_array_equal(gradient[:, :3], 0)
    np.testing.assert_array_equal(gradient[:, -5:], 0)
    assert_native(pullback, cotangent)


def test_compact_native_composes_with_adjacent_elementwise_operations():
    record = compact_record((65536,), source_offset=3)
    source = jnp.arange(65544, dtype=jnp.float32) / 64
    function = lambda value: jnp.tanh(mapped(jnp.sin(value), record, .75))
    assert_native(function, source)
    np.testing.assert_allclose(jax.jit(function)(source), jnp.tanh(reference(jnp.sin(source), record, .75)),
                               rtol=1e-6, atol=1e-6)


def test_batched_compact_complex_mapping_uses_one_native_operation():
    record = compact_record((128, 64), source_offset=3)
    source = complex_values((3, 8200))
    function = lambda value: mapped(value, record, .75 - .5j)
    assert_native(function, source)
    assert_close(jax.jit(function)(source), reference(source, record, .75 - .5j))


def test_complex_nonleading_vmap_jvp_and_vjp():
    record = compact_record((128, 64), source_offset=3)
    source = complex_values((8200, 3))
    direction = source * jnp.complex64(.25 + .125j)
    function = jax.vmap(lambda value: mapped(value, record, .75 - .5j), in_axes=1, out_axes=1)
    oracle = jax.vmap(lambda value: reference(value, record, .75 - .5j), in_axes=1, out_axes=1)
    assert_native(function, source)
    actual = jax.jvp(function, (source,), (direction,))
    expected = jax.jvp(oracle, (source,), (direction,))
    for result, wanted in zip(actual, expected, strict=True):
        assert_close(result, wanted)
    cotangent = complex_values(actual[0].shape)
    pullback = lambda value: jax.vjp(function, source)[1](value)[0]
    gradient = jax.jit(pullback)(cotangent)
    assert_close(gradient, jax.vjp(oracle, source)[1](cotangent)[0])
    np.testing.assert_array_equal(gradient[:3], 0)
    np.testing.assert_array_equal(gradient[-5:], 0)
    assert_native(pullback, cotangent)


def test_compact_hlo_operation_count_is_independent_of_element_count():
    counts = []
    for elements in (64, 16384):
        record = compact_record((elements,))
        source = complex_values((2, elements))
        text = assert_native(lambda value: mapped(value, record, .75 - .5j), source)
        counts.append(text.count("stablehlo."))
    assert counts[0] == counts[1]


@pytest.mark.parametrize("elements", [64, 65536])
def test_small_and_large_compact_complex_routes(elements):
    record = compact_record((elements,))
    source = complex_values((elements,))
    function = lambda value: mapped(value, record, .75 - .5j)
    assert_native(function, source)
    assert_close(jax.jit(function)(source), reference(source, record, .75 - .5j))


def test_positive_affine_permutation_uses_no_address_arrays():
    record = AffineRecord((8, 16), (1, 8), 0, (16, 1), 0)
    source = complex_values((2, 128))
    function = lambda value: mapped(value, record, .75 - .5j)
    assert_native(function, source)
    expected = source.reshape(2, 16, 8).transpose(0, 2, 1).reshape(2, 128) * jnp.complex64(.75 - .5j)
    assert_close(jax.jit(function)(source), expected)
