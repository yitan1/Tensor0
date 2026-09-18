"""Complex layout and arithmetic acceptance, independent of kernel selection."""

from __future__ import annotations

from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensor0._stride import StridedView, enable_threads, get_num_threads, scale, set_num_threads
from tensor0._stride._jax import accumulation_p
from tensor0._stride._layout import AffineRecord

from ._support import native_available


pytestmark = pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")
FACTORS = [complex(0., -0.), .125 + 1j, 2.5 + 3.5j, -1.25 + .75j]
COMPONENTS = np.asarray([0., -0., 1.25, -2.5, .125, 1e10, -1e10, 1e-10, 1e-20], dtype=np.float32)


def rank_two(rows, columns, forward):
    source = (1, rows) if forward else (columns, 1)
    destination = (columns, 1) if forward else (1, rows)
    return AffineRecord((rows, columns), source, 0, destination, 0)


def values(size):
    indices = jnp.arange(size, dtype=jnp.float32)
    return ((indices % 4093) / 1024 - 2 + 1j * (.5 - (indices % 4079) / 2048)).astype(jnp.complex64)


def finite_components(size):
    data = np.asarray([complex(real, imaginary) for real, imaginary in product(COMPONENTS, repeat=2)],
                      dtype=np.complex64)
    return jnp.asarray(np.resize(data, size))


def mapped(source, record, factor):
    return accumulation_p.bind(source, jnp.complex64(factor), records=(record,), coefficient_records=(0,),
                               output_size=int(np.prod(record.logical_shape)), dtype=jnp.dtype(jnp.complex64))


def reference(source, record, factor):
    source_indices = np.zeros(record.logical_shape, dtype=np.int64)
    destinations = np.zeros(record.logical_shape, dtype=np.int64)
    for axis, extent in enumerate(record.logical_shape):
        coordinate = np.arange(extent).reshape((1,) * axis + (extent,) + (1,) * (len(record.logical_shape) - axis - 1))
        source_indices += coordinate * record.source_strides[axis]
        destinations += coordinate * record.destination_strides[axis]
    output = jnp.zeros((*source.shape[:-1], source_indices.size), dtype=jnp.complex64)
    selected = source[..., source_indices.ravel()] * jnp.complex64(factor)
    return output.at[..., destinations.ravel()].set(selected)


def assert_close(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-6)


def assert_native(lowered):
    text = lowered.as_text().lower()
    assert text.count("custom_call") == 1 and "tensor0_stride_" in text
    assert "gather" not in text and "scatter" not in text


def restore_threads(previous):
    if previous is None:
        enable_threads()
    else:
        set_num_threads(previous)


def test_large_mixed_compact_mapping_under_thread_limits():
    size = (1 << 19) + 3
    record = AffineRecord((size,), (1,), 0, (1,), 0)
    source = jnp.arange(size, dtype=jnp.float32) / 1024 - 2
    operation = jax.jit(lambda value: mapped(value, record, .75 - .5j))
    expected = source * jnp.complex64(.75 - .5j)
    assert_native(operation.lower(source))
    previous = get_num_threads()
    try:
        for limit in (1, 4):
            set_num_threads(limit)
            assert_close(operation(source), expected)
    finally:
        restore_threads(previous)


@pytest.mark.parametrize("factor", FACTORS)
def test_real_to_complex_compact_finite_values(factor):
    source = jnp.asarray(np.resize(COMPONENTS, 73))
    record = AffineRecord((73,), (1,), 0, (1,), 0)
    actual = jax.jit(lambda value: mapped(value, record, factor))(source)
    assert_close(actual, source * jnp.complex64(factor))


@pytest.mark.parametrize("record", [AffineRecord((131075,), (1,), 0, (1,), 0),
                                    rank_two(256, 256, True), rank_two(517, 263, False)])
def test_complex_large_layouts_and_tails(record):
    source = values(int(np.prod(record.logical_shape)))
    operation = jax.jit(lambda value: mapped(value, record, 1.25 - .75j))
    assert_native(operation.lower(source))
    assert_close(operation(source), reference(source, record, 1.25 - .75j))


@pytest.mark.parametrize("forward", [False, True])
@pytest.mark.parametrize("factor", FACTORS)
def test_rank_two_finite_component_combinations_and_tails(forward, factor):
    record = rank_two(17, 13, forward)
    source = finite_components(221)
    assert_close(jax.jit(lambda value: mapped(value, record, factor))(source), reference(source, record, factor))


def test_complex_compact_batch_jvp_and_vjp():
    size = 65539
    record = AffineRecord((size,), (1,), 0, (1,), 0)
    source = jnp.stack((values(size), values(size) + 1j))
    direction = source * jnp.complex64(.25 + .125j)
    cotangent = source * jnp.complex64(-.5 + .25j)
    operation = lambda value: mapped(value, record, .75 - .5j)
    oracle = lambda value: value * jnp.complex64(.75 - .5j)
    assert_close(jax.jit(operation)(source), oracle(source))
    assert_close(jax.jit(jax.vmap(operation))(source), oracle(source))
    actual = jax.jit(lambda value, tangent: jax.jvp(operation, (value,), (tangent,)))(source, direction)
    expected = jax.jvp(oracle, (source,), (direction,))
    for result, wanted in zip(actual, expected, strict=True):
        assert_close(result, wanted)
    pullback = jax.jit(lambda cot: jax.vjp(operation, source)[1](cot)[0])
    assert_close(pullback(cotangent), jax.vjp(oracle, source)[1](cotangent)[0])
    assert_native(pullback.lower(cotangent))


@pytest.mark.parametrize("factor", FACTORS)
def test_selected_complex_scale_finite_components_and_tails(factor):
    base = finite_components(67)
    coefficient = jnp.complex64(factor)
    operation = jax.jit(lambda data, alpha: scale(StridedView(data, (67,), (1,), 0), alpha).data)
    assert_native(operation.lower(base, coefficient))
    assert_close(operation(base, coefficient), base * coefficient)


def test_selected_scale_batched_factors_under_thread_limits():
    size = 262147
    base = jnp.stack(tuple(values(size) + index for index in range(3)))
    factors = jnp.asarray([.75 - .5j, -1.25 + .25j, 2 + .125j], dtype=jnp.complex64)
    operation = jax.jit(lambda data, alpha: scale(StridedView(data, (size,), (1,), 0), alpha).data)
    expected = base * factors[:, None]
    assert_native(operation.lower(base, factors))
    previous = get_num_threads()
    try:
        for limit in (1, 8):
            set_num_threads(limit)
            assert_close(operation(base, factors), expected)
    finally:
        restore_threads(previous)
