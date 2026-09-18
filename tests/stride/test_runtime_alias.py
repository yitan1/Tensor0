from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensor0._stride import StridedView, scale
from tensor0._stride._ffi._descriptor import encode_layout
from tensor0._stride._ffi._registration import operation_target
from tensor0._stride._jax import update_p
from tensor0._stride._layout import AffineRecord

from ._support import native_available


pytestmark = pytest.mark.skipif(not native_available(), reason="native CPU stride is unavailable")


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.complex64])
@pytest.mark.parametrize("mapped", [False, True])
def test_unified_update_multirecord_scale_batches(dtype, mapped):
    records = (AffineRecord((3,), (2,), 0, (2,), 0), AffineRecord((2,), (2,), 1, (2,), 1))
    base = jnp.arange(20, dtype=jnp.float32).reshape(4, 5).astype(dtype)
    factors = jnp.asarray([1.25, -.75, .5, -1], dtype=dtype)
    if dtype == jnp.complex64:
        base = base + 1j * (base / 7)
        factors = factors + jnp.asarray([-.5j, .25j, 1j, -.5j], dtype=dtype)
    operation = lambda old, factor: update_p.bind(old, old, factor, jnp.int32(0), records=records)
    if mapped:
        operation = jax.vmap(operation)
    original = np.asarray(base).copy()
    np.testing.assert_allclose(jax.jit(operation)(base, factors), base * factors[:, None], rtol=2e-6, atol=1e-6)
    np.testing.assert_array_equal(base, original)


def test_public_scale_batched_ad_and_nested_pullback():
    base = jnp.linspace(-2, 3, 24, dtype=jnp.float32).reshape(2, 12)
    factors = jnp.asarray([1.25, -.75], dtype=jnp.float32)
    directions = (jnp.linspace(1, -1, 24, dtype=jnp.float32).reshape(2, 12), jnp.asarray([-.5, .25]))
    cotangent = jnp.linspace(-3, 2, 24, dtype=jnp.float32).reshape(2, 12)
    native = lambda old, factor: scale(StridedView(old, (5,), (2,), 1), factor).data
    reference = lambda old, factor: old.at[:, 1:11:2].set(old[:, 1:11:2] * factor[:, None])

    def derivatives(function, old, factor):
        forward = jax.jvp(function, (old, factor), directions)
        reverse = jax.vjp(function, old, factor)[1](cotangent)
        pullback = lambda cot: jax.vjp(lambda value: function(value, factor), old)[1](cot)[0]
        nested = jax.jvp(pullback, (cotangent,), (jnp.ones_like(cotangent),))
        return forward, reverse, nested

    actual = jax.jit(lambda old, factor: derivatives(native, old, factor))(base, factors)
    expected = derivatives(reference, base, factors)
    for result, wanted in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(result, wanted, rtol=2e-6, atol=2e-6)


def test_donation_reuses_batched_scale_base_buffer():
    base = jnp.arange(36, dtype=jnp.float32).reshape(3, 12)
    factor = jnp.asarray([2, -1, .5], dtype=jnp.float32)
    expected = np.asarray(base).copy()
    expected[:, 1:11:2] *= np.asarray(factor)[:, None]
    original_factor = np.asarray(factor).copy()
    pointer = base.unsafe_buffer_pointer()
    compiled = jax.jit(lambda old, value: scale(StridedView(old, (5,), (2,), 1), value).data,
                       donate_argnums=(0,)).lower(base, factor).compile()
    actual = compiled(base, factor)
    actual.block_until_ready()
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(factor, original_factor)
    assert base.is_deleted()
    assert actual.unsafe_buffer_pointer() == pointer
    memory = compiled.memory_analysis()
    assert memory is not None and memory.alias_size_in_bytes == actual.nbytes


@pytest.mark.parametrize("factor", [complex(np.nan, 1), complex(3e38, 3e38), 1.25 - .75j])
def test_donated_and_preserved_inputs_share_complex_arithmetic(factor):
    values = jnp.asarray([
        0j, complex(-0., 0.), complex(np.nextafter(np.float32(0), np.float32(1)), 0),
        complex(np.inf, 1), complex(-np.inf, -2), complex(np.nan, 3),
        complex(3e38, -3e38), 1.5 - 2.25j, -7 + .5j,
    ], dtype=jnp.complex64)
    coefficient = jnp.asarray(factor, dtype=jnp.complex64)
    operation = lambda old, value: scale(StridedView(old, (9,), (1,), 0), value).data
    expected = jax.jit(operation)(values, coefficient)
    donated = jnp.array(values, copy=True)
    actual = jax.jit(operation, donate_argnums=(0,))(donated, coefficient)
    actual.block_until_ready()
    assert donated.is_deleted()
    for result, wanted in ((actual.real, expected.real), (actual.imag, expected.imag)):
        np.testing.assert_allclose(result, wanted, rtol=2e-6, atol=1e-6, equal_nan=True)


@pytest.mark.parametrize("donate", [False, True])
def test_concurrent_calls_to_one_prepared_executable_are_independent(donate):
    operation = jax.jit(lambda old, value: scale(StridedView(old, (32,), (2,), 0), value).data,
                        donate_argnums=(0,) if donate else ())
    compiled = operation.lower(jax.ShapeDtypeStruct((64,), jnp.float32),
                               jax.ShapeDtypeStruct((), jnp.float32)).compile()

    def run(index):
        host = np.arange(64, dtype=np.float32) + index
        base = jnp.asarray(host)
        result = np.asarray(compiled(base, jnp.float32(index + 1)))
        if donate:
            assert base.is_deleted()
        else:
            np.testing.assert_array_equal(base, host)
        expected = host.copy()
        expected[::2] *= index + 1
        np.testing.assert_array_equal(result, expected)
        return result

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(run, range(8)))
    assert len(results) == 8


@pytest.mark.parametrize("reuse_base", [False, True])
def test_same_update_target_accepts_separate_output_and_base_reuse(reuse_base):
    records = (AffineRecord((3,), (2,), 0, (2,), 1),)
    layout = encode_layout(records, source_size=8, output_size=8)
    call = jax.ffi.ffi_call(
        operation_target("update", np.dtype(jnp.float32)), jax.ShapeDtypeStruct((2, 8), jnp.float32),
        input_output_aliases={1: 0} if reuse_base else {},
    )
    operation = jax.jit(lambda source, base, alpha, beta: call(source, base, alpha, beta, layout=layout),
                        donate_argnums=(1,) if reuse_base else ())
    source = jnp.arange(16, dtype=jnp.float32).reshape(2, 8)
    base = jnp.arange(16, dtype=jnp.float32).reshape(2, 8) + 20
    original_source, original_base = np.asarray(source).copy(), np.asarray(base).copy()
    expected = original_base.copy()
    expected[:, 1:6:2] = 2 * original_source[:, :5:2] - original_base[:, 1:6:2]
    pointer = base.unsafe_buffer_pointer()
    actual = operation(source, base, jnp.float32(2), jnp.float32(-1))
    actual.block_until_ready()
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(source, original_source)
    if reuse_base:
        assert base.is_deleted() and actual.unsafe_buffer_pointer() == pointer
    else:
        np.testing.assert_array_equal(base, original_base)


@pytest.mark.parametrize("factor", [0, 2])
def test_update_rejects_source_only_alias_even_for_zero_contribution(factor):
    layout = encode_layout((AffineRecord((8,), (1,), 0, (1,), 0),), source_size=8, output_size=8)
    call = jax.ffi.ffi_call(operation_target("update", np.dtype(jnp.float32)),
                            jax.ShapeDtypeStruct((1, 8), jnp.float32), input_output_aliases={0: 0})
    source = jnp.arange(8, dtype=jnp.float32).reshape(1, 8)
    base = source + 20
    with pytest.raises(Exception, match="unsupported source/result alias"):
        call(source, base, jnp.float32(factor), jnp.float32(0), layout=layout).block_until_ready()
    np.testing.assert_array_equal(source, np.arange(8).reshape(1, 8))
    np.testing.assert_array_equal(base, np.arange(8).reshape(1, 8) + 20)


@pytest.mark.parametrize("invalid", ["version", "truncated", "storage_size", "buffer_dtype"])
def test_update_ffi_validation_with_base_alias(invalid):
    layout = encode_layout((AffineRecord((8,), (1,), 0, (1,), 0),), source_size=8, output_size=8)
    dtype, shape = jnp.float32, (1, 8)
    if invalid == "version":
        layout[0] = 99
        message = "unsupported native layout version"
    elif invalid == "truncated":
        layout = layout[:-1]
        message = "layout rank exceeds descriptor length"
    elif invalid == "storage_size":
        shape = (1, 9)
        message = "buffer dimensions do not match layout"
    else:
        dtype = jnp.complex64
        message = "dtype|element type"
    call = jax.ffi.ffi_call(operation_target("update", np.dtype(jnp.float32)),
                            jax.ShapeDtypeStruct(shape, dtype), input_output_aliases={1: 0})
    source = jnp.arange(shape[-1], dtype=jnp.float32).reshape(shape).astype(dtype)
    base = source + 20
    with pytest.raises(Exception, match=message):
        call(source, base, jnp.float32(2), jnp.float32(0), layout=layout).block_until_ready()
    np.testing.assert_array_equal(source, np.arange(shape[-1]).reshape(shape))
    np.testing.assert_array_equal(base, np.arange(shape[-1]).reshape(shape) + 20)
