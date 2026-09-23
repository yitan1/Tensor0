"""Native Update records, buffers and storage execution."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensor0._stride._ffi._calls import execute_update
from tensor0._stride._ffi._descriptor import encode_layout
from tensor0._stride._ffi._registration import operation_target
from tensor0._stride._jax import update_p
from tensor0._stride._layout import AffineRecord

from tests.stride.support.availability import native_available
from tests.stride.support.data import PAIRS
from tests.stride.support.layouts import PARTIAL
from tests.stride.support.oracles.raw_update import COMPLETE, reference_update


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
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


@pytest.mark.parametrize("dtype,alpha", [(jnp.float16, -1.25), (jnp.bfloat16, -1.25),
                                        (jnp.float32, -1.25), (jnp.complex64, 1.25-0.75j),
                                        (jnp.int32, -3)])
@pytest.mark.parametrize("beta", [0, 1])
def test_same_dtype_assignment_and_accumulation(dtype, alpha, beta):
    records = (AffineRecord((17,), (1,), 0, (1,), 0),)
    source, base = jnp.arange(17, dtype=dtype) - 3, jnp.arange(17, dtype=dtype) + 5
    coefficient = jnp.asarray(alpha, dtype=dtype)
    operation = jax.jit(lambda old, new: update_p.bind(new, old, coefficient, jnp.int32(beta), records=records))
    actual = operation(base, source)
    expected = reference_update(source, base, records, coefficient, beta)
    np.testing.assert_allclose(actual, expected, rtol=5e-3 if dtype in (jnp.float16, jnp.bfloat16) else 2e-6,
                               atol=1e-6)
    lowered = operation.lower(base, source).as_text()
    assert lowered.count("stablehlo.custom_call") == 1
    assert operation_target("update", base.dtype) in lowered


@pytest.mark.parametrize("source_dtype,result_dtype", [(jnp.float16, jnp.float32),
                                                      (jnp.float32, jnp.complex64),
                                                      (jnp.complex64, jnp.float32)])
@pytest.mark.parametrize("beta", [0, 1])
def test_partial_mixed_dtype_updates(source_dtype, result_dtype, beta):
    source = jnp.linspace(-2, 3, 32).astype(source_dtype)
    base = jnp.linspace(3, -2, 50).astype(result_dtype)
    if jnp.iscomplexobj(source):
        source = source + 1j * source[::-1]
    if jnp.iscomplexobj(base):
        base = base + 1j * base[::-1]
    actual = jax.jit(lambda old, new: update_p.bind(
        new, old, jnp.float32(.5), jnp.int32(beta), records=PARTIAL))(base, source)
    expected = reference_update(source, base, PARTIAL, .5, beta)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-6)


@pytest.mark.parametrize("beta", [0, 1])
def test_complex_finite_updates(beta):
    source = jnp.asarray([0, complex(-0., 0.), 1.25+2j, -1.25-3j, .125+1j,
                          1e10-1e10j, 1.25-.75j, -2.5+4j], dtype=jnp.complex64)
    base = source[::-1]
    records = (AffineRecord((8,), (1,), 0, (1,), 0),)
    alpha = jnp.complex64(1.25-.75j)
    actual = jax.jit(lambda old, new: update_p.bind(new, old, alpha, jnp.int32(beta), records=records))(base, source)
    np.testing.assert_allclose(actual, reference_update(source, base, records, alpha, beta), rtol=2e-6, atol=1e-6)


@pytest.mark.parametrize("beta", [0, 1])
def test_partial_update_preserves_unselected_storage_and_inputs(beta):
    base, source = jnp.arange(50, dtype=jnp.float32) + 100, jnp.arange(32, dtype=jnp.float32)
    actual = update_p.bind(source, base, jnp.float32(.5), jnp.int32(beta), records=PARTIAL)
    np.testing.assert_array_equal(actual, reference_update(source, base, PARTIAL, .5, beta))
    untouched = np.ones(50, dtype=bool)
    untouched[2:50:3] = False
    np.testing.assert_array_equal(np.asarray(actual)[untouched].view(np.uint8), np.asarray(base)[untouched].view(np.uint8))
    np.testing.assert_array_equal(base, np.arange(50) + 100)
    np.testing.assert_array_equal(source, np.arange(32))


def test_complete_assignment_ignores_original_base():
    source = jnp.arange(16, dtype=jnp.float32)
    outputs = [update_p.bind(source, jnp.full(16, value, jnp.float32), jnp.float32(.5), jnp.int32(0),
                             records=COMPLETE) for value in (-7, 99)]
    np.testing.assert_array_equal(outputs[0], outputs[1])
    np.testing.assert_array_equal(outputs[0], reference_update(source, jnp.zeros(16), COMPLETE, .5, 0))


@pytest.fixture(autouse=False)
def enable_x64():
    with jax.enable_x64():
        yield


@pytest.mark.usefixtures("enable_x64")
@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
@pytest.mark.parametrize('source_dtype,base_dtype', PAIRS)
@pytest.mark.parametrize('output_size', [0, 3])
def test_empty_mixed_source_preserves_base(source_dtype, base_dtype, output_size):
    with jax.enable_x64():
        source = jnp.empty((2, 0), dtype=source_dtype)
        base = jnp.ones((2, output_size), dtype=base_dtype)
        layout = encode_layout((AffineRecord((0,), (1,), 0, (1,), 0),), source_size=0, output_size=output_size)
        np.testing.assert_array_equal(execute_update(source, base, jnp.int32(0), jnp.int32(0), layout=layout), base)
