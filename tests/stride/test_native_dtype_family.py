from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensor0._stride import StridedView, scale
from tensor0._stride._ffi._registration import operation_target
from tensor0._stride._jax import accumulation_p, copy_p, update_p
from tensor0._stride._layout import AffineRecord

from ._support import native_available


pytestmark = pytest.mark.skipif(not native_available(), reason="native CPU stride is unavailable")

DTYPE_CASES = (
    (jnp.bool_, False), (jnp.int8, -3), (jnp.int16, -3), (jnp.int32, -3), (jnp.int64, -3),
    (jnp.uint8, 3), (jnp.uint16, 3), (jnp.uint32, 3), (jnp.uint64, 3),
    (jnp.float16, -1.25), (jnp.bfloat16, -1.25), (jnp.float32, -1.25), (jnp.float64, -1.25),
    (jnp.complex64, 1.25 - .75j), (jnp.complex128, 1.25 - .75j),
)


def values(dtype, size, shift=0):
    indices = np.arange(size, dtype=np.int64) + shift
    if dtype == jnp.bool_:
        host = indices % 3 != 0
    elif jnp.issubdtype(dtype, jnp.unsignedinteger):
        host = indices * 71 + 200
    elif jnp.issubdtype(dtype, jnp.signedinteger):
        host = indices * 37 - 120
    else:
        host = indices.astype(np.float64) / 3 - 2
        if jnp.issubdtype(dtype, jnp.complexfloating):
            host = host + 1j * host[::-1]
    return jnp.asarray(np.asarray(host).astype(np.dtype(dtype)))


def assert_result(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    if actual.dtype == jnp.bool_ or jnp.issubdtype(actual.dtype, jnp.integer):
        np.testing.assert_array_equal(actual, expected)
    else:
        tolerance = (2e-14 if actual.dtype in (jnp.float64, jnp.complex128) else
                     .008 if actual.dtype == jnp.bfloat16 else .002 if actual.dtype == jnp.float16 else 2e-6)
        np.testing.assert_allclose(np.asarray(actual).astype(np.complex128), np.asarray(expected).astype(np.complex128),
                                   rtol=tolerance, atol=tolerance)


def assert_native(lowered, operation, dtype):
    text = lowered.as_text().lower()
    assert text.count("custom_call") == 1
    assert operation_target(operation, np.dtype(dtype)) in text
    assert "gather" not in text and "scatter" not in text


@pytest.mark.parametrize("dtype,factor", DTYPE_CASES)
def test_dtype_family_copy_map_update_and_selected_scale(dtype, factor):
    with jax.enable_x64():
        source, base = values(dtype, 35), values(dtype, 35, shift=11)
        coefficient = jnp.asarray(True if dtype == jnp.bool_ else factor, dtype=dtype)
        transpose = AffineRecord((5, 7), (1, 5), 0, (7, 1), 0)
        copy = jax.jit(lambda value: copy_p.bind(value, records=(transpose,), output_size=35, dtype=value.dtype))
        assert_result(copy(source), source.reshape(7, 5).T.ravel())
        assert_native(copy.lower(source), "copy", dtype)
        mapped = jax.jit(lambda value, scalar: accumulation_p.bind(
            value, scalar, records=(transpose,), coefficient_records=(0,), output_size=35, dtype=value.dtype,
        ))
        expected = (source * coefficient).astype(dtype)
        assert_result(mapped(source, coefficient), expected.reshape(7, 5).T.ravel())
        assert_native(mapped.lower(source, coefficient), "accumulation", dtype)
        record = AffineRecord((35,), (1,), 0, (1,), 0)
        update = jax.jit(lambda old, value, alpha, beta: update_p.bind(value, old, alpha, beta, records=(record,)))
        for beta in (0, 1):
            wanted = expected if beta == 0 else (expected + base).astype(dtype)
            assert_result(update(base, source, coefficient, jnp.asarray(beta, dtype=dtype)), wanted)
        assert_native(update.lower(base, source, coefficient, jnp.asarray(1, dtype=dtype)), "update", dtype)
        selected_base = values(dtype, 50, shift=11)
        factor_array = jnp.asarray(factor, dtype=dtype)
        selected = jax.jit(lambda old, scalar: scale(StridedView(old, (16,), (3,), 2), scalar).data)
        wanted = selected_base.at[2:50:3].set((selected_base[2:50:3] * factor_array).astype(dtype))
        assert_result(selected(selected_base, factor_array), wanted)
        assert_native(selected.lower(selected_base, factor_array), "update", dtype)


@pytest.mark.parametrize("source_dtype,result_dtype", [(jnp.int32, jnp.bool_), (jnp.int32, jnp.int8),
                                                       (jnp.int32, jnp.int16), (jnp.uint32, jnp.uint8)])
@pytest.mark.parametrize("beta", [0, 1])
def test_integer_narrowing_occurs_after_mixed_update(source_dtype, result_dtype, beta):
    source = values(source_dtype, 17)
    base = values(result_dtype, 17, shift=7)
    if result_dtype == jnp.bool_:
        source = source.at[:3].set(jnp.asarray([-1, 0, 1], dtype=source_dtype))
        base = base.at[:3].set(True)
    record = AffineRecord((17,), (1,), 0, (1,), 0)
    operation = jax.jit(lambda old, value: update_p.bind(
        value, old, jnp.asarray(1, dtype=source_dtype), jnp.asarray(beta, dtype=result_dtype), records=(record,),
    ))
    expected = (source + base.astype(source_dtype) if beta else source).astype(result_dtype)
    np.testing.assert_array_equal(operation(base, source), expected)
    assert_native(operation.lower(base, source), "update", result_dtype)


@pytest.mark.parametrize("dtype", [jnp.int8, jnp.int16, jnp.int32, jnp.int64,
                                   jnp.uint8, jnp.uint16, jnp.uint32, jnp.uint64])
def test_integer_same_dtype_multiply_and_add_wrap_at_storage_width(dtype):
    with jax.enable_x64():
        info = np.iinfo(dtype)
        source = jnp.asarray([info.min, info.max, 0, 1], dtype=dtype)
        base = jnp.asarray([1, info.max, 1, info.max], dtype=dtype)
        record = AffineRecord((4,), (1,), 0, (1,), 0)
        actual = jax.jit(lambda value, old: update_p.bind(
            value, old, jnp.asarray(3, dtype=dtype), jnp.asarray(1, dtype=dtype), records=(record,),
        ))(source, base)
        modulus = 1 << info.bits
        expected = []
        for value, old in zip(np.asarray(source), np.asarray(base), strict=True):
            result = (int(value) * 3 + int(old)) % modulus
            if info.min < 0 and result > info.max:
                result -= modulus
            expected.append(result)
        np.testing.assert_array_equal(actual, np.asarray(expected, dtype=dtype))


@pytest.mark.parametrize("source_dtype,result_dtype,factor", [
    (jnp.float64, jnp.float64, -1.25), (jnp.complex128, jnp.complex128, 1.25 - .75j),
    (jnp.float64, jnp.complex128, .5 + .75j), (jnp.complex128, jnp.float64, -.75),
])
def test_wide_broadcast_vjp_accumulates_without_hidden_narrowing(source_dtype, result_dtype, factor):
    with jax.enable_x64():
        source = values(source_dtype, 3) + jnp.asarray(2**-40, dtype=source_dtype)
        cotangent = values(result_dtype, 12, shift=5) + jnp.asarray(2**-39, dtype=result_dtype)
        coefficient = jnp.asarray(factor, dtype=result_dtype)
        record = AffineRecord((4, 3), (0, 1), 0, (3, 1), 0)
        native = lambda value: accumulation_p.bind(value, coefficient, records=(record,), coefficient_records=(0,),
                                                   output_size=12, dtype=np.dtype(result_dtype))

        def reference(value):
            result = jnp.broadcast_to(value, (4, 3)) * coefficient
            if result_dtype == jnp.float64:
                result = jnp.real(result)
            return result.astype(result_dtype).ravel()

        assert_result(jax.jit(native)(source), reference(source))
        pullback = jax.jit(lambda cot: jax.vjp(native, source)[1](cot)[0])
        assert_result(pullback(cotangent), jax.vjp(reference, source)[1](cotangent)[0])
        assert_native(pullback.lower(cotangent), "accumulation", source_dtype)


@pytest.mark.parametrize("factor", [1.25 - .75j, .125 + 1j, complex(2.5, -0.), 3 + 3j])
def test_complex128_finite_values_and_bilinear_transpose(factor):
    with jax.enable_x64():
        source = jnp.asarray([0., -0., 1.25, -2.5, .125, 2., -2.], dtype=jnp.float64)
        cotangent = jnp.asarray([complex(-0., -0.), 0j, 1.25, 2.5j, .125 + 1j, 3 + 3j, -3 - 3j], dtype=jnp.complex128)
        coefficient = jnp.asarray(factor, dtype=jnp.complex128)
        record = AffineRecord((7,), (1,), 0, (1,), 0)
        native = lambda value: accumulation_p.bind(value, coefficient, records=(record,), coefficient_records=(0,),
                                                   output_size=7, dtype=np.dtype(jnp.complex128))
        reference = lambda value: value * coefficient
        actual, gradient = jax.jit(lambda value, cot: (native(value), jax.vjp(native, value)[1](cot)[0]))(source, cotangent)
        assert_result(actual, reference(source))
        assert_result(gradient, jax.vjp(reference, source)[1](cotangent)[0])
        assert_result(jax.jit(native)(cotangent), reference(cotangent))
