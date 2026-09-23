"""Native scalar arithmetic, conversions and product stages."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensor0._stride import StridedView, scale
from tensor0._stride._jax import accumulation_p, copy_p, update_p
from tensor0._stride._layout import AffineRecord

from tests.stride.support.availability import native_available
from tests.stride.support.oracles.dtype_family import assert_native, assert_result, values
from tests.stride.support.oracles.product_stages import PARTIAL as PRODUCT_STAGES_PARTIAL
from tests.stride.support.oracles.scalar import PARTIAL as SCALAR_PARTIAL, assert_components
from tests.stride.support.oracles.scalar_paths import (
    PARTIAL as SCALAR_PATHS_PARTIAL,
    _CASES,
    assert_close,
    store,
)


pytestmark = pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')


@pytest.mark.parametrize("dtype,factor", [(jnp.float16, 1.3), (jnp.bfloat16, -.7)])
def test_narrow_float_copy_and_scaling_cover_every_storage_bit_pattern(dtype, factor):
    bits = np.arange(1 << 16, dtype=np.uint16)
    source = jnp.asarray(bits.view(np.dtype(dtype)))
    record = AffineRecord((source.size,), (1,), 0, (1,), 0)
    copied = jax.jit(lambda value: copy_p.bind(value, records=(record,), output_size=source.size, dtype=value.dtype))(source)
    np.testing.assert_array_equal(np.asarray(copied).view(np.uint16), bits)
    actual = jax.jit(lambda value, coefficient: accumulation_p.bind(
        value, coefficient, records=(record,), coefficient_records=(0,), output_size=source.size, dtype=value.dtype,
    ))(source, jnp.float32(factor))
    expected = np.asarray(jax.jit(
        lambda value: (value.astype(jnp.float32) * jnp.float32(factor)).astype(dtype),
    )(source)).astype(np.float32)
    actual = np.asarray(actual).astype(np.float32)
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    np.testing.assert_array_equal(np.isposinf(actual), np.isposinf(expected))
    np.testing.assert_array_equal(np.isneginf(actual), np.isneginf(expected))
    finite = np.isfinite(expected)
    np.testing.assert_allclose(actual[finite], expected[finite], rtol=.002 if dtype == jnp.float16 else .008, atol=0)


DTYPE_CASES = (
    (jnp.bool_, False), (jnp.int8, -3), (jnp.int16, -3), (jnp.int32, -3), (jnp.int64, -3),
    (jnp.uint8, 3), (jnp.uint16, 3), (jnp.uint32, 3), (jnp.uint64, 3),
    (jnp.float16, -1.25), (jnp.bfloat16, -1.25), (jnp.float32, -1.25), (jnp.float64, -1.25),
    (jnp.complex64, 1.25 - .75j), (jnp.complex128, 1.25 - .75j),
)


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


@pytest.mark.parametrize("source_dtype,result_dtype,coefficient_dtype", _CASES)
def test_native_computation_batch_branches_and_final_cast(
    source_dtype, result_dtype, coefficient_dtype,
):
    with jax.enable_x64():
        source = jnp.asarray([1.25, -2.5, 3.75], dtype=source_dtype)
        if jnp.issubdtype(source.dtype, jnp.complexfloating):
            source = source + jnp.asarray(.25j, dtype=source.dtype)
        base = jnp.arange(7, dtype=jnp.float32).astype(result_dtype)
        first = jnp.asarray([0, 1, .75, .75, .75], dtype=coefficient_dtype)
        second = jnp.asarray([.5, .5, 0, 1, .5], dtype=coefficient_dtype)
        sources = jnp.broadcast_to(source, (5, 3))
        bases = jnp.broadcast_to(base, (5, 7))
        def operation(old, values, alpha, beta):
            return update_p.bind(values, old, alpha, beta, records=SCALAR_PATHS_PARTIAL)
        expected_rows = []
        for alpha, beta in zip(first, second, strict=True):
            source_term = source if alpha == 1 else alpha * source
            base_term = base[1::2] if beta == 1 else beta * base[1::2]
            selected = base_term if alpha == 0 else source_term if beta == 0 else source_term + base_term
            expected_rows.append(store(base, selected))
        expected = jnp.stack(expected_rows)
        for function in (operation, jax.vmap(operation)):
            lowered = jax.jit(function).lower(bases, sources, first, second)
            assert lowered.as_text().count("custom_call") == 1
            assert "tensor0_stride_update_" in lowered.as_text()
            actual = lowered.compile()(bases, sources, first, second).block_until_ready()
            assert_close(actual, expected)
            np.testing.assert_array_equal(actual[:, ::2], bases[:, ::2])


@pytest.mark.parametrize("dtype,coefficient_dtype,value,increment", [
    ("float16", "float32", 65504., .001),
    ("bfloat16", "float32", 256., 1 / 256),
    ("float32", "float64", 2**24, 2**-25),
    ("complex64", "complex128", 2**24 * (1+1j), 2**-25 * (1+1j)),
])
def test_native_update_does_not_round_source_term_to_storage(
    dtype, coefficient_dtype, value, increment,
):
    with jax.enable_x64():
        source = jnp.full((3,), value, dtype=dtype)
        base = jnp.full((7,), value, dtype=dtype)
        first, second = jnp.asarray(1 + increment, dtype=coefficient_dtype), jnp.asarray(-1, dtype=coefficient_dtype)
        actual = jax.jit(lambda old, values, alpha, beta: update_p.bind(
            values, old, alpha, beta, records=SCALAR_PATHS_PARTIAL,
        ))(base, source, first, second)
        expected = store(base, first * source + second * base[1::2])
        assert_close(actual, expected)
        assert bool(jnp.all(actual[1::2] != 0))


@pytest.mark.parametrize("coefficient_dtype", ["float16", "bfloat16"])
def test_native_low_precision_coefficients_with_weak_float_storage(coefficient_dtype):
    with jax.enable_x64(False):
        source = jnp.broadcast_to(jnp.asarray(1.0003), (3,))
        base = jnp.broadcast_to(jnp.asarray(.2503), (7,))
        assert source.weak_type and base.weak_type
        first, second = jnp.asarray(.75, dtype=coefficient_dtype), jnp.asarray(.5, dtype=coefficient_dtype)
        actual = jax.jit(lambda old, values, alpha, beta: update_p.bind(
            values, old, alpha, beta, records=SCALAR_PATHS_PARTIAL,
        ))(base, source, first, second)
        expected = store(base, first.astype(source.dtype) * source.astype(source.dtype)
                         + second.astype(base.dtype) * base[1::2].astype(base.dtype))
        assert not actual.weak_type
        assert actual.dtype == expected.dtype
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-7)


def term(values, factor):
    if factor == 0:
        return None
    return values if factor == 1 else values * factor


def reference_update(base, source, alpha, beta):
    source_term, base_term = term(source, alpha), term(base[1::2], beta)
    if source_term is None:
        selected = jnp.zeros_like(base[1::2]) if base_term is None else base_term
    elif base_term is None:
        selected = source_term
    else:
        selected = source_term + base_term
    if not jnp.issubdtype(base.dtype, jnp.complexfloating):
        selected = jnp.real(selected)
    return base.at[1::2].set(selected.astype(base.dtype))


@pytest.mark.parametrize("alpha,beta", [(1., 0.), (0., 1.), (1., 1.), (1., 2.), (2., 1.), (2., 2.), (0., 0.)])
def test_update_short_circuit_branch_preserves_integer_precision(alpha, beta):
    source = jnp.asarray([2**24 + 1, 2**24 + 3, 2**24 + 5], dtype=jnp.int32)
    base = jnp.asarray([7, 2**24 + 6, 8, 2**24 + 9, 9, 2**24 + 10, 10], dtype=jnp.int32)
    alpha, beta = jnp.float32(alpha), jnp.float32(beta)
    actual = jax.jit(lambda old, new, first, second: update_p.bind(new, old, first, second, records=SCALAR_PARTIAL))(
        base, source, alpha, beta)
    np.testing.assert_array_equal(actual, reference_update(base, source, alpha, beta))


@pytest.mark.parametrize("static_factor", [.5, 1., 2.])
def test_explicit_coefficient_composition_batch_vmap_and_multirecord_layout(static_factor):
    source = jnp.asarray([2**24 + 1, 2**24 + 3, 2**24 + 5], dtype=jnp.int32)
    base = jnp.asarray([7, 2**24 + 7, 8, 2**24 + 9, 9, 2**24 + 11, 10], dtype=jnp.int32)
    records = tuple(AffineRecord((1,), (1,), index, (1,), 2 * index + 1) for index in range(3))
    pairs = [(first, second) for first in (0., 1., 2.) for second in (0., 1., 2.)]
    alpha = jnp.asarray([first for first, _ in pairs], dtype=jnp.float32)
    beta = jnp.asarray([second for _, second in pairs], dtype=jnp.float32)
    bases, sources = jnp.broadcast_to(base, (9, 7)), jnp.broadcast_to(source, (9, 3))
    operation = lambda old, new, first, second: update_p.bind(
        new, old, first * jnp.float32(static_factor), second, records=records)
    expected = jnp.stack([reference_update(base, source, jnp.float32(first) * jnp.float32(static_factor), jnp.float32(second))
                          for first, second in pairs])
    for function in (operation, jax.vmap(operation)):
        compiled = jax.jit(function)
        np.testing.assert_array_equal(compiled(bases, sources, alpha, beta), expected)
        assert compiled.lower(bases, sources, alpha, beta).as_text().count("custom_call") == 1


def test_copy_and_unit_scale_preserve_complex_identity_bits():
    source = jnp.asarray([complex(np.inf, 0), complex(0, np.inf), complex(-0., -0.)], dtype=jnp.complex64)
    copied = jax.jit(lambda value: copy_p.bind(value, records=SCALAR_PARTIAL, output_size=7, dtype=value.dtype))(source)
    np.testing.assert_array_equal(np.asarray(copied)[1::2].copy().view(np.uint32), np.asarray(source).view(np.uint32))
    np.testing.assert_array_equal(copied[::2], 0)
    for factor in (1, np.float32(1), np.complex64(1)):
        actual = jax.jit(lambda value, coefficient: scale(StridedView(value, (3,), (1,), 0), coefficient).data)(source, factor)
        np.testing.assert_array_equal(np.asarray(actual).view(np.uint32), np.asarray(source).view(np.uint32))


@pytest.mark.parametrize("source_dtype,result_dtype", [(jnp.float16, jnp.float32), (jnp.float32, jnp.float32),
    (jnp.float64, jnp.float64), (jnp.float32, jnp.complex64), (jnp.complex64, jnp.float32),
    (jnp.bool_, jnp.bool_), (jnp.int32, jnp.int8), (jnp.uint32, jnp.uint8)])
@pytest.mark.parametrize("static_factor", [None, 1, np.float32(.75), np.complex64(1 + 2j)])
def test_update_types_follow_bound_branches_and_final_conversion(source_dtype, result_dtype, static_factor):
    with jax.enable_x64():
        base, source = jnp.asarray([3, 120, 4, 100, 5, 80, 6], dtype=result_dtype), jnp.asarray([8, 9, 10], dtype=source_dtype)
        operation = jax.jit(lambda old, new, first, second: update_p.bind(new, old, first, second, records=SCALAR_PARTIAL))
        for alpha, beta in ((np.int64(1), np.int64(1)), (np.float64(.125), np.float32(2)),
                            (np.float64(0), np.complex64(1)), (np.float32(1), np.float64(0))):
            effective = jnp.asarray(alpha) if static_factor is None else jnp.asarray(alpha) * jnp.asarray(static_factor)
            actual = operation(base, source, effective, jnp.asarray(beta))
            assert_components(actual, reference_update(base, source, effective, jnp.asarray(beta)))
            np.testing.assert_array_equal(actual[::2], base[::2])


@pytest.mark.parametrize("dtype", [jnp.int8, jnp.uint8, jnp.int64, jnp.uint64, jnp.bool_])
def test_final_integer_cast_of_nonfinite_and_out_of_range_products(dtype):
    with jax.enable_x64():
        source = jnp.asarray([0, 1, 2, 63, 127], dtype=dtype)
        for factor in (np.float64(np.inf), np.float64(np.nan), np.float64(-1e30), np.float64(1e30), np.float64(.75)):
            actual = scale(StridedView(source, (5,), (1,), 0), factor).data
            expected = (factor * source.astype(jnp.result_type(factor, source))).astype(source.dtype)
            np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("size", [3, 9, 33])
def test_identity_half_conversion_tails_and_special_value_bits(size):
    source = jnp.resize(jnp.asarray([-0., 1., -2., np.inf, -np.inf, np.nan, 65504], dtype=jnp.float16), (size,))
    actual = jax.jit(lambda value: copy_p.bind(value, records=(AffineRecord((size,), (1,), 0, (1,), 0),),
                                              output_size=size, dtype=np.dtype(jnp.float32)))(source)
    np.testing.assert_array_equal(np.asarray(actual).view(np.uint32), np.asarray(source.astype(jnp.float32)).view(np.uint32))


@pytest.mark.parametrize("dtype,step", [(jnp.float16, 2**-10), (jnp.bfloat16, 2**-7)])
@pytest.mark.parametrize("base_factor", [0., 1., .25])
def test_final_update_write_rounds_low_precision_after_combining_terms(dtype, step, base_factor):
    with jax.enable_x64():
        midpoints = np.asarray([1 + step / 2, 1 + 1.5 * step], dtype=np.float32)
        values = np.concatenate((np.nextafter(midpoints, np.float32(0)), midpoints, np.nextafter(midpoints, np.float32(2))))
        values = np.concatenate((values, -values))
        source, base = jnp.ones((values.size, 3), dtype=dtype), jnp.full((values.size, 7), .5, dtype=dtype)
        coefficients, beta = jnp.asarray(values, dtype=jnp.float64), jnp.float64(base_factor)
        actual = jax.jit(lambda old, new, alpha, beta: update_p.bind(new, old, alpha, beta, records=SCALAR_PARTIAL))(
            base, source, coefficients, beta)
        expected = (coefficients[:, None] * source.astype(jnp.float64) + beta * .5).astype(dtype)
        np.testing.assert_array_equal(actual[:, 1::2], expected)
        np.testing.assert_array_equal(actual[:, ::2], base[:, ::2])


@pytest.mark.parametrize("dtype", ["bool", "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64",
                                   "float16", "bfloat16", "float32", "float64", "complex64", "complex128"])
def test_batched_coefficients_final_conversion_matches_jax(dtype):
    with jax.enable_x64():
        target = jnp.dtype(dtype)
        values = [0., 1., 2., 3., 7., 15., 31.] if jnp.issubdtype(target, jnp.unsignedinteger) else [-2., -1., 0., 1., 2., 3., 7.]
        if jnp.issubdtype(target, jnp.inexact):
            values += [1.00048828125, 1.00390625, 1.01171875, -.125]
        source = jnp.asarray(values, dtype=target)
        promoted = source.astype(jnp.complex128 if jnp.issubdtype(target, jnp.complexfloating) else jnp.float64)
        if jnp.issubdtype(target, jnp.complexfloating):
            promoted += 1.25j
            source = promoted.astype(target)
        count = source.size
        records = (AffineRecord((count,), (1,), 0, (1,), 0),)
        actual = jax.jit(lambda coefficients: update_p.bind(
            jnp.ones((count, count), dtype=target), jnp.zeros((count, count), dtype=target),
            coefficients, jnp.int32(0), records=records))(promoted)
        np.testing.assert_array_equal(actual, jnp.broadcast_to(source[:, None], actual.shape))


@pytest.mark.parametrize("source_dtype,result_dtype", [("int32", "int8"), ("int32", "int16"), ("uint32", "uint8"),
                                                       ("int64", "int64"), ("uint64", "uint64")])
def test_integer_conversion_preserves_boundary_bits(source_dtype, result_dtype):
    with jax.enable_x64():
        limits = np.iinfo(source_dtype)
        source = jnp.asarray(np.asarray([limits.min, limits.min + 1, 0, 1, limits.max - 1, limits.max], dtype=source_dtype))
        records = (AffineRecord((source.size,), (1,), 0, (1,), 0),)
        actual = jax.jit(lambda value: update_p.bind(value, jnp.zeros(value.shape, dtype=result_dtype),
                                                    jnp.int32(1), jnp.int32(0), records=records))(source)
        np.testing.assert_array_equal(actual, source.astype(result_dtype))


@pytest.mark.parametrize("source_is_narrow", [False, True])
def test_update_preserves_each_product_rounding(source_is_narrow):
    with jax.enable_x64():
        narrow_value = np.float32(1 - 2 ** -23)
        narrow_factor = jnp.asarray(1 + 2 ** -23, dtype=jnp.float32)
        wide_factor = jnp.asarray(2, dtype=jnp.float64)
        source = jnp.full((3,), narrow_value if source_is_narrow else -.5, dtype=jnp.float32)
        base = jnp.full((7,), -.5 if source_is_narrow else narrow_value, dtype=jnp.float32)
        alpha, beta = (narrow_factor, wide_factor) if source_is_narrow else (wide_factor, narrow_factor)
        function = jax.jit(lambda old, values, first, second: update_p.bind(
            values, old, first, second, records=PRODUCT_STAGES_PARTIAL,
        ))
        source_term, base_term = alpha * source, beta * base[1::2]
        assert (source_term.dtype, base_term.dtype) == (
            (jnp.float32, jnp.float64) if source_is_narrow else (jnp.float64, jnp.float32))
        expected = base.at[1::2].set((source_term + base_term).astype(base.dtype))
        before_source, before_base = np.asarray(source).copy(), np.asarray(base).copy()
        lowered = function.lower(base, source, alpha, beta)
        assert lowered.as_text().count("custom_call") == 1
        assert "tensor0_stride_update_f32" in lowered.as_text()
        actual = lowered.compile()(base, source, alpha, beta).block_until_ready()
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-7)
        np.testing.assert_array_equal(actual[::2], before_base[::2])
        np.testing.assert_array_equal(source, before_source)
        np.testing.assert_array_equal(base, before_base)


@pytest.mark.parametrize("promotion", ["standard", "strict"])
def test_weak_storage_uses_concrete_dtype_at_native_boundary(promotion):
    with jax.enable_x64(False), jax.numpy_dtype_promotion(promotion):
        source = jnp.broadcast_to(jnp.asarray(1.0003), (3,))
        base = jnp.broadcast_to(jnp.asarray(.25), (7,))
        assert source.weak_type and base.weak_type
        alpha = jnp.asarray(1, dtype=jnp.float16)
        beta = jnp.asarray(.3, dtype=jnp.float16)
        actual = jax.jit(lambda old, values, first, second: update_p.bind(
            values, old, first, second, records=PRODUCT_STAGES_PARTIAL,
        ))(base, source, alpha, beta)
        expected = np.asarray(base).copy()
        expected[1::2] = np.asarray(source) + np.float32(beta) * np.asarray(base)[1::2]
        assert actual.dtype == base.dtype and not actual.weak_type
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-7)


@pytest.mark.parametrize("source_is_narrow", [False, True])
@pytest.mark.parametrize("factor", [None, .5])
def test_mixed_product_stages_batch_zero_one_and_records(source_is_narrow, factor):
    with jax.enable_x64():
        source = jnp.arange(15, dtype=jnp.float32).reshape(5, 3) / 8
        base = jnp.arange(35, dtype=jnp.float32).reshape(5, 7) / 16
        alpha = jnp.asarray([0, 1, .75, 1, .75], dtype=jnp.float32 if source_is_narrow else jnp.float64)
        beta = jnp.asarray([.25, 1, 0, .25, 1], dtype=jnp.float64 if source_is_narrow else jnp.float32)
        records = (AffineRecord((2,), (1,), 0, (2,), 1),
                   AffineRecord((1,), (1,), 2, (1,), 5))

        def operation(old, values, first, second):
            effective = first if factor is None else first * jnp.asarray(factor, dtype=first.dtype)
            return update_p.bind(values, old, effective, second, records=records)

        effective = alpha if factor is None else alpha * jnp.asarray(factor, dtype=alpha.dtype)
        expected = base.at[:, 1::2].set(
            (effective[:, None] * source + beta[:, None] * base[:, 1::2]).astype(base.dtype))
        direct = jax.jit(operation)(base, source, alpha, beta)
        mapped = jax.jit(jax.vmap(operation))(base, source, alpha, beta)
        np.testing.assert_allclose(direct, expected, rtol=2e-6, atol=1e-7)
        np.testing.assert_allclose(mapped, expected, rtol=2e-6, atol=1e-7)
