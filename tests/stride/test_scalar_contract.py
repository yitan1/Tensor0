"""Stage-wise JAX typing, final casts, and structural mapping identities."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensor0._stride import StridedView, scale
from tensor0._stride._map import _execute_map
from tensor0._stride._native_descriptor import lower_plan
from tensor0._stride._ops._selected_scale import _build_strided_scale_plan, _selected_scale_alias
from tensor0._stride._ops._update import _execute_update
from tensor0._stride._plan import AffineRecord, CompleteMode, build_affine_plan
from tensor0._stride._testing import _native_call_count_for_tests, _reset_native_call_count_for_tests
from tensor0._stride._testing import _set_native_force_generic_for_tests
from tensor0._stride._testing import _set_native_disable_f16_f32_contiguous_simd_for_tests

from ._oracle import execute_update_reference


def _plan(source_dtype, result_dtype, factor=None):
    return build_affine_plan(
        records=(AffineRecord((3,), (1,), 0, (2,), 1, scale=factor),),
        source_size=3, output_size=7, source_dtype=source_dtype,
        result_dtype=result_dtype, coverage=CompleteMode.PARTIAL_UNIQUE_ZERO_FILL,
    )


def _assert_components(actual, expected):
    assert actual.dtype == expected.dtype
    np.testing.assert_allclose(actual.real, expected.real, rtol=3e-3, atol=0, equal_nan=True)
    np.testing.assert_allclose(actual.imag, expected.imag, rtol=3e-3, atol=0, equal_nan=True)


def _term(values, factor):
    if factor == 0:
        return None
    return values if factor == 1 else factor * values


def _update_reference(base, source, source_factor, base_factor, static_factor):
    coefficient = (source_factor if static_factor is None else
                   jnp.multiply(source_factor, static_factor))
    source_term = _term(source, coefficient)
    base_term = _term(base[1::2], base_factor)
    if source_term is None:
        selected = jnp.zeros_like(base[1::2]) if base_term is None else base_term
    elif base_term is None:
        selected = source_term
    else:
        selected = source_term + base_term
    return base.at[1::2].set(selected.astype(base.dtype))


@pytest.mark.parametrize("alpha,beta", [(1., 0.), (0., 1.), (1., 1.),
                                        (1., 2.), (2., 1.), (2., 2.), (0., 0.)])
def test_update_actual_branch_preserves_integer_precision(alpha, beta):
    source = jnp.asarray([2**24 + 1, 2**24 + 3, 2**24 + 5], dtype=jnp.int32)
    base = jnp.asarray([7, 2**24 + 6, 8, 2**24 + 9, 9, 2**24 + 10, 10], dtype=jnp.int32)
    plan = _plan(source.dtype, base.dtype)
    alpha, beta = np.float32(alpha), np.float32(beta)
    operation = jax.jit(lambda old, new, first, second: _execute_update(
        old, new, source_factor=first, base_factor=second, plan=plan,
    ))
    actual = operation(base, source, alpha, beta)
    expected = _update_reference(base, source, alpha, beta, None)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(execute_update_reference(base, source, alpha, beta, plan), expected)


@pytest.mark.parametrize("alias", [False, True])
def test_scale_classifies_weak_coefficient_before_product_conversion(alias):
    with jax.enable_x64():
        source = jnp.asarray([65504, 32752, -65504], dtype=jnp.float32)
        factor = 1.00000001
        plan = _build_strided_scale_plan((3,), (1,), 0, 3, "float32")
        def operation(data, coefficient):
            if alias:
                return _selected_scale_alias(data, coefficient, plan=plan)
            return scale(StridedView(data, (3,), (1,), 0), coefficient).data
        actual = jax.jit(operation)(source, factor)
        np.testing.assert_array_equal(actual, (factor * source).astype(source.dtype))


@pytest.mark.parametrize("force_generic", [False, True])
def test_branch_types_batch_vmap_and_static_coefficient_composition(force_generic):
    source = jnp.asarray([2**24 + 1, 2**24 + 3, 2**24 + 5], dtype=jnp.int32)
    base = jnp.asarray([7, 2**24 + 7, 8, 2**24 + 9, 9, 2**24 + 11, 10], dtype=jnp.int32)
    records = tuple(AffineRecord((1,), (1,), index, (1,), 2 * index + 1,
                                 scale=np.float32(factor))
                    for index, factor in enumerate((.5, 1., 2.)))
    plan = build_affine_plan(
        records=records, source_size=3, output_size=7, source_dtype="int32",
        result_dtype="int32", coverage=CompleteMode.PARTIAL_UNIQUE_ZERO_FILL,
    )
    factors = [(first, second) for first in (0., 1., 2.) for second in (0., 1., 2.)]
    alpha = jnp.asarray([first for first, _ in factors], dtype=jnp.float32)
    beta = jnp.asarray([second for _, second in factors], dtype=jnp.float32)
    bases = jnp.broadcast_to(base, (9, 7))
    sources = jnp.broadcast_to(source, (9, 3))
    expected = []
    for first, second in factors:
        result = base
        for index, record in enumerate(records):
            assert record.scale is not None
            source_term = _term(source[index], jnp.multiply(np.float32(first), record.scale))
            base_term = _term(base[2 * index + 1], np.float32(second))
            value = (jnp.int32(0) if base_term is None else base_term) if source_term is None else (
                source_term if base_term is None else source_term + base_term)
            result = result.at[2 * index + 1].set(value.astype(base.dtype))
        expected.append(result)
    def operation(old, new, first, second):
        return _execute_update(old, new, source_factor=first, base_factor=second, plan=plan)
    _set_native_force_generic_for_tests(force_generic)
    try:
        for function in (operation, jax.vmap(operation)):
            executable = jax.jit(function).lower(bases, sources, alpha, beta).compile()
            _reset_native_call_count_for_tests()
            actual = executable(bases, sources, alpha, beta).block_until_ready()
            assert _native_call_count_for_tests() == 1
            np.testing.assert_array_equal(actual, jnp.stack(expected))
        reference = jax.jit(lambda old, new, first, second:
                            execute_update_reference(old, new, first, second, plan))
        np.testing.assert_array_equal(reference(bases, sources, alpha, beta), jnp.stack(expected))
    finally:
        _set_native_force_generic_for_tests(False)


@pytest.mark.parametrize("static_factor", [None, np.float64(.5)])
@pytest.mark.parametrize("coefficient", [0., 1., 2.])
def test_actual_branch_update_keeps_unshortened_derivatives(static_factor, coefficient):
    with jax.enable_x64():
        source = jnp.asarray([1.25, 3.5, -2.75], dtype=jnp.float32)
        base = jnp.arange(7, dtype=jnp.float32)
        alpha, beta = jnp.float64(coefficient), jnp.float64(1.)
        plan = _plan(source.dtype, base.dtype, static_factor)
        def operation(old, new, first, second):
            return _execute_update(old, new, source_factor=first, base_factor=second, plan=plan)
        def reference(old, new, first, second):
            effective = first if static_factor is None else first * static_factor
            return old.at[1::2].set((effective * new + second * old[1::2]).astype(old.dtype))
        arguments = (base, source, alpha, beta)
        directions = tuple(jnp.ones_like(value) for value in arguments)
        def differential(function, *values):
            return jax.jvp(function, values, directions)[1]
        np.testing.assert_allclose(
            jax.jit(lambda *values: differential(operation, *values))(*arguments),
            differential(reference, *arguments), rtol=1e-6,
        )
        actual = jax.jit(jax.grad(lambda *values: jnp.sum(operation(*values)),
                                  argnums=(0, 1, 2, 3)))(*arguments)
        expected = jax.grad(lambda *values: jnp.sum(reference(*values)),
                            argnums=(0, 1, 2, 3))(*arguments)
        for value, wanted in zip(actual, expected, strict=True):
            np.testing.assert_allclose(value, wanted, rtol=1e-6)
        np.testing.assert_allclose(
            jax.jvp(lambda *values: differential(operation, *values), arguments, directions)[1],
            jax.jvp(lambda *values: differential(reference, *values), arguments, directions)[1],
            rtol=1e-6,
        )


@pytest.mark.parametrize("factor", [None, 1, np.float32(1), np.complex64(1)])
def test_identity_is_not_explicit_static_multiplication(factor):
    source = jnp.asarray([complex(np.inf, 0), complex(0, np.inf), complex(-0., -0.)],
                         dtype=jnp.complex64)
    if factor is not None:
        source = jnp.asarray([1.25 - .5j, -.25 + 3j, complex(-0., -0.)],
                             dtype=jnp.complex64)
    plan = _plan(source.dtype, source.dtype, factor)
    actual = _execute_map(source, plan=plan)
    expected = jnp.zeros(7, dtype=source.dtype).at[1::2].set(
        source if factor is None else factor * source,
    )
    _assert_components(actual, expected)
    if factor is None:
        np.testing.assert_array_equal(np.asarray(actual)[1::2].copy().view(np.uint8),
                                      np.asarray(source).view(np.uint8))


def test_static_scalar_keys_keep_type_weakness_and_payload():
    with jax.enable_x64():
        factors = (None, 1, np.int32(1), np.int64(1), 1., np.float32(1),
                   np.float64(1), 0., -0.)
        plans = [_plan("float16", "float16", factor) for factor in factors]
        assert len(set(plans)) == len(factors)
        first = _plan("float16", "float16", np.float64(1 + 1e-8))
        second = _plan("float16", "float16", np.float64(1 + 2e-8))
        assert first != second
        assert lower_plan(first) != lower_plan(second)


@pytest.mark.parametrize("factor", [1e-8, np.float32(1e-8), np.float64(1e-8)])
def test_static_and_dynamic_strong_factors_are_not_cast_to_storage(factor):
    with jax.enable_x64():
        source = jnp.asarray([65504, 32752, -65504], dtype=jnp.float16)
        mapped = _execute_map(source, plan=_plan(source.dtype, source.dtype, factor))
        expected = (factor * source).astype(source.dtype)
        _assert_components(mapped[1::2], expected)
        view = StridedView(source, (3,), (1,), 0)
        _assert_components(scale(view, factor).data, expected)
        _assert_components(jax.jit(lambda data, coefficient:
                                   scale(StridedView(data, (3,), (1,), 0), coefficient).data)(source, factor), expected)


@pytest.mark.parametrize("source_dtype,result_dtype", [
    (jnp.float16, jnp.float32), (jnp.float32, jnp.float32),
    (jnp.float64, jnp.float64), (jnp.float32, jnp.complex64),
    (jnp.complex64, jnp.float32), (jnp.bool_, jnp.bool_),
    (jnp.int32, jnp.int8), (jnp.uint32, jnp.uint8),
])
@pytest.mark.parametrize("static_factor", [None, 1, np.float32(.75), np.complex64(1+2j)])
def test_update_types_follow_actual_short_circuit_branch(source_dtype, result_dtype, static_factor):
    with jax.enable_x64():
        base = jnp.asarray([3, 120, 4, 100, 5, 80, 6], dtype=result_dtype)
        source = jnp.asarray([8, 9, 10], dtype=source_dtype)
        plan = _plan(source.dtype, base.dtype, static_factor)
        for source_factor, base_factor in [(np.int64(1), np.int64(1)),
                                           (np.float64(.125), np.float32(2)),
                                           (np.float64(0), np.complex64(1)),
                                           (np.float32(1), np.float64(0))]:
            actual = jax.jit(lambda old, new, alpha, beta: _execute_update(
                old, new, source_factor=alpha, base_factor=beta, plan=plan,
            ))(base, source, source_factor, base_factor)
            expected = _update_reference(base, source, source_factor, base_factor, static_factor)
            _assert_components(actual, expected)
            np.testing.assert_array_equal(actual[::2], base[::2])


@pytest.mark.parametrize("alias", [False, True])
@pytest.mark.parametrize("coefficient_dtype", [jnp.float32, jnp.float64, jnp.complex128])
def test_mixed_scale_jvp_vjp_higher_ad_and_single_call(alias, coefficient_dtype):
    with jax.enable_x64():
        base = jnp.asarray([1.25, 2.5, 3.75, 4.25, 5.5], dtype=jnp.float32)
        coefficient = jnp.asarray(1.25, dtype=coefficient_dtype)
        plan = _build_strided_scale_plan((2,), (2,), 1, 5, "float32")
        def operation(data, factor):
            if alias:
                return _selected_scale_alias(data, factor, plan=plan)
            return scale(StridedView(data, (2,), (2,), 1), factor).data
        def reference(data, factor):
            return data.at[1::2].set((factor * data[1::2]).astype(data.dtype))
        for function in (operation, reference):
            primal, pullback = jax.vjp(function, base, coefficient)
            cotangents = pullback(jnp.ones_like(primal))
            expected = jax.vjp(reference, base, coefficient)[1](jnp.ones_like(base))
            for actual, wanted in zip(cotangents, expected, strict=True):
                _assert_components(actual, wanted)
        for explicit in (False, True):
            def derivative(data, factor):
                if explicit:
                    return jax.jvp(operation, (data, factor),
                                   (jnp.ones_like(data), jnp.ones_like(factor)))[1]
                return jax.jvp(lambda values: operation(values, factor),
                               (data,), (jnp.ones_like(data),))[1]
            executable = jax.jit(derivative).lower(base, coefficient).compile()
            _reset_native_call_count_for_tests()
            actual = executable(base, coefficient).block_until_ready()
            assert _native_call_count_for_tests() == 1
            expected = jax.jvp(reference, (base, coefficient),
                               (jnp.ones_like(base), jnp.ones_like(coefficient)
                                if explicit else jnp.zeros_like(coefficient)))[1]
            _assert_components(actual, expected)
            second = jax.jvp(lambda factor: derivative(base, factor),
                              (coefficient,), (jnp.ones_like(coefficient),))[1]
            _assert_components(second, jnp.zeros_like(base).at[1::2].set(1))


@pytest.mark.parametrize("generic", [False, True])
def test_typed_update_multirecord_blocks_and_batch_coefficients(generic):
    with jax.enable_x64():
        records = tuple(
            AffineRecord((8, 16), (32, 2), offset, (32, 2), offset,
                         scale=factor)
            for offset, factor in ((0, np.float32(.75)), (1, np.complex128(1+2j)))
        )
        plan = build_affine_plan(
            records=records, source_size=256, output_size=256,
            source_dtype="float32", result_dtype="float32",
            coverage=CompleteMode.COMPLETE_UNIQUE,
        )
        base = jnp.arange(768, dtype=jnp.float32).reshape(3, 256)
        source = base / 8
        alpha = jnp.asarray([0, 1, .125], dtype=jnp.float64)
        beta = jnp.asarray([1, 0, 2], dtype=jnp.float64)
        _set_native_force_generic_for_tests(generic)
        jax.clear_caches()
        try:
            executable = jax.jit(lambda old, new, first, second: _execute_update(
                old, new, source_factor=first, base_factor=second, plan=plan,
            )).lower(base, source, alpha, beta).compile()
            _reset_native_call_count_for_tests()
            actual = executable(base, source, alpha, beta).block_until_ready()
            assert _native_call_count_for_tests() == 1
            expected = base
            for offset, record in enumerate(records):
                assert record.scale is not None
                for batch in range(base.shape[0]):
                    coefficient = alpha[batch] * record.scale
                    source_term = _term(source[batch, offset::2], coefficient)
                    base_term = _term(base[batch, offset::2], beta[batch])
                    result = base_term if source_term is None else (
                        source_term if base_term is None else source_term + base_term)
                    assert result is not None
                    expected = expected.at[batch, offset::2].set(result.astype(base.dtype))
            _assert_components(actual, expected)
        finally:
            _set_native_force_generic_for_tests(False)
            jax.clear_caches()


@pytest.mark.parametrize("factor", [np.float64(.125), np.complex128(1+2j)])
def test_static_mapping_cast_derivatives(factor):
    with jax.enable_x64():
        source = jnp.asarray([1.25, 2.5, -3.75], dtype=jnp.float32)
        operation = lambda data: _execute_map(data, plan=_plan("float32", "float32", factor))
        reference = lambda data: jnp.zeros(7, dtype=data.dtype).at[1::2].set(
            (factor * data).astype(data.dtype))
        actual = jax.jvp(operation, (source,), (jnp.ones_like(source),))
        expected = jax.jvp(reference, (source,), (jnp.ones_like(source),))
        for result, wanted in zip(actual, expected, strict=True):
            _assert_components(result, wanted)
        actual_vjp = jax.vjp(operation, source)[1](jnp.ones(7, dtype=source.dtype))[0]
        expected_vjp = jax.vjp(reference, source)[1](jnp.ones(7, dtype=source.dtype))[0]
        _assert_components(actual_vjp, expected_vjp)


@pytest.mark.parametrize("dtype", [jnp.int8, jnp.uint8, jnp.int64, jnp.uint64, jnp.bool_])
def test_final_integer_cast_of_nonfinite_and_out_of_range_products(dtype):
    with jax.enable_x64():
        source = jnp.asarray([0, 1, 2, 63, 127], dtype=dtype)
        view = StridedView(source, (5,), (1,), 0)
        for factor in (np.float64(np.inf), np.float64(np.nan), np.float64(-1e30),
                       np.float64(1e30), np.float64(.75)):
            actual = scale(view, factor).data
            promoted = source.astype(jnp.result_type(factor, source))
            expected = (factor * promoted).astype(source.dtype)
            np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("alias", [False, True])
@pytest.mark.parametrize("factor", [0, 1])
def test_strong_complex_zero_one_derivatives_use_ordinary_multiplication(alias, factor):
    with jax.enable_x64():
        base = jnp.asarray([1+2j, 3+4j, 5+6j], dtype=jnp.complex64)
        tangent = jnp.asarray([complex(1.25, -.5)] * 3, dtype=base.dtype)
        coefficient = jnp.asarray(factor, dtype=jnp.complex128)
        plan = _build_strided_scale_plan((3,), (1,), 0, 3, "complex64")
        def operation(data):
            if alias:
                return _selected_scale_alias(data, coefficient, plan=plan)
            return scale(StridedView(data, (3,), (1,), 0), coefficient).data
        actual = jax.jvp(operation, (base,), (tangent,))[1]
        expected = (coefficient * tangent).astype(base.dtype)
        _assert_components(actual, expected)


def test_weak_and_strong_coefficients_keep_actual_product_promotion():
    with jax.enable_x64():
        data = jnp.asarray([1e30, -1e30, 1], dtype=jnp.float32)
        operation = lambda values, coefficient: scale(
            StridedView(values, (3,), (1,), 0), coefficient,
        ).data
        for execute in (operation, jax.jit(operation)):
            for factor in (1e-46, np.float64(1e-46)):
                _assert_components(execute(data, factor), (factor * data).astype(data.dtype))


@pytest.mark.parametrize("explicit", [False, True])
def test_scale_jvp_preserves_coefficient_precision_before_product(explicit):
    with jax.enable_x64():
        base = jnp.asarray([1., 2., 3.], dtype=jnp.float32)
        direction = jnp.full_like(base, 1e30)
        coefficient = jnp.asarray(1e-46, dtype=jnp.float64)
        operation = lambda data, factor: scale(StridedView(data, (3,), (1,), 0), factor).data
        def derivative(data, factor):
            if explicit:
                return jax.jvp(operation, (data, factor),
                               (direction, jnp.zeros_like(factor)))[1]
            return jax.jvp(lambda values: operation(values, factor),
                           (data,), (direction,))[1]
        actual = jax.jit(derivative)(base, coefficient)
        expected = (coefficient * direction).astype(base.dtype)
        np.testing.assert_array_equal(actual, expected)
        assert bool(jnp.all(actual != 0))


@pytest.mark.parametrize("size", [3, 9, 33])
@pytest.mark.parametrize("disable_simd", [False, True])
def test_identity_half_conversion_preserves_tail_and_special_values(size, disable_simd):
    data = jnp.resize(jnp.asarray([-0., 1., -2., np.inf, -np.inf, np.nan, 65504],
                                 dtype=jnp.float16), (size,))
    plan = build_affine_plan(
        records=(AffineRecord((size,), (1,), 0, (1,), 0),),
        source_size=size, output_size=size, source_dtype="float16",
        result_dtype="float32", coverage=CompleteMode.COMPLETE_UNIQUE,
    )
    _set_native_disable_f16_f32_contiguous_simd_for_tests(disable_simd)
    try:
        actual = _execute_map(data, plan=plan)
        expected = data.astype(jnp.float32)
        np.testing.assert_array_equal(np.asarray(actual).view(np.uint32),
                                      np.asarray(expected).view(np.uint32))
    finally:
        _set_native_disable_f16_f32_contiguous_simd_for_tests(False)


@pytest.mark.parametrize("dtype,step", [("float16", 2 ** -10), ("bfloat16", 2 ** -7)])
@pytest.mark.parametrize("base_factor", [0., 1., .25])
def test_typed_update_final_write_rounds_low_precision_once(dtype, step, base_factor):
    with jax.enable_x64():
        midpoints = np.asarray([1 + step / 2, 1 + 1.5 * step], dtype=np.float32)
        values = np.concatenate((np.nextafter(midpoints, np.float32(0)), midpoints,
                                 np.nextafter(midpoints, np.float32(2))))
        values = np.concatenate((values, -values))
        source = jnp.ones((values.size, 3), dtype=dtype)
        base = jnp.full((values.size, 7), .5, dtype=dtype)
        plan = build_affine_plan(
            records=(AffineRecord((3,), (1,), 0, (2,), 1),),
            source_size=3, output_size=7,
            source_dtype=source.dtype, result_dtype=base.dtype,
            coverage=CompleteMode.PARTIAL_UNIQUE_ZERO_FILL,
        )
        coefficient = jnp.asarray(values, dtype=jnp.float64)
        previous_coefficient = jnp.float64(base_factor)
        execute = jax.jit(lambda old, values, alpha, beta: _execute_update(
            old, values, source_factor=alpha, base_factor=beta, plan=plan,
        ))
        actual = execute(base, source, coefficient, previous_coefficient)
        expected = (coefficient[:, None] * source.astype(jnp.float64)
                    + previous_coefficient * .5).astype(dtype)
        np.testing.assert_array_equal(actual[:, 1::2], expected)
        np.testing.assert_array_equal(actual[:, ::2], base[:, ::2])


@pytest.mark.parametrize("dtype", [
    "bool", "int8", "uint8", "int16", "uint16", "int32", "uint32",
    "int64", "uint64", "float16", "bfloat16", "float32", "float64",
    "complex64", "complex128",
])
def test_batched_coefficient_final_conversion_matches_jax(dtype):
    with jax.enable_x64():
        target = jnp.dtype(dtype)
        values = [-2., -1., 0., 1., 2., 3., 7.]
        if jnp.issubdtype(target, jnp.unsignedinteger):
            values = [0., 1., 2., 3., 7., 15., 31.]
        if jnp.issubdtype(target, jnp.inexact):
            values += [1.00048828125, 1.00390625, 1.01171875, -.125]
        source = jnp.asarray(values, dtype=target)
        promoted = source.astype(jnp.complex128 if jnp.issubdtype(target, jnp.complexfloating)
                                 else jnp.float64)
        if jnp.issubdtype(target, jnp.complexfloating):
            promoted = promoted + 1.25j
            source = promoted.astype(target)
        count = source.size
        plan = build_affine_plan(
            records=(AffineRecord((count,), (1,), 0, (1,), 0, scale=None),),
            source_size=count, output_size=count, source_dtype=target, result_dtype=target,
            coverage=CompleteMode.COMPLETE_UNIQUE,
        )
        actual = jax.jit(lambda coefficients: _execute_update(
            jnp.zeros((count, count), dtype=target),
            jnp.ones((count, count), dtype=target),
            source_factor=coefficients, base_factor=0, plan=plan,
        ))(promoted)
        np.testing.assert_array_equal(actual, jnp.broadcast_to(source[:, None], actual.shape))


@pytest.mark.parametrize("source_dtype,result_dtype", [
    ("int32", "int8"), ("int32", "int16"), ("uint32", "uint8"),
    ("int64", "int64"), ("uint64", "uint64"),
])
def test_integer_conversion_preserves_boundary_bits(source_dtype, result_dtype):
    with jax.enable_x64():
        limits = np.iinfo(source_dtype)
        values = np.asarray([limits.min, limits.min + 1, 0, 1, limits.max - 1, limits.max],
                            dtype=source_dtype)
        source = jnp.asarray(values)
        plan = build_affine_plan(
            records=(AffineRecord((source.size,), (1,), 0, (1,), 0, scale=1),),
            source_size=source.size, output_size=source.size,
            source_dtype=source_dtype, result_dtype=result_dtype,
            coverage=CompleteMode.COMPLETE_UNIQUE,
        )
        actual = jax.jit(lambda data: _execute_update(
            jnp.zeros((source.size,), dtype=result_dtype), data,
            source_factor=1, base_factor=0, plan=plan,
        ))(source)
        np.testing.assert_array_equal(actual, source.astype(result_dtype))
