"""Equivalent affine leaves share arithmetic across fresh and update calls."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensor0._stride import StridedView, scale
from tensor0._stride._map import _execute_map
from tensor0._stride._ops._update import _execute_update
from tensor0._stride._plan import AffineRecord, CompleteMode, build_affine_plan
from tensor0._stride._testing import (
    _native_call_count_for_tests,
    _native_leaf_kernel_masks_for_tests,
    _observe_native_leaf_kernels_for_tests,
    _reset_native_call_count_for_tests,
    _set_native_disable_f16_f16_contiguous_simd_for_tests,
    _set_native_force_generic_for_tests,
)

from ._oracle import execute_update_reference


def _layout(name):
    if name == "contiguous":
        return (33,), (1,)
    if name == "transpose":
        return (8, 16), (1, 8)
    if name == "broadcast":
        return (8, 16), (0, 1)
    return (2, 3, 16, 24), (384, 768, 1, 16)


@pytest.mark.parametrize("layout", ["contiguous", "transpose", "broadcast", "rank4"])
@pytest.mark.parametrize("dtype", ["float16", "float32", "complex64"])
@pytest.mark.parametrize("static_factor", [None, 1., -.75])
@pytest.mark.parametrize("partial", [False, True])
def test_fresh_and_overwrite_respect_their_scalar_contracts(layout, dtype, static_factor, partial):
    shape, strides = _layout(layout)
    source_size = 1 + sum((size - 1) * stride for size, stride in zip(shape, strides))
    selected_size = int(np.prod(shape))
    destination_strides = tuple(int(np.prod(shape[axis + 1:])) for axis in range(len(shape)))
    offset = 3 if partial else 0
    output_size = selected_size + 2 * offset
    factor = None if static_factor is None else np.dtype(dtype).type(static_factor)
    record = AffineRecord(
        shape, strides, 0, destination_strides, offset, scale=factor,
        source_broadcast_axes=tuple(axis for axis, stride in enumerate(strides) if stride == 0),
    )
    plan = build_affine_plan(
        records=(record,), source_size=source_size, output_size=output_size,
        source_dtype=dtype, result_dtype=dtype,
        coverage=(CompleteMode.PARTIAL_UNIQUE_ZERO_FILL if partial
                  else CompleteMode.COMPLETE_UNIQUE),
    )
    source = jnp.resize(jnp.asarray([-0., 1., -2., .125, -3.5, 128., 65504],
                                   dtype=dtype), (source_size,))
    base = jnp.full((output_size,), -7., dtype=dtype)
    fresh = jax.jit(lambda values: _execute_map(values, plan=plan))
    update = jax.jit(lambda old, values: _execute_update(
        old, values, source_factor=1, base_factor=0, plan=plan,
    ))
    _set_native_force_generic_for_tests(True)
    try:
        expected = _execute_map(source, plan=plan)
        expected.block_until_ready()
    finally:
        _set_native_force_generic_for_tests(False)
    for actual, reference_output in (
        (fresh(source), expected),
        (update(base, source), execute_update_reference(base, source, 1, 0, plan)),
    ):
        for component in (np.real, np.imag):
            selected = component(np.asarray(actual)[offset:offset + selected_size])
            reference = component(np.asarray(reference_output)[offset:offset + selected_size])
            np.testing.assert_allclose(selected, reference, rtol=2e-3, atol=1e-6)
    actual = update(base, source)
    if partial:
        np.testing.assert_array_equal(actual[:offset], base[:offset])
        np.testing.assert_array_equal(actual[-offset:], base[-offset:])
    _reset_native_call_count_for_tests()
    update(base, source).block_until_ready()
    assert _native_call_count_for_tests() == 1


@pytest.mark.parametrize("source_dtype,result_dtype", [
    ("float16", "float32"), ("float32", "float32"), ("float32", "complex64"),
])
@pytest.mark.parametrize("static_factor", [None, 1., 1.00000003, 1e-8])
def test_unit_dynamic_factor_preserves_effective_coefficient_stages(source_dtype, result_dtype, static_factor):
    with jax.enable_x64():
        factor = None if static_factor is None else np.float64(static_factor)
        source = jnp.asarray([-0., 1., 65504, .125, -3.5, 128.], dtype=source_dtype)
        base = jnp.full((source.size + 2,), np.nan, dtype=result_dtype)
        record = AffineRecord((source.size,), (1,), 0, (1,), 1, scale=factor)
        plan = build_affine_plan(
            records=(record,), source_size=source.size, output_size=base.size,
            source_dtype=source_dtype, result_dtype=result_dtype,
            coverage=CompleteMode.PARTIAL_UNIQUE_ZERO_FILL,
        )
        execute = jax.jit(lambda old, values, coefficient: _execute_update(
            old, values, source_factor=coefficient, base_factor=0, plan=plan,
        ))
        actual = execute(base, source, jnp.asarray(1., dtype=jnp.float32))
        expected = execute_update_reference(base, source, jnp.float32(1), 0, plan)[1:-1]
        for component in (jnp.real, jnp.imag):
            np.testing.assert_array_equal(component(actual[1:-1]), component(expected))
        assert bool(jnp.isnan(actual[0]) & jnp.isnan(actual[-1]))


def test_subnormal_effective_factor_agrees_across_update_entries():
    with jax.enable_x64():
        source = jnp.asarray([65504], dtype=jnp.float32)
        plan = build_affine_plan(
            records=(AffineRecord((1,), (1,), 0, (1,), 0, scale=np.float64(1e-46)),),
            source_size=1, output_size=1, source_dtype="float32", result_dtype="float32",
            coverage=CompleteMode.COMPLETE_UNIQUE,
        )
        updated = _execute_update(jnp.zeros_like(source), source, source_factor=1,
                                  base_factor=0, plan=plan)
        dynamic_plan = build_affine_plan(
            records=(AffineRecord((1,), (1,), 0, (1,), 0),),
            source_size=1, output_size=1, source_dtype="float32", result_dtype="float32",
            coverage=CompleteMode.COMPLETE_UNIQUE,
        )
        dynamic = _execute_update(jnp.zeros_like(source), source,
                                  source_factor=np.float64(1e-46), base_factor=0, plan=dynamic_plan)
        np.testing.assert_array_equal(np.asarray(updated).view(np.uint32),
                                      np.asarray(dynamic).view(np.uint32))


@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("partial", [False, True])
def test_promoted_complex_mapping_matches_fresh_and_jax(transpose, partial):
    shape = (9, 17)
    strides = (1, 9) if transpose else (17, 1)
    source = jnp.resize(jnp.asarray([-0., 1., -2., .125, -3.5, 128., 65504],
                                   dtype=jnp.float32), (153,))
    offset = 3 if partial else 0
    base = jnp.full((153 + 2 * offset,), complex(np.nan, np.nan), dtype=jnp.complex64)
    factor = np.complex64(1.25 - .5j)
    plan = build_affine_plan(
        records=(AffineRecord(shape, strides, 0, (17, 1), offset, scale=factor),),
        source_size=source.size, output_size=base.size,
        source_dtype="float32", result_dtype="complex64",
        coverage=(CompleteMode.PARTIAL_UNIQUE_ZERO_FILL if partial
                  else CompleteMode.COMPLETE_UNIQUE),
    )
    mapped = source.reshape((17, 9)).T if transpose else source.reshape(shape)
    expected = (mapped * factor).reshape(-1)
    fresh = jax.jit(lambda values: _execute_map(values, plan=plan))(source)
    updated = jax.jit(lambda old, values: _execute_update(
        old, values, source_factor=1, base_factor=0, plan=plan,
    ))(base, source)
    for actual in (fresh, updated):
        for component in (jnp.real, jnp.imag):
            np.testing.assert_array_equal(component(actual[offset:offset + source.size]),
                                          component(expected))


@pytest.mark.parametrize("explicit", [False, True])
def test_half_storage_derivatives_do_not_narrow_factor(explicit):
    source = jnp.ones((1,), dtype=jnp.float16)
    factor = jnp.asarray(1e-8, dtype=jnp.float32)
    direction = jnp.full_like(source, 65504)
    operation = lambda values, coefficient: scale(StridedView(values, (1,), (1,), 0), coefficient).data

    def derivative(values, coefficient):
        if explicit:
            tangent = jax.jvp(operation, (values, coefficient),
                              (direction, jnp.zeros_like(coefficient)))[1]
        else:
            tangent = jax.jvp(lambda data: operation(data, coefficient),
                              (values,), (direction,))[1]
        cotangent = jax.vjp(lambda data: operation(data, coefficient), values)[1](direction)[0]
        return tangent, cotangent

    for actual in jax.jit(derivative)(source, factor):
        np.testing.assert_array_equal(actual, (factor * direction).astype(source.dtype))
        assert bool(jnp.all(actual != 0))


@pytest.mark.parametrize(("dtype", "factor_value"), [
    ("float16", 0.), ("float16", 1.), ("float16", .75),
    ("complex64", 0.), ("complex64", 1.), ("complex64", .75),
    ("complex64", 1.25 - .5j),
])
@pytest.mark.parametrize("layout", ["contiguous", "forward", "reverse"])
def test_bound_static_and_dynamic_scaling_select_same_kernel(dtype, layout, factor_value):
    shape = (9, 17)
    source_strides = (1, 9) if layout == "forward" else (17, 1)
    destination_strides = (1, 9) if layout == "reverse" else (17, 1)
    values = [-0., 1., -2., .125, -3.5, 128., 65504]
    if dtype == "complex64":
        values.extend([1.5 - 2.25j, complex(.125, -3.5), complex(-0., -0.)])
    source = jnp.resize(jnp.asarray(values, dtype=dtype), (153,))
    factor = np.dtype(dtype).type(factor_value)

    def make_plan(static):
        return build_affine_plan(
            records=(AffineRecord(shape, source_strides, 0, destination_strides, 0,
                                  scale=static),),
            source_size=153, output_size=153, source_dtype=dtype, result_dtype=dtype,
            coverage=CompleteMode.COMPLETE_UNIQUE,
        )

    static_plan = make_plan(factor)
    dynamic_plan = make_plan(None)
    static = jax.jit(lambda values: _execute_map(values, plan=static_plan)).lower(source).compile()
    dynamic = jax.jit(lambda values, coefficient: _execute_update(
        jnp.zeros_like(values), values, source_factor=coefficient, base_factor=0,
        plan=dynamic_plan,
    )).lower(source, jnp.asarray(factor)).compile()
    outputs, masks = [], []
    try:
        for executable, operands in ((static, (source,)), (dynamic, (source, jnp.asarray(factor)))):
            _observe_native_leaf_kernels_for_tests(True)
            result = executable(*operands)
            result.block_until_ready()
            outputs.append(result)
            mask, supported = _native_leaf_kernel_masks_for_tests()
            masks.append(mask)
        expected_bit = (1 if layout == "contiguous" else 2) if dtype == "float16" else (
            4 if layout == "contiguous" else 8)
        if supported & expected_bit:
            assert masks[0] & expected_bit
            if factor_value not in (0, 1):
                assert masks[1] == masks[0]
        if factor_value in (0, 1):
            assert masks[1] == 0
        else:
            for component in (jnp.real, jnp.imag):
                np.testing.assert_array_equal(component(outputs[0]), component(outputs[1]))
    finally:
        _observe_native_leaf_kernels_for_tests(False)

    try:
        _set_native_force_generic_for_tests(True)
        static_reference = _execute_map(source, plan=static_plan)
        static_reference.block_until_ready()
        dynamic_reference = (jnp.zeros_like(source) if factor_value == 0 else
                             _execute_map(source, plan=dynamic_plan) if factor_value == 1 else
                             static_reference)
        dynamic_reference.block_until_ready()
    finally:
        _set_native_force_generic_for_tests(False)
    for actual, expected in zip(outputs, (static_reference, dynamic_reference), strict=True):
        for component in (np.real, np.imag):
            result, reference = component(np.asarray(actual)), component(np.asarray(expected))
            np.testing.assert_allclose(result, reference, rtol=2e-3, atol=1e-6)


@pytest.mark.parametrize("layout", ["contiguous", "forward", "reverse"])
def test_bound_half_scaling_simd_and_scalar_tails_agree(layout):
    shape = (11, 19)
    source_strides = (1, 11) if layout == "forward" else (19, 1)
    destination_strides = (1, 11) if layout == "reverse" else (19, 1)
    plan = build_affine_plan(
        records=(AffineRecord(shape, source_strides, 0, destination_strides, 0),),
        source_size=209, output_size=209, source_dtype="float16", result_dtype="float16",
        coverage=CompleteMode.COMPLETE_UNIQUE,
    )
    source = jnp.resize(jnp.asarray([-0., 1., -2., .125, -3.5, 128., 65504],
                                   dtype=jnp.float16), (209,))
    execute = jax.jit(lambda values, factor: _execute_update(
        jnp.zeros_like(values), values, source_factor=factor, base_factor=0, plan=plan,
    )).lower(source, jnp.float16(.75)).compile()
    try:
        optimized = execute(source, jnp.float16(.75))
        optimized.block_until_ready()
        _set_native_disable_f16_f16_contiguous_simd_for_tests(True)
        fallback = execute(source, jnp.float16(.75))
        np.testing.assert_array_equal(optimized, fallback)
    finally:
        _set_native_disable_f16_f16_contiguous_simd_for_tests(False)


@pytest.mark.parametrize("dtype", ["float16", "complex64"])
@pytest.mark.parametrize("factor_value", [0., 1.])
def test_bound_differential_scaling_keeps_zero_one_multiplication(dtype, factor_value):
    source = jnp.ones((153,), dtype=dtype)
    direction = jnp.full_like(source, complex(1.25, -.5) if dtype == "complex64" else 1.25)
    coefficient = jnp.asarray(factor_value, dtype=dtype)

    def derivative(values, factor, tangent):
        return jax.jvp(
            lambda data: scale(StridedView(data, (153,), (1,), 0), factor).data,
            (values,), (tangent,),
        )[1]

    execute = jax.jit(derivative).lower(source, coefficient, direction).compile()
    try:
        _observe_native_leaf_kernels_for_tests(True)
        _reset_native_call_count_for_tests()
        actual = execute(source, coefficient, direction)
        actual.block_until_ready()
        assert _native_call_count_for_tests() == 1
        mask, supported = _native_leaf_kernel_masks_for_tests()
        expected_bit = 1 if dtype == "float16" else 4
        if supported & expected_bit:
            assert mask & expected_bit
        expected = coefficient * direction
        for component in (jnp.real, jnp.imag):
            np.testing.assert_array_equal(component(actual), component(expected))
    finally:
        _observe_native_leaf_kernels_for_tests(False)


@pytest.mark.parametrize("layout", ["forward", "reverse"])
def test_bound_half_transpose_preserves_zero_cotangents(layout):
    shape = (11, 19)
    source_strides = (1, 11) if layout == "forward" else (19, 1)
    destination_strides = (1, 11) if layout == "reverse" else (19, 1)
    plan = build_affine_plan(
        records=(AffineRecord(shape, source_strides, 0, destination_strides, 0,
                              scale=np.float16(.75)),),
        source_size=209, output_size=209, source_dtype="float16", result_dtype="float16",
        coverage=CompleteMode.COMPLETE_UNIQUE,
    )
    source = jnp.ones((209,), dtype=jnp.float16)
    cotangent = jnp.full_like(source, -0.)
    execute = jax.jit(lambda values, direction: jax.vjp(
        lambda data: _execute_map(data, plan=plan), values,
    )[1](direction)[0])
    try:
        optimized = execute(source, cotangent)
        optimized.block_until_ready()
        _set_native_disable_f16_f16_contiguous_simd_for_tests(True)
        fallback = execute(source, cotangent)
        np.testing.assert_array_equal(optimized, fallback)
        np.testing.assert_array_equal(optimized, jnp.zeros_like(optimized))
    finally:
        _set_native_disable_f16_f16_contiguous_simd_for_tests(False)


@pytest.mark.parametrize("dtype", ["float32", "complex64"])
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("factor", [0, 1])
def test_short_copy_and_zero_preserve_storage_bits(dtype, transpose, factor):
    shape = (11, 19)
    count = int(np.prod(shape))
    component_count = count * (2 if dtype == "complex64" else 1)
    bits = np.resize(np.asarray([
        0x80000000, 0x7FC12345, 0xFFC23456, 0x7F800000, 0xFF800000, 0x3FA00000,
    ], dtype=np.uint32), component_count)
    source = jnp.asarray(bits.view(dtype))
    source_bits = np.asarray(source).copy().view(np.uint32)
    base = jnp.resize(source, (count + 4,))
    base_bits = np.asarray(base).copy().view(np.uint32)
    strides = (1, 11) if transpose else (19, 1)
    plan = build_affine_plan(
        records=(AffineRecord(shape, strides, 0, (19, 1), 2),),
        source_size=count, output_size=count + 4, source_dtype=dtype, result_dtype=dtype,
        coverage=CompleteMode.PARTIAL_UNIQUE_ZERO_FILL,
    )
    operation = jax.jit(lambda old, values, coefficient: _execute_update(
        old, values, source_factor=coefficient, base_factor=0, plan=plan,
    ))
    _reset_native_call_count_for_tests()
    actual = operation(base, source, jnp.asarray(factor, dtype=dtype))
    actual.block_until_ready()
    assert _native_call_count_for_tests() == 1
    selected = np.asarray(actual)[2:-2].copy()
    expected = np.asarray(source).reshape((19, 11)).T.reshape(-1) if transpose else np.asarray(source)
    if factor:
        np.testing.assert_array_equal(selected.view(np.uint32), expected.copy().view(np.uint32))
        fresh = jax.jit(lambda values: _execute_map(values, plan=plan))(source)
        np.testing.assert_array_equal(np.asarray(fresh)[2:-2].copy().view(np.uint32),
                                      expected.copy().view(np.uint32))
    else:
        np.testing.assert_array_equal(selected, np.zeros_like(selected))
    for region in (slice(None, 2), slice(-2, None)):
        np.testing.assert_array_equal(np.asarray(actual)[region].copy().view(np.uint32),
                                      np.asarray(base)[region].copy().view(np.uint32))
    np.testing.assert_array_equal(np.asarray(source).view(np.uint32), source_bits)
    np.testing.assert_array_equal(np.asarray(base).view(np.uint32), base_bits)
