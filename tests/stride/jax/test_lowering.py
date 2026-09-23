"""Native JAX lowering, fail-closed routing and compilation reuse."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.typing import DTypeLike

from tensor0._stride import StridedView, _jax, add, dotu, materialize, reduce_sum, scale
from tensor0._stride._ffi._descriptor import encode_layout
from tensor0._stride._ffi._registration import operation_target
from tensor0._stride._jax import accumulation_p, copy_p, reduction_p, update_p
from tensor0._stride._layout import AffineRecord

from tests.stride.support.availability import native_available
from tests.stride.support.layouts import PARTITIONS as MAPPING_PARTITIONS
from tests.stride.support.oracles.compact_complex import (
    assert_close,
    assert_native as compact_assert_native,
    compact_record,
    complex_values,
    mapped,
    reference,
)
from tests.stride.support.oracles.copy import PARTITIONS as COPY_PARTITIONS, reference_map
from tests.stride.support.oracles.generic import assert_native as generic_assert_native
from tests.stride.support.oracles.raw_update import COMPLETE
from tests.stride.support.views import _dense, _pitched


def test_reduce_sum_uses_native_reduction_for_supported_dtype() -> None:
    source = jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4)

    @jax.jit
    def operation(value: jax.Array) -> jax.Array:
        return reduce_sum(StridedView.from_dense(value, (3, 4)), (1,))

    actual = operation(source)
    actual.block_until_ready()

    np.testing.assert_array_equal(actual, jnp.sum(source, axis=2))
    stablehlo = str(operation.lower(source).compiler_ir("stablehlo"))
    assert stablehlo.count("stablehlo.custom_call") == 1
    assert "tensor0_stride_reduction_f32_cpu_v1" in stablehlo


def test_native_dot_uses_one_custom_call() -> None:
    left = jnp.arange(6, dtype=jnp.float32).reshape(2, 3) + 1
    right = jnp.linspace(-2, 2, 12, dtype=jnp.float32)

    @jax.jit
    def operation(left_data: jax.Array, right_data: jax.Array) -> jax.Array:
        return dotu(_dense(left_data), _pitched(right_data))

    actual = operation(left, right)
    actual.block_until_ready()
    np.testing.assert_allclose(
        actual,
        jnp.sum(left * materialize(_pitched(right))),
    )

    stablehlo = str(
        operation.lower(left, right).compiler_ir("stablehlo")
    ).lower()
    assert stablehlo.count("stablehlo.custom_call") == 1
    assert "tensor0_stride_dot_f32_cpu_v1" in stablehlo


@pytest.mark.parametrize(
    ("dtype", "alpha_value", "beta_value"),
    (
        (jnp.float16, 0.5, -2),
        (jnp.bfloat16, 0.5, -2),
        (jnp.float32, 0.5, -2),
        (jnp.float64, 0.5, -2),
        (jnp.complex64, 0.5 + 0.25j, -2 + 0.5j),
        (jnp.complex128, 0.5 + 0.25j, -2 + 0.5j),
    ),
)
def test_native_add_uses_one_custom_call(
    dtype: DTypeLike,
    alpha_value: float | complex,
    beta_value: float | complex,
) -> None:
    with jax.enable_x64():
        resolved_dtype = jnp.dtype(dtype)
        source = jnp.arange(6, dtype=jnp.float32).astype(resolved_dtype)
        source = jnp.reshape(source, (2, 3))
        destination = jnp.arange(12, dtype=jnp.float32).astype(resolved_dtype)
        alpha = jnp.asarray(alpha_value, dtype=resolved_dtype)
        beta = jnp.asarray(beta_value, dtype=resolved_dtype)

        @jax.jit
        def operation(
            coefficient: jax.Array,
            source_data: jax.Array,
            destination_coefficient: jax.Array,
            destination_data: jax.Array,
        ) -> jax.Array:
            return add(
                _pitched(destination_data),
                _dense(source_data),
                alpha=destination_coefficient,
                beta=coefficient,
            ).data

        actual = operation(alpha, source, beta, destination)
        actual.block_until_ready()
        destination_indices = jnp.asarray([2, 6, 10, 3, 7, 11])
        expected = destination.at[destination_indices].set(
            jnp.reshape(alpha * source, (-1,))
            + beta * destination[destination_indices]
        )
        np.testing.assert_allclose(actual, expected, rtol=5e-3, atol=5e-3)

        stablehlo = str(
            operation.lower(alpha, source, beta, destination).compiler_ir(
                "stablehlo"
            )
        ).lower()
        assert stablehlo.count("stablehlo.custom_call") == 1
        assert (
            operation_target("update", resolved_dtype) in stablehlo
        )


@pytest.mark.skipif(not native_available(), reason='native CPU stride unavailable')
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
    compact_assert_native(function, source)
    assert_close(jax.jit(function)(source), reference(source, record, factor))


@pytest.mark.skipif(not native_available(), reason='native CPU stride unavailable')
def test_compact_native_composes_with_adjacent_elementwise_operations():
    record = compact_record((65536,), source_offset=3)
    source = jnp.arange(65544, dtype=jnp.float32) / 64
    function = lambda value: jnp.tanh(mapped(jnp.sin(value), record, .75))
    compact_assert_native(function, source)
    np.testing.assert_allclose(jax.jit(function)(source), jnp.tanh(reference(jnp.sin(source), record, .75)),
                               rtol=1e-6, atol=1e-6)


@pytest.mark.skipif(not native_available(), reason='native CPU stride unavailable')
def test_batched_compact_complex_mapping_uses_one_native_operation():
    record = compact_record((128, 64), source_offset=3)
    source = complex_values((3, 8200))
    function = lambda value: mapped(value, record, .75 - .5j)
    compact_assert_native(function, source)
    assert_close(jax.jit(function)(source), reference(source, record, .75 - .5j))


@pytest.mark.skipif(not native_available(), reason='native CPU stride unavailable')
def test_compact_hlo_operation_count_is_independent_of_element_count():
    counts = []
    for elements in (64, 16384):
        record = compact_record((elements,))
        source = complex_values((2, elements))
        text = compact_assert_native(lambda value: mapped(value, record, .75 - .5j), source)
        counts.append(text.count("stablehlo."))
    assert counts[0] == counts[1]


@pytest.mark.skipif(not native_available(), reason='native CPU stride unavailable')
@pytest.mark.parametrize("elements", [64, 65536])
def test_small_and_large_compact_complex_routes(elements):
    record = compact_record((elements,))
    source = complex_values((elements,))
    function = lambda value: mapped(value, record, .75 - .5j)
    compact_assert_native(function, source)
    assert_close(jax.jit(function)(source), reference(source, record, .75 - .5j))


@pytest.mark.skipif(not native_available(), reason='native CPU stride unavailable')
def test_positive_affine_permutation_uses_no_address_arrays():
    record = AffineRecord((8, 16), (1, 8), 0, (16, 1), 0)
    source = complex_values((2, 128))
    function = lambda value: mapped(value, record, .75 - .5j)
    compact_assert_native(function, source)
    expected = source.reshape(2, 16, 8).transpose(0, 2, 1).reshape(2, 128) * jnp.complex64(.75 - .5j)
    assert_close(jax.jit(function)(source), expected)


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
@pytest.mark.parametrize("dtype", [jnp.float16, jnp.complex64])
@pytest.mark.parametrize("transpose", [False, True])
def test_composed_factor_uses_one_update_for_contiguous_and_transposed_layouts(dtype, transpose):
    strides = (1, 9) if transpose else (17, 1)
    records = (AffineRecord((9, 17), strides, 0, (17, 1), 1),)
    source = jnp.arange(153, dtype=jnp.float32).astype(dtype)
    base = jnp.full((155,), -7, dtype=dtype)
    factor = jnp.asarray(.5, dtype=dtype)
    lowered = jax.jit(lambda old, values, coefficient: update_p.bind(
        values, old, coefficient * jnp.asarray(1.5, dtype=dtype), jnp.asarray(0, dtype=dtype),
        records=records)).lower(base, source, factor)
    text = lowered.as_text()
    assert text.count("custom_call") == 1
    assert "tensor0_stride_update_" in text
    values = source.reshape(17, 9).T.reshape(-1) if transpose else source
    expected = base.at[1:-1].set(values * jnp.asarray(.75, dtype=dtype))
    actual = lowered.compile()(base, source, factor)
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=0)
    np.testing.assert_array_equal(actual[jnp.asarray([0, 154])], base[jnp.asarray([0, 154])])


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
def test_layout_encoding_occurs_once_per_compilation(monkeypatch):
    calls = []
    original = _jax.encode_layout

    def counted(records, **parameters):
        calls.append(records)
        return original(records, **parameters)

    monkeypatch.setattr(_jax, "encode_layout", counted)
    compiled = jax.jit(lambda value: copy_p.bind(value, records=COPY_PARTITIONS, output_size=16, dtype=value.dtype))
    source = jnp.arange(16, dtype=jnp.int32)
    for offset in (0, 1):
        actual = compiled(source + offset)
        np.testing.assert_array_equal(actual, np.asarray(source + offset).reshape(4, 4)[:, [2, 3, 0, 1]].ravel())
    assert calls == [COPY_PARTITIONS]


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
@pytest.mark.parametrize("x64_enabled", [False, True])
def test_large_signed_layout_words_are_not_canonicalized_as_array_operands(x64_enabled):
    with jax.enable_x64(x64_enabled):
        large = 2**31 + 17
        record = AffineRecord((1,), (large,), 0, (large + 2,), 0)
        layout = encode_layout((record,), source_size=1, output_size=1)
        assert layout.dtype == np.int64 and tuple(layout[-2:]) == (large, large + 2)
        operation = jax.jit(lambda value: copy_p.bind(value, records=(record,), output_size=1, dtype=value.dtype))
        source = jnp.asarray([3.5], dtype=jnp.float32)
        np.testing.assert_array_equal(operation(source), source)
        text = operation.lower(source).as_text()
        assert str(large) in text and str(large + 2) in text


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
def test_equal_layouts_reuse_trace_and_dynamic_coefficients_do_not_enter_cache_key():
    traces = []

    def apply(source, first, second, *, records):
        traces.append(records)
        return accumulation_p.bind(source, first, second, records=records, coefficient_records=(0, 1),
                                   output_size=16, dtype=source.dtype)

    operation = jax.jit(apply, static_argnames=("records",))
    source = jnp.arange(16, dtype=jnp.float32)
    equal = tuple(AffineRecord(record.logical_shape, record.source_strides, record.source_offset,
                               record.destination_strides, record.destination_offset) for record in COPY_PARTITIONS)
    unequal = tuple(AffineRecord(record.logical_shape, record.source_strides, record.source_offset,
                                 record.destination_strides, record.source_offset) for record in COPY_PARTITIONS)
    for records, factors in ((COPY_PARTITIONS, (1.25, -.5)), (equal, (-2., 3.)), (unequal, (.5, 2.))):
        actual = operation(source, *map(jnp.float32, factors), records=records)
        np.testing.assert_array_equal(actual, reference_map(np.asarray(source), records, factors, 16))
    assert traces == [COPY_PARTITIONS, unequal]


@pytest.mark.skipif(not native_available(), reason='native CPU stride unavailable')
def test_large_storage_mixed_pullback_lowering_has_no_address_arrays():
    record = AffineRecord((4,), (1,), 0, (1,), 0)
    source = jax.ShapeDtypeStruct((100000000,), jnp.float16)
    cotangent = jax.ShapeDtypeStruct((4,), jnp.float32)
    operation = lambda value: copy_p.bind(value, records=(record,), output_size=4, dtype=jnp.dtype(jnp.float32))
    pullback = jax.jit(lambda value, cot: jax.vjp(operation, value)[1](cot)[0])
    generic_assert_native(pullback.lower(source, cotangent))


def test_general_coefficients_share_compiled_update():
    values = jnp.arange(4, dtype=jnp.float32)
    traced = []

    @jax.jit
    def operation(data, coefficient):
        traced.append(1)
        return scale(StridedView(data, (4,), (1,), 0), coefficient).data

    for factor in (2, 3, 4, 2):
        np.testing.assert_array_equal(operation(values, jnp.float32(factor)), values * factor)
    assert traced == [1]
    lowered = operation.lower(values, jnp.float32(2)).as_text()
    assert lowered.count("stablehlo.custom_call") == 1
    assert "tensor0_stride_update_f32_cpu_v1" in lowered


@pytest.mark.parametrize("factor", [
    0, -0.0, 1, np.float64(1 + 1e-9), 2, np.float32(3), float("nan"), float("inf"),
])
def test_closed_and_dynamic_coefficients_use_update(factor):
    values = jnp.ones(4, dtype=jnp.float32)
    closed = jax.jit(lambda data: scale(StridedView(data, (4,), (1,), 0), factor).data)
    dynamic = jax.jit(lambda data, coefficient: scale(StridedView(data, (4,), (1,), 0), coefficient).data)
    coefficient = jnp.asarray(factor)
    np.testing.assert_allclose(closed(values), dynamic(values, coefficient), rtol=2e-6, atol=1e-6)
    for lowered in (closed.lower(values), dynamic.lower(values, coefficient)):
        assert "tensor0_stride_update_f32_cpu_v1" in lowered.as_text()
        assert lowered.as_text().count("stablehlo.custom_call") == 1


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
def test_many_small_output_groups_share_one_native_call():
    records = tuple(AffineRecord((1, 4), (4, 1), (2 * group + term) * 4, (1, 1), group)
                    for group in range(1024) for term in range(2))
    source = jnp.ones(8192, dtype=jnp.float32)
    parameters = dict(records=records, output_shapes=((1, 1),) * len(records),
                      reduction_axes=((False, True),) * len(records), output_size=1024, dtype=np.dtype(jnp.float32))
    compiled = jax.jit(lambda value: reduction_p.bind(value, **parameters))
    np.testing.assert_array_equal(compiled(source), np.full(1024, 8, dtype=np.float32))
    assert compiled.lower(source).as_text().count("custom_call") == 1


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
def test_reduction_non_cpu_lowering_fails_closed():
    operation = jax.jit(lambda value: reduce_sum(StridedView(value, (2, 3), (3, 1), 0)))
    with pytest.raises((RuntimeError, ValueError, NotImplementedError), match="tpu|TPU"):
        operation.trace(jax.ShapeDtypeStruct((6,), jnp.float32)).lower(lowering_platforms=("tpu",))


@pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")
@pytest.mark.parametrize("recipe", ["partitions", "compact", "many_tiny", "many_single"])
@pytest.mark.parametrize("scaled", [False, True])
def test_cpu_execution_ignores_recipe_size_and_record_count(recipe, scaled):
    if recipe == "partitions":
        records, size = MAPPING_PARTITIONS, 16
        order = np.arange(size).reshape(4, 4)[:, [2, 3, 0, 1]].ravel()
    elif recipe == "compact":
        size = 4096
        records, order = (AffineRecord((size,), (1,), 0, (1,), 0),), np.arange(size)
    else:
        record_size = 8 if recipe == "many_tiny" else 1
        size = 1024 * record_size
        records = tuple(AffineRecord((record_size,), (1,), offset, (1,), offset)
                        for offset in range(0, size, record_size))
        order = np.arange(size)
    source = jnp.arange(size, dtype=jnp.float32)

    def operation(value):
        if scaled:
            return accumulation_p.bind(value, *((jnp.float32(.75),) * len(records)), records=records,
                coefficient_records=tuple(range(len(records))), output_size=size, dtype=value.dtype)
        return copy_p.bind(value, records=records, output_size=size, dtype=value.dtype)

    lowered = jax.jit(operation).lower(source)
    text = lowered.as_text().lower()
    assert text.count("custom_call") == 1
    assert f"tensor0_stride_{'accumulation' if scaled else 'copy'}_f32_cpu_v1" in text
    assert "gather" not in text and "scatter" not in text and "iota" not in text
    expected = source[order] * (.75 if scaled else 1)
    np.testing.assert_array_equal(lowered.compile()(source), expected)


@pytest.mark.parametrize("primitive", [copy_p, accumulation_p])
def test_non_cpu_lowering_fails_without_fallback(primitive):
    record = AffineRecord((4,), (1,), 0, (1,), 0)
    function = lambda value: primitive.bind(value, records=(record,), output_size=4, dtype=value.dtype)
    traced = jax.jit(function).trace(jnp.zeros(4, dtype=jnp.float32))
    with pytest.raises(NotImplementedError, match=f"{primitive.name}.*not found for platform tpu"):
        traced.lower(lowering_platforms=("tpu",))


@pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")
@pytest.mark.parametrize("batch_count,shape,dtype", [
    (256, (4096,), jnp.float32), (8, (131072,), jnp.complex64),
    (2, (1025, 1024), jnp.complex64),
])
def test_large_direct_and_vmap_batches_use_one_native_call(batch_count, shape, dtype):
    size = int(np.prod(shape))
    strides = (1,) if len(shape) == 1 else (1, shape[0])
    destinations = (1,) if len(shape) == 1 else (shape[1], 1)
    record = AffineRecord(shape, strides, 0, destinations, 0)
    real = (jnp.arange(batch_count * size, dtype=jnp.float32).reshape(batch_count, size) % 127) / 32
    source = (real + 1j * (1 - real)).astype(dtype) if dtype == jnp.complex64 else real
    factor = jnp.asarray(.75 - .5j if dtype == jnp.complex64 else 1.25, dtype=dtype)
    operation = lambda value: accumulation_p.bind(value, factor, records=(record,), coefficient_records=(0,),
                                                  output_size=size, dtype=value.dtype)
    selected = source if len(shape) == 1 else source.reshape(batch_count, shape[1], shape[0]).transpose(0, 2, 1).reshape(batch_count, size)
    expected = selected * factor
    for function in (operation, jax.vmap(operation)):
        lowered = jax.jit(function).lower(source)
        text = lowered.as_text().lower()
        assert text.count("custom_call") == 1 and "tensor0_stride_accumulation_" in text
        assert "gather" not in text and "scatter" not in text and "iota" not in text
        np.testing.assert_allclose(lowered.compile()(source), expected, rtol=2e-6, atol=1e-6)


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
@pytest.mark.parametrize("stride", [1, 2])
def test_selected_scale_large_layout_lowers_without_address_arrays(stride):
    size = 1_100_000
    base = jax.ShapeDtypeStruct((stride * (size - 1) + 1,), jnp.float32)
    operation = jax.jit(lambda old, factor: scale(StridedView(old, (size,), (stride,), 0), factor).data)
    text = operation.lower(base, jax.ShapeDtypeStruct((), jnp.float32)).as_text().lower()
    assert text.count("custom_call") == 1
    assert operation_target("update", np.dtype(jnp.float32)) in text
    assert "gather" not in text and "scatter" not in text


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
def test_selected_scale_non_cpu_lowering_fails_closed():
    operation = jax.jit(lambda old, factor: scale(StridedView(old, (16,), (3,), 2), factor).data)
    traced = operation.trace(jax.ShapeDtypeStruct((50,), jnp.float32), jax.ShapeDtypeStruct((), jnp.float32))
    with pytest.raises((RuntimeError, NotImplementedError, ValueError), match="tpu|TPU"):
        traced.lower(lowering_platforms=("tpu",))


LARGE_SIZE = 1_100_000


@pytest.mark.parametrize("beta", [0, 1])
@pytest.mark.parametrize("strided", [False, True])
def test_large_update_lowering_has_no_materialized_addresses(beta, strided):
    stride = 2 if strided else 1
    size = LARGE_SIZE * stride
    records = (AffineRecord((LARGE_SIZE,), (stride,), 0, (stride,), 0),)
    argument = jax.ShapeDtypeStruct((size,), jnp.float32)
    operation = jax.jit(lambda old, new: update_p.bind(new, old, jnp.float32(1), jnp.int32(beta), records=records))
    lowered = operation.lower(argument, argument).as_text()
    assert "tensor0_stride_update_f32_cpu_v1" in lowered
    assert lowered.count("stablehlo.custom_call") == 1
    assert "stablehlo.gather" not in lowered and "stablehlo.scatter" not in lowered


def test_update_non_cpu_lowering_fails_closed():
    argument = jax.ShapeDtypeStruct((16,), jnp.float32)
    traced = jax.jit(lambda old, new: update_p.bind(
        new, old, jnp.float32(1), jnp.int32(1), records=COMPLETE)).trace(argument, argument)
    with pytest.raises((RuntimeError, NotImplementedError, ValueError), match="tpu|TPU"):
        traced.lower(lowering_platforms=("tpu",))
