from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
import os
from pathlib import Path
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensor0 import _native
from tensor0._stride import enable_threads, get_num_threads, set_num_threads
from tensor0._stride import _jax
from tensor0._stride._ffi._descriptor import encode_layout
from tensor0._stride._ffi._registration import operation_target
from tensor0._stride._jax import accumulation_p, copy_p
from tensor0._stride._layout import AffineRecord

from ._support import native_available


pytestmark = pytest.mark.skipif(not native_available(), reason="native CPU stride is unavailable")

PARTITIONS = (AffineRecord((4, 2), (4, 1), 0, (4, 1), 2),
              AffineRecord((4, 2), (4, 1), 2, (4, 1), 0))
LAYOUTS = [
    pytest.param(PARTITIONS, (1.25, -.5), 16, 16, id="partitions"),
    pytest.param((AffineRecord((3,), (2,), 0, (1,), 0),), (-2.,), 5, 3, id="gapped"),
    pytest.param((AffineRecord((2,), (1,), 0, (1,), 0), AffineRecord((2,), (1,), 3, (1,), 2)),
                 (1.5, -.5), 5, 4, id="gapped-records"),
    pytest.param((AffineRecord((16,), (2,), 1, (3,), 2),), (.5,), 32, 50, id="partial"),
    pytest.param((), (), 0, 8, id="empty-partial"),
    pytest.param((), (), 0, 0, id="empty-output"),
    pytest.param((AffineRecord((0,), (1,), 0, (1,), 0),), (1.,), 0, 8, id="empty-record"),
    pytest.param((AffineRecord((), (), 0, (), 0),), (2.5,), 1, 1, id="scalar"),
    pytest.param((AffineRecord((2, 4, 3, 8), (32, 1, 64, 4), 0, (96, 24, 8, 1), 0),),
                 (-.75,), 192, 192, id="rank4-tiled-shape"),
    pytest.param((AffineRecord((2, 8, 3, 12), (96, 1, 192, 8), 0, (288, 36, 12, 1), 0),),
                 (1.25,), 576, 576, id="rank4-avx-shape"),
    pytest.param((AffineRecord((1, 1, 1), (1, 1, 1), 0, (1, 1, 1), 0),
                  AffineRecord((2, 2, 1), (1, 2, 1), 1, (2, 1, 1), 1)),
                 (1., 1.), 5, 5, id="u1-two-record"),
    pytest.param((AffineRecord((12, 16, 8), (8, 96, 1), 0, (256, 8, 1), 0),
                  AffineRecord((12, 16, 8), (8, 96, 1), 1536, (256, 8, 1), 128),
                  AffineRecord((12, 16, 8), (8, 96, 1), 3072, (128, 8, 1), 3072)),
                 (1., 1., 1.), 4608, 4608, id="u1-three-record"),
]


def reference_map(source, records, factors, output_size):
    result = np.zeros((*source.shape[:-1], output_size), dtype=source.dtype)
    for record, factor in zip(records, factors, strict=True):
        addresses = []
        for strides, offset in ((record.source_strides, record.source_offset),
                                (record.destination_strides, record.destination_offset)):
            indices = np.full(record.logical_shape, offset, dtype=np.int64)
            for axis, (size, stride) in enumerate(zip(record.logical_shape, strides, strict=True)):
                shape = (1,) * axis + (size,) + (1,) * (len(record.logical_shape) - axis - 1)
                indices += np.arange(size).reshape(shape) * stride
            addresses.append(indices.ravel())
        values = source[..., addresses[0]].astype(np.complex64 if np.iscomplexobj(source) else np.float32)
        result[..., addresses[1]] = (values * factor).astype(source.dtype)
    return result


def assert_single_native_call(lowered):
    text = lowered.as_text().lower()
    assert text.count("custom_call") == 1
    assert "tensor0_stride_" in text
    assert "gather" not in text and "scatter" not in text


def test_build_versions_match_runtime():
    assert _native._stride_ffi_build_versions() == (jax.__version__, version("jaxlib"))


def test_layout_encoding_occurs_once_per_compilation(monkeypatch):
    calls = []
    original = _jax.encode_layout

    def counted(records, **parameters):
        calls.append(records)
        return original(records, **parameters)

    monkeypatch.setattr(_jax, "encode_layout", counted)
    compiled = jax.jit(lambda value: copy_p.bind(value, records=PARTITIONS, output_size=16, dtype=value.dtype))
    source = jnp.arange(16, dtype=jnp.int32)
    for offset in (0, 1):
        actual = compiled(source + offset)
        np.testing.assert_array_equal(actual, np.asarray(source + offset).reshape(4, 4)[:, [2, 3, 0, 1]].ravel())
    assert calls == [PARTITIONS]


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


def test_equal_layouts_reuse_trace_and_dynamic_coefficients_do_not_enter_cache_key():
    traces = []

    def apply(source, first, second, *, records):
        traces.append(records)
        return accumulation_p.bind(source, first, second, records=records, coefficient_records=(0, 1),
                                   output_size=16, dtype=source.dtype)

    operation = jax.jit(apply, static_argnames=("records",))
    source = jnp.arange(16, dtype=jnp.float32)
    equal = tuple(AffineRecord(record.logical_shape, record.source_strides, record.source_offset,
                               record.destination_strides, record.destination_offset) for record in PARTITIONS)
    unequal = tuple(AffineRecord(record.logical_shape, record.source_strides, record.source_offset,
                                 record.destination_strides, record.source_offset) for record in PARTITIONS)
    for records, factors in ((PARTITIONS, (1.25, -.5)), (equal, (-2., 3.)), (unequal, (.5, 2.))):
        actual = operation(source, *map(jnp.float32, factors), records=records)
        np.testing.assert_array_equal(actual, reference_map(np.asarray(source), records, factors, 16))
    assert traces == [PARTITIONS, unequal]


@pytest.mark.parametrize("records,factors,source_size,output_size", LAYOUTS)
@pytest.mark.parametrize("batch_shape", [(), (2, 3)])
def test_native_layouts_and_fresh_initialization(records, factors, source_size, output_size, batch_shape):
    source = jnp.arange(int(np.prod(batch_shape)) * source_size, dtype=jnp.float32).reshape(*batch_shape, source_size)
    coefficients = tuple(jnp.float32(value) for value in factors)
    operation = jax.jit(lambda value: accumulation_p.bind(
        value, *coefficients, records=records, coefficient_records=tuple(range(len(records))),
        output_size=output_size, dtype=value.dtype,
    ))
    np.testing.assert_array_equal(operation(source), reference_map(np.asarray(source), records, factors, output_size))
    assert_single_native_call(operation.lower(source))


@pytest.mark.parametrize("invalid,message", [
    ("source_size", "buffer dimensions do not match layout"),
    ("output_size", "buffer dimensions do not match layout"),
    ("batch_count", "copy batch dimensions do not match"),
    ("alias", "input and output buffers overlap"),
    ("overlapping_record", "injective|overlap"),
])
def test_copy_ffi_revalidates_batched_buffers_and_records(invalid, message):
    layout = encode_layout(PARTITIONS, source_size=16, output_size=16)
    source_shape, output_shape = (2, 16), (2, 16)
    if invalid == "source_size":
        source_shape = (2, 15)
    elif invalid == "output_size":
        output_shape = (2, 15)
    elif invalid == "batch_count":
        output_shape = (3, 16)
    elif invalid == "overlapping_record":
        layout[-2:] = (1, 1)
    call = jax.ffi.ffi_call(operation_target("copy", np.dtype(jnp.float32)),
                            jax.ShapeDtypeStruct(output_shape, jnp.float32),
                            input_output_aliases={0: 0} if invalid == "alias" else {})
    source = jnp.arange(np.prod(source_shape), dtype=jnp.float32).reshape(source_shape)
    with pytest.raises(Exception, match=message):
        call(source, layout=layout).block_until_ready()


@pytest.mark.parametrize("dtype,factor", [(jnp.float16, 1.3), (jnp.bfloat16, -.7),
                                         (jnp.complex64, .75 - .5j), (jnp.int32, -1)])
@pytest.mark.parametrize("partial", [False, True])
def test_typed_scaling_transpose_and_partial_zero_fill(dtype, factor, partial):
    record = AffineRecord((5, 7), (1, 5), 0, (7, 1), 3 if partial else 0)
    output_size = 41 if partial else 35
    source = (jnp.arange(105, dtype=jnp.int32).reshape(3, 35) % 17 - 8).astype(dtype)
    if dtype == jnp.complex64:
        source = source + .5j * source
    if dtype == jnp.int32:
        source = source.at[0, :2].set(jnp.asarray([np.iinfo(np.int32).min, np.iinfo(np.int32).max]))
    coefficient = jnp.asarray(factor, dtype=jnp.complex64 if dtype == jnp.complex64 else
                              jnp.int32 if dtype == jnp.int32 else jnp.float32)
    operation = jax.jit(lambda value: accumulation_p.bind(value, coefficient, records=(record,),
                         coefficient_records=(0,), output_size=output_size, dtype=value.dtype))
    expected_values = (source * coefficient).astype(dtype).reshape(3, 7, 5).transpose(0, 2, 1).reshape(3, 35)
    expected = jnp.zeros((3, output_size), dtype=dtype).at[:, record.destination_offset:record.destination_offset + 35].set(expected_values)
    actual = operation(source)
    if dtype == jnp.int32:
        np.testing.assert_array_equal(actual, expected)
    else:
        np.testing.assert_allclose(np.asarray(actual).astype(np.complex64), np.asarray(expected).astype(np.complex64),
                                   rtol=2e-6, atol=1e-6)
    assert_single_native_call(operation.lower(source))


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


@pytest.mark.parametrize("mode", ["leading", "trailing", "nested"])
def test_vmap_axes_use_one_batched_native_call(mode):
    apply = lambda value: copy_p.bind(value, records=PARTITIONS, output_size=16, dtype=value.dtype)
    indices = np.arange(16).reshape(4, 4)[:, [2, 3, 0, 1]].ravel()
    if mode == "leading":
        source = jnp.arange(48, dtype=jnp.float32).reshape(3, 16)
        operation = jax.jit(jax.vmap(apply))
        expected = np.asarray(source)[:, indices]
    elif mode == "trailing":
        source = jnp.arange(48, dtype=jnp.float32).reshape(16, 3)
        operation = jax.jit(jax.vmap(apply, in_axes=1, out_axes=1))
        expected = np.asarray(source)[indices, :]
    else:
        source = jnp.arange(96, dtype=jnp.float32).reshape(16, 2, 3)
        operation = jax.jit(jax.vmap(jax.vmap(apply, in_axes=1, out_axes=1), in_axes=2, out_axes=2))
        expected = np.asarray(source)[indices, :, :]
    np.testing.assert_array_equal(operation(source), expected)
    assert_single_native_call(operation.lower(source))


@pytest.mark.parametrize("compact", [False, True])
def test_nine_dimensional_layout_no_longer_requires_rank_compression(compact):
    strides = tuple((2 if compact else 3)**axis for axis in range(8, -1, -1))
    record = AffineRecord((2,) * 9, strides, 0, tuple(2**axis for axis in range(8, -1, -1)), 0)
    source = jnp.arange(sum(strides) + 1, dtype=jnp.float32)
    actual = jax.jit(lambda value: copy_p.bind(value, records=(record,), output_size=512, dtype=value.dtype))(source)
    np.testing.assert_array_equal(actual, reference_map(np.asarray(source), (record,), (1,), 512))


def test_rank70_broadcast_forward_jvp_and_transpose():
    record = AffineRecord((1,) * 69 + (3,), (0,) * 70, 0, (0,) * 69 + (1,), 0)
    operation = lambda value: copy_p.bind(value, records=(record,), output_size=3, dtype=value.dtype)
    source, direction = jnp.asarray([3.5]), jnp.asarray([-1.25])
    actual, tangent = jax.jit(lambda value, dot: jax.jvp(operation, (value,), (dot,)))(source, direction)
    np.testing.assert_array_equal(actual, np.full(3, 3.5))
    np.testing.assert_array_equal(tangent, np.full(3, -1.25))
    np.testing.assert_array_equal(jax.jit(lambda cot: jax.vjp(operation, source)[1](cot)[0])(jnp.asarray([1., -2., 4.])), [3.])


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float16, jnp.bfloat16, jnp.complex64, jnp.int32])
@pytest.mark.parametrize("layout_kind", ["compact", "transpose", "partial"])
def test_large_scaled_layouts_match_single_worker_and_concurrent_execution(dtype, layout_kind):
    rows, columns = 512, 513
    size = rows * columns
    offset = 32 if layout_kind == "partial" else 0
    output_size = size + 2 * offset
    record = (AffineRecord((size,), (1,), 0, (1,), 0) if layout_kind == "compact" else
              AffineRecord((rows, columns), (1, rows), 0, (columns, 1), offset))
    source = (jnp.arange(size, dtype=jnp.int32) % 64 - 32).astype(dtype)
    coefficient = jnp.asarray(-1, dtype=jnp.int32)
    operation = jax.jit(lambda value: accumulation_p.bind(value, coefficient, records=(record,),
                         coefficient_records=(0,), output_size=output_size, dtype=value.dtype)).lower(source).compile()
    previous = get_num_threads()
    try:
        for workers in (1, 4):
            set_num_threads(workers)
            actual = operation(source)
            np.testing.assert_array_equal(actual, reference_map(np.asarray(source), (record,), (-1,), output_size))
        with ThreadPoolExecutor(max_workers=4) as executor:
            values = [source + jnp.asarray(index, dtype=dtype) for index in range(4)]
            outputs = list(executor.map(operation, values))
        for value, actual in zip(values, outputs, strict=True):
            np.testing.assert_array_equal(actual, reference_map(np.asarray(value), (record,), (-1,), output_size))
    finally:
        if previous is None:
            enable_threads()
        else:
            set_num_threads(previous)


def _check_sharding():
    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    assert len(jax.devices()) == 2
    mesh = Mesh(np.asarray(jax.devices()), ("device",))
    batches = NamedSharding(mesh, PartitionSpec("device", None))
    replicated = NamedSharding(mesh, PartitionSpec())
    partitioned = NamedSharding(mesh, PartitionSpec("device"))
    size = 131072
    record = AffineRecord((size,), (1,), 0, (1,), 0)
    apply = lambda value: copy_p.bind(value, records=(record,), output_size=size, dtype=value.dtype)
    host = np.arange(2 * size, dtype=np.float32).reshape(2, size)
    for values, sharding in ((host, batches), (host[0], replicated)):
        source = jax.device_put(values, sharding)
        compiled = jax.jit(apply, in_shardings=sharding, out_shardings=sharding).lower(source).compile()
        actual = compiled(source)
        np.testing.assert_array_equal(actual, values)
        assert actual.sharding.is_equivalent_to(sharding, values.ndim)
        text = compiled.as_text().lower()
        assert "tensor0_stride_copy_f32_cpu_v1" in text
        assert "all-gather" not in text and "all-reduce" not in text
    np.testing.assert_array_equal(jax.pmap(apply)(jnp.asarray(host)), host)
    source = jax.device_put(host[0], partitioned)
    with pytest.raises(Exception, match="cannot shard the packed storage axis"):
        jax.jit(apply, in_shardings=partitioned, out_shardings=partitioned).lower(source).compile()
    assert not any(name == "tensor0._stride1" or name.startswith("tensor0._stride1.") or
                   name == "tensor0.operations._strided" for name in sys.modules)


def test_multi_device_copy_batch_replication_pmap_and_storage_boundary():
    environment = os.environ.copy()
    environment["JAX_PLATFORMS"] = "cpu"
    environment["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    completed = subprocess.run([sys.executable, "-c", "from tests.stride.test_ffi import _check_sharding; _check_sharding()"],
                               cwd=Path(__file__).resolve().parents[2], env=environment,
                               capture_output=True, text=True, timeout=180)
    assert completed.returncode == 0, completed.stdout + completed.stderr
