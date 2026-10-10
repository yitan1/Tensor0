"""CUDA address accumulation with provably disjoint output owners."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensor0 import _native
from tensor0._stride import _jax
from tensor0._stride._layout import AffineRecord



def accumulate(source, records, size, *coefficients, indices=()):
    return _jax.accumulation_p.bind(source, *coefficients, records=records,
                                   output_size=size, dtype=source.dtype, coefficient_records=indices)


CASES = [
    (AffineRecord((3, 2), (2, 1), 0, (4, 1), 1), 6, 12),  # injective
    (AffineRecord((2, 2), (1, 1), 0, (2, 1), 0), 3, 4),  # repeated source reads stay legal
    (AffineRecord((2, 3, 2), (6, 2, 1), 0, (0, -2, 0), 5), 12, 8),  # fibers
    (AffineRecord((2, 2, 2), (4, 2, 1), 0, (0, 2, 1), 0), 8, 4),
    (AffineRecord((3, 2), (0, -1), 1, (0, 0), 2), 2, 4),
    (AffineRecord((1, 3), (-(1 << 63), 1), 0, ((1 << 63) - 1, -1), 2), 3, 5),
    (AffineRecord((), (), 1, (), 2), 3, 4),
    (AffineRecord((1,) * 20 + (2,), (0,) * 20 + (1,), 0, (0,) * 21, 1), 2, 3),
    (AffineRecord((0,), (1,), 0, (1,), 0), 0, 4),
    (AffineRecord((0,), (1,), 0, (1,), 0), 0, 0),
    (AffineRecord((33, 35), (0, 0), 0, (35, 1), 0), 1, 1155),
    (AffineRecord((257,), (-1,), 256, (2,), 1), 257, 519),
]


@pytest.mark.parametrize("record,source_size,output_size", CASES)
@pytest.mark.parametrize("dtype", ["float32", "float64", "complex64", "complex128"])
@pytest.mark.parametrize("batches", [0, 3])
def test_layouts(cuda_device, record, source_size, output_size, dtype, batches):
    with jax.enable_x64():
        source = (np.arange(batches * source_size) % 7).astype(dtype).reshape(batches, source_size)
        if dtype.startswith("complex"):
            source += 1j * source
        expected = np.zeros((batches, output_size), dtype)
        for index in np.ndindex(record.logical_shape):
            src = record.source_offset + sum(i * s for i, s in zip(index, record.source_strides))
            dst = record.destination_offset + sum(i * s for i, s in zip(index, record.destination_strides))
            expected[:, dst] += source[:, src]
        value = jax.device_put(source, cuda_device)
        operation = lambda s: accumulate(s, (record,), output_size)
        for result in (jax.jit(operation)(value), jax.vmap(operation)(value)):
            np.testing.assert_array_equal(result, expected)
            assert result.devices() == {cuda_device}


MATRIX = [(storage, coefficient) for storage, coefficients in (
    ("float32", ("int32", "float32")), ("float64", ("int32", "float64")),
    ("complex64", ("int32", "float32", "complex64")),
    ("complex128", ("int32", "float64", "complex128")),
) for coefficient in coefficients]


@pytest.mark.parametrize("storage,coefficient", MATRIX)
@pytest.mark.parametrize("shape", [(), (1,), (3,)])
def test_coefficient_matrix_sparse_empty_records(cuda_device, storage, coefficient, shape):
    records = (AffineRecord((0,), (1,), 0, (1,), 0),
               AffineRecord((2, 2), (2, 1), 0, (2, 1), 0),
               AffineRecord((2, 2), (2, 1), 0, (0, 1), 0))
    with jax.enable_x64():
        source = np.arange(12).reshape(3, 4).astype(storage) + 1
        if storage.startswith("complex"):
            source += 1j * source
        for factor in (0, 1, 2):
            factors = np.full(shape, factor, coefficient)
            if shape == (3,): factors[:] = [0, 1, 2]
            if coefficient.startswith("complex") and factor == 2: factors += 1j
            expected = np.zeros((3, 5), storage)
            for r, record in enumerate(records):
                for index in np.ndindex(record.logical_shape):
                    src = sum(i * s for i, s in zip(index, record.source_strides))
                    dst = sum(i * s for i, s in zip(index, record.destination_strides))
                    expected[:, dst] += source[:, src] * (factors.reshape(-1) if r == 2 else 1)
            args = tuple(jax.device_put(v, cuda_device) for v in (source, factors))
            # Empty record retains its own coefficient, not record 1's identity.
            result = jax.jit(lambda s, c: accumulate(s, records, 5, c, c, indices=(0, 2)))(*args)
            np.testing.assert_allclose(result, expected, rtol=2e-6, atol=2e-6)


def test_record_order_and_fiber_grouping(cuda_device):
    records = (AffineRecord((1,), (1,), 0, (1,), 0),
               AffineRecord((2,), (1,), 1, (0,), 0))
    source = jax.device_put(np.array([2**24, 1, -2**24], np.float32), cuda_device)
    operation = jax.jit(lambda s: accumulate(s, records, 1))
    # Starting the second fiber at zero, then adding old, incorrectly produces 1.
    for _ in range(5): np.testing.assert_array_equal(operation(source), [0])
    records = tuple(AffineRecord((), (), i, (), 0) for i in range(3))
    np.testing.assert_array_equal(jax.jit(lambda s: accumulate(s, records, 1))(source), [0])


@pytest.mark.parametrize("storage,coefficient", MATRIX)
def test_numeric_shortcuts_subnormal_and_large_s32(cuda_device, storage, coefficient):
    real = np.float32 if storage in ("float32", "complex64") else np.float64
    with jax.enable_x64():
        records = (AffineRecord((2,), (1,), 0, (0,), 0),)
        for source, factor, expected in (
            ([np.nan, np.inf], 0, 0), ([1, 2], 1, 3),
            ([np.nextafter(real(0), real(1))] * 2, 1, real(np.nextafter(real(0), real(1)) * 2)),
            ([-0.0, -0.0], 1, 0),
        ):
            args = [jax.device_put(np.asarray(v, dtype=d), cuda_device)
                    for v, d in ((source, storage), (factor, coefficient))]
            result = jax.jit(lambda s, c: accumulate(s, records, 1, c, indices=(0,)))(*args)
            np.testing.assert_array_equal(result, [expected])
            assert not np.signbit(np.asarray(result).real[0])
        source = np.asarray([1.25, -0.5], storage)
        factor = np.asarray(2**24 + 1 if coefficient == "int32" else 2, coefficient)
        operation = jax.jit(lambda s, c: accumulate(s, records, 1, c, indices=(0,)))
        cpu = operation(*(jax.device_put(v, jax.devices("cpu")[0]) for v in (source, factor)))
        gpu = operation(*(jax.device_put(v, cuda_device) for v in (source, factor)))
        np.testing.assert_array_equal(gpu, cpu)


@pytest.mark.parametrize("dtype", ["float32", "float64", "complex64", "complex128"])
@pytest.mark.parametrize("strides", [(0, 1), (1, 1), (2, 1)])
def test_copy_and_update_reverse(cuda_device, dtype, strides):
    records = (AffineRecord((2, 2), strides, 0, (2, 1), 0),)
    with jax.enable_x64():
        source = np.arange(4).astype(dtype)
        weights = np.array([1, 2, 3, 4], dtype)
        if dtype.startswith("complex"): weights += 1j
        for operation in (
            lambda s: _jax.copy_p.bind(s, records=records, output_size=4, dtype=s.dtype),
            lambda s: _jax.update_p.bind(s, jnp.zeros(4, s.dtype), jnp.array(2, s.dtype),
                                          jnp.array(0, s.dtype), records=records),
        ):
            reverse = jax.jit(lambda s, w: jax.vjp(operation, s)[1](w)[0])
            if strides == (1, 1):
                for device in (jax.devices("cpu")[0], cuda_device):
                    with pytest.raises(Exception, match="cannot prove injective output owners"):
                        reverse(*(jax.device_put(v, device) for v in (source, weights)))
            else:
                cpu = reverse(*(jax.device_put(v, jax.devices("cpu")[0]) for v in (source, weights)))
                gpu = reverse(*(jax.device_put(v, cuda_device) for v in (source, weights)))
                np.testing.assert_array_equal(gpu, cpu)


def test_concurrent_hlo_and_coefficient_reverse(cuda_device):
    records = [(AffineRecord((2, 2), (2, 1), 0, strides, 0),) for strides in ((2, 1), (0, 1))]
    operations = [jax.jit(lambda s, c, rec=rec: accumulate(s, rec, 4, c, indices=(0,))) for rec in records]
    inputs = [tuple(jax.device_put(v, cuda_device) for v in
                    (np.arange(4, dtype=np.float32) + i, np.array(i + 1, np.float32))) for i in range(12)]
    for i in (0, 1): operations[i](*inputs[i]).block_until_ready()
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda i: operations[i % 2](*inputs[i]), range(12)))
    for i, result in enumerate(results):
        expected = np.array([i, i + 1, i + 2, i + 3]) if i % 2 == 0 else np.array([2 * i + 2, 2 * i + 4, 0, 0])
        np.testing.assert_array_equal(result, expected * (i + 1))
    text = str(operations[0].lower(*inputs[0]).compiler_ir("stablehlo"))
    assert "tensor0_stride_accumulation_f32_cuda_v1" in text
    assert "_cpu_v1" not in text
    source, coefficient = inputs[0]
    gradient = jax.jit(jax.grad(lambda c: _jax.update_p.bind(source, jnp.zeros_like(source), c,
                    jnp.float32(0), records=(AffineRecord((4,), (1,), 0, (1,), 0),)).sum()))(coefficient)
    np.testing.assert_array_equal(gradient, 6)



def test_partition_callbacks_preserve_platform(monkeypatch):
    from jax.sharding import Mesh, NamedSharding, PartitionSpec
    mesh = Mesh(np.asarray(jax.devices("cpu")[:1]), ("batch",))
    sharding = NamedSharding(mesh, PartitionSpec("batch", None))
    shape = SimpleNamespace(shape=(1, 4), sharding=sharding)
    _, function, _, _ = _jax._partition_accumulation((), 4, np.dtype("float32"), (), False, "cuda", mesh, (shape,), shape)
    observed = []
    monkeypatch.setattr(_jax, "execute_accumulation", lambda *args, **kwargs: observed.append(kwargs))
    function(np.zeros((1, 4), np.float32))
    assert observed[0]["platform"] == "cuda"
    assert _jax._infer_accumulation_sharding((), 4, np.dtype("float32"), (), False, "cuda", mesh, (shape,), shape) == sharding
    assert _jax._propagate_accumulation_sharding((), 4, np.dtype("float32"), (), False, "cuda", mesh, shape) == sharding


@pytest.mark.parametrize("storage,coefficient", MATRIX)
def test_nonfinite_real_complex_products_and_per_contribution_scaling(cuda_device, storage, coefficient):
    real = np.float32 if storage in ("float32", "complex64") else np.float64
    with jax.enable_x64():
        values = np.array([np.inf, -np.inf, np.nan, np.finfo(real).max, -np.finfo(real).max], storage)
        if storage.startswith("complex"):
            values.imag = [2, 3, 4, 0, 0]
        records = (AffineRecord((3,), (1,), 0, (1,), 0),
                   AffineRecord((2,), (1,), 3, (0,), 3))
        for factor in (0, 1, 2):
            c = np.asarray(factor, coefficient)
            operation = jax.jit(lambda s, a: accumulate(s, records, 5, a, a, indices=(0, 1)))
            cpu = np.asarray(operation(*(jax.device_put(v, jax.devices("cpu")[0]) for v in (values, c))))
            gpu = np.asarray(operation(*(jax.device_put(v, cuda_device) for v in (values, c))))
            # Explicit component comparison distinguishes real and complex scaling.
            for observed, expected in ((gpu.real, cpu.real), (gpu.imag, cpu.imag)):
                np.testing.assert_array_equal(np.isnan(observed), np.isnan(expected))
                np.testing.assert_array_equal(np.isposinf(observed), np.isposinf(expected))
                np.testing.assert_array_equal(np.isneginf(observed), np.isneginf(expected))
                finite = np.isfinite(expected)
                np.testing.assert_array_equal(observed[finite], expected[finite])


def test_i64_metadata_without_x64(cuda_device):
    empty = AffineRecord((0, 1), (1, 1 << 35), 0, (0, -(1 << 34)), 0)
    record = AffineRecord((1, 2), (1 << 35, 1), 0, (-(1 << 34), 0), 0)
    with jax.enable_x64(False):
        value = jax.device_put(np.array([2, 3], np.float32), cuda_device)
        operation = jax.jit(lambda s: accumulate(s, (empty, record), 1))
        text = str(operation.lower(value).compiler_ir("stablehlo"))
        assert "34359738368" in text and "i64" in text
        np.testing.assert_array_equal(operation(value), [5])


@pytest.mark.parametrize("indices", [(1,), (-1,), (0, 0), (1, 0)])
def test_invalid_sparse_indices(cuda_device, indices):
    record = AffineRecord((0,), (1,), 0, (1,), 0)
    value = jax.device_put(np.zeros(1, np.float32), cuda_device)
    coefficient = jax.device_put(np.float32(0), cuda_device)
    with pytest.raises(Exception, match="record indices|record count|nonnegative int64"):
        jax.jit(lambda s, c: accumulate(s, (record,), 1, *((c,) * len(indices)), indices=indices))(value, coefficient).block_until_ready()


def test_cpu_only_accumulation_diagnostic():
    if getattr(_native, "_stride_cuda_available", lambda: False)():
        pytest.skip("requires CPU-only extension")
    try:
        device = jax.devices("cuda")[0]
    except RuntimeError:
        pytest.skip("CUDA device is unavailable")
    source = jax.device_put(np.arange(4, dtype=np.float32), device)
    with pytest.raises(RuntimeError, match="CUDA accumulation is unavailable"):
        jax.jit(lambda s: accumulate(s, (), 4))(source)


def test_grid_stride_second_iteration(cuda_device):
    size = 65535 * 256 + 7
    record = AffineRecord((size,), (0,), 0, (1,), 0)
    source = jax.device_put(np.array([3], np.float32), cuda_device)
    result = jax.jit(lambda s: accumulate(s, (record,), size))(source)
    assert np.all(np.asarray(result) == 3)


@pytest.mark.parametrize("dtype", ["float32", "float64", "complex64", "complex128"])
def test_update_both_input_vjp_and_coefficient_jvp(cuda_device, dtype):
    record = AffineRecord((2, 2), (1, 1), 0, (3, 1), 1)
    with jax.enable_x64():
        source = np.arange(3).astype(dtype) + 1
        base = np.arange(8).astype(dtype)
        alpha, beta = np.asarray(2, dtype), np.asarray(3, dtype)
        if dtype.startswith("complex"):
            source += 1j
            alpha += 1j
            beta -= 1j
        operation = lambda s, b, a, c: _jax.update_p.bind(s, b, a, c, records=(record,))
        reverse = jax.jit(lambda s, b, a, c: jax.vjp(lambda x, y: operation(x, y, a, c), s, b)[1](jnp.ones_like(b)))
        forward = jax.jit(lambda s, b, a, c: jax.jvp(lambda x, y: operation(s, b, x, y),
                            (a, c), (jnp.ones_like(a), jnp.ones_like(c))))
        cpu_args = tuple(jax.device_put(v, jax.devices("cpu")[0]) for v in (source, base, alpha, beta))
        gpu_args = tuple(jax.device_put(v, cuda_device) for v in (source, base, alpha, beta))
        for args in (cpu_args, gpu_args):
            with pytest.raises(Exception, match="cannot prove injective output owners"):
                reverse(*args)
        for actual, expected in zip(forward(*gpu_args), forward(*cpu_args)):
            np.testing.assert_array_equal(actual, expected)
