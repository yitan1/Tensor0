"""Optional CUDA forward-copy integration (no CPU fallback)."""

from concurrent.futures import ThreadPoolExecutor

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensor0 import _native
from tensor0._stride._jax import copy_p
from tensor0._stride._layout import AffineRecord


@pytest.fixture
def cuda_device():
    if not getattr(_native, "_stride_cuda_available", lambda: False)():
        pytest.skip("native CUDA copy is unavailable")
    try:
        return jax.devices("cuda")[0]
    except RuntimeError:
        pytest.skip("CUDA device is unavailable")


def _copy(source, records, output_size, dtype=None):
    return copy_p.bind(source, records=records, output_size=output_size,
                       dtype=source.dtype if dtype is None else np.dtype(dtype))


@pytest.mark.parametrize("dtype", ["float32", "float64", "complex64", "complex128"])
def test_cuda_copy_affine_batches_eager_jit_vmap(cuda_device, dtype):
    records = (AffineRecord((2, 3), (-3, 1), 3, (1, 3), 1),)
    with jax.enable_x64():
        values = np.arange(24).reshape(2, 2, 6).astype(dtype)
        if np.issubdtype(values.dtype, np.complexfloating):
            values += 1j * (values + 1)
        source = jax.device_put(values, cuda_device)
        expected = np.zeros((2, 2, 10), dtype=dtype)
        for i in range(2):
            for j in range(3):
                expected[..., 1 + i + 3 * j] = values[..., 3 - 3 * i + j]
        operation = lambda value: _copy(value, records, 10)
        compiled = jax.jit(operation)
        for result in (operation(source), compiled(source), jax.vmap(operation)(source)):
            np.testing.assert_array_equal(result, expected)
            assert result.devices() == {cuda_device}
        text = str(compiled.lower(source).compiler_ir("stablehlo"))
        assert f"tensor0_stride_copy_{ {'float32': 'f32', 'float64': 'f64', 'complex64': 'c64', 'complex128': 'c128'}[dtype]}_cuda_v1" in text
        assert "_cpu_v1" not in text


@pytest.mark.parametrize("records,source_shape,output_size", [
    ((), (2, 3), 5),
    ((AffineRecord((0,), (1,), 0, (1,), 0),), (2, 0), 4),
    ((), (0, 3), 4),
    ((), (2, 3), 0),
])
def test_cuda_copy_empty(cuda_device, records, source_shape, output_size):
    source = jax.device_put(np.zeros(source_shape, np.float32), cuda_device)
    result = jax.jit(lambda value: _copy(value, records, output_size))(source)
    np.testing.assert_array_equal(result, np.zeros((*source_shape[:-1], output_size), np.float32))


def test_cuda_copy_i64_metadata_with_x64_disabled(cuda_device):
    # An unused singleton stride is deliberately not representable as int32.
    records = (AffineRecord((1,), (1 << 33,), 0, (1 << 34,), 1),)
    with jax.enable_x64(False):
        source = jax.device_put(np.array([7], np.float32), cuda_device)
        operation = jax.jit(lambda value: _copy(value, records, 3))
        text = str(operation.lower(source).compiler_ir("stablehlo"))
        assert "tensor<10xi64>" in text
        assert "8589934592" in text
        assert "17179869184" in text
        np.testing.assert_array_equal(operation(source), [0, 7, 0])


@pytest.mark.parametrize("source_dtype,result_dtype", [
    ("int32", "int32"), ("float16", "float16"), ("float32", "complex64"),
])
def test_cuda_copy_unsupported_dtype(cuda_device, source_dtype, result_dtype):
    source = jax.device_put(np.arange(4).astype(source_dtype), cuda_device)
    with pytest.raises(NotImplementedError, match="same-dtype"):
        jax.jit(lambda value: _copy(value, (), 4, result_dtype))(source)


def test_cuda_repeated_concurrent_async_copy(cuda_device):
    records = (AffineRecord((32,), (-1,), 31, (1,), 1),)
    source = jax.device_put(np.arange(32, dtype=np.float32), cuda_device)
    operation = jax.jit(lambda value: _copy(value, records, 34))
    operation(source).block_until_ready()
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda _: operation(source), range(16)))
    for result in results:
        np.testing.assert_array_equal(result, np.r_[0, np.arange(32)[::-1], 0])


def test_explicit_cpu_compilation_on_cuda_host(cuda_device):
    # CUDA is available, but placement must choose the CPU lowering independently.
    source = jax.device_put(np.arange(4, dtype=np.float32), jax.devices("cpu")[0])
    records = (AffineRecord((4,), (-1,), 3, (1,), 0),)
    operation = jax.jit(lambda value: _copy(value, records, 4))
    text = str(operation.lower(source).compiler_ir("stablehlo"))
    assert "tensor0_stride_copy_f32_cpu_v1" in text
    assert "_cuda_v1" not in text
    np.testing.assert_array_equal(operation(source), [3, 2, 1, 0])


@pytest.mark.parametrize("records,source_size,output_size", [
    ((AffineRecord((), (), 2, (), 3),), 4, 5),
    ((AffineRecord((2, 3), (0, 1), 1, (3, 1), 0),), 4, 6),
    ((AffineRecord((2, 2), (3, 1), 1, (-4, -1), 6),), 6, 8),
    ((AffineRecord((2,), (1,), 0, (1,), 0),
      AffineRecord((2,), (1,), 3, (1,), 4)), 5, 7),
    ((AffineRecord((1,) * 8 + (3,), (0,) * 8 + (1,), 0,
                  (0,) * 8 + (2,), 1),), 3, 7),
])
def test_cuda_copy_address_cases(cuda_device, records, source_size, output_size):
    values = np.arange(source_size, dtype=np.float32) + 1
    expected = np.zeros(output_size, np.float32)
    for record in records:
        for index in np.ndindex(record.logical_shape):
            source_address = record.source_offset + sum(i * s for i, s in zip(index, record.source_strides))
            destination_address = record.destination_offset + sum(i * s for i, s in zip(index, record.destination_strides))
            expected[destination_address] = values[source_address]
    source = jax.device_put(values, cuda_device)
    result = jax.jit(lambda value: _copy(value, records, output_size))(source)
    np.testing.assert_array_equal(result, expected)


def test_partition_callback_preserves_compilation_platform(monkeypatch):
    from types import SimpleNamespace
    from tensor0._stride import _jax
    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    mesh = Mesh(np.asarray(jax.devices("cpu")[:1]), ("batch",))
    sharding = NamedSharding(mesh, PartitionSpec("batch", None))
    shape = SimpleNamespace(shape=(1, 4), sharding=sharding)
    _, function, _, _ = _jax._partition_copy((), 4, np.dtype("float32"), "cuda", mesh, (shape,), shape)
    observed = []
    monkeypatch.setattr(_jax, "execute_copy", lambda source, **kwargs: observed.append(kwargs))
    function(np.zeros((1, 4), np.float32))
    assert observed[0]["platform"] == "cuda"


@pytest.mark.parametrize("dtype", ["float32", "float64", "complex64", "complex128"])
def test_cuda_copy_preserves_storage_bits(cuda_device, dtype):
    component = np.dtype("float32" if dtype in ("float32", "complex64") else "float64")
    integer = np.dtype("uint32" if component.itemsize == 4 else "uint64")
    words = ([0, 0x80000000, 0x7f800000, 0xff800000, 0x7fc12345, 1]
             if component.itemsize == 4 else
             [0, 0x8000000000000000, 0x7ff0000000000000, 0xfff0000000000000,
              0x7ff8123456789abc, 1])
    values = np.array(words * 2, dtype=integer).view(dtype)
    records = (AffineRecord((values.size,), (-1,), values.size - 1, (2,), 1),)
    expected = np.zeros(2 * values.size + 1, dtype=dtype)
    expected[1::2] = values[::-1]
    with jax.enable_x64():
        source = jax.device_put(values, cuda_device)
        result = jax.jit(lambda value: _copy(value, records, expected.size))(source)
        np.testing.assert_array_equal(np.asarray(result).view(integer), expected.view(integer))
        np.testing.assert_array_equal(np.asarray(source).view(integer), values.view(integer))


def test_cuda_async_independent_layouts_and_buffers(cuda_device):
    records = [(AffineRecord((32,), (1,), 0, (2,), offset),) for offset in (0, 1)]
    operations = [jax.jit(lambda value, rec=rec: _copy(value, rec, 66)) for rec in records]
    sources = [jax.device_put(np.arange(32, dtype=np.float32) + i, cuda_device) for i in range(12)]
    for operation in operations:
        operation(sources[0]).block_until_ready()
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda i: operations[i % 2](sources[i]), range(12)))
    for i, result in enumerate(results):
        expected = np.zeros(66, np.float32)
        expected[i % 2:64:2] = np.arange(32) + i
        np.testing.assert_array_equal(result, expected)


def test_cuda_copy_jvp_and_transpose(cuda_device):
    records = (AffineRecord((4,), (-1,), 3, (1,), 1),)
    source = jax.device_put(np.arange(4, dtype=np.float32), cuda_device)
    tangent = jax.device_put(np.ones(4, np.float32), cuda_device)
    operation = lambda value: _copy(value, records, 6)
    primal, derivative = jax.jit(lambda value, direction: jax.jvp(operation, (value,), (direction,)))(source, tangent)
    np.testing.assert_array_equal(primal, [0, 3, 2, 1, 0, 0])
    np.testing.assert_array_equal(derivative, [0, 1, 1, 1, 1, 0])
    np.testing.assert_array_equal(jax.jit(jax.grad(lambda value: jnp.sum(operation(value))))(source), np.ones(4))


@pytest.mark.parametrize("dtype", ["float32", "float64", "complex64", "complex128"])
def test_cuda_copy_multiple_blocks_and_batch_tails(cuda_device, dtype):
    records = (AffineRecord((257,), (-1,), 256, (2,), 1),)
    values = np.arange(3 * 257).reshape(3, 257).astype(dtype)
    expected = np.zeros((3, 519), dtype=dtype)
    expected[:, 1:514:2] = values[:, ::-1]
    with jax.enable_x64():
        source = jax.device_put(values, cuda_device)
        result = jax.jit(lambda value: _copy(value, records, 519))(source)
        np.testing.assert_array_equal(result, expected)


def test_cuda_copy_grid_stride_second_iteration(cuda_device):
    size = 65535 * 256 + 17
    records = (AffineRecord((size,), (0,), 0, (1,), 0),)
    source = jax.device_put(np.array([3.25], np.float32), cuda_device)
    result = jax.jit(lambda value: _copy(value, records, size))(source)
    np.testing.assert_array_equal(result, np.full(size, 3.25, np.float32))


def test_cpu_only_extension_rejects_cuda_lowering():
    if getattr(_native, "_stride_cuda_available", lambda: False)():
        pytest.skip("requires a CPU-only extension")
    try:
        device = jax.devices("cuda")[0]
    except RuntimeError:
        pytest.skip("CUDA device is unavailable")
    source = jax.device_put(np.arange(4, dtype=np.float32), device)
    with pytest.raises(RuntimeError, match="CUDA copy is unavailable"):
        jax.jit(lambda value: _copy(value, (), 4))(source)
