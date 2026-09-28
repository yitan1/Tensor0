"""Same-storage CUDA Dot and the coefficient AD subset it unlocks."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensor0 import _native
from tensor0._stride import _jax, StridedView, dotu, dotc
from tensor0._stride._layout import AffineRecord



DTYPES = ["float32", "float64", "complex64", "complex128"]
CASES = [
    (AffineRecord((3, 2), (2, 1), 1, (-2, 1), 5), 8, 8),
    (AffineRecord((3, 2), (0, -1), 1, (1, 1), 0), 2, 5),
    (AffineRecord((1, 3), (-(1 << 63), 1), 0, ((1 << 63) - 1, -1), 2), 3, 3),
    (AffineRecord((), (), 1, (), 2), 3, 4),
    (AffineRecord((1,) * 20 + (2,), (0,) * 20 + (1,), 0, (0,) * 21, 1), 2, 3),
    (AffineRecord((0,), (1,), 0, (1,), 0), 0, 0),
    (AffineRecord((1024,), (-1,), 1023, (1,), 0), 1024, 1024),
    (AffineRecord((2051,), (-1,), 2050, (0,), 1), 2051, 3),
]


def dot(left, right, records, conjugate=False, dtype=None):
    return _jax.dot_p.bind(left, right, records=records, conjugate_left=conjugate, dtype=dtype)


@pytest.mark.parametrize("record,left_size,right_size", CASES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("conjugate", [False, True])
def test_layouts(cuda_device, record, left_size, right_size, dtype, conjugate):
    with jax.enable_x64():
        left = (np.arange(3 * left_size) % 7).astype(dtype).reshape(3, left_size)
        right = (np.arange(3 * right_size) % 5).astype(dtype).reshape(3, right_size)
        if dtype.startswith("complex"):
            left += 1j * left / 2
            right -= 1j * right
        expected = np.zeros(3, dtype)
        for index in np.ndindex(record.logical_shape):
            a = record.source_offset + sum(i * s for i, s in zip(index, record.source_strides))
            b = record.destination_offset + sum(i * s for i, s in zip(index, record.destination_strides))
            expected += (left[:, a].conj() if conjugate else left[:, a]) * right[:, b]
        args = tuple(jax.device_put(v, cuda_device) for v in (left, right))
        run = lambda a, b: dot(a, b, (record,), conjugate)
        for result in (jax.jit(run)(*args), jax.jit(jax.vmap(run))(*args)):
            np.testing.assert_allclose(result, expected, rtol=2e-6, atol=2e-6)
            assert result.devices() == {cuda_device}
        assert jax.jit(run)(args[0][:0], args[1][:0]).shape == (0,)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("conjugate", [False, True])
def test_public_dot_jvp_vjp(cuda_device, dtype, conjugate):
    with jax.enable_x64(), jax.default_device(cuda_device):
        a = jnp.arange(6, dtype=dtype) / 4
        b = jnp.arange(4, dtype=dtype) / 3
        if dtype.startswith("complex"):
            a = a + 1j * a
            b = b - .5j * b
        op = dotc if conjugate else dotu
        run = lambda x, y: op(StridedView(x, (2, 2), (0, 1), 0), StridedView(y, (2, 2), (0, -1), 2))
        ai, bi = jnp.array([0, 1, 0, 1]), jnp.array([2, 1, 2, 1])
        reference = lambda x, y: jnp.sum((jnp.conj(x[ai]) if conjugate else x[ai]) * y[bi])
        for actual, expected in zip(jax.jit(lambda x, y: jax.jvp(run, (x, y), (x, y)))(a, b), jax.jvp(reference, (a, b), (a, b))):
            np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
        cotangent = jnp.asarray(1 + 2j if dtype.startswith("complex") else 2, dtype)
        actual = jax.jit(lambda x, y, c: jax.vjp(run, x, y)[1](c))(a, b, cotangent)
        expected = jax.vjp(reference, a, b)[1](cotangent)
        for x, y in zip(actual, expected): np.testing.assert_allclose(x, y, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(), (1,), (3,)])
@pytest.mark.parametrize("operation", ["update", "accumulation"])
def test_coefficient_vjp(cuda_device, dtype, shape, operation):
    records = (AffineRecord((3,), (-1,), 3, (2,), 1),)
    with jax.enable_x64():
        source = np.arange(12).reshape(3, 4).astype(dtype) / 4
        base = np.arange(24).reshape(3, 8).astype(dtype) / 3
        if dtype.startswith("complex"):
            source += 1j * source
            base -= .5j * base
        factor = np.full(shape, 1.5 + .25j if dtype.startswith("complex") else 1.5, dtype)
        def run(s, b, a, c):
            if operation == "update": return _jax.update_p.bind(s, b, a, c, records=records)
            return _jax.accumulation_p.bind(s, a, c, records=records * 2, coefficient_records=(0, 1), output_size=8, dtype=s.dtype)
        reverse = jax.jit(lambda s, b, a, c, g: jax.vjp(run, s, b, a, c)[1](g))
        args = source, base, factor, factor, base
        cpu = reverse(*(jax.device_put(v, jax.devices("cpu")[0]) for v in args))
        gpu = reverse(*(jax.device_put(v, cuda_device) for v in args))
        for actual, expected in zip(gpu, cpu):
            np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("operation", ["update", "accumulation"])
def test_real_coefficient_complex_storage_vjp_remains_unsupported(cuda_device, operation):
    records = (AffineRecord((3,), (1,), 0, (1,), 0),)
    with jax.default_device(cuda_device):
        source = jnp.ones(3, jnp.complex64)
        def run(factor):
            if operation == "update": return _jax.update_p.bind(source, source, factor, factor, records=records)
            return _jax.accumulation_p.bind(source, factor, records=records, coefficient_records=(0,), output_size=3, dtype=source.dtype)
        with pytest.raises(NotImplementedError, match="CUDA dot supports only same-dtype"):
            jax.jit(lambda c: jax.vjp(run, c)[1](source))(jnp.float32(2))


@pytest.mark.parametrize("left,right,result", [("float32", "float64", "float64"), ("complex64", "complex64", "float32"), ("float16", "float16", "float16"), ("int32", "int32", "int32")])
def test_unsupported_dtype(cuda_device, left, right, result):
    with jax.enable_x64():
        a, b = (jax.device_put(np.ones(3, d), cuda_device) for d in (left, right))
        with pytest.raises(NotImplementedError, match="CUDA dot supports only same-dtype"):
            jax.jit(lambda x, y: dot(x, y, (), dtype=np.dtype(result)))(a, b)


def test_multirecord_order_concurrent_and_hlo(cuda_device):
    records = tuple(AffineRecord((), (), i, (), i) for i in range(3))
    a = jax.device_put(np.array([2**24, 1, -2**24], np.float32), cuda_device)
    b = jax.device_put(np.ones(3, np.float32), cuda_device)
    np.testing.assert_array_equal(jax.jit(lambda x, y: dot(x, y, records))(a, b), 0)
    operations = [jax.jit(lambda x, y, n=n: dot(x, y, (AffineRecord((n,), (1,), 0, (-1,), n - 1),))) for n in (2051, 3077)]
    args = [(jax.device_put(np.ones(3077, np.float32) * i, cuda_device), jax.device_put(np.ones(3077, np.float32), cuda_device)) for i in range(12)]
    for op in operations: op(*args[0]).block_until_ready()
    with ThreadPoolExecutor(4) as pool:
        values = list(pool.map(lambda i: operations[i % 2](*args[i]), range(12)))
    np.testing.assert_array_equal(values, [i * (2051 if i % 2 == 0 else 3077) for i in range(12)])
    lowered = operations[0].lower(*args[0])
    text = lowered.as_text()
    assert "tensor0_stride_dot_f32_cuda_v1" in text and "_cpu_v1" not in text
    module = lowered.compiler_ir(dialect="stablehlo")
    calls = [inner for operation in module.operation.regions[0].blocks[0].operations
             for region in operation.regions for block in region.blocks
             for inner in block.operations if inner.name == "stablehlo.custom_call"]
    assert len(calls) == 1
    assert [tuple(result.type.shape) for result in calls[0].results] == [(1,), (1, 3)]
    # The second structured result is the XLA-owned partial scratch.


def test_partition_callback_platform(cuda_device, monkeypatch):
    mesh = jax.sharding.Mesh(np.asarray([cuda_device]), ("x",))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    shape = SimpleNamespace(shape=(4,), sharding=sharding)
    result = SimpleNamespace(shape=(), sharding=sharding)
    _, function, _, _ = _jax._partition_dot((), False, np.dtype("float32"), "cuda", mesh, (shape, shape), result)
    seen = []
    monkeypatch.setattr(_jax, "execute_dot", lambda *a, **kw: seen.append(kw))
    function(np.ones(4), np.ones(4))
    assert seen[0]["platform"] == "cuda"
    assert _jax._infer_dot_sharding((), False, np.dtype("float32"), "cuda", mesh, (shape, shape), result) == sharding


def test_cpu_only_dot_diagnostic():
    if getattr(_native, "_stride_cuda_available", lambda: False)(): pytest.skip("requires CPU-only extension")
    try: device = jax.devices("cuda")[0]
    except RuntimeError: pytest.skip("CUDA device is unavailable")
    a = jax.device_put(np.ones(3, np.float32), device)
    with pytest.raises(RuntimeError, match="CUDA dot is unavailable"):
        jax.jit(lambda x: dot(x, x, ()))(a)


@pytest.mark.parametrize("dtype", DTYPES)
def test_random_long_dot_accuracy(cuda_device, dtype):
    with jax.enable_x64():
        rng = np.random.default_rng(19)
        left, right = (rng.normal(size=(3, 100003)).astype(dtype) for _ in range(2))
        if dtype.startswith("complex"):
            left += 1j * rng.normal(size=left.shape)
            right += 1j * rng.normal(size=right.shape)
        records = (AffineRecord((100003,), (1,), 0, (1,), 0),)
        args = tuple(jax.device_put(v, cuda_device) for v in (left, right))
        run = jax.jit(lambda x, y: dot(x, y, records, True))
        result = np.asarray(run(*args))
        wide = np.complex128 if dtype.startswith("complex") else np.float64
        expected = np.sum(left.astype(wide).conj() * right.astype(wide), axis=-1)
        tolerance = 2e-5 if dtype in ("float32", "complex64") else 2e-12
        np.testing.assert_allclose(result, expected, rtol=tolerance, atol=tolerance)
        for _ in range(3): np.testing.assert_array_equal(run(*args), result)


@pytest.mark.parametrize("dtype", DTYPES)
def test_subnormal_nonfinite_and_empty(cuda_device, dtype):
    real = np.float32 if dtype in ("float32", "complex64") else np.float64
    with jax.enable_x64():
        tiny = np.nextafter(real(0), real(1))
        records = (AffineRecord((1,), (1,), 0, (1,), 0),)
        # The CPU runtime can flush subnormals; check CUDA RN against exact bits,
        # not a CPU execution oracle for this particular arithmetic contract.
        tiny_args = (jax.device_put(np.array([v], dtype), cuda_device) for v in (tiny, 2))
        preserved = np.asarray(jax.jit(lambda x, y: dot(x, y, records))(*tiny_args))
        np.testing.assert_array_equal(preserved.real, real(tiny * 2))
        np.testing.assert_array_equal(preserved.imag, 0)
        for a, b in ((np.inf, 2), (np.nan, 2), (np.finfo(real).max, 2)):
            arrays = np.array([a], dtype), np.array([b], dtype)
            run = jax.jit(lambda x, y: dot(x, y, records))
            actual = np.asarray(run(*(jax.device_put(v, cuda_device) for v in arrays)))
            expected = np.asarray(run(*(jax.device_put(v, jax.devices("cpu")[0]) for v in arrays)))
            for x, y in ((actual.real, expected.real), (actual.imag, expected.imag)):
                np.testing.assert_array_equal(np.isnan(x), np.isnan(y))
                np.testing.assert_array_equal(np.isinf(x), np.isinf(y))
                np.testing.assert_array_equal(x[np.isfinite(y)], y[np.isfinite(y)])
        a = jax.device_put(np.ones(3, dtype), cuda_device)
        result = np.asarray(jax.jit(lambda x: dot(x, x, ()))(a))
        assert result == 0 and not np.signbit(result.real)


def test_long_single_owner_fiber_and_x64_disabled(cuda_device):
    count = 4097
    records = (AffineRecord((count,), (0,), 0, (0,), 0),)
    with jax.enable_x64():
        a = jax.device_put(np.ones(1, np.float64), cuda_device)
        np.testing.assert_array_equal(jax.jit(lambda x: dot(x, x, records))(a), count)
    with jax.enable_x64(False):
        a = jax.device_put(np.ones(3, np.float32), cuda_device)
        run = jax.jit(lambda x: dot(x, x, (AffineRecord((3,), (1,), 0, (1,), 0),)))
        np.testing.assert_array_equal(run(a), 3)
        assert "xi64>" in run.lower(a).as_text()


@pytest.mark.parametrize("dtype", DTYPES)
def test_multirecord_sequential_fiber_with_empty_record(cuda_device, dtype):
    records = (AffineRecord((2051,), (1,), 0, (1,), 0),
               AffineRecord((0,), (1,), 0, (1,), 0),
               AffineRecord((1,), (0,), 0, (0,), 0),
               AffineRecord((1025,), (1,), 0, (-1,), 1024))
    with jax.enable_x64():
        value = jax.device_put(np.ones((2, 2051), dtype), cuda_device)
        run = jax.jit(lambda x: dot(x, x, records))
        for _ in range(3): np.testing.assert_array_equal(run(value), [3077, 3077])
