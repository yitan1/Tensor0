"""Optional CUDA Update integration; no CPU fallback or coefficient casts."""

from concurrent.futures import ThreadPoolExecutor
from itertools import product
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensor0 import _native
from tests.stride.support.availability import cuda_device_or_skip
from tensor0._stride import _jax
from tensor0._stride._layout import AffineRecord


@pytest.fixture
def cuda_device():
    return cuda_device_or_skip()


def _update(source, base, alpha, beta, records):
    return _jax.update_p.bind(source, base, alpha, beta, records=records)


MATRIX = [(storage, alpha, beta) for storage, coefficients in (
    ("float32", ("int32", "float32")), ("float64", ("int32", "float64")),
    ("complex64", ("int32", "float32", "complex64")),
    ("complex128", ("int32", "float64", "complex128")),
) for alpha, beta in product(coefficients, repeat=2)]


@pytest.mark.parametrize("storage,alpha_dtype,beta_dtype", MATRIX)
def test_cuda_update_matrix_nine_branches(cuda_device, storage, alpha_dtype, beta_dtype):
    records = (AffineRecord((4,), (-1,), 3, (2,), 1),)
    with jax.enable_x64():
        source = np.arange(36).reshape(9, 4).astype(storage) + 1
        base = np.arange(90).reshape(9, 10).astype(storage) - 7
        if storage.startswith("complex"):
            source += 1j * (source + 1)
            base += 1j * (base - 1)
        alpha = np.array([a for a, b in product((0, 1, 2), repeat=2)], dtype=alpha_dtype)
        beta = np.array([b for a, b in product((0, 1, 3), repeat=2)], dtype=beta_dtype)
        expected = base.copy()
        expected[:, 1:8:2] = alpha[:, None] * source[:, ::-1] + beta[:, None] * base[:, 1:8:2]
        args = tuple(jax.device_put(value, cuda_device) for value in (source, base, alpha, beta))
        operation = lambda s, b, a, c: _update(s, b, a, c, records)
        compiled = jax.jit(operation)
        for result in (operation(*args), compiled(*args), jax.vmap(operation)(*args)):
            np.testing.assert_allclose(result, expected, rtol=2e-6, atol=2e-6)
            assert result.devices() == {cuda_device}
        for actual, original in zip(args, (source, base, alpha, beta)):
            np.testing.assert_array_equal(actual, original)
        text = str(compiled.lower(*args).compiler_ir("stablehlo"))
        assert "tensor0_stride_update_" in text and "_cuda_v1" in text
        assert "_cpu_v1" not in text
        assert "output_operand_aliases" in text


@pytest.mark.parametrize("records,source_size,output_size", [
    ((), 3, 5), ((AffineRecord((0,), (1,), 0, (1,), 0),), 0, 4),
    ((), 0, 0), ((AffineRecord((), (), 2, (), 3),), 4, 5),
    ((AffineRecord((2, 3), (0, 1), 1, (3, 1), 0),), 4, 7),
    ((AffineRecord((2, 2), (3, 1), 1, (-4, -1), 6),), 6, 8),
    ((AffineRecord((2,), (1,), 0, (1,), 0), AffineRecord((2,), (1,), 3, (1,), 4)), 5, 7),
    ((AffineRecord((1, 3), (999, 1), 0, (888, 2), 1),), 3, 8),
    ((AffineRecord((257,), (-1,), 256, (2,), 1),), 257, 519),
])
@pytest.mark.parametrize("batch_shape", [(0,), (3,), (2, 3)])
def test_cuda_update_layouts_empty_holes(cuda_device, records, source_size, output_size, batch_shape):
    source = np.arange(np.prod(batch_shape) * source_size, dtype=np.float32).reshape(*batch_shape, source_size)
    base = np.full((*batch_shape, output_size), -0.0, np.float32)
    expected = base.copy()
    for record in records:
        for index in np.ndindex(record.logical_shape):
            src = record.source_offset + sum(i * s for i, s in zip(index, record.source_strides))
            dst = record.destination_offset + sum(i * s for i, s in zip(index, record.destination_strides))
            expected[..., dst] = 2 * source[..., src]
    args = tuple(jax.device_put(v, cuda_device) for v in (source, base, np.array([2], np.int32), np.array(0, np.int32)))
    result = jax.jit(lambda s, b, a, c: _update(s, b, a, c, records))(*args)
    np.testing.assert_array_equal(np.asarray(result).view(np.uint32), expected.view(np.uint32))


def test_cuda_update_i64_metadata_x64_disabled(cuda_device):
    records = (AffineRecord((0, 1), (1, 1 << 33), 0, (1, 1 << 34), 1),
               AffineRecord((1,), (1,), 0, (1,), 1))
    with jax.enable_x64(False):
        source = jax.device_put(np.array([7], np.float32), cuda_device)
        base = jax.device_put(np.array([2, 3, 4], np.float32), cuda_device)
        operation = jax.jit(lambda s, b: _update(s, b, jnp.int32(1), jnp.int32(0), records))
        text = str(operation.lower(source, base).compiler_ir("stablehlo"))
        assert "tensor<18xi64>" in text and "8589934592" in text and "17179869184" in text
        np.testing.assert_array_equal(operation(source, base), [2, 7, 4])


def test_cuda_update_jvp_and_input_reverse(cuda_device):
    records = (AffineRecord((3,), (-1,), 2, (2,), 1),)
    source, base = (jax.device_put(np.arange(n, dtype=np.float32), cuda_device) for n in (3, 8))
    operation = lambda s, b: _update(s, b, jnp.float32(2), jnp.float32(3), records)
    primal, tangent = jax.jit(lambda s, b: jax.jvp(operation, (s, b), (jnp.ones_like(s), jnp.ones_like(b))))(source, base)
    np.testing.assert_array_equal(primal, [0, 7, 2, 11, 4, 15, 6, 7])
    np.testing.assert_array_equal(tangent, [1, 5, 1, 5, 1, 5, 1, 1])
    np.testing.assert_array_equal(jax.jit(jax.grad(lambda s: operation(s, base).sum()))(source), [2, 2, 2])
    _, coefficient_tangent = jax.jit(lambda a: jax.jvp(
        lambda x: _update(source, base, x, jnp.float32(0), records),
        (a,), (jnp.ones_like(a),)))(jax.device_put(np.float32(2), cuda_device))
    np.testing.assert_array_equal(coefficient_tangent, [0, 2, 0, 1, 0, 0, 0, 0])


def test_cuda_update_concurrent_distinct_calls(cuda_device):
    records = [(AffineRecord((32,), (-1,), 31, (2,), offset),) for offset in (0, 1)]
    operations = [jax.jit(lambda s, b, a, c, rec=rec: _update(s, b, a, c, rec)) for rec in records]
    inputs = [tuple(jax.device_put(v, cuda_device) for v in (
        np.arange(96, dtype=np.float32).reshape(3, 32) + i,
        np.full((3, 66), i, np.float32),
        np.array(i + 1, np.int32) if i % 2 == 0 else np.full(3, i + 1, np.float32),
        np.array([2], np.float32))) for i in range(12)]
    for i in (0, 1):
        operations[i](*inputs[i]).block_until_ready()
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda i: operations[i % 2](*inputs[i]), range(12)))
    for i, result in enumerate(results):
        expected = np.full((3, 66), i, np.float32)
        expected[:, i % 2:64:2] = (np.arange(96).reshape(3, 32)[:, ::-1] + i) * (i + 1) + 2 * i
        np.testing.assert_array_equal(result, expected)


def test_update_partition_callbacks_preserve_platform(monkeypatch):
    from jax.sharding import Mesh, NamedSharding, PartitionSpec
    from jaxlib.mlir import ir
    from jax.interpreters import mlir

    mesh = Mesh(np.asarray(jax.devices("cpu")[:1]), ("batch",))
    sharding = NamedSharding(mesh, PartitionSpec("batch", None))
    shape = SimpleNamespace(shape=(1, 4), sharding=sharding)
    coefficient = SimpleNamespace(shape=())
    shapes = (shape, shape, coefficient, coefficient)
    _, function, _, _ = _jax._partition_update((), "cuda", mesh, shapes, shape)
    observed = []
    monkeypatch.setattr(_jax, "execute_update", lambda *args, **kwargs: observed.append(kwargs))
    function(np.zeros((1, 4)), np.zeros((1, 4)), np.float32(1), np.float32(0))
    assert observed[0]["platform"] == "cuda"
    assert _jax._infer_update_sharding((), "cuda", mesh, shapes, shape) == sharding
    assert _jax._propagate_update_sharding((), "cuda", mesh, shape) == sharding
    with mlir.make_ir_context(), ir.Location.unknown():
        f32 = ir.F32Type.get()
        types = [ir.RankedTensorType.get(s.shape, f32) for s in shapes]
        assert _jax._update_sharding_rule((), "cuda", mesh, types, types[1:2]) == (
            "batch0 source, batch0 result, ,  -> batch0 result")


def test_cuda_host_explicit_cpu_update(cuda_device):
    source = jax.device_put(np.arange(4, dtype=np.float32), jax.devices("cpu")[0])
    records = (AffineRecord((4,), (-1,), 3, (1,), 0),)
    operation = jax.jit(lambda s: _update(s, s, jnp.int32(2), jnp.int32(1), records))
    text = str(operation.lower(source).compiler_ir("stablehlo"))
    assert "tensor0_stride_update_f32_cpu_v1" in text and "_cuda_v1" not in text
    np.testing.assert_array_equal(operation(source), [6, 5, 4, 3])


@pytest.mark.parametrize("alpha,beta", list(product((0, 1, 2), repeat=2)))
def test_cuda_update_nonfinite_branch_classification(cuda_device, alpha, beta):
    source = np.array([np.nan, np.inf, -np.inf, -0.0, np.finfo(np.float32).max], np.float32)
    base = np.array([np.inf, np.nan, np.inf, -0.0, -np.finfo(np.float32).max, -0.0], np.float32)
    records = (AffineRecord((5,), (1,), 0, (1,), 0),)
    with np.errstate(all="ignore"):
        left = source if alpha == 1 else alpha * source
        right = base[:5] if beta == 1 else beta * base[:5]
        selected = (np.zeros(5, np.float32) if alpha == beta == 0 else
                    right if alpha == 0 else left if beta == 0 else left + right)
    args = tuple(jax.device_put(v, cuda_device) for v in (source, base))
    result = np.asarray(jax.jit(lambda s, b: _update(s, b, jnp.int32(alpha), jnp.int32(beta), records))(*args))
    np.testing.assert_array_equal(result[:5], selected)
    np.testing.assert_array_equal(np.signbit(result[:5][selected == 0]), np.signbit(selected[selected == 0]))
    assert np.signbit(result[-1])


@pytest.mark.parametrize("coefficient_dtype", ["float32", "complex64"])
def test_cuda_update_real_vs_complex_nonfinite(cuda_device, coefficient_dtype):
    values = np.empty(3, np.complex64)
    values.real = [np.inf, 2, -0.0]
    values.imag = [3, np.inf, -0.0]
    records = (AffineRecord((3,), (1,), 0, (1,), 0),)
    operation = jax.jit(lambda s, b, a: _update(s, b, a, jnp.int32(0), records))
    outputs = []
    for device in (jax.devices("cpu")[0], cuda_device):
        args = tuple(jax.device_put(v, device) for v in (values, np.zeros_like(values), np.array(2, coefficient_dtype)))
        outputs.append(np.asarray(operation(*args)))
    for component in (np.real, np.imag):
        np.testing.assert_array_equal(component(outputs[0]), component(outputs[1]))


@pytest.mark.parametrize("source_dtype,base_dtype,coefficient_dtype", [
    ("float32", "float32", "int64"), ("float64", "float64", "float32"),
    ("complex64", "complex64", "float64"), ("float32", "float32", "complex64"),
    ("float16", "float16", "int32"), ("float32", "float64", "int32"),
])
def test_cuda_update_rejects_unsupported_types(cuda_device, source_dtype, base_dtype, coefficient_dtype):
    with jax.enable_x64():
        args = tuple(jax.device_put(v, cuda_device) for v in (
            np.ones(4, source_dtype), np.ones(4, base_dtype), np.array(2, coefficient_dtype)))
        with pytest.raises(NotImplementedError, match="CUDA update"):
            jax.jit(lambda s, b, a: _update(s, b, a, jnp.int32(0), ()))(*args)


def test_cuda_update_complex_coefficients(cuda_device):
    records = (AffineRecord((4,), (1,), 0, (1,), 0),)
    source = np.array([1 + 2j, 3 - 4j, -2j, 7], np.complex64)
    base = source[::-1].copy()
    args = tuple(jax.device_put(v, cuda_device) for v in (source, base))
    result = jax.jit(lambda s, b: _update(s, b, jnp.complex64(2 + 3j), jnp.complex64(-1 + 2j), records))(*args)
    np.testing.assert_allclose(result, (2 + 3j) * source + (-1 + 2j) * base, rtol=2e-6)


@pytest.mark.parametrize("dtype", ["float32", "float64", "complex64", "complex128"])
def test_cuda_update_identity_bits_and_live_inputs(cuda_device, dtype):
    integer = np.uint32 if dtype in ("float32", "complex64") else np.uint64
    words = ([0, 0x80000000, 0x7f800000, 0xff800000, 0x7fc12345, 1]
             if integer is np.uint32 else
             [0, 0x8000000000000000, 0x7ff0000000000000, 0xfff0000000000000,
              0x7ff8123456789abc, 1])
    values = np.array(words * 2, integer).view(dtype)
    records = (AffineRecord((values.size,), (-1,), values.size - 1, (2,), 1),)
    base = np.array(words * 6, integer).view(dtype)
    expected = base.copy()
    expected[1:2 * values.size:2] = values[::-1]
    with jax.enable_x64():
        source, original = (jax.device_put(v, cuda_device) for v in (values, base))
        result = jax.jit(lambda s, b: _update(s, b, jnp.int32(1), jnp.int32(0), records))(source, original)
        np.testing.assert_array_equal(np.asarray(result).view(integer), expected.view(integer))
        np.testing.assert_array_equal(np.asarray(source).view(integer), values.view(integer))
        np.testing.assert_array_equal(np.asarray(original).view(integer), base.view(integer))


def test_cuda_update_donated_base(cuda_device):
    source = jax.device_put(np.arange(4, dtype=np.float32), cuda_device)
    base = jax.device_put(np.arange(8, dtype=np.float32), cuda_device)
    records = (AffineRecord((4,), (-1,), 3, (2,), 1),)
    operation = jax.jit(lambda s, b: _update(s, b, jnp.int32(2), jnp.int32(1), records), donate_argnums=(1,))
    np.testing.assert_array_equal(operation(source, base), [0, 7, 2, 7, 4, 7, 6, 7])


@pytest.mark.parametrize("storage", ["float32", "float64", "complex64", "complex128"])
def test_cuda_update_large_s32_binding_matches_cpu(cuda_device, storage):
    real = np.float32 if storage in ("float32", "complex64") else np.float64
    with jax.enable_x64():
        alpha = np.array([2**24 - 1, 2**24 + 1, -(2**24 + 1), -2**31, 2**31 - 1], np.int32)
        beta = alpha[::-1].copy()
        source = np.array([[1.25, -2.5, 3.75]] * 5, dtype=storage)
        base = np.array([[2, 4, -3, 8, .5, 7, 6]] * 5, dtype=storage)
        if np.issubdtype(source.dtype, np.complexfloating):
            source += 1j * source.real.astype(real) / 2
            base -= 1j * base.real.astype(real) / 3
        records = (AffineRecord((3,), (1,), 0, (2,), 1),)
        operation = jax.jit(lambda s, b, a, c: _update(s, b, a, c, records))
        args = source, base, alpha, beta
        cpu = operation(*(jax.device_put(value, jax.devices("cpu")[0]) for value in args))
        gpu = operation(*(jax.device_put(value, cuda_device) for value in args))
        np.testing.assert_array_equal(gpu, cpu)


@pytest.mark.parametrize("storage", ["float32", "float64", "complex64", "complex128"])
def test_cuda_update_nonfinite_coefficients_match_cpu(cuda_device, storage):
    real = np.float32 if storage in ("float32", "complex64") else np.float64
    is_complex = storage.startswith("complex")
    with jax.enable_x64():
        cases = [0., 1., -2., np.inf, -np.inf, np.nan]
        if is_complex:
            cases += [complex(1, -0.), complex(0, -0.), complex(.5, 2),
                      complex(np.inf, 1), complex(1, np.nan)]
        pairs = list(product(cases, repeat=2))
        alpha = np.asarray([p[0] for p in pairs], dtype=storage)
        beta = np.asarray([p[1] for p in pairs], dtype=storage)
        row = np.array([0., -0., 1., -3., np.inf, -np.inf, np.nan], dtype=storage)
        if is_complex:
            row.imag = np.array([0., -0., np.inf, 2., 0., np.nan, -np.inf], dtype=real)
        source = np.tile(row, (len(pairs), 1))
        base = np.tile(row[::-1], (len(pairs), 1))
        records = (AffineRecord((7,), (1,), 0, (1,), 0),)
        operation = jax.jit(lambda s, b, a, c: _update(s, b, a, c, records))
        args = source, base, alpha, beta
        cpu = np.asarray(operation(*(jax.device_put(v, jax.devices("cpu")[0]) for v in args)))
        gpu = np.asarray(operation(*(jax.device_put(v, cuda_device) for v in args)))
        for actual, expected in ((gpu.real, cpu.real), (gpu.imag, cpu.imag)):
            np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
            np.testing.assert_array_equal(np.isposinf(actual), np.isposinf(expected))
            np.testing.assert_array_equal(np.isneginf(actual), np.isneginf(expected))
            mask = np.isfinite(expected)
            np.testing.assert_array_equal(actual[mask], expected[mask])


@pytest.mark.parametrize("storage", ["float32", "float64", "complex64", "complex128"])
def test_cuda_update_subnormal_arithmetic_and_separate_products(cuda_device, storage):
    real = np.float32 if storage in ("float32", "complex64") else np.float64
    with jax.enable_x64():
        tiny = np.nextafter(real(0), real(1))
        largest = np.finfo(real).max
        source = np.array([tiny, -tiny, largest], dtype=storage)
        base = np.array([tiny, -tiny, -largest], dtype=storage)
        if storage.startswith("complex"):
            source.imag = source.real
            base.imag = base.real
        records = (AffineRecord((3,), (1,), 0, (1,), 0),)
        operation = jax.jit(lambda s, b: _update(s, b, jnp.asarray(2, dtype=real), jnp.int32(1), records))
        result = np.asarray(operation(jax.device_put(source, cuda_device), jax.device_put(base, cuda_device)))
        np.testing.assert_array_equal(result.real[:2], np.array([3 * tiny, -3 * tiny], dtype=real))
        assert np.isposinf(result.real[2])  # A fused 2*x + (-x) would incorrectly stay finite.
        if storage.startswith("complex"):
            np.testing.assert_array_equal(result.imag[:2], np.array([3 * tiny, -3 * tiny], dtype=real))
            assert np.isposinf(result.imag[2])


@pytest.mark.parametrize("storage", ["float32", "float64", "complex64", "complex128"])
def test_cuda_public_scale_and_add(cuda_device, storage):
    from tensor0._stride import StridedView, add, scale

    with jax.enable_x64(), jax.default_device(cuda_device):
        values = np.arange(8).astype(storage)
        other = np.arange(4).astype(storage) + 10
        source = jax.device_put(values, cuda_device)
        rhs = jax.device_put(other, cuda_device)
        operation = jax.jit(lambda x: scale(StridedView(x, (4,), (2,), 1), 2).data)
        expected = values.copy()
        expected[1::2] *= 2
        np.testing.assert_array_equal(operation(source), expected)
        operation = jax.jit(lambda x, y: add(StridedView(x, (4,), (2,), 1),
                                            StridedView(y, (4,), (-1,), 3), alpha=2, beta=3).data)
        expected[1::2] += 3 * other[::-1]
        np.testing.assert_array_equal(operation(source, rhs), expected)
        np.testing.assert_array_equal(source, values)


def test_cpu_only_extension_rejects_cuda_update():
    if getattr(_native, "_stride_cuda_available", lambda: False)():
        pytest.skip("requires a CPU-only extension")
    try:
        device = jax.devices("cuda")[0]
    except RuntimeError:
        pytest.skip("CUDA device is unavailable")
    source = jax.device_put(np.arange(4, dtype=np.float32), device)
    with pytest.raises(RuntimeError, match="CUDA update is unavailable"):
        jax.jit(lambda value: _update(value, value, jnp.int32(1), jnp.int32(0), ()))(source)
