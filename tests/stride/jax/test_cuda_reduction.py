"""CUDA reduction roles, fibers, coefficients and existing AD rules."""

from concurrent.futures import ThreadPoolExecutor
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


def reduce_records(source, records, size, *coefficients, indices=()):
    axes = tuple(tuple(stride == 0 for stride in r.destination_strides) for r in records)
    shapes = tuple(tuple(1 if flag else extent for flag, extent in zip(flags, r.logical_shape))
                   for r, flags in zip(records, axes))
    return _jax.reduction_p.bind(source, *coefficients, records=records,
        output_shapes=shapes, reduction_axes=axes,
        output_size=size, dtype=source.dtype, coefficient_records=indices)


CASES = [
    (AffineRecord((3, 2), (2, 1), 0, (4, 1), 1), 6, 12),  # injective
    (AffineRecord((2, 3, 2), (6, 2, 1), 0, (0, -2, 0), 5), 12, 8),  # fibers
    (AffineRecord((2, 2, 2), (1, 1, 1), 0, (0, 2, 1), 0), 4, 4),
    (AffineRecord((3, 2), (0, -1), 1, (0, 0), 2), 2, 4),
    (AffineRecord((1, 3), (-(1 << 63), 1), 0, ((1 << 63) - 1, -1), 2), 3, 5),
    (AffineRecord((), (), 1, (), 2), 3, 4),
    (AffineRecord((1,) * 20 + (2,), (0,) * 20 + (1,), 0, (0,) * 21, 1), 2, 3),
    (AffineRecord((0,), (1,), 0, (1,), 0), 0, 4),
    (AffineRecord((0,), (1,), 0, (1,), 0), 0, 0),
    (AffineRecord((33, 35), (0, 0), 0, (1, 0), 0), 1, 70),
    (AffineRecord((2, 0, 3), (3, 1, 1), 0, (4, 0, 1), 1), 0, 9),
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
        operation = lambda s: reduce_records(s, (record,), output_size)
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
            result = jax.jit(lambda s, c: reduce_records(s, records, 5, c, c, indices=(0, 2)))(*args)
            np.testing.assert_allclose(result, expected, rtol=2e-6, atol=2e-6)


def test_record_order_and_fiber_grouping(cuda_device):
    records = (AffineRecord((1,), (1,), 0, (1,), 0),
               AffineRecord((2,), (1,), 1, (0,), 0))
    source = jax.device_put(np.array([2**24, 1, -2**24], np.float32), cuda_device)
    operation = jax.jit(lambda s: reduce_records(s, records, 1))
    # Starting the second fiber at zero, then adding old, incorrectly produces 1.
    for _ in range(5): np.testing.assert_array_equal(operation(source), [0])
    records = tuple(AffineRecord((), (), i, (), 0) for i in range(3))
    np.testing.assert_array_equal(jax.jit(lambda s: reduce_records(s, records, 1))(source), [0])


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
            result = jax.jit(lambda s, c: reduce_records(s, records, 1, c, indices=(0,)))(*args)
            np.testing.assert_array_equal(result, [expected])
            assert not np.signbit(np.asarray(result).real[0])
        source = np.asarray([1.25, -0.5], storage)
        factor = np.asarray(2**24 + 1 if coefficient == "int32" else 2, coefficient)
        operation = jax.jit(lambda s, c: reduce_records(s, records, 1, c, indices=(0,)))
        cpu = operation(*(jax.device_put(v, jax.devices("cpu")[0]) for v in (source, factor)))
        gpu = operation(*(jax.device_put(v, cuda_device) for v in (source, factor)))
        np.testing.assert_array_equal(gpu, cpu)


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
            operation = jax.jit(lambda s, a: reduce_records(s, records, 5, a, a, indices=(0, 1)))
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
        operation = jax.jit(lambda s: reduce_records(s, (empty, record), 1))
        text = str(operation.lower(value).compiler_ir("stablehlo"))
        assert "34359738368" in text and "i64" in text
        np.testing.assert_array_equal(operation(value), [5])


@pytest.mark.parametrize("indices", [(1,), (-1,), (0, 0), (1, 0)])
def test_invalid_sparse_indices(cuda_device, indices):
    record = AffineRecord((0,), (1,), 0, (1,), 0)
    value = jax.device_put(np.zeros(1, np.float32), cuda_device)
    coefficient = jax.device_put(np.float32(0), cuda_device)
    with pytest.raises(Exception, match="record indices|record count|nonnegative int64"):
        jax.jit(lambda s, c: reduce_records(s, (record,), 1, *((c,) * len(indices)), indices=indices))(value, coefficient).block_until_ready()


def test_cpu_only_reduction_diagnostic():
    if getattr(_native, "_stride_cuda_available", lambda: False)():
        pytest.skip("requires CPU-only extension")
    try:
        device = jax.devices("cuda")[0]
    except RuntimeError:
        pytest.skip("CUDA device is unavailable")
    source = jax.device_put(np.arange(4, dtype=np.float32), device)
    with pytest.raises(RuntimeError, match="CUDA reduction is unavailable"):
        jax.jit(lambda s: reduce_records(s, (), 4))(source)


def test_grid_stride_second_iteration(cuda_device):
    size = 65535 * 256 + 7
    record = AffineRecord((size,), (0,), 0, (1,), 0)
    source = jax.device_put(np.array([3], np.float32), cuda_device)
    result = jax.jit(lambda s: reduce_records(s, (record,), size))(source)
    assert np.all(np.asarray(result) == 3)



@pytest.mark.parametrize("dtype", ["float32", "float64", "complex64", "complex128"])
@pytest.mark.parametrize("coefficient_shape", [(), (1,), (2,)])
def test_reduction_ad(cuda_device, dtype, coefficient_shape):
    records = (AffineRecord((2, 2), (2, 1), 0, (0, 1), 1),
               AffineRecord((2,), (-1,), 2, (0,), 1))
    with jax.enable_x64():
        source = np.arange(8).reshape(2, 4).astype(dtype) + 1
        coefficient = np.full(coefficient_shape, 2, dtype)
        if dtype.startswith("complex"):
            source += 1j
            coefficient += 1j
        operation = lambda s, c: reduce_records(s, records, 4, c, indices=(0,))
        transforms = (
            jax.jit(lambda s, c: jax.jvp(operation, (s, c), (jnp.ones_like(s), jnp.ones_like(c)))),
            jax.jit(lambda s, c: jax.vjp(operation, s, c)[1](jnp.ones((2, 4), s.dtype))),
        )
        gpu = tuple(jax.device_put(v, cuda_device) for v in (source, coefficient))
        cpu = tuple(jax.device_put(v, jax.devices("cpu")[0]) for v in (source, coefficient))
        for transform in transforms:
            for actual, expected in zip(transform(*gpu), transform(*cpu)):
                np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("ignored_stride", [-(1 << 63), (1 << 63) - 1, -99])
def test_original_roles_and_optimized_axes(cuda_device, ignored_stride):
    # Adjacent map axes merge on host; singleton disappears. Nonzero stride on
    # reduced output axes is ignored, not used to classify device fibers.
    record = AffineRecord((2, 2, 1, 2), (4, 2, -(1 << 63), 1), 0, (2, 1, 0, ignored_stride), 1)
    value = jax.device_put(np.arange(8, dtype=np.float32), cuda_device)
    result = jax.jit(lambda s: _jax.reduction_p.bind(s, records=(record,),
        output_shapes=((2, 2, 1, 1),), reduction_axes=((False, False, False, True),),
        output_size=7, dtype=s.dtype))(value)
    np.testing.assert_array_equal(result, [0, 1, 5, 9, 13, 0, 0])


def test_unsigned_extent_without_x64(cuda_device):
    record = AffineRecord(((1 << 64) - 1,), (0,), 0, (99,), 0)
    with jax.enable_x64(False):
        source, factor = (jax.device_put(np.array(v, np.float32), cuda_device) for v in ([2], 0))
        operation = jax.jit(lambda s, c: _jax.reduction_p.bind(s, c, records=(record,),
            output_shapes=((1,),), reduction_axes=((True,),), output_size=1,
            dtype=s.dtype, coefficient_records=(0,)))
        np.testing.assert_array_equal(operation(source, factor), [0])
        text = str(operation.lower(source, factor).compiler_ir("stablehlo"))
        assert "i64" in text and "_reduction_f32_cuda_v1" in text and "_cpu_v1" not in text


def test_concurrent(cuda_device):
    records = (AffineRecord((2, 3), (3, 1), 0, (0, -1), 2),)
    operation = jax.jit(lambda s: reduce_records(s, records, 5))
    values = [jax.device_put(np.arange(6, dtype=np.float32) + i, cuda_device) for i in range(10)]
    operation(values[0]).block_until_ready()
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(operation, values))
    for i, result in enumerate(results):
        np.testing.assert_array_equal(result, [7 + 2*i, 5 + 2*i, 3 + 2*i, 0, 0])


def test_partition_callbacks_preserve_platform(monkeypatch):
    from jax.sharding import Mesh, NamedSharding, PartitionSpec
    mesh = Mesh(np.asarray(jax.devices("cpu")[:1]), ("batch",))
    sharding = NamedSharding(mesh, PartitionSpec("batch", None))
    shape = SimpleNamespace(shape=(1, 4), sharding=sharding)
    args = ((), (), (), 4, np.dtype("float32"), (), "cuda", mesh)
    _, function, _, _ = _jax._partition_reduction(*args, (shape,), shape)
    observed = []
    monkeypatch.setattr(_jax, "execute_reduction", lambda *a, **k: observed.append(k))
    function(np.zeros((1, 4), np.float32))
    assert observed[0]["platform"] == "cuda"
    assert _jax._infer_reduction_sharding(*args, (shape,), shape) == sharding
    assert _jax._propagate_reduction_sharding(*args, shape) == sharding

@pytest.mark.parametrize("dtype", ["float32", "float64", "complex64", "complex128"])
@pytest.mark.parametrize("axes", [None, (), (0,), (1,), (0, 2)])
def test_public_reduce_sum(cuda_device, dtype, axes):
    from tensor0._stride import StridedView, reduce_sum
    with jax.enable_x64():
        source = np.arange(12).astype(dtype)
        if dtype.startswith("complex"): source += 1j
        value = jax.device_put(source, cuda_device)
        operation = jax.jit(lambda s: reduce_sum(StridedView(s, (2, 3, 2), (6, 2, -1), 1), axes))
        dense = source.reshape(2, 3, 2)[..., ::-1]
        np.testing.assert_allclose(operation(value), dense.sum(axis=axes), rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("storage,result,coefficient", [
    ("float16", "float16", "float16"), ("int32", "int32", "int32"),
    ("float32", "float64", "float32"), ("float32", "float32", "int64"),
    ("complex64", "complex64", "float64"),
])
def test_unsupported_types(cuda_device, storage, result, coefficient):
    with jax.enable_x64():
        s, c = (jax.device_put(np.array(v, dtype), cuda_device)
                for v, dtype in (([1], storage), (1, coefficient)))
        with pytest.raises(NotImplementedError, match="same-dtype|coefficient dtype"):
            jax.jit(lambda x, y: _jax.reduction_p.bind(x, y,
                records=(AffineRecord((1,), (1,), 0, (0,), 0),), output_shapes=((1,),),
                reduction_axes=((True,),), output_size=1, dtype=np.dtype(result),
                coefficient_records=(0,)))(s, c)

@pytest.mark.parametrize("dtype", ["float32", "float64", "complex64", "complex128"])
def test_packed_multirecord_trace(cuda_device, dtype):
    from tests.stride.support.oracles.trace import packed_trace, reference_trace
    with jax.enable_x64():
        source = np.arange(24).reshape(2, 12).astype(dtype)
        if dtype.startswith("complex"): source += 1j
        args = tuple(jax.device_put(np.asarray(v, dtype), cuda_device) for v in (source, 2, -1))
        actual = jax.jit(packed_trace)(*args)
        expected = reference_trace(*args)
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
        assert actual.devices() == {cuda_device}


def test_complex_storage_real_coefficient_vjp_remains_unsupported(cuda_device):
    source = jax.device_put(np.array([1+2j, 3+4j], np.complex64), cuda_device)
    coefficient = jax.device_put(np.float32(2), cuda_device)
    record = AffineRecord((2,), (1,), 0, (0,), 0)
    with pytest.raises(NotImplementedError, match="CUDA dot supports only same-dtype"):
        jax.jit(jax.grad(lambda c: jnp.real(reduce_records(source, (record,), 1, c, indices=(0,))[0])))(coefficient)

@pytest.mark.parametrize("dtype", ["float32", "float64", "complex64", "complex128"])
def test_su2_tensortrace_smoke(cuda_device, dtype):
    from tensor0 import SU2Irrep, TensorMap, hom, space, tensortrace
    half = space(SU2Irrep, {1: 1})
    source_space = hom((half, half, half, half.dual()), ())
    with jax.enable_x64(dtype in ("float64", "complex128")):
        source = np.array([1, 2], dtype)
        if dtype.startswith("complex"): source += 1j
        def operation(data):
            return tensortrace(TensorMap(source_space, data), axes=((3,), (0,)),
                               output=((1, 2), ())).storage.data
        source_gpu = jax.device_put(source, cuda_device)
        source_cpu = jax.device_put(source, jax.devices("cpu")[0])
        compiled = jax.jit(operation)
        np.testing.assert_allclose(compiled(source_gpu), compiled(source_cpu), rtol=2e-6, atol=2e-6)
        reverse = jax.jit(lambda s: jax.vjp(operation, s)[1](jnp.ones_like(operation(s)))[0])
        np.testing.assert_allclose(reverse(source_gpu), reverse(source_cpu), rtol=2e-6, atol=2e-6)
        text = str(compiled.lower(source_gpu).compiler_ir("stablehlo"))
        assert "_reduction_" in text and "_cuda_v1" in text and "_cpu_v1" not in text

@pytest.mark.parametrize("dtype", ["float32", "float64", "complex64", "complex128"])
def test_no_records_clear_output(cuda_device, dtype):
    with jax.enable_x64():
        value = jax.device_put(np.zeros((2, 0), dtype), cuda_device)
        result = jax.jit(lambda s: reduce_records(s, (), 4))(value)
        np.testing.assert_array_equal(result, np.zeros((2, 4), dtype))


@pytest.mark.parametrize("dtype", ["float32", "complex64"])
def test_su2_narrow_storage_wide_plan_coefficients_rejected(cuda_device, dtype):
    from tensor0 import SU2Irrep, TensorMap, hom, space, tensortrace
    half = space(SU2Irrep, {1: 1})
    source_space = hom((half, half, half, half.dual()), ())
    with jax.enable_x64():
        source = jax.device_put(np.array([1, 2], dtype), cuda_device)
        with pytest.raises(NotImplementedError, match="coefficient dtype float64"):
            jax.jit(lambda data: tensortrace(TensorMap(source_space, data),
                axes=((3,), (0,)), output=((1, 2), ())).storage.data)(source)
