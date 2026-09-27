"""CUDA Reduction executable-owned, explicit-role host state lifecycle."""

import gc
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import jax
import numpy as np
import pytest

from tensor0 import _native
from tests.stride.support.availability import cuda_device_or_skip
from tensor0._stride._ffi import _calls, _registration
from tensor0._stride._ffi._descriptor import encode_reduction_layout
from tensor0._stride._layout import AffineRecord
from tests.stride.support.paths import REPO_ROOT


@pytest.fixture
def cuda_device():
    return cuda_device_or_skip()


def _layout(role=True, reverse=False):
    records = (AffineRecord((0,), (1,), 0, (1,), 0),
               AffineRecord((2, 2), (2, 1), 0, (1 if role else 2, 0 if role else 1), 0),
               AffineRecord((2,), (-1 if reverse else 1,), 1 if reverse else 0, (1,), 3))
    roles = ((False,), (False, role), (False,))
    shapes = ((0,), (2, 1 if role else 2), (2,))
    return encode_reduction_layout(records, output_shapes=shapes, reduction_axes=roles,
                                   source_size=4, output_size=5)


def _expected(host, factor, role=True, reverse=False, mapping=False):
    result = np.zeros((host.shape[0], 5), host.dtype)
    for batch in range(host.shape[0]):
        for row in range(2):
            for column in range(2):
                dst = row * (1 if role else 2) + (0 if role else column)
                result[batch, dst] += (1 if mapping else factor[batch]) * host[batch, row * 2 + column]
        for column in range(2):
            result[batch, 3 + column] += (factor[batch] if mapping else 1) * host[batch, 1 - column if reverse else column]
    return result


def _operation(layout, indices=(0, 1)):
    return jax.jit(lambda source, empty, factor: _calls.execute_reduction(
        source, (empty, factor), coefficient_records=indices, layout=layout,
        output_size=5, platform="cuda"))


def test_cpu_only_reduction_stats_none():
    if _native._stride_cuda_available():
        pytest.skip("requires CPU-only extension")
    assert _native._stride_cuda_reduction_prepared_stats() is None


@pytest.mark.parametrize("change,message", [("version", "unsupported"),
                                              ("role", "not boolean"),
                                              ("overflow", "layout element count overflows"),
                                              ("index", "record indices")])
def test_static_compile_rejects_without_state(cuda_device, change, message):
    with jax.enable_x64():
        layout = _layout().view("<i8").copy()
        indices = (0, 1)
        if change == "version": layout[0] = 99
        if change == "role": layout[24] = 2  # tested below against the raw role position
        if change == "overflow":
            layout = np.asarray([1, 4, 5, 1, 3, 0, 0, -1, 2, 0, 0, 0, 0,
                                 1, 2, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
        if change == "index": indices = (-1, 1)
        dtype = np.dtype("float32")
        target = _registration.cuda_reduction_target(dtype, dtype, dtype, dtype)
        call = jax.ffi.ffi_call(target, (jax.ShapeDtypeStruct((2, 5), dtype),
                                         jax.ShapeDtypeStruct((2, 0), dtype)))
        source = jax.device_put(np.arange(8, dtype=dtype).reshape(2, 4), cuda_device)
        coefficient = jax.device_put(np.array(1, dtype), cuda_device)
        table = jax.device_put(layout, cuda_device)
        operation = jax.jit(lambda a, b, c, d: call(a, b, c, d, layout=layout,
                             coefficient_records=np.asarray(indices, np.int64)))
        before = _native._stride_cuda_reduction_prepared_stats()[0]
        with pytest.raises(Exception, match=message):
            operation.lower(source, table, coefficient, coefficient).compile()
        assert _native._stride_cuda_reduction_prepared_stats()[0] == before


@pytest.mark.parametrize("dtype", ["float32", "float64", "complex64", "complex128"])
@pytest.mark.parametrize("role,reverse", [(True, False), (False, True)])
def test_compiled_reuse_roles_sparse_coefficients_and_dtypes(cuda_device, dtype, role, reverse):
    with jax.enable_x64():
        layout = _layout(role, reverse)
        source = np.arange(12, dtype=np.float32).reshape(3, 4).astype(dtype)
        if dtype.startswith("complex"): source += 1j
        empty = np.array(0, np.int32)
        op = _operation(layout)
        inputs = tuple(jax.device_put(v, cuda_device) for v in (source, empty, np.array([0, 1, 2], np.int32)))
        before = _native._stride_cuda_reduction_prepared_stats()[0]
        compiled = op.lower(*inputs).compile()
        assert _native._stride_cuda_reduction_prepared_stats()[0] == before + 1
        for offset in range(3):
            factors = np.array([offset, 1, 2], np.int32)
            actual = compiled(*(jax.device_put(v, cuda_device) for v in (source + offset, empty, factors)))
            np.testing.assert_array_equal(actual, _expected(source + offset, factors, role, reverse))
        assert _native._stride_cuda_reduction_prepared_stats()[0] == before + 1


def test_concurrent_coefficients_and_executable_isolation(cuda_device):
    with jax.enable_x64():
        source = np.arange(8, dtype=np.float32).reshape(2, 4)
        empty = jax.device_put(np.array(9, np.int32), cuda_device)
        operations = [_operation(_layout(role, reverse)) for role, reverse in ((True, False), (False, True))]
        sample = tuple(jax.device_put(v, cuda_device) for v in (source, np.array([1, 2], np.int32)))
        before = _native._stride_cuda_reduction_prepared_stats()[0]
        compiled = [op.lower(sample[0], empty, sample[1]).compile() for op in operations]
        assert _native._stride_cuda_reduction_prepared_stats()[0] == before + 2
        def invoke(index):
            role, reverse = ((True, False), (False, True))[index % 2]
            host = source + index
            factor = np.array([index % 3, 2], np.int32)
            actual = compiled[index % 2](jax.device_put(host, cuda_device), empty,
                                          jax.device_put(factor, cuda_device))
            return actual, _expected(host, factor, role, reverse)
        with ThreadPoolExecutor(max_workers=4) as pool:
            outputs = list(pool.map(invoke, range(12)))
        for actual, expected in outputs: np.testing.assert_array_equal(actual, expected)
        assert _native._stride_cuda_reduction_prepared_stats()[0] == before + 2
        del compiled, operations
        jax.clear_caches()
        gc.collect()


def test_independent_coefficient_mapping_and_other_state_stats(cuda_device):
    with jax.enable_x64():
        layout = _layout()
        host = np.arange(8, dtype=np.float32).reshape(2, 4) + 1
        factors = np.array([0, 2], np.int32)
        args = tuple(jax.device_put(value, cuda_device) for value in (
            host, np.array(9, np.int32), factors))
        gc.collect()
        other = (_native._stride_native_prepared_stats(), *(
            getattr(_native, f"_stride_cuda_{name}_prepared_stats")()
            for name in ("copy", "update", "accumulation", "dot")))
        before = _native._stride_cuda_reduction_prepared_stats()[0]
        operations = [_operation(layout, indices) for indices in ((0, 1), (0, 2))]
        compiled = [operation.lower(*args).compile() for operation in operations]
        assert _native._stride_cuda_reduction_prepared_stats()[0] == before + 2
        for plan, mapping in zip(compiled, (False, True), strict=True):
            np.testing.assert_array_equal(plan(*args), _expected(host, factors, mapping=mapping))
        assert (_native._stride_native_prepared_stats(), *(
            getattr(_native, f"_stride_cuda_{name}_prepared_stats")()
            for name in ("copy", "update", "accumulation", "dot"))) == other


@pytest.mark.parametrize("destruction", ["local", "global", "exit", "compile_only", "lazy_registration"])
def test_isolated_lifetime(cuda_device, destruction):
    script = ("from tests.stride.support.workers.cuda_reduction_lifecycle import _check_lifecycle; "
              f"_check_lifecycle({destruction!r})")
    completed = subprocess.run([sys.executable, "-c", script], cwd=REPO_ROOT,
                               capture_output=True, text=True, timeout=180)
    assert completed.returncode == 0, completed.stdout + completed.stderr
