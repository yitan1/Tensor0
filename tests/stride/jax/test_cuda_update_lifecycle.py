"""CUDA Update prepared-state validation, reuse, and isolation."""

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
from tensor0._stride._ffi._descriptor import encode_layout
from tensor0._stride._layout import AffineRecord
from tests.stride.support.paths import REPO_ROOT


@pytest.fixture
def cuda_device():
    return cuda_device_or_skip()


def _layout(reverse=False):
    return encode_layout((AffineRecord((6,), (-1 if reverse else 1,),
                                       5 if reverse else 0, (2,), 1),),
                         source_size=6, output_size=13)


def _expected(source, base, alpha, beta, reverse=False):
    result = base.copy()
    a = np.broadcast_to(alpha, (source.shape[0],))
    b = np.broadcast_to(beta, (source.shape[0],))
    values = source[:, ::-1] if reverse else source
    result[:, 1:13:2] = a[:, None] * values + b[:, None] * base[:, 1:13:2]
    return result


def _operation(layout):
    return jax.jit(lambda source, base, alpha, beta: _calls.execute_update(
        source, base, alpha, beta, layout=layout, platform="cuda"))


def _raw_operation(layout, dtype=np.float32):
    dtype = np.dtype(dtype)
    target = _registration.cuda_update_target(dtype, dtype, dtype, dtype)
    call = jax.ffi.ffi_call(target, jax.ShapeDtypeStruct((1, 13), dtype),
                            input_output_aliases={1: 0})
    return jax.jit(lambda source, base, alpha, beta, table:
                   call(source, base, alpha, beta, table, layout=layout))


def test_cpu_only_cuda_update_stats_are_none():
    if _native._stride_cuda_available():
        pytest.skip("requires a CPU-only extension")
    assert _native._stride_cuda_update_prepared_stats() is None


@pytest.mark.parametrize("mutation,message", [
    ("version", "unsupported native layout version"),
    ("destination", "injective|overlap"),
])
def test_raw_descriptor_rejected_at_compile_before_state(cuda_device, mutation, message):
    layout = _layout()
    if mutation == "version":
        layout[0] = 99
    else:
        # Destination stride zero writes all six elements to the same address.
        layout[-1] = 0
    with jax.enable_x64():
        source = jax.device_put(np.arange(6, dtype=np.float32).reshape(1, 6), cuda_device)
        base = jax.device_put(np.ones((1, 13), np.float32), cuda_device)
        coefficient = jax.device_put(np.array(1, np.float32), cuda_device)
        table = jax.device_put(layout, cuda_device)
        operation = _raw_operation(layout)
        created = _native._stride_cuda_update_prepared_stats()[0]
        with pytest.raises(Exception, match=message):
            operation.lower(source, base, coefficient, coefficient, table).compile()
        assert _native._stride_cuda_update_prepared_stats()[0] == created


@pytest.mark.parametrize("shape", [(), (1,), (3,)])
def test_compiled_update_dynamic_coefficients_and_inputs(cuda_device, shape):
    with jax.enable_x64():
        layout = _layout(True)
        source = jax.device_put(np.arange(18, dtype=np.float32).reshape(3, 6), cuda_device)
        base = jax.device_put(np.arange(39, dtype=np.float32).reshape(3, 13), cuda_device)
        def coefficient(value):
            return jax.device_put(np.full(shape, value, np.float32), cuda_device)
        operation = _operation(layout)
        before = _native._stride_cuda_update_prepared_stats()[0]
        compiled = operation.lower(source, base, coefficient(0), coefficient(0)).compile()
        assert _native._stride_cuda_update_prepared_stats()[0] == before + 1
        for index, (a, b) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1), (2, -1), (-1, 2))):
            host_source = np.arange(18, dtype=np.float32).reshape(3, 6) + index
            host_base = np.arange(39, dtype=np.float32).reshape(3, 13) - index
            result = compiled(jax.device_put(host_source, cuda_device),
                              jax.device_put(host_base, cuda_device), coefficient(a), coefficient(b))
            assert result.devices() == {cuda_device}
            np.testing.assert_array_equal(result, _expected(host_source, host_base, a, b, True))
        assert _native._stride_cuda_update_prepared_stats()[0] == before + 1


def test_concurrent_calls_keep_coefficients_and_bases_independent(cuda_device):
    with jax.enable_x64():
        operation = _operation(_layout())
        source = jax.device_put(np.zeros((3, 6), np.float32), cuda_device)
        base = jax.device_put(np.zeros((3, 13), np.float32), cuda_device)
        coefficient = jax.device_put(np.zeros(3, np.float32), cuda_device)
        before = _native._stride_cuda_update_prepared_stats()[0]
        compiled = operation.lower(source, base, coefficient, coefficient).compile()
        assert _native._stride_cuda_update_prepared_stats()[0] == before + 1

        def submit(i):
            host_source = np.arange(18, dtype=np.float32).reshape(3, 6) + i
            host_base = np.full((3, 13), i + 1, np.float32)
            a = np.array([0, 1, 2], np.float32) + i % 2
            b = np.array([2, 0, -1], np.float32) + i % 3
            result = compiled(*(jax.device_put(value, cuda_device)
                                for value in (host_source, host_base, a, b)))
            return result, _expected(host_source, host_base, a, b)

        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(submit, range(12)))
        for result, expected in results:
            np.testing.assert_array_equal(result, expected)
        assert _native._stride_cuda_update_prepared_stats()[0] == before + 1


def test_independent_layouts_executables_dtypes_and_release(cuda_device):
    with jax.enable_x64():
        gc.collect()
        before = _native._stride_cuda_update_prepared_stats()[0]
        copy_before = _native._stride_cuda_copy_prepared_stats()
        cpu_before = _native._stride_native_prepared_stats()
        retained = []
        for dtype, reverse in ((np.float32, False), (np.float32, True),
                               (np.complex64, False), (np.complex64, True)):
            source_host = np.arange(12).reshape(2, 6).astype(dtype)
            base_host = np.arange(26).reshape(2, 13).astype(dtype)
            source = jax.device_put(source_host, cuda_device)
            base = jax.device_put(base_host, cuda_device)
            a = jax.device_put(np.array(2, dtype), cuda_device)
            b = jax.device_put(np.array(-1, dtype), cuda_device)
            operation = _operation(_layout(reverse))
            compiled = operation.lower(source, base, a, b).compile()
            retained.append((operation, compiled, source, base, a, b,
                             _expected(source_host, base_host, 2, -1, reverse)))
        assert _native._stride_cuda_update_prepared_stats()[0] == before + 4
        for _ in range(2):
            for _, compiled, source, base, a, b, expected in retained:
                np.testing.assert_array_equal(compiled(source, base, a, b), expected)
        removed = retained.pop(0)
        operation, compiled = removed[:2]
        del removed, compiled
        operation.clear_cache()
        del operation
        gc.collect()
        for _, compiled, source, base, a, b, expected in retained:
            np.testing.assert_array_equal(compiled(source, base, a, b), expected)
        assert _native._stride_cuda_update_prepared_stats()[0] == before + 4
        assert _native._stride_cuda_copy_prepared_stats() == copy_before
        assert _native._stride_native_prepared_stats() == cpu_before


@pytest.mark.parametrize("destruction", ["local", "global", "exit", "compile_only", "lazy_registration"])
def test_isolated_queued_launch_and_destruction(cuda_device, destruction):
    script = ("from tests.stride.support.workers.cuda_update_lifecycle import _check_lifecycle; "
              f"_check_lifecycle({destruction!r})")
    completed = subprocess.run([sys.executable, "-c", script], cwd=REPO_ROOT,
                               capture_output=True, text=True, timeout=180)
    assert completed.returncode == 0, completed.stdout + completed.stderr
