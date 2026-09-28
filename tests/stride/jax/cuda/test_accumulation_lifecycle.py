"""CUDA Accumulation prepared-state validation, reuse, and isolation."""

import gc
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import jax
import numpy as np
import pytest

from tensor0 import _native
from tensor0._stride._ffi import _calls, _registration
from tensor0._stride._ffi._descriptor import encode_layout
from tensor0._stride._layout import AffineRecord
from tests.stride.support.paths import REPO_ROOT



EMPTY = AffineRecord((2, 0), (3, 1), 0, (2, 1), 0)
MAP = AffineRecord((4,), (1,), 0, (2,), 1)
FIBER = AffineRecord((2, 3), (3, 1), 2, (0, 1), 2)
MAP_TWO = AffineRecord((2, 3), (3, 1), 4, (3, 1), 4)


def _records(reverse=False):
    return (EMPTY, MAP, MAP_TWO if reverse else FIBER, FIBER if reverse else MAP_TWO)


def _layout(records):
    return encode_layout(records, source_size=12, output_size=10)


def _expected(host, records, indices, coefficients):
    result = np.zeros((host.shape[0], 10), dtype=host.dtype)
    factors = dict(zip(indices, coefficients, strict=True))
    for index, record in enumerate(records):
        for coordinate in np.ndindex(record.logical_shape):
            src = record.source_offset + sum(i * s for i, s in zip(coordinate, record.source_strides))
            dst = record.destination_offset + sum(i * s for i, s in zip(coordinate, record.destination_strides))
            for batch in range(host.shape[0]):
                coefficient = factors.get(index)
                factor = 1 if coefficient is None else np.asarray(coefficient).reshape(-1)[0 if np.asarray(coefficient).size == 1 else batch]
                result[batch, dst] += factor * host[batch, src]
    return result


def _operation(layout, indices):
    return jax.jit(lambda source, *coefficients: _calls.execute_accumulation(
        source, coefficients, coefficient_records=indices, layout=layout,
        output_size=10, platform="cuda"))


def _raw_operation(layout, indices, coefficient_count=2):
    dtype = np.dtype("float32")
    target = _registration.cuda_accumulation_target(dtype, dtype, *((dtype,) * coefficient_count))
    call = jax.ffi.ffi_call(target, (jax.ShapeDtypeStruct((2, 10), dtype),
                                         jax.ShapeDtypeStruct((2, 0), dtype)))
    return jax.jit(lambda source, table, *coefficients: call(
        source, table, *coefficients, layout=layout,
        coefficient_records=np.asarray(indices, dtype=np.int64))[0])


def test_cpu_only_cuda_accumulation_stats_are_none():
    if _native._stride_cuda_available():
        pytest.skip("requires a CPU-only extension")
    assert _native._stride_cuda_accumulation_prepared_stats() is None


@pytest.mark.parametrize("mutation,message", [
    ("version", "unsupported native layout version"),
    ("address", "source address exceeds storage"),
])
def test_raw_descriptor_rejected_at_compile_before_state(cuda_device, mutation, message):
    layout = _layout(_records())
    if mutation == "version":
        layout[0] = 99
    else:
        layout[1] = 2  # Source addresses in later records exceed the declared size.
    with jax.enable_x64():
        source = jax.device_put(np.arange(24, dtype=np.float32).reshape(2, 12), cuda_device)
        table = jax.device_put(layout, cuda_device)
        coefficient = jax.device_put(np.array(1, np.float32), cuda_device)
        operation = _raw_operation(layout, (0, 2))
        before = _native._stride_cuda_accumulation_prepared_stats()[0]
        with pytest.raises(Exception, match=message):
            operation.lower(source, table, coefficient, coefficient).compile()
        assert _native._stride_cuda_accumulation_prepared_stats()[0] == before


@pytest.mark.parametrize("indices", [(-1, 2), (0, 0), (2, 1), (0, 4)])
def test_static_coefficient_records_rejected_before_creation(cuda_device, indices):
    with jax.enable_x64():
        layout = _layout(_records())
        source = jax.device_put(np.arange(24, dtype=np.float32).reshape(2, 12), cuda_device)
        table = jax.device_put(np.asarray(_native._stride_pack_owner_fiber("accumulation", layout.tolist()), np.int64), cuda_device)
        coefficient = jax.device_put(np.array(1, np.float32), cuda_device)
        operation = _raw_operation(layout, indices)
        before = _native._stride_cuda_accumulation_prepared_stats()[0]
        with pytest.raises(Exception, match="coefficient records|record indices"):
            operation.lower(source, table, coefficient, coefficient).compile()
        assert _native._stride_cuda_accumulation_prepared_stats()[0] == before


def test_dynamic_rejections_leave_other_live_executable_usable(cuda_device):
    with jax.enable_x64():
        layout = _layout(_records())
        source = jax.device_put(np.arange(24, dtype=np.float32).reshape(2, 12), cuda_device)
        table = jax.device_put(np.asarray(_native._stride_pack_owner_fiber("accumulation", layout.tolist()), np.int64), cuda_device)
        one = jax.device_put(np.array(1, np.float32), cuda_device)
        operation = _raw_operation(layout, (0, 2))
        valid = operation.lower(source, table, one, one).compile()
        expected = _expected(np.asarray(source), _records(), (0, 2), (1, 1))
        np.testing.assert_array_equal(valid(source, table, one, one), expected)
        before = _native._stride_cuda_accumulation_prepared_stats()[0]
        bad_table = jax.device_put(np.zeros(3, np.int64), cuda_device)
        invalid = operation.lower(source, bad_table, one, one).compile()
        with pytest.raises(Exception, match="descriptor operand shape"):
            invalid(source, bad_table, one, one).block_until_ready()
        # The static attribute declares two records but the call has one coefficient.
        mismatch = _raw_operation(layout, (0, 2), coefficient_count=1)
        invalid_count = mismatch.lower(source, table, one).compile()
        with pytest.raises(Exception, match="coefficient record count|coefficient.*operands"):
            invalid_count(source, table, one).block_until_ready()
        np.testing.assert_array_equal(valid(source, table, one, one), expected)
        # Different input signatures compile two additional executables/states.
        # Same-state dynamic rejection/reuse is covered by the native contract.
        assert _native._stride_cuda_accumulation_prepared_stats()[0] == before + 2


@pytest.mark.parametrize("shape", [(), (1,), (3,)])
def test_fixed_executable_dynamic_coefficients_and_ordered_overlap(cuda_device, shape):
    with jax.enable_x64():
        records = _records()
        indices = (0, 2)  # A coefficient on an empty record must not shift later indices.
        operation = _operation(_layout(records), indices)
        host = np.arange(36, dtype=np.float32).reshape(3, 12)
        source = jax.device_put(host, cuda_device)
        def coefficient(value):
            return jax.device_put(np.full(shape, value, np.float32), cuda_device)
        before = _native._stride_cuda_accumulation_prepared_stats()[0]
        compiled = operation.lower(source, coefficient(0), coefficient(0)).compile()
        assert _native._stride_cuda_accumulation_prepared_stats()[0] == before + 1
        for step, (empty, factor) in enumerate(((0, 0), (1, 1), (3, 2), (-2, -1))):
            values = host + step
            result = compiled(jax.device_put(values, cuda_device), coefficient(empty), coefficient(factor))
            assert result.devices() == {cuda_device}
            np.testing.assert_array_equal(result, _expected(values, records, indices, (empty, factor)))
        assert _native._stride_cuda_accumulation_prepared_stats()[0] == before + 1


def test_batched_coefficients_and_concurrent_independent_calls(cuda_device):
    with jax.enable_x64():
        records, indices = _records(), (1, 3)
        operation = _operation(_layout(records), indices)
        source = jax.device_put(np.zeros((3, 12), np.float32), cuda_device)
        coefficient = jax.device_put(np.zeros(3, np.float32), cuda_device)
        before = _native._stride_cuda_accumulation_prepared_stats()[0]
        compiled = operation.lower(source, coefficient, coefficient).compile()
        assert _native._stride_cuda_accumulation_prepared_stats()[0] == before + 1

        def submit(i):
            host = np.arange(36, dtype=np.float32).reshape(3, 12) + i
            first = np.array([0, 1, 2], np.float32) + i % 2
            second = np.array([2, 0, -1], np.float32) + i % 3
            result = compiled(*(jax.device_put(value, cuda_device) for value in (host, first, second)))
            return result, _expected(host, records, indices, (first, second))

        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(submit, range(12)))
        for result, expected in results:
            np.testing.assert_array_equal(result, expected)
        assert _native._stride_cuda_accumulation_prepared_stats()[0] == before + 1


def test_distinct_layouts_and_same_count_coefficient_mappings_isolate(cuda_device):
    with jax.enable_x64():
        gc.collect()
        before = _native._stride_cuda_accumulation_prepared_stats()[0]
        other = (_native._stride_native_prepared_stats(),
                 _native._stride_cuda_copy_prepared_stats(),
                 _native._stride_cuda_update_prepared_stats())
        host = np.arange(24, dtype=np.float32).reshape(2, 12)
        source = jax.device_put(host, cuda_device)
        first = jax.device_put(np.array(0, np.float32), cuda_device)
        second = jax.device_put(np.array(2, np.float32), cuda_device)
        retained = []
        for reverse, indices in ((False, (0, 2)), (False, (1, 3)), (True, (0, 2))):
            records = _records(reverse)
            operation = _operation(_layout(records), indices)
            compiled = operation.lower(source, first, second).compile()
            retained.append((operation, compiled, _expected(host, records, indices, (0, 2))))
        assert _native._stride_cuda_accumulation_prepared_stats()[0] == before + 3
        for _ in range(2):
            for _, compiled, expected in retained:
                np.testing.assert_array_equal(compiled(source, first, second), expected)
        operation, compiled, expected = retained.pop(0)
        del compiled, expected
        operation.clear_cache()
        del operation
        gc.collect()
        for _, compiled, expected in retained:
            np.testing.assert_array_equal(compiled(source, first, second), expected)
        assert _native._stride_cuda_accumulation_prepared_stats()[0] == before + 3
        assert (_native._stride_native_prepared_stats(),
                _native._stride_cuda_copy_prepared_stats(),
                _native._stride_cuda_update_prepared_stats()) == other


@pytest.mark.parametrize("destruction", ["local", "global", "exit", "compile_only", "lazy_registration"])
def test_isolated_queued_launch_and_destruction(cuda_device, destruction):
    script = ("from tests.stride.support.workers.cuda_accumulation_lifecycle import _check_lifecycle; "
              f"_check_lifecycle({destruction!r})")
    completed = subprocess.run([sys.executable, "-c", script], cwd=REPO_ROOT,
                               capture_output=True, text=True, timeout=180)
    assert completed.returncode == 0, completed.stdout + completed.stderr
