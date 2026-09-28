"""CUDA Dot prepared-state validation, reuse, isolation and executable lifetime."""

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



EMPTY = AffineRecord((2, 0), (1, 1), 0, (1, 1), 0)


def _records(reverse=False):
    return (EMPTY, AffineRecord((1025,), (-1 if reverse else 1,),
                                1024 if reverse else 0, (1,), 0),
            AffineRecord((), (), 0, (), 0))


def _layout(records):
    return encode_layout(records, source_size=1025, output_size=1025)


def _expected(left, right, records, conjugate):
    result = np.zeros(left.shape[0], dtype=left.dtype)
    for record in records:
        for index in np.ndindex(record.logical_shape):
            a = record.source_offset + sum(i * s for i, s in zip(index, record.source_strides))
            b = record.destination_offset + sum(i * s for i, s in zip(index, record.destination_strides))
            result += (left[:, a].conj() if conjugate else left[:, a]) * right[:, b]
    return result


def _inputs(dtype, batches=2, shift=0):
    shape = (batches, 1025)
    left = np.resize(np.array([1, -2, 0, 3], dtype=dtype), shape).copy()
    right = np.resize(np.array([2, 0, -1, 1], dtype=dtype), shape).copy()
    left += shift % 3
    if np.issubdtype(dtype, np.complexfloating):
        left += 1j * np.resize(np.array([2, -1, 1], dtype=dtype), shape)
        right += 1j * np.resize(np.array([-1, 2, 0], dtype=dtype), shape)
    return left, right


def _operation(layout, conjugate):
    return jax.jit(lambda left, right: _calls.execute_dot(
        left, right, layout=layout, conjugate_left=conjugate, platform="cuda"))


def _raw_operation(layout, conjugate, dtype=np.float32):
    dtype = np.dtype(dtype)
    target = _registration.cuda_dot_target(dtype, dtype, dtype)
    call = jax.ffi.ffi_call(target, (jax.ShapeDtypeStruct((2,), dtype),
                                     jax.ShapeDtypeStruct((2, 2), dtype)))
    return jax.jit(lambda left, right, table: call(
        left, right, table, layout=layout, conjugate_left=np.int64(conjugate))[0])


def test_cpu_only_cuda_dot_stats_are_none():
    if _native._stride_cuda_available():
        pytest.skip("requires a CPU-only extension")
    assert _native._stride_cuda_dot_prepared_stats() is None


@pytest.mark.parametrize("mutation,message", [
    ("conjugate", "conjugate_left must be zero or one"),
    ("version", "unsupported native layout version"),
    ("address", "source address exceeds storage"),
])
def test_static_attributes_rejected_at_compile_before_state(cuda_device, mutation, message):
    with jax.enable_x64():
        layout = _layout(_records())
        conjugate = 0
        if mutation == "conjugate":
            conjugate = 2
            # An invalid layout too must not mask conjugate validation.
            layout[0] = 99
        elif mutation == "version":
            layout[0] = 99
        else:
            layout[1] = 2
        left, right = (jax.device_put(v, cuda_device) for v in _inputs(np.float32))
        table = jax.device_put(layout, cuda_device)
        operation = _raw_operation(layout, conjugate)
        before = _native._stride_cuda_dot_prepared_stats()[0]
        with pytest.raises(Exception, match=message):
            operation.lower(left, right, table).compile()
        assert _native._stride_cuda_dot_prepared_stats()[0] == before


@pytest.mark.parametrize("conjugate", [-1, 3])
def test_other_invalid_conjugate_values(cuda_device, conjugate):
    with jax.enable_x64():
        layout = _layout(_records())
        left, right = (jax.device_put(v, cuda_device) for v in _inputs(np.float32))
        table = jax.device_put(layout, cuda_device)
        before = _native._stride_cuda_dot_prepared_stats()[0]
        with pytest.raises(Exception, match="conjugate_left must be zero or one"):
            _raw_operation(layout, conjugate).lower(left, right, table).compile()
        assert _native._stride_cuda_dot_prepared_stats()[0] == before


def test_compiled_dynamic_inputs_concurrent(cuda_device):
    with jax.enable_x64():
        records = _records(True)
        operation = _operation(_layout(records), True)
        a, b = (jax.device_put(v, cuda_device) for v in _inputs(np.complex64))
        before = _native._stride_cuda_dot_prepared_stats()[0]
        compiled = operation.lower(a, b).compile()
        assert _native._stride_cuda_dot_prepared_stats()[0] == before + 1

        def submit(i):
            left, right = _inputs(np.complex64, shift=i)
            output = compiled(jax.device_put(left, cuda_device), jax.device_put(right, cuda_device))
            return output, _expected(left, right, records, True)

        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(submit, range(12)))
        for result, expected in results:
            assert result.devices() == {cuda_device}
            np.testing.assert_allclose(result, expected, rtol=2e-6, atol=2e-6)
        assert _native._stride_cuda_dot_prepared_stats()[0] == before + 1


def test_distinct_layout_conjugation_and_four_dtypes_isolate(cuda_device):
    with jax.enable_x64():
        gc.collect()
        before = _native._stride_cuda_dot_prepared_stats()[0]
        unrelated = (_native._stride_native_prepared_stats(),
                     _native._stride_cuda_copy_prepared_stats(),
                     _native._stride_cuda_update_prepared_stats(),
                     _native._stride_cuda_accumulation_prepared_stats())
        retained = []
        for dtype in (np.float32, np.float64, np.complex64, np.complex128):
            for reverse, conjugate in ((False, False), (False, True), (True, True)):
                records = _records(reverse)
                left, right = _inputs(dtype)
                a, b = (jax.device_put(v, cuda_device) for v in (left, right))
                operation = _operation(_layout(records), conjugate)
                compiled = operation.lower(a, b).compile()
                retained.append((operation, compiled, a, b,
                                 _expected(left, right, records, conjugate)))
        assert _native._stride_cuda_dot_prepared_stats()[0] == before + len(retained)
        for _ in range(2):
            for _, compiled, a, b, expected in retained:
                np.testing.assert_allclose(compiled(a, b), expected, rtol=2e-6, atol=2e-6)
        # Complex conjugation and reversed source addresses must have distinct outputs.
        assert not np.array_equal(retained[6][-1], retained[7][-1])
        assert not np.array_equal(retained[7][-1], retained[8][-1])
        operation, compiled, *_ = retained.pop(0)
        del compiled
        operation.clear_cache()
        del operation
        gc.collect()
        for _, compiled, a, b, expected in retained:
            np.testing.assert_allclose(compiled(a, b), expected, rtol=2e-6, atol=2e-6)
        assert _native._stride_cuda_dot_prepared_stats()[0] == before + 12
        assert (_native._stride_native_prepared_stats(),
                _native._stride_cuda_copy_prepared_stats(),
                _native._stride_cuda_update_prepared_stats(),
                _native._stride_cuda_accumulation_prepared_stats()) == unrelated


@pytest.mark.parametrize("invalid", ["descriptor", "output", "left"])
def test_bad_dynamic_signature_does_not_poison_live_executable(cuda_device, invalid):
    # Different JAX signatures create different prepared states. Same-state
    # malformed dynamic buffers are checked by the native contract tests.
    with jax.enable_x64():
        records = _records()
        layout = _layout(records)
        left, right = _inputs(np.float32)
        a, b = (jax.device_put(v, cuda_device) for v in (left, right))
        table = jax.device_put(np.asarray(_native._stride_pack_owner_fiber("dot", layout.tolist()), np.int64), cuda_device)
        valid_op = _raw_operation(layout, 0)
        valid = valid_op.lower(a, b, table).compile()
        np.testing.assert_allclose(valid(a, b, table), _expected(left, right, records, False), rtol=2e-6, atol=2e-6)
        before = _native._stride_cuda_dot_prepared_stats()[0]
        if invalid == "descriptor":
            table = jax.device_put(np.zeros(3, np.int64), cuda_device)
            bad = valid_op.lower(a, b, table).compile()
            args, message = (a, b, table), "descriptor operand shape"
        elif invalid == "output":
            target = _registration.cuda_dot_target(np.dtype("float32"), np.dtype("float32"), np.dtype("float32"))
            call = jax.ffi.ffi_call(target, (jax.ShapeDtypeStruct((3,), np.float32),
                                             jax.ShapeDtypeStruct((2, 2), np.float32)))
            bad = jax.jit(lambda x, y, d: call(x, y, d, layout=layout, conjugate_left=np.int64(0))[0]).lower(a, b, table).compile()
            args, message = (a, b, table), "buffer dimensions"
        else:
            a = jax.device_put(np.zeros((2, 3), np.float32), cuda_device)
            bad = valid_op.lower(a, b, table).compile()
            args, message = (a, b, table), "buffer dimensions"
        with pytest.raises(Exception, match=message):
            bad(*args).block_until_ready()
        np.testing.assert_allclose(valid(
            jax.device_put(left, cuda_device), b, jax.device_put(np.asarray(
                _native._stride_pack_owner_fiber("dot", layout.tolist()), np.int64), cuda_device)),
            _expected(left, right, records, False), rtol=2e-6, atol=2e-6)
        assert _native._stride_cuda_dot_prepared_stats()[0] == before + 1


@pytest.mark.parametrize("destruction", ["local", "global", "exit", "compile_only", "lazy_registration"])
def test_isolated_queued_launch_and_destruction(cuda_device, destruction):
    script = ("from tests.stride.support.workers.cuda_dot_lifecycle import _check_lifecycle; "
              f"_check_lifecycle({destruction!r})")
    completed = subprocess.run([sys.executable, "-c", script], cwd=REPO_ROOT,
                               capture_output=True, text=True, timeout=180)
    assert completed.returncode == 0, completed.stdout + completed.stderr
