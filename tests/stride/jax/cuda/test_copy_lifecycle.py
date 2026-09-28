"""CUDA Copy host-state validation, reuse and executable lifetime."""

import gc
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import jax
import numpy as np
import pytest

from tensor0 import _native
from tensor0._stride._ffi import _registration
from tensor0._stride._ffi._descriptor import encode_layout
from tensor0._stride._layout import AffineRecord
from tests.stride.support.paths import REPO_ROOT



def _layout(reverse=False):
    return encode_layout((AffineRecord((6,), (-1 if reverse else 1,),
                                      5 if reverse else 0, (1,), 0),),
                         source_size=6, output_size=6)


def _operation(layout, dtype=np.float32):
    dtype = np.dtype(dtype)
    target = _registration.cuda_copy_target(dtype, dtype)
    call = jax.ffi.ffi_call(target, jax.ShapeDtypeStruct((1, 6), dtype))
    # Keep the descriptor operand explicit, independent of the host attribute.
    return jax.jit(lambda source, table: call(source, table, layout=layout))


def test_cpu_only_cuda_copy_stats_are_none():
    if _native._stride_cuda_available():
        pytest.skip("requires a CPU-only extension")
    assert _native._stride_cuda_copy_prepared_stats() is None


@pytest.mark.parametrize("mutation,message", [
    ("version", "unsupported native layout version"),
    ("destination", "injective|overlap"),
])
def test_raw_descriptor_rejected_at_compile_before_state(cuda_device, mutation, message):
    layout = _layout()
    if mutation == "version":
        layout[0] = 99
    else:
        layout[-1] = 0
    with jax.enable_x64():
        source = jax.device_put(np.arange(6, dtype=np.float32).reshape(1, 6), cuda_device)
        table = jax.device_put(layout, cuda_device)
        operation = _operation(layout)
        created = _native._stride_cuda_copy_prepared_stats()[0]
        with pytest.raises(Exception, match=message):
            operation.lower(source, table).compile()
        assert _native._stride_cuda_copy_prepared_stats()[0] == created


def test_compiled_copy_repeated_and_concurrent_reuse(cuda_device):
    with jax.enable_x64():
        layout = _layout(True)
        table = jax.device_put(np.asarray(_native._stride_pack_owner_fiber("copy", layout.tolist()), np.int64), cuda_device)
        hosts = [np.arange(6, dtype=np.float32).reshape(1, 6) + i for i in range(12)]
        inputs = [jax.device_put(value, cuda_device) for value in hosts]
        operation = _operation(layout)
        before = _native._stride_cuda_copy_prepared_stats()[0]
        compiled = operation.lower(inputs[0], table).compile()
        assert _native._stride_cuda_copy_prepared_stats()[0] == before + 1
        for _ in range(3):
            np.testing.assert_array_equal(compiled(inputs[0], table), hosts[0][:, ::-1])
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda value: compiled(value, table), inputs))
        for result, host in zip(results, hosts, strict=True):
            assert result.devices() == {cuda_device}
            np.testing.assert_array_equal(result, host[:, ::-1])
        assert _native._stride_cuda_copy_prepared_stats()[0] == before + 1


def test_independent_layouts_executables_and_dtypes(cuda_device):
    with jax.enable_x64():
        before = _native._stride_cuda_copy_prepared_stats()[0]
        retained = []
        for dtype, reverse in [(np.float32, False), (np.float32, True),
                               (np.complex64, False), (np.complex64, True)]:
            host = np.arange(6).reshape(1, 6).astype(dtype)
            if dtype == np.complex64:
                host += 1j * (host + 2)
            source = jax.device_put(host, cuda_device)
            layout = _layout(reverse)
            table = jax.device_put(np.asarray(_native._stride_pack_owner_fiber("copy", layout.tolist()), np.int64), cuda_device)
            operation = _operation(layout, dtype)
            compiled = operation.lower(source, table).compile()
            retained.append((operation, compiled, source, table, host[:, ::-1] if reverse else host))
        assert _native._stride_cuda_copy_prepared_stats()[0] == before + 4
        for _ in range(3):
            results = [compiled(source, table) for _, compiled, source, table, _ in retained]
            for result, entry in zip(results, retained, strict=True):
                np.testing.assert_array_equal(result, entry[-1])
        assert _native._stride_cuda_copy_prepared_stats()[0] == before + 4
        removed = retained.pop(0)
        operation, compiled = removed[:2]
        del removed, compiled
        operation.clear_cache()
        del operation
        gc.collect()
        # Other tests may own collectible states, so only created is exact here.
        for _, compiled, source, table, expected in retained:
            np.testing.assert_array_equal(compiled(source, table), expected)
        assert _native._stride_cuda_copy_prepared_stats()[0] == before + 4


def test_rejected_descriptor_operand_does_not_poison_live_executable(cuda_device):
    # Executable operand shapes are fixed. A malformed shape needs its own
    # compilation; this checks isolation, not recovery within that executable.
    with jax.enable_x64():
        layout = _layout(True)
        source = jax.device_put(np.arange(6, dtype=np.float32).reshape(1, 6), cuda_device)
        table = jax.device_put(np.asarray(_native._stride_pack_owner_fiber("copy", layout.tolist()), np.int64), cuda_device)
        operation = _operation(layout)
        valid = operation.lower(source, table).compile()
        np.testing.assert_array_equal(valid(source, table), np.arange(6)[None, ::-1])
        bad_table = jax.device_put(np.zeros(3, np.int64), cuda_device)
        invalid = operation.lower(source, bad_table).compile()
        before = _native._stride_cuda_copy_prepared_stats()[0]
        with pytest.raises(Exception, match="descriptor operand shape"):
            invalid(source, bad_table).block_until_ready()
        np.testing.assert_array_equal(valid(source, table), np.arange(6)[None, ::-1])
        assert _native._stride_cuda_copy_prepared_stats()[0] == before


def test_fresh_registration_is_cuda_only_with_five_prepared_bundles(cuda_device):
    script = ("from tests.stride.support.workers.cuda_registration "
              "import check_fresh_registration; check_fresh_registration()")
    completed = subprocess.run([sys.executable, "-c", script], cwd=REPO_ROOT,
                               capture_output=True, text=True, timeout=180)
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize("destruction", ["local", "global", "exit", "compile_only", "lazy_registration"])
def test_isolated_queued_launch_and_destruction(cuda_device, destruction):
    script = ("from tests.stride.support.workers.cuda_copy_lifecycle import _check_lifecycle; "
              f"_check_lifecycle({destruction!r})")
    # Inherit PYTHONPATH verbatim: remote runs may select a candidate extension.
    completed = subprocess.run([sys.executable, "-c", script], cwd=REPO_ROOT,
                               capture_output=True, text=True, timeout=180)
    assert completed.returncode == 0, completed.stdout + completed.stderr
