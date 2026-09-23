"""Prepared descriptor validation, reuse and isolated lifecycles."""

import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensor0 import _native
from tensor0._stride._ffi import _registration
from tensor0._stride._ffi._descriptor import encode_layout
from tensor0._stride._jax import copy_p
from tensor0._stride._layout import AffineRecord

from tests.stride.support.availability import native_available
from tests.stride.support.paths import REPO_ROOT


pytestmark = pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')


@pytest.mark.parametrize("mutation,message", [
    ("version", "unsupported native layout version"),
    ("truncated", "layout rank exceeds descriptor length"),
    ("extended", "layout descriptor has trailing words"),
    ("negative_shape", "layout size or offset is negative"),
    ("source_stride", "source"),
    ("destination_stride", "injective|overlap"),
    ("destination_offset", "destination"),
])
def test_prepared_compile_validates_address_descriptor_before_creating_state(mutation, message):
    layout = encode_layout((AffineRecord((2, 3), (3, 1), 0, (3, 1), 0),), source_size=6, output_size=6)
    if mutation == "version":
        layout[0] = 99
    elif mutation == "truncated":
        layout = layout[:-1]
    elif mutation == "extended":
        layout = np.append(layout, np.int64(0))
    elif mutation == "negative_shape":
        layout[7] = -1
    elif mutation == "source_stride":
        layout[9] = 100
    elif mutation == "destination_stride":
        layout[11] = 0
    else:
        layout[6] = 6
    call = jax.ffi.ffi_call(_registration.operation_target("copy", np.dtype(jnp.float32)),
                            jax.ShapeDtypeStruct((1, 6), jnp.float32))
    source = jnp.arange(6, dtype=jnp.float32).reshape(1, 6)
    created = _native._stride_native_prepared_stats()[0]
    with pytest.raises(Exception, match=message):
        jax.jit(lambda value: call(value, layout=layout)).lower(source).compile()
    assert _native._stride_native_prepared_stats()[0] == created


@pytest.mark.parametrize("invalid,message", [
    ("rank", "copy storage must be rank-two"),
    ("dtype", "dtype|element type"),
    ("alias", "input and output buffers overlap"),
])
def test_prepared_copy_enforces_buffer_contract(invalid, message):
    layout = encode_layout((AffineRecord((6,), (1,), 0, (1,), 0),), source_size=6, output_size=6)
    source = jnp.arange(6, dtype=jnp.float32)
    if invalid != "rank":
        source = source.reshape(1, 6)
    result_dtype = jnp.complex64 if invalid == "dtype" else jnp.float32
    call = jax.ffi.ffi_call(_registration.operation_target("copy", np.dtype(jnp.float32)),
                            jax.ShapeDtypeStruct((1, 6), result_dtype),
                            input_output_aliases={0: 0} if invalid == "alias" else {})
    with pytest.raises(Exception, match=message):
        call(source, layout=layout).block_until_ready()
    np.testing.assert_array_equal(source.ravel(), np.arange(6))


def test_many_record_prepared_executable_is_reused_concurrently():
    records = tuple(AffineRecord((8,), (1,), index * 8, (1,), index * 8) for index in range(1024))
    source = jnp.arange(8192, dtype=jnp.float32)
    operation = jax.jit(lambda value: copy_p.bind(value, records=records, output_size=8192, dtype=np.dtype(jnp.float32)))
    before = _native._stride_native_prepared_stats()[0]
    compiled = operation.lower(source).compile()
    np.testing.assert_array_equal(compiled(source), source)
    created = _native._stride_native_prepared_stats()[0]
    assert created == before + 1
    inputs = [source + offset for offset in range(4)]
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(compiled, inputs))
    for actual, expected in zip(results, inputs, strict=True):
        np.testing.assert_array_equal(actual, expected)
    assert _native._stride_native_prepared_stats()[0] == created


@pytest.mark.parametrize("operation", ["copy", "update", "reduction", "dot", "mixed_ad"])
@pytest.mark.parametrize("destruction", ["local", "global", "exit"])
def test_prepared_reuse_reload_destruction_and_process_exit(operation, destruction):
    script = ("from tests.stride.support.workers.prepared_lifecycle import _check_lifecycle; "
              f"_check_lifecycle({operation!r}, {destruction!r})")
    completed = subprocess.run([sys.executable, "-c", script], cwd=REPO_ROOT,
                               capture_output=True, text=True, timeout=180)
    assert completed.returncode == 0, completed.stdout + completed.stderr
