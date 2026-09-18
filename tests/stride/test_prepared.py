from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import gc
import importlib
from pathlib import Path
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensor0 import _native
from tensor0._stride import StridedView, dotu, enable_threads, get_num_threads, materialize, reduce_sum, scale, set_num_threads
from tensor0._stride._ffi import _registration
from tensor0._stride._ffi._descriptor import encode_layout
from tensor0._stride._jax import accumulation_p, copy_p
from tensor0._stride._layout import AffineRecord

from ._support import native_available


pytestmark = pytest.mark.skipif(not native_available(), reason="native CPU stride is unavailable")


@pytest.mark.parametrize("factor", [1., -.75])
@pytest.mark.parametrize("shape,batches,workers", [((8, 8, 8, 8), 3, 1), ((32, 16, 16, 16), 1, 4)])
def test_rank4_prepared_layout_batches_and_worker_limits(factor, shape, batches, workers):
    outer, rows, inner, columns = shape
    size = int(np.prod(shape))
    record = AffineRecord(shape, (inner * columns, outer * inner * columns, 1, inner), 0,
                          (rows * inner * columns, inner * columns, columns, 1), 0)
    source = jnp.arange(batches * size, dtype=jnp.float32).reshape(batches, size)
    operation = jax.jit(lambda value, coefficient: accumulation_p.bind(
        value, coefficient, records=(record,), coefficient_records=(0,), output_size=size, dtype=np.dtype(jnp.float32),
    ))
    previous = get_num_threads()
    try:
        set_num_threads(workers)
        actual = operation(source, jnp.float32(factor))
        actual.block_until_ready()
    finally:
        if previous is None:
            enable_threads()
        else:
            set_num_threads(previous)
    expected = np.asarray(source).reshape(batches, rows, outer, columns, inner).transpose(0, 2, 1, 4, 3)
    np.testing.assert_array_equal(actual, expected.reshape(batches, size) * factor)


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


def _check_lifecycle(operation, destruction):
    source = jnp.arange(12, dtype=jnp.float32)
    if operation == "copy":
        function = lambda value, factor: materialize(StridedView(value, (3, 4), (1, 3), 0))
        expected = lambda value, factor: np.asarray(value).reshape(4, 3).T
    elif operation == "update":
        function = lambda value, factor: scale(StridedView(value, (6,), (2,), 0), factor).data
        expected = lambda value, factor: np.where(np.arange(12) % 2 == 0, np.asarray(value) * factor, value)
    elif operation == "reduction":
        function = lambda value, factor: reduce_sum(StridedView(value, (3, 4), (4, 1), 0), (1,))
        expected = lambda value, factor: np.asarray(value).reshape(3, 4).sum(axis=1)
    elif operation == "dot":
        function = lambda value, factor: dotu(StridedView(value, (12,), (1,), 0), StridedView(value, (12,), (1,), 0))
        expected = lambda value, factor: np.sum(np.asarray(value) ** 2)
    else:
        mapping = lambda value: materialize(StridedView(value, (12,), (1,), 0), dtype=jnp.complex64)
        cotangent = jnp.full((12,), 1 + 2j, dtype=jnp.complex64)
        function = lambda value, factor: (mapping(value), jax.vjp(mapping, value)[1](cotangent)[0])
        expected = lambda value, factor: (np.asarray(value).astype(np.complex64), np.ones(12, dtype=np.float32))
    baseline = _native._stride_native_prepared_stats()
    compiled = jax.jit(function)

    def execute(offset, factor):
        actual = compiled(source + offset, jnp.float32(factor))
        jax.block_until_ready(actual)
        for result, wanted in zip(jax.tree.leaves(actual), jax.tree.leaves(expected(source + offset, factor)), strict=True):
            np.testing.assert_array_equal(result, wanted)

    execute(0, 2)
    states = 2 if operation == "mixed_ad" else 1
    created, destroyed = _native._stride_native_prepared_stats()
    assert created == baseline[0] + states
    assert destroyed == baseline[1]
    execute(3, -1)
    assert _native._stride_native_prepared_stats() == (created, destroyed)
    importlib.reload(_registration)
    _registration.operation_target("copy" if operation == "mixed_ad" else operation, np.dtype(jnp.float32))
    execute(5, .5)
    assert _native._stride_native_prepared_stats() == (created, destroyed)
    if destruction != "exit":
        if destruction == "local":
            compiled.clear_cache()
        else:
            jax.clear_caches()
        gc.collect()
        assert _native._stride_native_prepared_stats() == (created, destroyed + states)
        execute(7, 3)
        assert _native._stride_native_prepared_stats() == (created + states, destroyed + states)
        compiled.clear_cache()
        gc.collect()
        assert _native._stride_native_prepared_stats() == (created + states, destroyed + 2 * states)
    assert not any(name == "tensor0._stride1" or name.startswith("tensor0._stride1.") or
                   name == "tensor0.operations._strided" for name in sys.modules)


@pytest.mark.parametrize("operation", ["copy", "update", "reduction", "dot", "mixed_ad"])
@pytest.mark.parametrize("destruction", ["local", "global", "exit"])
def test_prepared_reuse_reload_destruction_and_process_exit(operation, destruction):
    script = ("from tests.stride.test_prepared import _check_lifecycle; "
              f"_check_lifecycle({operation!r}, {destruction!r})")
    completed = subprocess.run([sys.executable, "-c", script], cwd=Path(__file__).resolve().parents[2],
                               capture_output=True, text=True, timeout=180)
    assert completed.returncode == 0, completed.stdout + completed.stderr
