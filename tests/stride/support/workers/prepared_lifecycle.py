"""Isolated prepared lifecycle checks."""

import gc
import importlib
import sys

import jax
import jax.numpy as jnp
import numpy as np

from tensor0 import _native
from tensor0._stride import StridedView, dotu, materialize, reduce_sum, scale
from tensor0._stride._ffi import _registration


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
