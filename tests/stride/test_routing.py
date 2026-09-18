"""Native routing boundaries exercised without legacy route-decision objects."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensor0 import _native
from tensor0._stride import _jax
from tensor0._stride._ffi import _registration
from tensor0._stride._ffi._calls import execute_copy
from tensor0._stride._ffi._descriptor import encode_layout
from tensor0._stride._jax import accumulation_p, copy_p
from tensor0._stride._layout import AffineRecord

from ._support import native_available


PARTITIONS = (AffineRecord((4, 2), (4, 1), 0, (4, 1), 2),
              AffineRecord((4, 2), (4, 1), 2, (4, 1), 0))


@pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")
@pytest.mark.parametrize("recipe", ["partitions", "compact", "many_tiny", "many_single"])
@pytest.mark.parametrize("scaled", [False, True])
def test_cpu_execution_ignores_recipe_size_and_record_count(recipe, scaled):
    if recipe == "partitions":
        records, size = PARTITIONS, 16
        order = np.arange(size).reshape(4, 4)[:, [2, 3, 0, 1]].ravel()
    elif recipe == "compact":
        size = 4096
        records, order = (AffineRecord((size,), (1,), 0, (1,), 0),), np.arange(size)
    else:
        record_size = 8 if recipe == "many_tiny" else 1
        size = 1024 * record_size
        records = tuple(AffineRecord((record_size,), (1,), offset, (1,), offset)
                        for offset in range(0, size, record_size))
        order = np.arange(size)
    source = jnp.arange(size, dtype=jnp.float32)

    def operation(value):
        if scaled:
            return accumulation_p.bind(value, *((jnp.float32(.75),) * len(records)), records=records,
                coefficient_records=tuple(range(len(records))), output_size=size, dtype=value.dtype)
        return copy_p.bind(value, records=records, output_size=size, dtype=value.dtype)

    lowered = jax.jit(operation).lower(source)
    text = lowered.as_text().lower()
    assert text.count("custom_call") == 1
    assert f"tensor0_stride_{'accumulation' if scaled else 'copy'}_f32_cpu_v1" in text
    assert "gather" not in text and "scatter" not in text and "iota" not in text
    expected = source[order] * (.75 if scaled else 1)
    np.testing.assert_array_equal(lowered.compile()(source), expected)


@pytest.mark.parametrize("failure,message", [
    ("missing", "CPU execution is unavailable"),
    ("unavailable", "CPU execution is unavailable"),
    ("versions", "different JAX versions"),
])
def test_unavailable_cpu_call_fails_without_fallback(failure, message, monkeypatch):
    monkeypatch.setattr(_registration, "_REGISTERED", set())
    if failure == "missing":
        monkeypatch.delattr(_native, "_stride_native_registration", raising=False)
    elif failure == "unavailable":
        monkeypatch.setattr(_native, "_stride_ffi_available", lambda: False)
    else:
        monkeypatch.setattr(_native, "_stride_ffi_available", lambda: True)
        monkeypatch.setattr(_native, "_stride_ffi_build_versions", lambda: ("invalid", "invalid"))
    layout = encode_layout((AffineRecord((4,), (1,), 0, (1,), 0),), source_size=4, output_size=4)
    with pytest.raises(RuntimeError, match=message):
        execute_copy(jnp.arange(4, dtype=jnp.float32), layout=layout, output_size=4)
    assert not _registration._REGISTERED


@pytest.mark.parametrize("primitive", [copy_p, accumulation_p])
def test_non_cpu_lowering_fails_without_fallback(primitive):
    record = AffineRecord((4,), (1,), 0, (1,), 0)
    function = lambda value: primitive.bind(value, records=(record,), output_size=4, dtype=value.dtype)
    traced = jax.jit(function).trace(jnp.zeros(4, dtype=jnp.float32))
    with pytest.raises(NotImplementedError, match=f"{primitive.name}.*not found for platform tpu"):
        traced.lower(lowering_platforms=("tpu",))


@pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")
@pytest.mark.parametrize("batch_count,shape,dtype", [
    (256, (4096,), jnp.float32), (8, (131072,), jnp.complex64),
    (2, (1025, 1024), jnp.complex64),
])
def test_large_direct_and_vmap_batches_use_one_native_call(batch_count, shape, dtype):
    size = int(np.prod(shape))
    strides = (1,) if len(shape) == 1 else (1, shape[0])
    destinations = (1,) if len(shape) == 1 else (shape[1], 1)
    record = AffineRecord(shape, strides, 0, destinations, 0)
    real = (jnp.arange(batch_count * size, dtype=jnp.float32).reshape(batch_count, size) % 127) / 32
    source = (real + 1j * (1 - real)).astype(dtype) if dtype == jnp.complex64 else real
    factor = jnp.asarray(.75 - .5j if dtype == jnp.complex64 else 1.25, dtype=dtype)
    operation = lambda value: accumulation_p.bind(value, factor, records=(record,), coefficient_records=(0,),
                                                  output_size=size, dtype=value.dtype)
    selected = source if len(shape) == 1 else source.reshape(batch_count, shape[1], shape[0]).transpose(0, 2, 1).reshape(batch_count, size)
    expected = selected * factor
    for function in (operation, jax.vmap(operation)):
        lowered = jax.jit(function).lower(source)
        text = lowered.as_text().lower()
        assert text.count("custom_call") == 1 and "tensor0_stride_accumulation_" in text
        assert "gather" not in text and "scatter" not in text and "iota" not in text
        np.testing.assert_allclose(lowered.compile()(source), expected, rtol=2e-6, atol=1e-6)


def test_unknown_sharding_is_rejected_at_storage_boundary():
    with pytest.raises(ValueError, match="requires NamedSharding"):
        _jax._batch_storage_sharding(SimpleNamespace(shape=(2, 16), sharding=None))


def _check_sharding(mode):
    from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec

    assert len(jax.devices()) == 2
    mesh = Mesh(np.asarray(jax.devices()), ("device",),
                axis_types=(AxisType.Explicit if mode == "explicit" else AxisType.Auto,))
    batches = NamedSharding(mesh, PartitionSpec("device", None))
    replicated = NamedSharding(mesh, PartitionSpec())
    partitioned = NamedSharding(mesh, PartitionSpec("device"))
    host = np.arange(64, dtype=np.float32).reshape(4, 16)
    order = np.arange(16).reshape(4, 4)[:, [2, 3, 0, 1]].ravel()
    for scaled in (False, True):
        def operation(value):
            if scaled:
                return accumulation_p.bind(value, jnp.float32(.75), jnp.float32(.75), records=PARTITIONS,
                    coefficient_records=(0, 1), output_size=16, dtype=value.dtype)
            return copy_p.bind(value, records=PARTITIONS, output_size=16, dtype=value.dtype)

        for sharding in (batches, replicated):
            source = jax.device_put(host, sharding)
            executable = jax.jit(operation, in_shardings=sharding, out_shardings=sharding).lower(source).compile()
            actual = executable(source)
            np.testing.assert_array_equal(actual, host[:, order] * (.75 if scaled else 1))
            assert actual.sharding.is_equivalent_to(sharding, 2)
            text = executable.as_text().lower()
            assert "tensor0_stride_" in text
            assert "all-gather" not in text and "all-reduce" not in text
        source = jax.device_put(host[0], partitioned)
        with pytest.raises(Exception, match="cannot shard the packed storage axis"):
            jax.jit(operation, in_shardings=partitioned, out_shardings=replicated).lower(source).compile()
    assert not any(name == "tensor0._stride1" or name.startswith("tensor0._stride1.") or
                   name == "tensor0.operations._strided" for name in sys.modules)


@pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")
@pytest.mark.parametrize("mode", ["auto", "explicit"])
def test_mapping_batch_replication_and_storage_sharding_boundaries(mode):
    environment = os.environ.copy()
    environment["JAX_PLATFORMS"] = "cpu"
    environment["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    script = f"from tests.stride.test_routing import _check_sharding; _check_sharding({mode!r})"
    completed = subprocess.run([sys.executable, "-c", script], cwd=Path(__file__).resolve().parents[2],
                               env=environment, capture_output=True, text=True, timeout=180)
    assert completed.returncode == 0, completed.stdout + completed.stderr
