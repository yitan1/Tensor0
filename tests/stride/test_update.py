"""Assignment and accumulation through the shared native Update boundary."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensor0._stride import enable_threads, get_num_threads, set_num_threads
from tensor0._stride._ffi._calls import execute_update
from tensor0._stride._ffi._descriptor import encode_layout
from tensor0._stride._ffi._registration import operation_target
from tensor0._stride._jax import update_p
from tensor0._stride._layout import AffineRecord

from ._support import native_available


PARTIAL = (AffineRecord((16,), (2,), 1, (3,), 2),)
COMPLETE = (AffineRecord((4, 2), (4, 1), 0, (4, 1), 2),
            AffineRecord((4, 2), (4, 1), 2, (4, 1), 0))
LARGE_SIZE = 1_100_000


def reference_update(source, base, records, alpha, beta):
    result = base
    for record in records:
        source_indices = np.full(record.logical_shape, record.source_offset, dtype=np.int32)
        destination_indices = np.full(record.logical_shape, record.destination_offset, dtype=np.int32)
        for axis, extent in enumerate(record.logical_shape):
            shape = (1,) * axis + (extent,) + (1,) * (len(record.logical_shape) - axis - 1)
            coordinates = np.arange(extent, dtype=np.int32).reshape(shape)
            source_indices += coordinates * record.source_strides[axis]
            destination_indices += coordinates * record.destination_strides[axis]
        source_indices = source_indices.reshape(-1)
        destination_indices = destination_indices.reshape(-1)
        term = source[..., source_indices] if alpha == 1 else alpha * source[..., source_indices]
        if beta:
            term = term + base[..., destination_indices]
        if not jnp.iscomplexobj(base):
            term = jnp.real(term)
        result = result.at[..., destination_indices].set(term.astype(base.dtype))
    return result


@pytest.mark.parametrize("operand", ["source", "base"])
@pytest.mark.parametrize("shape,error", [((), "at least one dimension"), ((2, 4), "batch shapes")])
def test_update_storage_shape_validation(operand, shape, error):
    arguments = [jnp.ones(4), jnp.ones(4), jnp.float32(1), jnp.float32(0)]
    arguments[0 if operand == "source" else 1] = jnp.ones(shape)
    layout = encode_layout((AffineRecord((4,), (1,), 0, (1,), 0),), source_size=4, output_size=4)
    for operation in (execute_update, jax.jit(lambda *values: execute_update(*values, layout=layout))):
        with pytest.raises(ValueError, match=error):
            if operation is execute_update:
                operation(*arguments, layout=layout)
            else:
                operation(*arguments)


@pytest.mark.parametrize("operand", range(4))
def test_update_rejects_non_array_operands(operand):
    arguments = [jnp.ones(4), jnp.ones(4), jnp.float32(1), jnp.float32(0)]
    arguments[operand] = np.asarray(arguments[operand])
    layout = encode_layout((), source_size=4, output_size=4)
    with pytest.raises(TypeError, match="JAX Array or Tracer"):
        execute_update(*arguments, layout=layout)


@pytest.mark.parametrize("operand", ["source", "base"])
def test_update_rejects_out_of_bounds_layout(operand):
    source = jnp.ones(3 if operand == "source" else 4)
    base = jnp.ones(3 if operand == "base" else 4)
    records = (AffineRecord((4,), (1,), 0, (1,), 0),)
    with pytest.raises(Exception, match="address exceeds storage"):
        update_p.bind(source, base, jnp.float32(1), jnp.float32(0), records=records).block_until_ready()
    np.testing.assert_array_equal(base, np.ones(base.shape))


def test_update_rejects_unsupported_storage_pair():
    with pytest.raises(Exception, match="unsupported update source/storage dtype pair"):
        update_p.bind(jnp.ones(4, jnp.complex64), jnp.ones(4, jnp.int32),
                      jnp.int32(1), jnp.int32(0), records=()).block_until_ready()


def test_update_abstract_result_preserves_base_shape_and_dtype():
    base = jax.ShapeDtypeStruct((2, 4), jnp.float32)
    source = jax.ShapeDtypeStruct((2, 6), jnp.float16)
    result = jax.eval_shape(lambda new, old: update_p.bind(
        new, old, jnp.float32(1), jnp.float32(0), records=()), source, base)
    assert result.shape == base.shape and result.dtype == base.dtype


@pytest.mark.parametrize("dtype,alpha", [(jnp.float16, -1.25), (jnp.bfloat16, -1.25),
                                        (jnp.float32, -1.25), (jnp.complex64, 1.25-0.75j),
                                        (jnp.int32, -3)])
@pytest.mark.parametrize("beta", [0, 1])
def test_same_dtype_assignment_and_accumulation(dtype, alpha, beta):
    records = (AffineRecord((17,), (1,), 0, (1,), 0),)
    source, base = jnp.arange(17, dtype=dtype) - 3, jnp.arange(17, dtype=dtype) + 5
    coefficient = jnp.asarray(alpha, dtype=dtype)
    operation = jax.jit(lambda old, new: update_p.bind(new, old, coefficient, jnp.int32(beta), records=records))
    actual = operation(base, source)
    expected = reference_update(source, base, records, coefficient, beta)
    np.testing.assert_allclose(actual, expected, rtol=5e-3 if dtype in (jnp.float16, jnp.bfloat16) else 2e-6,
                               atol=1e-6)
    lowered = operation.lower(base, source).as_text()
    assert lowered.count("stablehlo.custom_call") == 1
    assert operation_target("update", base.dtype) in lowered


@pytest.mark.parametrize("source_dtype,result_dtype", [(jnp.float16, jnp.float32),
                                                      (jnp.float32, jnp.complex64),
                                                      (jnp.complex64, jnp.float32)])
@pytest.mark.parametrize("beta", [0, 1])
def test_partial_mixed_dtype_updates(source_dtype, result_dtype, beta):
    source = jnp.linspace(-2, 3, 32).astype(source_dtype)
    base = jnp.linspace(3, -2, 50).astype(result_dtype)
    if jnp.iscomplexobj(source):
        source = source + 1j * source[::-1]
    if jnp.iscomplexobj(base):
        base = base + 1j * base[::-1]
    actual = jax.jit(lambda old, new: update_p.bind(
        new, old, jnp.float32(.5), jnp.int32(beta), records=PARTIAL))(base, source)
    expected = reference_update(source, base, PARTIAL, .5, beta)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-6)


@pytest.mark.parametrize("beta", [0, 1])
def test_complex_finite_updates(beta):
    source = jnp.asarray([0, complex(-0., 0.), 1.25+2j, -1.25-3j, .125+1j,
                          1e10-1e10j, 1.25-.75j, -2.5+4j], dtype=jnp.complex64)
    base = source[::-1]
    records = (AffineRecord((8,), (1,), 0, (1,), 0),)
    alpha = jnp.complex64(1.25-.75j)
    actual = jax.jit(lambda old, new: update_p.bind(new, old, alpha, jnp.int32(beta), records=records))(base, source)
    np.testing.assert_allclose(actual, reference_update(source, base, records, alpha, beta), rtol=2e-6, atol=1e-6)


@pytest.mark.parametrize("beta", [0, 1])
def test_partial_update_preserves_unselected_storage_and_inputs(beta):
    base, source = jnp.arange(50, dtype=jnp.float32) + 100, jnp.arange(32, dtype=jnp.float32)
    actual = update_p.bind(source, base, jnp.float32(.5), jnp.int32(beta), records=PARTIAL)
    np.testing.assert_array_equal(actual, reference_update(source, base, PARTIAL, .5, beta))
    untouched = np.ones(50, dtype=bool)
    untouched[2:50:3] = False
    np.testing.assert_array_equal(np.asarray(actual)[untouched].view(np.uint8), np.asarray(base)[untouched].view(np.uint8))
    np.testing.assert_array_equal(base, np.arange(50) + 100)
    np.testing.assert_array_equal(source, np.arange(32))


def test_complete_assignment_ignores_original_base():
    source = jnp.arange(16, dtype=jnp.float32)
    outputs = [update_p.bind(source, jnp.full(16, value, jnp.float32), jnp.float32(.5), jnp.int32(0),
                             records=COMPLETE) for value in (-7, 99)]
    np.testing.assert_array_equal(outputs[0], outputs[1])
    np.testing.assert_array_equal(outputs[0], reference_update(source, jnp.zeros(16), COMPLETE, .5, 0))


@pytest.mark.parametrize("beta", [0, 1])
@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("transpose_layout", [False, True])
def test_update_jvp_vjp_and_linear_transpose(beta, mixed, transpose_layout):
    records = (AffineRecord((17, 13), (1, 17), 0, (13, 1), 0),) if transpose_layout else PARTIAL
    source_size, output_size = (221, 221) if transpose_layout else (32, 50)
    source = jnp.linspace(-2, 2, source_size)
    base = jnp.linspace(-3, 4, output_size)
    if mixed:
        base = base + 1j * base[::-1]
    directions = (jnp.ones_like(base) * .5, jnp.linspace(-1, 1, source_size))
    cotangent = jnp.linspace(-4, 3, output_size).astype(base.dtype)
    if mixed:
        cotangent = cotangent + 1j * cotangent[::-1]
    operation = lambda old, new: update_p.bind(new, old, jnp.float32(.5), jnp.int32(beta), records=records)
    reference = lambda old, new: reference_update(new, old, records, .5, beta)
    actual_jvp = jax.jit(lambda old, new: jax.jvp(operation, (old, new), directions))(base, source)
    expected_jvp = jax.jvp(reference, (base, source), directions)
    for actual, expected in zip(actual_jvp, expected_jvp, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-6)
    actual_vjp = jax.jit(jax.vjp(operation, base, source)[1])(cotangent)
    expected_vjp = jax.vjp(reference, base, source)[1](cotangent)
    actual_transpose = jax.jit(jax.linear_transpose(operation, base, source))(cotangent)
    for actual, expected, transposed in zip(actual_vjp, expected_vjp, actual_transpose, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-6)
        np.testing.assert_allclose(transposed, expected, rtol=2e-6, atol=1e-6)
    if beta:
        np.testing.assert_array_equal(actual_vjp[0], cotangent)
    else:
        selected = actual_vjp[0] if transpose_layout else actual_vjp[0][2:50:3]
        np.testing.assert_array_equal(selected, jnp.zeros_like(selected))


@pytest.mark.parametrize("beta", [0, 1])
@pytest.mark.parametrize("records,source_size,output_size", [(PARTIAL, 32, 50), (COMPLETE, 16, 16)])
@pytest.mark.parametrize("in_axes", [(0, 0), (0, None), (None, 0)])
def test_update_vmap_operand_combinations(beta, records, source_size, output_size, in_axes):
    base, source = jnp.arange(output_size, dtype=jnp.float32), jnp.arange(source_size, dtype=jnp.float32)
    operands = (jnp.stack((base, 2 * base, -base)) if in_axes[0] == 0 else base,
                jnp.stack((source, 3 * source, -source)) if in_axes[1] == 0 else source)
    operation = lambda old, new: update_p.bind(new, old, jnp.float32(.5), jnp.int32(beta), records=records)
    reference = lambda old, new: reference_update(new, old, records, .5, beta)
    np.testing.assert_allclose(jax.jit(jax.vmap(operation, in_axes=in_axes))(*operands),
                               jax.vmap(reference, in_axes=in_axes)(*operands), rtol=2e-6, atol=1e-6)


@pytest.mark.parametrize("beta", [0, 1])
@pytest.mark.parametrize("strided", [False, True])
def test_large_update_lowering_has_no_materialized_addresses(beta, strided):
    stride = 2 if strided else 1
    size = LARGE_SIZE * stride
    records = (AffineRecord((LARGE_SIZE,), (stride,), 0, (stride,), 0),)
    argument = jax.ShapeDtypeStruct((size,), jnp.float32)
    operation = jax.jit(lambda old, new: update_p.bind(new, old, jnp.float32(1), jnp.int32(beta), records=records))
    lowered = operation.lower(argument, argument).as_text()
    assert "tensor0_stride_update_f32_cpu_v1" in lowered
    assert lowered.count("stablehlo.custom_call") == 1
    assert "stablehlo.gather" not in lowered and "stablehlo.scatter" not in lowered


@pytest.mark.parametrize("beta", [0, 1])
@pytest.mark.parametrize("layout_kind", ["transpose", "negative", "broadcast"])
@pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")
def test_large_layout_updates_with_thread_limits(beta, layout_kind):
    rows, columns = 512, 512
    size = rows * columns
    if layout_kind == "transpose":
        record = AffineRecord((rows, columns), (1, rows), 0, (columns, 1), 0)
        source = jnp.resize(jnp.array([0., -0., jnp.inf, -jnp.inf, jnp.nan, 1.5, -2.]), (size,))
    elif layout_kind == "negative":
        record = AffineRecord((rows, columns), (-columns, 1), (rows - 1) * columns, (1, rows), 0)
        source = (jnp.arange(size, dtype=jnp.float32) % 17) * .125
    else:
        record = AffineRecord((rows, columns), (1, 0), 0, (columns, 1), 0)
        source = jnp.linspace(-2, 3, rows)
    base = jnp.linspace(-3, 2, size)
    operation = jax.jit(lambda old, new: update_p.bind(new, old, jnp.float32(1.25), jnp.int32(beta), records=(record,)))
    expected = reference_update(source, base, (record,), 1.25, beta)
    original = get_num_threads()
    try:
        for workers in (1, 4):
            set_num_threads(workers)
            np.testing.assert_allclose(operation(base, source), expected, rtol=2e-6, atol=1e-6)
    finally:
        enable_threads() if original is None else set_num_threads(original)


def test_update_non_cpu_lowering_fails_closed():
    argument = jax.ShapeDtypeStruct((16,), jnp.float32)
    traced = jax.jit(lambda old, new: update_p.bind(
        new, old, jnp.float32(1), jnp.int32(1), records=COMPLETE)).trace(argument, argument)
    with pytest.raises((RuntimeError, NotImplementedError, ValueError), match="tpu|TPU"):
        traced.lower(lowering_platforms=("tpu",))


@pytest.mark.parametrize("beta", [0, 1])
@pytest.mark.parametrize("donate", [False, True])
def test_update_base_reuse_respects_functional_input_protection(beta, donate):
    source, base = jnp.arange(32, dtype=jnp.float32), jnp.arange(50, dtype=jnp.float32)
    expected = reference_update(source, base, PARTIAL, .5, beta)
    operation = jax.jit(lambda old, new: update_p.bind(new, old, jnp.float32(.5), jnp.int32(beta), records=PARTIAL),
                        donate_argnums=(0,) if donate else ())
    lowered = operation.lower(base, source)
    text = lowered.as_text()
    assert "output_operand_aliases" in text and "operand_index = 1" in text
    assert "tensor0_stride_update_f32_cpu_v1" in text
    executable = lowered.compile()
    memory = executable.memory_analysis()
    assert memory is not None
    assert memory.alias_size_in_bytes == (base.nbytes if donate else 0)
    actual = executable(base, source)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(source, np.arange(32))
    if donate:
        assert base.is_deleted()
    else:
        np.testing.assert_array_equal(base, np.arange(50))


def _check_sharding(mode):
    from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec

    mesh = Mesh(np.asarray(jax.devices()), ("device",),
                axis_types=(AxisType.Explicit if mode == "explicit" else AxisType.Auto,))
    assert len(jax.devices()) == 2
    batches = NamedSharding(mesh, PartitionSpec("device", None))
    base_host = np.arange(100, dtype=np.float32).reshape(2, 50)
    source_host = np.arange(64, dtype=np.float32).reshape(2, 32)
    base, source = jax.device_put(base_host, batches), jax.device_put(source_host, batches)
    for beta in (0, 1):
        operation = lambda old, new: update_p.bind(new, old, jnp.float32(.5), jnp.int32(beta), records=PARTIAL)
        executable = jax.jit(operation, in_shardings=(batches, batches), out_shardings=batches).lower(base, source).compile()
        actual = executable(base, source)
        expected = base_host.copy()
        expected[:, 2:50:3] = .5 * source_host[:, 1:32:2] + beta * base_host[:, 2:50:3]
        np.testing.assert_array_equal(actual, expected)
        assert actual.sharding.is_equivalent_to(batches, 2)
        text = executable.as_text().lower()
        assert "tensor0_stride_update_f32_cpu_v1" in text
        assert "all-gather" not in text and "all-reduce" not in text
    for operand in ("source", "base", "output"):
        partitioned = NamedSharding(mesh, PartitionSpec("device"))
        replicated = NamedSharding(mesh, PartitionSpec())
        shardings = (partitioned if operand == "base" else replicated,
                     partitioned if operand == "source" else replicated)
        output = partitioned if operand == "output" else replicated
        arguments = tuple(jax.device_put(value, sharding) for value, sharding in
                          zip((base_host[0], source_host[0]), shardings, strict=True))
        compiled = jax.jit(operation, in_shardings=shardings, out_shardings=output)
        if operand == "output" and mode == "explicit":
            executable = compiled.lower(*arguments).compile()
            actual = executable(*arguments)
            expected = base_host[0].copy()
            expected[2:50:3] += .5 * source_host[0, 1:32:2]
            np.testing.assert_array_equal(actual, expected)
            assert actual.sharding.is_equivalent_to(output, 1)
            calls = [line for line in executable.as_text().splitlines()
                     if 'custom_call_target="tensor0_stride_update_f32_cpu_v1"' in line]
            assert len(calls) == 1 and "f32[1,50]" in calls[0]
        else:
            with pytest.raises(Exception, match="cannot shard the packed storage axis"):
                compiled.lower(*arguments).compile()
    assert not any(name == "tensor0._stride1" or name.startswith("tensor0._stride1.") or
                   name == "tensor0.operations._strided" for name in sys.modules)


@pytest.mark.parametrize("mode", ["auto", "explicit"])
def test_update_batch_sharding_and_storage_boundary(mode):
    environment = os.environ.copy()
    environment["JAX_PLATFORMS"] = "cpu"
    environment["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    script = f"from tests.stride.test_update import _check_sharding; _check_sharding({mode!r})"
    completed = subprocess.run([sys.executable, "-c", script], cwd=Path(__file__).resolve().parents[2],
                               env=environment, capture_output=True, text=True, timeout=180)
    assert completed.returncode == 0, completed.stdout + completed.stderr
