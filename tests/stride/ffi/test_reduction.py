"""Raw Reduction buffer, dtype and coefficient protocol contracts."""

from math import prod

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensor0 import _native
from tensor0._stride._ffi._calls import execute_reduction
from tensor0._stride._ffi._registration import operation_target

from tests.stride.support.availability import native_available
from tests.stride.support.data import REDUCTION_DTYPES
from tests.stride.support.ffi import REDUCTION_FIBER, _encode_reduction_records, _raw_reduction_call


pytestmark = pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')


@pytest.mark.parametrize("invalid,message", [("version", "unsupported native reduction layout version"),
    ("axes", "axis flag is not boolean"), ("output_shape", "shapes do not match axis roles"),
    ("dtype", "dtype|element type")])
def test_reduction_ffi_rejects_invalid_protocol_or_typed_output(invalid, message):
    layout = _encode_reduction_records((dict(source_shape=(2, 3), source_strides=(3, 1), source_offset=0, output_shape=(2, 1), output_strides=(1, 1), output_offset=0, reduction_axes=(False, True)),), source_size=6, output_size=2)
    words = layout.view("<u8")
    if invalid == "version":
        words[0] = 99
    elif invalid == "axes":
        words[-1] = 2
    elif invalid == "output_shape":
        words[12] = 3
    dtype = jnp.complex64 if invalid == "dtype" else jnp.float32
    call = jax.ffi.ffi_call(operation_target("reduction", np.dtype(jnp.float32)), jax.ShapeDtypeStruct((1, 2), dtype))
    with pytest.raises(Exception, match=message):
        call(jnp.ones((1, 6), dtype=jnp.float32), layout=layout,
             coefficient_records=np.asarray([], dtype=np.int64)).block_until_ready()


@pytest.mark.parametrize("dtype", REDUCTION_DTYPES)
@pytest.mark.parametrize("batch_shape", [(), (3,), (2, 3), (0,), (2, 0)])
@pytest.mark.parametrize("shared", ["scalar", "singleton", "batch"])
def test_reduction_batches_coefficient_shapes_and_types(dtype, batch_shape, shared):
    with jax.enable_x64():
        source = jnp.broadcast_to(jnp.asarray([1, 2, 3], dtype=jnp.float32), (*batch_shape, 3))
        factors = jnp.asarray(np.arange(prod(batch_shape)).reshape(batch_shape) % 3, dtype=dtype)
        if shared != "batch":
            factors = jnp.asarray(1 if shared == "scalar" else [2], dtype=dtype)
        layout = _encode_reduction_records((REDUCTION_FIBER,), source_size=3, output_size=1)
        execute = jax.jit(lambda data, coefficient: execute_reduction(
            data, (coefficient,), coefficient_records=(0,), layout=layout, output_size=1))
        result = execute(source, factors)
        expected_factor = np.asarray(factors).reshape(()) if shared != "batch" else np.asarray(factors)
        expected = np.broadcast_to(6 * expected_factor, batch_shape)[..., None]
        np.testing.assert_array_equal(result, expected)
        text = execute.lower(source, factors).as_text()
        assert text.count("stablehlo.custom_call @tensor0_stride_reduction_f32_cpu_v1") == 1


def test_reduction_batches_invalid_coefficient_batch_shapes():
    layout = _encode_reduction_records((dict(REDUCTION_FIBER, source_shape=(1,)),), source_size=1, output_size=1)
    source = jnp.ones((3, 1))
    for coefficient in (jnp.ones(2), jnp.ones((3, 1)), jnp.ones(0)):
        with pytest.raises(Exception, match="batch count"):
            _raw_reduction_call(source, layout, shape=(3, 1), coefficients=(coefficient,), indices=(0,)).block_until_ready()


# Keep every unscaled conversion; typed factors use diagonal and float32 anchors.
@pytest.mark.parametrize("source_dtype,result_dtype,coefficient_dtype", [
    pytest.param(source, result, coefficient, id=f"{coefficient}-{result}-{source}")
    for coefficient in (None, *REDUCTION_DTYPES)
    for result in REDUCTION_DTYPES for source in REDUCTION_DTYPES
    if coefficient is None or source == result or source == "float32" or result == "float32"
])
def test_dtype_combinations(source_dtype, result_dtype, coefficient_dtype):
    with jax.enable_x64():
        values = np.asarray([1, 2, 3])
        if source_dtype.startswith("complex"):
            values = values + 1j * values
        source = jnp.asarray(values, dtype=source_dtype)
        coefficients = () if coefficient_dtype is None else (jnp.asarray(2, dtype=coefficient_dtype),)
        mapped = source if not coefficients else source * coefficients[0]
        expected = jnp.sum(mapped)
        if jnp.iscomplexobj(expected) and result_dtype not in ("bool", "complex64", "complex128"):
            expected = expected.real
        layout = _encode_reduction_records((REDUCTION_FIBER,), source_size=3, output_size=1)
        result = execute_reduction(source, coefficients, coefficient_records=() if not coefficients else (0,),
                                   layout=layout, output_size=1, dtype=result_dtype)
        assert result.dtype == jnp.dtype(result_dtype)
        np.testing.assert_array_equal(result, np.asarray(expected.astype(result_dtype)).reshape(1))


@pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3), (0,), (2, 0, 3)])
def test_real_outputs_heterogeneous_coefficients_and_batches(batch_shape):
    records = (
        dict(source_shape=(2, 2), source_strides=(2, 1), source_offset=0, output_shape=(2, 1), output_strides=(2, 2**63 - 1), output_offset=1, reduction_axes=(False, True)),
        dict(source_shape=(2,), source_strides=(-1,), source_offset=5, output_shape=(2,), output_strides=(-2,), output_offset=3, reduction_axes=(False,)),
        dict(source_shape=(), source_strides=(), source_offset=6, output_shape=(), output_strides=(), output_offset=1, reduction_axes=()),
        dict(source_shape=(0,), source_strides=(1,), source_offset=7, output_shape=(1,), output_strides=(2**63 - 1,), output_offset=4, reduction_axes=(True,)),
    )
    layout = _encode_reduction_records(records, source_size=7, output_size=5)
    multiplier = np.arange(1, prod(batch_shape) + 1).reshape((*batch_shape, 1))
    source = jnp.asarray(multiplier * np.arange(1, 8), dtype=jnp.int32)
    execute = jax.jit(lambda data, half, integer: execute_reduction(
        data, (half, integer, jnp.float32(np.nan)), coefficient_records=(1, 2, 3),
        layout=layout, output_size=5, dtype="float32"))
    for half in (.5, 1.5):
        expected = multiplier * [0, 3 + 5 * half + 14, 0, 7 + 6 * half, 0]
        np.testing.assert_array_equal(execute(source, jnp.float32(half), jnp.int32(2)), expected)


def test_python_argument_contracts_and_raw_ffi_ad_rejection():
    layout = _encode_reduction_records((REDUCTION_FIBER,), source_size=3, output_size=1)
    source = jnp.ones(3)
    with pytest.raises(TypeError, match="JAX Array"):
        execute_reduction(np.ones(3), layout=layout, output_size=1)
    with pytest.raises(ValueError, match="dimension"):
        execute_reduction(jnp.float32(1), layout=layout, output_size=1)
    with pytest.raises(TypeError, match="explicit dtypes"):
        execute_reduction(source, (2,), coefficient_records=(0,), layout=layout, output_size=1)
    with pytest.raises(ValueError, match="batch shape"):
        execute_reduction(source, (jnp.ones(2),), coefficient_records=(0,), layout=layout, output_size=1)
    with pytest.raises(ValueError, match="nonnegative int64"):
        execute_reduction(source, coefficient_records=(True,), layout=layout, output_size=1)
    with pytest.raises(NotImplementedError, match="does not support"):
        execute_reduction(source, layout=layout, output_size=1, dtype=jnp.float8_e4m3fn)
    with pytest.raises(ValueError, match="cannot be differentiated"):
        jax.grad(lambda data: execute_reduction(data, layout=layout, output_size=1).sum())(source)


def test_outer_vmap_and_prepared_reuse():
    layout = _encode_reduction_records((REDUCTION_FIBER,), source_size=3, output_size=1)
    execute = jax.jit(lambda source, factor: execute_reduction(
        source, (factor,), coefficient_records=(0,), layout=layout, output_size=1))
    source = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)
    factors = jnp.asarray([2, 3], dtype=jnp.float32)
    np.testing.assert_array_equal(jax.vmap(execute)(source, factors), [[6], [36]])
    compiled = execute.lower(source, jnp.float32(2)).compile()
    created = _native._stride_native_prepared_stats()[0]
    for factor in (2, 3, 4):
        np.testing.assert_array_equal(compiled(source, jnp.float32(factor)), [[3 * factor], [12 * factor]])
    assert _native._stride_native_prepared_stats()[0] == created
    assert "tensor0_stride_reduction_f32_cpu_v1" in execute.lower(source, jnp.float32(2)).as_text()


@pytest.mark.parametrize("indices,count", [((0,), 0), ((), 1), ((-1,), 1), ((1,), 1), ((0, 0), 2)])
def test_invalid_coefficient_association(indices, count):
    layout = _encode_reduction_records((REDUCTION_FIBER,), source_size=3, output_size=1)
    with pytest.raises(Exception, match="coefficient.*(count|indices)"):
        _raw_reduction_call(jnp.ones((1, 3)), layout, coefficients=(jnp.float32(2),) * count, indices=indices).block_until_ready()


def test_invalid_buffers_and_aliases():
    layout = _encode_reduction_records((REDUCTION_FIBER,), source_size=3, output_size=1)
    for source, shape in ((jnp.ones(3), (1, 1)), (jnp.ones((1, 4)), (1, 1)),
                          (jnp.ones((1, 3)), (2, 1)), (jnp.ones((1, 3)), (1, 2))):
        with pytest.raises(Exception, match="rank-two|dimensions"):
            _raw_reduction_call(source, layout, shape=shape).block_until_ready()
    for coefficient in (jnp.ones(2), jnp.ones((1, 1)), jnp.float8_e4m3fn(1)):
        with pytest.raises(Exception, match="batch count|unsupported scalar dtype"):
            _raw_reduction_call(jnp.ones((1, 3)), layout, coefficients=(coefficient,), indices=(0,)).block_until_ready()
    with pytest.raises(Exception, match="unsupported scalar dtype"):
        _raw_reduction_call(jnp.ones((1, 3), dtype=jnp.float8_e4m3fn), layout).block_until_ready()
    identity = dict(REDUCTION_FIBER, output_shape=(3,), reduction_axes=(False,))
    layout = _encode_reduction_records((identity,), source_size=3, output_size=3)
    with pytest.raises(Exception, match="overlap"):
        _raw_reduction_call(jnp.ones((1, 3)), layout, shape=(1, 3), aliases={0: 0}).block_until_ready()


def test_layout_bytes_and_native_validation():
    layout = _encode_reduction_records((REDUCTION_FIBER,), source_size=3, output_size=1)
    np.testing.assert_array_equal(layout.view("<u8"), [1, 3, 1, 1, 1, 0, 0, 3, 1, 1, 1, 1])
    for invalid in (layout[:-1], np.concatenate((layout, np.zeros(8, dtype=np.uint8)))):
        with pytest.raises(Exception, match="partial word|trailing words"):
            execute_reduction(jnp.ones(3), layout=invalid, output_size=1).block_until_ready()
    for changes in ({"output_shape": (2,)}, {"source_strides": (2,)}, {"output_offset": 1}):
        invalid = _encode_reduction_records((dict(REDUCTION_FIBER, **changes),), source_size=3, output_size=1)
        with pytest.raises(Exception, match="shape|address|storage"):
            execute_reduction(jnp.ones(3), layout=invalid, output_size=1).block_until_ready()
    overlapping = dict(source_shape=(2, 2), source_strides=(2, 1), source_offset=0, output_shape=(2, 2), output_strides=(1, 1), output_offset=0, reduction_axes=(False, False))
    invalid = _encode_reduction_records((overlapping,), source_size=4, output_size=3)
    with pytest.raises(Exception, match="injective|overlap"):
        execute_reduction(jnp.ones(4), layout=invalid, output_size=3).block_until_ready()
