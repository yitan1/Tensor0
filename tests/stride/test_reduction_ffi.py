"""Reduction FFI protocol, dtype matrix, and buffer contracts."""

from math import prod

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensor0 import _native
from tensor0._stride._ffi._calls import execute_reduction
from tensor0._stride._ffi._descriptor import encode_reduction_layout

from ._support import native_available
from .test_native_reduction import (
    REDUCTION_DTYPES,
    REDUCTION_FIBER,
    _encode_reduction_records,
    _raw_reduction_call,
)

pytestmark = pytest.mark.skipif(not native_available(), reason="native CPU stride is unavailable")


@pytest.mark.parametrize("changes,source_size,output_size,words", [
    ({}, 3, 1, [1, 3, 1, 1, 1, 0, 0, 3, 1, 1, 1, 1]),
    ({"source_shape": (2, 3), "source_strides": (3, 1), "output_shape": (1, 1),
      "output_strides": (1, 1), "reduction_axes": (True, True)},
     6, 1, [1, 6, 1, 1, 2, 0, 0, 2, 3, 3, 1, 1, 1, 1, 1, 1, 1]),
    ({"source_shape": (0,), "output_offset": 2, "output_strides": (2**63 - 1,)},
     0, 3, [1, 0, 3, 1, 1, 0, 2, 0, 1, 1, 2**63 - 1, 1]),
    ({"source_offset": 2, "source_strides": (-1,)},
     3, 1, [1, 3, 1, 1, 1, 2, 0, 3, 2**64 - 1, 1, 1, 1]),
    ({"source_strides": (0,)}, 1, 1, [1, 1, 1, 1, 1, 0, 0, 3, 0, 1, 1, 1]),
    ({"source_shape": (), "source_strides": (), "source_offset": 2,
      "output_shape": (), "output_strides": (), "output_offset": 1, "reduction_axes": ()},
     3, 2, [1, 3, 2, 1, 0, 2, 1]),
    ({"source_shape": (2, 2), "source_strides": (2, 1), "output_shape": (2, 1),
      "output_strides": (2, 2**63 - 1), "output_offset": 1, "reduction_axes": (False, True)},
     4, 5, [1, 4, 5, 1, 2, 0, 1, 2, 2, 2, 1, 2, 1, 2, 2**63 - 1, 0, 1]),
    ({"source_shape": (2**64 - 1,), "source_strides": (-(2**63),),
      "source_offset": 2**63 - 1, "output_offset": 2**63 - 1, "output_strides": (2**63 - 1,)},
     2**64 - 1, 2**64 - 1,
     [1, 2**64 - 1, 2**64 - 1, 1, 1, 2**63 - 1, 2**63 - 1, 2**64 - 1, 2**63, 1, 2**63 - 1, 1]),
])
def test_single_record_protocol_bytes_unchanged(changes, source_size, output_size, words):
    encoded = encode_reduction_layout(**dict(REDUCTION_FIBER, **changes),
                                      source_size=source_size, output_size=output_size)
    assert encoded.dtype == np.uint8
    assert encoded.tobytes() == b"".join(word.to_bytes(8, "little") for word in words)



@pytest.mark.parametrize("source_dtype", REDUCTION_DTYPES)
@pytest.mark.parametrize("result_dtype", REDUCTION_DTYPES)
@pytest.mark.parametrize("coefficient_dtype", (None, *REDUCTION_DTYPES))
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
        layout = encode_reduction_layout(**REDUCTION_FIBER, source_size=3, output_size=1)
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


def test_record_continuation_and_local_writeback():
    with jax.enable_x64():
        records = (dict(REDUCTION_FIBER, source_shape=(1,)), dict(REDUCTION_FIBER, source_shape=(2,), source_offset=1))
        layout = _encode_reduction_records(records, source_size=3, output_size=1)
        source = jnp.asarray([1, 1e20, -1e20], dtype=jnp.float32)
        np.testing.assert_array_equal(execute_reduction(source, layout=layout, output_size=1), [0])
        source = jnp.asarray([2**24, 1, -(2**24)], dtype=jnp.float64)
        local = encode_reduction_layout(**REDUCTION_FIBER, source_size=3, output_size=1)
        separate = _encode_reduction_records(tuple(
            dict(REDUCTION_FIBER, source_shape=(1,), source_offset=index) for index in range(3)
        ), source_size=3, output_size=1)
        np.testing.assert_array_equal(execute_reduction(source, layout=local, output_size=1, dtype="float32"), [1])
        np.testing.assert_array_equal(execute_reduction(source, layout=separate, output_size=1, dtype="float32"), [0])


def test_mapped_value_is_not_narrowed_before_add():
    with jax.enable_x64():
        records = tuple(dict(REDUCTION_FIBER, source_shape=(1,), source_offset=index) for index in range(2))
        layout = _encode_reduction_records(records, source_size=2, output_size=1)
        result = execute_reduction(jnp.asarray([-1, 1], dtype=jnp.float32),
                                   (jnp.float64(1 + 2**-24),), coefficient_records=(1,),
                                   layout=layout, output_size=1)
        np.testing.assert_array_equal(result, [np.float32(2**-24)])


def test_shared_planner_order_counterexample():
    record = dict(source_shape=(3, 3, 2), source_strides=(2, 1, 3), source_offset=0, output_shape=(1, 1, 1), output_strides=(1, 1, 1), output_offset=0, reduction_axes=(True, True, True))
    layout = encode_reduction_layout(**record, source_size=10, output_size=1)
    source = jnp.asarray([0, 0, 1e20, -1e20, 1, 0, 0, 0, 0, 0], dtype=jnp.float32)
    np.testing.assert_array_equal(execute_reduction(source, layout=layout, output_size=1), [1])


def test_nonzero_destination_stride_writes_back_each_sum():
    with jax.enable_x64():
        record = dict(source_shape=(2,), source_strides=(1,), source_offset=0, output_shape=(2,), output_strides=(1,), output_offset=0, reduction_axes=(False,))
        records = (record, dict(record, source_offset=2))
        layout = _encode_reduction_records(records, source_size=4, output_size=2)
        source = jnp.asarray([-1, -1, 1 + 2**-24, 1 + 2**-24], dtype=jnp.float64)
        np.testing.assert_array_equal(execute_reduction(source, layout=layout, output_size=2, dtype="float32"),
                                      [np.float32(2**-24)] * 2)


def test_unit_skips_promotion_and_zero_skips_input():
    with jax.enable_x64():
        layout = encode_reduction_layout(**dict(REDUCTION_FIBER, source_shape=(1,)), source_size=1, output_size=1)
        source = jnp.asarray([16777217], dtype=jnp.int32)
        np.testing.assert_array_equal(execute_reduction(source, layout=layout, output_size=1, dtype="int64"), [16777217])
        np.testing.assert_array_equal(execute_reduction(source, (jnp.float32(1),), coefficient_records=(0,),
                                                       layout=layout, output_size=1, dtype="int64"), [16777217])
        result = execute_reduction(jnp.asarray([np.inf], dtype=jnp.float32), (jnp.float32(0),),
                                   coefficient_records=(0,), layout=layout, output_size=1)
        np.testing.assert_array_equal(result, [0])


def test_integer_mapping_wrap_and_final_conversion():
    layout = encode_reduction_layout(**dict(REDUCTION_FIBER, source_shape=(2,)), source_size=2, output_size=1)
    mapped = execute_reduction(jnp.asarray([100, 100], dtype=jnp.int8), (jnp.int8(2),),
                               coefficient_records=(0,), layout=layout, output_size=1, dtype="int32")
    np.testing.assert_array_equal(mapped, [-112])
    result = execute_reduction(jnp.asarray([.9, .9], dtype=jnp.float32), layout=layout,
                               output_size=1, dtype="int32")
    np.testing.assert_array_equal(result, [1])


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_low_precision_mapping_rounds_before_wider_sum(dtype):
    source = jnp.asarray([.33325, .7, 1.01], dtype=dtype)
    coefficient = jnp.asarray(.33325, dtype=dtype)
    layout = encode_reduction_layout(**REDUCTION_FIBER, source_size=3, output_size=1)
    expected = jnp.sum((source * coefficient).astype(jnp.float32))
    result = execute_reduction(source, (coefficient,), coefficient_records=(0,),
                               layout=layout, output_size=1, dtype="float32")
    np.testing.assert_array_equal(result, np.asarray(expected).reshape(1))


def test_broadcast_and_negative_output_stride():
    record = dict(source_shape=(2, 3), source_strides=(1, 0), source_offset=0, output_shape=(2, 1), output_strides=(-1, 0), output_offset=2, reduction_axes=(False, True))
    layout = encode_reduction_layout(**record, source_size=2, output_size=4)
    np.testing.assert_array_equal(execute_reduction(jnp.asarray([1, 2], dtype=jnp.int32),
                                                   layout=layout, output_size=4), [0, 6, 3, 0])


@pytest.mark.parametrize("batch_shape", [(), (2,), (0,), (2, 0)])
def test_empty_input_preserves_real_output_initialization(batch_shape):
    record = dict(REDUCTION_FIBER, source_shape=(0,), output_offset=2, output_strides=(2**63 - 1,))
    layout = encode_reduction_layout(**record, source_size=0, output_size=3)
    source = jnp.zeros((*batch_shape, 0), dtype=jnp.float32)
    result = execute_reduction(source, (jnp.float32(np.nan),), coefficient_records=(0,), layout=layout, output_size=3)
    np.testing.assert_array_equal(result, np.zeros((*batch_shape, 3)))
    no_records = _encode_reduction_records((), source_size=0, output_size=3)
    np.testing.assert_array_equal(execute_reduction(source, layout=no_records, output_size=3), result)


def test_zero_output_and_large_unsigned_extent():
    empty = dict(source_shape=(0, 2), source_strides=(2, 1), source_offset=0, output_shape=(0, 1), output_strides=(1, 1), output_offset=0, reduction_axes=(False, True))
    layout = encode_reduction_layout(**empty, source_size=0, output_size=0)
    assert execute_reduction(jnp.zeros(0), layout=layout, output_size=0).shape == (0,)
    huge = dict(REDUCTION_FIBER, source_shape=(2**63,), source_strides=(0,), output_strides=(-(2**63),))
    layout = encode_reduction_layout(**huge, source_size=1, output_size=1)
    assert execute_reduction(jnp.zeros((0, 1)), layout=layout, output_size=1).shape == (0, 1)


def test_high_rank_and_real_coefficient():
    with jax.enable_x64():
        record = dict(source_shape=(1,) * 12, source_strides=(2**63 - 1,) * 12, source_offset=0, output_shape=(1,) * 12, output_strides=(2**63 - 1,) * 12, output_offset=0, reduction_axes=(True,) * 12)
        layout = encode_reduction_layout(**record, source_size=1, output_size=1)
        source = jnp.asarray([complex(1, np.inf)], dtype=jnp.complex64)
        actual = np.asarray(execute_reduction(source, (jnp.float32(2),), coefficient_records=(0,),
                                              layout=layout, output_size=1, dtype="complex128"))
        assert actual[0].real == 2 and np.isposinf(actual[0].imag)


def test_python_argument_contracts_and_raw_ffi_ad_rejection():
    layout = encode_reduction_layout(**REDUCTION_FIBER, source_size=3, output_size=1)
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
    layout = encode_reduction_layout(**REDUCTION_FIBER, source_size=3, output_size=1)
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
    layout = encode_reduction_layout(**REDUCTION_FIBER, source_size=3, output_size=1)
    with pytest.raises(Exception, match="coefficient.*(count|indices)"):
        _raw_reduction_call(jnp.ones((1, 3)), layout, coefficients=(jnp.float32(2),) * count, indices=indices).block_until_ready()


def test_invalid_buffers_and_aliases():
    layout = encode_reduction_layout(**REDUCTION_FIBER, source_size=3, output_size=1)
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
    layout = encode_reduction_layout(**identity, source_size=3, output_size=3)
    with pytest.raises(Exception, match="overlap"):
        _raw_reduction_call(jnp.ones((1, 3)), layout, shape=(1, 3), aliases={0: 0}).block_until_ready()


@pytest.mark.parametrize("changes", [
    {"source_shape": (2**64,)}, {"source_strides": (-(2**63) - 1,)}, {"source_offset": -1},
    {"output_offset": 2**63}, {"reduction_axes": (1,)}, {"output_shape": ()},
])
def test_encoding_rejects_invalid_representation(changes):
    with pytest.raises(ValueError):
        encode_reduction_layout(**dict(REDUCTION_FIBER, **changes), source_size=3, output_size=1)


def test_layout_bytes_and_native_validation():
    layout = encode_reduction_layout(**REDUCTION_FIBER, source_size=3, output_size=1)
    np.testing.assert_array_equal(layout.view("<u8"), [1, 3, 1, 1, 1, 0, 0, 3, 1, 1, 1, 1])
    for invalid in (layout[:-1], np.concatenate((layout, np.zeros(8, dtype=np.uint8)))):
        with pytest.raises(Exception, match="partial word|trailing words"):
            execute_reduction(jnp.ones(3), layout=invalid, output_size=1).block_until_ready()
    for changes in ({"output_shape": (2,)}, {"source_strides": (2,)}, {"output_offset": 1}):
        invalid = encode_reduction_layout(**dict(REDUCTION_FIBER, **changes), source_size=3, output_size=1)
        with pytest.raises(Exception, match="shape|address|storage"):
            execute_reduction(jnp.ones(3), layout=invalid, output_size=1).block_until_ready()
    overlapping = dict(source_shape=(2, 2), source_strides=(2, 1), source_offset=0, output_shape=(2, 2), output_strides=(1, 1), output_offset=0, reduction_axes=(False, False))
    invalid = encode_reduction_layout(**overlapping, source_size=4, output_size=3)
    with pytest.raises(Exception, match="injective|overlap"):
        execute_reduction(jnp.ones(4), layout=invalid, output_size=3).block_until_ready()
