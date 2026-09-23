"""Native repeated-address accumulation and record execution."""

from math import prod

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensor0._stride._ffi._calls import execute_accumulation
from tensor0._stride._ffi._descriptor import encode_layout
from tensor0._stride._layout import AffineRecord

from tests.stride.support.availability import native_available
from tests.stride.support.layouts import OVERLAP


pytestmark = pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')


@pytest.fixture(autouse=False)
def enable_x64():
    with jax.enable_x64():
        yield


@pytest.mark.usefixtures("enable_x64")
@pytest.mark.parametrize('batch_shape', [(), (2,), (2, 3), (0,), (2, 0)])
@pytest.mark.parametrize('record', [OVERLAP, AffineRecord((2, 2, 2), (-1, -2, -4), 7, (1, -1, 0), 2), AffineRecord((2, 2, 2), (1, 2, 4), 0, (0, 1, 1), 2), AffineRecord((0,), (-(1 << 63),), 8, ((1 << 63) - 1,), 8), AffineRecord((), (), 2, (), 3), AffineRecord((1,) * 70, ((1 << 63) - 1,) * 70, 2, (-(1 << 63),) * 70, 3)])
def test_layouts_and_batches(batch_shape, record):
    source = jnp.arange(prod(batch_shape) * 8, dtype=jnp.float32).reshape((*batch_shape, 8))
    expected = np.zeros((*batch_shape, 8), np.float32)
    for coordinates in np.ndindex(record.logical_shape):
        source_index = record.source_offset + sum((axis * stride for axis, stride in zip(coordinates, record.source_strides)))
        output_index = record.destination_offset + sum((axis * stride for axis, stride in zip(coordinates, record.destination_strides)))
        expected[..., output_index] += np.asarray(source)[..., source_index]
    layout = encode_layout((record,), source_size=8, output_size=8)
    execute = lambda data: execute_accumulation(data, layout=layout, output_size=8)
    for call in (execute, jax.jit(execute)):
        np.testing.assert_array_equal(call(source), expected)


@pytest.mark.usefixtures("enable_x64")
@pytest.mark.parametrize('batch_shape', [(), (3,), (2, 3), (0,), (2, 0)])
def test_record_coefficients_and_reuse(batch_shape):
    records = (OVERLAP, AffineRecord((0,), (1,), 8, (1,), 8), AffineRecord((2, 2), (-1, -2), 7, (-1, -1), 3))
    source = jnp.arange(prod(batch_shape) * 8, dtype=jnp.float32).reshape((*batch_shape, 8)) + 1
    original = np.asarray(source).copy()
    factors = jnp.asarray(np.arange(prod(batch_shape)).reshape(batch_shape) % 3, dtype=jnp.int32)
    layout = encode_layout(records, source_size=8, output_size=8)
    execute = jax.jit(lambda data, first, second: execute_accumulation(data, (first, second), coefficient_records=(0, 2), layout=layout, output_size=8))
    for shift in (0, 2):
        first, second = (jnp.float32(2 + shift), factors + shift)
        expected = np.zeros((*batch_shape, 8), np.float32)
        for record, coefficient in ((records[0], first), (records[2], second)):
            for coordinates in np.ndindex(record.logical_shape):
                source_index = record.source_offset + sum((axis * stride for axis, stride in zip(coordinates, record.source_strides)))
                output_index = record.destination_offset + sum((axis * stride for axis, stride in zip(coordinates, record.destination_strides)))
                expected[..., output_index] += original[..., source_index] * np.asarray(coefficient)
        np.testing.assert_array_equal(execute(source, first, second), expected)
    np.testing.assert_array_equal(source, original)
    hlo = execute.lower(source, jnp.float32(2), factors).compiler_ir(dialect='hlo').as_hlo_text()
    assert hlo.count('custom-call(') == (1 if prod(batch_shape) else 0)


@pytest.mark.usefixtures("enable_x64")
def test_empty_storage_and_no_records():
    for records in ((), (AffineRecord((0,), (1,), 0, (1,), 5),)):
        np.testing.assert_array_equal(execute_accumulation(jnp.empty(0, jnp.float32), layout=encode_layout(records, source_size=0, output_size=5), output_size=5), np.zeros(5))
    assert execute_accumulation(jnp.empty((2, 0)), layout=encode_layout((), source_size=0, output_size=0), output_size=0).shape == (2, 0)


@pytest.mark.usefixtures("enable_x64")
@pytest.mark.parametrize('batch_shape', [(), (2,), (2, 3), (0,), (2, 0, 3)])
@pytest.mark.parametrize('per_record', [False, True])
def test_dynamic_record_factors_batches_and_empty_records(batch_shape, per_record):
    records = (AffineRecord((0,), (1,), 0, (1,), 0), AffineRecord((2, 3), (1, 2), 0, (3, 1), 1), AffineRecord((3,), (0,), 6, (1,), 7))
    layout = encode_layout(records, source_size=7, output_size=11)
    source = jnp.arange(prod(batch_shape) * 7, dtype=jnp.int16).reshape((*batch_shape, 7))
    factors = jnp.asarray([99, 0.5, -0.25] if per_record else 0.5, dtype=jnp.float32)
    execute = jax.jit(lambda values, coefficients: execute_accumulation(values, tuple(coefficients) if per_record else (coefficients,) * 3, coefficient_records=(0, 1, 2), layout=layout, output_size=11, dtype='float32'))
    for coefficients in (factors, factors + 1):
        expected = np.zeros((*batch_shape, 11), dtype=np.float32)
        factor = np.asarray(coefficients)
        expected[..., 1:7] = np.asarray(source)[..., [0, 2, 4, 1, 3, 5]] * (factor[1] if per_record else factor)
        expected[..., 7:10] = np.asarray(source)[..., 6:7] * (factor[2] if per_record else factor)
        np.testing.assert_array_equal(execute(source, coefficients), expected)


@pytest.mark.usefixtures("enable_x64")
def test_empty_source_initializes_real_output():
    layout = encode_layout((AffineRecord((0,), (1,), 0, (1,), 2),), source_size=0, output_size=4)
    result = execute_accumulation(jnp.zeros(0, dtype=jnp.int16), (jnp.float32(np.nan),), coefficient_records=(0,), layout=layout, output_size=4, dtype='float32')
    np.testing.assert_array_equal(result, np.zeros(4))
