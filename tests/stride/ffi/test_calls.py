"""Raw native calls, initialization and layout reuse."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensor0._stride._ffi._calls import execute_accumulation, execute_copy, execute_update
from tensor0._stride._ffi._descriptor import encode_layout
from tensor0._stride._ffi._registration import operation_target
from tensor0._stride._jax import accumulation_p
from tensor0._stride._layout import AffineRecord

from tests.stride.support.availability import native_available
from tests.stride.support.oracles.copy import PARTITIONS, assert_single_native_call, reference_map


LAYOUTS = [
    pytest.param(PARTITIONS, (1.25, -.5), 16, 16, id="partitions"),
    pytest.param((AffineRecord((3,), (2,), 0, (1,), 0),), (-2.,), 5, 3, id="gapped"),
    pytest.param((AffineRecord((2,), (1,), 0, (1,), 0), AffineRecord((2,), (1,), 3, (1,), 2)),
                 (1.5, -.5), 5, 4, id="gapped-records"),
    pytest.param((AffineRecord((16,), (2,), 1, (3,), 2),), (.5,), 32, 50, id="partial"),
    pytest.param((), (), 0, 8, id="empty-partial"),
    pytest.param((), (), 0, 0, id="empty-output"),
    pytest.param((AffineRecord((0,), (1,), 0, (1,), 0),), (1.,), 0, 8, id="empty-record"),
    pytest.param((AffineRecord((), (), 0, (), 0),), (2.5,), 1, 1, id="scalar"),
    pytest.param((AffineRecord((2, 4, 3, 8), (32, 1, 64, 4), 0, (96, 24, 8, 1), 0),),
                 (-.75,), 192, 192, id="rank4-tiled-shape"),
    pytest.param((AffineRecord((2, 8, 3, 12), (96, 1, 192, 8), 0, (288, 36, 12, 1), 0),),
                 (1.25,), 576, 576, id="rank4-avx-shape"),
    pytest.param((AffineRecord((1, 1, 1), (1, 1, 1), 0, (1, 1, 1), 0),
                  AffineRecord((2, 2, 1), (1, 2, 1), 1, (2, 1, 1), 1)),
                 (1., 1.), 5, 5, id="u1-two-record"),
    pytest.param((AffineRecord((12, 16, 8), (8, 96, 1), 0, (256, 8, 1), 0),
                  AffineRecord((12, 16, 8), (8, 96, 1), 1536, (256, 8, 1), 128),
                  AffineRecord((12, 16, 8), (8, 96, 1), 3072, (128, 8, 1), 3072)),
                 (1., 1., 1.), 4608, 4608, id="u1-three-record"),
]


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
@pytest.mark.parametrize("records,factors,source_size,output_size", LAYOUTS)
@pytest.mark.parametrize("batch_shape", [(), (2, 3)])
def test_native_layouts_and_fresh_initialization(records, factors, source_size, output_size, batch_shape):
    source = jnp.arange(int(np.prod(batch_shape)) * source_size, dtype=jnp.float32).reshape(*batch_shape, source_size)
    coefficients = tuple(jnp.float32(value) for value in factors)
    operation = jax.jit(lambda value: accumulation_p.bind(
        value, *coefficients, records=records, coefficient_records=tuple(range(len(records))),
        output_size=output_size, dtype=value.dtype,
    ))
    np.testing.assert_array_equal(operation(source), reference_map(np.asarray(source), records, factors, output_size))
    assert_single_native_call(operation.lower(source))


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
@pytest.mark.parametrize("invalid,message", [
    ("source_size", "buffer dimensions do not match layout"),
    ("output_size", "buffer dimensions do not match layout"),
    ("batch_count", "copy batch dimensions do not match"),
    ("alias", "input and output buffers overlap"),
    ("overlapping_record", "injective|overlap"),
])
def test_copy_ffi_revalidates_batched_buffers_and_records(invalid, message):
    layout = encode_layout(PARTITIONS, source_size=16, output_size=16)
    source_shape, output_shape = (2, 16), (2, 16)
    if invalid == "source_size":
        source_shape = (2, 15)
    elif invalid == "output_size":
        output_shape = (2, 15)
    elif invalid == "batch_count":
        output_shape = (3, 16)
    elif invalid == "overlapping_record":
        layout[-2:] = (1, 1)
    call = jax.ffi.ffi_call(operation_target("copy", np.dtype(jnp.float32)),
                            jax.ShapeDtypeStruct(output_shape, jnp.float32),
                            input_output_aliases={0: 0} if invalid == "alias" else {})
    source = jnp.arange(np.prod(source_shape), dtype=jnp.float32).reshape(source_shape)
    with pytest.raises(Exception, match=message):
        call(source, layout=layout).block_until_ready()


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
@pytest.mark.parametrize("dtype,factor", [(jnp.float16, 1.3), (jnp.bfloat16, -.7),
                                         (jnp.complex64, .75 - .5j), (jnp.int32, -1)])
@pytest.mark.parametrize("partial", [False, True])
def test_typed_scaling_transpose_and_partial_zero_fill(dtype, factor, partial):
    record = AffineRecord((5, 7), (1, 5), 0, (7, 1), 3 if partial else 0)
    output_size = 41 if partial else 35
    source = (jnp.arange(105, dtype=jnp.int32).reshape(3, 35) % 17 - 8).astype(dtype)
    if dtype == jnp.complex64:
        source = source + .5j * source
    if dtype == jnp.int32:
        source = source.at[0, :2].set(jnp.asarray([np.iinfo(np.int32).min, np.iinfo(np.int32).max]))
    coefficient = jnp.asarray(factor, dtype=jnp.complex64 if dtype == jnp.complex64 else
                              jnp.int32 if dtype == jnp.int32 else jnp.float32)
    operation = jax.jit(lambda value: accumulation_p.bind(value, coefficient, records=(record,),
                         coefficient_records=(0,), output_size=output_size, dtype=value.dtype))
    expected_values = (source * coefficient).astype(dtype).reshape(3, 7, 5).transpose(0, 2, 1).reshape(3, 35)
    expected = jnp.zeros((3, output_size), dtype=dtype).at[:, record.destination_offset:record.destination_offset + 35].set(expected_values)
    actual = operation(source)
    if dtype == jnp.int32:
        np.testing.assert_array_equal(actual, expected)
    else:
        np.testing.assert_allclose(np.asarray(actual).astype(np.complex64), np.asarray(expected).astype(np.complex64),
                                   rtol=2e-6, atol=1e-6)
    assert_single_native_call(operation.lower(source))


requires_native = pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")


@requires_native
def test_layout_reuse_does_not_capture_factors_or_initialization():
    layout = encode_layout((AffineRecord((2,), (1,), 0, (1,), 1),), source_size=2, output_size=4)
    before = layout.tobytes()
    source, base = jnp.asarray([2., 3.], dtype=jnp.float32), jnp.full(4, 7., dtype=jnp.float32)
    np.testing.assert_array_equal(execute_copy(source, layout=layout, output_size=4), [0, 2, 3, 0])
    for factor in (1, 2, -1):
        coefficient = jnp.float32(factor)
        mapped = execute_accumulation(source, (coefficient,), coefficient_records=(0,), layout=layout, output_size=4)
        np.testing.assert_array_equal(mapped, [0, 2 * factor, 3 * factor, 0])
        for beta in (0, 1):
            updated = execute_update(source, base, coefficient, jnp.int32(beta), layout=layout)
            np.testing.assert_array_equal(updated, [7, 2 * factor + 7 * beta, 3 * factor + 7 * beta, 7])
    assert layout.tobytes() == before
