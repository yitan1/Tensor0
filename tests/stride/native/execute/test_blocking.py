"""Native blocked mixed-precision Reduction and Dot execution."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensor0._stride import enable_threads, get_num_threads, set_num_threads
from tensor0._stride._ffi._calls import execute_dot, execute_reduction
from tensor0._stride._ffi._descriptor import encode_layout, encode_reduction_layout
from tensor0._stride._layout import AffineRecord

from tests.stride.support.availability import native_available
from tests.stride.support.oracles.reduction import assert_close


pytestmark = pytest.mark.skipif(not native_available(), reason='native CPU stride unavailable')


@pytest.fixture(autouse=False)
def restore_worker_limit():
    previous = get_num_threads()
    with jax.enable_x64():
        yield
    if previous is None:
        enable_threads()
    else:
        set_num_threads(previous)


DTYPES = [
    ("float16", "float32", "float32"),
    ("bfloat16", "float32", "float32"),
    ("float32", "float32", "float16"),
    ("float32", "float32", "bfloat16"),
    ("float64", "float32", "float32"),
    ("complex64", "float64", "complex64"),
]


def values(size, batches, dtype):
    signs = np.where(np.arange(size) % 2, -1, 1)
    data = signs[None, :] * np.arange(1, batches + 1)[:, None] * 1.03125
    if dtype == "complex64":
        data = data + signs[None, :] * 0.25j
    return jnp.asarray(data, dtype=dtype)


@pytest.mark.usefixtures("restore_worker_limit")
@pytest.mark.parametrize("source_dtype,right_dtype,result_dtype", DTYPES)
@pytest.mark.parametrize("workers", [1, 4])
@pytest.mark.parametrize("batches", [1, 3])
def test_blocked_dot_mixed_storage(source_dtype, right_dtype, result_dtype, workers, batches):
    set_num_threads(workers)
    rows, columns = 257, 263
    size = rows * columns
    left = values(size, batches, source_dtype)
    right = jnp.ones((batches, size), dtype=right_dtype)
    records = (
        AffineRecord((rows, columns), (-1, rows), rows - 1, (columns, 1), 0),
        AffineRecord((), (), 0, (), 0),
    )
    layout = encode_layout(records, source_size=size, output_size=size)
    operation = jax.jit(lambda lhs, rhs: execute_dot(
        lhs, rhs, layout=layout, conjugate_left=True, dtype=jnp.dtype(result_dtype),
    ))
    actual = operation(left, right)
    # Odd dimensions leave one positive term; the scalar record adds another.
    expected = (2 * jnp.conj(left[:, 0])).astype(result_dtype)
    assert_close(actual, expected)
    np.testing.assert_array_equal(actual, operation(left, right))


@pytest.mark.usefixtures("restore_worker_limit")
@pytest.mark.parametrize("source_dtype,unused_right_dtype,result_dtype", DTYPES)
@pytest.mark.parametrize("workers", [1, 4])
@pytest.mark.parametrize("batches", [1, 3])
def test_blocked_output_domains_mixed_storage(source_dtype, unused_right_dtype, result_dtype, workers, batches):
    set_num_threads(workers)
    rows, columns = 129, 131
    output = rows * columns
    source = values(3 * output, batches, source_dtype)
    record = AffineRecord((rows, columns, 3), (1, rows, output), 0, (columns, 1, 1), 2)
    layout = encode_reduction_layout(
        (record, record), output_shapes=((rows, columns, 1),) * 2,
        reduction_axes=((False, False, True),) * 2,
        source_size=3 * output, output_size=output + 4,
    )
    operation = jax.jit(lambda data, factor: execute_reduction(
        data, (factor,), coefficient_records=(0,), layout=layout,
        output_size=output + 4, dtype=jnp.dtype(result_dtype),
    ))
    actual = operation(source, jnp.float32(0.5))
    # The three slabs have alternating signs; transpose only the spatial axes.
    mapped = source[:, :output].reshape(batches, columns, rows).swapaxes(1, 2).reshape(batches, output)
    expected = jnp.zeros((batches, output + 4), dtype=result_dtype)
    expected = expected.at[:, 2:output + 2].set((mapped * jnp.float32(1.5)).astype(result_dtype))
    assert_close(actual, expected)
    np.testing.assert_array_equal(actual, operation(source, jnp.float32(0.5)))
