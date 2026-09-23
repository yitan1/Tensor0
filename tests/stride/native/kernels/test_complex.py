"""Native complex kernel dispatch, tails and arithmetic."""

from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensor0._stride import StridedView, scale
from tensor0._stride._layout import AffineRecord

from tests.stride.support.availability import native_available
from tests.stride.support.oracles.complex_kernels import assert_close, assert_native, mapped, values


pytestmark = pytest.mark.skipif(not native_available(), reason='native CPU stride unavailable')


FACTORS = [complex(0., -0.), .125 + 1j, 2.5 + 3.5j, -1.25 + .75j]


COMPONENTS = np.asarray([0., -0., 1.25, -2.5, .125, 1e10, -1e10, 1e-10, 1e-20], dtype=np.float32)


def rank_two(rows, columns, forward):
    source = (1, rows) if forward else (columns, 1)
    destination = (columns, 1) if forward else (1, rows)
    return AffineRecord((rows, columns), source, 0, destination, 0)


def finite_components(size):
    data = np.asarray([complex(real, imaginary) for real, imaginary in product(COMPONENTS, repeat=2)],
                      dtype=np.complex64)
    return jnp.asarray(np.resize(data, size))


def reference(source, record, factor):
    source_indices = np.zeros(record.logical_shape, dtype=np.int64)
    destinations = np.zeros(record.logical_shape, dtype=np.int64)
    for axis, extent in enumerate(record.logical_shape):
        coordinate = np.arange(extent).reshape((1,) * axis + (extent,) + (1,) * (len(record.logical_shape) - axis - 1))
        source_indices += coordinate * record.source_strides[axis]
        destinations += coordinate * record.destination_strides[axis]
    output = jnp.zeros((*source.shape[:-1], source_indices.size), dtype=jnp.complex64)
    selected = source[..., source_indices.ravel()] * jnp.complex64(factor)
    return output.at[..., destinations.ravel()].set(selected)


@pytest.mark.parametrize("factor", FACTORS)
def test_real_to_complex_compact_finite_values(factor):
    source = jnp.asarray(np.resize(COMPONENTS, 73))
    record = AffineRecord((73,), (1,), 0, (1,), 0)
    actual = jax.jit(lambda value: mapped(value, record, factor))(source)
    assert_close(actual, source * jnp.complex64(factor))


@pytest.mark.parametrize("record", [AffineRecord((131075,), (1,), 0, (1,), 0),
                                    rank_two(256, 256, True), rank_two(517, 263, False)])
def test_complex_large_layouts_and_tails(record):
    source = values(int(np.prod(record.logical_shape)))
    operation = jax.jit(lambda value: mapped(value, record, 1.25 - .75j))
    assert_native(operation.lower(source))
    assert_close(operation(source), reference(source, record, 1.25 - .75j))


@pytest.mark.parametrize("forward", [False, True])
@pytest.mark.parametrize("factor", FACTORS)
def test_rank_two_finite_component_combinations_and_tails(forward, factor):
    record = rank_two(17, 13, forward)
    source = finite_components(221)
    assert_close(jax.jit(lambda value: mapped(value, record, factor))(source), reference(source, record, factor))


@pytest.mark.parametrize("factor", FACTORS)
def test_selected_complex_scale_finite_components_and_tails(factor):
    base = finite_components(67)
    coefficient = jnp.complex64(factor)
    operation = jax.jit(lambda data, alpha: scale(StridedView(data, (67,), (1,), 0), alpha).data)
    assert_native(operation.lower(base, coefficient))
    assert_close(operation(base, coefficient), base * coefficient)
