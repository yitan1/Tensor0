"""Real subblock metadata and view-update integration."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensor0 import SU2Irrep, hom, space
from tensor0._stride import StridedView, add, materialize, scale
from tensor0.structure import get_degeneracystructure

from tests.stride.support.availability import native_available
from tests.stride.support.oracles.materialize import _view, reference_materialize


def test_subblock_metadata_can_be_materialized_directly() -> None:
    factor = space(SU2Irrep, {0: 2, 1: 3})
    target = hom((factor, factor), (factor, factor))
    layout = get_degeneracystructure(target)
    subblock = layout.subblockstructure[0]
    source = jnp.arange(layout.total_dim, dtype=jnp.float32)
    expected = reference_materialize(
        source,
        tuple(subblock.sizes),
        tuple(subblock.strides),
        subblock.offset,
    )
    actual = materialize(
        _view(
            source,
            tuple(subblock.sizes),
            tuple(subblock.strides),
            subblock.offset,
        ),
    )

    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


@pytest.fixture(params=[0, 3, 8], ids=["first", "offset-noncompact", "last"])
def subblock_case(request):
    factor = space(SU2Irrep, {0: 2, 1: 3})
    layout = get_degeneracystructure(hom((factor, factor), (factor, factor)))
    block = layout.subblockstructure[request.param]
    sizes, strides = tuple(block.sizes), tuple(block.strides)
    indices = np.asarray([
        block.offset + sum(item * stride for item, stride in zip(index, strides, strict=True))
        for index in np.ndindex(sizes)
    ], dtype=np.int32).reshape(sizes)
    assert np.unique(indices).size == indices.size
    assert np.all((indices >= 0) & (indices < layout.total_dim))
    return layout.total_dim, sizes, strides, block.offset, indices


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
@pytest.mark.parametrize("operation", ["assign", "accumulate", "scale"])
def test_real_subblock_updates_jit_vmap_and_ad(subblock_case, operation):
    storage_size, sizes, strides, offset, indices = subblock_case
    storage = jnp.linspace(-2, 3, storage_size, dtype=jnp.float32)
    if operation == "scale":
        operand = jnp.float32(-1.25)
        direction = jnp.float32(.75)
        native = lambda old, value: scale(StridedView(old, sizes, strides, offset), value).data
        reference = lambda old, value: old.at[indices].set(old[indices] * value)
    else:
        operand = jnp.linspace(4, -1, indices.size, dtype=jnp.float32).reshape(sizes)
        direction = jnp.full_like(operand, .75)
        alpha = 0 if operation == "assign" else 1
        native = lambda old, value: add(
            StridedView(old, sizes, strides, offset), StridedView.from_dense(value, sizes),
            alpha=alpha, beta=1,
        ).data
        reference = lambda old, value: old.at[indices].set(alpha * old[indices] + value)
    expected = reference(storage, operand)
    original_storage, original_operand = np.asarray(storage).copy(), np.asarray(operand).copy()
    for actual in (native(storage, operand), jax.jit(native)(storage, operand)):
        np.testing.assert_array_equal(actual, expected)
        selected = np.zeros(storage_size, dtype=bool)
        selected[indices.ravel()] = True
        assert np.asarray(actual)[~selected].tobytes() == original_storage[~selected].tobytes()
    np.testing.assert_array_equal(storage, original_storage)
    np.testing.assert_array_equal(operand, original_operand)
    text = jax.jit(native).lower(storage, operand).as_text().lower()
    assert text.count("custom_call") == 1
    assert "tensor0_stride_update_f32_cpu_v1" in text
    assert "gather" not in text and "scatter" not in text

    storage_direction = jnp.linspace(1, -2, storage_size, dtype=jnp.float32)
    cotangent = jnp.linspace(-3, 4, storage_size, dtype=jnp.float32)

    def derivatives(function, old, value):
        forward = jax.jvp(function, (old, value), (storage_direction, direction))
        reverse = jax.vjp(function, old, value)[1](cotangent)
        return forward, reverse

    actual = jax.jit(lambda old, value: derivatives(native, old, value))(storage, operand)
    expected = derivatives(reference, storage, operand)
    for result, wanted in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(result, wanted, rtol=2e-6, atol=2e-6)

    storage_batch = jnp.stack((storage, 2 * storage, -storage))
    operand_batch = jnp.stack((operand, -.5 * operand, 3 * operand))
    np.testing.assert_array_equal(jax.jit(jax.vmap(native))(storage_batch, operand_batch),
                                  jax.vmap(reference)(storage_batch, operand_batch))


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
def test_real_subblock_assignment_preserves_complex_special_value_bits(subblock_case):
    storage_size, sizes, strides, offset, indices = subblock_case
    storage = jnp.full(storage_size, complex(-0., 0.), dtype=jnp.complex64)
    pattern = jnp.asarray([
        complex(np.inf, 0.), complex(0., np.inf), complex(-np.inf, 2.), complex(np.nan, -3.),
    ], dtype=jnp.complex64)
    value = jnp.resize(pattern, sizes)
    expected = np.asarray(storage).copy()
    expected[indices] = np.asarray(value)
    native = lambda old, source: add(
        StridedView(old, sizes, strides, offset), StridedView.from_dense(source, sizes), alpha=0, beta=1,
    ).data
    for actual in (native(storage, value), jax.jit(native)(storage, value)):
        np.testing.assert_array_equal(np.asarray(actual).view(np.uint32), expected.view(np.uint32))
