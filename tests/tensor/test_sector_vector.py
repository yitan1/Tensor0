from dataclasses import FrozenInstanceError

import jax
import jax.numpy as jnp
import pytest

from tensor0 import (
    FermionParity,
    SectorVector,
    TensorMap,
    U1Irrep,
    VectorStorage,
    hom,
    space,
)


def _u1_hom():
    v = space(U1Irrep, {0: 2, 1: 3})
    return hom((v,), (v,))


def _assert_array_equal(left, right):
    assert bool(jnp.array_equal(left, right))


def test_sectorvector_accepts_raw_data_and_vector_storage_matching_space_dims():
    bond = space(U1Irrep, {0: 2, 1: 3})
    data = jnp.arange(5, dtype=jnp.float32)
    raw_vector = SectorVector(bond, data)

    assert raw_vector.space is bond
    assert isinstance(raw_vector.storage, VectorStorage)
    assert raw_vector.storage.data is data

    storage = VectorStorage(data)
    storage_vector = SectorVector(bond, storage)

    assert storage_vector.space is bond
    assert storage_vector.storage is storage
    assert storage_vector.storage.data is data


def test_sectorvector_is_frozen_and_uses_identity_equality():
    bond = space(U1Irrep, {0: 2, 1: 3})
    data = jnp.arange(5, dtype=jnp.float32)
    left = SectorVector(bond, data)
    right = SectorVector(bond, data)

    assert left == left
    assert left != right
    with pytest.raises(FrozenInstanceError):
        setattr(left, "storage", VectorStorage(data))


def test_sectorvector_rejects_non_elementary_space():
    with pytest.raises(TypeError, match="SectorVector requires an ElementarySpace"):
        SectorVector(_u1_hom(), jnp.arange(1))  # pyright: ignore[reportArgumentType]


def test_sectorvector_rejects_wrong_storage_shape():
    bond = space(U1Irrep, {0: 2, 1: 3})

    with pytest.raises(ValueError, match="expected.*5.*actual.*4"):
        SectorVector(bond, jnp.arange(4))

    with pytest.raises(ValueError, match="1D"):
        SectorVector(bond, jnp.zeros((5, 1)))


def test_sectorvector_block_access_by_sector_key():
    bond = space(U1Irrep, {0: 2, 1: 3})
    data = jnp.arange(5, dtype=jnp.float32)
    vector = SectorVector(bond, data)

    _assert_array_equal(vector.block(0), data[0:2])
    _assert_array_equal(vector.block((1,)), data[2:5])

    with pytest.raises(TypeError, match="sector key"):
        vector.block("bad")  # pyright: ignore[reportArgumentType]

    with pytest.raises(KeyError):
        vector.block(2)


def test_sectorvector_blocks_returns_space_order_pairs():
    bond = space(U1Irrep, {1: 3, 0: 2})
    data = jnp.arange(5, dtype=jnp.float32)
    vector = SectorVector(bond, data)

    blocks = vector.blocks()

    assert [sector for sector, _block in blocks] == [(0,), (1,)]
    _assert_array_equal(blocks[0][1], data[0:2])
    _assert_array_equal(blocks[1][1], data[2:5])


def test_sectorvector_supports_tuple_product_sector_keys():
    sector_type = U1Irrep @ FermionParity
    bond = space(sector_type, {(0, 0): 2, (1, 1): 3})
    data = jnp.arange(5, dtype=jnp.float32)
    vector = SectorVector(bond, data)

    _assert_array_equal(vector.block((0, 0)), data[0:2])
    _assert_array_equal(vector.block((1, 1)), data[2:5])


def test_sectorvector_to_diagonal_returns_diagonal_tensormap():
    bond = space(U1Irrep, {0: 2, 1: 3})
    data = jnp.arange(1, 6, dtype=jnp.float32)
    vector = SectorVector(bond, data)

    diagonal = vector.to_diagonal()

    assert isinstance(diagonal, TensorMap)
    assert diagonal.space == hom((bond,), (bond,))
    assert diagonal.storage.data.dtype == data.dtype
    _assert_array_equal(diagonal.block(0), jnp.diag(data[0:2]))
    _assert_array_equal(diagonal.block(1), jnp.diag(data[2:5]))


def test_sectorvector_pytree_children_are_storage_data():
    bond = space(U1Irrep, {0: 2, 1: 3})
    data = jnp.arange(5, dtype=jnp.float32)
    vector = SectorVector(bond, data)

    leaves, treedef = jax.tree_util.tree_flatten(vector)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)

    assert len(leaves) == 1
    assert leaves[0] is vector.storage.data
    assert isinstance(rebuilt, SectorVector)
    assert rebuilt.space == vector.space
    assert rebuilt.storage.data is vector.storage.data


def test_sectorvector_tree_unflatten_reconstructs_sectorvector():
    bond = space(U1Irrep, {0: 2, 1: 3})
    vector = SectorVector(bond, jnp.arange(5, dtype=jnp.float32))
    _leaves, treedef = jax.tree_util.tree_flatten(vector)
    new_data = jnp.arange(5, dtype=jnp.float32) + 20

    rebuilt = jax.tree_util.tree_unflatten(treedef, (new_data,))

    assert isinstance(rebuilt, SectorVector)
    assert rebuilt.space == vector.space
    assert rebuilt.storage.data is new_data

    with pytest.raises(ValueError, match="storage data length mismatch"):
        jax.tree_util.tree_unflatten(treedef, (jnp.arange(4),))
