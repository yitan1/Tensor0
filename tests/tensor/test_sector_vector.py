from dataclasses import FrozenInstanceError

import jax
import jax.numpy as jnp
import pytest

from tensor0 import (
    DiagonalTensorMap,
    FermionParity,
    SectorVector,
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

    assert not hasattr(raw_vector, "space")
    assert raw_vector.sector_type == U1Irrep
    assert raw_vector.sectors == (((0,), 2), ((1,), 3))
    assert isinstance(raw_vector.storage, VectorStorage)
    assert raw_vector.storage.data is data

    storage = VectorStorage(data)
    storage_vector = SectorVector(bond, storage)

    assert storage_vector.sector_type == U1Irrep
    assert storage_vector.sectors == (((0,), 2), ((1,), 3))
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


def test_sectorvector_exposes_read_only_block_mapping_helpers():
    bond = space(U1Irrep, {1: 3, 0: 2})
    data = jnp.arange(5, dtype=jnp.float32)
    vector = SectorVector(bond, data)

    assert vector.blocksectors == ((0,), (1,))
    assert vector.keys() == ((0,), (1,))
    assert vector.hasblock(0)
    assert not vector.hasblock(2)

    values = tuple(vector.values())
    pairs = tuple(vector.pairs())

    assert tuple(sector for sector, _block in pairs) == ((0,), (1,))
    _assert_array_equal(values[0], data[0:2])
    _assert_array_equal(values[1], data[2:5])
    _assert_array_equal(vector.get(0), data[0:2])
    assert vector.get(2, "missing") == "missing"


def test_sectorvector_copy_and_similar_preserve_sector_structure():
    bond = space(U1Irrep, {0: 2, 1: 3})
    data = jnp.arange(5, dtype=jnp.float32)
    vector = SectorVector(bond, data)

    copied = vector.copy()
    same_layout = vector.similar()
    typed = vector.similar(jnp.float16)
    target_space = space(U1Irrep, {0: 1})
    resized = vector.similar(space=target_space)

    assert isinstance(copied, SectorVector)
    assert copied.sector_type == vector.sector_type
    assert copied.sectors == vector.sectors
    _assert_array_equal(copied.storage.data, data)

    assert same_layout.sectors == vector.sectors
    assert same_layout.storage.data.shape == (5,)
    assert same_layout.storage.data.dtype == data.dtype

    assert typed.sectors == vector.sectors
    assert typed.storage.data.shape == (5,)
    assert typed.storage.data.dtype == jnp.dtype(jnp.float16)

    assert resized.sectors == (((0,), 1),)
    assert resized.storage.data.shape == (1,)
    assert resized.storage.data.dtype == data.dtype


def test_sectorvector_supports_tuple_product_sector_keys():
    sector_type = U1Irrep @ FermionParity
    bond = space(sector_type, {(0, 0): 2, (1, 1): 3})
    data = jnp.arange(5, dtype=jnp.float32)
    vector = SectorVector(bond, data)

    _assert_array_equal(vector.block((0, 0)), data[0:2])
    _assert_array_equal(vector.block((1, 1)), data[2:5])


def test_sectorvector_constructed_from_dual_space_keeps_visible_sector_labels():
    dual_bond = space(U1Irrep, {1: 2}, dual=True)
    data = jnp.arange(2, dtype=jnp.float32)
    vector = SectorVector(dual_bond, data)

    assert vector.sector_type == U1Irrep
    assert vector.sectors == (((-1,), 2),)
    _assert_array_equal(vector.block(-1), data)

    diagonal = vector.to_diagonal()

    assert diagonal.domain == space(U1Irrep, {-1: 2})
    assert not diagonal.domain.is_dual


def test_sectorvector_to_diagonal_returns_diagonal_tensormap():
    bond = space(U1Irrep, {0: 2, 1: 3})
    data = jnp.arange(1, 6, dtype=jnp.float32)
    vector = SectorVector(bond, data)

    diagonal = vector.to_diagonal()

    assert isinstance(diagonal, DiagonalTensorMap)
    assert diagonal.domain == bond
    assert diagonal.storage.data.dtype == data.dtype
    _assert_array_equal(diagonal.diag().block(0), data[0:2])
    _assert_array_equal(diagonal.diag().block(1), data[2:5])


def test_sectorvector_pytree_children_are_storage_data():
    bond = space(U1Irrep, {0: 2, 1: 3})
    data = jnp.arange(5, dtype=jnp.float32)
    vector = SectorVector(bond, data)

    leaves, treedef = jax.tree_util.tree_flatten(vector)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)

    assert len(leaves) == 1
    assert leaves[0] is vector.storage.data
    assert isinstance(rebuilt, SectorVector)
    assert rebuilt.sector_type == vector.sector_type
    assert rebuilt.sectors == vector.sectors
    assert rebuilt.storage.data is vector.storage.data

    new_data = jnp.arange(5, dtype=jnp.float32) + 20

    rebuilt_with_new_data = jax.tree_util.tree_unflatten(treedef, (new_data,))

    assert isinstance(rebuilt_with_new_data, SectorVector)
    assert rebuilt_with_new_data.sector_type == vector.sector_type
    assert rebuilt_with_new_data.sectors == vector.sectors
    assert rebuilt_with_new_data.storage.data is new_data

    with pytest.raises(ValueError, match="storage data length mismatch"):
        jax.tree_util.tree_unflatten(treedef, (jnp.arange(4),))
