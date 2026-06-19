from dataclasses import FrozenInstanceError

import jax
import jax.numpy as jnp
import pytest

from tensor0 import DiagonalTensorMap, SectorVector, TensorMap, U1Irrep, hom, space
from tests.cases import assert_allclose


def test_diagonal_tensormap_exposes_domain_and_storage():
    bond = space(U1Irrep, {0: 2, 1: 3})
    data = jnp.arange(5, dtype=jnp.float32)
    diagonal = DiagonalTensorMap(bond, data)

    assert diagonal.domain is bond
    assert diagonal.storage.data is data
    assert diagonal.space == hom((bond,), (bond,))
    with pytest.raises(FrozenInstanceError):
        setattr(diagonal, "storage", diagonal.storage)


def test_diagonal_tensormap_rejects_invalid_domain_and_storage_shape():
    bond = space(U1Irrep, {0: 2, 1: 3})

    with pytest.raises(TypeError, match="DiagonalTensorMap requires an ElementarySpace"):
        DiagonalTensorMap(hom((bond,), (bond,)), jnp.arange(5))  # pyright: ignore[reportArgumentType]

    with pytest.raises(ValueError, match="expected.*5.*actual.*4"):
        DiagonalTensorMap(bond, jnp.arange(4))

    with pytest.raises(ValueError, match="1D"):
        DiagonalTensorMap(bond, jnp.zeros((5, 1)))


def test_diagonal_tensormap_diag_returns_sectorvector_view():
    bond = space(U1Irrep, {0: 2, 1: 3})
    data = jnp.arange(5, dtype=jnp.float32)
    diagonal = DiagonalTensorMap(bond, data)

    values = diagonal.diag()

    assert isinstance(values, SectorVector)
    assert not hasattr(values, "space")
    assert values.sector_type == U1Irrep
    assert values.sectors == bond.sectors
    assert values.storage.data is diagonal.storage.data
    assert_allclose(values.block(0), data[0:2])
    assert_allclose(values.block(1), data[2:5])


def test_diagonal_tensormap_to_tensor_map_expands_dense_diagonal_blocks():
    bond = space(U1Irrep, {0: 2, 1: 3})
    data = jnp.arange(1, 6, dtype=jnp.float32)
    diagonal = DiagonalTensorMap(bond, data)

    tensor = diagonal.to_tensor_map()

    assert isinstance(tensor, TensorMap)
    assert tensor.space == hom((bond,), (bond,))
    assert_allclose(tensor.block(0), jnp.diag(data[0:2]))
    assert_allclose(tensor.block(1), jnp.diag(data[2:5]))


def test_diagonal_tensormap_pytree_children_are_storage_data():
    bond = space(U1Irrep, {0: 2, 1: 3})
    data = jnp.arange(5, dtype=jnp.float32)
    diagonal = DiagonalTensorMap(bond, data)

    leaves, treedef = jax.tree_util.tree_flatten(diagonal)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)

    assert len(leaves) == 1
    assert leaves[0] is diagonal.storage.data
    assert isinstance(rebuilt, DiagonalTensorMap)
    assert rebuilt.domain == diagonal.domain
    assert rebuilt.storage.data is diagonal.storage.data
