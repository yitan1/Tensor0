from dataclasses import FrozenInstanceError

import jax
import jax.numpy as jnp
import pytest

import tensor0
import tensor0.tensor as tensor_api
from tensor0 import (
    DiagonalTensorMap,
    FermionParity,
    ProductSpace,
    SU2Irrep,
    SectorVector,
    TensorMap,
    U1Irrep,
    hom,
    space,
)
from tests.cases import assert_allclose, assert_tensormap_blocks_allclose


def _u1_diagonal(data=None):
    index_space = space(U1Irrep, {0: 2, 1: 3})
    if data is None:
        data = jnp.arange(1, 6, dtype=jnp.float32)
    return DiagonalTensorMap(index_space, data)


def test_diagonal_tensormap_exposes_shared_tensor_metadata():
    diagonal = _u1_diagonal()
    expanded = diagonal.to_tensor_map()

    assert diagonal.index_space == space(U1Irrep, {0: 2, 1: 3})
    assert isinstance(diagonal.domain, ProductSpace)
    assert isinstance(diagonal.codomain, ProductSpace)
    assert diagonal.domain.spaces == (diagonal.index_space,)
    assert diagonal.codomain.spaces == (diagonal.index_space,)
    assert diagonal.space == hom(diagonal.codomain, diagonal.domain)
    assert (diagonal.numout, diagonal.numin, diagonal.numind) == (1, 1, 2)
    assert diagonal.codomainind == (0,)
    assert diagonal.domainind == (1,)
    assert diagonal.allind == (0, 1)
    assert diagonal.ndim == 2
    assert diagonal.output_axes == (0,)
    assert diagonal.input_axes == (1,)
    assert diagonal.axes == (0, 1)
    assert diagonal.dim == expanded.dim
    assert diagonal.dims == expanded.dims
    assert diagonal.shape == expanded.shape
    assert diagonal.dtype == jnp.dtype(jnp.float32)
    assert diagonal.blocksectors == expanded.blocksectors
    assert diagonal.block_sectors == diagonal.blocksectors
    assert diagonal.fusiontrees == expanded.fusiontrees
    predicate = diagonal.is_diagonal()
    assert predicate.shape == ()
    assert predicate.dtype == jnp.dtype(jnp.bool_)
    assert bool(predicate)
    assert repr(diagonal) == (
        "DiagonalTensorMap(dims=(5, 5), blocks=2, dtype=float32)"
    )


def test_diagonal_tensormap_is_immutable_and_keeps_storage_leaf():
    data = jnp.arange(5, dtype=jnp.float32)
    diagonal = _u1_diagonal(data)

    assert diagonal.storage.data is data
    with pytest.raises(FrozenInstanceError):
        setattr(diagonal, "storage", diagonal.storage)
    with pytest.raises(FrozenInstanceError):
        setattr(diagonal, "index_space", diagonal.index_space)


def test_diagonal_tensormap_rejects_invalid_index_space_and_storage_shape():
    index_space = space(U1Irrep, {0: 2, 1: 3})

    with pytest.raises(TypeError, match="DiagonalTensorMap requires an ElementarySpace"):
        DiagonalTensorMap(
            hom(  # pyright: ignore[reportArgumentType]
                (index_space,),
                (index_space,),
            ),
            jnp.arange(5),
        )

    with pytest.raises(ValueError, match="expected.*5.*actual.*4"):
        DiagonalTensorMap(index_space, jnp.arange(4))

    with pytest.raises(ValueError, match="1D"):
        DiagonalTensorMap(index_space, jnp.zeros((5, 1)))


def test_diagonal_tensormap_diag_returns_sectorvector_view():
    diagonal = _u1_diagonal()

    values = diagonal.diag()

    assert isinstance(values, SectorVector)
    assert not hasattr(values, "space")
    assert values.sector_type == U1Irrep
    assert values.sectors == diagonal.index_space.sectors
    assert values.storage.data is diagonal.storage.data
    assert_allclose(values.block(0), diagonal.storage.data[0:2])
    assert_allclose(values.block(1), diagonal.storage.data[2:5])


def test_diagonal_tensormap_blocks_match_expanded_tensor_map():
    diagonal = _u1_diagonal()
    expanded = diagonal.to_tensor_map()

    assert diagonal.hasblock(0)
    assert diagonal.hasblock((1,))
    assert not diagonal.hasblock(2)
    assert tuple(sector for sector, _block in diagonal.blocks()) == (
        diagonal.blocksectors
    )
    for sector, block in diagonal.blocks():
        assert_allclose(block, expanded.block(sector))
        assert_allclose(diagonal.block(sector), expanded.block(sector))
    with pytest.raises(KeyError):
        diagonal.block(2)


@pytest.mark.parametrize(
    ("sector_type", "sector_dims"),
    [
        (U1Irrep, {0: 2, 1: 1}),
        (SU2Irrep, {0: 2, 2: 1}),
        (FermionParity, {0: 2, 1: 1}),
        (U1Irrep @ FermionParity, {(0, 0): 2, (1, 1): 1}),
    ],
    ids=["u1", "su2", "fermionic", "product-sector"],
)
def test_diagonal_tensormap_expansion_supports_sector_families(
    sector_type,
    sector_dims,
):
    index_space = space(sector_type, sector_dims)
    data = jnp.arange(1, 4, dtype=jnp.float32)
    diagonal = DiagonalTensorMap(index_space, data)
    expanded = diagonal.to_tensor_map()

    assert diagonal.space == expanded.space
    for sector, values in diagonal.diag().blocks():
        assert_allclose(expanded.block(sector), jnp.diag(values))


def test_diagonal_tensormap_arithmetic_preserves_diagonal_storage():
    left = _u1_diagonal()
    right = _u1_diagonal(jnp.arange(5, 0, -1, dtype=jnp.float32))

    results = (
        (-left, -left.storage.data),
        (left + right, left.storage.data + right.storage.data),
        (left - right, left.storage.data - right.storage.data),
        (2.5 * left, 2.5 * left.storage.data),
        (left * 2.5, 2.5 * left.storage.data),
        (left / 2.0, left.storage.data / 2.0),
        (left.zero_like(), jnp.zeros_like(left.storage.data)),
        (left.scale(3), 3 * left.storage.data),
        (left.add(right, alpha=2, beta=-1), 2 * left.storage.data - right.storage.data),
        (tensor0.zero_like(left), jnp.zeros_like(left.storage.data)),
        (tensor0.scale(left, 3), 3 * left.storage.data),
        (tensor0.add(left, right), left.storage.data + right.storage.data),
    )
    for result, expected in results:
        assert isinstance(result, DiagonalTensorMap)
        assert result.index_space == left.index_space
        assert_allclose(result.storage.data, expected)

    promoted = left + _u1_diagonal(left.storage.data.astype(jnp.complex64) * 1j)
    assert promoted.dtype == jnp.dtype(jnp.complex64)


def test_diagonal_tensormap_arithmetic_rejects_invalid_inputs_and_spaces():
    diagonal = _u1_diagonal()
    incompatible = DiagonalTensorMap(
        space(U1Irrep, {0: 1, 1: 4}),
        jnp.ones((5,), dtype=jnp.float32),
    )

    with pytest.raises(ValueError, match="spaces.*not compatible"):
        diagonal.add(incompatible)
    with pytest.raises(TypeError, match="scalar"):
        diagonal.scale(jnp.ones((2,)))
    with pytest.raises(TypeError, match="scalar"):
        _ = diagonal / jnp.ones((2,))


def test_diagonal_tensormap_components_and_adjoint_preserve_structure():
    data = jnp.arange(1, 6, dtype=jnp.float32) * jnp.asarray(
        1.0 + 2.0j,
        dtype=jnp.complex64,
    )
    diagonal = _u1_diagonal(data)

    real_part = tensor0.real(diagonal)
    imag_part = tensor0.imag(diagonal)
    adjoint = tensor0.adjoint(diagonal)
    complex_part = tensor0.complex(real_part)

    for result in (real_part, imag_part, adjoint, complex_part):
        assert isinstance(result, DiagonalTensorMap)
        assert result.index_space == diagonal.index_space
    assert_allclose(real_part.storage.data, jnp.real(data))
    assert_allclose(imag_part.storage.data, jnp.imag(data))
    assert_allclose(adjoint.storage.data, jnp.conj(data))
    assert complex_part.dtype == jnp.dtype(jnp.complex64)
    assert_allclose(complex_part.storage.data, jnp.real(data).astype(jnp.complex64))

    real_diagonal = _u1_diagonal()
    assert real_diagonal.real() is real_diagonal
    assert real_diagonal.adjoint() is real_diagonal
    assert_allclose(real_diagonal.imag().storage.data, jnp.zeros((5,)))


def test_diagonal_tensormap_norm_matches_expanded_weighted_norm():
    index_space = space(SU2Irrep, {0: 1, 2: 2})
    diagonal = DiagonalTensorMap(
        index_space,
        jnp.asarray([2.0, -3.0, 4.0], dtype=jnp.float32),
    )
    expanded = diagonal.to_tensor_map()

    for p in (1, 2, 3, jnp.inf):
        assert_allclose(diagonal.norm(p), expanded.norm(p))
        assert_allclose(tensor0.norm(diagonal, p), expanded.norm(p))
    assert diagonal.norm().dtype == jnp.dtype(jnp.float32)

    with pytest.raises(ValueError, match="positive finite p or inf"):
        diagonal.norm(0)


def test_diagonal_tensormap_inverse_and_pseudoinverse_zero_cutoff_semantics():
    diagonal = _u1_diagonal(
        jnp.asarray([4.0, 0.1, 0.0, -2.0, 8.0], dtype=jnp.float32),
    )

    inverse = diagonal.inverse()
    pseudoinverse = diagonal.pseudoinverse(atol=0.2, rtol=0.0)

    assert isinstance(inverse, DiagonalTensorMap)
    assert bool(jnp.isinf(inverse.storage.data[2]))
    assert_allclose(
        pseudoinverse.storage.data,
        jnp.asarray([0.25, 0.0, 0.0, -0.5, 0.125], dtype=jnp.float32),
    )
    assert_allclose(
        diagonal.pseudoinverse(atol=0.0, rtol=0.0).storage.data,
        jnp.asarray([0.25, 10.0, 0.0, -0.5, 0.125], dtype=jnp.float32),
    )

    with pytest.raises(ValueError, match="atol.*non-negative"):
        diagonal.pseudoinverse(atol=-1.0)
    with pytest.raises(ValueError, match="rtol.*non-negative"):
        diagonal.pseudoinverse(rtol=-1.0)


def test_diagonal_tensormap_supports_all_composition_combinations():
    diagonal = _u1_diagonal()
    expanded = diagonal.to_tensor_map()
    expected = expanded @ expanded

    diagonal_result = diagonal @ diagonal
    left_mixed = diagonal @ expanded
    right_mixed = expanded @ diagonal

    assert isinstance(diagonal_result, DiagonalTensorMap)
    assert_tensormap_blocks_allclose(diagonal_result.to_tensor_map(), expected)
    assert_tensormap_blocks_allclose(left_mixed, expected)
    assert_tensormap_blocks_allclose(right_mixed, expected)


def test_diagonal_tensormap_comparisons_use_mathematical_values():
    diagonal = _u1_diagonal()
    expanded = diagonal.to_tensor_map()
    promoted = expanded.astype(jnp.complex64)
    offdiagonal = TensorMap(
        expanded.space,
        expanded.storage.data.at[1].set(0.25),
    )

    for left, right in ((diagonal, expanded), (expanded, diagonal)):
        assert bool(tensor_api.equal(left, right))
        assert bool(tensor_api.allclose(left, right, rtol=0.0, atol=0.0))

    assert not bool(tensor_api.equal(diagonal, promoted))
    assert bool(tensor_api.allclose(diagonal, promoted, rtol=0.0, atol=0.0))
    assert not bool(tensor_api.equal(diagonal, offdiagonal))
    assert not bool(tensor_api.allclose(diagonal, offdiagonal, rtol=0.0, atol=0.1))
    assert bool(tensor_api.is_diagonal(diagonal))


def test_diagonal_tensormap_composition_rejects_incompatible_spaces():
    diagonal = _u1_diagonal()
    incompatible = DiagonalTensorMap(
        space(U1Irrep, {0: 1}),
        jnp.ones((1,), dtype=jnp.float32),
    )

    with pytest.raises(ValueError, match="spaces are not composable"):
        _ = diagonal @ incompatible
    with pytest.raises(ValueError, match="spaces are not composable"):
        _ = diagonal @ incompatible.to_tensor_map()
    with pytest.raises(ValueError, match="spaces are not composable"):
        _ = diagonal.to_tensor_map() @ incompatible


def test_diagonal_tensormap_empty_space_operations_are_well_defined():
    diagonal = DiagonalTensorMap(
        space(U1Irrep, {}),
        jnp.zeros((0,), dtype=jnp.float32),
    )

    assert diagonal.blocks() == ()
    assert diagonal.to_tensor_map().storage.data.shape == (0,)
    assert_allclose(diagonal.norm(), jnp.asarray(0.0, dtype=jnp.float32))
    assert_allclose(diagonal.norm(jnp.inf), jnp.asarray(0.0, dtype=jnp.float32))
    assert diagonal.inverse().storage.data.shape == (0,)
    assert diagonal.pseudoinverse().storage.data.shape == (0,)


def test_diagonal_tensormap_pytree_children_are_storage_data():
    diagonal = _u1_diagonal()

    leaves, treedef = jax.tree_util.tree_flatten(diagonal)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)

    assert len(leaves) == 1
    assert leaves[0] is diagonal.storage.data
    assert isinstance(rebuilt, DiagonalTensorMap)
    assert rebuilt.index_space == diagonal.index_space
    assert rebuilt.storage.data is diagonal.storage.data


def test_diagonal_tensormap_arithmetic_composition_and_pseudoinverse_jit():
    diagonal = _u1_diagonal()
    expanded = diagonal.to_tensor_map()

    arithmetic = jax.jit(lambda value: (2 * value - value).adjoint())(diagonal)
    diagonal_product = jax.jit(lambda left, right: left @ right)(diagonal, diagonal)
    mixed_product = jax.jit(lambda left, right: left @ right)(expanded, diagonal)
    pseudoinverse = jax.jit(
        lambda value: value.pseudoinverse(atol=2.0, rtol=0.0),
    )(diagonal)

    assert_allclose(arithmetic.storage.data, diagonal.storage.data)
    assert_allclose(diagonal_product.storage.data, diagonal.storage.data**2)
    assert_tensormap_blocks_allclose(
        mixed_product,
        expanded @ diagonal.to_tensor_map(),
    )
    assert_allclose(
        pseudoinverse.storage.data,
        jnp.asarray([0.0, 0.0, 1 / 3, 1 / 4, 1 / 5], dtype=jnp.float32),
    )


def test_diagonal_tensormap_numerical_paths_are_differentiable():
    diagonal = _u1_diagonal()
    expanded = diagonal.to_tensor_map()

    norm_gradient = jax.grad(lambda value: value.norm())(diagonal)
    composition_gradient = jax.grad(
        lambda value: jnp.sum((expanded @ value).storage.data),
    )(diagonal)
    inverse_gradient = jax.grad(
        lambda value: jnp.sum(value.inverse().storage.data),
    )(diagonal)
    pseudoinverse_gradient = jax.grad(
        lambda value: jnp.sum(
            value.pseudoinverse(atol=0.0, rtol=0.0).storage.data,
        ),
    )(diagonal)

    assert isinstance(norm_gradient, DiagonalTensorMap)
    assert isinstance(composition_gradient, DiagonalTensorMap)
    assert isinstance(inverse_gradient, DiagonalTensorMap)
    assert isinstance(pseudoinverse_gradient, DiagonalTensorMap)
    assert bool(jnp.all(jnp.isfinite(norm_gradient.storage.data)))
    assert bool(jnp.all(jnp.isfinite(composition_gradient.storage.data)))
    assert_allclose(inverse_gradient.storage.data, -1 / diagonal.storage.data**2)
    assert_allclose(
        pseudoinverse_gradient.storage.data,
        -1 / diagonal.storage.data**2,
    )

    with_zero = _u1_diagonal(
        jnp.asarray([4.0, 0.1, 0.0, -2.0, 8.0], dtype=jnp.float32),
    )
    cutoff_gradient = jax.jit(
        jax.grad(
            lambda value: jnp.sum(
                value.pseudoinverse(atol=0.2, rtol=0.0).storage.data,
            ),
        ),
    )(with_zero)
    assert_allclose(
        cutoff_gradient.storage.data,
        jnp.asarray([-1 / 16, 0.0, 0.0, -1 / 4, -1 / 64], dtype=jnp.float32),
    )
