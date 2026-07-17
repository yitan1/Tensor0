import jax
import jax.numpy as jnp
import pytest

from tensor0 import (
    FermionParity,
    SU2Irrep,
    TensorMap,
    U1Irrep,
    U1SU2Irrep,
    _native,
    from_dense,
    fuse,
    hom,
    space,
    zero_space,
)
import tensor0.tensor as tensor_api
import tensor0.tensor.constructors as constructors
from tensor0.tensor.constructors import identity, isometry, isomorphism, unitary
from tests.cases import assert_allclose, float_data_for


def _product(*factors):
    return hom(factors, ()).codomain


def _assert_tensors_allclose(left, right):
    assert left.space == right.space
    assert bool(tensor_api.allclose(left, right, rtol=1e-5, atol=1e-6))


def test_identity_is_neutral_for_elementary_and_product_spaces():
    left = space(U1Irrep, {0: 2, 1: 3})
    right = space(U1Irrep, {0: 5, 1: 7})
    target = hom((left,), (right,))
    tensor = TensorMap(target, float_data_for(target))

    _assert_tensors_allclose(identity(left) @ tensor, tensor)
    _assert_tensors_allclose(tensor @ identity(right), tensor)

    product = _product(left, right)
    product_target = hom(product, product)
    product_tensor = TensorMap(product_target, float_data_for(product_target))
    product_identity = identity(product)

    _assert_tensors_allclose(product_identity @ product_tensor, product_tensor)
    _assert_tensors_allclose(product_tensor @ product_identity, product_tensor)


@pytest.mark.parametrize(
    ("sector_type", "sector_dims"),
    [
        (U1Irrep, {0: 2, 1: 3}),
        (SU2Irrep, {0: 2, 1: 3}),
        (FermionParity, {0: 2, 1: 3}),
        (U1SU2Irrep, {(0, 0): 2, (1, 1): 3}),
    ],
    ids=["u1", "su2", "fermion", "product-sector"],
)
def test_identity_uses_canonical_block_bases_for_supported_family_categories(
    sector_type,
    sector_dims,
):
    factor = space(sector_type, sector_dims)
    result = identity(factor, dtype=jnp.complex64)

    assert result.dtype == jnp.dtype(jnp.complex64)
    for _coupled, block in result.blocks():
        assert_allclose(block, jnp.eye(block.shape[0], dtype=jnp.complex64))


def test_product_to_fused_isomorphism_and_unitary_follow_fusion_tree_gauge():
    half = space(SU2Irrep, {1: 1})
    product = _product(half, half)
    fused = fuse(product)

    forward = isomorphism(fused, product)
    reverse = isomorphism(product, fused)
    _assert_tensors_allclose(forward.adjoint(), reverse)
    _assert_tensors_allclose(forward @ reverse, identity(fused))
    _assert_tensors_allclose(reverse @ forward, identity(product))

    canonical_coefficients = tuple(
        jnp.asarray(_native.fusiontree_pair_tensor(row, col))
        for row, col in forward.fusiontrees
    )
    assert_allclose(forward.to_dense(), jnp.concatenate(canonical_coefficients, axis=0))

    split = unitary(product, fused, dtype=jnp.complex64)
    merge = unitary(fused, product, dtype=jnp.complex64)
    _assert_tensors_allclose(split.adjoint(), merge)
    _assert_tensors_allclose(split @ merge, identity(product, dtype=jnp.complex64))
    _assert_tensors_allclose(merge @ split, identity(fused, dtype=jnp.complex64))


def test_dual_product_identity_and_fused_maps_support_dense_roundtrip():
    half = space(SU2Irrep, {1: 1})
    product = _product(half, half.dual())
    fused = fuse(product)
    product_identity = identity(product)
    merge = isomorphism(fused, product)
    split = isomorphism(product, fused)

    dense_identity = product_identity.to_dense()
    dense_merge = merge.to_dense()
    dense_split = split.to_dense()

    assert_allclose(
        dense_identity.reshape((4, 4)),
        jnp.eye(4, dtype=product_identity.dtype),
    )
    assert_allclose(dense_split, jnp.transpose(dense_merge, (1, 2, 0)))
    for tensor, dense in (
        (product_identity, dense_identity),
        (merge, dense_merge),
        (split, dense_split),
    ):
        _assert_tensors_allclose(from_dense(tensor.space, dense), tensor)


def test_isomorphism_supports_equal_visible_spaces_with_different_dual_metadata():
    domain = space(U1Irrep, {1: 2, -1: 3}).dual()
    codomain = space(U1Irrep, {-1: 2, 1: 3})

    forward = isomorphism(codomain, domain)
    reverse = isomorphism(domain, codomain)

    _assert_tensors_allclose(forward.adjoint(), reverse)
    _assert_tensors_allclose(forward @ reverse, identity(codomain))
    _assert_tensors_allclose(reverse @ forward, identity(domain))


def test_isometry_has_left_inverse_and_projector_for_product_domain():
    half = space(SU2Irrep, {1: 1})
    domain = _product(half, half)
    codomain = space(SU2Irrep, {0: 2, 2: 3})

    result = isometry(codomain, domain, dtype=jnp.float32)
    left_inverse = result.adjoint() @ result
    projector = result @ result.adjoint()

    _assert_tensors_allclose(left_inverse, identity(domain, dtype=jnp.float32))
    _assert_tensors_allclose(projector @ projector, projector)
    assert not bool(
        tensor_api.allclose(
            projector,
            identity(codomain, dtype=jnp.float32),
            rtol=1e-5,
            atol=1e-6,
        ),
    )


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.complex64])
def test_canonical_constructors_preserve_explicit_dtype(dtype):
    factor = space(U1Irrep, {0: 2})

    results = (
        identity(factor, dtype=dtype),
        isomorphism(factor, factor, dtype=dtype),
        unitary(factor, factor, dtype=dtype),
        isometry(factor, factor, dtype=dtype),
    )

    assert all(result.dtype == jnp.dtype(dtype) for result in results)


def test_canonical_constructor_default_dtype_matches_jax_default():
    factor = space(U1Irrep, {0: 1})

    assert identity(factor).dtype == jnp.zeros(()).dtype


def test_identity_supports_typed_scalar_and_zero_spaces():
    scalar_product = hom((), (), sector_type=U1Irrep).codomain
    scalar_identity = identity(scalar_product)
    zero_identity = identity(zero_space(U1Irrep))

    assert_allclose(scalar_identity.scalar(), jnp.asarray(1.0))
    assert scalar_identity.storage.data.shape == (1,)
    assert zero_identity.storage.data.shape == (0,)
    assert zero_identity.blocksectors == ()


@pytest.mark.parametrize("constructor", [isomorphism, unitary])
def test_nonisomorphic_constructors_fail_before_block_allocation(monkeypatch, constructor):
    codomain = space(U1Irrep, {0: 2})
    domain = space(U1Irrep, {0: 3})

    def fail_allocation(*_args, **_kwargs):
        raise AssertionError("numerical allocation must follow space validation")

    monkeypatch.setattr(constructors.jnp, "eye", fail_allocation)
    with pytest.raises(ValueError, match="isomorphic codomain and domain"):
        constructor(codomain, domain)


def test_nonmonomorphic_isometry_fails_before_block_allocation(monkeypatch):
    codomain = space(U1Irrep, {0: 2})
    domain = space(U1Irrep, {0: 3})

    def fail_allocation(*_args, **_kwargs):
        raise AssertionError("numerical allocation must follow space validation")

    monkeypatch.setattr(constructors.jnp, "eye", fail_allocation)
    with pytest.raises(ValueError, match="domain to be monomorphic"):
        isometry(codomain, domain)


def test_constructor_inputs_reject_nonspaces_and_mismatched_families():
    u1 = space(U1Irrep, {0: 1})
    su2 = space(SU2Irrep, {0: 1})

    with pytest.raises(TypeError, match="identity.*ElementarySpace or ProductSpace"):
        identity(object())  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError, match="isomorphic codomain and domain"):
        isomorphism(u1, su2)
    with pytest.raises(ValueError, match="domain to be monomorphic"):
        isometry(u1, su2)


def test_constructor_composition_is_jittable_and_differentiable():
    factor = space(U1Irrep, {0: 2, 1: 1})
    target = hom((factor,), (factor,))
    data = float_data_for(target)

    @jax.jit
    def apply_identity(storage):
        tensor = TensorMap(target, storage)
        return identity(factor, dtype=storage.dtype) @ tensor

    result = apply_identity(data)
    assert_allclose(result.storage.data, data)

    def objective(storage):
        result = apply_identity(storage)
        return jnp.sum(result.storage.data**2)

    assert_allclose(jax.grad(objective)(data), 2 * data)
