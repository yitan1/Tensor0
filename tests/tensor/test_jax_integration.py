import jax
import jax.numpy as jnp
import pytest

from tensor0 import (
    FermionParity,
    SU2Irrep,
    TensorMap,
    U1Irrep,
    hom,
    permute,
    repartition,
    space,
    tensorcontract,
    twist,
)
from tests.cases import assert_allclose, float_data_for


def _u1_hom():
    v = space(U1Irrep, {0: 2, 1: 3})
    return hom((v,), (v,))


def _u1_hom_same_total_dim_with_different_metadata():
    v = space(U1Irrep, {0: 2, 2: 3})
    return hom((v,), (v,))


def _u1_data():
    return jnp.arange(13)


def _u1_composition_tensors():
    v = space(U1Irrep, {0: 2, 1: 3})
    w = space(U1Irrep, {0: 5, 1: 7})
    x = space(U1Irrep, {0: 11, 1: 13})
    a_space = hom((v,), (w,))
    b_space = hom((w,), (x,))
    return (
        TensorMap(a_space, float_data_for(a_space)),
        TensorMap(b_space, float_data_for(b_space)),
    )


def _partial_contraction_tensors():
    a = space(U1Irrep, {0: 2})
    x = space(U1Irrep, {0: 3})
    c = space(U1Irrep, {0: 4})
    b = space(U1Irrep, {0: 5})
    d = space(U1Irrep, {0: 6})
    left_space = hom((a, x), (c,))
    right_space = hom((x.dual(), b), (d,))
    return (
        TensorMap(left_space, float_data_for(left_space)),
        TensorMap(right_space, float_data_for(right_space)),
    )


def _assert_jitted_transform_matches_eager(tensor, transform):
    result = jax.jit(transform)(tensor)
    expected = transform(tensor)

    assert isinstance(result, TensorMap)
    assert result.space == expected.space
    assert_allclose(result.storage.data, expected.storage.data)


def test_tensormap_pytree_roundtrip_preserves_static_space_metadata():
    tensor = TensorMap(_u1_hom(), _u1_data())

    leaves, treedef = jax.tree_util.tree_flatten(tensor)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    new_data = _u1_data() + 20
    rebuilt_with_new_data = jax.tree_util.tree_unflatten(treedef, (new_data,))

    assert len(leaves) == 1
    assert leaves[0] is tensor.storage.data
    assert isinstance(rebuilt, TensorMap)
    assert rebuilt.space == tensor.space
    assert rebuilt.storage.data is tensor.storage.data
    assert isinstance(rebuilt_with_new_data, TensorMap)
    assert rebuilt_with_new_data.space == tensor.space
    assert rebuilt_with_new_data.storage.data is new_data

    with pytest.raises(ValueError, match="storage data length mismatch"):
        jax.tree_util.tree_unflatten(treedef, (jnp.arange(12),))


def test_tensormap_pytree_aux_uses_static_space_metadata():
    first = TensorMap(_u1_hom(), _u1_data())
    second = TensorMap(_u1_hom(), _u1_data() + 100)
    different_metadata = TensorMap(
        _u1_hom_same_total_dim_with_different_metadata(),
        _u1_data(),
    )

    _first_leaves, first_treedef = jax.tree_util.tree_flatten(first)
    _second_leaves, second_treedef = jax.tree_util.tree_flatten(second)
    _different_leaves, different_treedef = jax.tree_util.tree_flatten(
        different_metadata,
    )

    assert first_treedef == second_treedef
    assert first_treedef != different_treedef


def test_jitted_composition_uses_storage_as_dynamic_leaf():
    left, right = _u1_composition_tensors()
    same_space_left = TensorMap(left.space, left.storage.data * 2.0 + 1.0)

    @jax.jit
    def compose(a, b):
        return a @ b

    left_leaves, left_treedef = jax.tree_util.tree_flatten(left)
    same_leaves, same_treedef = jax.tree_util.tree_flatten(same_space_left)
    result = compose(same_space_left, right)
    expected = same_space_left @ right

    assert left_treedef == same_treedef
    assert len(left_leaves) == 1
    assert len(same_leaves) == 1
    assert same_leaves[0] is same_space_left.storage.data
    assert result.space == expected.space
    assert_allclose(result.storage.data, expected.storage.data)


def test_jitted_tensorcontract_matches_eager_with_static_metadata():
    left, right = _partial_contraction_tensors()
    axes = ((1,), (0,))
    output = (((0, 0), (1, 1)), ((0, 2), (1, 2)))
    trace_count = 0

    @jax.jit
    def contract(a, b):
        nonlocal trace_count
        trace_count += 1
        return tensorcontract(a, b, axes=axes, output=output)

    result = contract(left, right)
    expected = tensorcontract(left, right, axes=axes, output=output)

    updated_left = TensorMap(left.space, left.storage.data * 2.0 + 1.0)
    updated_result = contract(updated_left, right)
    updated_expected = tensorcontract(
        updated_left,
        right,
        axes=axes,
        output=output,
    )

    assert result.space == expected.space
    assert_allclose(result.storage.data, expected.storage.data)
    assert updated_result.space == updated_expected.space
    assert_allclose(updated_result.storage.data, updated_expected.storage.data)
    assert trace_count == 1


def test_value_and_grad_through_jitted_composition_loss():
    left, right = _u1_composition_tensors()
    right_blocks = dict(right.blocks())

    @jax.jit
    def loss(a):
        composed = a @ right
        return jnp.sum(composed.storage.data * composed.storage.data)

    value, gradient = jax.value_and_grad(loss)(left)

    assert value.shape == ()
    assert isinstance(gradient, TensorMap)
    assert gradient.space == left.space
    for coupled, left_block in left.blocks():
        right_block = right_blocks[coupled]
        expected = 2.0 * (left_block @ right_block) @ right_block.T
        assert_allclose(gradient.block(coupled), expected)


def test_value_and_grad_through_jitted_tensorcontract_matches_composition_rule():
    left, right = _u1_composition_tensors()
    right_blocks = dict(right.blocks())

    @jax.jit
    def loss(a):
        result = tensorcontract(
            a,
            right,
            axes=((1,), (0,)),
            output=(((0, 0),), ((1, 1),)),
        )
        return jnp.sum(result.storage.data * result.storage.data)

    value, gradient = jax.value_and_grad(loss)(left)

    assert value.shape == ()
    assert isinstance(gradient, TensorMap)
    assert gradient.space == left.space
    for coupled, left_block in left.blocks():
        right_block = right_blocks[coupled]
        expected = 2.0 * (left_block @ right_block) @ right_block.T
        assert_allclose(gradient.block(coupled), expected)


def test_grad_through_jitted_composition_loss_handles_missing_middle_sector():
    v = space(U1Irrep, {0: 2, 1: 3})
    w = space(U1Irrep, {0: 5})
    x = space(U1Irrep, {0: 7, 1: 11})
    left_space = hom((v,), (w,))
    right_space = hom((w,), (x,))
    left = TensorMap(left_space, float_data_for(left_space))
    right = TensorMap(right_space, float_data_for(right_space))

    @jax.jit
    def loss(a):
        composed = a @ right
        return jnp.sum(composed.storage.data * composed.storage.data)

    gradient = jax.grad(loss)(left)
    expected_value = jnp.sum((left.block(0) @ right.block(0)) ** 2)

    assert isinstance(gradient, TensorMap)
    assert gradient.space == left.space
    assert tuple(coupled for coupled, _block in gradient.blocks()) == ((0,),)
    assert_allclose(
        jnp.asarray(loss(left)).reshape(()),
        jnp.asarray(expected_value).reshape(()),
    )
    assert_allclose(
        gradient.block(0),
        2.0 * (left.block(0) @ right.block(0)) @ right.block(0).T,
    )


def test_jit_repartition_matches_eager_transform():
    v = space(U1Irrep, {0: 2})
    w = space(U1Irrep, {0: 3})
    x = space(U1Irrep, {0: 5})
    h = hom((v,), (w, x))
    tensor = TensorMap(h, float_data_for(h))

    _assert_jitted_transform_matches_eager(tensor, lambda value: repartition(value, 2))


def test_jit_su2_permute_matches_eager_transform():
    half = space(SU2Irrep, {1: 1})
    h = hom((half, half, half), (half,))
    tensor = TensorMap(h, float_data_for(h))

    _assert_jitted_transform_matches_eager(
        tensor,
        lambda value: permute(value, ((1, 2), (0, 3))),
    )


def test_jitted_grad_through_fermion_odd_swap_phase():
    odd_a = space(FermionParity, {1: 1})
    odd_b = space(FermionParity, {1: 1})
    h = hom((odd_a, odd_b), ())
    tensor = TensorMap(h, jnp.array([2.0], dtype=jnp.float32))

    def loss(value):
        swapped = permute(value, ((1, 0), ()))
        return jnp.sum(swapped.storage.data)

    gradient = jax.jit(jax.grad(loss))(tensor)

    assert isinstance(gradient, TensorMap)
    assert gradient.space == tensor.space
    assert_allclose(gradient.storage.data, jnp.array([-1.0], dtype=jnp.float32))


def test_grad_through_twist_preserves_fermionic_subblock_sign():
    factor = space(FermionParity, {0: 1, 1: 1})
    target = hom((factor,), (factor,))
    tensor = TensorMap(target, jnp.array([2.0, 3.0]))
    weights = jnp.array([5.0, 7.0])

    def loss(value):
        return jnp.sum(twist(value, 0).storage.data * weights)

    gradient = jax.jit(jax.grad(loss))(tensor)

    assert gradient.space == target
    assert_allclose(gradient.storage.data, jnp.array([5.0, -7.0]))
