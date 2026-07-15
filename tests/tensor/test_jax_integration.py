import jax
import jax.numpy as jnp
import pytest

from tensor0 import (
    FermionParity,
    SU2Irrep,
    TensorMap,
    U1Irrep,
    contract,
    flip,
    hom,
    idx,
    ncon,
    permute,
    repartition,
    scalar,
    space,
    tensorcontract,
    tensortrace,
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


def _partial_trace_tensor():
    open_factor = space(U1Irrep, {0: 2})
    traced = space(U1Irrep, {0: 3})
    target = hom((open_factor, traced), (open_factor, traced))
    return TensorMap(target, float_data_for(target))


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


def test_jitted_sector_indexing_and_lazy_subblock_iteration_match_eager():
    factor = space(U1Irrep, {0: 1, 1: 2})
    target = hom((factor,), (factor,))
    tensor = TensorMap(target, float_data_for(target))

    def read(value):
        total = jnp.asarray(0, dtype=value.storage.data.dtype)
        for _pair, block in value.subblocks():
            total = total + jnp.sum(block)
        return value[1, -1], total

    indexed, total = jax.jit(read)(tensor)
    expected_indexed, expected_total = read(tensor)

    assert_allclose(indexed, expected_indexed)
    assert_allclose(total, expected_total)

    pair = tensor.fusiontrees[1]

    def sector_objective(data):
        block = TensorMap(target, data)[1, -1]
        return jnp.sum(block**2)

    def tree_objective(data):
        block = TensorMap(target, data)[pair]
        return jnp.sum(block**2)

    assert_allclose(
        jax.jit(jax.grad(sector_objective))(tensor.storage.data),
        jax.grad(tree_objective)(tensor.storage.data),
    )


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


def test_jitted_tensortrace_matches_eager_with_static_metadata():
    tensor = _partial_trace_tensor()
    trace_count = 0

    @jax.jit
    def trace(value):
        nonlocal trace_count
        trace_count += 1
        return tensortrace(
            value,
            axes=((1,), (3,)),
            output=((0,), (2,)),
        )

    result = trace(tensor)
    expected = tensortrace(
        tensor,
        axes=((1,), (3,)),
        output=((0,), (2,)),
    )

    updated = TensorMap(tensor.space, tensor.storage.data * 2.0 + 1.0)
    updated_result = trace(updated)
    updated_expected = tensortrace(
        updated,
        axes=((1,), (3,)),
        output=((0,), (2,)),
    )

    assert result.space == expected.space
    assert_allclose(result.storage.data, expected.storage.data)
    assert updated_result.space == updated_expected.space
    assert_allclose(updated_result.storage.data, updated_expected.storage.data)
    assert trace_count == 1


def test_jitted_network_contract_reuses_static_labels_for_new_storage():
    left, right = _u1_composition_tensors()
    trace_count = 0

    @jax.jit
    def network(a, b):
        nonlocal trace_count
        trace_count += 1
        return contract(
            (a, "a,x"),
            (b, "x,b"),
            output="a;b",
        )

    result = network(left, right)
    expected = contract(
        (left, "a,x"),
        (right, "x,b"),
        output=("a", "b"),
    )
    updated = TensorMap(left.space, left.storage.data * 2.0 + 1.0)
    updated_result = network(updated, right)
    updated_expected = contract(
        (updated, "a,x"),
        (right, "x,b"),
        output=("a", "b"),
    )

    assert result.space == expected.space
    assert_allclose(result.storage.data, expected.storage.data)
    assert updated_result.space == updated_expected.space
    assert_allclose(updated_result.storage.data, updated_expected.storage.data)
    assert trace_count == 1


def test_jitted_ncon_reuses_static_labels_for_new_storage():
    left, right = _u1_composition_tensors()
    trace_count = 0

    @jax.jit
    def network(a, b):
        nonlocal trace_count
        trace_count += 1
        return ncon(
            (a, b),
            ((-1, 1), (1, -2)),
        )

    result = network(left, right)
    expected = ncon(
        (left, right),
        ((-1, 1), (1, -2)),
    )
    updated = TensorMap(left.space, left.storage.data * 2.0 + 1.0)
    updated_result = network(updated, right)
    updated_expected = ncon(
        (updated, right),
        ((-1, 1), (1, -2)),
    )

    assert result.space == expected.space
    assert_allclose(result.storage.data, expected.storage.data)
    assert updated_result.space == updated_expected.space
    assert_allclose(updated_result.storage.data, updated_expected.storage.data)
    assert trace_count == 1


def test_jitted_network_contract_retraces_for_labels_output_or_homspace():
    tensor = TensorMap(_u1_hom(), _u1_data())
    changed_storage = TensorMap(tensor.space, _u1_data() + 10)
    changed_space = TensorMap(
        _u1_hom_same_total_dim_with_different_metadata(),
        _u1_data(),
    )
    trace_count = 0

    def network(value, labels, output):
        nonlocal trace_count
        trace_count += 1
        return contract(idx(value, labels), output=output)

    compiled = jax.jit(network, static_argnames=("labels", "output"))
    compiled(tensor, "a,b", ("a", "b"))
    compiled(changed_storage, "a,b", ("a", "b"))
    compiled(tensor, "a,b", ("b", "a"))
    compiled(tensor, "b,a", ("a", "b"))
    compiled(changed_space, "a,b", ("a", "b"))

    assert trace_count == 4


def test_value_and_grad_through_three_operand_network():
    v = space(U1Irrep, {0: 2})
    x = space(U1Irrep, {0: 3})
    y = space(U1Irrep, {0: 4})
    w = space(U1Irrep, {0: 2})
    tensors = (
        TensorMap(hom((v,), (x,)), float_data_for(hom((v,), (x,)))),
        TensorMap(hom((x,), (y,)), float_data_for(hom((x,), (y,)))),
        TensorMap(hom((y,), (w,)), float_data_for(hom((y,), (w,)))),
    )

    def named_loss(a, b, c):
        result = contract(
            idx(a, "v,x"),
            idx(b, "x,y"),
            idx(c, "y,w"),
            output=("v", "w"),
            order=("y", "x"),
        )
        return jnp.sum(result.storage.data**2)

    def integer_loss(a, b, c):
        result = ncon(
            (a, b, c),
            ((-1, 1), (1, 2), (2, -2)),
            order=(2, 1),
        )
        return jnp.sum(result.storage.data**2)

    eager = jax.value_and_grad(named_loss, argnums=(0, 1, 2))(*tensors)
    compiled = jax.jit(
        jax.value_and_grad(named_loss, argnums=(0, 1, 2))
    )(*tensors)
    integer = jax.jit(
        jax.value_and_grad(integer_loss, argnums=(0, 1, 2))
    )(*tensors)

    assert_allclose(compiled[0], eager[0])
    for actual, expected in zip(compiled[1], eager[1], strict=True):
        assert actual.space == expected.space
        assert_allclose(actual.storage.data, expected.storage.data)
    assert_allclose(integer[0], eager[0])
    for actual, expected in zip(integer[1], eager[1], strict=True):
        assert actual.space == expected.space
        assert_allclose(actual.storage.data, expected.storage.data)


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


def test_value_and_grad_through_jitted_tensortrace_storage_leaf():
    factor = space(U1Irrep, {0: 2})
    target = hom((factor,), (factor,))
    tensor = TensorMap(
        target,
        jnp.asarray([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32),
    )

    @jax.jit
    def loss(value):
        result = tensortrace(
            value,
            axes=((0,), (1,)),
            output=((), ()),
        )
        traced = scalar(result)
        return traced * traced

    value, gradient = jax.value_and_grad(loss)(tensor)
    gradient_leaves, _gradient_treedef = jax.tree_util.tree_flatten(gradient)

    assert_allclose(value, jnp.asarray(25.0, dtype=jnp.float32))
    assert isinstance(gradient, TensorMap)
    assert gradient.space == target
    assert len(gradient_leaves) == 1
    assert gradient_leaves[0] is gradient.storage.data
    assert_allclose(
        gradient.storage.data,
        jnp.asarray([10.0, 0.0, 0.0, 10.0], dtype=jnp.float32),
    )


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


def test_jitted_grad_through_su2_nontrivial_basis_transform():
    half = space(SU2Irrep, {1: 1})
    target = hom((half, half, half, half), ())
    data = jnp.asarray([1.0, 2.0], dtype=jnp.float32)

    def loss(storage):
        result = permute(
            TensorMap(target, storage),
            ((1, 2, 3), (0,)),
        )
        return jnp.sum(result.storage.data)

    eager = jax.grad(loss)(data)
    compiled = jax.jit(jax.grad(loss))(data)

    assert_allclose(compiled, eager)
    assert not bool(jnp.allclose(compiled, jnp.ones_like(compiled)))


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


def test_jit_flip_matches_eager_with_static_destination_metadata():
    factor = space(FermionParity, {0: 1, 1: 1})
    target = hom((factor, factor), (factor,))
    tensor = TensorMap(target, float_data_for(target))

    _assert_jitted_transform_matches_eager(
        tensor,
        lambda value: flip(value, (0, 2)),
    )


def test_jitted_grad_through_flip_preserves_column_z_isomorphism_sign():
    factor = space(FermionParity, {0: 1, 1: 1})
    target = hom((factor,), (factor,))
    tensor = TensorMap(target, jnp.array([2.0, 3.0]))
    weights = jnp.array([5.0, 7.0])

    def loss(value):
        return jnp.sum(flip(value, 1).storage.data * weights)

    gradient = jax.jit(jax.grad(loss))(tensor)

    assert gradient.space == target
    assert_allclose(gradient.storage.data, jnp.array([5.0, -7.0]))
