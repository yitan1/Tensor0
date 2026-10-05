"""Selected physical contraction layouts and their mathematical output ordering."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tensor0.operations.contractions.primitives as primitives
from tensor0 import (
    FermionParity, SU2Irrep, TensorMap, U1Irrep, U1SU2Irrep,
    Z2Irrep, Z3Irrep, Z4Irrep, hom, space, tensorcontract, to_dense,
)
from tensor0.structure import get_degeneracystructure


def _value(target, seed):
    count = get_degeneracystructure(target).total_dim
    x = jnp.arange(count, dtype=jnp.float32) + seed
    return TensorMap(target, (jnp.sin(x * .17) + 1j * jnp.cos(x * .13)).astype(jnp.complex64))


@pytest.mark.parametrize("family,charges", [
    (Z2Irrep, {0: 2, 1: 2}),
    (U1Irrep, {0: 2, 1: 2}),
    (SU2Irrep, {0: 1, 1: 1}),
    (U1SU2Irrep, {(0, 0): 1, (0, 1): 1}),
])
def test_selected_reverse_matches_independent_dense_product_and_reduces_copies(
    monkeypatch, family, charges,
):
    factor = space(family, charges)
    target = hom((factor,), (factor,))
    left, right = _value(target, 1), _value(target, 3)
    axes = ((0,), (1,))
    output = (((1, 0),), ((0, 1),))
    plan = primitives._select_contraction_plan(
        (target, target), axes, ((1,), (0,)), ((1,), (0,)), True,
    )
    assert plan.reversed_operands
    assert plan.copy_cost == 0
    original_permute = primitives.permute
    materialized = []

    def recording(value, permutation):
        if not primitives._is_identity_permutation(value.space, *permutation):
            materialized.append(permutation)
        return original_permute(value, permutation)

    monkeypatch.setattr(primitives, "permute", recording)
    result = tensorcontract(left, right, axes=axes, output=output)
    assert materialized == []
    expected = np.einsum(
        "ki,jk->ji", np.asarray(to_dense(left)), np.asarray(to_dense(right)),
    )
    np.testing.assert_allclose(np.asarray(to_dense(result)), expected, rtol=2e-5, atol=2e-5)

    select = primitives._select_contraction_plan

    def nonreversed(spaces, contracted, opened, destination, allow_reversal):
        return select(spaces, contracted, opened, destination, False)

    monkeypatch.setattr(primitives, "_select_contraction_plan", nonreversed)
    materialized.clear()
    baseline = tensorcontract(left, right, axes=axes, output=output)
    assert len(materialized) == 3
    np.testing.assert_allclose(
        np.asarray(result.storage.data), np.asarray(baseline.storage.data),
        rtol=2e-5, atol=2e-5,
    )

    def objective(a, b):
        value = tensorcontract(TensorMap(target, a), TensorMap(target, b), axes=axes, output=output)
        return jnp.real(jnp.vdot(value.storage.data, value.storage.data))

    old_gradient = jax.grad(objective, argnums=(0, 1))(left.storage.data, right.storage.data)
    monkeypatch.setattr(primitives, "_select_contraction_plan", select)
    new_gradient = jax.grad(objective, argnums=(0, 1))(left.storage.data, right.storage.data)
    for actual, reference in zip(new_gradient, old_gradient, strict=True):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(reference), rtol=2e-5, atol=2e-5)


def test_tie_keeps_original_orientation_and_fermion_does_not_reverse():
    for family in (U1Irrep, FermionParity):
        factor = space(family, {0: 2})
        target = hom((factor,), (factor,))
        plan = primitives._select_contraction_plan(
            (target, target), ((1,), (0,)), ((0,), (1,)), ((0,), (1,)),
            family is U1Irrep,
        )
        assert not plan.reversed_operands
        assert plan.left_permutation == ((0,), (1,))
        assert plan.right_permutation == ((0,), (1,))
    target = hom((space(FermionParity, {0: 2, 1: 2}),), (space(FermionParity, {0: 2, 1: 2}),))
    left, right = _value(target, 1), _value(target, 3)
    result = tensorcontract(left, right, axes=((0,), (1,)), output=(((1, 0),), ((0, 1),)))
    assert result.space == target

@pytest.mark.parametrize("family,charges", [
    (SU2Irrep, {0: 1, 1: 1}),
    (U1SU2Irrep, {(0, 0): 1, (0, 1): 1, (1, 0): 1}),
])
@pytest.mark.parametrize("open_rank", (2, 3))
def test_high_rank_nonabelian_reversal_preserves_fusion_and_output(
    monkeypatch, family, charges, open_rank,
):
    factor = space(family, charges)
    left_space = hom((factor,), (factor,) * open_rank)
    right_space = hom((factor,) * open_rank, (factor,))
    left, right = _value(left_space, 1), _value(right_space, 7)
    axes = ((0,), (open_rank,))
    output = (
        tuple((1, axis) for axis in range(open_rank)),
        tuple((0, axis) for axis in range(1, open_rank + 1)),
    )
    select = primitives._select_contraction_plan
    plan = select(
        (left_space, right_space), axes,
        (tuple(range(1, open_rank + 1)), tuple(range(open_rank))),
        (tuple(range(open_rank, 2 * open_rank)), tuple(range(open_rank))),
        True,
    )
    assert plan.reversed_operands
    result = tensorcontract(left, right, axes=axes, output=output)

    def nonreversed(spaces, contracted, opened, destination, allow_reversal):
        return select(spaces, contracted, opened, destination, False)

    monkeypatch.setattr(primitives, "_select_contraction_plan", nonreversed)
    reference = tensorcontract(left, right, axes=axes, output=output)
    assert result.space == reference.space
    np.testing.assert_allclose(
        np.asarray(result.storage.data), np.asarray(reference.storage.data),
        rtol=2e-5, atol=2e-5,
    )

    def objective(a, b):
        value = tensorcontract(TensorMap(left_space, a), TensorMap(right_space, b),
                               axes=axes, output=output)
        return jnp.real(jnp.vdot(value.storage.data, value.storage.data))

    old_grad = jax.grad(objective, argnums=(0, 1))(left.storage.data, right.storage.data)
    monkeypatch.setattr(primitives, "_select_contraction_plan", select)
    new_grad = jax.grad(objective, argnums=(0, 1))(left.storage.data, right.storage.data)
    for actual, expected in zip(new_grad, old_grad, strict=True):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=2e-5, atol=2e-5)

@pytest.mark.parametrize("family,charges", [
    (Z2Irrep, {0: 2, 1: 1}),
    (U1Irrep, {-1: 2, 0: 2, 1: 1}),
    (SU2Irrep, {0: 2, 1: 1}),
    (U1SU2Irrep, {(0, 0): 2, (0, 1): 1}),
])
@pytest.mark.parametrize("flags", [(True, False), (False, True), (True, True)])
@pytest.mark.parametrize("rank", [2, 3, 4])
def test_conjugated_contraction_reassociation_dense_oracle(family, charges, flags, rank):
    factor = space(family, charges)
    left_space = hom((factor,), (factor,) * (rank - 1))
    right_space = hom((factor,) * (rank - 1), (factor,))
    left, right = _value(left_space, 1), _value(right_space, 4)
    visible = tuple(
        hom(target.domain, target.codomain) if flag else target
        for target, flag in zip((left_space, right_space), flags, strict=True)
    )
    # Pick a compatible pair in original-axis coordinates, including mixed adjoint partitions.
    pairs = [
        (a, b) for a in range(rank) for b in range(rank)
        if visible[0][primitives._adjoint_axis(left_space, a) if flags[0] else a].dual()
        == visible[1][primitives._adjoint_axis(right_space, b) if flags[1] else b]
    ]
    assert pairs
    a, b = pairs[-1]
    left_open = tuple(i for i in range(rank) if i != a)
    right_open = tuple(i for i in range(rank) if i != b)
    refs = tuple((1, i) for i in right_open) + tuple((0, i) for i in left_open)
    output = (refs[::2], refs[1::2])
    result = tensorcontract(left, right, axes=((a,), (b,)), output=output, conjugate=flags)
    dense = [np.asarray(to_dense(t)) for t in (left, right)]
    for operand, flag in enumerate(flags):
        if flag:
            dense[operand] = np.conj(dense[operand])
    expected = np.tensordot(*dense, axes=((a,), (b,)))
    canonical = tuple((0, i) for i in left_open) + tuple((1, i) for i in right_open)
    expected = np.transpose(expected, tuple(canonical.index(ref) for group in output for ref in group))
    assert result.space == hom(
        tuple(visible[ref[0]][primitives._adjoint_axis((left_space, right_space)[ref[0]], ref[1])
            if flags[ref[0]] else ref[1]] for ref in output[0]),
        tuple(visible[ref[0]][primitives._adjoint_axis((left_space, right_space)[ref[0]], ref[1])
            if flags[ref[0]] else ref[1]].dual() for ref in output[1]),
    )
    assert result.storage.data.dtype == jnp.complex64
    np.testing.assert_allclose(np.asarray(to_dense(result)), expected, rtol=4e-5, atol=4e-5)


def test_conjugated_selected_reversal_and_both_gradients():
    factor = space(SU2Irrep, {0: 2, 1: 1})
    target = hom((factor,), (factor,))
    left, right = _value(target, 1), _value(target, 3)
    axes = ((1,), (0,))
    output = (((1, 1),), ((0, 0),))
    plan = primitives._select_contraction_plan(
        (target, target), ((0,), (1,)), ((1,), (0,)), ((1,), (0,)), True,
    )
    assert plan.reversed_operands

    def compute(a, b):
        return tensorcontract(TensorMap(target, a), TensorMap(target, b),
                              axes=axes, output=output, conjugate=(True, True))

    result = compute(left.storage.data, right.storage.data)
    expected = np.einsum("ik,kj->ji", np.conj(np.asarray(to_dense(left))),
                         np.conj(np.asarray(to_dense(right))))
    np.testing.assert_allclose(np.asarray(to_dense(result)), expected, rtol=4e-5, atol=4e-5)
    assert result.space == target and result.storage.data.dtype == jnp.complex64

    def loss(a, b):
        data = compute(a, b).storage.data
        return jnp.real(jnp.vdot(data, data))

    def old_loss(a, b):
        # Explicit original adjoint-first route, in the selected reversed order.
        first = primitives.permute(TensorMap(target, b).adjoint(), plan.left_permutation)
        second = primitives.permute(TensorMap(target, a).adjoint(), plan.right_permutation)
        twisted = primitives.twist(
            second, tuple(i for i in range(len(plan.right_permutation[0]))
                          if second.space[i].is_dual),
        )
        data = primitives.permute(
            primitives._compose(first, twisted), plan.output_permutation,
        ).storage.data
        return jnp.real(jnp.vdot(data, data))

    gradients = jax.grad(loss, argnums=(0, 1))(left.storage.data, right.storage.data)
    reference = jax.grad(old_loss, argnums=(0, 1))(left.storage.data, right.storage.data)
    for actual, expected in zip(gradients, reference, strict=True):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=4e-5, atol=4e-5)

@pytest.mark.parametrize("flags", [(False, True), (True, False), (True, True)])
def test_fermionic_adjoint_keeps_original_preparation_and_twist(monkeypatch, flags):
    odd = space(FermionParity, {1: 1})
    target = hom((odd,), (odd,))
    left, right = _value(target, 2), _value(target, 5)
    calls = []
    original = primitives._prepare_contraction_operand

    def observe(value, permutation, conjugate, symmetric_boson):
        calls.append((conjugate, symmetric_boson))
        return original(value, permutation, conjugate, symmetric_boson)

    monkeypatch.setattr(primitives, "_prepare_contraction_operand", observe)
    axes = ((1,), (0,)) if all(flags) else ((0,), (0,))
    output = (((0, 1 - axes[0][0]),), ((1, 1 - axes[1][0]),))
    result = tensorcontract(left, right, axes=axes, output=output, conjugate=flags)
    assert calls == [(flags[0], False), (flags[1], False)]
    spaces = tuple(hom(target.domain, target.codomain) if flag else target
                   for flag in flags)
    mapped = tuple((primitives._adjoint_axis(target, ax) if flag else ax,)
                   for ax, flag in zip((axes[0][0], axes[1][0]), flags, strict=True))
    opened = tuple((primitives._adjoint_axis(target, 1 - ax) if flag else 1 - ax,)
                   for ax, flag in zip((axes[0][0], axes[1][0]), flags, strict=True))
    plan = primitives._select_contraction_plan(spaces, mapped, opened, ((0,), (1,)), False)
    prepared_left = primitives.permute(left.adjoint() if flags[0] else left,
                                       plan.left_permutation)
    prepared_right = primitives.permute(right.adjoint() if flags[1] else right,
                                        plan.right_permutation)
    reference = primitives.permute(primitives._compose(
        prepared_left,
        primitives.twist(prepared_right, (0,) if prepared_right.space[0].is_dual else ()),
    ), plan.output_permutation)
    assert result.space == reference.space
    assert result.storage.data.dtype == reference.storage.data.dtype
    np.testing.assert_allclose(np.asarray(result.storage.data), np.asarray(reference.storage.data),
                               rtol=2e-5, atol=2e-5)


def test_rank_four_two_contract_axes_with_both_adjoints(monkeypatch):
    import tensor0.operations.transforms as transforms

    composed = []
    original = primitives._permute_to_adjoint_destination

    def recording(value, *permutation):
        parent = value.space.permute(*permutation)
        transformer = transforms._treepermuter(value.space, parent, *permutation)
        composed.append(any(entry.transform.shape[0] > 1 and entry.transform.shape[1] > 1
                            for entry in transformer.generic_data)
                        if transformer.kind == "generic" else False)
        return original(value, *permutation)

    monkeypatch.setattr(primitives, "_permute_to_adjoint_destination", recording)
    factor = space(SU2Irrep, {0: 2, 1: 1})
    left_space = hom((factor,) * 3, (factor,))
    right_space = hom((factor,), (factor,) * 3)
    left, right = _value(left_space, 2), _value(right_space, 7)
    axes = ((0, 1), (1, 2))
    output = (((1, 0), (0, 3)), ((1, 3), (0, 2)))
    result = tensorcontract(left, right, axes=axes, output=output,
                            conjugate=(True, True))
    assert len(composed) == 2 and any(composed)
    expected = np.tensordot(np.conj(np.asarray(to_dense(left))),
                            np.conj(np.asarray(to_dense(right))), axes=axes)
    # Canonical open order: left 2,3 then right 0,3.
    expected = np.transpose(expected, (2, 1, 3, 0))
    assert result.storage.data.dtype == jnp.complex64
    np.testing.assert_allclose(np.asarray(to_dense(result)), expected, rtol=4e-5, atol=4e-5)

@pytest.mark.parametrize("family,charges", [
    (SU2Irrep, {0: 2, 1: 1}),
    (U1SU2Irrep, {(0, 0): 2, (0, 1): 1, (1, 0): 1}),
])
def test_mixed_fusion_paths_use_eager_canonical_adjoint(family, charges, monkeypatch):
    """A genuinely off-diagonal group must avoid quadratic affine metadata."""
    import tensor0.operations.transforms as transforms

    factor = space(family, charges)
    source_space = hom((factor,), (factor,) * 3)
    parent_axes = ((0, 1), (2, 3))
    parent_space = source_space.permute(*parent_axes)
    transformer = transforms._treepermuter(source_space, parent_space, *parent_axes)
    assert transformer.kind == "generic"
    mixed = [entry for entry in transformer.generic_data if entry.transform.shape[0] > 1
             and entry.transform.shape[1] > 1]
    assert mixed
    assert any(np.count_nonzero(np.asarray(entry.transform) -
                                np.diag(np.diag(np.asarray(entry.transform)))) > 0
               for entry in mixed)
    source = _value(source_space, 9)
    eager_calls = []
    old_permute = transforms.permute

    def record_eager(value, permutation):
        eager_calls.append(permutation)
        return old_permute(value, permutation)

    monkeypatch.setattr(transforms, "permute", record_eager)

    def direct(data):
        return transforms._permute_to_adjoint_destination(
            TensorMap(source_space, data), *parent_axes,
        )

    actual = direct(source.storage.data)
    assert eager_calls == [parent_axes]
    reference = transforms.permute(source, parent_axes).adjoint()
    assert actual.space == reference.space
    np.testing.assert_allclose(np.asarray(actual.storage.data),
                               np.asarray(reference.storage.data), rtol=5e-5, atol=5e-5)
    parent_dense = np.transpose(np.asarray(to_dense(source)), parent_axes[0] + parent_axes[1])
    expected = np.conj(np.transpose(parent_dense, (2, 3, 0, 1)))
    np.testing.assert_allclose(np.asarray(to_dense(actual)), expected, rtol=5e-5, atol=5e-5)

    def loss(fn, data):
        output = fn(data).storage.data
        return jnp.real(jnp.vdot(output, output))

    expected_grad = jax.grad(lambda data: loss(
        lambda x: transforms.permute(TensorMap(source_space, x), parent_axes).adjoint(),
        data,
    ))(source.storage.data)
    actual_grad = jax.grad(lambda data: loss(direct, data))(source.storage.data)
    np.testing.assert_allclose(np.asarray(actual_grad), np.asarray(expected_grad),
                               rtol=5e-5, atol=5e-5)

@pytest.mark.parametrize("dtype", [jnp.float16, jnp.complex64])
def test_direct_adjoint_empty_storage_matches_established_dtype(dtype):
    import tensor0.operations.transforms as transforms

    factor = space(SU2Irrep, {1: 1})
    source_space = hom((factor,), ())
    assert get_degeneracystructure(source_space).total_dim == 0
    value = TensorMap(source_space, jnp.zeros((0,), dtype=dtype))
    permutation = ((), (0,))
    actual = transforms._permute_to_adjoint_destination(value, *permutation)
    expected = transforms.permute(value, permutation).adjoint()
    assert actual.space == expected.space
    assert actual.storage.data.dtype == expected.storage.data.dtype
    assert actual.storage.data.shape == expected.storage.data.shape == (0,)

@pytest.mark.parametrize("family,charges", [
    (SU2Irrep, {1: 1}),
    (U1SU2Irrep, {(0, 1): 1}),
])
def test_rank_six_five_path_both_adjoint_contracts_and_all_dense_gradients(
    family, charges, monkeypatch,
):
    import tensor0.operations.transforms as transforms

    factor = space(family, charges)
    target = hom((factor,) * 3, (factor,) * 3)
    size = get_degeneracystructure(target).total_dim
    grid = jnp.arange(size, dtype=jnp.float32)
    a = (jnp.sin(grid * .71 + .3) + 1j * jnp.cos(grid * .37 + .9)).astype(jnp.complex64)
    b = (jnp.sin(grid * .17 + .7) + 1j * jnp.cos(grid * .29 + .5)).astype(jnp.complex64)
    axes = ((0, 2, 4), (3, 5, 1))
    output = (((1, 4), (0, 1), (1, 0)), ((0, 5), (1, 2), (0, 3)))
    calls = []
    original = primitives._permute_to_adjoint_destination

    def record(value, *permutation):
        transformer = transforms._treepermuter(
            value.space, value.space.permute(*permutation), *permutation,
        )
        assert permutation != (tuple(range(value.numout)),
                               tuple(range(value.numout, value.numind)))
        calls.append(max((max(group.transform.shape)
                          for group in transformer.generic_data), default=0))
        return original(value, *permutation)

    monkeypatch.setattr(primitives, "_permute_to_adjoint_destination", record)
    eager_calls = []
    original_permute = transforms.permute

    def record_eager(value, permutation):
        eager_calls.append(permutation)
        return original_permute(value, permutation)

    monkeypatch.setattr(transforms, "permute", record_eager)

    def candidate(x, y):
        return tensorcontract(TensorMap(target, x), TensorMap(target, y),
                              axes=axes, output=output, conjugate=(True, True))

    def dense(x, y):
        left = to_dense(TensorMap(target, x))
        right = to_dense(TensorMap(target, y))
        canonical = jnp.tensordot(jnp.conj(left), jnp.conj(right), axes=axes)
        refs = tuple((0, i) for i in (1, 3, 5)) + tuple((1, i) for i in (0, 2, 4))
        return jnp.transpose(canonical, tuple(refs.index(ref) for group in output for ref in group))

    actual = to_dense(candidate(a, b))
    np.testing.assert_allclose(np.asarray(actual), np.asarray(dense(a, b)),
                               rtol=4e-5, atol=4e-5)
    assert len(calls) == 2 and calls == [5, 5]
    assert len(eager_calls) == 2

    def loss(result):
        return jnp.real(jnp.vdot(result, result))
    actual_grads = jax.grad(lambda x, y: loss(to_dense(candidate(x, y))),
                            argnums=(0, 1))(a, b)
    dense_grads = jax.grad(lambda x, y: loss(dense(x, y)), argnums=(0, 1))(a, b)
    for actual_grad, dense_grad in zip(actual_grads, dense_grads, strict=True):
        np.testing.assert_allclose(np.asarray(actual_grad), np.asarray(dense_grad),
                                   rtol=5e-5, atol=5e-5)

@pytest.mark.parametrize("family,charges", [
    (U1Irrep, {0: 2, 1: 1}),
    (Z2Irrep, {0: 2, 1: 1}),
    (Z3Irrep, {0: 2, 1: 1}),
    (Z4Irrep, {0: 2, 1: 1}),
    (SU2Irrep, {0: 2, 1: 1}),
    (U1SU2Irrep, {(0, 0): 2, (0, 1): 1}),
])
def test_scalar_group_direct_destination_without_eager_fallback(family, charges, monkeypatch):
    import tensor0.operations.transforms as transforms

    factor = space(family, charges)
    source_space = hom((factor, factor), (factor,))
    permutation = ((2,), (0, 1))
    parent_space = source_space.permute(*permutation)
    transformer = transforms._treepermuter(source_space, parent_space, *permutation)
    assert transformer.kind == "abelian" or all(
        group.transform.shape == (1, 1) for group in transformer.generic_data
    )
    value = _value(source_space, 3)
    original_permute = transforms.permute

    def disallow_eager(*_args):
        raise AssertionError("scalar groups must compose destination descriptors directly")

    monkeypatch.setattr(transforms, "permute", disallow_eager)
    result = transforms._permute_to_adjoint_destination(value, *permutation)
    monkeypatch.setattr(transforms, "permute", original_permute)
    expected = original_permute(value, permutation).adjoint()
    assert result.space == expected.space
    np.testing.assert_allclose(np.asarray(result.storage.data),
                               np.asarray(expected.storage.data), rtol=5e-5, atol=5e-5)
    parent_dense = np.transpose(np.asarray(to_dense(value)), permutation[0] + permutation[1])
    dense = np.conj(np.transpose(parent_dense, (1, 2, 0)))
    np.testing.assert_allclose(np.asarray(to_dense(result)), dense, rtol=5e-5, atol=5e-5)
