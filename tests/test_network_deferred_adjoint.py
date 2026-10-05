"""Adjoint-visible network leaves versus independent eager logical operands."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tensor0.operations.contractions.index_notation as notation
from tensor0 import (ComplexSpace, FermionParity, SU2Irrep, TensorMap,
                     U1Irrep, U1SU2Irrep, Z2Irrep, Z3Irrep, Z4Irrep, hom, ncon, space, to_dense)
from tensor0.operations.contractions.primitives import _adjoint_axis
from tensor0.structure import get_degeneracystructure


@pytest.mark.parametrize('family,sectors', [
    (None, None), (Z2Irrep, {0: 2, 1: 1}),
    (U1Irrep, {-1: 1, 0: 2, 1: 1}),
    (Z3Irrep, {0: 2, 1: 1}), (Z4Irrep, {0: 2, 1: 1}),
    (SU2Irrep, {0: 2, 1: 1}),
    (U1SU2Irrep, {(0, 0): 2, (1, 1): 1}),
    (FermionParity, {0: 2, 1: 1}),
])
@pytest.mark.parametrize('flags', [(True, False, False), (False, True, False),
                                   (True, True, True)])
@pytest.mark.parametrize('order', [(1, 2), (2, 1)])
def test_deferred_leaf_adjoint_matches_explicit_eager_and_dense_oracle(
    monkeypatch, family, sectors, flags, order,
):
    factor = ComplexSpace(2) if family is None else space(family, sectors)
    groups = ((-1, 1, -2), (1, 2, -3), (2, -4, -5))
    seen = set()
    logical, tensors, original_labels = [], [], []
    for index, labels in enumerate(groups):
        numout = 2 if index != 1 else 1
        visible = tuple(factor.dual() if label > 0 and label in seen else factor
                        for label in labels)
        seen.update(label for label in labels if label > 0)
        target = hom(visible[:numout], tuple(leg.dual() for leg in visible[numout:]))
        size = get_degeneracystructure(target).total_dim
        data = jnp.arange(1, size + 1, dtype=jnp.float32)
        value = TensorMap(target, (jnp.sin(data * 0.17 + index)
                                   + 1j * jnp.cos(data * 0.13 + index)).astype(jnp.complex64))
        logical.append(value)
        parent = value.adjoint() if flags[index] else value
        tensors.append(parent)
        original_labels.append(tuple(labels[_adjoint_axis(parent.space, axis)]
                                     for axis in range(parent.numind)) if flags[index] else labels)
    output = ((-5, -1, -3), (-4, -2))
    captured = []
    primitive = notation.tensorcontract
    def record(*args, **kwargs):
        captured.append(kwargs['conjugate'])
        return primitive(*args, **kwargs)
    monkeypatch.setattr(notation, 'tensorcontract', record)
    def candidate(*values):
        return ncon(tuple(TensorMap(t.space, data) for t, data in zip(tensors, values, strict=True)),
                    tuple(original_labels), conjugate=flags, order=order, output=output)
    def reference(*values):
        nodes = tuple(TensorMap(t.space, data).adjoint() if flag else TensorMap(t.space, data)
                      for t, data, flag in zip(tensors, values, flags, strict=True))
        return ncon(nodes, groups, order=order, output=output)
    inputs = tuple(t.storage.data for t in tensors)
    actual, expected = candidate(*inputs), reference(*inputs)
    assert actual.space == expected.space
    np.testing.assert_allclose(np.asarray(actual.storage.data), np.asarray(expected.storage.data), rtol=2e-5, atol=2e-5)
    if family is not FermionParity:
        dense = [np.asarray(to_dense(value)) for value in logical]
        raw = np.einsum('axb,xyc,yde->eacdb', *dense, optimize=True)
        np.testing.assert_allclose(np.asarray(to_dense(actual)), raw, rtol=2e-5, atol=2e-5)
    if family not in (None, FermionParity):
        assert any(any(pair) for pair in captured)
    else:
        assert all(not any(pair) for pair in captured)
    if family in (None, SU2Irrep, U1SU2Irrep):
        objective = lambda fn: lambda *data: jnp.real(jnp.sum(fn(*data).storage.data))
        grad_actual = jax.grad(objective(candidate), argnums=(0, 1, 2))(*inputs)
        grad_expected = jax.grad(objective(reference), argnums=(0, 1, 2))(*inputs)
        for left, right in zip(grad_actual, grad_expected, strict=True):
            np.testing.assert_allclose(np.asarray(left), np.asarray(right), rtol=3e-5, atol=3e-5)

@pytest.mark.parametrize("family,charges", [
    (SU2Irrep, {1: 1}), (U1SU2Irrep, {(0, 1): 1}),
])
def test_five_fusion_path_pending_network_matches_dense_and_all_input_ad(
    family, charges, monkeypatch,
):
    from tensor0.operations import transforms
    from tensor0.operations.contractions import primitives

    factor = space(family, charges)
    rank_six = hom((factor,) * 3, (factor,) * 3)
    link = hom((factor.dual(),), (factor,))
    spaces = (rank_six, rank_six, link)
    values = []
    for index, target in enumerate(spaces):
        grid = jnp.arange(get_degeneracystructure(target).total_dim, dtype=jnp.float32)
        values.append((jnp.sin(grid * (.13 + index * .08) + index)
                       + 1j * jnp.cos(grid * (.23 + index * .04) + index * .5)).astype(jnp.complex64))
    labels = ((1, -1, 2, -2, 3, -3), (-4, 3, -5, 1, 4, 2), (4, -6))
    output = ((-1, -2, -3), (-4, -5, -6))
    seen = []
    original = primitives._permute_to_adjoint_destination

    def record(value, *permutation):
        transformer = transforms._treepermuter(
            value.space, value.space.permute(*permutation), *permutation,
        )
        assert permutation != (tuple(range(value.numout)),
                               tuple(range(value.numout, value.numind)))
        seen.append(max((max(group.transform.shape)
                         for group in transformer.generic_data), default=0))
        return original(value, *permutation)

    monkeypatch.setattr(primitives, "_permute_to_adjoint_destination", record)
    eager_calls = []
    original_permute = transforms.permute

    def record_eager(value, permutation):
        eager_calls.append(permutation)
        return original_permute(value, permutation)

    monkeypatch.setattr(transforms, "permute", record_eager)

    def candidate(*data):
        tensors = tuple(TensorMap(target, array)
                        for target, array in zip(spaces, data, strict=True))
        return ncon(tensors, labels, conjugate=(True, True, False),
                    order=(1, 2, 3, 4), output=output)

    def dense(*data):
        left, right, last = (to_dense(TensorMap(target, array))
                             for target, array in zip(spaces, data, strict=True))
        return jnp.einsum("xaybzc,dzexry,rf->abcdef", jnp.conj(left),
                          jnp.conj(right), last)

    actual = to_dense(candidate(*values))
    np.testing.assert_allclose(np.asarray(actual), np.asarray(dense(*values)),
                               rtol=5e-5, atol=5e-5)
    assert seen == [5, 5]
    assert len(eager_calls) == 2

    def loss(result):
        return jnp.real(jnp.vdot(result, result))

    actual_grads = jax.grad(lambda *data: loss(to_dense(candidate(*data))),
                            argnums=(0, 1, 2))(*values)
    dense_grads = jax.grad(lambda *data: loss(dense(*data)),
                           argnums=(0, 1, 2))(*values)
    for actual_grad, expected_grad in zip(actual_grads, dense_grads, strict=True):
        np.testing.assert_allclose(np.asarray(actual_grad), np.asarray(expected_grad),
                                   rtol=8e-5, atol=8e-5)
