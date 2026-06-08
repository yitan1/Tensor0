import jax.numpy as jnp
import pytest

from tensor0 import (
    FermionParity,
    TensorMap,
    U1Irrep,
    braid,
    hom,
    permute,
    repartition,
    space,
    transpose,
    to_dense,
)
from tests.cases import assert_allclose, transform_cases, zero_based_float_data_for


def _assert_dense_transpose(result, tensor, expected_space, axes):
    assert result.space == expected_space
    assert_allclose(to_dense(result), jnp.transpose(to_dense(tensor), axes))


def _simple_u1_tensor():
    v = space(U1Irrep, {0: 2})
    h = hom((v,), (v,))
    return TensorMap(h, jnp.arange(4, dtype=jnp.float32))


def _odd_odd_fermion_tensor():
    odd_a = space(FermionParity, {1: 1})
    odd_b = space(FermionParity, {1: 1})
    h = hom((odd_a, odd_b), ())
    return TensorMap(h, jnp.array([2.0], dtype=jnp.float32))


def test_permute_rejects_non_tensormap_input():
    with pytest.raises(TypeError, match=r"permute\(\) requires a TensorMap"):
        permute(object(), ((0,), (1,)))  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize(
    "permutation",
    [((0,), (0,)), ((0,), (2,))],
    ids=["duplicate-visible-index", "out-of-range-visible-index"],
)
def test_permute_rejects_invalid_visible_index_permutations(permutation):
    tensor = _simple_u1_tensor()

    with pytest.raises(ValueError, match="visible index"):
        permute(tensor, permutation)


@pytest.mark.parametrize("case", transform_cases(), ids=lambda case: case.name)
def test_permute_matches_public_dense_transpose_for_space_cases(case):
    tensor = TensorMap(case.space, zero_based_float_data_for(case.space))

    result = permute(tensor, case.permutation)

    _assert_dense_transpose(result, tensor, case.expected_space, case.dense_axes)


def test_repartition_matches_explicit_permute():
    v = space(U1Irrep, {0: 2})
    w = space(U1Irrep, {0: 3})
    x = space(U1Irrep, {0: 5})
    h = hom((v,), (w, x))
    tensor = TensorMap(h, zero_based_float_data_for(h))

    result = repartition(tensor, 2)
    expected = permute(tensor, ((0, 2), (1,)))

    assert result.space == expected.space
    assert_allclose(result.storage.data, expected.storage.data)


def test_fermion_parity_odd_odd_permute_matches_public_dense_phase():
    tensor = _odd_odd_fermion_tensor()

    result = permute(tensor, ((1, 0), ()))

    assert_allclose(to_dense(result), -jnp.transpose(to_dense(tensor), (1, 0)))


def test_braid_matches_default_permute_for_identity_levels():
    tensor = _odd_odd_fermion_tensor()

    result = braid(tensor, ((1, 0), ()), (0, 1))
    expected = permute(tensor, ((1, 0), ()))

    assert result.space == expected.space
    assert_allclose(result.storage.data, expected.storage.data)


def test_transpose_default_matches_public_dense_transpose():
    v = space(U1Irrep, {0: 2})
    w = space(U1Irrep, {0: 3})
    x = space(U1Irrep, {0: 5})
    h = hom((v,), (w, x))
    tensor = TensorMap(h, zero_based_float_data_for(h))

    result = transpose(tensor)

    _assert_dense_transpose(
        result,
        tensor,
        hom((x.dual(), w.dual()), (v.dual(),)),
        (2, 1, 0),
    )
