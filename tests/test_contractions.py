import jax.numpy as jnp
import pytest

from tensor0 import (
    FermionNumber,
    FermionParity,
    FermionParitySU2Irrep,
    FermionParityU1Irrep,
    FermionParityU1SU2Irrep,
    SU2Irrep,
    TensorMap,
    U1Irrep,
    U1SU2Irrep,
    Z2Irrep,
    Z3Irrep,
    Z4Irrep,
    get_degeneracystructure,
    hom,
    space,
    tensorcontract,
    to_dense,
)
from tests.cases import assert_allclose


def _tensor(target):
    size = get_degeneracystructure(target).total_dim
    return TensorMap(target, jnp.arange(1, size + 1, dtype=jnp.float32))


def _dense_contract(left, right, axes, output):
    left_axes, right_axes = axes
    result = jnp.tensordot(
        to_dense(left),
        to_dense(right),
        axes=(left_axes, right_axes),
    )
    left_contracted = set(left_axes)
    right_contracted = set(right_axes)
    canonical_refs = tuple(
        (0, axis) for axis in range(left.numind) if axis not in left_contracted
    ) + tuple(
        (1, axis) for axis in range(right.numind) if axis not in right_contracted
    )
    canonical_positions = {
        ref: position for position, ref in enumerate(canonical_refs)
    }
    permutation = tuple(
        canonical_positions[ref] for group in output for ref in group
    )
    if permutation != tuple(range(len(permutation))):
        result = jnp.transpose(result, permutation)
    return result


class _ExplodingVectorData:
    def __init__(self, length):
        self.shape = (length,)

    def __getitem__(self, key):
        if isinstance(key, slice) and key.start == 0 and key.stop == 0:
            return jnp.zeros((0,))
        raise AssertionError("tensorcontract validation must not access storage")


@pytest.mark.parametrize(
    ("metadata", "error", "match"),
    [
        (
            {"axes": [(1,), (0,)]},
            TypeError,
            "axes.*tuple",
        ),
        (
            {"axes": ((True,), (0,))},
            TypeError,
            "axis.*int",
        ),
        (
            {"axes": ((-1,), (0,))},
            ValueError,
            "axis.*out of range",
        ),
        (
            {"axes": ((0, 0), (0, 1))},
            ValueError,
            "exactly once",
        ),
        (
            {"axes": ((1,), ())},
            ValueError,
            "same number",
        ),
        (
            {"output": ((), ((1, 1),))},
            ValueError,
            "exactly once",
        ),
        (
            {"output": (((0, 0), (0, 0)), ((1, 1),))},
            ValueError,
            "exactly once",
        ),
        (
            {"output": (((2, 0),), ((1, 1),))},
            ValueError,
            "operand.*0 or 1",
        ),
        (
            {"output": (((0, -1),), ((1, 1),))},
            ValueError,
            "output axis.*out of range",
        ),
        (
            {"conjugate": (False,)},
            TypeError,
            "conjugate.*tuple.*length 2",
        ),
        (
            {"conjugate": (False, 0)},
            TypeError,
            "conjugate.*bool",
        ),
    ],
    ids=[
        "axes-outer-not-tuple",
        "bool-axis",
        "negative-axis",
        "duplicate-contracted-axis",
        "unequal-contracted-counts",
        "incomplete-coverage",
        "duplicate-output-reference",
        "invalid-output-operand",
        "negative-output-axis",
        "conjugate-wrong-length",
        "conjugate-non-bool",
    ],
)
def test_tensorcontract_rejects_invalid_metadata(metadata, error, match):
    factor = space(U1Irrep, {0: 1})
    left = _tensor(hom((factor,), (factor,)))
    right = _tensor(hom((factor,), (factor,)))
    arguments = {
        "axes": ((1,), (0,)),
        "output": (((0, 0),), ((1, 1),)),
        "conjugate": (False, False),
    }
    arguments.update(metadata)

    with pytest.raises(error, match=match):
        tensorcontract(left, right, **arguments)


def test_tensorcontract_rejects_non_tensor_inputs():
    factor = space(U1Irrep, {0: 1})
    tensor = _tensor(hom((factor,), (factor,)))
    metadata = {
        "axes": ((1,), (0,)),
        "output": (((0, 0),), ((1, 1),)),
        "conjugate": (False, False),
    }

    with pytest.raises(TypeError, match="left and right.*TensorMap"):
        tensorcontract(object(), tensor, **metadata)
    with pytest.raises(TypeError, match="left and right.*TensorMap"):
        tensorcontract(tensor, object(), **metadata)


def test_tensorcontract_rejects_mismatched_sector_families():
    u1 = space(U1Irrep, {0: 1})
    z2 = space(Z2Irrep, {0: 1})
    left = _tensor(hom((u1,), (u1,)))
    right = _tensor(hom((z2,), (z2,)))

    with pytest.raises(ValueError, match="sector family"):
        tensorcontract(
            left,
            right,
            axes=((1,), (0,)),
            output=(((0, 0),), ((1, 1),)),
            conjugate=(False, False),
        )


def test_tensorcontract_rejects_non_dual_contracted_spaces():
    charge_zero = space(U1Irrep, {0: 1})
    charge_one = space(U1Irrep, {1: 1})
    left = _tensor(hom((charge_zero,), (charge_zero,)))
    right = _tensor(hom((charge_one,), (charge_one,)))

    with pytest.raises(ValueError, match="dual.*compatible"):
        tensorcontract(
            left,
            right,
            axes=((1,), (0,)),
            output=(((0, 0),), ((1, 1),)),
            conjugate=(False, False),
        )


def test_tensorcontract_composition_matches_matmul():
    a = space(U1Irrep, {0: 2})
    x = space(U1Irrep, {0: 3})
    b = space(U1Irrep, {0: 4})
    left = _tensor(hom((a,), (x,)))
    right = _tensor(hom((x,), (b,)))

    result = tensorcontract(
        left,
        right,
        axes=((1,), (0,)),
        output=(((0, 0),), ((1, 1),)),
    )
    expected = left @ right

    assert result.space == expected.space
    assert_allclose(result.storage.data, expected.storage.data)


def test_tensorcontract_fermion_parity_composition_without_twist_is_positive():
    odd = space(FermionParity, {1: 1})
    target = hom((odd,), (odd,))
    left = TensorMap(target, jnp.array([2.0], dtype=jnp.float32))
    right = TensorMap(target, jnp.array([3.0], dtype=jnp.float32))

    result = tensorcontract(
        left,
        right,
        axes=((1,), (0,)),
        output=(((0, 0),), ((1, 1),)),
    )

    assert_allclose(result.storage.data, jnp.array([6.0], dtype=jnp.float32))


def test_tensorcontract_fermion_parity_dual_right_codomain_applies_twist():
    odd = space(FermionParity, {1: 1})
    odd_dual = odd.dual()
    left = TensorMap(
        hom((odd,), (odd_dual,)),
        jnp.array([2.0], dtype=jnp.float32),
    )
    right = TensorMap(
        hom((odd_dual,), (odd,)),
        jnp.array([3.0], dtype=jnp.float32),
    )

    result = tensorcontract(
        left,
        right,
        axes=((1,), (0,)),
        output=(((0, 0),), ((1, 1),)),
    )

    # TensorKit: twist(odd) * 2 * 3 == -6.
    assert_allclose(result.storage.data, jnp.array([-6.0], dtype=jnp.float32))


@pytest.mark.parametrize(
    ("sector_type", "sector", "expected_sign"),
    [
        (FermionNumber, (0, 1), -1),
        (FermionParityU1Irrep, (1, 0), -1),
        (U1SU2Irrep, (0, 0), 1),
        (FermionParitySU2Irrep, (1, 0), -1),
        (FermionParityU1SU2Irrep, (1, 0, 0), -1),
    ],
    ids=[
        "fermion-number",
        "fermion-parity-u1",
        "u1-su2",
        "fermion-parity-su2",
        "fermion-parity-u1-su2",
    ],
)
def test_tensorcontract_product_sector_twist_is_multiplicative(
    sector_type,
    sector,
    expected_sign,
):
    factor = space(sector_type, {sector: 1})
    dual = factor.dual()
    left = TensorMap(
        hom((factor,), (dual,)),
        jnp.array([2.0], dtype=jnp.float32),
    )
    right = TensorMap(
        hom((dual,), (factor,)),
        jnp.array([3.0], dtype=jnp.float32),
    )

    result = tensorcontract(
        left,
        right,
        axes=((1,), (0,)),
        output=(((0, 0),), ((1, 1),)),
    )

    assert_allclose(
        result.storage.data,
        jnp.array([expected_sign * 6.0], dtype=jnp.float32),
    )


@pytest.mark.parametrize(
    "sector_type",
    [Z2Irrep, Z3Irrep, Z4Irrep],
    ids=["z2", "z3", "z4"],
)
def test_tensorcontract_zn_sector_composition_has_trivial_twist(sector_type):
    factor = space(sector_type, {1: 1})
    target = hom((factor,), (factor,))
    left = TensorMap(target, jnp.array([2.0], dtype=jnp.float32))
    right = TensorMap(target, jnp.array([3.0], dtype=jnp.float32))

    result = tensorcontract(
        left,
        right,
        axes=((1,), (0,)),
        output=(((0, 0),), ((1, 1),)),
    )

    assert_allclose(result.storage.data, jnp.array([6.0], dtype=jnp.float32))


def test_tensorcontract_left_conjugation_uses_tensor_adjoint():
    factor = space(U1Irrep, {0: 1})
    target = hom((factor,), (factor,))
    left = TensorMap(target, jnp.array([2.0 + 3.0j], dtype=jnp.complex64))
    right = TensorMap(target, jnp.array([5.0 - 7.0j], dtype=jnp.complex64))

    result = tensorcontract(
        left,
        right,
        axes=((0,), (0,)),
        output=(((0, 1),), ((1, 1),)),
        conjugate=(True, False),
    )

    assert_allclose(
        result.storage.data,
        jnp.array([-11.0 - 29.0j], dtype=jnp.complex64),
    )


def test_tensorcontract_right_conjugation_remaps_twist_after_permutation():
    odd = space(FermionParity, {1: 1})
    even = space(FermionParity, {0: 1})
    left = TensorMap(
        hom((odd,), (odd,)),
        jnp.array([2.0], dtype=jnp.float32),
    )
    right = TensorMap(
        hom((odd, even), (odd,)),
        jnp.array([3.0], dtype=jnp.float32),
    )

    result = tensorcontract(
        left,
        right,
        axes=((0, 1), (0, 2)),
        output=(((1, 1),), ()),
        conjugate=(False, True),
    )

    assert_allclose(result.storage.data, jnp.array([-6.0], dtype=jnp.float32))


def test_tensorcontract_partial_contraction_matches_dense():
    a = space(U1Irrep, {0: 2})
    x = space(U1Irrep, {0: 3})
    b = space(U1Irrep, {0: 4})
    c = space(U1Irrep, {0: 5})
    d = space(U1Irrep, {0: 6})
    left = _tensor(hom((a, x), (c,)))
    right = _tensor(hom((x.dual(), b), (d,)))
    axes = ((1,), (0,))
    output = (((0, 0), (1, 1)), ((0, 2), (1, 2)))

    result = tensorcontract(left, right, axes=axes, output=output)

    assert result.numout == 2
    assert result.numin == 2
    assert_allclose(to_dense(result), _dense_contract(left, right, axes, output))


def test_tensorcontract_su2_nontrivial_fusion_tree_transform_matches_dense():
    half = space(SU2Irrep, {1: 1})
    left = _tensor(hom((half, half, half), (half,)))
    right = _tensor(hom((half.dual(), half, half), (half,)))
    axes = ((2,), (0,))
    output = (((0, 0), (0, 1), (1, 1), (1, 2)), ((0, 3), (1, 3)))

    result = tensorcontract(left, right, axes=axes, output=output)

    assert_allclose(to_dense(result), _dense_contract(left, right, axes, output))


def test_tensorcontract_without_contracted_axes_matches_dense_tensor_product():
    a = space(U1Irrep, {0: 2})
    b = space(U1Irrep, {0: 3})
    left = _tensor(hom((a,), ()))
    right = _tensor(hom((), (b,)))
    axes = ((), ())
    output = (((0, 0),), ((1, 0),))

    result = tensorcontract(left, right, axes=axes, output=output)

    assert_allclose(to_dense(result), _dense_contract(left, right, axes, output))


def test_tensorcontract_all_visible_axes_matches_dense_scalar():
    a = space(U1Irrep, {0: 2})
    x = space(U1Irrep, {0: 3})
    left = _tensor(hom((a,), (x,)))
    right = _tensor(hom((x,), (a,)))
    axes = ((0, 1), (1, 0))
    output = ((), ())

    result = tensorcontract(left, right, axes=axes, output=output)

    assert result.numind == 0
    assert_allclose(to_dense(result), _dense_contract(left, right, axes, output))


def test_tensorcontract_rejects_invalid_metadata_before_accessing_storage():
    factor = space(U1Irrep, {0: 1})
    left_space = hom((factor,), (factor,))
    right_space = hom((factor,), (factor,))
    left = TensorMap(
        left_space,
        _ExplodingVectorData(get_degeneracystructure(left_space).total_dim),
    )
    right = TensorMap(
        right_space,
        _ExplodingVectorData(get_degeneracystructure(right_space).total_dim),
    )

    with pytest.raises(ValueError, match="exactly once"):
        tensorcontract(
            left,
            right,
            axes=((1,), (0,)),
            output=((), ((1, 1),)),
        )
