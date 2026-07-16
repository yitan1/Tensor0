import jax
import jax.numpy as jnp
import pytest

import tensor0.operations.contractions.primitives as contractions
import tensor0.operations.transforms as transforms
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
    contract,
    hom,
    idx,
    ncon,
    permute,
    scalar,
    space,
    tensorcontract,
    tensortrace,
    to_dense,
)
from tensor0.structure import get_degeneracystructure
from tests.cases import (
    InaccessibleVectorData,
    assert_allclose,
    is_contiguous_subblock,
)


# Shared fixtures and independent dense oracles.
_SECTOR_CASES = (
    pytest.param(U1Irrep, 0, 1, id="u1"),
    pytest.param(Z2Irrep, 0, 1, id="z2"),
    pytest.param(Z3Irrep, 0, 1, id="z3"),
    pytest.param(Z4Irrep, 0, 1, id="z4"),
    pytest.param(SU2Irrep, 1, 2, id="su2"),
    pytest.param(FermionParity, 1, -1, id="fermion-parity"),
    pytest.param(FermionNumber, (0, 1), -1, id="fermion-number"),
    pytest.param(
        FermionParityU1Irrep,
        (1, 0),
        -1,
        id="fermion-parity-u1",
    ),
    pytest.param(U1SU2Irrep, (0, 1), 2, id="u1-su2"),
    pytest.param(
        FermionParitySU2Irrep,
        (1, 1),
        -2,
        id="fermion-parity-su2",
    ),
    pytest.param(
        FermionParityU1SU2Irrep,
        (1, 0, 1),
        -2,
        id="fermion-parity-u1-su2",
    ),
)


def _tensor(target, dtype=jnp.float32):
    size = get_degeneracystructure(target).total_dim
    data = jnp.arange(1, size + 1, dtype=jnp.float32)
    if jnp.issubdtype(dtype, jnp.complexfloating):
        data = data + 1j * (data + 1)
    return TensorMap(target, data.astype(dtype))


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


def _metadata_only_tensor(target):
    size = get_degeneracystructure(target).total_dim
    return TensorMap(target, InaccessibleVectorData(size))


def _neutral_endomorphism():
    factor = space(U1Irrep, {0: 1})
    return _metadata_only_tensor(hom((factor,), (factor,)))


# Binary contraction primitive.
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
        tensorcontract(object(), tensor, **metadata)  # pyright: ignore[reportArgumentType]
    with pytest.raises(TypeError, match="left and right.*TensorMap"):
        tensorcontract(tensor, object(), **metadata)  # pyright: ignore[reportArgumentType]


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


def test_tensorcontract_product_sector_applies_component_twist():
    factor = space(FermionParityU1SU2Irrep, {(1, 0, 0): 1})
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
        jnp.array([-6.0], dtype=jnp.float32),
    )


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
        InaccessibleVectorData(get_degeneracystructure(left_space).total_dim),
    )
    right = TensorMap(
        right_space,
        InaccessibleVectorData(get_degeneracystructure(right_space).total_dim),
    )

    with pytest.raises(ValueError, match="exactly once"):
        tensorcontract(
            left,
            right,
            axes=((1,), (0,)),
            output=((), ((1, 1),)),
        )


# Single-tensor trace primitive.
@pytest.mark.parametrize(
    ("metadata", "error", "match"),
    [
        ({"axes": [(0,), (1,)]}, TypeError, "axes.*tuple.*length 2"),
        ({"axes": ((0,), [1])}, TypeError, "trace partitions.*tuple"),
        ({"axes": ((True,), (1,))}, TypeError, "trace axis.*int"),
        ({"axes": ((-1,), (1,))}, ValueError, "trace axis.*out of range"),
        ({"axes": ((0, 1), (1,))}, ValueError, "same number"),
        ({"axes": ((0, 0), (1, 1))}, ValueError, "exactly once"),
        ({"output": [(), ()]}, TypeError, "output.*tuple.*length 2"),
        ({"output": ([], ())}, TypeError, "output partitions.*tuple"),
        ({"output": ((True,), ())}, TypeError, "output axis.*int"),
        ({"output": ((-1,), ())}, ValueError, "output axis.*out of range"),
        ({"conjugate": 0}, TypeError, "conjugate.*bool"),
    ],
    ids=[
        "axes-outer-not-tuple",
        "axes-partition-not-tuple",
        "bool-trace-axis",
        "negative-trace-axis",
        "unequal-pair-counts",
        "duplicate-trace-axis",
        "output-outer-not-tuple",
        "output-partition-not-tuple",
        "bool-output-axis",
        "negative-output-axis",
        "conjugate-non-bool",
    ],
)
def test_tensortrace_rejects_invalid_metadata_before_storage_access(
    metadata,
    error,
    match,
):
    tensor = _neutral_endomorphism()
    arguments = {
        "axes": ((0,), (1,)),
        "output": ((), ()),
        "conjugate": False,
    }
    arguments.update(metadata)

    with pytest.raises(error, match=match):
        tensortrace(tensor, **arguments)


def test_tensortrace_rejects_non_tensor_input():
    with pytest.raises(TypeError, match="tensor.*TensorMap"):
        tensortrace(
            object(),  # pyright: ignore[reportArgumentType]
            axes=((), ()),
            output=((), ()),
        )


def test_tensortrace_rejects_non_dual_trace_spaces_before_storage_access():
    charge_one = space(U1Irrep, {1: 1})
    charge_two = space(U1Irrep, {2: 1})
    tensor = _metadata_only_tensor(hom((charge_one,), (charge_two,)))

    with pytest.raises(ValueError, match="trace axes.*dual-compatible"):
        tensortrace(tensor, axes=((0,), (1,)), output=((), ()))


def test_tensortrace_empty_trace_accepts_output_permutation():
    first = space(U1Irrep, {0: 2})
    second = space(U1Irrep, {0: 3})
    third = space(U1Irrep, {0: 4})
    tensor = _tensor(hom((first, second), (third,)))
    expected = permute(tensor, ((2, 0), (1,)))

    result = tensortrace(
        tensor,
        axes=((), ()),
        output=((2, 0), (1,)),
    )

    assert result.space == expected.space
    assert_allclose(result.storage.data, expected.storage.data)


def test_tensortrace_empty_trace_identity_reuses_input(monkeypatch):
    factor = space(U1Irrep, {0: 2})
    tensor = _tensor(hom((factor,), (factor,)))

    def fail_metadata(*_args):
        raise AssertionError("zero trace must not request native trace metadata")

    monkeypatch.setattr(
        contractions._native,
        "trace_transformer",
        fail_metadata,
    )

    result = tensortrace(
        tensor,
        axes=((), ()),
        output=((0,), (1,)),
    )

    assert result is tensor


def test_tensortrace_rebuilds_trace_metadata_and_reuses_cached_basis(monkeypatch):
    factor = space(U1Irrep, {0: 2})
    tensor = _tensor(hom((factor,), (factor,)))
    native_trace_transformer = contractions._native.trace_transformer
    native_tree_braider = transforms._native.tree_braider
    trace_calls = 0
    basis_calls = 0

    def count_trace_calls(*args):
        nonlocal trace_calls
        trace_calls += 1
        return native_trace_transformer(*args)

    def count_basis_calls(*args):
        nonlocal basis_calls
        basis_calls += 1
        return native_tree_braider(*args)

    transforms._clear_tree_transformer_caches_for_tests()
    monkeypatch.setattr(
        contractions._native,
        "trace_transformer",
        count_trace_calls,
    )
    monkeypatch.setattr(transforms._native, "tree_braider", count_basis_calls)
    try:
        first = tensortrace(tensor, axes=((0,), (1,)), output=((), ()))
        second = tensortrace(tensor, axes=((0,), (1,)), output=((), ()))
    finally:
        transforms._clear_tree_transformer_caches_for_tests()

    assert trace_calls == 2
    assert basis_calls == 1
    assert_allclose(second.storage.data, first.storage.data)


def test_tensortrace_conjugation_remaps_original_codomain_axes():
    traced = space(U1Irrep, {0: 2})
    open_left = space(U1Irrep, {0: 3})
    open_middle = space(U1Irrep, {0: 4})
    open_right = space(U1Irrep, {0: 5})
    tensor = _tensor(
        hom(
            (traced, traced.dual()),
            (open_left, open_middle, open_right),
        ),
        dtype=jnp.complex64,
    )
    expected_destination = hom(
        (open_left, open_right),
        (open_middle.dual(),),
    )
    adjoint_space = hom(tensor.space.domain, tensor.space.codomain)
    assert adjoint_space[0].dual() != adjoint_space[1]
    assert adjoint_space[3].dual() == adjoint_space[4]

    result = tensortrace(
        tensor,
        axes=((0,), (1,)),
        output=((2, 4), (3,)),
        conjugate=True,
    )
    dense = to_dense(tensor)
    adjoint_dense = jnp.transpose(jnp.conj(dense), (2, 3, 4, 0, 1))
    canonical_dense = jnp.transpose(adjoint_dense, (0, 2, 3, 1, 4))
    expected = jnp.trace(canonical_dense, axis1=2, axis2=4)

    assert result.space == expected_destination
    assert_allclose(to_dense(result), expected)


def test_tensortrace_partial_trace_matches_dense_oracle():
    open_out = space(U1Irrep, {0: 2})
    traced = space(U1Irrep, {0: 3})
    open_in = space(U1Irrep, {0: 4})
    tensor = _tensor(hom((open_out, traced), (open_in, traced)))
    expected = jnp.trace(to_dense(tensor), axis1=1, axis2=3)

    result = tensortrace(
        tensor,
        axes=((1,), (3,)),
        output=((0,), (2,)),
    )

    assert result.space == hom((open_out,), (open_in,))
    assert_allclose(to_dense(result), expected)


@pytest.mark.parametrize(
    ("sector_type", "sector", "coefficient"),
    _SECTOR_CASES,
)
def test_tensortrace_full_trace_across_sector_families(
    sector_type,
    sector,
    coefficient,
):
    factor = space(sector_type, {sector: 1})
    tensor = TensorMap(
        hom((factor,), (factor,)),
        jnp.asarray([3.0], dtype=jnp.float32),
    )

    result = tensortrace(tensor, axes=((0,), (1,)), output=((), ()))

    assert result.numind == 0
    assert result.space.codomain.sector_spec == factor.sector_spec
    assert_allclose(scalar(result), coefficient * tensor.storage.data[0])


def test_tensortrace_su2_partial_trace_matches_dense_oracle():
    half = space(SU2Irrep, {1: 1})
    tensor = _tensor(hom((half, half), (half, half)))

    result = tensortrace(
        tensor,
        axes=((1,), (3,)),
        output=((0,), (2,)),
    )
    expected = jnp.trace(to_dense(tensor), axis1=1, axis2=3)

    assert result.space == hom((half,), (half,))
    assert_allclose(to_dense(result), expected)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.complex64])
def test_tensortrace_su2_grouped_transform_matches_dense_oracle(dtype, monkeypatch):
    half = space(SU2Irrep, {1: 1})
    source = hom(
        (half, half, half, half.dual()),
        (),
    )
    tensor = _tensor(source, dtype=dtype)

    def fail_permute(*_args):
        raise AssertionError("nonempty tensortrace must not build canonical storage")

    monkeypatch.setattr(contractions, "permute", fail_permute)
    result = tensortrace(
        tensor,
        axes=((3,), (0,)),
        output=((1, 2), ()),
    )
    canonical_dense = jnp.transpose(to_dense(tensor), (1, 2, 3, 0))
    expected = jnp.trace(canonical_dense, axis1=2, axis2=3)

    assert result.space == hom((half, half), ())
    assert_allclose(to_dense(result), expected)


def test_tensortrace_su2_grouped_transform_handles_multiple_destination_rows():
    half = space(SU2Irrep, {1: 1})
    unit = space(SU2Irrep, {0: 1})
    source = hom(
        (half, half, half, half, unit, unit.dual()),
        (),
    )
    destination = hom((half, half, half, half), ())
    tensor = _tensor(source)

    result = tensortrace(
        tensor,
        axes=((4,), (5,)),
        output=((1, 2, 3, 0), ()),
    )
    canonical_dense = jnp.transpose(to_dense(tensor), (1, 2, 3, 0, 4, 5))
    expected = jnp.trace(canonical_dense, axis1=4, axis2=5)

    assert result.space == destination
    assert_allclose(to_dense(result), expected)


def test_tensortrace_su2_grouped_transform_is_jittable_and_differentiable():
    half = space(SU2Irrep, {1: 1})
    source = hom((half, half, half, half.dual()), ())

    @jax.jit
    @jax.value_and_grad
    def traced_sum(data):
        result = tensortrace(
            TensorMap(source, data),
            axes=((3,), (0,)),
            output=((1, 2), ()),
        )
        return jnp.sum(result.storage.data)

    value, gradient = traced_sum(jnp.asarray([1.0, 2.0], dtype=jnp.float32))
    expected_gradient = jnp.asarray(
        [-(0.5**0.5), 1.5**0.5],
        dtype=jnp.float32,
    )

    assert_allclose(value, jnp.dot(expected_gradient, jnp.asarray([1.0, 2.0])))
    assert_allclose(gradient, expected_gradient)


def test_tensortrace_dual_fermion_orientation_matches_trace():
    factor = space(FermionParity, {1: 1}).dual()
    tensor = TensorMap(
        hom((factor,), (factor,)),
        jnp.asarray([3 + 4j], dtype=jnp.complex64),
    )

    result = tensortrace(tensor, axes=((0,), (1,)), output=((), ()))

    assert_allclose(scalar(result), tensor.storage.data[0])
    assert_allclose(scalar(result), tensor.tr())


def test_tensortrace_reversed_fermion_orientation_matches_canonical_trace():
    odd = space(FermionParity, {1: 1})
    tensor = TensorMap(
        hom((odd,), (odd,)),
        jnp.asarray([3 + 4j], dtype=jnp.complex64),
    )

    result = tensortrace(tensor, axes=((1,), (0,)), output=((), ()))
    canonical = permute(tensor, ((1,), (0,)))
    expected = tensortrace(canonical, axes=((0,), (1,)), output=((), ()))

    assert_allclose(result.storage.data, expected.storage.data)


def test_tensortrace_partial_trace_reorders_and_repartitions_open_axes():
    first_out = space(U1Irrep, {0: 2})
    traced = space(U1Irrep, {0: 2})
    second_out = space(U1Irrep, {0: 3})
    first_in = space(U1Irrep, {0: 4})
    second_in = space(U1Irrep, {0: 5})
    tensor = _tensor(
        hom(
            (first_out, traced, second_out),
            (first_in, traced, second_in),
        )
    )

    result = tensortrace(
        tensor,
        axes=((1,), (4,)),
        output=((5, 0), (2, 3)),
    )
    expected = jnp.trace(to_dense(tensor), axis1=1, axis2=4)
    expected = jnp.transpose(expected, (3, 0, 1, 2))

    assert result.space == hom(
        (second_in.dual(), first_out),
        (second_out.dual(), first_in),
    )
    assert_allclose(to_dense(result), expected)


def test_tensortrace_multiple_pairs_updates_column_axis_after_each_trace():
    open_out = space(U1Irrep, {0: 2})
    first_trace = space(U1Irrep, {0: 2})
    second_trace = space(U1Irrep, {0: 3})
    open_in = space(U1Irrep, {0: 4})
    tensor = _tensor(
        hom(
            (open_out, first_trace, second_trace),
            (open_in, first_trace, second_trace),
        )
    )

    result = tensortrace(
        tensor,
        axes=((1, 2), (4, 5)),
        output=((0,), (3,)),
    )
    expected = jnp.einsum("aijbij->ab", to_dense(tensor))

    assert result.space == hom((open_out,), (open_in,))
    assert_allclose(to_dense(result), expected)


def test_tensortrace_handles_strided_subblocks():
    first = space(U1Irrep, {0: 2, 1: 1})
    second = space(U1Irrep, {0: 1, 1: 1})
    traced = space(U1Irrep, {0: 1})
    source = hom((first, second, traced), (first, second, traced))
    destination = hom((first, second), (first, second))
    source_degeneracy = get_degeneracystructure(source)
    destination_degeneracy = get_degeneracystructure(destination)
    assert any(
        not is_contiguous_subblock(subblock)
        for subblock in source_degeneracy.subblockstructure
    )
    assert any(
        not is_contiguous_subblock(subblock)
        for subblock in destination_degeneracy.subblockstructure
    )
    tensor = _tensor(source)

    result = tensortrace(
        tensor,
        axes=((2,), (5,)),
        output=((0, 1), (3, 4)),
    )
    expected = jnp.trace(to_dense(tensor), axis1=2, axis2=5)

    assert result.space == destination
    assert_allclose(to_dense(result), expected)


@pytest.mark.parametrize(
    "dtype",
    [jnp.float32, jnp.complex64],
    ids=["float32", "complex64"],
)
def test_tensortrace_unit_coefficients_preserve_dtype(dtype):
    factor = space(U1Irrep, {0: 2})
    tensor = _tensor(hom((factor,), (factor,)), dtype=dtype)

    result = tensortrace(tensor, axes=((0,), (1,)), output=((), ()))

    assert result.numind == 0
    assert result.space.codomain.sector_spec == factor.sector_spec
    assert result.storage.data.dtype == dtype
    assert_allclose(scalar(result), tensor.tr())


def test_tensortrace_nonunit_coefficient_promotes_float16_storage():
    half = space(SU2Irrep, {1: 1})
    tensor = TensorMap(
        hom((half,), (half,)),
        jnp.asarray([3], dtype=jnp.float16),
    )

    result = tensortrace(tensor, axes=((0,), (1,)), output=((), ()))

    assert result.storage.data.dtype == jnp.float32
    assert_allclose(scalar(result), jnp.asarray(6, dtype=jnp.float32))


# Explicit contraction order.
def test_contract_custom_order_matches_reference():
    a = space(U1Irrep, {0: 5})
    x = space(U1Irrep, {0: 2})
    y = space(U1Irrep, {0: 3})
    b = space(U1Irrep, {0: 2})
    left = _tensor(hom((a,), (x,)))
    middle = _tensor(hom((x,), (y,)))
    right = _tensor(hom((y,), (b,)))

    result = contract(
        idx(left, "a,x"),
        idx(middle, "x,y"),
        idx(right, "y,b"),
        output=("a", "b"),
        order=["y", "x"],
    )
    expected = left @ (middle @ right)

    assert result.space == expected.space
    assert_allclose(result.storage.data, expected.storage.data)


# Named-index frontend.
def test_idx_and_output_forms_normalize_labels_and_empty_groups():
    factor = space(U1Irrep, {0: 1})
    tensor = _tensor(hom((factor, factor), (factor,)))

    indexed_result = contract(
        idx(tensor, " left, β,\n env "),
        output=(" β ", " env, left "),
    )
    tuple_result = contract(
        (tensor, "left,β,env"),
        output=" β ; env, left ",
    )
    expected = tensortrace(
        tensor,
        axes=((), ()),
        output=((1,), (2, 0)),
    )

    assert indexed_result.space == expected.space
    assert tuple_result.space == expected.space
    assert_allclose(indexed_result.storage.data, expected.storage.data)
    assert_allclose(tuple_result.storage.data, expected.storage.data)

    rank_zero = tensortrace(
        _tensor(hom((factor,), (factor,))),
        axes=((0,), (1,)),
        output=((), ()),
    )
    empty_string_result = contract(idx(rank_zero, " \n "), output=";")
    empty_tuple_result = contract((rank_zero, ""), output=("", ""))

    assert empty_string_result.space == rank_zero.space
    assert empty_tuple_result.space == rank_zero.space
    assert_allclose(empty_string_result.storage.data, rank_zero.storage.data)
    assert_allclose(empty_tuple_result.storage.data, rank_zero.storage.data)


@pytest.mark.parametrize(
    ("labels", "match"),
    [
        ("1a", "invalid label"),
        ("a b", "invalid label"),
        ("a,,b", "invalid label"),
        ("[a]", "invalid label"),
    ],
)
def test_idx_rejects_invalid_labels(labels, match):
    factor = space(U1Irrep, {0: 1})
    tensor = _tensor(hom((factor,), ()))

    with pytest.raises(ValueError, match=match):
        idx(tensor, labels)


def test_idx_rejects_non_string_labels():
    factor = space(U1Irrep, {0: 1})
    tensor = _tensor(hom((factor,), ()))

    with pytest.raises(TypeError, match="labels.*string"):
        idx(tensor, None)  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize(
    "output",
    [None, ("a",), (["a"], "b")],
)
def test_contract_rejects_invalid_output_structure(output):
    factor = space(U1Irrep, {0: 1})
    tensor = _tensor(hom((factor,), ()))

    with pytest.raises(TypeError, match="output.*string"):
        contract(idx(tensor, "a"), output=output)


@pytest.mark.parametrize(
    ("output", "match"),
    [
        ("a,b", "exactly one"),
        ("a;b;c", "exactly one"),
        ("a b;c", "invalid label"),
        ("a,,b;c", "invalid label"),
    ],
)
def test_contract_rejects_invalid_output_strings(output, match):
    factor = space(U1Irrep, {0: 1})
    tensor = _tensor(hom((factor,), ()))

    with pytest.raises(ValueError, match=match):
        contract(idx(tensor, "a"), output=output)


@pytest.mark.parametrize(
    ("order", "error", "match"),
    [
        ("x", TypeError, "order.*tuple or list"),
        ((1,), TypeError, "order.*only strings"),
        (("x y",), ValueError, "invalid label"),
        ((), ValueError, "every contracted label"),
        (("x", "x"), ValueError, "every contracted label"),
        (("x", "unknown"), ValueError, "every contracted label"),
    ],
)
def test_contract_rejects_invalid_order(order, error, match):
    factor = space(U1Irrep, {0: 1})
    left = _tensor(hom((factor,), (factor,)))
    right = _tensor(hom((factor,), (factor,)))

    with pytest.raises(error, match=match):
        contract(
            idx(left, "a,x"),
            idx(right, "x,b"),
            output=("a", "b"),
            order=order,
        )


def test_contract_rejects_missing_or_non_tensor_operands():
    factor = space(U1Irrep, {0: 1})
    tensor = _tensor(hom((factor,), (factor,)))

    with pytest.raises(ValueError, match="at least one operand"):
        contract(output=("", ""))
    with pytest.raises(TypeError, match="idx.*tuple"):
        contract(tensor, output=("a", "a"))  # pyright: ignore[reportArgumentType]
    with pytest.raises(TypeError, match="TensorMap"):
        idx(object(), "")  # pyright: ignore[reportArgumentType]


def test_contract_rejects_invalid_tuple_operands():
    factor = space(U1Irrep, {0: 1})
    tensor = _tensor(hom((factor,), (factor,)))

    for operand in ((tensor,), (tensor, "a,b", False, False), [tensor, "a,b"]):
        with pytest.raises(TypeError, match="idx.*tuple"):
            contract(operand, output="a;b")  # pyright: ignore[reportArgumentType]
    with pytest.raises(TypeError, match="TensorMap"):
        contract((object(), "a,b"), output="a;b")  # pyright: ignore[reportArgumentType]
    with pytest.raises(TypeError, match="labels.*string"):
        contract((tensor, ("a", "b")), output="a;b")  # pyright: ignore[reportArgumentType]
    with pytest.raises(TypeError, match="conjugate.*bool"):
        contract((tensor, "a,b", 1), output="a;b")  # pyright: ignore[reportArgumentType]


def test_idx_rejects_invalid_conjugation_metadata():
    factor = space(U1Irrep, {0: 1})
    tensor = _tensor(hom((factor,), (factor,)))

    with pytest.raises(TypeError, match="conjugate.*bool"):
        idx(tensor, "a,b", conjugate=1)  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize(
    ("labels", "output", "match"),
    [
        ("a", ("a", ""), "one label per tensor index"),
        ("a,b", ("a", ""), "contracted label 'b'.*exactly twice"),
        ("a,b", ("a,a", "b"), "output label.*exactly once"),
        ("a,b", ("unknown", "a"), "output label 'unknown'.*exactly once"),
        ("a,a,b", ("a,b", ""), "output label 'a'.*exactly once"),
        ("a,a,a", ("", ""), "contracted label 'a'.*exactly twice"),
    ],
)
def test_contract_rejects_invalid_label_coverage(labels, output, match):
    factor = space(U1Irrep, {0: 1})
    rank_two = _tensor(hom((factor,), (factor,)))
    rank_three = _tensor(hom((factor, factor, factor), ()))
    tensor = rank_three if len(labels.split(",")) == 3 else rank_two

    with pytest.raises(ValueError, match=match):
        contract(idx(tensor, labels), output=output)


def test_contract_rejects_non_dual_self_trace_before_storage_access():
    one = space(U1Irrep, {1: 1})
    two = space(U1Irrep, {2: 1})
    tensor = _metadata_only_tensor(hom((one,), (two,)))

    with pytest.raises(ValueError, match="label 'x'.*dual-compatible"):
        contract(idx(tensor, "x,x"), output=("", ""))


def test_contract_validates_later_edge_before_any_storage_access():
    neutral = space(U1Irrep, {0: 1})
    one = space(U1Irrep, {1: 1})
    two = space(U1Irrep, {2: 1})
    first = _metadata_only_tensor(hom((neutral,), (neutral,)))
    second = _metadata_only_tensor(hom((neutral,), (one,)))
    third = _metadata_only_tensor(hom((two,), (neutral,)))

    with pytest.raises(ValueError, match="label 'bad'.*dual-compatible"):
        contract(
            idx(first, "a,x"),
            idx(second, "x,bad"),
            idx(third, "bad,c"),
            output=("a", "c"),
        )


def test_contract_rejects_sector_family_mismatch_for_disconnected_network():
    u1 = space(U1Irrep, {0: 1})
    z2 = space(Z2Irrep, {0: 1})
    first = _metadata_only_tensor(hom((u1,), ()))
    second = _metadata_only_tensor(hom((z2,), ()))

    with pytest.raises(ValueError, match="same sector family"):
        contract(
            idx(first, "a"),
            idx(second, "b"),
            output=("a,b", ""),
        )


def test_contract_single_tensor_permutation_and_repartition():
    a = space(U1Irrep, {0: 2})
    b = space(U1Irrep, {0: 3})
    c = space(U1Irrep, {0: 4})
    tensor = _tensor(hom((a, b), (c,)))

    result = contract(idx(tensor, "a,b,c"), output=("c", "b,a"))
    expected = tensortrace(
        tensor,
        axes=((), ()),
        output=((2,), (1, 0)),
    )

    assert result.space == expected.space
    assert_allclose(result.storage.data, expected.storage.data)


def test_contract_single_tensor_conjugate_matches_adjoint():
    a = space(U1Irrep, {0: 2})
    b = space(U1Irrep, {0: 3})
    tensor = _tensor(hom((a,), (b,)), jnp.complex64)

    result = contract(
        idx(tensor, "a,b", conjugate=True),
        output=("b", "a"),
    )
    expected = tensor.adjoint()

    assert result.space == expected.space
    assert_allclose(result.storage.data, expected.storage.data)


def test_contract_self_trace_after_conjugation_uses_effective_partition():
    a = space(U1Irrep, {0: 2})
    b = space(U1Irrep, {0: 3})
    x = space(U1Irrep, {0: 4})
    tensor = _tensor(hom((a, x), (b, x)), jnp.complex64)

    result = contract(
        idx(tensor, "a,x,b,x", conjugate=True),
        output=("b", "a"),
    )
    expected = tensortrace(
        tensor,
        axes=((1,), (3,)),
        output=((2,), (0,)),
        conjugate=True,
    )

    assert result.space == expected.space
    assert_allclose(result.storage.data, expected.storage.data)


def test_contract_single_tensor_trace_matches_primitive():
    a = space(U1Irrep, {0: 2})
    x = space(U1Irrep, {0: 3})
    tensor = _tensor(hom((a, x), (a, x)))

    result = contract(
        idx(tensor, "a,x,a,x"),
        output=("", ""),
        order=("a", "x"),
    )
    expected = tensortrace(
        tensor,
        axes=((0, 1), (2, 3)),
        output=((), ()),
    )

    assert result.space == expected.space
    assert_allclose(result.storage.data, expected.storage.data)


def test_contract_two_tensors_matches_binary_primitive():
    a = space(U1Irrep, {0: 2})
    x = space(U1Irrep, {0: 3})
    b = space(U1Irrep, {0: 4})
    left = _tensor(hom((a,), (x,)))
    right = _tensor(hom((x,), (b,)))

    result = contract(
        (left, "a,x"),
        idx(right, "x,b"),
        output="a;b",
    )
    expected = tensorcontract(
        left,
        right,
        axes=((1,), (0,)),
        output=(((0, 0),), ((1, 1),)),
    )

    assert result.space == expected.space
    assert_allclose(result.storage.data, expected.storage.data)


def test_contract_binary_conjugation_matches_primitive():
    a = space(U1Irrep, {0: 2})
    x = space(U1Irrep, {0: 3})
    b = space(U1Irrep, {0: 4})
    left = _tensor(hom((x,), (a,)), jnp.complex64)
    right = _tensor(hom((x,), (b,)), jnp.complex64)

    result = contract(
        (left, "x,a", True),
        (right, "x,b"),
        output=("a", "b"),
    )
    expected = tensorcontract(
        left,
        right,
        axes=((0,), (0,)),
        output=(((0, 1),), ((1, 1),)),
        conjugate=(True, False),
    )

    assert result.space == expected.space
    assert_allclose(result.storage.data, expected.storage.data)


def test_contract_multiple_shared_labels_matches_binary_primitive():
    a = space(U1Irrep, {0: 2})
    x = space(U1Irrep, {0: 3})
    y = space(U1Irrep, {0: 4})
    b = space(U1Irrep, {0: 5})
    left = _tensor(hom((a,), (x, y)))
    right = _tensor(hom((x, y), (b,)))
    result = contract(
        idx(left, "a,x,y"),
        idx(right, "x,y,b"),
        output=("a", "b"),
        order=("y", "x"),
    )
    expected = tensorcontract(
        left,
        right,
        axes=((1, 2), (0, 1)),
        output=(((0, 0),), ((1, 2),)),
    )

    assert result.space == expected.space
    assert_allclose(result.storage.data, expected.storage.data)


def test_contract_three_tensors_matches_sequential_composition():
    a = space(U1Irrep, {0: 2})
    x = space(U1Irrep, {0: 3})
    y = space(U1Irrep, {0: 4})
    b = space(U1Irrep, {0: 5})
    tensors = (
        _tensor(hom((a,), (x,))),
        _tensor(hom((x,), (y,))),
        _tensor(hom((y,), (b,))),
    )
    result = contract(
        idx(tensors[0], "a,x"),
        idx(tensors[1], "x,y"),
        idx(tensors[2], "y,b"),
        output=("a", "b"),
    )
    expected = tensors[0] @ tensors[1] @ tensors[2]

    assert result.space == expected.space
    assert_allclose(result.storage.data, expected.storage.data)


def test_contract_disconnected_network_uses_tensor_product():
    a = space(U1Irrep, {0: 2})
    b = space(U1Irrep, {0: 3})
    first = _tensor(hom((a,), ()))
    second = _tensor(hom((b,), ()))

    result = contract(
        idx(first, "a"),
        idx(second, "b"),
        output=("b", "a"),
        order=(),
    )
    expected = tensorcontract(
        first,
        second,
        axes=((), ()),
        output=(((1, 0),), ((0, 0),)),
    )

    assert result.space == expected.space
    assert_allclose(result.storage.data, expected.storage.data)


@pytest.mark.parametrize(
    ("sector_type", "sector"),
    [
        pytest.param(SU2Irrep, 1, id="su2"),
        pytest.param(
            FermionParityU1SU2Irrep,
            (1, 0, 1),
            id="fermion-parity-u1-su2",
        ),
    ],
)
def test_contract_composition_supports_additional_sector_families(
    sector_type,
    sector,
):
    factor = space(sector_type, {sector: 1})
    left = _tensor(hom((factor,), (factor,)))
    right = _tensor(hom((factor,), (factor,)))

    result = contract(
        idx(left, "a,x"),
        idx(right, "x,b"),
        output=("a", "b"),
    )
    expected = left @ right

    assert result.space == expected.space
    assert_allclose(result.storage.data, expected.storage.data)


def test_ncon_default_and_explicit_output_match_named_contract():
    a = space(U1Irrep, {0: 2})
    x = space(U1Irrep, {0: 3})
    b = space(U1Irrep, {0: 4})
    left = _tensor(hom((a,), (x,)))
    right = _tensor(hom((x,), (b,)))

    default_result = ncon(
        [left, right],
        [[-1, 1], [1, -2]],
    )
    default_expected = contract(
        idx(left, "a,x"),
        idx(right, "x,b"),
        output=("a", "b"),
    )
    explicit_result = ncon(
        (left, right),
        ((-1, 1), (1, -2)),
        output=[[-2], [-1]],
    )
    explicit_expected = contract(
        idx(left, "a,x"),
        idx(right, "x,b"),
        output=("b", "a"),
    )

    assert default_result.space == default_expected.space
    assert explicit_result.space == explicit_expected.space
    assert_allclose(default_result.storage.data, default_expected.storage.data)
    assert_allclose(explicit_result.storage.data, explicit_expected.storage.data)


def test_ncon_default_output_sorts_negative_labels_within_each_side():
    first = space(U1Irrep, {0: 2})
    second = space(U1Irrep, {0: 3})
    third = space(U1Irrep, {0: 4})
    tensor = _tensor(hom((first, second), (third,)))

    result = ncon((tensor,), ((-3, -1, -2),))
    expected = contract(
        idx(tensor, "first,second,third"),
        output=("second,first", "third"),
    )

    assert result.space == expected.space
    assert_allclose(result.storage.data, expected.storage.data)


def test_ncon_custom_order_contracts_all_shared_labels_together():
    a = space(U1Irrep, {0: 2})
    x = space(U1Irrep, {0: 3})
    y = space(U1Irrep, {0: 4})
    b = space(U1Irrep, {0: 5})
    left = _tensor(hom((a,), (x, y)))
    right = _tensor(hom((x, y), (b,)))

    result = ncon(
        (left, right),
        ((-1, 1, 2), (1, 2, -2)),
        order=[2, 1],
    )
    expected = tensorcontract(
        left,
        right,
        axes=((1, 2), (0, 1)),
        output=(((0, 0),), ((1, 2),)),
    )

    assert result.space == expected.space
    assert_allclose(result.storage.data, expected.storage.data)


def test_ncon_self_trace_conjugation_consumes_order_label():
    a = space(U1Irrep, {0: 2})
    b = space(U1Irrep, {0: 3})
    x = space(U1Irrep, {0: 4})
    tensor = _tensor(hom((a, x), (b, x)), jnp.complex64)

    result = ncon(
        (tensor,),
        ((-1, 1, -2, 1),),
        conjugate=(True,),
        order=(1,),
    )
    expected = contract(
        idx(tensor, "a,x,b,x", conjugate=True),
        output=("b", "a"),
    )

    assert result.space == expected.space
    assert_allclose(result.storage.data, expected.storage.data)


def test_ncon_empty_order_disconnected_and_rank_zero_networks():
    a = space(U1Irrep, {0: 2})
    b = space(U1Irrep, {0: 3})
    tensor = _tensor(hom((a,), (b,)))

    copied = ncon((tensor,), ((-1, -2),), order=())
    assert copied.space == tensor.space
    assert_allclose(copied.storage.data, tensor.storage.data)

    first = _tensor(hom((a,), ()))
    second = _tensor(hom((b,), ()))
    disconnected = ncon(
        (first, second),
        ((-1,), (-2,)),
    )
    disconnected_expected = contract(
        idx(first, "a"),
        idx(second, "b"),
        output=("a,b", ""),
    )
    assert disconnected.space == disconnected_expected.space
    assert_allclose(
        disconnected.storage.data,
        disconnected_expected.storage.data,
    )

    factor = space(U1Irrep, {0: 2})
    left = _tensor(hom((factor,), (factor,)))
    right = _tensor(hom((factor,), (factor,)))
    rank_zero = ncon(
        (left, right),
        ((1, 2), (2, 1)),
        order=(1, 2),
    )
    rank_zero_expected = contract(
        idx(left, "x,y"),
        idx(right, "y,x"),
        output=("", ""),
    )
    assert rank_zero.numind == 0
    assert rank_zero.space == rank_zero_expected.space
    assert_allclose(rank_zero.storage.data, rank_zero_expected.storage.data)


def test_ncon_rejects_invalid_containers_labels_and_conjugation():
    factor = space(U1Irrep, {0: 1})
    tensor = _tensor(hom((factor,), (factor,)))

    with pytest.raises(TypeError, match="tensors.*tuple or list"):
        ncon(tensor, ((-1, -2),))  # pyright: ignore[reportArgumentType]
    with pytest.raises(TypeError, match="labels.*tuple or list"):
        ncon((tensor,), object())  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError, match="at least one tensor"):
        ncon((), ())
    with pytest.raises(ValueError, match="same length"):
        ncon((tensor,), ())
    with pytest.raises(TypeError, match="TensorMap"):
        ncon((object(),), ((-1, -2),))  # pyright: ignore[reportArgumentType]
    with pytest.raises(TypeError, match="labels for tensor 0.*tuple or list"):
        ncon((tensor,), ("-1,-2",))  # pyright: ignore[reportArgumentType]
    with pytest.raises(TypeError, match="only integers"):
        ncon((tensor,), ((True, -2),))
    with pytest.raises(ValueError, match="label 0"):
        ncon((tensor,), ((0, -1),))
    with pytest.raises(ValueError, match="visible rank"):
        ncon((tensor,), ((-1,),))
    with pytest.raises(TypeError, match="conjugate.*tuple or list"):
        ncon((tensor,), ((-1, -2),), conjugate=True)  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError, match="number of tensors"):
        ncon((tensor,), ((-1, -2),), conjugate=())
    with pytest.raises(TypeError, match="only bool"):
        ncon((tensor,), ((-1, -2),), conjugate=(0,))  # pyright: ignore[reportArgumentType]


def test_ncon_rejects_invalid_occurrences_order_and_output():
    factor = space(U1Irrep, {0: 1})
    tensor = _tensor(hom((factor,), (factor,)))
    left = _tensor(hom((factor,), (factor,)))
    right = _tensor(hom((factor,), (factor,)))
    tensors = (left, right)
    labels = ((-1, 1), (1, -2))

    with pytest.raises(ValueError, match="positive label 1.*twice"):
        ncon((tensor,), ((-1, 1),))
    with pytest.raises(ValueError, match="negative label -1.*once"):
        ncon((tensor,), ((-1, -1),))
    with pytest.raises(ValueError, match="every positive label"):
        ncon(tensors, labels, order=())
    with pytest.raises(ValueError, match="only positive"):
        ncon(tensors, labels, order=(-1,))
    with pytest.raises(ValueError, match="every positive label"):
        ncon(tensors, labels, order=(1, 1))
    with pytest.raises(TypeError, match="output partitions"):
        ncon(tensors, labels, output=(-1, -2))  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError, match="codomain and domain"):
        ncon(tensors, labels, output=((-1,),))  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError, match="only negative"):
        ncon(tensors, labels, output=((1,), (-2,)))
    with pytest.raises(ValueError, match="every negative label"):
        ncon(tensors, labels, output=((-1,), ()))


def test_ncon_validates_complete_network_before_storage_access():
    neutral = space(U1Irrep, {0: 1})
    one = space(U1Irrep, {1: 1})
    two = space(U1Irrep, {2: 1})
    first = _metadata_only_tensor(hom((neutral,), (neutral,)))
    second = _metadata_only_tensor(hom((neutral,), (one,)))
    third = _metadata_only_tensor(hom((two,), (neutral,)))

    with pytest.raises(ValueError, match="label 2.*dual-compatible"):
        ncon(
            (first, second, third),
            ((-1, 1), (1, 2), (2, -2)),
        )

    u1 = space(U1Irrep, {0: 1})
    z2 = space(Z2Irrep, {0: 1})
    u1_tensor = _metadata_only_tensor(hom((u1,), ()))
    z2_tensor = _metadata_only_tensor(hom((z2,), ()))
    with pytest.raises(ValueError, match="same sector family"):
        ncon(
            (u1_tensor, z2_tensor),
            ((-1,), (-2,)),
        )
