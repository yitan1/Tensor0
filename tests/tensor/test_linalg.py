import jax
import jax.numpy as jnp
import pytest

import tensor0
import tensor0.tensor as tensor_api
from tensor0 import (
    FermionParity,
    SU2Irrep,
    SectorDict,
    TensorMap,
    U1Irrep,
    contract,
    hom,
    idx,
    space,
    tensor_product,
    to_dense,
    zero_space,
)
from tensor0.structure import get_degeneracystructure
from tensor0.tensor.linalg import (
    inverse,
    is_isometric,
    is_positive_definite,
    is_unitary,
    left_solve,
    pseudoinverse,
    right_solve,
)
from tests.cases import (
    InaccessibleVectorData,
    assert_allclose,
    float_data_for,
    tensor_map_matmul_cases,
)


def _u1_hom():
    factor = space(U1Irrep, {0: 2, 1: 3})
    return hom((factor,), (factor,))


def _u1_hom_same_total_dim_with_different_metadata():
    factor = space(U1Irrep, {0: 2, 2: 3})
    return hom((factor,), (factor,))


def _u1_rectangular_hom():
    codomain = space(U1Irrep, {0: 2, 1: 3})
    domain = space(U1Irrep, {0: 3, 1: 2})
    return hom((codomain,), (domain,))


def _u1_data():
    return jnp.arange(13)


def _int_data_for(target):
    total_dim = get_degeneracystructure(target).total_dim
    return jnp.arange(1, total_dim + 1, dtype=jnp.int32)


def _tensor(target, dtype=jnp.float32):
    size = get_degeneracystructure(target).total_dim
    data = jnp.arange(1, size + 1, dtype=jnp.float32)
    if jnp.issubdtype(dtype, jnp.complexfloating):
        data = data + 1j * (data + 1)
    return TensorMap(target, data.astype(dtype))


def _metadata_only_tensor(target):
    size = get_degeneracystructure(target).total_dim
    return TensorMap(target, InaccessibleVectorData(size))


def _invertible_tensor(target, dtype=jnp.float32):
    zero = TensorMap(
        target,
        jnp.zeros(get_degeneracystructure(target).total_dim, dtype=dtype),
    )
    blocks = {}
    for index, (coupled, block) in enumerate(zero.blocks()):
        values = jnp.arange(block.size, dtype=jnp.float32).reshape(block.shape)
        values = values / max(block.size, 1)
        values = values + (index + block.shape[0] + 1) * jnp.eye(block.shape[0])
        if jnp.issubdtype(dtype, jnp.complexfloating):
            values = values + 0.1j * values.T
        blocks[coupled] = values.astype(dtype)
    return tensor0.from_blocks(target, blocks, dtype=dtype)


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


def _assert_tensormap_matmul_contract(case):
    left = _tensor(case.left_space)
    right = _tensor(case.right_space)

    result = left @ right

    assert isinstance(result, TensorMap)
    assert result.space == case.result_space

    left_blocks = dict(left.blocks())
    right_blocks = dict(right.blocks())
    for coupled, block in result.blocks():
        assert_allclose(block, left_blocks[coupled] @ right_blocks[coupled])


# Basic algebra and reductions.


def test_zero_like_preserves_space_dtype_and_zeroes_storage():
    target = _u1_hom()
    tensor = TensorMap(target, _int_data_for(target))

    result = tensor0.zero_like(tensor)

    assert result.space == target
    assert result.storage.data.dtype == tensor.storage.data.dtype
    assert_allclose(result.storage.data, jnp.zeros_like(tensor.storage.data))


def test_tensormap_scalar_and_linear_operations_match_functional_forms():
    target = _u1_hom()
    left = TensorMap(target, float_data_for(target))
    right = TensorMap(target, jnp.arange(left.dim, dtype=jnp.float32) * 0.25)

    negated = -left
    summed = left + right
    subtracted = left - right
    left_scaled = 2.5 * left
    right_scaled = left * 2.5
    divided = left / 2.0
    functional_add = tensor0.add(left, right, alpha=2.0, beta=-0.5)
    functional_scale = tensor_api.scale(left, 3.0)

    assert negated.space == target
    assert_allclose(negated.storage.data, -left.storage.data)
    assert_allclose(summed.storage.data, left.storage.data + right.storage.data)
    assert_allclose(subtracted.storage.data, left.storage.data - right.storage.data)
    assert_allclose(left_scaled.storage.data, 2.5 * left.storage.data)
    assert_allclose(right_scaled.storage.data, left.storage.data * 2.5)
    assert_allclose(divided.storage.data, left.storage.data / 2.0)
    assert_allclose(
        functional_add.storage.data,
        2.0 * left.storage.data - 0.5 * right.storage.data,
    )
    assert_allclose(functional_scale.storage.data, 3.0 * left.storage.data)


def test_tensormap_linear_operations_promote_dtype():
    target = _u1_hom()
    int_tensor = TensorMap(target, _int_data_for(target))
    float_tensor = TensorMap(target, float_data_for(target).astype(jnp.float32))

    added = tensor0.add(int_tensor, float_tensor)
    scalar = jnp.array(1.0 + 2.0j, dtype=jnp.complex64)
    complex_scaled = tensor_api.scale(int_tensor, scalar)

    assert added.storage.data.dtype == jnp.result_type(
        int_tensor.storage.data,
        float_tensor.storage.data,
        1,
        1,
    )
    assert_allclose(
        added.storage.data,
        int_tensor.storage.data + float_tensor.storage.data,
    )
    assert complex_scaled.storage.data.dtype == jnp.result_type(
        int_tensor.storage.data,
        scalar,
    )
    assert_allclose(complex_scaled.storage.data, int_tensor.storage.data * scalar)


def test_tensormap_add_rejects_space_mismatch_before_data_operations():
    left_space = _u1_hom()
    right_space = _u1_hom_same_total_dim_with_different_metadata()
    left = TensorMap(left_space, float_data_for(left_space))
    right = TensorMap(right_space, InaccessibleVectorData())

    with pytest.raises(ValueError, match="space|compatible|mismatch"):
        tensor0.add(left, right)


def test_tensormap_linear_functions_reject_non_tensormap_inputs():
    tensor = TensorMap(_u1_hom(), _u1_data())

    with pytest.raises(TypeError, match="zero_like.*TensorMap"):
        tensor0.zero_like(object())  # pyright: ignore[reportArgumentType, reportCallIssue]

    with pytest.raises(TypeError, match="scale.*TensorMap"):
        tensor0.scale(object(), 2.0)  # pyright: ignore[reportArgumentType, reportCallIssue]

    with pytest.raises(TypeError, match="add.*TensorMap"):
        tensor0.add(object(), tensor)  # pyright: ignore[reportArgumentType, reportCallIssue]

    with pytest.raises(TypeError, match="add.*TensorMap"):
        tensor0.add(tensor, object())  # pyright: ignore[reportArgumentType, reportCallIssue]


def test_tensormap_linear_operations_reject_non_scalar_coefficients():
    tensor = TensorMap(_u1_hom(), _u1_data())
    other = TensorMap(_u1_hom(), _u1_data())

    with pytest.raises((TypeError, ValueError), match="scalar"):
        tensor0.scale(tensor, jnp.ones((1,)))

    with pytest.raises((TypeError, ValueError), match="scalar"):
        tensor0.add(tensor, other, alpha=jnp.ones((1,)))

    with pytest.raises((TypeError, ValueError), match="scalar"):
        tensor0.add(tensor, other, beta=jnp.ones((1,)))

    with pytest.raises(TypeError):
        _ = tensor * other

    with pytest.raises(TypeError):
        _ = tensor / jnp.ones((1,))


def test_tensormap_linear_operations_are_jit_compatible():
    target = _u1_hom()
    tensor = TensorMap(target, float_data_for(target))

    result = jax.jit(lambda value: 2.0 * value + value)(tensor)

    assert result.space == target
    assert_allclose(result.storage.data, 3.0 * tensor.storage.data)


def test_tensormap_adjoint_is_jit_compatible():
    target = _u1_rectangular_hom()
    data = (
        jnp.arange(
            1,
            get_degeneracystructure(target).total_dim + 1,
            dtype=jnp.float32,
        )
        * jnp.array(1.0 + 2.0j, dtype=jnp.complex64)
    )
    tensor = TensorMap(target, data)
    expected = tensor.adjoint()

    result = jax.jit(lambda value: value.adjoint())(tensor)

    assert result.space == expected.space
    assert_allclose(result.storage.data, expected.storage.data)


def test_tensormap_inner_and_dot_conjugate_first_argument():
    target = _u1_hom()
    left = TensorMap(
        target,
        jnp.arange(
            1,
            get_degeneracystructure(target).total_dim + 1,
            dtype=jnp.float32,
        )
        * (1.0 + 2.0j),
    )
    right = TensorMap(
        target,
        jnp.arange(1, left.dim + 1, dtype=jnp.float32) * (3.0 - 1.0j),
    )
    expected = sum(
        jnp.vdot(left.block(coupled), right.block(coupled))
        for coupled, _block in left.blocks()
    )

    inner = tensor0.inner(left, right)

    assert_allclose(jnp.asarray(inner), jnp.asarray(expected))
    assert_allclose(jnp.asarray(tensor_api.dot(left, right)), jnp.asarray(inner))


def test_tensormap_inner_norm_and_trace_use_su2_quantum_dimension_weights():
    factor = space(SU2Irrep, {0: 1, 1: 1})
    target = hom((factor,), (factor,))
    left = tensor0.from_blocks(
        target,
        {
            0: jnp.array([[2.0]], dtype=jnp.float32),
            1: jnp.array([[3.0]], dtype=jnp.float32),
        },
    )
    right = tensor0.from_blocks(
        target,
        {
            0: jnp.array([[5.0]], dtype=jnp.float32),
            1: jnp.array([[7.0]], dtype=jnp.float32),
        },
    )

    expected_inner = 1.0 * 2.0 * 5.0 + 2.0 * 3.0 * 7.0
    expected_norm = jnp.sqrt(1.0 * 2.0**2 + 2.0 * 3.0**2)
    expected_trace = 1.0 * 2.0 + 2.0 * 3.0

    assert tensor0.inner(left, right) != jnp.vdot(
        left.storage.data,
        right.storage.data,
    )
    assert_allclose(jnp.asarray(tensor0.inner(left, right)), jnp.asarray(expected_inner))
    assert_allclose(jnp.asarray(tensor0.norm(left)), jnp.asarray(expected_norm))
    assert_allclose(jnp.asarray(tensor0.tr(left)), jnp.asarray(expected_trace))


def test_tensormap_norm_supports_p_values_and_rejects_invalid_p():
    factor = space(SU2Irrep, {0: 1, 1: 1})
    target = hom((factor,), (factor,))
    tensor = tensor0.from_blocks(
        target,
        {
            0: jnp.array([[-2.0]], dtype=jnp.float32),
            1: jnp.array([[3.0]], dtype=jnp.float32),
        },
    )

    assert_allclose(
        tensor0.norm(tensor, p=2),
        jnp.sqrt(jnp.real(tensor0.inner(tensor, tensor))),
    )
    assert_allclose(
        tensor0.norm(tensor, p=1),
        jnp.asarray(1.0 * 2.0 + 2.0 * 3.0),
    )
    assert_allclose(tensor_api.norm(tensor, p=jnp.inf), jnp.asarray(3.0))

    with pytest.raises(ValueError, match="norm.*p|p.*norm"):
        tensor0.norm(tensor, p=0)

    with pytest.raises(ValueError, match="norm.*p|p.*norm"):
        tensor0.norm(tensor, p=float("nan"))


def test_tensormap_norm_returns_real_dtype_for_complex_nondefault_p():
    target = _u1_hom()
    tensor = TensorMap(
        target,
        jnp.asarray(float_data_for(target), dtype=jnp.float32)
        * jnp.array(1.0 + 2.0j, dtype=jnp.complex64),
    )

    one_norm = tensor0.norm(tensor, p=1)
    inf_norm = tensor0.norm(tensor, p=jnp.inf)

    expected_abs = jnp.abs(tensor.storage.data)
    assert one_norm.dtype == expected_abs.dtype
    assert inf_norm.dtype == expected_abs.dtype
    assert_allclose(jnp.asarray(one_norm), jnp.sum(expected_abs))
    assert_allclose(jnp.asarray(inf_norm), jnp.max(expected_abs))


def test_tensormap_normalize_scales_storage_by_weighted_norm():
    target = _u1_hom()
    tensor = TensorMap(target, float_data_for(target))

    result = tensor0.normalize(tensor)

    assert isinstance(result, TensorMap)
    assert result.space == tensor.space
    assert_allclose(result.storage.data, tensor.storage.data / tensor0.norm(tensor))


def test_tensormap_adjoint_swaps_space_and_conjugates_blocks():
    target = _u1_rectangular_hom()
    tensor = tensor0.from_blocks(
        target,
        {
            0: jnp.array(
                [[1.0 + 2.0j, 3.0 - 4.0j, 5.0 + 6.0j], [7.0, 8.0j, 9.0]],
                dtype=jnp.complex64,
            ),
            1: jnp.array(
                [[2.0 - 1.0j, 3.0], [4.0j, 5.0], [6.0, 7.0 + 8.0j]],
                dtype=jnp.complex64,
            ),
        },
    )

    result = tensor0.adjoint(tensor)

    assert isinstance(result, TensorMap)
    assert result.space == hom(target.domain, target.codomain)
    for coupled, block in tensor.blocks():
        assert_allclose(result.block(coupled), jnp.conj(block.T))
    assert_allclose(tensor_api.adjoint(result).storage.data, tensor.storage.data)


def test_tensormap_component_wrappers_preserve_space():
    target = _u1_hom()
    data = (
        jnp.arange(
            1,
            get_degeneracystructure(target).total_dim + 1,
            dtype=jnp.float32,
        )
        * jnp.array(1.0 + 2.0j, dtype=jnp.complex64)
    )
    tensor = TensorMap(target, data)

    real_part = tensor_api.real(tensor)
    imag_part = tensor0.imag(tensor)
    complex_part = tensor.complex()

    for result in (real_part, imag_part, complex_part):
        assert isinstance(result, TensorMap)
        assert result.space == target

    assert_allclose(real_part.storage.data, jnp.real(data))
    assert_allclose(imag_part.storage.data, jnp.imag(data))
    assert complex_part.storage.data.dtype == jnp.result_type(data, 1j)
    assert_allclose(
        complex_part.storage.data,
        data.astype(complex_part.storage.data.dtype),
    )

    real_tensor = TensorMap(target, float_data_for(target).astype(jnp.float32))
    real_imag = tensor0.imag(real_tensor)
    real_complex = tensor0.complex(real_tensor)

    assert real_imag.space == target
    assert real_complex.space == target
    assert_allclose(real_imag.storage.data, jnp.zeros_like(real_tensor.storage.data))
    assert real_complex.storage.data.dtype == jnp.result_type(
        real_tensor.storage.data,
        1j,
    )
    assert_allclose(
        real_complex.storage.data,
        real_tensor.storage.data.astype(real_complex.storage.data.dtype),
    )


def test_tensormap_does_not_expose_ambiguous_public_conj():
    tensor = TensorMap(_u1_hom(), _u1_data())

    assert not hasattr(tensor, "conj")
    assert not hasattr(tensor0, "conj")
    assert not hasattr(tensor_api, "conj")


def test_tensormap_trace_rejects_non_endomorphism_before_data_operations():
    codomain = space(U1Irrep, {0: 2})
    domain = space(U1Irrep, {0: 3})
    target = hom((codomain,), (domain,))
    tensor = TensorMap(
        target,
        InaccessibleVectorData(get_degeneracystructure(target).total_dim),
    )

    with pytest.raises(ValueError, match="trace|domain|codomain|square"):
        tensor0.tr(tensor)


def test_tensormap_reductions_reject_invalid_inputs_and_space_mismatch():
    tensor = TensorMap(_u1_hom(), _u1_data())
    other = TensorMap(
        _u1_hom_same_total_dim_with_different_metadata(),
        InaccessibleVectorData(),
    )

    with pytest.raises(TypeError, match="inner.*TensorMap"):
        tensor0.inner(object(), tensor)  # pyright: ignore[reportArgumentType]

    with pytest.raises(TypeError, match="norm.*TensorMap"):
        tensor0.norm(object())  # pyright: ignore[reportArgumentType]

    with pytest.raises(TypeError, match="scalar.*TensorMap"):
        tensor0.scalar(object())  # pyright: ignore[reportArgumentType]

    with pytest.raises(TypeError, match="tr.*TensorMap"):
        tensor0.tr(object())  # pyright: ignore[reportArgumentType]

    with pytest.raises(TypeError, match="adjoint.*TensorMap"):
        tensor0.adjoint(object())  # pyright: ignore[reportArgumentType, reportCallIssue]

    with pytest.raises(TypeError, match="real.*TensorMap"):
        tensor0.real(object())  # pyright: ignore[reportArgumentType, reportCallIssue]

    with pytest.raises(TypeError, match="imag.*TensorMap"):
        tensor0.imag(object())  # pyright: ignore[reportArgumentType, reportCallIssue]

    with pytest.raises(TypeError, match="complex.*TensorMap"):
        tensor0.complex(object())  # pyright: ignore[reportArgumentType, reportCallIssue]

    with pytest.raises(ValueError, match="space|compatible|mismatch"):
        tensor0.inner(tensor, other)


# Diagonal construction and comparison.


def test_tensormap_diag_diagm_and_isdiag_round_trip_u1_blocks():
    codomain = space(U1Irrep, {0: 2, 1: 1})
    domain = space(U1Irrep, {0: 2, 1: 3})
    target = hom((codomain,), (domain,))
    tensor = tensor0.from_blocks(
        target,
        {
            0: jnp.diag(jnp.array([1.0, 2.0], dtype=jnp.float32)),
            1: jnp.array([[3.0, 0.0, 0.0]], dtype=jnp.float32),
        },
    )

    values = tensor0.diag(tensor)
    rebuilt = tensor_api.diagm(codomain, domain, values)

    assert isinstance(values, SectorDict)
    assert tuple(values) == ((0,), (1,))
    assert_allclose(values[0], jnp.array([1.0, 2.0], dtype=jnp.float32))
    assert_allclose(values[1], jnp.array([3.0], dtype=jnp.float32))
    assert tensor0.is_diagonal(tensor)
    assert tensor0.is_diagonal(rebuilt)
    assert_allclose(rebuilt.storage.data, tensor.storage.data)

    offdiagonal = tensor0.from_blocks(
        target,
        {
            0: jnp.array([[1.0, 4.0], [0.0, 2.0]], dtype=jnp.float32),
            1: jnp.array([[3.0, 0.0, 0.0]], dtype=jnp.float32),
        },
    )
    assert not tensor_api.is_diagonal(offdiagonal)


def test_tensormap_diag_and_diagm_support_multi_leg_blocks():
    left = space(U1Irrep, {0: 1, 1: 1})
    right = space(U1Irrep, {0: 1})
    target = hom((left, right), (right, left))
    blocks = {}
    zero = TensorMap(target, jnp.zeros(get_degeneracystructure(target).total_dim))
    for coupled, block in zero.blocks():
        block_array = jnp.zeros(block.shape, dtype=jnp.float32)
        diagonal_indices = jnp.arange(min(block.shape))
        blocks[coupled] = block_array.at[diagonal_indices, diagonal_indices].set(
            jnp.arange(1, len(diagonal_indices) + 1, dtype=jnp.float32),
        )
    tensor = tensor0.from_blocks(target, blocks)

    values = tensor0.diag(tensor)
    rebuilt = tensor0.diagm(target.codomain, target.domain, values)

    assert isinstance(values, SectorDict)
    for coupled, block in tensor.blocks():
        assert_allclose(values[coupled], jnp.diag(block))
    assert tensor0.is_diagonal(tensor)
    assert tensor0.is_diagonal(rebuilt)
    assert_allclose(rebuilt.storage.data, tensor.storage.data)


def test_tensormap_diag_diagm_and_is_diagonal_reject_invalid_inputs():
    factor = space(U1Irrep, {0: 2})
    values = {0: jnp.arange(2, dtype=jnp.float32)}

    with pytest.raises(TypeError, match="diag.*TensorMap"):
        tensor0.diag(object())  # pyright: ignore[reportArgumentType]

    with pytest.raises(TypeError, match="is_diagonal.*TensorMap"):
        tensor0.is_diagonal(object())  # pyright: ignore[reportArgumentType]

    with pytest.raises(TypeError, match="diagm.*ProductSpace|diagm.*space"):
        tensor0.diagm(object(), factor, values)  # pyright: ignore[reportArgumentType]

    with pytest.raises(TypeError, match="diagm.*mapping|diagm.*values"):
        tensor0.diagm(factor, factor, object())  # pyright: ignore[reportArgumentType]


def test_tensormap_equal_is_exact_space_and_dtype_sensitive():
    target = _u1_hom()
    data = jnp.arange(get_degeneracystructure(target).total_dim, dtype=jnp.float32)
    tensor = TensorMap(target, data)
    same = TensorMap(target, data.copy())
    changed = TensorMap(target, data.at[0].set(-1))
    changed_dtype = TensorMap(target, data.astype(jnp.complex64))
    changed_space = TensorMap(
        _u1_hom_same_total_dim_with_different_metadata(),
        InaccessibleVectorData(),
    )

    results = (
        tensor_api.equal(tensor, same),
        tensor_api.equal(tensor, changed),
        tensor_api.equal(tensor, changed_dtype),
        tensor_api.equal(tensor, changed_space),
    )

    assert all(result.shape == () for result in results)
    assert all(result.dtype == jnp.dtype(jnp.bool_) for result in results)
    assert bool(results[0])
    assert not bool(results[1])
    assert not bool(results[2])
    assert not bool(results[3])


def test_tensormap_allclose_uses_explicit_tolerances_and_dtype_promotion():
    target = _u1_hom()
    size = get_degeneracystructure(target).total_dim
    integers = TensorMap(target, jnp.arange(size, dtype=jnp.int32))
    close_floats = TensorMap(
        target,
        jnp.arange(size, dtype=jnp.float32).at[1].add(0.05),
    )
    changed_space = TensorMap(
        _u1_hom_same_total_dim_with_different_metadata(),
        InaccessibleVectorData(),
    )

    close = tensor_api.allclose(integers, close_floats, rtol=0.0, atol=0.1)
    far = tensor_api.allclose(integers, close_floats, rtol=0.0, atol=0.01)
    mismatch = tensor_api.allclose(
        integers,
        changed_space,
        rtol=0.0,
        atol=0.1,
    )

    for result in (close, far, mismatch):
        assert result.shape == ()
        assert result.dtype == jnp.dtype(jnp.bool_)
    assert bool(close)
    assert not bool(far)
    assert not bool(mismatch)

    with pytest.raises(TypeError):
        tensor_api.allclose(  # pyright: ignore[reportCallIssue]
            integers,
            close_floats,
        )
    with pytest.raises(TypeError):
        tensor_api.allclose(
            integers,
            close_floats,
            0.0,  # pyright: ignore[reportCallIssue]
            0.1,
        )


@pytest.mark.parametrize("operation", [tensor_api.equal, tensor_api.allclose])
def test_tensormap_comparison_helpers_reject_invalid_inputs(operation):
    tensor = TensorMap(_u1_hom(), _u1_data())
    kwargs = {} if operation is tensor_api.equal else {"rtol": 0.0, "atol": 0.0}

    with pytest.raises(TypeError, match=rf"{operation.__name__}.*TensorMap"):
        operation(object(), tensor, **kwargs)
    with pytest.raises(TypeError, match=rf"{operation.__name__}.*TensorMap"):
        operation(tensor, object(), **kwargs)


def test_tensormap_diagm_rejects_missing_unexpected_and_wrong_shape_blocks():
    codomain = space(U1Irrep, {0: 2})
    domain = space(U1Irrep, {0: 2})

    with pytest.raises(ValueError, match="missing"):
        tensor0.diagm(codomain, domain, {})

    with pytest.raises(ValueError, match="unexpected"):
        tensor0.diagm(codomain, domain, {0: jnp.arange(2), 1: jnp.arange(1)})

    with pytest.raises(ValueError, match="shape|length"):
        tensor0.diagm(codomain, domain, {0: jnp.arange(3)})

    tensor = tensor0.diagm(
        codomain,
        domain,
        {0: jnp.arange(2), 1: jnp.arange(0)},
    )
    assert_allclose(tensor.block(0), jnp.diag(jnp.arange(2)))


# Inverse, pseudoinverse, and direct solves.


@pytest.mark.parametrize(
    ("sector_type", "sector_dims", "dtype"),
    [
        (U1Irrep, {0: 2, 1: 1}, jnp.float32),
        (SU2Irrep, {0: 2, 1: 1}, jnp.float32),
        (FermionParity, {0: 2, 1: 1}, jnp.float32),
        (U1Irrep @ FermionParity, {(0, 0): 2, (1, 1): 1}, jnp.complex64),
    ],
    ids=["u1", "su2", "fermionic", "product-sector-complex"],
)
def test_inverse_matches_blockwise_oracle_and_two_sided_identities(
    sector_type,
    sector_dims,
    dtype,
):
    factor = space(sector_type, sector_dims)
    value = _invertible_tensor(hom((factor,), (factor,)), dtype=dtype)

    result = inverse(value)

    assert result.space == hom(value.domain, value.codomain)
    for coupled, block in value.blocks():
        assert_allclose(result.block(coupled), jnp.linalg.inv(block))
        assert_allclose(block @ result.block(coupled), jnp.eye(block.shape[0]))
        assert_allclose(result.block(coupled) @ block, jnp.eye(block.shape[0]))


def test_inverse_preserves_reversed_multileg_product_space_metadata():
    left = space(U1Irrep, {0: 1, 1: 1})
    right = space(U1Irrep, {0: 1, -1: 1})
    target = hom((left, right), (right, left))
    value = _invertible_tensor(target)

    result = value.inverse()

    assert result.space == hom(target.domain, target.codomain)
    assert result.codomain == target.domain
    assert result.domain == target.codomain
    assert_allclose((value @ result).storage.data, tensor0.identity(target.codomain).storage.data)
    assert_allclose((result @ value).storage.data, tensor0.identity(target.domain).storage.data)


def test_pseudoinverse_satisfies_moore_penrose_identities_and_cutoff():
    codomain = space(U1Irrep, {0: 3, 1: 2})
    domain = space(U1Irrep, {0: 2, 1: 3})
    target = hom((codomain,), (domain,))
    value = tensor0.from_blocks(
        target,
        {
            0: jnp.asarray(
                [[4.0, 0.0], [0.0, 0.1], [0.0, 0.0]],
                dtype=jnp.float32,
            ),
            1: jnp.asarray(
                [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
                dtype=jnp.float32,
            ),
        },
    )

    result = pseudoinverse(value, atol=0.0, rtol=0.0)
    cutoff_result = pseudoinverse(value, atol=0.2, rtol=0.0)

    assert result.space == hom(target.domain, target.codomain)
    assert_allclose(
        cutoff_result.block(0),
        jnp.asarray([[0.25, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    )
    assert_allclose(
        cutoff_result.block(1),
        jnp.asarray([[1.0, 0.0], [0.0, 0.5], [0.0, 0.0]]),
    )
    assert_allclose((value @ result @ value).storage.data, value.storage.data)
    assert_allclose((result @ value @ result).storage.data, result.storage.data)
    assert_allclose(
        (value @ result).adjoint().storage.data,
        (value @ result).storage.data,
    )
    assert_allclose(
        (result @ value).adjoint().storage.data,
        (result @ value).storage.data,
    )


def test_inverse_and_pseudoinverse_cover_scalar_empty_diagonal_jit_and_grad():
    scalar_space = hom((), (), sector_type=U1Irrep)
    scalar = TensorMap(scalar_space, jnp.asarray([4.0], dtype=jnp.float32))
    empty_factor = zero_space(U1Irrep)
    empty = TensorMap(
        hom((empty_factor,), (empty_factor,)),
        jnp.zeros((0,), dtype=jnp.float32),
    )
    diagonal = tensor0.DiagonalTensorMap(
        space(U1Irrep, {0: 2}),
        jnp.asarray([2.0, 0.0], dtype=jnp.float32),
    )

    compiled_inverse = jax.jit(inverse)(scalar)
    compiled_pseudoinverse = jax.jit(
        lambda value: pseudoinverse(value, atol=0.0, rtol=0.0),
    )(scalar)
    gradient = jax.grad(
        lambda value: jnp.sum(value.pseudoinverse(rtol=0.0).storage.data),
    )(scalar)

    assert_allclose(compiled_inverse.storage.data, jnp.asarray([0.25]))
    assert_allclose(compiled_pseudoinverse.storage.data, jnp.asarray([0.25]))
    assert_allclose(gradient.storage.data, jnp.asarray([-1.0 / 16.0]))
    assert inverse(empty).storage.data.shape == (0,)
    assert pseudoinverse(empty).storage.data.shape == (0,)
    assert isinstance(inverse(diagonal), tensor0.DiagonalTensorMap)
    assert_allclose(
        pseudoinverse(diagonal, rtol=0.0).storage.data,
        jnp.asarray([0.5, 0.0]),
    )


def test_pseudoinverse_default_rtol_matches_diagonal_reduced_blocks():
    diagonal = tensor0.DiagonalTensorMap(
        space(SU2Irrep, {0: 1, 2: 1}),
        jnp.asarray([1.0, 2.0e-6], dtype=jnp.float32),
    )

    diagonal_result = pseudoinverse(diagonal)
    ordinary_result = pseudoinverse(diagonal.to_tensor_map())

    assert_allclose(
        diagonal_result.to_tensor_map().storage.data,
        ordinary_result.storage.data,
    )
    assert_allclose(diagonal_result.block(2), jnp.asarray([[5.0e5]]))


def test_inverse_and_pseudoinverse_validate_before_numerical_storage_access():
    codomain = space(U1Irrep, {0: 2})
    domain = space(U1Irrep, {0: 3})
    rectangular_space = hom((codomain,), (domain,))
    rectangular = _metadata_only_tensor(rectangular_space)

    with pytest.raises(ValueError, match="inverse.*isomorphic|square"):
        inverse(rectangular)
    with pytest.raises(ValueError, match="atol.*non-negative"):
        pseudoinverse(rectangular, atol=-1.0)
    with pytest.raises(ValueError, match="rtol.*non-negative"):
        pseudoinverse(rectangular, rtol=-1.0)
    with pytest.raises(TypeError, match="inverse.*TensorMap"):
        inverse(object())  # pyright: ignore[reportArgumentType, reportCallIssue]
    with pytest.raises(TypeError, match="pseudoinverse.*TensorMap"):
        pseudoinverse(  # pyright: ignore[reportCallIssue]
            object(),  # pyright: ignore[reportArgumentType]
        )


def test_left_and_right_solve_have_explicit_equation_orientations():
    middle = space(U1Irrep, {0: 2, 1: 1})
    right = space(U1Irrep, {0: 1, 1: 2})
    left = space(U1Irrep, {0: 3, 1: 2})
    operator = tensor0.from_blocks(
        hom((middle,), (middle,)),
        {
            0: jnp.asarray([[2.0, 1.0], [0.0, 3.0]]),
            1: jnp.asarray([[4.0]]),
        },
    )
    rhs = _tensor(hom((middle,), (right,)))
    lhs = _tensor(hom((left,), (middle,)))

    left_result = left_solve(operator, rhs)
    right_result = right_solve(lhs, operator)

    assert left_result.space == hom(operator.domain, rhs.domain)
    assert right_result.space == hom(lhs.codomain, operator.codomain)
    assert_allclose((operator @ left_result).storage.data, rhs.storage.data)
    assert_allclose((right_result @ operator).storage.data, lhs.storage.data)
    for coupled, block in rhs.blocks():
        assert_allclose(
            left_result.block(coupled),
            jnp.linalg.solve(operator.block(coupled), block),
        )
    for coupled, block in lhs.blocks():
        assert_allclose(
            right_result.block(coupled),
            jnp.linalg.solve(operator.block(coupled).T, block.T).T,
        )


@pytest.mark.parametrize(
    ("sector_type", "sector_dims"),
    [
        (SU2Irrep, {0: 2, 1: 1}),
        (FermionParity, {0: 2, 1: 1}),
        (U1Irrep @ FermionParity, {(0, 0): 2, (1, 1): 1}),
    ],
    ids=["su2", "fermionic", "product-sector"],
)
def test_direct_solves_cover_sector_families_jit_and_grad(
    sector_type,
    sector_dims,
):
    factor = space(sector_type, sector_dims)
    target = hom((factor,), (factor,))
    operator = _invertible_tensor(target)
    rhs = _tensor(target)

    left_result = jax.jit(left_solve)(operator, rhs)
    right_result = jax.jit(right_solve)(rhs, operator)
    gradient = jax.grad(
        lambda value: jnp.sum(left_solve(operator, value).storage.data),
    )(rhs)

    assert_allclose((operator @ left_result).storage.data, rhs.storage.data)
    assert_allclose((right_result @ operator).storage.data, rhs.storage.data)
    assert bool(jnp.all(jnp.isfinite(gradient.storage.data)))


def test_direct_solves_cover_scalar_and_empty_spaces():
    scalar_space = hom((), (), sector_type=U1Irrep)
    operator = TensorMap(scalar_space, jnp.asarray([2.0]))
    rhs = TensorMap(scalar_space, jnp.asarray([6.0]))
    empty_factor = zero_space(U1Irrep)
    empty_space = hom((empty_factor,), (empty_factor,))
    empty = TensorMap(empty_space, jnp.zeros((0,), dtype=jnp.float32))

    assert_allclose(left_solve(operator, rhs).storage.data, jnp.asarray([3.0]))
    assert_allclose(right_solve(rhs, operator).storage.data, jnp.asarray([3.0]))
    assert left_solve(empty, empty).storage.data.shape == (0,)
    assert right_solve(empty, empty).storage.data.shape == (0,)


def test_direct_solves_validate_complete_equations_before_storage_access():
    operator_factor = space(U1Irrep, {0: 2})
    incompatible_factor = space(U1Irrep, {0: 3})
    operator_space = hom((operator_factor,), (operator_factor,))
    operator = _metadata_only_tensor(operator_space)
    incompatible = _metadata_only_tensor(
        hom((incompatible_factor,), (incompatible_factor,)),
    )

    with pytest.raises(ValueError, match="left_solve.*codomain"):
        left_solve(operator, incompatible)
    with pytest.raises(ValueError, match="right_solve.*domain"):
        right_solve(incompatible, operator)
    with pytest.raises(TypeError, match="left_solve.*TensorMap"):
        left_solve(object(), incompatible)  # pyright: ignore[reportArgumentType]
    with pytest.raises(TypeError, match="right_solve.*TensorMap"):
        right_solve(incompatible, object())  # pyright: ignore[reportArgumentType]


# Numerical structure predicates.


def test_isometric_and_unitary_predicates_cover_embeddings_and_tolerances():
    codomain = space(U1Irrep, {0: 3, 1: 2})
    domain = space(U1Irrep, {0: 2, 1: 1})
    embedding = tensor0.isometry(codomain, domain, dtype=jnp.float32)
    square = tensor0.unitary(domain, domain, dtype=jnp.complex64)
    perturbed = TensorMap(
        square.space,
        square.storage.data.at[0].add(1.0e-4),
    )
    coisometry = embedding.adjoint()

    assert bool(is_isometric(embedding))
    assert not bool(is_isometric(embedding, side="right"))
    assert not bool(is_isometric(coisometry))
    assert bool(is_isometric(coisometry, side="right"))
    assert not bool(is_unitary(embedding))
    assert bool(is_isometric(square))
    assert bool(is_unitary(square))
    assert not bool(is_unitary(perturbed))
    assert bool(is_unitary(perturbed, atol=3.0e-4))


def test_isometric_and_unitary_static_false_avoids_numerical_storage():
    small = space(U1Irrep, {0: 1})
    large = space(U1Irrep, {0: 2})
    impossible_isometry = _metadata_only_tensor(hom((small,), (large,)))
    impossible_coisometry = _metadata_only_tensor(hom((large,), (small,)))
    nonisomorphic = _metadata_only_tensor(hom((small,), (large,)))

    isometric = is_isometric(impossible_isometry)
    coisometric = is_isometric(impossible_coisometry, side="right")
    unitary = is_unitary(nonisomorphic)

    assert isometric.shape == () and isometric.dtype == jnp.dtype(jnp.bool_)
    assert coisometric.shape == () and coisometric.dtype == jnp.dtype(jnp.bool_)
    assert unitary.shape == () and unitary.dtype == jnp.dtype(jnp.bool_)
    assert not bool(isometric)
    assert not bool(coisometric)
    assert not bool(unitary)


def test_positive_definite_checks_hermiticity_spectrum_and_cutoff():
    factor = space(U1Irrep, {0: 2, 1: 1})
    target = hom((factor,), (factor,))
    positive = tensor0.from_blocks(
        target,
        {
            0: jnp.asarray([[2.0, 0.5j], [-0.5j, 3.0]], dtype=jnp.complex64),
            1: jnp.asarray([[0.1]], dtype=jnp.complex64),
        },
    )
    indefinite = tensor0.from_blocks(
        target,
        {0: jnp.diag(jnp.asarray([1.0, -0.1])), 1: jnp.asarray([[2.0]])},
    )
    nonhermitian = tensor0.from_blocks(
        target,
        {0: jnp.asarray([[2.0, 1.0], [0.0, 3.0]]), 1: jnp.asarray([[1.0]])},
    )

    assert bool(is_positive_definite(positive))
    assert not bool(is_positive_definite(positive, atol=0.2))
    assert not bool(is_positive_definite(indefinite))
    assert not bool(is_positive_definite(nonhermitian))


def test_numerical_predicates_are_jittable_scalar_booleans_and_empty_true():
    factor = space(SU2Irrep, {0: 1, 1: 1})
    unitary_value = tensor0.unitary(factor, factor)
    positive = tensor0.identity(factor)
    empty_factor = zero_space(U1Irrep)
    empty = TensorMap(
        hom((empty_factor,), (empty_factor,)),
        jnp.zeros((0,), dtype=jnp.float32),
    )

    results = (
        jax.jit(is_isometric)(unitary_value),
        jax.jit(lambda value: is_isometric(value, side="right"))(unitary_value),
        jax.jit(is_unitary)(unitary_value),
        jax.jit(is_positive_definite)(positive),
        is_isometric(empty),
        is_isometric(empty, side="right"),
        is_unitary(empty),
        is_positive_definite(empty),
    )

    assert all(result.shape == () for result in results)
    assert all(result.dtype == jnp.dtype(jnp.bool_) for result in results)
    assert all(bool(result) for result in results)


def test_isometric_predicate_rejects_unknown_side():
    value = _invertible_tensor(_u1_hom())

    with pytest.raises(ValueError, match="side.*left.*right"):
        is_isometric(
            value,
            side="center",  # pyright: ignore[reportArgumentType]
        )


@pytest.mark.parametrize(
    "operation",
    [is_isometric, is_unitary, is_positive_definite],
)
def test_numerical_predicates_reject_invalid_inputs_and_tolerances(operation):
    value = _invertible_tensor(_u1_hom())

    with pytest.raises(TypeError, match=rf"{operation.__name__}.*TensorMap"):
        operation(object())
    with pytest.raises(ValueError, match="atol.*non-negative"):
        operation(value, atol=-1.0)
    with pytest.raises(ValueError, match="rtol.*non-negative"):
        operation(value, rtol=-1.0)


def test_positive_definite_rejects_non_endomorphism_before_storage_access():
    rectangular = _metadata_only_tensor(_u1_rectangular_hom())

    with pytest.raises(ValueError, match="endomorphism"):
        is_positive_definite(rectangular)


# Composition.


@pytest.mark.parametrize("case", tensor_map_matmul_cases(), ids=lambda case: case.name)
def test_tensormap_satisfies_matmul_contract(case):
    _assert_tensormap_matmul_contract(case)


def test_tensormap_matmul_zero_fills_result_sector_missing_from_middle_space():
    left_space = space(U1Irrep, {0: 2, 1: 3})
    middle_space = space(U1Irrep, {0: 5})
    right_space = space(U1Irrep, {0: 7, 1: 11})
    a_space = hom((left_space,), (middle_space,))
    b_space = hom((middle_space,), (right_space,))
    left = TensorMap(a_space, float_data_for(a_space))
    right = TensorMap(b_space, float_data_for(b_space))

    result = left @ right

    assert result.space == hom((left_space,), (right_space,))
    assert_allclose(result.block(0), left.block(0) @ right.block(0))
    assert_allclose(
        result.block(1),
        jnp.zeros((3, 11), dtype=result.storage.data.dtype),
    )


def test_tensormap_matmul_promotes_dtype_for_products_and_zero_blocks():
    left_space = space(U1Irrep, {0: 2, 1: 3})
    middle_space = space(U1Irrep, {0: 5})
    right_space = space(U1Irrep, {0: 7, 1: 11})
    a_space = hom((left_space,), (middle_space,))
    b_space = hom((middle_space,), (right_space,))
    left = TensorMap(a_space, _int_data_for(a_space))
    right = TensorMap(b_space, float_data_for(b_space))

    result = left @ right

    expected_dtype = jnp.result_type(left.storage.data, right.storage.data)
    assert result.storage.data.dtype == expected_dtype
    for _coupled, block in result.blocks():
        assert block.dtype == expected_dtype


def test_tensormap_matmul_rejects_incompatible_middle_space():
    left_space = space(U1Irrep, {0: 2})
    middle_space = space(U1Irrep, {0: 3})
    different_middle_space = space(U1Irrep, {0: 4})
    right_space = space(U1Irrep, {0: 5})
    a_space = hom((left_space,), (middle_space,))
    b_space = hom((different_middle_space,), (right_space,))
    left = TensorMap(a_space, float_data_for(a_space))
    right = TensorMap(b_space, float_data_for(b_space))

    with pytest.raises(ValueError, match="composable"):
        _ = left @ right


# Tensor product.


def test_tensor_product_fixes_multileg_output_order_and_matches_contract():
    left_out_0 = space(U1Irrep, {0: 2})
    left_out_1 = space(U1Irrep, {0: 3})
    left_in = space(U1Irrep, {0: 2})
    right_out = space(U1Irrep, {0: 2})
    right_in_0 = space(U1Irrep, {0: 3})
    right_in_1 = space(U1Irrep, {0: 2})
    left = _tensor(hom((left_out_0, left_out_1), (left_in,)))
    right = _tensor(hom((right_out,), (right_in_0, right_in_1)))
    output = (
        ((0, 0), (0, 1), (1, 0)),
        ((0, 2), (1, 1), (1, 2)),
    )

    result = tensor_product(left, right)
    expected = contract(
        idx(left, "left_out_0,left_out_1,left_in"),
        idx(right, "right_out,right_in_0,right_in_1"),
        output=(
            "left_out_0,left_out_1,right_out",
            "left_in,right_in_0,right_in_1",
        ),
        order=(),
    )
    expected_space = hom(
        (left_out_0, left_out_1, right_out),
        (left_in, right_in_0, right_in_1),
    )

    assert result.space == expected_space
    assert_allclose(result.storage.data, expected.storage.data)
    assert_allclose(to_dense(result), _dense_contract(left, right, ((), ()), output))


@pytest.mark.parametrize(
    ("sector_type", "sector_dims", "dtype"),
    [
        (SU2Irrep, {0: 1, 1: 1}, jnp.float32),
        (FermionParity, {0: 1, 1: 1}, jnp.float32),
        (U1Irrep @ FermionParity, {(0, 0): 1, (1, 1): 1}, jnp.complex64),
    ],
    ids=["su2", "fermionic", "product-sector-complex"],
)
def test_tensor_product_matches_sector_dense_oracles(
    sector_type,
    sector_dims,
    dtype,
):
    factor = space(sector_type, sector_dims)
    target = hom((factor,), (factor,))
    left = _tensor(target, dtype=dtype)
    right = _tensor(target)
    output = (((0, 0), (1, 0)), ((0, 1), (1, 1)))

    result = tensor_product(left, right)

    assert result.space == hom((factor, factor), (factor, factor))
    assert result.dtype == jnp.result_type(left.dtype, right.dtype)
    assert_allclose(to_dense(result), _dense_contract(left, right, ((), ()), output))


def test_tensor_product_includes_scalar_operands():
    scalar_space = hom((), (), sector_type=U1Irrep)
    left_scalar = TensorMap(scalar_space, jnp.asarray([2.5], dtype=jnp.float32))
    right_scalar = TensorMap(scalar_space, jnp.asarray([3.0], dtype=jnp.float32))
    factor = space(U1Irrep, {0: 2, 1: 1})
    tensor = _tensor(hom((factor,), (factor,)))

    scalar_tensor = tensor_product(left_scalar, tensor)
    tensor_scalar = tensor_product(tensor, left_scalar)
    scalar_scalar = tensor_product(left_scalar, right_scalar)

    assert scalar_tensor.space == tensor.space
    assert tensor_scalar.space == tensor.space
    assert_allclose(scalar_tensor.storage.data, 2.5 * tensor.storage.data)
    assert_allclose(tensor_scalar.storage.data, 2.5 * tensor.storage.data)
    assert scalar_scalar.numind == 0
    assert_allclose(scalar_scalar.scalar(), jnp.asarray(7.5, dtype=jnp.float32))


def test_tensor_product_handles_empty_spaces():
    empty = zero_space(U1Irrep)
    factor = space(U1Irrep, {0: 2})
    left = TensorMap(
        hom((empty,), ()),
        jnp.zeros((0,), dtype=jnp.float32),
    )
    right = _tensor(hom((factor,), ()))

    eager = tensor_product(left, right)
    compiled = jax.jit(tensor_product)(left, right)

    assert eager.space == hom((empty, factor), ())
    assert eager.storage.data.shape == (0,)
    assert compiled.space == eager.space
    assert compiled.storage.data.shape == (0,)


def test_tensor_product_is_jittable_and_differentiable():
    factor = space(U1Irrep, {0: 2})
    target = hom((factor,), (factor,))
    left = _tensor(target)
    right = _tensor(target)

    eager = tensor_product(left, right)
    compiled = jax.jit(tensor_product)(left, right)

    def squared_norm(left_value, right_value):
        product = tensor_product(left_value, right_value)
        return jnp.real(jnp.vdot(product.storage.data, product.storage.data))

    left_grad, right_grad = jax.jit(
        jax.grad(squared_norm, argnums=(0, 1)),
    )(left, right)
    left_norm_squared = jnp.vdot(left.storage.data, left.storage.data)
    right_norm_squared = jnp.vdot(right.storage.data, right.storage.data)

    assert_allclose(compiled.storage.data, eager.storage.data)
    assert left_grad.space == left.space
    assert right_grad.space == right.space
    assert_allclose(
        left_grad.storage.data,
        2 * left.storage.data * right_norm_squared,
    )
    assert_allclose(
        right_grad.storage.data,
        2 * right.storage.data * left_norm_squared,
    )


def test_tensor_product_rejects_invalid_operands_and_sector_mismatch():
    factor = space(U1Irrep, {0: 1})
    tensor = _tensor(hom((factor,), (factor,)))

    with pytest.raises(TypeError, match="left and right.*TensorMap"):
        tensor_product(object(), tensor)  # pyright: ignore[reportArgumentType]
    with pytest.raises(TypeError, match="left and right.*TensorMap"):
        tensor_product(tensor, object())  # pyright: ignore[reportArgumentType]

    su2_factor = space(SU2Irrep, {0: 1})
    left = _metadata_only_tensor(hom((factor,), ()))
    right = _metadata_only_tensor(hom((su2_factor,), ()))

    with pytest.raises(ValueError, match="same sector family"):
        tensor_product(left, right)
