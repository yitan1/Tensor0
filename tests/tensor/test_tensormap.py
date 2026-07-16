from dataclasses import FrozenInstanceError

import tensor0
import tensor0.tensor as tensor_api
import tensor0.tensor.tensor_map as tensor_map_module
import jax
import jax.numpy as jnp
import pytest

from tensor0 import (
    FermionNumber,
    FermionParity,
    FermionParityU1Irrep,
    SU2Irrep,
    SectorDict,
    TensorMap,
    U1Irrep,
    VectorStorage,
    Z2Irrep,
    Z3Irrep,
    Z4Irrep,
    _native,
    from_dense,
    hom,
    space,
    to_dense,
)
from tensor0.structure import get_degeneracystructure, get_sectorstructure
from tests.cases import (
    InaccessibleVectorData,
    assert_allclose,
    dense_roundtrip_cases,
    float_data_for,
    tensor_map_cases,
    tensor_map_matmul_cases,
)


def _u1_hom():
    v = space(U1Irrep, {0: 2, 1: 3})
    return hom((v,), (v,))


def _u1_hom_same_total_dim_with_different_metadata():
    v = space(U1Irrep, {0: 2, 2: 3})
    return hom((v,), (v,))


def _u1_rectangular_hom():
    codomain = space(U1Irrep, {0: 2, 1: 3})
    domain = space(U1Irrep, {0: 3, 1: 2})
    return hom((codomain,), (domain,))


def _u1_data():
    return jnp.arange(13)


def _int_data_for(h):
    total_dim = get_degeneracystructure(h).total_dim
    return jnp.arange(1, total_dim + 1, dtype=jnp.int32)


def _tensor_for(space_obj):
    return TensorMap(space_obj, float_data_for(space_obj))


def _metadata_cases():
    empty = _native.make_product_space(U1Irrep, ())
    u1 = space(U1Irrep, {0: 2, 1: 3})
    half = space(SU2Irrep, {1: 1})

    return (
        ("scalar", hom(empty, empty)),
        ("one-sided", hom((u1,), ())),
        ("u1", hom((u1,), (u1,))),
        ("su2", hom((half,), (half,))),
    )


def _assert_tensormap_storage_contract(case):
    data = float_data_for(case.space)
    storage = VectorStorage(data)
    tensor = TensorMap(case.space, storage)
    raw_tensor = TensorMap(case.space, data)

    assert tensor.space == case.space
    assert tensor.storage is storage
    assert tensor.storage.data is data
    assert raw_tensor.space == case.space
    assert isinstance(raw_tensor.storage, VectorStorage)
    assert raw_tensor.storage.data is data


def _assert_tensormap_blocks_contract(case):
    tensor = _tensor_for(case.space)
    sectorstructure = get_sectorstructure(case.space)

    blocks = tensor.blocks()

    assert isinstance(blocks, tuple)
    assert tuple(coupled for coupled, _block in blocks) == sectorstructure.blocksectors
    for coupled, block in blocks:
        assert_allclose(tensor.block(coupled), block)


def _assert_tensormap_metadata_contract(space_obj):
    tensor = _tensor_for(space_obj)
    degeneracystructure = get_degeneracystructure(space_obj)

    assert tensor.codomain == space_obj.codomain
    assert tensor.domain == space_obj.domain
    assert tensor.numout == space_obj.numout
    assert tensor.numin == space_obj.numin
    assert tensor.numind == space_obj.numind
    assert tensor.codomainind == tuple(range(space_obj.numout))
    assert tensor.domainind == tuple(range(space_obj.numout, space_obj.numind))
    assert tensor.allind == tuple(range(space_obj.numind))
    assert tensor.dim == degeneracystructure.total_dim
    assert tensor.dims == tuple(_native.product_dims(space_obj.codomain)) + tuple(
        _native.product_dims(space_obj.domain),
    )
    assert tensor.blocksectors == tuple(coupled for coupled, _block in tensor.blocks())
    assert len(tensor.fusiontrees) == len(tensor.subblocks())
    assert tensor.ndim == tensor.numind
    assert tensor.shape == tensor.dims
    assert tensor.dtype == tensor.storage.data.dtype
    assert tensor.output_axes == tensor.codomainind
    assert tensor.input_axes == tensor.domainind
    assert tensor.axes == tensor.allind
    assert tensor.block_sectors == tensor.blocksectors


def _assert_tensormap_matmul_contract(case):
    left = _tensor_for(case.left_space)
    right = _tensor_for(case.right_space)

    result = left @ right

    assert isinstance(result, TensorMap)
    assert result.space == case.result_space

    left_blocks = dict(left.blocks())
    right_blocks = dict(right.blocks())
    for coupled, block in result.blocks():
        assert_allclose(block, left_blocks[coupled] @ right_blocks[coupled])


def _u1_blocks_for(h):
    tensor = TensorMap(h, float_data_for(h))
    return dict(tensor.blocks())


def test_vector_storage_is_frozen_and_preserves_data():
    data = _u1_data()
    storage = VectorStorage(data)

    assert storage.data is data
    with pytest.raises(FrozenInstanceError):
        setattr(storage, "data", jnp.arange(13))


def test_vector_storage_rejects_string_and_bytes_data():
    with pytest.raises(TypeError, match="storage data"):
        VectorStorage("bad")

    with pytest.raises(TypeError, match="storage data"):
        VectorStorage(b"bad")


def test_tensormap_is_frozen_and_exposes_space_and_storage():
    h = _u1_hom()
    tensor = TensorMap(h, _u1_data())

    assert tensor.space is h
    with pytest.raises(FrozenInstanceError):
        setattr(tensor, "storage", tensor.storage)


@pytest.mark.parametrize("case", tensor_map_cases(), ids=lambda case: case.name)
def test_tensormap_satisfies_storage_contract(case):
    _assert_tensormap_storage_contract(case)


def test_tensormap_rejects_non_hom_space():
    with pytest.raises(TypeError, match="^TensorMap requires a HomSpace$"):
        TensorMap(object(), jnp.arange(1))  # pyright: ignore[reportArgumentType]


def test_tensormap_rejects_storage_data_without_shape():
    h = _u1_hom()

    with pytest.raises(TypeError, match="storage data.*shape"):
        TensorMap(h, object())


def test_tensormap_rejects_wrong_vector_data_length():
    h = _u1_hom()

    with pytest.raises(ValueError, match="expected.*13.*actual.*12"):
        TensorMap(h, jnp.arange(12))


def test_tensormap_rejects_non_1d_storage_data():
    h = _u1_hom()

    with pytest.raises(ValueError, match="1D"):
        TensorMap(h, jnp.zeros((13, 1)))


def test_tensormap_block_rejects_invalid_sector_keys():
    tensor = TensorMap(_u1_hom(), _u1_data())

    with pytest.raises(TypeError, match="sector key"):
        tensor.block("bad")  # pyright: ignore[reportArgumentType]

    with pytest.raises(KeyError):
        tensor.block(2)


@pytest.mark.parametrize("case", tensor_map_cases(), ids=lambda case: case.name)
def test_tensormap_satisfies_blocks_contract(case):
    _assert_tensormap_blocks_contract(case)


@pytest.mark.parametrize(
    "space_obj",
    [space_obj for _name, space_obj in _metadata_cases()],
    ids=[name for name, _space_obj in _metadata_cases()],
)
def test_tensormap_exposes_basic_metadata_interfaces(space_obj):
    _assert_tensormap_metadata_contract(space_obj)


def test_tensormap_hasblock_uses_sectorstructure_key_semantics():
    tensor = TensorMap(_u1_hom(), _u1_data())

    assert tensor.hasblock(0)
    assert tensor.hasblock((1,))
    assert not tensor.hasblock(2)

    with pytest.raises(TypeError, match="sector key"):
        tensor.hasblock("bad")

    product_case = next(
        case for case in tensor_map_cases() if case.name == "u1-fermion-product"
    )
    product_tensor = _tensor_for(product_case.space)

    with pytest.raises(ValueError, match="width"):
        product_tensor.hasblock((0,))


def test_tensormap_zeros_and_ones_allocate_expected_storage_and_blocks():
    h = _u1_hom()

    zero = tensor0.zeros(h)
    one = tensor_api.ones(h, dtype=jnp.float32)

    assert zero.space == h
    assert zero.storage.data.shape == (get_degeneracystructure(h).total_dim,)
    assert zero.storage.data.dtype == jnp.zeros(()).dtype
    for _coupled, block in zero.blocks():
        assert_allclose(block, jnp.zeros(block.shape, dtype=zero.storage.data.dtype))

    assert one.space == h
    assert one.storage.data.shape == (get_degeneracystructure(h).total_dim,)
    assert one.storage.data.dtype == jnp.dtype(jnp.float32)
    for _coupled, block in one.blocks():
        assert_allclose(block, jnp.ones(block.shape, dtype=jnp.float32))


def test_tensormap_zeros_and_ones_reject_non_hom_space():
    with pytest.raises(TypeError, match="zeros.*HomSpace"):
        tensor0.zeros(object())  # pyright: ignore[reportArgumentType]

    with pytest.raises(TypeError, match="ones.*HomSpace"):
        tensor0.ones(object())  # pyright: ignore[reportArgumentType]


def test_tensormap_copy_astype_and_similar_allocate_independent_storage():
    h = _u1_hom()
    tensor = TensorMap(h, float_data_for(h))
    other_h = _u1_hom_same_total_dim_with_different_metadata()

    copied = tensor.copy()
    cast = tensor.astype(jnp.complex64)
    similar = tensor.similar()
    retargeted = tensor.similar(space=other_h, dtype=jnp.int32)

    assert copied.space == h
    assert copied.storage.data is not tensor.storage.data
    assert_allclose(copied.storage.data, tensor.storage.data)

    assert cast.space == h
    assert cast.storage.data.dtype == jnp.dtype(jnp.complex64)
    assert_allclose(cast.storage.data, tensor.storage.data.astype(jnp.complex64))

    assert similar.space == h
    assert similar.storage.data.dtype == tensor.storage.data.dtype
    assert_allclose(similar.storage.data, jnp.zeros_like(tensor.storage.data))

    assert retargeted.space == other_h
    assert retargeted.storage.data.dtype == jnp.dtype(jnp.int32)
    assert_allclose(retargeted.storage.data, jnp.zeros_like(retargeted.storage.data))


def test_from_blocks_builds_tensormap_and_converts_dtype():
    h = _u1_hom()
    blocks = _u1_blocks_for(h)

    tensor = tensor0.from_blocks(h, blocks)
    cast = tensor_api.from_blocks(h, blocks.items(), dtype=jnp.complex64)

    assert tensor.space == h
    for coupled, block in blocks.items():
        assert_allclose(tensor.block(coupled), block)

    assert cast.storage.data.dtype == jnp.dtype(jnp.complex64)
    for coupled, block in blocks.items():
        assert_allclose(cast.block(coupled), block.astype(jnp.complex64))


def test_from_blocks_rejects_non_hom_space():
    with pytest.raises(TypeError, match="from_blocks.*HomSpace"):
        tensor0.from_blocks(object(), {})  # pyright: ignore[reportArgumentType]


def test_from_blocks_rejects_missing_expected_block():
    h = _u1_hom()
    blocks = _u1_blocks_for(h)
    blocks.pop((1,))

    with pytest.raises(ValueError, match="missing|no data"):
        tensor0.from_blocks(h, blocks)


def test_from_blocks_rejects_unexpected_non_empty_block():
    h = _u1_hom()
    blocks = list(_u1_blocks_for(h).items())
    blocks.append(((2,), jnp.ones((1, 1), dtype=jnp.float32)))

    with pytest.raises(ValueError, match="unexpected|not expected"):
        tensor0.from_blocks(h, blocks)


def test_from_blocks_ignores_unexpected_empty_block():
    h = _u1_hom()
    blocks = list(_u1_blocks_for(h).items())
    blocks.append(((2,), jnp.ones((0,), dtype=jnp.float32)))

    tensor = tensor0.from_blocks(h, blocks)

    for coupled, block in _u1_blocks_for(h).items():
        assert_allclose(tensor.block(coupled), block)


def test_from_blocks_rejects_wrong_block_shape():
    h = _u1_hom()
    blocks = _u1_blocks_for(h)
    blocks[(0,)] = jnp.zeros((1, 1), dtype=jnp.float32)

    with pytest.raises(ValueError, match=r"\(0,\).*shape.*\(1, 1\).*expected"):
        tensor0.from_blocks(h, blocks)


def test_from_blocks_rejects_duplicate_sector():
    h = _u1_hom()
    blocks = _u1_blocks_for(h)

    with pytest.raises(ValueError, match="multiple"):
        tensor0.from_blocks(h, [(0, blocks[(0,)]), ((0,), blocks[(0,)])])


def test_zero_like_preserves_space_dtype_and_zeroes_storage():
    h = _u1_hom()
    tensor = TensorMap(h, _int_data_for(h))

    result = tensor0.zero_like(tensor)

    assert result.space == h
    assert result.storage.data.dtype == tensor.storage.data.dtype
    assert_allclose(result.storage.data, jnp.zeros_like(tensor.storage.data))


def test_tensormap_scalar_and_linear_operations_match_functional_forms():
    h = _u1_hom()
    left = TensorMap(h, float_data_for(h))
    right = TensorMap(h, jnp.arange(left.dim, dtype=jnp.float32) * 0.25)

    negated = -left
    summed = left + right
    subtracted = left - right
    left_scaled = 2.5 * left
    right_scaled = left * 2.5
    divided = left / 2.0
    functional_add = tensor0.add(left, right, alpha=2.0, beta=-0.5)
    functional_scale = tensor_api.scale(left, 3.0)

    assert negated.space == h
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
    h = _u1_hom()
    int_tensor = TensorMap(h, _int_data_for(h))
    float_tensor = TensorMap(h, float_data_for(h).astype(jnp.float32))

    added = tensor0.add(int_tensor, float_tensor)
    complex_scaled = tensor_api.scale(int_tensor, jnp.array(1.0 + 2.0j, dtype=jnp.complex64))

    assert added.storage.data.dtype == jnp.result_type(
        int_tensor.storage.data,
        float_tensor.storage.data,
        1,
        1,
    )
    assert_allclose(added.storage.data, int_tensor.storage.data + float_tensor.storage.data)
    assert complex_scaled.storage.data.dtype == jnp.result_type(
        int_tensor.storage.data,
        jnp.array(1.0 + 2.0j, dtype=jnp.complex64),
    )
    assert_allclose(
        complex_scaled.storage.data,
        int_tensor.storage.data * jnp.array(1.0 + 2.0j, dtype=jnp.complex64),
    )


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
    h = _u1_hom()
    tensor = TensorMap(h, float_data_for(h))

    result = jax.jit(lambda x: 2.0 * x + x)(tensor)

    assert result.space == h
    assert_allclose(result.storage.data, 3.0 * tensor.storage.data)


def test_tensormap_adjoint_is_jit_compatible():
    h = _u1_rectangular_hom()
    data = (
        jnp.arange(1, get_degeneracystructure(h).total_dim + 1, dtype=jnp.float32)
        * jnp.array(1.0 + 2.0j, dtype=jnp.complex64)
    )
    tensor = TensorMap(h, data)
    expected = tensor.adjoint()

    result = jax.jit(lambda x: x.adjoint())(tensor)

    assert result.space == expected.space
    assert_allclose(result.storage.data, expected.storage.data)


def test_tensormap_inner_and_dot_conjugate_first_argument():
    h = _u1_hom()
    left = TensorMap(
        h,
        jnp.arange(1, get_degeneracystructure(h).total_dim + 1, dtype=jnp.float32)
        * (1.0 + 2.0j),
    )
    right = TensorMap(h, jnp.arange(1, left.dim + 1, dtype=jnp.float32) * (3.0 - 1.0j))
    expected = sum(
        jnp.vdot(left.block(coupled), right.block(coupled))
        for coupled, _block in left.blocks()
    )

    inner = tensor0.inner(left, right)

    assert_allclose(jnp.asarray(inner), jnp.asarray(expected))
    assert_allclose(jnp.asarray(tensor_api.dot(left, right)), jnp.asarray(inner))


def test_tensormap_inner_norm_and_trace_use_su2_quantum_dimension_weights():
    v = space(SU2Irrep, {0: 1, 1: 1})
    h = hom((v,), (v,))
    left = tensor0.from_blocks(
        h,
        {
            0: jnp.array([[2.0]], dtype=jnp.float32),
            1: jnp.array([[3.0]], dtype=jnp.float32),
        },
    )
    right = tensor0.from_blocks(
        h,
        {
            0: jnp.array([[5.0]], dtype=jnp.float32),
            1: jnp.array([[7.0]], dtype=jnp.float32),
        },
    )

    expected_inner = 1.0 * 2.0 * 5.0 + 2.0 * 3.0 * 7.0
    expected_norm = jnp.sqrt(1.0 * 2.0**2 + 2.0 * 3.0**2)
    expected_trace = 1.0 * 2.0 + 2.0 * 3.0

    assert tensor0.inner(left, right) != jnp.vdot(left.storage.data, right.storage.data)
    assert_allclose(jnp.asarray(tensor0.inner(left, right)), jnp.asarray(expected_inner))
    assert_allclose(jnp.asarray(tensor0.norm(left)), jnp.asarray(expected_norm))
    assert_allclose(jnp.asarray(tensor0.tr(left)), jnp.asarray(expected_trace))


def test_tensormap_norm_supports_p_values_and_rejects_invalid_p():
    v = space(SU2Irrep, {0: 1, 1: 1})
    h = hom((v,), (v,))
    tensor = tensor0.from_blocks(
        h,
        {
            0: jnp.array([[-2.0]], dtype=jnp.float32),
            1: jnp.array([[3.0]], dtype=jnp.float32),
        },
    )

    assert_allclose(tensor0.norm(tensor, p=2), jnp.sqrt(jnp.real(tensor0.inner(tensor, tensor))))
    assert_allclose(tensor0.norm(tensor, p=1), jnp.asarray(1.0 * 2.0 + 2.0 * 3.0))
    assert_allclose(tensor_api.norm(tensor, p=jnp.inf), jnp.asarray(3.0))

    with pytest.raises(ValueError, match="norm.*p|p.*norm"):
        tensor0.norm(tensor, p=0)

    with pytest.raises(ValueError, match="norm.*p|p.*norm"):
        tensor0.norm(tensor, p=float("nan"))


def test_tensormap_norm_returns_real_dtype_for_complex_nondefault_p():
    h = _u1_hom()
    tensor = TensorMap(
        h,
        jnp.asarray(float_data_for(h), dtype=jnp.float32)
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
    h = _u1_hom()
    tensor = TensorMap(h, float_data_for(h))

    result = tensor0.normalize(tensor)

    assert isinstance(result, TensorMap)
    assert result.space == tensor.space
    assert_allclose(result.storage.data, tensor.storage.data / tensor0.norm(tensor))


def test_tensormap_adjoint_swaps_space_and_conjugates_blocks():
    h = _u1_rectangular_hom()
    tensor = tensor0.from_blocks(
        h,
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
    assert result.space == hom(h.domain, h.codomain)
    for coupled, block in tensor.blocks():
        assert_allclose(result.block(coupled), jnp.conj(block.T))
    assert_allclose(tensor_api.adjoint(result).storage.data, tensor.storage.data)


def test_tensormap_component_wrappers_preserve_space():
    h = _u1_hom()
    data = (
        jnp.arange(1, get_degeneracystructure(h).total_dim + 1, dtype=jnp.float32)
        * jnp.array(1.0 + 2.0j, dtype=jnp.complex64)
    )
    tensor = TensorMap(h, data)

    real_part = tensor_api.real(tensor)
    imag_part = tensor0.imag(tensor)
    complex_part = tensor.complex()

    for result in (real_part, imag_part, complex_part):
        assert isinstance(result, TensorMap)
        assert result.space == h

    assert_allclose(real_part.storage.data, jnp.real(data))
    assert_allclose(imag_part.storage.data, jnp.imag(data))
    assert complex_part.storage.data.dtype == jnp.result_type(data, 1j)
    assert_allclose(
        complex_part.storage.data,
        data.astype(complex_part.storage.data.dtype),
    )

    real_tensor = TensorMap(h, float_data_for(h).astype(jnp.float32))
    real_imag = tensor0.imag(real_tensor)
    real_complex = tensor0.complex(real_tensor)

    assert real_imag.space == h
    assert real_complex.space == h
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


def test_tensormap_method_aliases_match_canonical_forms():
    h = _u1_hom()
    tensor = TensorMap(h, float_data_for(h))

    normalized = tensor.normalized(p=1)

    assert_allclose(tensor.to_dense(), to_dense(tensor))
    assert_allclose(normalized.storage.data, tensor.normalize(p=1).storage.data)
    assert_allclose(tensor.trace(), tensor.tr())


def test_tensormap_scalar_and_repr_are_public_api():
    empty = _native.make_product_space(U1Irrep, ())
    h = hom(empty, empty)
    tensor = TensorMap(h, jnp.array([3.5], dtype=jnp.float32))

    assert_allclose(tensor0.scalar(tensor), jnp.asarray(3.5, dtype=jnp.float32))
    assert_allclose(tensor.scalar(), jnp.asarray(3.5, dtype=jnp.float32))
    assert (
        repr(tensor)
        == "TensorMap(numout=0, numin=0, dims=(), blocks=1, dtype=float32)"
    )

    v = space(U1Irrep, {0: 1})
    non_scalar = TensorMap(hom((v,), ()), jnp.array([2.0], dtype=jnp.float32))
    with pytest.raises(ValueError, match="scalar.*visible indices"):
        non_scalar.scalar()


def test_tensormap_trace_rejects_non_endomorphism_before_data_operations():
    codomain = space(U1Irrep, {0: 2})
    domain = space(U1Irrep, {0: 3})
    h = hom((codomain,), (domain,))
    tensor = TensorMap(h, InaccessibleVectorData(get_degeneracystructure(h).total_dim))

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


def test_tensormap_diag_diagm_and_isdiag_round_trip_u1_blocks():
    codomain = space(U1Irrep, {0: 2, 1: 1})
    domain = space(U1Irrep, {0: 2, 1: 3})
    h = hom((codomain,), (domain,))
    tensor = tensor0.from_blocks(
        h,
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
    assert tensor0.isdiag(tensor)
    assert tensor0.isdiag(rebuilt)
    assert_allclose(rebuilt.storage.data, tensor.storage.data)

    offdiagonal = tensor0.from_blocks(
        h,
        {
            0: jnp.array([[1.0, 4.0], [0.0, 2.0]], dtype=jnp.float32),
            1: jnp.array([[3.0, 0.0, 0.0]], dtype=jnp.float32),
        },
    )
    assert not tensor_api.isdiag(offdiagonal)


def test_tensormap_diag_and_diagm_support_multi_leg_blocks():
    v1 = space(U1Irrep, {0: 1, 1: 1})
    v2 = space(U1Irrep, {0: 1})
    h = hom((v1, v2), (v2, v1))
    blocks = {}
    zero = TensorMap(h, jnp.zeros(get_degeneracystructure(h).total_dim))
    for coupled, block in zero.blocks():
        block_array = jnp.zeros(block.shape, dtype=jnp.float32)
        diagonal_indices = jnp.arange(min(block.shape))
        blocks[coupled] = block_array.at[diagonal_indices, diagonal_indices].set(
            jnp.arange(1, len(diagonal_indices) + 1, dtype=jnp.float32),
        )
    tensor = tensor0.from_blocks(h, blocks)

    values = tensor0.diag(tensor)
    rebuilt = tensor0.diagm(h.codomain, h.domain, values)

    assert isinstance(values, SectorDict)
    for coupled, block in tensor.blocks():
        assert_allclose(values[coupled], jnp.diag(block))
    assert tensor0.isdiag(tensor)
    assert tensor0.isdiag(rebuilt)
    assert_allclose(rebuilt.storage.data, tensor.storage.data)


def test_tensormap_diag_diagm_and_isdiag_reject_invalid_inputs():
    v = space(U1Irrep, {0: 2})
    values = {0: jnp.arange(2, dtype=jnp.float32)}

    with pytest.raises(TypeError, match="diag.*TensorMap"):
        tensor0.diag(object())  # pyright: ignore[reportArgumentType]

    with pytest.raises(TypeError, match="isdiag.*TensorMap"):
        tensor0.isdiag(object())  # pyright: ignore[reportArgumentType]

    with pytest.raises(TypeError, match="diagm.*ProductSpace|diagm.*space"):
        tensor0.diagm(object(), v, values)  # pyright: ignore[reportArgumentType]

    with pytest.raises(TypeError, match="diagm.*mapping|diagm.*values"):
        tensor0.diagm(v, v, object())  # pyright: ignore[reportArgumentType]


def test_tensormap_equal_is_exact_space_and_dtype_sensitive():
    h = _u1_hom()
    data = jnp.arange(get_degeneracystructure(h).total_dim, dtype=jnp.float32)
    tensor = TensorMap(h, data)
    same = TensorMap(h, data.copy())
    changed = TensorMap(h, data.at[0].set(-1))
    changed_dtype = TensorMap(h, data.astype(jnp.complex64))
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
    h = _u1_hom()
    size = get_degeneracystructure(h).total_dim
    integers = TensorMap(h, jnp.arange(size, dtype=jnp.int32))
    close_floats = TensorMap(
        h,
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

    tensor = tensor0.diagm(codomain, domain, {0: jnp.arange(2), 1: jnp.arange(0)})
    assert_allclose(tensor.block(0), jnp.diag(jnp.arange(2)))


def test_tensormap_subblocks_are_lazy_reiterable_and_match_indexed_access(monkeypatch):
    tensor = _tensor_for(_u1_hom())
    sectorstructure = get_sectorstructure(tensor.space)
    degeneracystructure = get_degeneracystructure(tensor.space)
    expected_count = sectorstructure.fusiontree_pair_count
    native_get_subblock = tensor_map_module._get_subblock
    get_calls = 0

    class SectorStructureProxy:
        def __getattr__(self, name):
            return getattr(sectorstructure, name)

        @property
        def fusiontree_pairs(self):
            raise AssertionError("subblocks() must not materialize all fusion tree pairs")

    class DegeneracyStructureProxy:
        def __getattr__(self, name):
            return getattr(degeneracystructure, name)

        @property
        def subblockstructure(self):
            raise AssertionError("subblocks() must not materialize all subblock structures")

    def count_gets(*args, **kwargs):
        nonlocal get_calls
        get_calls += 1
        return native_get_subblock(*args, **kwargs)

    monkeypatch.setattr(tensor_map_module, "_get_subblock", count_gets)
    monkeypatch.setattr(
        tensor_map_module,
        "get_sectorstructure",
        lambda _space: SectorStructureProxy(),
    )
    monkeypatch.setattr(
        tensor_map_module,
        "get_degeneracystructure",
        lambda _space: DegeneracyStructureProxy(),
    )

    subblocks = tensor.subblocks()

    assert len(subblocks) == expected_count
    assert not hasattr(subblocks, "index")
    assert not hasattr(subblocks, "count")
    assert get_calls == 0

    first_item = next(iter(subblocks))
    assert get_calls == 1

    first = tuple(subblocks)
    assert get_calls == 1 + expected_count
    second = tuple(subblocks)
    assert get_calls == 1 + 2 * expected_count
    assert first_item[0] == first[0][0]
    assert_allclose(first_item[1], first[0][1])

    indexed_pair, indexed = subblocks[0]
    assert indexed_pair == first[0][0]
    assert_allclose(indexed, first[0][1])

    last_pair, last = subblocks[-1]
    assert last_pair == first[-1][0]
    assert_allclose(last, first[-1][1])

    sliced = subblocks[:1]
    assert isinstance(sliced, tuple)
    assert sliced[0][0] == first[0][0]
    assert_allclose(sliced[0][1], first[0][1])

    with pytest.raises(IndexError, match="subblock index out of range"):
        subblocks[expected_count]

    for ((row_tree, col_tree), subblock), (repeated_pair, repeated) in zip(
        first,
        second,
        strict=True,
    ):
        assert repeated_pair == (row_tree, col_tree)
        assert_allclose(repeated, subblock)
        assert_allclose(tensor.subblock(row_tree, col_tree), subblock)
        assert_allclose(tensor.subblock((row_tree, col_tree)), subblock)
        assert_allclose(tensor[row_tree, col_tree], subblock)


@pytest.mark.parametrize(
    ("sector_type", "sector", "visible_dual"),
    [
        (U1Irrep, 1, -1),
        (FermionParity, 1, 1),
        (Z2Irrep, 1, 1),
        (Z3Irrep, 1, 2),
        (Z4Irrep, 1, 3),
        (FermionNumber, (1, 1), (-1, 1)),
        (FermionParityU1Irrep, (1, 1), (1, -1)),
    ],
)
def test_tensormap_sector_indexing_supports_all_unique_fusion_families(
    sector_type,
    sector,
    visible_dual,
):
    factor = space(sector_type, {sector: 1})
    tensor = _tensor_for(hom((factor,), (factor,)))
    pair = tensor.fusiontrees[0]

    assert_allclose(tensor[(sector, visible_dual)], tensor[pair])


def test_tensormap_unique_fusion_sector_indexing_uses_visible_domain_sectors():
    factor = space(U1Irrep, {1: 2})
    target = hom((factor,), (factor,))
    tensor = _tensor_for(target)
    row_tree, col_tree = tensor.fusiontrees[0]
    expected = tensor[row_tree, col_tree]

    assert_allclose(tensor.subblock((1, -1)), expected)
    assert_allclose(tensor[1, -1], expected)

    dual_factor = space(U1Irrep, {1: 1}, dual=True)
    dual_tensor = _tensor_for(hom((dual_factor,), (dual_factor,)))
    dual_pair = dual_tensor.fusiontrees[0]
    assert_allclose(dual_tensor[-1, 1], dual_tensor[dual_pair])

    product = space(FermionNumber, {(1, 1): 1})
    product_tensor = _tensor_for(hom((product,), (product,)))
    product_pair = product_tensor.fusiontrees[0]
    assert_allclose(
        product_tensor[(1, 1), (-1, 1)],
        product_tensor[product_pair],
    )

    unit_product = space(FermionNumber, {(0, 0): 1})
    empty_product = _native.make_product_space(FermionNumber, ())
    rank_one = _tensor_for(hom((unit_product,), empty_product))
    assert_allclose(rank_one[((0, 0),)], rank_one[rank_one.fusiontrees[0]])


def test_tensormap_sector_indexing_validates_style_rank_and_channel():
    factor = space(U1Irrep, {1: 1})
    tensor = _tensor_for(hom((factor,), (factor,)))

    with pytest.raises(ValueError, match="length 2"):
        tensor[(1,)]
    with pytest.raises(KeyError):
        tensor[0, -1]

    multi_factor = space(U1Irrep, {0: 1, 1: 1})
    multi_tensor = _tensor_for(
        hom((multi_factor, multi_factor), (multi_factor, multi_factor))
    )
    with pytest.raises(KeyError):
        multi_tensor[1, 0, 0, 0]

    half = space(SU2Irrep, {1: 1})
    su2_tensor = _tensor_for(hom((half,), (half,)))
    with pytest.raises(ValueError, match="UniqueFusion"):
        su2_tensor[1, 1]

    empty = _native.make_product_space(U1Irrep, ())
    scalar = _tensor_for(hom(empty, empty))
    assert_allclose(scalar[()], scalar.storage.data.reshape(()))


def test_tensormap_subblock_rejects_invalid_tree_inputs():
    h = _u1_hom()
    tensor = TensorMap(h, _u1_data())
    row_tree, col_tree = get_sectorstructure(h).fusiontree_pairs[0]
    other_tree, _ = get_sectorstructure(
        _u1_hom_same_total_dim_with_different_metadata(),
    ).fusiontree_pairs[1]
    su2 = space(SU2Irrep, {1: 1})
    su2_sectorstructure = get_sectorstructure(hom((su2,), (su2,)))
    su2_row_tree, su2_col_tree = su2_sectorstructure.fusiontree_pairs[0]

    with pytest.raises(TypeError, match="FusionTree"):
        tensor.subblock(object(), col_tree)  # pyright: ignore[reportArgumentType]

    with pytest.raises(TypeError, match="two arguments.*FusionTree"):
        tensor.subblock(1, -1)  # pyright: ignore[reportArgumentType]

    with pytest.raises(TypeError, match="FusionTree"):
        tensor.subblock((row_tree, -1))  # pyright: ignore[reportArgumentType]

    with pytest.raises(TypeError, match="FusionTree|tuple of sectors"):
        tensor[0]  # pyright: ignore[reportArgumentType]

    with pytest.raises(KeyError):
        tensor.subblock(other_tree, col_tree)

    with pytest.raises(KeyError):
        tensor.subblock(su2_row_tree, su2_col_tree)

    assert_allclose(tensor[row_tree, col_tree], tensor.subblock(row_tree, col_tree))


@pytest.mark.parametrize("case", tensor_map_matmul_cases(), ids=lambda case: case.name)
def test_tensormap_satisfies_matmul_contract(case):
    _assert_tensormap_matmul_contract(case)


def test_tensormap_matmul_zero_fills_result_sector_missing_from_middle_space():
    v = space(U1Irrep, {0: 2, 1: 3})
    w = space(U1Irrep, {0: 5})
    x = space(U1Irrep, {0: 7, 1: 11})
    a_space = hom((v,), (w,))
    b_space = hom((w,), (x,))
    a = TensorMap(a_space, float_data_for(a_space))
    b = TensorMap(b_space, float_data_for(b_space))

    result = a @ b

    assert result.space == hom((v,), (x,))
    assert_allclose(result.block(0), a.block(0) @ b.block(0))
    assert_allclose(
        result.block(1),
        jnp.zeros((3, 11), dtype=result.storage.data.dtype),
    )


def test_tensormap_matmul_promotes_dtype_for_products_and_zero_blocks():
    v = space(U1Irrep, {0: 2, 1: 3})
    w = space(U1Irrep, {0: 5})
    x = space(U1Irrep, {0: 7, 1: 11})
    a_space = hom((v,), (w,))
    b_space = hom((w,), (x,))
    a = TensorMap(a_space, _int_data_for(a_space))
    b = TensorMap(b_space, float_data_for(b_space))

    result = a @ b

    expected_dtype = jnp.result_type(a.storage.data, b.storage.data)
    assert result.storage.data.dtype == expected_dtype
    for _coupled, block in result.blocks():
        assert block.dtype == expected_dtype


def test_tensormap_matmul_rejects_incompatible_middle_space():
    v = space(U1Irrep, {0: 2})
    w = space(U1Irrep, {0: 3})
    different_w = space(U1Irrep, {0: 4})
    x = space(U1Irrep, {0: 5})
    a_space = hom((v,), (w,))
    b_space = hom((different_w,), (x,))
    a = TensorMap(a_space, float_data_for(a_space))
    b = TensorMap(b_space, float_data_for(b_space))

    with pytest.raises(ValueError, match="composable"):
        _ = a @ b


def test_dense_roundtrip_for_scalar_and_one_sided_spaces():
    empty = _native.make_product_space(U1Irrep, ())
    scalar_h = hom(empty, empty)
    scalar = TensorMap(scalar_h, jnp.array([3.0], dtype=jnp.float32))

    scalar_dense = to_dense(scalar)
    scalar_rebuilt = from_dense(scalar_h, scalar_dense)
    scalar_from_matrix = from_dense(scalar_h, jnp.array([[3.0]], dtype=jnp.float32))

    assert scalar_dense.shape == ()
    assert float(scalar_dense) == pytest.approx(3.0)
    assert_allclose(scalar_rebuilt.storage.data, scalar.storage.data)
    assert_allclose(scalar_from_matrix.storage.data, scalar.storage.data)

    v = space(U1Irrep, {0: 2})
    h = hom((v,), ())
    tensor = TensorMap(h, jnp.array([0.25, 0.5], dtype=jnp.float32))

    dense = to_dense(tensor)
    rebuilt = from_dense(h, dense)

    assert dense.shape == (2,)
    assert_allclose(dense, tensor.storage.data)
    assert_allclose(rebuilt.storage.data, tensor.storage.data)


@pytest.mark.parametrize("case", dense_roundtrip_cases(), ids=lambda case: case.name)
def test_dense_roundtrip_matches_tensor_storage_for_space_cases(case):
    tensor = TensorMap(case.space, float_data_for(case.space))

    dense = to_dense(tensor)
    rebuilt = from_dense(case.space, dense)

    assert dense.shape == case.dense_shape
    assert rebuilt.space == tensor.space
    assert_allclose(rebuilt.storage.data, tensor.storage.data)


def test_dense_conversion_matches_su2_half_operator_convention():
    half = space(SU2Irrep, {1: 1})
    h = hom((half,), (half,))
    tensor = TensorMap(h, jnp.array([1.0], dtype=jnp.float32))

    dense = to_dense(tensor)
    rebuilt = from_dense(h, dense)

    assert dense.shape == (2, 2)
    assert_allclose(dense, jnp.eye(2, dtype=jnp.float32))
    assert_allclose(rebuilt.storage.data, tensor.storage.data)


def test_dense_conversion_matches_su2_dual_leg_convention():
    half = space(SU2Irrep, {1: 1})
    h = hom((half.dual(), half), ())
    tensor = TensorMap(h, jnp.array([1.0], dtype=jnp.float32))

    dense = to_dense(tensor)
    rebuilt = from_dense(h, dense)

    assert dense.shape == (2, 2)
    assert_allclose(dense, -jnp.eye(2, dtype=jnp.float32) / jnp.sqrt(2.0))
    assert_allclose(rebuilt.storage.data, tensor.storage.data)


def test_from_dense_accepts_matrix_shape_with_row_major_reshape():
    a = space(U1Irrep, {0: 2})
    b = space(U1Irrep, {0: 3})
    c = space(U1Irrep, {0: 4})
    h = hom((a, b), (c,))
    tensor = TensorMap(h, float_data_for(h))
    dense = to_dense(tensor)

    rebuilt_from_matrix = from_dense(h, jnp.reshape(dense, (6, 4)))

    assert_allclose(rebuilt_from_matrix.storage.data, tensor.storage.data)
    assert_allclose(to_dense(rebuilt_from_matrix), dense)


def test_from_dense_rejects_incompatible_shape():
    v = space(U1Irrep, {0: 2})
    h = hom((v,), (v,))

    with pytest.raises(ValueError, match="dense shape"):
        from_dense(h, jnp.zeros((2, 3), dtype=jnp.float32))


def test_from_dense_rejects_components_outside_symmetry_structure():
    v = space(U1Irrep, {0: 2, 1: 1})
    h = hom((v,), (v,))
    dense = jnp.zeros((3, 3), dtype=jnp.float32).at[0, 2].set(1.0)

    with pytest.raises(ValueError, match="symmetry structure"):
        from_dense(h, dense)


def test_public_dense_path_has_no_debug_element_cap():
    v = space(U1Irrep, {0: 65})
    h = hom((v,), (v,))
    tensor = TensorMap(h, jnp.ones((65 * 65,), dtype=jnp.float32))

    dense = to_dense(tensor)
    rebuilt = from_dense(h, dense)

    assert dense.shape == (65, 65)
    assert_allclose(rebuilt.storage.data, tensor.storage.data)
