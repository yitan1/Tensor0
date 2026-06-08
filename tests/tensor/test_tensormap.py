from dataclasses import FrozenInstanceError

import jax.numpy as jnp
import pytest

from tensor0 import (
    SU2Irrep,
    TensorMap,
    U1Irrep,
    VectorStorage,
    _native,
    from_dense,
    get_degeneracystructure,
    get_sectorstructure,
    hom,
    space,
    to_dense,
)
from tests.cases import (
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


def _u1_data():
    return jnp.arange(13)


def _int_data_for(h):
    total_dim = get_degeneracystructure(h).total_dim
    return jnp.arange(1, total_dim + 1, dtype=jnp.int32)


def _gather_expected_subblock(data, subblock):
    indices = jnp.zeros(subblock.sizes, dtype=jnp.result_type(subblock.offset))
    for axis, (size, stride) in enumerate(zip(subblock.sizes, subblock.strides)):
        shape = (1,) * axis + (size,) + (1,) * (len(subblock.sizes) - axis - 1)
        indices = indices + jnp.arange(size).reshape(shape) * stride
    return data[(indices + subblock.offset).reshape(-1)].reshape(subblock.sizes)


def _tensor_for(space_obj):
    return TensorMap(space_obj, float_data_for(space_obj))


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


def _assert_tensormap_subblocks_contract(case):
    tensor = _tensor_for(case.space)
    sectorstructure = get_sectorstructure(case.space)
    degeneracystructure = get_degeneracystructure(case.space)

    subblocks = tensor.subblocks()

    assert len(subblocks) == len(sectorstructure.fusiontree_pairs)
    for ((row, col), subblock), (expected_row, expected_col), layout in zip(
        subblocks,
        sectorstructure.fusiontree_pairs,
        degeneracystructure.subblockstructure,
        strict=True,
    ):
        assert row.static_key == expected_row.static_key
        assert col.static_key == expected_col.static_key
        assert_allclose(tensor.subblock(row, col), subblock)
        assert_allclose(
            subblock,
            _gather_expected_subblock(tensor.storage.data, layout),
        )


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


@pytest.mark.parametrize("case", tensor_map_cases(), ids=lambda case: case.name)
def test_tensormap_satisfies_subblocks_contract(case):
    _assert_tensormap_subblocks_contract(case)


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

    with pytest.raises(TypeError, match="FusionTree"):
        tensor[0]  # pyright: ignore[index]

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
    assert_allclose(result.block(0), a.block(0) @ b.block(0))
    assert_allclose(
        result.block(1),
        jnp.zeros((3, 11), dtype=expected_dtype),
    )


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

    rebuilt_from_full = from_dense(h, dense)
    rebuilt_from_matrix = from_dense(h, jnp.reshape(dense, (6, 4)))

    assert_allclose(rebuilt_from_full.storage.data, tensor.storage.data)
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
