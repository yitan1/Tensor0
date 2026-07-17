from dataclasses import FrozenInstanceError

import tensor0
import tensor0.tensor as tensor_api
import tensor0.tensor.tensor_map as tensor_map_module
import jax.numpy as jnp
import pytest

from tensor0 import (
    FermionNumber,
    FermionParity,
    FermionParityU1Irrep,
    SU2Irrep,
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
)


def _u1_hom():
    v = space(U1Irrep, {0: 2, 1: 3})
    return hom((v,), (v,))


def _u1_hom_same_total_dim_with_different_metadata():
    v = space(U1Irrep, {0: 2, 2: 3})
    return hom((v,), (v,))


def _u1_data():
    return jnp.arange(13)


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
