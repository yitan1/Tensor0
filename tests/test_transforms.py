import math

import jax.numpy as jnp
import pytest

import tensor0.operations.transforms as transforms
from tensor0 import (
    FermionParity,
    SU2Irrep,
    TensorMap,
    U1Irrep,
    braid,
    get_degeneracystructure,
    get_sectorstructure,
    hom,
    permute,
    repartition,
    space,
    transpose,
    to_dense,
    twist,
)
from tests.cases import (
    assert_allclose,
    is_contiguous_subblock,
    transform_cases,
    zero_based_float_data_for,
)


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


def _data_for(target):
    size = get_degeneracystructure(target).total_dim
    return jnp.arange(1, size + 1, dtype=jnp.float32)


class _ExplodingVectorData:
    def __init__(self, length):
        self.shape = (length,)

    def __getitem__(self, key):
        if isinstance(key, slice) and key.start == 0 and key.stop == 0:
            return jnp.zeros((0,))
        raise AssertionError("twist validation must not access storage")


def test_native_tree_transformer_payload_uses_canonical_indices():
    u1 = space(U1Irrep, {0: 1})
    u1_source = hom((u1, u1), ())
    u1_destination = u1_source.permute((1,), (0,))
    abelian = transforms._native.tree_braider(
        u1_source,
        u1_destination,
        get_sectorstructure(u1_source),
        get_sectorstructure(u1_destination),
        (1,),
        (0,),
        (0, 1),
        (),
    )

    assert abelian.kind == "abelian"
    assert abelian.abelian_data is abelian.abelian_data
    assert tuple((entry.src, entry.dst) for entry in abelian.abelian_data) == ((0, 0),)
    assert isinstance(abelian.abelian_data[0].src, int)
    assert isinstance(abelian.abelian_data[0].dst, int)

    half = space(SU2Irrep, {1: 1})
    su2_source = hom((half, half, half, half), ())
    su2_destination = su2_source.permute((1, 2, 3), (0,))
    generic = transforms._native.tree_transposer(
        su2_source,
        su2_destination,
        get_sectorstructure(su2_source),
        get_sectorstructure(su2_destination),
        (1, 2, 3),
        (0,),
    )
    generic_data = generic.generic_data
    group = generic_data[0]
    transform = group.transform

    assert generic.kind == "generic"
    assert generic.generic_data is generic_data
    assert group.transform is transform
    assert transform.shape == (2, 2)
    assert not transform.flags.writeable
    assert group.src_indices == (0, 1)
    assert group.dst_indices == (0, 1)


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


@pytest.mark.parametrize(
    ("operation", "native_name"),
    [
        (lambda tensor: permute(tensor, ((1,), (0,))), "tree_braider"),
        (lambda tensor: transpose(tensor, ((1,), (0,))), "tree_transposer"),
    ],
    ids=["braider", "transposer"],
)
def test_tree_transformer_cache_reuses_metadata_across_degeneracies(
    monkeypatch,
    operation,
    native_name,
):
    first_left = space(U1Irrep, {0: 2})
    first_right = space(U1Irrep, {0: 3})
    first_tensor = TensorMap(
        hom((first_left, first_right), ()),
        jnp.arange(6, dtype=jnp.float32),
    )
    second_left = space(U1Irrep, {0: 4})
    second_right = space(U1Irrep, {0: 5})
    second_tensor = TensorMap(
        hom((second_left, second_right), ()),
        jnp.arange(20, dtype=jnp.float32),
    )
    native_builder = getattr(transforms._native, native_name)
    builder_calls = 0

    def count_builder(*args):
        nonlocal builder_calls
        builder_calls += 1
        return native_builder(*args)

    transforms._clear_tree_transformer_caches_for_tests()
    monkeypatch.setattr(transforms._native, native_name, count_builder)
    try:
        first = operation(first_tensor)
        second = operation(second_tensor)
    finally:
        transforms._clear_tree_transformer_caches_for_tests()

    assert builder_calls == 1
    _assert_dense_transpose(
        first,
        first_tensor,
        first_tensor.space.permute((1,), (0,)),
        (1, 0),
    )
    _assert_dense_transpose(
        second,
        second_tensor,
        second_tensor.space.permute((1,), (0,)),
        (1, 0),
    )


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


def test_repartition_rejects_nout_larger_than_tensor_rank():
    tensor = _simple_u1_tensor()

    with pytest.raises(ValueError, match="tensor rank"):
        repartition(tensor, tensor.numind + 1)


def test_u1_index_transform_resolves_strided_source_and_destination_subblocks():
    factor = space(U1Irrep, {0: 2, 1: 1})
    source = hom((factor, factor), (factor, factor))
    permutation = ((1, 2), (0, 3))
    destination = source.permute(*permutation)
    transformer = transforms._treebraider(
        source,
        destination,
        *permutation,
        tuple(range(source.numind)),
    )
    source_subblocks = get_degeneracystructure(source).subblockstructure
    destination_subblocks = get_degeneracystructure(destination).subblockstructure
    source_indices = tuple(entry.src for entry in transformer.abelian_data)
    destination_indices = tuple(entry.dst for entry in transformer.abelian_data)

    assert any(
        not is_contiguous_subblock(source_subblocks[index])
        for index in source_indices
    )
    assert any(
        not is_contiguous_subblock(destination_subblocks[index])
        for index in destination_indices
    )

    tensor = TensorMap(source, zero_based_float_data_for(source))
    result = permute(tensor, permutation)
    _assert_dense_transpose(result, tensor, destination, (1, 2, 0, 3))


def test_index_transform_preserves_existing_dtype_rules():
    u1_left = space(U1Irrep, {0: 2})
    u1_right = space(U1Irrep, {0: 3})
    u1_tensor = TensorMap(
        hom((u1_left, u1_right), ()),
        jnp.arange(6, dtype=jnp.float16),
    )
    assert permute(u1_tensor, ((1,), (0,))).storage.data.dtype == jnp.float16

    half = space(SU2Irrep, {1: 1})
    su2_source = hom((half, half, half, half), ())
    permutation = ((1, 2, 3), (0,))
    float_tensor = TensorMap(
        su2_source,
        jnp.asarray([1, 2], dtype=jnp.float16),
    )
    complex_tensor = TensorMap(
        su2_source,
        jnp.asarray([1 + 2j, 3 + 4j], dtype=jnp.complex64),
    )

    assert permute(float_tensor, permutation).storage.data.dtype == jnp.float32
    assert permute(complex_tensor, permutation).storage.data.dtype == jnp.complex64


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


def test_twist_passes_the_cached_sectorstructure_to_native(monkeypatch):
    factor = space(FermionParity, {0: 1, 1: 1})
    target = hom((factor,), (factor,))
    tensor = TensorMap(target, _data_for(target))
    expected = get_sectorstructure(target)
    native_twist_subblock_factors = transforms._native.twist_subblock_factors
    seen_structure = None

    def recording_twist_subblock_factors(space, sectorstructure, indices, inv):
        nonlocal seen_structure
        seen_structure = sectorstructure
        return native_twist_subblock_factors(
            space,
            sectorstructure,
            indices,
            inv,
        )

    monkeypatch.setattr(
        transforms._native,
        "twist_subblock_factors",
        recording_twist_subblock_factors,
    )

    twist(tensor, 0)

    assert seen_structure is expected


def test_twist_layout_cancellation_bypasses_degeneracy_lookup(monkeypatch):
    factor = space(FermionParity, {0: 1, 1: 1})
    target = hom((factor,), (factor,))
    tensor = TensorMap(target, _data_for(target))

    def explode(*args, **kwargs):
        raise AssertionError("cancelled twist must not request degeneracy layout")

    monkeypatch.setattr(transforms, "get_degeneracystructure", explode)

    assert twist(tensor, (0, 1)) is tensor


def test_twist_reweights_parity_endomorphism_without_changing_space_or_dtype():
    factor = space(FermionParity, {0: 1, 1: 1})
    target = hom((factor,), (factor,))
    tensor = TensorMap(target, _data_for(target))
    original = tensor.storage.data.copy()

    codomain = twist(tensor, 0)
    domain = twist(tensor, (1,))

    expected = jnp.array([1.0, -2.0], dtype=jnp.float32)
    assert bool(jnp.array_equal(codomain.storage.data, expected))
    assert bool(jnp.array_equal(domain.storage.data, expected))
    assert codomain.space is target
    assert domain.space is target
    assert codomain.storage.data is not tensor.storage.data
    assert domain.storage.data is not tensor.storage.data
    assert codomain.storage.data.dtype == tensor.storage.data.dtype
    assert domain.storage.data.dtype == tensor.storage.data.dtype
    assert bool(jnp.array_equal(tensor.storage.data, original))


def test_twist_reweights_multielement_strided_subblocks_and_is_involutive():
    factor = space(FermionParity, {0: 2, 1: 1})
    target = hom((factor, factor), (factor, factor))
    tensor = TensorMap(target, _data_for(target))
    subblockstructure = get_degeneracystructure(target).subblockstructure

    assert any(
        math.prod(subblock.sizes) > 1 and not is_contiguous_subblock(subblock)
        for subblock in subblockstructure
    )

    result = twist(tensor, 0)
    restored = twist(result, 0)

    assert result.space is target
    assert result.storage.data.dtype == tensor.storage.data.dtype
    assert not bool(jnp.array_equal(result.storage.data, tensor.storage.data))
    assert bool(jnp.array_equal(restored.storage.data, tensor.storage.data))


def test_twist_inverse_supports_product_sector_types():
    product = space(U1Irrep @ FermionParity, {(0, 0): 1, (0, 1): 1})
    target = hom((product,), (product,))
    tensor = TensorMap(target, _data_for(target))

    result = twist(tensor, 0, inv=True)

    expected = jnp.array([1.0, -2.0], dtype=jnp.float32)
    assert bool(jnp.array_equal(result.storage.data, expected))
    assert result.space is target


@pytest.mark.parametrize(
    ("sector_type", "sectors"),
    [
        (U1Irrep, {0: 1, 1: 1}),
        (FermionParity, {0: 1}),
    ],
    ids=["bosonic", "trivial-fermionic"],
)
def test_twist_semantic_identity_bypasses_layout_lookup(
    monkeypatch,
    sector_type,
    sectors,
):
    factor = space(sector_type, sectors)
    target = hom((factor,), (factor,))
    tensor = TensorMap(
        target,
        _ExplodingVectorData(get_degeneracystructure(target).total_dim),
    )

    def explode(*args, **kwargs):
        raise AssertionError("semantic identity twist must not request layout")

    monkeypatch.setattr(transforms, "get_sectorstructure", explode)
    monkeypatch.setattr(transforms, "get_degeneracystructure", explode)

    assert twist(tensor, 0) is tensor


def test_twist_empty_indices_bypass_native_after_inv_validation(monkeypatch):
    factor = space(U1Irrep, {0: 1})
    target = hom((factor,), (factor,))
    tensor = TensorMap(target, _data_for(target))

    def explode(*args, **kwargs):
        raise AssertionError("empty twist must bypass native factor computation")

    monkeypatch.setattr(transforms._native, "twist_is_trivial", explode)
    monkeypatch.setattr(transforms._native, "twist_subblock_factors", explode)
    monkeypatch.setattr(transforms, "get_sectorstructure", explode)
    monkeypatch.setattr(transforms, "get_degeneracystructure", explode)

    assert twist(tensor, ()) is tensor
    with pytest.raises(TypeError, match=r"inv.*bool"):
        twist(tensor, (), inv=1)  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize(
    ("indices", "error", "match"),
    [
        ([0], TypeError, "integer|tuple"),
        ((True,), TypeError, "integer"),
        ((-1,), ValueError, "non-negative"),
        ((0, 0), ValueError, "unique"),
        ((2,), ValueError, "range"),
    ],
)
def test_twist_rejects_invalid_indices_before_accessing_storage(indices, error, match):
    factor = space(U1Irrep, {0: 1})
    target = hom((factor,), (factor,))
    tensor = TensorMap(
        target,
        _ExplodingVectorData(get_degeneracystructure(target).total_dim),
    )

    with pytest.raises(error, match=match):
        twist(tensor, indices)


def test_twist_rejects_invalid_tensor_and_inv_inputs():
    with pytest.raises(TypeError, match=r"twist.*TensorMap"):
        twist(object(), 0)  # pyright: ignore[reportArgumentType]

    factor = space(U1Irrep, {0: 1})
    target = hom((factor,), (factor,))
    tensor = TensorMap(
        target,
        _ExplodingVectorData(get_degeneracystructure(target).total_dim),
    )
    with pytest.raises(TypeError, match=r"inv.*bool"):
        twist(tensor, 0, inv=1)  # pyright: ignore[reportArgumentType]
