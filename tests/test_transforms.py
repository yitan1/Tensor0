import jax.numpy as jnp
import pytest

import tensor0.transforms as transforms
from tensor0 import (
    FermionParity,
    SU2Irrep,
    TensorMap,
    U1Irrep,
    Z2Irrep,
    braid,
    get_degeneracystructure,
    get_sectorstructure,
    hom,
    permute,
    repartition,
    space,
    transpose,
    to_dense,
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

    with pytest.raises(ValueError):
        transform[0, 0] = 0.0


def test_native_tree_transformer_rejects_mixed_cached_sector_families():
    u1 = space(U1Irrep, {0: 1})
    z2 = space(Z2Irrep, {0: 1})
    source = hom((u1, u1), ())
    destination = source.permute((1,), (0,))
    z2_source = hom((z2, z2), ())
    z2_destination = z2_source.permute((1,), (0,))

    with pytest.raises(ValueError, match="same sector family"):
        transforms._native.tree_braider(
            source,
            destination,
            get_sectorstructure(z2_source),
            get_sectorstructure(z2_destination),
            (1,),
            (0,),
            (0, 1),
            (),
        )

    other = space(U1Irrep, {1: 1})
    other_source = hom((other, other), ())
    with pytest.raises(ValueError, match="source sectorstructure does not match"):
        transforms._native.tree_braider(
            source,
            destination,
            get_sectorstructure(other_source),
            get_sectorstructure(destination),
            (1,),
            (0,),
            (0, 1),
            (),
        )


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


def test_simplefusion_scalar_transform_does_not_stack(monkeypatch):
    half = space(SU2Irrep, {1: 1})
    source = hom((half, half), ())
    tensor = TensorMap(source, jnp.asarray([3.0], dtype=jnp.float32))

    def fail_stack(*_args, **_kwargs):
        raise AssertionError("1x1 tree transform must not stack source blocks")

    monkeypatch.setattr(transforms.jnp, "stack", fail_stack)
    result = permute(tensor, ((1,), (0,)))

    assert result.storage.data.shape == (1,)


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
