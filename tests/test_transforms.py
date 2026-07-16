import math

import jax.numpy as jnp
import pytest

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
    braid,
    flip,
    get_degeneracystructure,
    get_sectorstructure,
    hom,
    insertleftunit,
    insertrightunit,
    permute,
    repartition,
    removeunit,
    space,
    tensorcontract,
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
        raise AssertionError("index transform validation must not access storage")


def test_native_transform_payloads_use_canonical_indices():
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

    su2_flip_destination = su2_source.flip((0, 2))
    su2_flip = transforms._native.flip_entries(
        su2_source,
        su2_flip_destination,
        get_sectorstructure(su2_source),
        get_sectorstructure(su2_flip_destination),
        (0, 2),
    )

    assert su2_flip == ((0, 1.0), (1, 1.0))


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


def test_twist_only_materializes_nontrivial_subblock_metadata(monkeypatch):
    factor = space(FermionParity, {0: 1, 1: 1})
    target = hom((factor,), (factor,))
    tensor = TensorMap(target, _data_for(target))
    degeneracystructure = get_degeneracystructure(target)
    accessed = []

    class DegeneracyStructureProxy:
        @property
        def subblockstructure(self):
            raise AssertionError("twist must not materialize all subblock metadata")

        def subblock_at(self, index):
            accessed.append(index)
            return degeneracystructure.subblock_at(index)

    monkeypatch.setattr(
        transforms,
        "get_degeneracystructure",
        lambda _space: DegeneracyStructureProxy(),
    )

    result = twist(tensor, 0)

    assert accessed == [1]
    assert bool(jnp.array_equal(result.storage.data, jnp.array([1.0, -2.0])))


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


@pytest.mark.parametrize("operation", [twist, flip], ids=["twist", "flip"])
def test_index_transform_empty_indices_validate_inv_and_return_input(operation):
    factor = space(U1Irrep, {0: 1})
    target = hom((factor,), (factor,))
    tensor = TensorMap(target, _data_for(target))

    assert operation(tensor, ()) is tensor
    with pytest.raises(TypeError, match=r"inv.*bool"):
        operation(tensor, (), inv=1)  # pyright: ignore[reportArgumentType]


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
@pytest.mark.parametrize("operation", [twist, flip], ids=["twist", "flip"])
def test_index_transform_rejects_invalid_indices_before_accessing_storage(
    operation,
    indices,
    error,
    match,
):
    factor = space(U1Irrep, {0: 1})
    target = hom((factor,), (factor,))
    tensor = TensorMap(
        target,
        _ExplodingVectorData(get_degeneracystructure(target).total_dim),
    )

    with pytest.raises(error, match=match):
        operation(tensor, indices)


@pytest.mark.parametrize("operation", [twist, flip], ids=["twist", "flip"])
def test_index_transform_rejects_invalid_tensor_and_inv_inputs(operation):
    with pytest.raises(TypeError, match=rf"{operation.__name__}.*TensorMap"):
        operation(object(), 0)  # pyright: ignore[reportArgumentType]

    factor = space(U1Irrep, {0: 1})
    target = hom((factor,), (factor,))
    tensor = TensorMap(
        target,
        _ExplodingVectorData(get_degeneracystructure(target).total_dim),
    )
    with pytest.raises(TypeError, match=r"inv.*bool"):
        operation(tensor, 0, inv=1)  # pyright: ignore[reportArgumentType]


def test_space_flip_preserves_visible_sectors_and_differs_from_dual():
    factor = space(U1Irrep, {-2: 3, 1: 5})

    flipped = factor.flip()

    assert dict(flipped.sectors) == dict(factor.sectors)
    assert flipped.is_dual
    assert flipped != factor.dual()
    assert flipped.flip() == factor


def test_hom_flip_updates_factors_and_roundtrips():
    left = space(U1Irrep, {-1: 2, 2: 3})
    right = space(U1Irrep, {-2: 5, 1: 7}).dual()
    target = hom((left,), (right,))

    flipped = target.flip((0, 1))

    assert flipped == hom((left.flip(),), (right.flip(),))
    assert flipped.flip((0, 1)) == target


def test_flip_tensorkit_row_and_column_coefficients_for_fermion_parity():
    odd = space(FermionParity, {1: 1})
    target = hom((odd,), (odd,))
    tensor = TensorMap(target, jnp.array([2.0], dtype=jnp.float32))

    row = flip(tensor, 0)
    column = flip(tensor, 1)

    # TensorKit's Z-isomorphism formulas give row=1 and column=theta=-1
    # when both source fusion-tree flags are non-dual.
    assert_allclose(row.storage.data, jnp.array([2.0], dtype=jnp.float32))
    assert_allclose(column.storage.data, jnp.array([-2.0], dtype=jnp.float32))
    assert row.space == target.flip((0,))
    assert column.space == target.flip((1,))


def test_flip_forward_inverse_roundtrips_and_forward_is_not_involutory():
    half = space(SU2Irrep, {1: 1})
    target = hom((half,), (half,))
    tensor = TensorMap(target, jnp.array([2.0], dtype=jnp.float32))

    forward_inverse = flip(flip(tensor, 0), 0, inv=True)
    inverse_forward = flip(flip(tensor, 0, inv=True), 0)
    twice_forward = flip(flip(tensor, 0), 0)

    assert forward_inverse.space == target
    assert inverse_forward.space == target
    assert_allclose(forward_inverse.storage.data, tensor.storage.data)
    assert_allclose(inverse_forward.storage.data, tensor.storage.data)
    assert_allclose(twice_forward.storage.data, -tensor.storage.data)


@pytest.mark.parametrize(
    "sector_type",
    [FermionParity, SU2Irrep],
    ids=["fermion-parity", "su2"],
)
def test_flip_matching_contracted_legs_preserves_contraction(sector_type):
    factor = space(sector_type, {1: 1})
    target = hom((factor,), (factor,))
    left = TensorMap(target, jnp.array([2.0], dtype=jnp.float32))
    right = TensorMap(target, jnp.array([3.0], dtype=jnp.float32))
    kwargs = {
        "axes": ((1,), (0,)),
        "output": (((0, 0),), ((1, 1),)),
    }

    expected = tensorcontract(left, right, **kwargs)
    result = tensorcontract(flip(left, 1), flip(right, 0), **kwargs)

    assert result.space == expected.space
    assert_allclose(result.storage.data, expected.storage.data)


def test_u1_flip_applies_nonidentity_entry_mapping():
    factor = space(U1Irrep, {-1: 1, 1: 1})
    target = hom((factor, factor), ())
    tensor = TensorMap(target, jnp.array([10.0, 20.0], dtype=jnp.float32))

    result = flip(tensor, 1)

    assert_allclose(result.storage.data, jnp.array([20.0, 10.0], dtype=jnp.float32))


def test_su2_multitree_scalar_flip_preserves_float16_and_roundtrips():
    half = space(SU2Irrep, {1: 1})
    target = hom((half, half, half, half), ())
    tensor = TensorMap(target, jnp.array([1.0, 2.0], dtype=jnp.float16))

    result = flip(tensor, (0, 2))
    restored = flip(result, (0, 2), inv=True)

    assert result.storage.data.dtype == jnp.float16
    assert restored.space == target
    assert_allclose(restored.storage.data, tensor.storage.data)


@pytest.mark.parametrize(
    ("sector_type", "sector"),
    [
        (Z2Irrep, 1),
        (Z3Irrep, 1),
        (Z4Irrep, 1),
        (FermionNumber, (1, 1)),
        (FermionParityU1Irrep, (1, 1)),
        (U1SU2Irrep, (1, 1)),
        (FermionParitySU2Irrep, (1, 1)),
        (FermionParityU1SU2Irrep, (1, 1, 1)),
    ],
    ids=[
        "z2",
        "z3",
        "z4",
        "fermion-number",
        "fermion-parity-u1",
        "u1-su2",
        "fermion-parity-su2",
        "fermion-parity-u1-su2",
    ],
)
def test_flip_roundtrips_additional_exported_sector_families(sector_type, sector):
    factor = space(sector_type, {sector: 1})
    target = hom((factor,), (factor,))
    tensor = TensorMap(target, _data_for(target))

    flipped = flip(tensor, (0, 1))
    restored = flip(flipped, (0, 1), inv=True)

    assert restored.space == target
    assert_allclose(restored.storage.data, tensor.storage.data)


@pytest.mark.parametrize("partition", ["codomain", "domain"])
def test_flip_handles_one_sided_spaces(partition):
    unit = space(U1Irrep, {0: 2})
    target = hom((unit,), ()) if partition == "codomain" else hom((), (unit,))
    tensor = TensorMap(target, _data_for(target))

    restored = flip(flip(tensor, 0), 0, inv=True)

    assert restored.space == target
    assert_allclose(restored.storage.data, tensor.storage.data)


def test_flip_handles_complex_strided_degeneracy_subblocks():
    factor = space(FermionParity, {0: 2, 1: 1})
    target = hom((factor, factor), (factor, factor))
    data = _data_for(target).astype(jnp.complex64)
    data = data + 1j * (data + 1)
    tensor = TensorMap(target, data)
    subblocks = get_degeneracystructure(target).subblockstructure

    assert any(
        math.prod(subblock.sizes) > 1 and not is_contiguous_subblock(subblock)
        for subblock in subblocks
    )
    result = flip(tensor, (0, 2))
    restored = flip(result, (0, 2), inv=True)

    assert result.storage.data.dtype == tensor.storage.data.dtype
    assert restored.space == target
    assert_allclose(restored.storage.data, tensor.storage.data)


def test_unit_insertion_distinguishes_partition_boundary_and_dual_flags():
    out0 = space(U1Irrep, {0: 2, 1: 1})
    out1 = space(U1Irrep, {0: 1, -1: 2})
    in0 = space(U1Irrep, {0: 2, 1: 1})
    in1 = space(U1Irrep, {0: 1, -1: 2})
    target = hom((out0, out1), (in0, in1))
    tensor = TensorMap(target, _data_for(target))
    position = tensor.numout
    unit = space(U1Irrep, {0: 1}, dual=True)

    left = insertleftunit(tensor, position, dual=True)
    right = insertrightunit(tensor, position, dual=True)

    expected_dims = tensor.dims[:position] + (1,) + tensor.dims[position:]
    assert left.dims == right.dims == expected_dims
    assert left.codomain == target.codomain
    assert left.domain.spaces == (unit,) + target.domain.spaces
    assert left.space[position] == unit.dual()
    assert right.codomain.spaces == target.codomain.spaces + (unit,)
    assert right.domain == target.domain
    assert right.space[position] == unit

    for result in (left, right):
        assert result.storage is tensor.storage
        restored = removeunit(result, position)
        assert restored.space == tensor.space
        assert restored.storage is tensor.storage


def test_unit_insertion_preserves_indexed_fusiontree_and_visible_sector_access():
    factor = space(U1Irrep, {0: 1, 1: 2})
    target = hom((factor,), (factor,))
    tensor = TensorMap(target, _data_for(target))
    result = insertleftunit(tensor, 0)

    source_pair = tensor.fusiontrees[-1]
    result_pair = result.fusiontrees[-1]
    row_tree, col_tree = result_pair
    visible_key = tuple(sector[0] for sector in row_tree.uncoupled) + tuple(
        -sector[0] for sector in col_tree.uncoupled
    )

    source_by_tree = tensor[source_pair]
    result_by_tree = result[result_pair]
    assert_allclose(result[visible_key], result_by_tree)
    assert_allclose(result_by_tree.reshape(-1), source_by_tree.reshape(-1))


def test_unit_insertion_defaults_cover_rank_zero_and_regular_spaces():
    unit = space(U1Irrep, {0: 1})
    empty = hom((unit,), ()).domain
    scalar_space = hom(empty, empty)
    scalar_tensor = TensorMap(scalar_space, jnp.asarray([2.0], dtype=jnp.float32))

    scalar_left = insertleftunit(scalar_tensor)
    scalar_right = insertrightunit(scalar_tensor)
    assert (scalar_left.numout, scalar_left.numin) == (0, 1)
    assert (scalar_right.numout, scalar_right.numin) == (1, 0)
    assert removeunit(scalar_left, 0).space == scalar_space
    assert removeunit(scalar_right, 0).space == scalar_space

    assert scalar_left.storage is scalar_tensor.storage
    assert scalar_right.storage is scalar_tensor.storage

    factor = space(U1Irrep, {0: 2})
    target = hom((factor,), (factor,))
    tensor = TensorMap(target, _data_for(target))
    left = insertleftunit(tensor)
    right = insertrightunit(tensor)

    assert left.space == right.space
    for result in (left, right):
        assert result.storage is tensor.storage
        assert removeunit(result, tensor.numind).space == target


def test_unit_operations_validate_complete_call_before_storage_access():
    with pytest.raises(TypeError, match=r"insertleftunit\(\) requires a TensorMap"):
        insertleftunit(object())  # pyright: ignore[reportArgumentType]
    with pytest.raises(TypeError, match=r"insertrightunit\(\) requires a TensorMap"):
        insertrightunit(object())  # pyright: ignore[reportArgumentType]
    with pytest.raises(TypeError, match=r"removeunit\(\) requires a TensorMap"):
        removeunit(object(), 0)  # pyright: ignore[reportArgumentType]

    factor = space(U1Irrep, {0: 2})
    target = hom((factor,), (factor,))
    tensor = TensorMap(target, _ExplodingVectorData(_data_for(target).size))

    for operation in (insertleftunit, insertrightunit):
        for position in (True, 1.5):
            with pytest.raises(TypeError, match="position"):
                operation(tensor, position)  # pyright: ignore[reportArgumentType]
        for position in (-1, tensor.numind + 1):
            with pytest.raises(ValueError, match="out of range"):
                operation(tensor, position)
        with pytest.raises(TypeError, match="dual"):
            operation(tensor, dual=0)  # pyright: ignore[reportArgumentType]

        result = operation(tensor)
        assert result.storage is tensor.storage

    for index in (True, 1.5):
        with pytest.raises(TypeError, match="index"):
            removeunit(tensor, index)  # pyright: ignore[reportArgumentType]
    for index in (-1, tensor.numind):
        with pytest.raises(ValueError, match="out of range"):
            removeunit(tensor, index)

    nonunit = space(U1Irrep, {1: 1})
    nonunit_space = hom((nonunit,), target.domain)
    nonunit_tensor = TensorMap(
        nonunit_space,
        _ExplodingVectorData(get_degeneracystructure(nonunit_space).total_dim),
    )
    with pytest.raises(ValueError, match="canonical unit space"):
        removeunit(nonunit_tensor, 0)


@pytest.mark.parametrize(
    ("sector_type", "unit_sector"),
    [
        (U1Irrep, 0),
        (SU2Irrep, 0),
        (FermionParity, 0),
        (Z2Irrep, 0),
        (Z3Irrep, 0),
        (Z4Irrep, 0),
        (FermionNumber, (0, 0)),
        (FermionParityU1Irrep, (0, 0)),
        (U1SU2Irrep, (0, 0)),
        (FermionParitySU2Irrep, (0, 0)),
        (FermionParityU1SU2Irrep, (0, 0, 0)),
    ],
    ids=[
        "u1",
        "su2",
        "fermion-parity",
        "z2",
        "z3",
        "z4",
        "fermion-number",
        "fermion-parity-u1",
        "u1-su2",
        "fermion-parity-su2",
        "fermion-parity-u1-su2",
    ],
)
def test_unit_operations_smoke_all_exported_sector_families(
    sector_type,
    unit_sector,
):
    factor = space(sector_type, {unit_sector: 2})
    target = hom((factor,), (factor,))
    tensor = TensorMap(target, _data_for(target))

    left = insertleftunit(tensor, 0, dual=True)
    right = insertrightunit(tensor)

    assert removeunit(left, 0).space == target
    assert removeunit(right, tensor.numind).space == target
    assert left.storage is tensor.storage
    assert right.storage is tensor.storage
