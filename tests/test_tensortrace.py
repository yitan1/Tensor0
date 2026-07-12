import jax
import jax.numpy as jnp
import pytest

import tensor0.contractions as contractions
import tensor0.structure.layout as layout_module
import tensor0.transforms as transforms
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
    get_sectorstructure,
    hom,
    permute,
    scalar,
    space,
    tensortrace,
    to_dense,
    twist,
)
from tests.cases import assert_allclose, is_contiguous_subblock


class _ExplodingVectorData:
    def __init__(self, length):
        self.shape = (length,)

    def __getitem__(self, key):
        if isinstance(key, slice) and key.start == 0 and key.stop == 0:
            return jnp.zeros((0,))
        raise AssertionError("tensortrace validation must not access storage")


def _metadata_only_tensor(target):
    size = get_degeneracystructure(target).total_dim
    return TensorMap(target, _ExplodingVectorData(size))


def _tensor(target, dtype=jnp.float32):
    size = get_degeneracystructure(target).total_dim
    data = jnp.arange(1, size + 1, dtype=jnp.float32)
    if jnp.issubdtype(dtype, jnp.complexfloating):
        data = data + 1j * (data + 1)
    return TensorMap(target, data.astype(dtype))


def _neutral_endomorphism():
    factor = space(U1Irrep, {0: 1})
    return _metadata_only_tensor(hom((factor,), (factor,)))


def _native_trace_transformer(source, destination, permutation=None):
    if permutation is None:
        permutation = (
            tuple(range(source.numout)),
            tuple(range(source.numout, source.numind)),
        )
    canonical = source.permute(*permutation)
    basis_transformer = transforms._treepermuter(
        source,
        canonical,
        *permutation,
    )
    return contractions._native.trace_transformer(
        canonical,
        destination,
        get_sectorstructure(canonical),
        get_sectorstructure(destination),
        basis_transformer,
    )


def _trace_contributions(source, destination):
    transformer = _native_trace_transformer(source, destination)
    if transformer.kind == "abelian":
        return tuple(
            (entry.src, entry.dst, entry.coeff)
            for entry in transformer.abelian_data
        )

    contributions = []
    for group in transformer.generic_data:
        for row, destination_index in enumerate(group.dst_indices):
            for column, source_index in enumerate(group.src_indices):
                coefficient = group.transform[row, column]
                if coefficient != 0:
                    contributions.append(
                        (source_index, destination_index, float(coefficient))
                    )
    return tuple(sorted(contributions))


def test_native_trace_transformer_reuses_tree_transformer_payload():
    factor = space(U1Irrep, {0: 1, 1: 1})
    source = hom((factor, factor), (factor, factor))
    destination = hom((factor,), (factor,))

    transformer = _native_trace_transformer(source, destination)
    metadata = transformer.abelian_data

    assert isinstance(transformer, contractions._native.TreeTransformer)
    assert transformer.kind == "abelian"
    assert tuple((entry.src, entry.dst, entry.coeff) for entry in metadata) == (
        (0, 0, 1.0),
        (1, 1, 1.0),
        (4, 0, 1.0),
        (5, 1, 1.0),
    )
    assert isinstance(metadata, tuple)
    assert all(isinstance(entry.src, int) and isinstance(entry.dst, int) for entry in metadata)


def test_native_rank_zero_trace_metadata_separates_sector_families():
    u1 = space(U1Irrep, {1: 1})
    half = space(SU2Irrep, {1: 1})
    u1_empty = contractions._native.make_product_space(U1Irrep, ())
    su2_empty = contractions._native.make_product_space(SU2Irrep, ())
    u1_destination = contractions._native.make_hom_products(u1_empty, u1_empty)
    su2_destination = contractions._native.make_hom_products(su2_empty, su2_empty)

    layout_module._clear_layout_caches_for_tests()
    try:
        assert _trace_contributions(
            hom((u1,), (u1,)),
            u1_destination,
        ) == ((0, 0, 1.0),)
        assert _trace_contributions(
            hom((half,), (half,)),
            su2_destination,
        ) == ((0, 0, 2.0),)
    finally:
        layout_module._clear_layout_caches_for_tests()


def test_native_trace_transformer_rejects_mixed_sector_families():
    u1 = space(U1Irrep, {0: 1})
    z2 = space(Z2Irrep, {0: 1})
    source = hom((u1, u1), (u1, u1))
    destination = hom((u1,), (u1,))
    z2_source = hom((z2, z2), (z2, z2))
    z2_destination = hom((z2,), (z2,))
    permutation = ((0, 1), (2, 3))
    canonical = source.permute(*permutation)
    basis_transformer = transforms._treepermuter(
        source,
        canonical,
        *permutation,
    )

    with pytest.raises(ValueError, match="same sector family"):
        contractions._native.trace_transformer(
            canonical,
            destination,
            get_sectorstructure(z2_source),
            get_sectorstructure(z2_destination),
            basis_transformer,
        )


@pytest.mark.parametrize(
    ("metadata", "error", "match"),
    [
        ({"axes": [(0,), (1,)]}, TypeError, "axes.*tuple.*length 2"),
        ({"axes": ((0,), [1])}, TypeError, "trace partitions.*tuple"),
        ({"axes": ((True,), (1,))}, TypeError, "trace axis.*int"),
        ({"axes": ((-1,), (1,))}, ValueError, "trace axis.*out of range"),
        ({"axes": ((0,), (2,))}, ValueError, "trace axis.*out of range"),
        ({"axes": ((0, 1), (1,))}, ValueError, "same number"),
        ({"axes": ((0, 0), (1, 1))}, ValueError, "exactly once"),
        ({"axes": ((0,), (1,)), "output": ((0,), ())}, ValueError, "exactly once"),
        ({"axes": ((), ()), "output": ((0,), ())}, ValueError, "exactly once"),
        ({"axes": ((), ()), "output": ((0, 0), (1,))}, ValueError, "exactly once"),
        ({"output": [(), ()]}, TypeError, "output.*tuple.*length 2"),
        ({"output": ([], ())}, TypeError, "output partitions.*tuple"),
        ({"output": ((True,), ())}, TypeError, "output axis.*int"),
        ({"output": ((-1,), ())}, ValueError, "output axis.*out of range"),
        ({"output": ((2,), ())}, ValueError, "output axis.*out of range"),
        ({"conjugate": 0}, TypeError, "conjugate.*bool"),
    ],
    ids=[
        "axes-outer-not-tuple",
        "axes-partition-not-tuple",
        "bool-trace-axis",
        "negative-trace-axis",
        "out-of-range-trace-axis",
        "unequal-pair-counts",
        "duplicate-trace-axis",
        "overlapping-trace-output-axis",
        "missing-axis",
        "duplicate-output-axis",
        "output-outer-not-tuple",
        "output-partition-not-tuple",
        "bool-output-axis",
        "negative-output-axis",
        "out-of-range-output-axis",
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
        tensortrace(object(), axes=((), ()), output=((), ()))


def test_tensortrace_rejects_non_dual_trace_spaces_before_storage_access():
    charge_one = space(U1Irrep, {1: 1})
    charge_two = space(U1Irrep, {2: 1})
    tensor = _metadata_only_tensor(hom((charge_one,), (charge_two,)))

    with pytest.raises(ValueError, match="trace axes.*dual-compatible"):
        tensortrace(tensor, axes=((0,), (1,)), output=((), ()))


def test_tensortrace_empty_trace_accepts_output_permutation(monkeypatch):
    first = space(U1Irrep, {0: 2})
    second = space(U1Irrep, {0: 3})
    third = space(U1Irrep, {0: 4})
    tensor = _tensor(hom((first, second), (third,)))
    expected = permute(tensor, ((2, 0), (1,)))

    def fail_trace_destination(*_args):
        raise AssertionError("empty trace must not build a trace destination")

    monkeypatch.setattr(
        contractions._native,
        "make_product_space",
        fail_trace_destination,
    )
    monkeypatch.setattr(
        contractions._native,
        "make_hom_products",
        fail_trace_destination,
    )

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
    native_sector_builder = layout_module._native.build_sectorstructure
    trace_calls = 0
    basis_calls = 0
    layout_builds = 0

    def count_trace_calls(*args):
        nonlocal trace_calls
        trace_calls += 1
        return native_trace_transformer(*args)

    def count_basis_calls(*args):
        nonlocal basis_calls
        basis_calls += 1
        return native_tree_braider(*args)

    def count_layout_builds(*args):
        nonlocal layout_builds
        layout_builds += 1
        return native_sector_builder(*args)

    transforms._clear_tree_transformer_caches_for_tests()
    layout_module._clear_layout_caches_for_tests()
    monkeypatch.setattr(
        contractions._native,
        "trace_transformer",
        count_trace_calls,
    )
    monkeypatch.setattr(transforms._native, "tree_braider", count_basis_calls)
    monkeypatch.setattr(
        layout_module._native,
        "build_sectorstructure",
        count_layout_builds,
    )
    try:
        first = tensortrace(tensor, axes=((0,), (1,)), output=((), ()))
        second = tensortrace(tensor, axes=((0,), (1,)), output=((), ()))
    finally:
        transforms._clear_tree_transformer_caches_for_tests()
        layout_module._clear_layout_caches_for_tests()

    assert trace_calls == 2
    assert basis_calls == 1
    assert layout_builds == 2
    assert_allclose(second.storage.data, first.storage.data)


def test_tensortrace_full_trace_builds_typed_rank_zero_destination(monkeypatch):
    factor = space(U1Irrep, {0: 3})
    tensor = _tensor(hom((factor,), (factor,)))
    products = []
    make_product_space = contractions._native.make_product_space

    def capture_product_space(sector_spec, spaces):
        result = make_product_space(sector_spec, spaces)
        products.append(result)
        return result

    monkeypatch.setattr(
        contractions._native,
        "make_product_space",
        capture_product_space,
    )

    result = tensortrace(tensor, axes=((0,), (1,)), output=((), ()))

    assert len(products) == 2
    assert all(len(product) == 0 for product in products)
    assert all(product.sector_spec == factor.sector_spec for product in products)
    assert result.numind == 0
    assert_allclose(scalar(result), tensor.tr())


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


def test_tensortrace_conjugation_remaps_original_domain_axes():
    traced = space(U1Irrep, {0: 2})
    open_left = space(U1Irrep, {0: 3})
    open_right = space(U1Irrep, {0: 4})
    tensor = _tensor(
        hom((open_left, open_right), (traced, traced.dual()))
    )
    expected_destination = hom((open_left.dual(),), (open_right,))
    adjoint_space = hom(tensor.space.domain, tensor.space.codomain)
    assert adjoint_space[2].dual() != adjoint_space[3]
    assert adjoint_space[0].dual() == adjoint_space[1]

    result = tensortrace(
        tensor,
        axes=((2,), (3,)),
        output=((0,), (1,)),
        conjugate=True,
    )

    assert result.space == expected_destination


def test_tensortrace_partial_trace_matches_dense_oracle(monkeypatch):
    open_out = space(U1Irrep, {0: 2})
    traced = space(U1Irrep, {0: 3})
    open_in = space(U1Irrep, {0: 4})
    tensor = _tensor(hom((open_out, traced), (open_in, traced)))
    source_subblock = get_degeneracystructure(tensor.space).subblockstructure[0]
    assert is_contiguous_subblock(source_subblock)
    expected = jnp.trace(to_dense(tensor), axis1=1, axis2=3)

    def fail_transpose(*_args, **_kwargs):
        raise AssertionError("canonical trace must not transpose source subblocks")

    monkeypatch.setattr(contractions.jnp, "transpose", fail_transpose)

    result = tensortrace(
        tensor,
        axes=((1,), (3,)),
        output=((0,), (2,)),
    )

    assert result.space == hom((open_out,), (open_in,))
    assert_allclose(to_dense(result), expected)


@pytest.mark.parametrize(
    "sector_type",
    [Z2Irrep, Z3Irrep, Z4Irrep],
    ids=["z2", "z3", "z4"],
)
def test_tensortrace_zn_partial_trace_matches_dense_oracle(sector_type):
    factor = space(sector_type, {0: 1, 1: 1})
    tensor = _tensor(hom((factor, factor), (factor, factor)))

    result = tensortrace(
        tensor,
        axes=((1,), (3,)),
        output=((0,), (2,)),
    )
    expected = jnp.trace(to_dense(tensor), axis1=1, axis2=3)

    assert result.space == hom((factor,), (factor,))
    assert_allclose(to_dense(result), expected)


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
    assert_allclose(result.storage.data, jnp.asarray([3.5], dtype=jnp.float32))
    assert_allclose(to_dense(result), expected)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.complex64])
def test_tensortrace_su2_grouped_transform_matches_dense_oracle(dtype, monkeypatch):
    half = space(SU2Irrep, {1: 1})
    source = hom(
        (half, half, half, half.dual()),
        (),
    )
    tensor = _tensor(source, dtype=dtype)
    transformer = _native_trace_transformer(
        source,
        hom((half, half), ()),
        ((1, 2, 3), (0,)),
    )
    assert transformer.kind == "generic"
    assert len(transformer.generic_data) == 1
    assert transformer.generic_data[0].transform.shape == (1, 2)

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
    permutation = ((1, 2, 3, 0, 4), (5,))
    tensor = _tensor(source)
    transformer = _native_trace_transformer(
        source,
        destination,
        permutation,
    )

    assert transformer.kind == "generic"
    assert len(transformer.generic_data) == 1
    assert transformer.generic_data[0].transform.shape == (2, 2)

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


def test_tensortrace_simplefusion_scalar_path_does_not_stack(monkeypatch):
    half = space(SU2Irrep, {1: 1})
    tensor = _tensor(hom((half,), (half,)))

    def fail_stack(*_args, **_kwargs):
        raise AssertionError("1x1 trace transform must not stack source blocks")

    monkeypatch.setattr(contractions.jnp, "stack", fail_stack)
    result = tensortrace(tensor, axes=((0,), (1,)), output=((), ()))

    assert_allclose(scalar(result), 2 * tensor.storage.data[0])


@pytest.mark.parametrize(
    ("is_dual", "expected_sign"),
    [(False, -1), (True, 1)],
    ids=["nondual-left", "dual-left"],
)
def test_tensortrace_fermion_parity_orientation_matches_tensorkit(
    is_dual,
    expected_sign,
):
    odd = space(FermionParity, {1: 1})
    factor = odd.dual() if is_dual else odd
    tensor = TensorMap(
        hom((factor,), (factor,)),
        jnp.asarray([3 + 4j], dtype=jnp.complex64),
    )

    result = tensortrace(tensor, axes=((0,), (1,)), output=((), ()))

    assert_allclose(scalar(result), expected_sign * tensor.storage.data[0])
    if is_dual:
        assert_allclose(scalar(result), tensor.tr())
    else:
        assert not bool(jnp.allclose(scalar(result), tensor.tr()))
        assert_allclose(scalar(result), twist(tensor, 0).tr())


def test_tensortrace_fermion_permutation_and_twist_are_fused():
    odd = space(FermionParity, {1: 1})
    tensor = TensorMap(
        hom((odd,), (odd,)),
        jnp.asarray([3 + 4j], dtype=jnp.complex64),
    )

    result = tensortrace(tensor, axes=((1,), (0,)), output=((), ()))
    canonical = permute(tensor, ((1,), (0,)))
    expected = tensortrace(canonical, axes=((0,), (1,)), output=((), ()))

    assert_allclose(result.storage.data, expected.storage.data)


@pytest.mark.parametrize(
    ("sector_type", "sector", "expected_coefficient"),
    [
        (FermionNumber, (0, 1), -1),
        (FermionParityU1Irrep, (1, 0), -1),
        (U1SU2Irrep, (0, 1), 2),
        (FermionParitySU2Irrep, (1, 1), -2),
        (FermionParityU1SU2Irrep, (1, 0, 1), -2),
    ],
    ids=[
        "fermion-number",
        "fermion-parity-u1",
        "u1-su2",
        "fermion-parity-su2",
        "fermion-parity-u1-su2",
    ],
)
def test_tensortrace_product_sector_orientation_matches_tensorkit(
    sector_type,
    sector,
    expected_coefficient,
):
    factor = space(sector_type, {sector: 1})
    tensor = TensorMap(
        hom((factor,), (factor,)),
        jnp.asarray([3.0], dtype=jnp.float32),
    )

    result = tensortrace(tensor, axes=((0,), (1,)), output=((), ()))

    assert_allclose(scalar(result), expected_coefficient * tensor.storage.data[0])
    assert_allclose(scalar(result), twist(tensor, 0).tr())
    if expected_coefficient < 0:
        assert not bool(jnp.allclose(scalar(result), tensor.tr()))
    else:
        assert_allclose(scalar(result), tensor.tr())


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


def test_tensortrace_coalesces_strided_subblocks(monkeypatch):
    factor = space(U1Irrep, {0: 2, 1: 1})
    source = hom((factor, factor, factor), (factor, factor, factor))
    destination = hom((factor, factor), (factor, factor))
    source_degeneracy = get_degeneracystructure(source)
    destination_degeneracy = get_degeneracystructure(destination)
    metadata = _trace_contributions(source, destination)
    source_indices = tuple(src for src, _dst, _coeff in metadata)
    destination_indices = tuple(dst for _src, dst, _coeff in metadata)
    assert len(set(destination_indices)) < len(destination_indices)
    assert any(
        not is_contiguous_subblock(source_degeneracy.subblockstructure[index])
        for index in source_indices
    )
    assert any(
        not is_contiguous_subblock(
            destination_degeneracy.subblockstructure[index]
        )
        for index in destination_indices
    )
    tensor = _tensor(source)
    scatter_calls = []
    add_to_subblock = contractions._add_to_subblock

    def count_scatter(storage, subblock, value):
        scatter_calls.append(subblock.static_key)
        return add_to_subblock(storage, subblock, value)

    monkeypatch.setattr(contractions, "_add_to_subblock", count_scatter)

    result = tensortrace(
        tensor,
        axes=((2,), (5,)),
        output=((0, 1), (3, 4)),
    )
    expected = jnp.trace(to_dense(tensor), axis1=2, axis2=5)

    assert result.space == destination
    assert len(scatter_calls) == len(set(destination_indices))
    assert len(scatter_calls) == len(set(scatter_calls))
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
