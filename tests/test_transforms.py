import math
import sys
from collections.abc import Callable
from types import SimpleNamespace
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tensor0.operations.transforms as transforms
from tensor0 import (
    ComplexSpace,
    FermionNumber,
    FermionParity,
    FermionParitySU2Irrep,
    FermionParityU1Irrep,
    FermionParityU1SU2Irrep,
    SU2Irrep,
    TensorMap,
    Trivial,
    U1Irrep,
    U1SU2Irrep,
    Z2Irrep,
    Z3Irrep,
    Z4Irrep,
    braid,
    flip,
    from_dense,
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
from tensor0._stride import StridedView, add, materialize
from tensor0._stride import _tensor_ops as _transform
from tensor0._stride._tensor_ops import _strided_affine_transform, _strided_tree_transform
from tests.stride.support.availability import native_available
from tensor0.structure import get_degeneracystructure, get_sectorstructure
from tests.cases import (
    InaccessibleVectorData,
    assert_allclose,
    is_contiguous_subblock,
    transform_cases,
    zero_based_float_data_for,
)


@pytest.fixture
def checked_affine_destinations(monkeypatch):
    calls = []
    assert transforms._strided_affine_transform is _strided_affine_transform
    assert transforms._strided_tree_transform is _strided_tree_transform

    def execute(source, **arguments):
        arguments["entries"] = tuple(arguments["entries"])
        destinations = [
            subblock.offset + sum(index * stride for index, stride in zip(coordinates, subblock.strides, strict=True))
            for _, destination_index, _ in arguments["entries"]
            for subblock in (arguments["destination_subblocks"][destination_index],)
            for coordinates in np.ndindex(tuple(subblock.sizes))
        ]
        assert sorted(destinations) == list(range(arguments["output_size"]))
        calls.append(1)
        return _strided_affine_transform(source, **arguments)

    monkeypatch.setattr(transforms, "_strided_affine_transform", execute)
    yield
    assert calls


@pytest.fixture
def checked_abelian_tree_destinations(monkeypatch):
    calls = []
    assert transforms._strided_tree_transform is _strided_tree_transform

    def execute(source, **parameters):
        assert parameters["transformer"].kind == "abelian"
        _tree_address_terms(parameters)
        calls.append(1)
        return _strided_tree_transform(source, **parameters)

    monkeypatch.setattr(transforms, "_strided_tree_transform", execute)
    yield
    assert calls


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


def _large_u1_permute_case():
    left = space(U1Irrep, {0: 16, 1: 16})
    right = space(U1Irrep, {0: 12, 1: 12})
    incoming = space(U1Irrep, {0: 8, 1: 8})
    target = hom((left, right), (incoming,))
    permutation = ((1,), (0, 2))
    data = _data_for(target) / get_degeneracystructure(target).total_dim
    return TensorMap(target, data), permutation


def _large_su2_permute_case():
    factor = space(SU2Irrep, {0: 8, 1: 8})
    target = hom((factor, factor), (factor, factor))
    permutation = ((1, 0), (3, 2))
    data = _data_for(target) / get_degeneracystructure(target).total_dim
    return TensorMap(target, data), permutation


def _transpose_reference_block(block, permutation):
    if not permutation:
        return block
    return jnp.transpose(block, permutation)


def _apply_abelian_reference(
    result,
    source,
    permutation,
    data,
    source_subblocks,
    destination_subblocks,
):
    for entry in data:
        source_subblock = source_subblocks[entry.src]
        source_view = StridedView(
            source,
            tuple(source_subblock.sizes),
            tuple(source_subblock.strides),
            source_subblock.offset,
        )
        block = materialize(
            source_view,
            dtype=result.dtype,
        )
        block = _transpose_reference_block(block, permutation)
        destination_subblock = destination_subblocks[entry.dst]
        destination_view = StridedView(
            result,
            tuple(destination_subblock.sizes),
            tuple(destination_subblock.strides),
            destination_subblock.offset,
        )
        result = add(
            destination_view,
            StridedView.from_dense(
                jnp.asarray(entry.coeff, dtype=result.dtype) * block,
                destination_view.sizes,
            ),
        ).data
    return result


def _apply_generic_reference(
    result,
    source,
    permutation,
    data,
    source_subblocks,
    destination_subblocks,
):
    for entry in data:
        transform = jnp.asarray(entry.transform, dtype=result.dtype)
        source_indices = entry.src_indices
        destination_indices = entry.dst_indices
        if (
            transform.size == 1
            and len(source_indices) == 1
            and len(destination_indices) == 1
        ):
            source_subblock = source_subblocks[source_indices[0]]
            block = materialize(
                StridedView(
                    source,
                    tuple(source_subblock.sizes),
                    tuple(source_subblock.strides),
                    source_subblock.offset,
                ),
                dtype=result.dtype,
            )
            block = _transpose_reference_block(block, permutation)
            destination_subblock = destination_subblocks[destination_indices[0]]
            destination_view = StridedView(
                result,
                tuple(destination_subblock.sizes),
                tuple(destination_subblock.strides),
                destination_subblock.offset,
            )
            result = add(
                destination_view,
                StridedView.from_dense(
                    transform.reshape(()) * block,
                    destination_view.sizes,
                ),
            ).data
            continue

        source_sizes = tuple(source_subblocks[source_indices[0]].sizes)
        source_rows = tuple(
            materialize(
                StridedView(
                    source,
                    tuple(source_subblocks[index].sizes),
                    tuple(source_subblocks[index].strides),
                    source_subblocks[index].offset,
                ),
                dtype=result.dtype,
            ).reshape(-1)
            for index in source_indices
        )
        destination_rows = transform @ jnp.stack(source_rows, axis=0)

        for row, destination_index in enumerate(destination_indices):
            block = destination_rows[row, :].reshape(source_sizes)
            block = _transpose_reference_block(block, permutation)
            destination_subblock = destination_subblocks[destination_index]
            destination_view = StridedView(
                result,
                tuple(destination_subblock.sizes),
                tuple(destination_subblock.strides),
                destination_subblock.offset,
            )
            result = add(
                destination_view,
                StridedView.from_dense(block, destination_view.sizes),
            ).data
    return result


@pytest.mark.parametrize(
    "factory",
    [_large_u1_permute_case, _large_su2_permute_case],
    ids=["abelian", "generic"],
)
def test_tree_transform_rejects_unsupported_tpu_lowering(factory):
    tensor, permutation = factory()
    run = jax.jit(
        lambda source: (
            permute(
                TensorMap(tensor.space, source),
                permutation,
            ).storage.data
        )
    )

    with pytest.raises(
        NotImplementedError,
        match="tensor0_stride_.*not found for platform tpu",
    ):
        run.trace(tensor.storage.data).lower(lowering_platforms=("tpu",))


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


@pytest.mark.skipif(not native_available(), reason="native CPU stride is unavailable")
@pytest.mark.parametrize("dtype", [jnp.float16, jnp.float32, jnp.complex64])
def test_large_abelian_permute_uses_one_native_stride_call_without_indices(checked_abelian_tree_destinations, dtype):
    tensor, permutation = _large_u1_permute_case()
    source = tensor.storage.data.astype(dtype)
    if dtype == jnp.complex64:
        source = source + 1j * source
    tensor = TensorMap(tensor.space, source)
    destination = tensor.space.permute(*permutation)
    _, oracle = _tree_functions(_tree_parameters_for(tensor.space, permutation))

    run = jax.jit(
        lambda source: (
            permute(
                TensorMap(tensor.space, source),
                permutation,
            ).storage.data
        )
    )
    lowered = run.lower(tensor.storage.data)
    stablehlo = str(lowered.compiler_ir(dialect="stablehlo"))
    compiled = lowered.compile()
    actual = compiled(tensor.storage.data)
    actual.block_until_ready()

    assert "tensor0_stride_accumulation_" in stablehlo
    assert "@tensor0_stride1_" not in stablehlo

    _assert_dense_transpose(
        TensorMap(destination, actual),
        tensor,
        destination,
        (1, 0, 2),
    )
    assert stablehlo.count("stablehlo.custom_call") == 1
    assert "signed_permutation" not in stablehlo
    assert "stablehlo.gather" not in stablehlo
    assert "stablehlo.scatter" not in stablehlo
    assert "stablehlo.dot_general" not in stablehlo
    assert "stablehlo.multiply" not in stablehlo
    for multiplier in (1, 2):
        assert_allclose(compiled(source * multiplier), oracle(source * multiplier))
    assert_allclose(jax.jit(jax.vmap(run))(jnp.stack((source, source * 2))),
                    jnp.stack((oracle(source), oracle(source * 2))))

@pytest.mark.usefixtures("checked_abelian_tree_destinations")
def test_large_abelian_permute_preserves_jvp_vjp_and_batching():
    tensor, permutation = _large_u1_permute_case()
    destination = tensor.space.permute(*permutation)
    transformer = transforms._treepermuter(
        tensor.space,
        destination,
        *permutation,
    )
    source_layout = get_degeneracystructure(tensor.space)
    destination_layout = get_degeneracystructure(destination)
    p = permutation[0] + permutation[1]

    def run(source):
        return permute(TensorMap(tensor.space, source), permutation).storage.data

    def baseline(source):
        result = jnp.zeros(
            (destination_layout.total_dim,),
            dtype=transforms._transform_result_dtype(source, transformer),
        )
        return _apply_abelian_reference(
            result,
            source,
            p,
            transformer.abelian_data,
            source_layout.subblockstructure,
            destination_layout.subblockstructure,
        )

    source = tensor.storage.data
    tangent = jnp.linspace(0.0, 1.0, source.size, dtype=source.dtype)
    primal, actual_tangent = jax.jvp(run, (source,), (tangent,))
    expected = baseline(source)
    expected_tangent = baseline(tangent)
    _, pullback = jax.vjp(run, source)
    cotangent = jnp.linspace(-1.0, 1.0, primal.size, dtype=primal.dtype)
    actual_cotangent = pullback(cotangent)[0]
    _, baseline_pullback = jax.vjp(baseline, source)
    expected_cotangent = baseline_pullback(cotangent)[0]
    batch = jnp.stack((source, 2 * source))

    assert_allclose(primal, expected)
    assert_allclose(actual_tangent, expected_tangent)
    assert_allclose(actual_cotangent, expected_cotangent)
    assert_allclose(jax.jit(jax.vmap(run))(batch), jnp.stack((expected, 2 * expected)))


@pytest.mark.skipif(not native_available(), reason="native CPU stride is unavailable")
def test_large_su2_permute_uses_pack_matmul_unpack_without_indices():
    tensor, permutation = _large_su2_permute_case()
    destination = tensor.space.permute(*permutation)
    transformer = transforms._treepermuter(
        tensor.space,
        destination,
        *permutation,
    )
    source_layout = get_degeneracystructure(tensor.space)
    destination_layout = get_degeneracystructure(destination)
    p = permutation[0] + permutation[1]

    def baseline(source):
        result = jnp.zeros(
            (destination_layout.total_dim,),
            dtype=transforms._transform_result_dtype(source, transformer),
        )
        return _apply_generic_reference(
            result,
            source,
            p,
            transformer.generic_data,
            source_layout.subblockstructure,
            destination_layout.subblockstructure,
        )

    run = jax.jit(
        lambda source: (
            permute(
                TensorMap(tensor.space, source),
                permutation,
            ).storage.data
        )
    )
    lowered = run.lower(tensor.storage.data)
    stablehlo = str(lowered.compiler_ir(dialect="stablehlo"))
    expected = baseline(tensor.storage.data)
    expected.block_until_ready()
    actual = run(tensor.storage.data)
    actual.block_until_ready()

    assert_allclose(actual, expected)
    assert stablehlo.count("stablehlo.custom_call") == 3
    for target in ("copy", "accumulation", "update"):
        assert f"tensor0_stride_{target}_" in stablehlo
    assert "@tensor0_stride1_" not in stablehlo
    assert "signed_permutation" not in stablehlo
    expected_dots = sum(
        len(entry.src_indices) > 1 or len(entry.dst_indices) > 1
        for entry in transformer.generic_data
    )
    assert stablehlo.count("stablehlo.dot_general") == expected_dots
    assert "stablehlo.gather" not in stablehlo
    assert "stablehlo.scatter" not in stablehlo


@pytest.mark.skipif(not native_available(), reason="native CPU stride is unavailable")
def test_large_float16_su2_permute_matches_grouped_reference():
    tensor, permutation = _large_su2_permute_case()
    source = tensor.storage.data.astype(jnp.float16)
    lowered = jax.jit(
        lambda value: permute(
            TensorMap(tensor.space, value),
            permutation,
        ).storage.data
    ).lower(source)
    stablehlo = str(lowered.compiler_ir(dialect="stablehlo"))
    run = lowered.compile()

    destination = tensor.space.permute(*permutation)
    transformer = transforms._treepermuter(tensor.space, destination, *permutation)
    destination_layout = get_degeneracystructure(destination)
    expected = _apply_generic_reference(
        jnp.zeros((destination_layout.total_dim,), dtype=jnp.float32),
        source,
        permutation[0] + permutation[1],
        transformer.generic_data,
        get_degeneracystructure(tensor.space).subblockstructure,
        destination_layout.subblockstructure,
    )
    actual = run(source)
    actual.block_until_ready()

    assert actual.dtype == jnp.float32
    assert stablehlo.count("stablehlo.custom_call") == 3
    for target in ("copy", "accumulation", "update"):
        assert f"tensor0_stride_{target}_" in stablehlo
    assert "@tensor0_stride1_" not in stablehlo
    assert "stablehlo.gather" not in stablehlo
    assert "stablehlo.scatter" not in stablehlo
    assert_allclose(actual, expected)


def test_large_su2_permute_preserves_jvp_vjp_and_batching():
    tensor, permutation = _large_su2_permute_case()
    destination = tensor.space.permute(*permutation)
    transformer = transforms._treepermuter(
        tensor.space,
        destination,
        *permutation,
    )
    source_layout = get_degeneracystructure(tensor.space)
    destination_layout = get_degeneracystructure(destination)
    p = permutation[0] + permutation[1]

    def run(source):
        return permute(TensorMap(tensor.space, source), permutation).storage.data

    def baseline(source):
        result = jnp.zeros(
            (destination_layout.total_dim,),
            dtype=transforms._transform_result_dtype(source, transformer),
        )
        return _apply_generic_reference(
            result,
            source,
            p,
            transformer.generic_data,
            source_layout.subblockstructure,
            destination_layout.subblockstructure,
        )

    source = tensor.storage.data
    tangent = jnp.linspace(0.0, 1.0, source.size, dtype=source.dtype)
    primal, actual_tangent = jax.jvp(run, (source,), (tangent,))
    expected, expected_tangent = jax.jvp(
        baseline,
        (source,),
        (tangent,),
    )
    cotangent = jnp.linspace(-1.0, 1.0, primal.size, dtype=primal.dtype)
    _, pullback = jax.vjp(run, source)
    _, baseline_pullback = jax.vjp(baseline, source)
    batch = jnp.stack((source, 2 * source))

    assert_allclose(primal, expected)
    assert_allclose(actual_tangent, expected_tangent)
    assert_allclose(pullback(cotangent)[0], baseline_pullback(cotangent)[0])
    assert_allclose(jax.jit(jax.vmap(run))(batch), jnp.stack((expected, 2 * expected)))


def test_large_complex_su2_permute_matches_pack_matmul_unpack_reference():
    real_tensor, permutation = _large_su2_permute_case()
    source = real_tensor.storage.data.astype(jnp.complex64) * (1.0 + 0.25j)
    tensor = TensorMap(real_tensor.space, source)
    destination = tensor.space.permute(*permutation)
    transformer = transforms._treepermuter(
        tensor.space,
        destination,
        *permutation,
    )
    source_layout = get_degeneracystructure(tensor.space)
    destination_layout = get_degeneracystructure(destination)
    p = permutation[0] + permutation[1]
    expected = _apply_generic_reference(
        jnp.zeros((destination_layout.total_dim,), dtype=source.dtype),
        source,
        p,
        transformer.generic_data,
        source_layout.subblockstructure,
        destination_layout.subblockstructure,
    )
    expected.block_until_ready()
    run = jax.jit(
        lambda value: (
            permute(
                TensorMap(tensor.space, value),
                permutation,
            ).storage.data
        )
    )
    stablehlo = str(run.lower(source).compiler_ir(dialect="stablehlo"))
    actual = run(source)
    actual.block_until_ready()

    assert "stablehlo.gather" not in stablehlo
    assert "stablehlo.scatter" not in stablehlo
    assert_allclose(actual, expected)


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


@pytest.mark.usefixtures("checked_abelian_tree_destinations")
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


@pytest.mark.usefixtures("checked_abelian_tree_destinations")
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
        not is_contiguous_subblock(source_subblocks[index]) for index in source_indices
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


@pytest.mark.usefixtures("checked_abelian_tree_destinations")
def test_fermion_parity_odd_odd_permute_matches_public_dense_phase():
    tensor = _odd_odd_fermion_tensor()

    result = permute(tensor, ((1, 0), ()))

    assert_allclose(to_dense(result), -jnp.transpose(to_dense(tensor), (1, 0)))


@pytest.mark.usefixtures("checked_abelian_tree_destinations")
def test_braid_matches_default_permute_for_identity_levels():
    tensor = _odd_odd_fermion_tensor()

    result = braid(tensor, ((1, 0), ()), (0, 1))
    expected = permute(tensor, ((1, 0), ()))

    assert result.space == expected.space
    assert_allclose(result.storage.data, expected.storage.data)


@pytest.mark.usefixtures("checked_abelian_tree_destinations")
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


@pytest.mark.usefixtures("checked_affine_destinations")
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


def test_nontrivial_twist_requires_jax_backed_storage():
    factor = space(FermionParity, {0: 1, 1: 1})
    target = hom((factor,), (factor,))
    tensor = TensorMap(
        target,
        InaccessibleVectorData(get_degeneracystructure(target).total_dim),
    )

    with pytest.raises(TypeError, match=r"^twist\(\) requires JAX-backed storage"):
        twist(tensor, 0)


@pytest.mark.skipif(
    not native_available(),
    reason="native CPU stride is unavailable",
)
@pytest.mark.parametrize("dtype", [jnp.float16, jnp.float32, jnp.complex64])
def test_twist_and_flip_each_lower_as_one_complete_native_map(checked_affine_destinations, dtype):
    factor = space(FermionParity, {0: 2, 1: 1})
    target = hom((factor, factor), (factor, factor))
    source = _data_for(target).astype(dtype)
    if dtype == jnp.complex64:
        source = source + 1j * (source + 1)
    operations = (
        jax.jit(
            lambda data: twist(TensorMap(target, data), 0).storage.data
        ),
        jax.jit(
            lambda data: flip(TensorMap(target, data), (0, 2)).storage.data
        ),
    )

    for operation in operations:
        lowered = operation.lower(source)
        stablehlo = str(lowered.compiler_ir("stablehlo"))
        compiled = lowered.compile()
        result = compiled(source)
        result.block_until_ready()

        assert stablehlo.count("stablehlo.custom_call") == 1
        assert "tensor0_stride_accumulation_" in stablehlo
        assert "@tensor0_stride1_" not in stablehlo
        for name in ("stablehlo.gather", "stablehlo.scatter", "stablehlo.multiply", "stride_update_"):
            assert name not in stablehlo
        assert_allclose(compiled(source * 2), result * 2)
        for count in (0, 2):
            batch = jnp.broadcast_to(source, (count, *source.shape))
            assert_allclose(jax.jit(jax.vmap(operation))(batch),
                            jnp.broadcast_to(result, (count, *source.shape)))


@pytest.mark.usefixtures("checked_affine_destinations")
def test_twist_and_flip_complete_maps_support_jax_transformations():
    factor = space(FermionParity, {0: 2, 1: 1})
    target = hom((factor, factor), (factor, factor))
    source = _data_for(target)
    tangent = jnp.linspace(-1.0, 1.0, source.size, dtype=source.dtype)
    cotangent = jnp.linspace(1.0, -1.0, source.size, dtype=source.dtype)
    operations = (
        lambda data: twist(TensorMap(target, data), 0).storage.data,
        lambda data: flip(TensorMap(target, data), (0, 2)).storage.data,
    )

    for operation in operations:
        primal, actual_tangent = jax.jvp(operation, (source,), (tangent,))
        pullback = jax.vjp(operation, source)[1]
        source_cotangent = pullback(cotangent)[0]

        assert_allclose(actual_tangent, operation(tangent))
        assert_allclose(
            jnp.vdot(primal, cotangent),
            jnp.vdot(source, source_cotangent),
        )
        assert_allclose(
            jax.vmap(operation)(jnp.stack((source, 2 * source))),
            jnp.stack((primal, 2 * primal)),
        )


@pytest.mark.usefixtures("checked_affine_destinations")
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


@pytest.mark.usefixtures("checked_affine_destinations")
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
        InaccessibleVectorData(get_degeneracystructure(target).total_dim),
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
        operation(tensor, (), inv=1)


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
        InaccessibleVectorData(get_degeneracystructure(target).total_dim),
    )

    with pytest.raises(error, match=match):
        operation(tensor, indices)


@pytest.mark.parametrize("operation", [twist, flip], ids=["twist", "flip"])
def test_index_transform_rejects_invalid_tensor_and_inv_inputs(operation):
    with pytest.raises(TypeError, match=rf"{operation.__name__}.*TensorMap"):
        operation(object(), 0)

    factor = space(U1Irrep, {0: 1})
    target = hom((factor,), (factor,))
    tensor = TensorMap(
        target,
        InaccessibleVectorData(get_degeneracystructure(target).total_dim),
    )
    with pytest.raises(TypeError, match=r"inv.*bool"):
        operation(tensor, 0, inv=1)


def test_data_transform_rejects_non_jax_storage_after_metadata_validation():
    factor = space(U1Irrep, {0: 2})
    target = hom((factor,), (factor,))
    tensor = TensorMap(
        target,
        InaccessibleVectorData(get_degeneracystructure(target).total_dim),
    )

    with pytest.raises(TypeError, match="flip.*JAX-backed storage"):
        flip(tensor, 0)
    with pytest.raises(TypeError, match="tree transform.*JAX-backed storage"):
        transpose(tensor)


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


@pytest.mark.usefixtures("checked_affine_destinations")
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


@pytest.mark.usefixtures("checked_affine_destinations")
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


@pytest.mark.usefixtures("checked_affine_destinations")
def test_u1_flip_applies_nonidentity_entry_mapping():
    factor = space(U1Irrep, {-1: 1, 1: 1})
    target = hom((factor, factor), ())
    tensor = TensorMap(target, jnp.array([10.0, 20.0], dtype=jnp.float32))

    result = flip(tensor, 1)

    assert_allclose(result.storage.data, jnp.array([20.0, 10.0], dtype=jnp.float32))


@pytest.mark.usefixtures("checked_affine_destinations")
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
@pytest.mark.usefixtures("checked_affine_destinations")
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


@pytest.mark.usefixtures("checked_affine_destinations")
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
    tensor = TensorMap(target, InaccessibleVectorData(_data_for(target).size))

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
        InaccessibleVectorData(get_degeneracystructure(nonunit_space).total_dim),
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


def test_tensormap_transform_methods_match_functional_forms():
    tensor = _odd_odd_fermion_tensor()
    inserted_left = insertleftunit(tensor, 1, dual=True)
    cases: tuple[
        tuple[
            TensorMap,
            str,
            Callable[..., TensorMap],
            tuple[object, ...],
            dict[str, object],
        ],
        ...,
    ] = (
        (tensor, "flip", flip, ((0,),), {"inv": True}),
        (tensor, "twist", twist, (0,), {"inv": True}),
        (tensor, "insertleftunit", insertleftunit, (1,), {"dual": True}),
        (tensor, "insertrightunit", insertrightunit, (1,), {"dual": True}),
        (inserted_left, "removeunit", removeunit, (1,), {}),
    )

    for source, method_name, function, args, kwargs in cases:
        actual = getattr(source, method_name)(*args, **kwargs)
        expected = cast(Callable[..., TensorMap], function)(source, *args, **kwargs)

        assert actual.space == expected.space
        assert actual.dtype == expected.dtype
        assert_allclose(actual.storage.data, expected.storage.data)


@pytest.mark.parametrize(
    ("method_name", "function", "args", "kwargs", "error"),
    [
        ("flip", flip, ([0],), {}, TypeError),
        ("twist", twist, (0,), {"inv": 1}, TypeError),
        ("insertleftunit", insertleftunit, (True,), {}, TypeError),
        ("insertrightunit", insertrightunit, (None,), {"dual": 0}, TypeError),
        ("removeunit", removeunit, (True,), {}, TypeError),
    ],
)
def test_tensormap_transform_methods_match_functional_errors(
    method_name,
    function,
    args,
    kwargs,
    error,
):
    tensor = _simple_u1_tensor()

    with pytest.raises(error) as functional_error:
        function(tensor, *args, **kwargs)
    with pytest.raises(error) as method_error:
        getattr(tensor, method_name)(*args, **kwargs)

    assert str(method_error.value) == str(functional_error.value)


@pytest.mark.parametrize(
    "operation",
    [
        lambda tensor, p: permute(tensor, p),
        lambda tensor, p: braid(tensor, p, (5, 1, 2, 0)),
        lambda tensor, p: transpose(tensor, p),
    ],
    ids=["permute", "bosonic-braid", "transpose"],
)
def test_trivial_index_transforms_match_direct_dense_oracle_with_dual_legs(
    operation,
):
    target = hom(
        (ComplexSpace(2), ComplexSpace(3, dual=True)),
        (ComplexSpace(4), ComplexSpace(5, dual=True)),
    )
    dense = jnp.arange(120, dtype=jnp.float32).reshape(2, 3, 4, 5) * (1.0 + 0.25j)
    tensor = from_dense(target, dense)
    permutation = ((1, 3), (0, 2))

    result = operation(tensor, permutation)

    assert result.space == target.permute(*permutation)
    assert result.numout == result.numin == 2
    assert result.storage.data.dtype == tensor.storage.data.dtype
    assert jnp.array_equal(to_dense(result), jnp.transpose(dense, (1, 3, 0, 2)))


def test_trivial_index_transforms_preserve_identity_and_rank_zero_storage_reuse():
    target = hom((), (), sector_type=Trivial)
    tensor = from_dense(target, jnp.asarray(3.0, dtype=jnp.float32))
    identity = ((), ())

    assert permute(tensor, identity) is tensor
    assert braid(tensor, identity, ()) is tensor
    assert transpose(tensor, identity) is tensor
    assert transpose(tensor) is tensor

    rank_two_target = hom((ComplexSpace(2),), (ComplexSpace(3),))
    rank_two = TensorMap(rank_two_target, jnp.zeros((6,), dtype=jnp.float32))
    assert braid(rank_two, ((0,), (1,)), (0, 1)) is rank_two
    u1_factor = space(U1Irrep, {0: 2})
    u1_target = hom((u1_factor,), (u1_factor,))
    u1_tensor = TensorMap(u1_target, jnp.zeros((4,), dtype=jnp.float32))
    assert braid(u1_tensor, ((0,), (1,)), (0, 1)) is u1_tensor


@pytest.mark.parametrize(
    "operation",
    [
        lambda tensor: permute(tensor, ((0,), (0,))),
        lambda tensor: braid(tensor, ((1,), (0,)), (0,)),
        lambda tensor: transpose(tensor, ((0,), (0,))),
    ],
    ids=[
        "permute-duplicate",
        "braid-level-count",
        "transpose-duplicate",
    ],
)
def test_trivial_index_transform_validation_matches_nontrivial_path(operation):
    trivial_target = hom((ComplexSpace(2),), (ComplexSpace(3),))
    u1_factor = space(U1Irrep, {0: 2})
    u1_target = hom((u1_factor,), (u1_factor,))
    tensors = (
        TensorMap(trivial_target, jnp.zeros((6,), dtype=jnp.float32)),
        TensorMap(u1_target, jnp.zeros((4,), dtype=jnp.float32)),
    )
    errors = []

    for tensor in tensors:
        with pytest.raises((TypeError, ValueError)) as error:
            operation(tensor)
        errors.append((type(error.value), str(error.value)))

    assert errors[0] == errors[1]


@pytest.mark.parametrize(
    ("level", "error", "message"),
    [
        (-1, ValueError, "braid levels must be non-negative"),
        (
            2 * sys.maxsize + 2,
            OverflowError,
            "braid level is too large for a platform unsigned integer",
        ),
    ],
)
def test_braid_levels_reject_values_outside_platform_unsigned_range(
    level,
    error,
    message,
):
    trivial_target = hom((ComplexSpace(2),), (ComplexSpace(3),))
    u1_factor = space(U1Irrep, {0: 2})
    u1_target = hom((u1_factor,), (u1_factor,))
    permutation = ((1,), (0,))

    for target in (trivial_target, u1_target):
        tensor = TensorMap(
            target,
            jnp.zeros((get_degeneracystructure(target).total_dim,), dtype=jnp.float32),
        )
        with pytest.raises(error, match=message):
            braid(tensor, permutation, (level, 0))


def test_trivial_permute_and_braid_accept_noncyclic_permutations():
    target = hom(
        (ComplexSpace(2), ComplexSpace(3)),
        (ComplexSpace(4), ComplexSpace(5)),
    )
    dense = jnp.arange(120, dtype=jnp.float32).reshape(2, 3, 4, 5)
    tensor = from_dense(target, dense)
    permutation = ((1, 0), (3, 2))
    expected = jnp.transpose(dense, (1, 0, 3, 2))

    assert jnp.array_equal(to_dense(permute(tensor, permutation)), expected)
    assert jnp.array_equal(
        to_dense(braid(tensor, permutation, (5, 1, 2, 0))),
        expected,
    )


def test_trivial_transpose_noncyclic_validation_matches_nontrivial_path():
    trivial_target = hom(
        (ComplexSpace(2), ComplexSpace(3)),
        (ComplexSpace(4), ComplexSpace(5)),
    )
    u1_factor = space(U1Irrep, {0: 2})
    u1_target = hom((u1_factor, u1_factor), (u1_factor, u1_factor))
    tensors = (
        TensorMap(trivial_target, jnp.zeros((120,), dtype=jnp.float32)),
        TensorMap(
            u1_target,
            jnp.zeros(
                (get_degeneracystructure(u1_target).total_dim,),
                dtype=jnp.float32,
            ),
        ),
    )
    errors = []

    for tensor in tensors:
        with pytest.raises(ValueError) as error:
            transpose(tensor, ((1, 0), (3, 2)))
        errors.append(str(error.value))

    assert errors == [
        "fusion tree transpose requires a cyclic planar permutation",
        "fusion tree transpose requires a cyclic planar permutation",
    ]


def test_zero_storage_transpose_rejects_noncyclic_permutations():
    trivial_factors = (
        ComplexSpace(2),
        ComplexSpace(0),
        ComplexSpace(3),
        ComplexSpace(4),
    )
    empty_u1 = space(U1Irrep, {})
    u1_factor = space(U1Irrep, {0: 2})
    u1_factors = (u1_factor, empty_u1, u1_factor, u1_factor)
    targets = (
        hom(trivial_factors[:2], trivial_factors[2:]),
        hom(u1_factors[:2], u1_factors[2:]),
    )
    permutation = ((1, 0), (3, 2))

    for target in targets:
        tensor = TensorMap(target, jnp.zeros((0,), dtype=jnp.float32))
        with pytest.raises(
            ValueError,
            match="fusion tree transpose requires a cyclic planar permutation",
        ):
            transpose(tensor, permutation)


def test_trivial_index_transform_execution_bypasses_transformer_metadata(
    monkeypatch,
):
    target = hom(
        (ComplexSpace(2), ComplexSpace(3)),
        (ComplexSpace(4), ComplexSpace(5)),
    )
    dense = jnp.arange(120, dtype=jnp.float32).reshape(2, 3, 4, 5)
    tensor = from_dense(target, dense)
    permutation = ((1, 3), (0, 2))

    def explode(*_args, **_kwargs):
        raise AssertionError("Trivial numerical execution must not request metadata")

    monkeypatch.setattr(transforms, "_treepermuter", explode)
    monkeypatch.setattr(transforms, "_treebraider", explode)
    monkeypatch.setattr(transforms, "_treetransposer", explode)
    monkeypatch.setattr(transforms, "get_sectorstructure", explode)
    monkeypatch.setattr(transforms, "get_degeneracystructure", explode)

    results = (
        permute(tensor, permutation),
        braid(tensor, permutation, (0, 1, 2, 3)),
        transpose(tensor, permutation),
    )

    expected = jnp.transpose(dense, (1, 3, 0, 2))
    assert all(jnp.array_equal(to_dense(result), expected) for result in results)


def test_trivial_index_transform_handles_zero_axis_jit_vmap_and_grad():
    target = hom(
        (ComplexSpace(2), ComplexSpace(0)),
        (ComplexSpace(3), ComplexSpace(4)),
    )
    destination = target.permute((1, 3), (0, 2))
    dense = jnp.zeros((2, 0, 3, 4), dtype=jnp.float32)

    @jax.jit
    def run(value):
        tensor = TensorMap(target, value.reshape(-1))
        return transpose(tensor, ((1, 3), (0, 2))).storage.data

    expected = jnp.transpose(dense, (1, 3, 0, 2)).reshape(-1)
    batch = jnp.stack((dense, dense))

    assert destination.numout == destination.numin == 2
    assert jnp.array_equal(run(dense), expected)
    assert jnp.array_equal(jax.vmap(run)(batch), jnp.stack((expected, expected)))
    assert jnp.array_equal(
        jax.grad(lambda value: jnp.sum(run(value) ** 2))(dense),
        jnp.zeros_like(dense),
    )


def test_apply_trivial_index_transform_jaxpr_is_transpose_and_reshape_only():
    target = hom(
        (ComplexSpace(2), ComplexSpace(3)),
        (ComplexSpace(4), ComplexSpace(5)),
    )
    permutation = ((1, 3), (0, 2))
    destination = target.permute(*permutation)
    data = jnp.arange(120, dtype=jnp.float32)

    primitives = {
        equation.primitive.name
        for equation in jax.make_jaxpr(
            lambda value: (
                transforms._apply_trivial_index_transform(
                    TensorMap(target, value),
                    destination,
                    *permutation,
                ).storage.data
            ),
        )(data).jaxpr.eqns
    }

    assert primitives == {"reshape", "transpose"}


def _tree_parameters_for(target, permutation):
    destination = target.permute(*permutation)
    return dict(
        source_layout=get_degeneracystructure(target),
        destination_layout=get_degeneracystructure(destination),
        transformer=transforms._treepermuter(target, destination, *permutation),
        permutation=permutation[0] + permutation[1],
    )



def _tree_address_terms(parameters):
    terms = []
    for entry in parameters["transformer"].abelian_data:
        source = parameters["source_layout"].subblockstructure[entry.src]
        destination = parameters["destination_layout"].subblockstructure[entry.dst]
        permutation = parameters["permutation"]
        shape = tuple(source.sizes[axis] for axis in permutation)
        strides = tuple(source.strides[axis] for axis in permutation)
        assert shape == tuple(destination.sizes)
        source_indices = [source.offset + sum(index * stride for index, stride in zip(coordinate, strides))
                          for coordinate in np.ndindex(shape)]
        destination_indices = [destination.offset + sum(index * stride for index, stride in zip(coordinate, destination.strides))
                               for coordinate in np.ndindex(shape)]
        terms.append((source_indices, destination_indices, entry.coeff))
    destinations = [index for _, indices, _ in terms for index in indices]
    assert sorted(destinations) == list(range(parameters["destination_layout"].total_dim))
    return terms



def _tree_functions(parameters):
    terms = _tree_address_terms(parameters)
    def execute(source):
        return _transform._strided_tree_transform(source, result_dtype=source.dtype, **parameters)
    def oracle(source):
        result = jnp.zeros((*source.shape[:-1], parameters["destination_layout"].total_dim), dtype=source.dtype)
        for source_indices, destination_indices, coefficient in terms:
            values = source[..., jnp.asarray(source_indices, dtype=jnp.int32)]
            result = result.at[..., jnp.asarray(destination_indices, dtype=jnp.int32)].set(
                values * jnp.asarray(coefficient, dtype=source.dtype))
        return result
    return execute, oracle



def test_abelian_handoff_reuses_affine_adapter(monkeypatch):
    factor = jnp.float32(2)
    source = jnp.ones(2)
    source_layout = SimpleNamespace(subblockstructure=(object(),))
    destination_layout = SimpleNamespace(subblockstructure=(object(),), total_dim=2)
    transformer = SimpleNamespace(kind="abelian", abelian_data=(SimpleNamespace(src=0, dst=0, coeff=factor),))
    calls = []
    def capture(values, **arguments):
        arguments["entries"] = tuple(arguments["entries"])
        calls.append((values, arguments))
        return values
    monkeypatch.setattr(_transform, "_strided_affine_transform", capture)
    actual = _transform._strided_tree_transform(
        source, source_layout=source_layout, destination_layout=destination_layout,
        transformer=transformer, result_dtype=source.dtype, permutation=(0,),
    )
    assert actual is source
    assert len(calls) == 1
    values, arguments = calls[0]
    assert values is source
    assert arguments["source_subblocks"] is source_layout.subblockstructure
    assert arguments["destination_subblocks"] is destination_layout.subblockstructure
    assert arguments["entries"][0][:2] == (0, 0)
    assert arguments["entries"][0][2] is factor
    assert arguments["permutation"] == (0,)
    assert arguments["output_size"] == 2
    assert arguments["result_dtype"] == source.dtype
    assert arguments["shape_error"] == "Abelian tree transform subblock shapes are inconsistent"



@pytest.mark.parametrize("sector", [U1Irrep, FermionParity])
@pytest.mark.parametrize(
    "batch_shape,dtype",
    [
        pytest.param(batch_shape, dtype, id=f"batch_shape{index}-{dtype}")
        for index, batch_shape in enumerate([(), (2,), (2, 3), (0,), (2, 0)])
        for dtype in ["float16", "bfloat16", "float32", "float64", "complex64", "complex128"]
        if batch_shape == () or dtype in {"float32", "complex64"}
    ],
)
def test_abelian_tree_metadata_numerics_and_source_ad(sector, dtype, batch_shape):
    with jax.enable_x64():
        factor = space(sector, {0: 2, 1: 1})
        parameters = _tree_parameters_for(hom((factor, factor), (factor,)), ((1,), (0, 2)))
        assert parameters["transformer"].kind == "abelian"
        execute, oracle = _tree_functions(parameters)
        source_size = parameters["source_layout"].total_dim
        source = (jnp.arange(math.prod(batch_shape) * source_size) % 5).astype(dtype).reshape((*batch_shape, source_size))
        if dtype.startswith("complex"):
            source = source + 1j * (source - 2)
        tangent = jnp.ones_like(source)
        for actual, expected in zip(jax.jit(lambda values: jax.jvp(execute, (values,), (tangent,)))(source),
                                    jax.jvp(oracle, (source,), (tangent,)), strict=True):
            np.testing.assert_array_equal(actual, expected)
        cotangent = jnp.ones((*batch_shape, parameters["destination_layout"].total_dim), dtype=source.dtype)
        expected = jax.vjp(oracle, source)[1](cotangent)[0]
        np.testing.assert_array_equal(jax.jit(jax.vjp(execute, source)[1])(cotangent)[0], expected)
        np.testing.assert_array_equal(jax.linear_transpose(execute, source)(cotangent)[0], expected)



def test_generic_tree_uses_grouped_adapter():
    half = space(SU2Irrep, {1: 1})
    target = hom((half, half, half, half), ())
    permutation = ((1, 2, 3), (0,))
    parameters = _tree_parameters_for(target, permutation)
    assert parameters["transformer"].kind == "generic"
    tensor = TensorMap(target, jnp.asarray([1, 2], dtype=jnp.float32))
    result = permute(tensor, permutation)
    expected = _apply_generic_reference(
        jnp.zeros(parameters["destination_layout"].total_dim), tensor.storage.data,
        parameters["permutation"], parameters["transformer"].generic_data,
        parameters["source_layout"].subblockstructure, parameters["destination_layout"].subblockstructure,
    )
    np.testing.assert_allclose(result.storage.data, expected, rtol=1e-6, atol=1e-6)



def test_unknown_kind_does_not_touch_layouts():
    with pytest.raises(ValueError, match="unsupported tree transformer kind"):
        _transform._strided_tree_transform(
            None, source_layout=None, destination_layout=None, result_dtype=jnp.float32,
            permutation=(), transformer=SimpleNamespace(kind="unknown"),
        )



def test_empty_abelian_transform():
    layout = SimpleNamespace(subblockstructure=(), total_dim=0)
    transformer = SimpleNamespace(kind="abelian", abelian_data=())
    run = lambda source: _transform._strided_tree_transform(
        source, source_layout=layout, destination_layout=layout, result_dtype=source.dtype,
        permutation=(), transformer=transformer,
    )
    source = jnp.zeros((2, 0))
    result = jax.jit(run)(source)
    assert result.shape == (2, 0)
    np.testing.assert_array_equal(jax.vjp(run, source)[1](result)[0], source)
