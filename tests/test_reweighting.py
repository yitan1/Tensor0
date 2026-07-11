import math

import jax.numpy as jnp
import pytest
import tensor0.transforms as transforms

from tensor0 import (
    FermionParity,
    TensorMap,
    U1Irrep,
    get_degeneracystructure,
    get_sectorstructure,
    hom,
    space,
    twist,
)
from tensor0 import _native


def _data_for(target):
    size = get_degeneracystructure(target).total_dim
    return jnp.arange(1, size + 1, dtype=jnp.float32)


def _row_major_strides(sizes):
    strides = []
    stride = 1
    for size in reversed(sizes):
        strides.append(stride)
        stride *= size
    return tuple(reversed(strides))


class _ExplodingVectorData:
    def __init__(self, length):
        self.shape = (length,)

    def __getitem__(self, key):
        if isinstance(key, slice) and key.start == 0 and key.stop == 0:
            return jnp.zeros((0,))
        raise AssertionError("twist validation must not access storage")


def test_native_twist_subblock_factors_are_immutable_and_follow_subblock_order():
    factor = space(FermionParity, {0: 1, 1: 1})
    target = hom((factor,), (factor,))

    factors = _native.twist_subblock_factors(
        target,
        get_sectorstructure(target),
        (0,),
        False,
    )

    assert isinstance(factors, tuple)
    assert factors == (1.0, -1.0)


def test_twist_passes_the_cached_sectorstructure_to_native(monkeypatch):
    factor = space(FermionParity, {0: 1, 1: 1})
    target = hom((factor,), (factor,))
    tensor = TensorMap(target, _data_for(target))
    expected = get_sectorstructure(target)
    native_twist_subblock_factors = _native.twist_subblock_factors

    def recording_twist_subblock_factors(space, sectorstructure, indices, inv):
        assert sectorstructure is expected
        return native_twist_subblock_factors(
            space,
            sectorstructure,
            indices,
            inv,
        )

    monkeypatch.setattr(
        _native,
        "twist_subblock_factors",
        recording_twist_subblock_factors,
    )

    result = twist(tensor, 0)

    assert bool(
        jnp.array_equal(
            result.storage.data,
            jnp.array([1.0, -2.0], dtype=jnp.float32),
        )
    )


def test_native_twist_classification_and_layout_cancellation_are_distinct():
    parity_factor = space(FermionParity, {0: 1, 1: 1})
    parity_target = hom((parity_factor,), (parity_factor,))
    assert not _native.twist_is_trivial(parity_target, (0, 1))
    assert (
        _native.twist_subblock_factors(
            parity_target,
            get_sectorstructure(parity_target),
            (0, 1),
            False,
        )
        is None
    )

    u1_factor = space(U1Irrep, {0: 1, 1: 1})
    u1_target = hom((u1_factor,), (u1_factor,))
    assert _native.twist_is_trivial(u1_target, (0,))


def test_twist_layout_cancellation_bypasses_degeneracy_lookup(monkeypatch):
    factor = space(FermionParity, {0: 1, 1: 1})
    target = hom((factor,), (factor,))
    tensor = TensorMap(target, _data_for(target))

    def explode(*args, **kwargs):
        raise AssertionError("cancelled twist must not request degeneracy layout")

    monkeypatch.setattr(transforms, "get_degeneracystructure", explode)

    assert twist(tensor, (0, 1)) is tensor


@pytest.mark.parametrize(
    ("indices", "match"),
    [
        ((2,), "range"),
        ((0, 0), "unique"),
    ],
)
def test_native_twist_is_trivial_validates_before_bosonic_identity(indices, match):
    factor = space(U1Irrep, {0: 1})
    target = hom((factor,), (factor,))

    with pytest.raises(ValueError, match=match):
        _native.twist_is_trivial(target, indices)


def test_twist_reweights_parity_endomorphism_without_changing_space_or_dtype():
    factor = space(FermionParity, {0: 1, 1: 1})
    target = hom((factor,), (factor,))
    tensor = TensorMap(target, _data_for(target))

    codomain = twist(tensor, 0)
    domain = twist(tensor, (1,))
    both = twist(tensor, (0, 1))

    expected = jnp.array([1.0, -2.0], dtype=jnp.float32)
    assert bool(jnp.array_equal(codomain.storage.data, expected))
    assert bool(jnp.array_equal(domain.storage.data, expected))
    assert codomain.space is target
    assert domain.space is target
    assert codomain.storage.data is not tensor.storage.data
    assert domain.storage.data is not tensor.storage.data
    assert codomain.storage.data.dtype == tensor.storage.data.dtype
    assert domain.storage.data.dtype == tensor.storage.data.dtype
    assert both is tensor


def test_twist_scales_only_nonidentity_subblocks(monkeypatch):
    factor = space(FermionParity, {0: 1, 1: 1})
    target = hom((factor,), (factor,))
    tensor = TensorMap(target, _data_for(target))
    original = tensor.storage.data.copy()
    scale_subblock = transforms._scale_subblock
    recorded_factors = []

    def recording_scale_subblock(data, subblock, coefficient):
        recorded_factors.append(float(coefficient))
        return scale_subblock(data, subblock, coefficient)

    monkeypatch.setattr(transforms, "_scale_subblock", recording_scale_subblock)

    result = twist(tensor, 0)

    assert recorded_factors == [-1.0]
    assert bool(
        jnp.array_equal(
            result.storage.data,
            jnp.array([1.0, -2.0], dtype=jnp.float32),
        )
    )
    assert bool(jnp.array_equal(tensor.storage.data, original))
    assert result.space is target
    assert result.storage.data.dtype == tensor.storage.data.dtype


def test_twist_reweights_multielement_strided_subblocks_in_canonical_order():
    factor = space(FermionParity, {0: 2, 1: 3})
    target = hom((factor, factor), (factor, factor))
    tensor = TensorMap(target, _data_for(target))
    subblockstructure = get_degeneracystructure(target).subblockstructure

    assert all(math.prod(subblock.sizes) > 1 for subblock in subblockstructure)
    assert any(
        tuple(subblock.strides) != _row_major_strides(tuple(subblock.sizes))
        for subblock in subblockstructure
    )

    factors = _native.twist_subblock_factors(
        target,
        get_sectorstructure(target),
        (0,),
        False,
    )
    assert factors is not None
    result = twist(tensor, 0)

    for factor_value, source, destination in zip(
        factors,
        tensor.subblocks(),
        result.subblocks(),
        strict=True,
    ):
        source_pair, source_data = source
        destination_pair, destination_data = destination
        assert tuple(tree.static_key for tree in destination_pair) == tuple(
            tree.static_key for tree in source_pair
        )
        expected = jnp.asarray(factor_value, dtype=source_data.dtype) * source_data
        assert bool(jnp.array_equal(destination_data, expected))

    restored = twist(result, 0)
    assert bool(jnp.array_equal(restored.storage.data, tensor.storage.data))


def test_twist_inverse_supports_product_sector_types():
    product = space(U1Irrep @ FermionParity, {(0, 0): 1, (0, 1): 1})
    target = hom((product,), (product,))
    tensor = TensorMap(target, _data_for(target))

    result = twist(tensor, 0, inv=True)

    expected = jnp.array([1.0, -2.0], dtype=jnp.float32)
    assert bool(jnp.array_equal(result.storage.data, expected))
    assert result.space is target


def test_twist_noop_returns_identical_tensor_without_accessing_storage():
    factor = space(U1Irrep, {0: 1})
    target = hom((factor,), (factor,))
    tensor = TensorMap(
        target,
        _ExplodingVectorData(get_degeneracystructure(target).total_dim),
    )

    assert twist(tensor, ()) is tensor
    assert twist(tensor, 0) is tensor


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
    tensor = TensorMap(target, _data_for(target))

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

    monkeypatch.setattr(_native, "twist_is_trivial", explode)
    monkeypatch.setattr(_native, "twist_subblock_factors", explode)
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
