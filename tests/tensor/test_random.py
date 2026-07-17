import jax
import jax.numpy as jnp
import pytest

import tensor0.tensor.constructors as constructors_module
from tensor0 import (
    FermionParity,
    SU2Irrep,
    TensorMap,
    U1Irrep,
    hom,
    space,
    storage_dim,
)
from tensor0.tensor.constructors import random_normal


def _random_space_cases():
    u1 = space(U1Irrep, {0: 2, 1: 1})
    su2 = space(SU2Irrep, {0: 2, 2: 1})
    fermionic = space(FermionParity, {0: 2, 1: 1})
    product = space(
        U1Irrep @ FermionParity,
        {(0, 0): 2, (1, 1): 1},
    )
    return (
        ("u1", hom((u1,), (u1,))),
        ("su2", hom((su2,), (su2,))),
        ("fermionic", hom((fermionic,), (fermionic,))),
        ("product-sector", hom((product,), (product,))),
    )


@pytest.mark.parametrize(
    ("_name", "target"),
    _random_space_cases(),
    ids=[name for name, _target in _random_space_cases()],
)
def test_random_normal_samples_packed_storage_for_sector_families(_name, target):
    key = jax.random.key(17)

    result = random_normal(key, target, dtype=jnp.float32)
    expected = jax.random.normal(
        key,
        shape=(storage_dim(target),),
        dtype=jnp.float32,
    )

    assert isinstance(result, TensorMap)
    assert result.space is target
    assert result.storage.data.shape == (storage_dim(target),)
    assert result.dtype == jnp.dtype(jnp.float32)
    assert bool(jnp.array_equal(result.storage.data, expected))


def test_random_normal_is_reproducible_and_does_not_replace_or_split_key():
    target = _random_space_cases()[0][1]
    key = jax.random.key(23)
    original_key_data = jax.random.key_data(key)

    first = random_normal(key, target, dtype=jnp.float32)
    second = random_normal(key, target, dtype=jnp.float32)
    distinct = random_normal(jax.random.key(24), target, dtype=jnp.float32)

    assert bool(jnp.array_equal(first.storage.data, second.storage.data))
    assert not bool(jnp.array_equal(first.storage.data, distinct.storage.data))
    assert bool(jnp.array_equal(jax.random.key_data(key), original_key_data))


def test_random_normal_default_and_explicit_dtypes_follow_jax():
    target = _random_space_cases()[0][1]
    key = jax.random.key(31)

    default = random_normal(key, target)
    explicit_real = random_normal(key, target, dtype=jnp.float16)
    explicit_complex = random_normal(key, target, dtype=jnp.complex64)

    expected_default = jax.random.normal(key, (storage_dim(target),))
    expected_real = jax.random.normal(
        key,
        (storage_dim(target),),
        dtype=jnp.float16,
    )
    expected_complex = jax.random.normal(
        key,
        (storage_dim(target),),
        dtype=jnp.complex64,
    )

    for result, expected in (
        (default, expected_default),
        (explicit_real, expected_real),
        (explicit_complex, expected_complex),
    ):
        assert result.dtype == expected.dtype
        assert bool(jnp.array_equal(result.storage.data, expected))

    assert jnp.issubdtype(default.dtype, jnp.floating)
    assert explicit_complex.dtype == jnp.dtype(jnp.complex64)


def test_random_normal_empty_and_rank_zero_spaces():
    empty_factor = space(U1Irrep, {})
    empty_target = hom((empty_factor,), (empty_factor,))
    scalar_target = hom((), (), sector_type=U1Irrep)
    key = jax.random.key(41)

    empty = random_normal(key, empty_target, dtype=jnp.complex64)
    scalar = random_normal(key, scalar_target, dtype=jnp.float32)

    assert empty.space is empty_target
    assert empty.storage.data.shape == (0,)
    assert empty.dtype == jnp.dtype(jnp.complex64)
    assert scalar.space is scalar_target
    assert scalar.storage.data.shape == (1,)
    assert bool(
        jnp.array_equal(
            scalar.storage.data,
            jax.random.normal(key, (1,), dtype=jnp.float32),
        ),
    )


def test_random_normal_validates_space_before_sampling(monkeypatch):
    def explode(*_args, **_kwargs):
        raise AssertionError("random sampling must not run for an invalid space")

    monkeypatch.setattr(constructors_module.random, "normal", explode)

    with pytest.raises(TypeError, match=r"random_normal\(\) requires a HomSpace"):
        random_normal(jax.random.key(0), object())  # pyright: ignore[reportArgumentType]


def test_random_normal_rejects_non_floating_dtype():
    target = _random_space_cases()[0][1]

    with pytest.raises(ValueError, match="float or complex dtype"):
        random_normal(jax.random.key(0), target, dtype=jnp.int32)


def test_random_normal_jit_captures_static_space_and_reuses_key_signature():
    target = _random_space_cases()[2][1]
    trace_count = 0

    @jax.jit
    def build(key):
        nonlocal trace_count
        trace_count += 1
        return random_normal(key, target, dtype=jnp.complex64)

    first_key = jax.random.key(51)
    second_key = jax.random.key(52)
    first = build(first_key)
    second = build(second_key)

    assert first.space is target
    assert second.space is target
    assert trace_count == 1
    assert bool(
        jnp.array_equal(
            first.storage.data,
            jax.random.normal(
                first_key,
                (storage_dim(target),),
                dtype=jnp.complex64,
            ),
        ),
    )
    assert not bool(jnp.array_equal(first.storage.data, second.storage.data))
