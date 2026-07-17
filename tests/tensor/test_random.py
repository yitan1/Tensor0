import jax
import jax.numpy as jnp
import pytest

import tensor0.tensor.constructors as constructors_module
from tensor0 import (
    FermionParity,
    SU2Irrep,
    TensorMap,
    U1Irrep,
    U1SU2Irrep,
    fuse,
    hom,
    space,
    storage_dim,
    zero_space,
)
import tensor0.tensor as tensor_api
from tensor0.tensor.constructors import identity, random_isometry, random_normal
from tests.cases import assert_allclose, float_data_for


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


def _random_isometry_space_cases():
    return (
        (
            "u1",
            space(U1Irrep, {0: 3, 1: 2}),
            space(U1Irrep, {0: 2, 1: 1}),
        ),
        (
            "su2",
            space(SU2Irrep, {0: 3, 2: 2}),
            space(SU2Irrep, {0: 2, 2: 1}),
        ),
        (
            "fermionic",
            space(FermionParity, {0: 3, 1: 2}),
            space(FermionParity, {0: 2, 1: 1}),
        ),
        (
            "product-sector",
            space(U1SU2Irrep, {(0, 0): 3, (1, 1): 2}),
            space(U1SU2Irrep, {(0, 0): 2, (1, 1): 1}),
        ),
    )


@pytest.mark.parametrize(
    ("_name", "codomain", "domain"),
    _random_isometry_space_cases(),
    ids=[name for name, _codomain, _domain in _random_isometry_space_cases()],
)
def test_random_isometry_preserves_homspace_and_is_isometric(
    _name,
    codomain,
    domain,
):
    result = random_isometry(
        jax.random.key(61),
        codomain,
        domain,
        dtype=jnp.complex64,
    )

    assert result.space == hom((codomain,), (domain,))
    assert result.dtype == jnp.dtype(jnp.complex64)
    assert bool(
        tensor_api.allclose(
            result.adjoint() @ result,
            identity(domain, dtype=jnp.complex64),
            rtol=1e-5,
            atol=1e-6,
        ),
    )


def test_random_isometry_preserves_product_and_fused_partitions():
    half = space(SU2Irrep, {1: 1})
    product_domain = hom((half, half), ()).codomain
    fused_codomain = space(SU2Irrep, {0: 2, 2: 2})

    product_to_fused = random_isometry(
        jax.random.key(62),
        fused_codomain,
        product_domain,
    )
    assert product_to_fused.space == hom((fused_codomain,), product_domain)
    assert product_to_fused.space.domain == product_domain

    factor = space(U1Irrep, {0: 2})
    product_codomain = hom((factor, factor), ()).codomain
    fused_domain = fuse(hom((factor,), ()).codomain)
    fused_to_product = random_isometry(
        jax.random.key(63),
        product_codomain,
        fused_domain,
    )
    assert fused_to_product.space == hom(product_codomain, (fused_domain,))
    assert fused_to_product.space.codomain == product_codomain

    for result, domain in (
        (product_to_fused, product_domain),
        (fused_to_product, fused_domain),
    ):
        assert bool(
            tensor_api.allclose(
                result.adjoint() @ result,
                identity(domain),
                rtol=1e-5,
                atol=1e-6,
            ),
        )


def test_random_isometry_uses_positive_qr_phase_convention():
    codomain = space(U1Irrep, {0: 4, 1: 3})
    domain = space(U1Irrep, {0: 3, 1: 2})
    target = hom((codomain,), (domain,))
    key = jax.random.key(64)
    sample = random_normal(key, target, dtype=jnp.complex64)
    result = random_isometry(key, codomain, domain, dtype=jnp.complex64)

    remainder = result.adjoint() @ sample
    for _coupled, block in remainder.blocks():
        diagonal = jnp.diag(block)
        assert_allclose(jnp.tril(block, -1), jnp.zeros_like(block))
        assert_allclose(jnp.imag(diagonal), jnp.zeros_like(jnp.imag(diagonal)))
        assert bool(jnp.all(jnp.real(diagonal) >= 0))


@pytest.mark.parametrize("dtype", [None, jnp.float32, jnp.complex64])
def test_random_isometry_is_reproducible_and_preserves_dtype(dtype):
    codomain = space(U1Irrep, {0: 3})
    domain = space(U1Irrep, {0: 2})
    key = jax.random.key(65)
    original_key_data = jax.random.key_data(key)

    first = random_isometry(key, codomain, domain, dtype=dtype)
    second = random_isometry(key, codomain, domain, dtype=dtype)
    distinct = random_isometry(jax.random.key(66), codomain, domain, dtype=dtype)

    expected_dtype = jax.random.normal(key, (), dtype=dtype).dtype
    assert first.dtype == expected_dtype
    assert bool(jnp.array_equal(first.storage.data, second.storage.data))
    assert not bool(jnp.array_equal(first.storage.data, distinct.storage.data))
    assert bool(jnp.array_equal(jax.random.key_data(key), original_key_data))


def test_random_isometry_supports_scalar_and_zero_spaces():
    scalar = hom((), (), sector_type=U1Irrep).codomain
    zero = zero_space(U1Irrep)
    key = jax.random.key(67)

    scalar_result = random_isometry(key, scalar, scalar, dtype=jnp.complex64)
    zero_result = random_isometry(key, zero, zero, dtype=jnp.float32)

    assert scalar_result.space == hom(scalar, scalar)
    assert_allclose(jnp.abs(scalar_result.scalar()), jnp.asarray(1, dtype=jnp.float32))
    assert zero_result.space == hom((zero,), (zero,))
    assert zero_result.storage.data.shape == (0,)
    assert zero_result.dtype == jnp.dtype(jnp.float32)


def test_random_isometry_validates_inputs_before_sampling(monkeypatch):
    codomain = space(U1Irrep, {0: 2})
    oversized_domain = space(U1Irrep, {0: 3})

    def explode(*_args, **_kwargs):
        raise AssertionError("random sampling must follow static validation")

    monkeypatch.setattr(constructors_module.random, "normal", explode)

    with pytest.raises(
        TypeError,
        match=r"random_isometry\(\).*codomain.*ElementarySpace or ProductSpace",
    ):
        random_isometry(
            jax.random.key(0),
            object(),  # pyright: ignore[reportArgumentType]
            oversized_domain,
        )
    with pytest.raises(ValueError, match="domain to be monomorphic"):
        random_isometry(jax.random.key(0), codomain, oversized_domain)
    with pytest.raises(ValueError, match="float or complex dtype"):
        random_isometry(
            jax.random.key(0),
            codomain,
            codomain,
            dtype=jnp.int32,
        )


def test_random_isometry_is_jittable_and_supports_downstream_gradients():
    codomain = space(U1Irrep, {0: 3, 1: 2})
    domain = space(U1Irrep, {0: 2, 1: 1})

    @jax.jit
    def build(key):
        return random_isometry(key, codomain, domain, dtype=jnp.float32)

    key = jax.random.key(68)
    result = build(key)
    eager = random_isometry(key, codomain, domain, dtype=jnp.float32)
    assert bool(jnp.array_equal(result.storage.data, eager.storage.data))

    operator_space = hom((domain,), (domain,))
    data = float_data_for(operator_space)

    def objective(storage):
        operator = TensorMap(operator_space, storage)
        transformed = result @ operator
        return jnp.sum(transformed.storage.data**2)

    assert_allclose(jax.grad(objective)(data), 2 * data)
