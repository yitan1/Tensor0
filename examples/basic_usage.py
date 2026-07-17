from __future__ import annotations

import jax
import jax.numpy as jnp

from tensor0 import (
    FermionParity,
    SU2Irrep,
    TensorMap,
    U1Irrep,
    eigh_full,
    from_blocks,
    from_dense,
    hom,
    identity,
    isometry,
    is_hermitian,
    qr_compact,
    random_normal,
    space,
    svd_compact,
    tensor_product,
    to_dense,
)
from tensor0.structure import get_degeneracystructure


def _data_for(hom_space, *, scale=0.1):
    total_dim = get_degeneracystructure(hom_space).total_dim
    return jnp.arange(1, total_dim + 1, dtype=jnp.float32) * scale


def _assert_allclose(actual, expected):
    assert actual.shape == expected.shape
    assert bool(jnp.allclose(actual, expected, rtol=1e-5, atol=1e-6))


def layout_and_storage_example():
    v = space(U1Irrep, {0: 2, 1: 3})
    h = hom((v,), (v,))
    tensor = TensorMap(h, _data_for(h))

    assert tensor.block(0).shape == (2, 2)
    assert tensor.block((1,)).shape == (3, 3)
    assert [sector for sector, _block in tensor.blocks()] == [(0,), (1,)]
    return tensor


def dense_roundtrip_example(tensor):
    dense = to_dense(tensor)
    rebuilt = from_dense(tensor.space, dense)

    assert rebuilt.space == tensor.space
    _assert_allclose(rebuilt.storage.data, tensor.storage.data)


def composition_and_svd_example():
    v = space(U1Irrep, {0: 2, 1: 3})
    w = space(U1Irrep, {0: 5, 1: 7})
    x = space(U1Irrep, {0: 11, 1: 13})
    a_space = hom((v,), (w,))
    b_space = hom((w,), (x,))
    a = TensorMap(a_space, _data_for(a_space))
    b = TensorMap(b_space, _data_for(b_space))

    composed = a @ b
    assert composed.space == hom((v,), (x,))
    for coupled, block in composed.blocks():
        _assert_allclose(block, a.block(coupled) @ b.block(coupled))

    left = space(U1Irrep, {0: 2, 1: 4})
    right = space(U1Irrep, {0: 3, 1: 2})
    svd_space = hom((left,), (right,))
    tensor = TensorMap(svd_space, _data_for(svd_space))
    u, s, vh = svd_compact(tensor)
    reconstructed = u @ s @ vh

    assert reconstructed.space == tensor.space
    for coupled, block in tensor.blocks():
        _assert_allclose(reconstructed.block(coupled), block)


def transform_example():
    v = space(U1Irrep, {0: 2, 1: 1})
    w = space(U1Irrep, {0: 1, 1: 2})
    x = space(U1Irrep, {1: 1})
    u1_space = hom((v, w), (x,))
    u1_tensor = TensorMap(u1_space, _data_for(u1_space, scale=1.0))
    u1_permuted = u1_tensor.permute(((1,), (0, 2)))

    assert u1_permuted.space == hom((w,), (v.dual(), x))
    _assert_allclose(
        to_dense(u1_permuted),
        jnp.transpose(to_dense(u1_tensor), (1, 0, 2)),
    )

    odd_a = space(FermionParity, {1: 1})
    odd_b = space(FermionParity, {1: 1})
    fermion_space = hom((odd_a, odd_b), ())
    fermion_tensor = TensorMap(fermion_space, jnp.array([2.0], dtype=jnp.float32))
    fermion_permuted = fermion_tensor.permute(((1, 0), ()))

    _assert_allclose(
        to_dense(fermion_permuted),
        -jnp.transpose(to_dense(fermion_tensor), (1, 0)),
    )

    half = space(SU2Irrep, {1: 1})
    su2_space = hom((half, half, half), (half,))
    su2_tensor = TensorMap(su2_space, _data_for(su2_space, scale=1.0))
    su2_permuted = su2_tensor.permute(((1, 2), (0, 3)))

    _assert_allclose(
        to_dense(su2_permuted),
        jnp.transpose(to_dense(su2_tensor), (1, 2, 0, 3)),
    )


def canonical_and_random_example():
    small = space(U1Irrep, {0: 2, 1: 1})
    large = space(U1Irrep, {0: 3, 1: 2})
    target = hom((small,), (small,))
    sample = random_normal(jax.random.key(0), target, dtype=jnp.float32)
    unit = identity(small, dtype=jnp.float32)

    _assert_allclose((unit @ sample).storage.data, sample.storage.data)

    embedding = isometry(large, small, dtype=jnp.float32)
    _assert_allclose(
        (embedding.adjoint() @ embedding).storage.data,
        unit.storage.data,
    )

    product = tensor_product(unit, unit)
    assert product.codomain.spaces == (small, small)
    assert product.domain.spaces == (small, small)


def blockwise_factorization_example():
    factor = space(U1Irrep, {0: 2})
    target = hom((factor,), (factor,))
    tensor = from_blocks(
        target,
        {
            0: jnp.asarray(
                [[2.0, 1.0], [1.0, 3.0]],
                dtype=jnp.float32,
            ),
        },
    )

    q, r = qr_compact(tensor)
    assert q.domain == r.codomain
    _assert_allclose((q @ r).storage.data, tensor.storage.data)

    assert bool(is_hermitian(tensor))
    eigenvalues, eigenvectors = eigh_full(tensor)
    reconstructed = eigenvectors @ eigenvalues @ eigenvectors.adjoint()
    _assert_allclose(reconstructed.storage.data, tensor.storage.data)


def jax_example():
    v = space(U1Irrep, {0: 2, 1: 3})
    w = space(U1Irrep, {0: 5, 1: 7})
    x = space(U1Irrep, {0: 11, 1: 13})
    a_space = hom((v,), (w,))
    b_space = hom((w,), (x,))
    a = TensorMap(a_space, _data_for(a_space))
    b = TensorMap(b_space, _data_for(b_space))

    @jax.jit
    def compose(left, right):
        return left @ right

    jitted = compose(a, b)
    eager = a @ b
    _assert_allclose(jitted.storage.data, eager.storage.data)

    def loss(data):
        candidate = TensorMap(a.space, data)
        return jnp.sum((candidate @ b).storage.data ** 2)

    value, gradient = jax.value_and_grad(loss)(a.storage.data)

    assert value.shape == ()
    assert gradient.shape == a.storage.data.shape


def main():
    tensor = layout_and_storage_example()
    dense_roundtrip_example(tensor)
    composition_and_svd_example()
    transform_example()
    canonical_and_random_example()
    blockwise_factorization_example()
    jax_example()
    print("Tensor0 basic usage OK")


if __name__ == "__main__":
    main()
