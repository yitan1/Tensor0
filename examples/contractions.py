from __future__ import annotations

import jax
import jax.numpy as jnp

from tensor0 import (
    FermionParity,
    TensorMap,
    U1Irrep,
    contract,
    hom,
    idx,
    ncon,
    scalar,
    space,
    tensorcontract,
    tensortrace,
    twist,
)
from tensor0.structure import get_degeneracystructure


def _data_for(hom_space, *, scale=0.1):
    total_dim = get_degeneracystructure(hom_space).total_dim
    return jnp.arange(1, total_dim + 1, dtype=jnp.float32) * scale


def _tensor(hom_space, *, scale=0.1):
    return TensorMap(hom_space, _data_for(hom_space, scale=scale))


def _assert_allclose(actual, expected):
    assert actual.shape == expected.shape
    assert bool(jnp.allclose(actual, expected, rtol=1e-5, atol=1e-6))


def primitive_examples():
    a = space(U1Irrep, {0: 2})
    x = space(U1Irrep, {0: 3})
    b = space(U1Irrep, {0: 4})
    left = _tensor(hom((a,), (x,)))
    right = _tensor(hom((x,), (b,)))

    contracted = tensorcontract(
        left,
        right,
        axes=((1,), (0,)),
        output=(((0, 0),), ((1, 1),)),
    )
    composed = left @ right
    assert contracted.space == composed.space
    _assert_allclose(contracted.storage.data, composed.storage.data)

    source = _tensor(hom((a, x), (a, x)))
    partial = tensortrace(
        source,
        axes=((1,), (3,)),
        output=((0,), (2,)),
    )
    rank_zero = tensortrace(
        partial,
        axes=((0,), (1,)),
        output=((), ()),
    )
    assert rank_zero.numind == 0
    assert scalar(rank_zero).shape == ()


def network_examples():
    a = space(U1Irrep, {0: 2})
    x = space(U1Irrep, {0: 3})
    y = space(U1Irrep, {0: 4})
    b = space(U1Irrep, {0: 2})
    left = _tensor(hom((a,), (x,)))
    middle = _tensor(hom((x,), (y,)))
    right = _tensor(hom((y,), (b,)))

    named = contract(
        idx(left, "a,x"),
        idx(middle, "x,y"),
        idx(right, "y,b"),
        output=("a", "b"),
        order=("y", "x"),
    )
    integer = ncon(
        (left, middle, right),
        ((-1, 1), (1, 2), (2, -2)),
        order=(2, 1),
        output=((-1,), (-2,)),
    )
    assert named.space == integer.space
    _assert_allclose(named.storage.data, integer.storage.data)
    return left, middle, right, named


def fermionic_twist_example():
    parity = space(FermionParity, {0: 1, 1: 1})
    target = hom((parity,), (parity,))
    tensor = TensorMap(target, jnp.array([1.0, 2.0], dtype=jnp.float32))
    twisted = twist(tensor, 0)

    _assert_allclose(
        twisted.storage.data,
        jnp.array([1.0, -2.0], dtype=jnp.float32),
    )


def jax_example(left, middle, right, eager):
    @jax.jit
    def network(a, b, c):
        return contract(
            idx(a, "a,x"),
            idx(b, "x,y"),
            idx(c, "y,b"),
            output=("a", "b"),
            order=("y", "x"),
        )

    compiled = network(left, middle, right)
    _assert_allclose(compiled.storage.data, eager.storage.data)

    def loss(data):
        candidate = TensorMap(left.space, data)
        result = network(candidate, middle, right)
        return jnp.sum(result.storage.data**2)

    value, gradient = jax.value_and_grad(loss)(left.storage.data)
    assert value.shape == ()
    assert gradient.shape == left.storage.data.shape


def main():
    primitive_examples()
    left, middle, right, result = network_examples()
    fermionic_twist_example()
    jax_example(left, middle, right, result)
    print("Tensor0 contraction examples OK")


if __name__ == "__main__":
    main()
