"""Verify installed public CPU operations; run with python -I outside the checkout."""

from pathlib import Path
import sys
import jax
import jax.numpy as jnp
import numpy as np
import tensor0
from tensor0 import TensorMap, U1Irrep, Vect, from_dense, hom, space, to_dense
import tensor0._native as native

assert Path(tensor0.__file__).resolve().is_relative_to(Path(sys.prefix))
assert Path(native.__file__).resolve().is_relative_to(Path(sys.prefix))
assert jax.default_backend() == 'cpu'
target = hom((Vect(2),), (Vect(2),))
data = jnp.arange(1, 5, dtype=jnp.float32).reshape(2, 2)
tensor = from_dense(target, data)
np.testing.assert_allclose(to_dense(tensor), data)
np.testing.assert_allclose(to_dense(tensor @ tensor), data @ data)
np.testing.assert_allclose(to_dense(tensor.permute(((1,), (0,)))), data.T)
composed = jax.jit(lambda a: a @ a)(tensor)
np.testing.assert_allclose(to_dense(composed), data @ data)

def loss(array):
    candidate = from_dense(target, array)
    transformed = candidate.permute(((1,), (0,)))
    return jnp.sum(to_dense(transformed) ** 2)

value, gradient = jax.jit(jax.value_and_grad(loss))(data)
np.testing.assert_allclose(value, jnp.sum(data ** 2))
np.testing.assert_allclose(gradient, 2 * data)
factor = space(U1Irrep, {0: 2, 1: 1})
symmetric_space = hom((factor,), (factor,))
symmetric_data = jnp.arange(1, 6, dtype=jnp.float32)
symmetric = TensorMap(symmetric_space, symmetric_data)
symmetric_composed = jax.jit(lambda a: a @ a)(symmetric)
for sector, block in symmetric.blocks():
    np.testing.assert_allclose(symmetric_composed.block(sector), block @ block)

def symmetric_loss(array):
    candidate = TensorMap(symmetric_space, array)
    return sum(jnp.sum(block ** 2) for _, block in candidate.blocks())

symmetric_value, symmetric_gradient = jax.jit(jax.value_and_grad(symmetric_loss))(symmetric_data)
np.testing.assert_allclose(symmetric_value, jnp.sum(symmetric_data ** 2))
np.testing.assert_allclose(symmetric_gradient, 2 * symmetric_data)
print('Installed wheel Trivial/U1 CPU ops, jit and grad OK:', tensor0.__file__)
