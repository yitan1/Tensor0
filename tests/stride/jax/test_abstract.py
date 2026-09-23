"""JAX abstract evaluation and result metadata."""

import jax
import jax.numpy as jnp

from tensor0._stride._jax import update_p


def test_update_abstract_result_preserves_base_shape_and_dtype():
    base = jax.ShapeDtypeStruct((2, 4), jnp.float32)
    source = jax.ShapeDtypeStruct((2, 6), jnp.float16)
    result = jax.eval_shape(lambda new, old: update_p.bind(
        new, old, jnp.float32(1), jnp.float32(0), records=()), source, base)
    assert result.shape == base.shape and result.dtype == base.dtype
