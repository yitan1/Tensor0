from __future__ import annotations

from jax import Array
import jax.numpy as jnp


def positive_qr(block: Array) -> tuple[Array, Array]:
    """Return reduced QR factors with positive-real nonzero diagonal."""
    q_block, r_block = jnp.linalg.qr(block, mode="reduced")
    diagonal = jnp.diag(r_block)
    magnitude = jnp.abs(diagonal)
    nonzero = magnitude != 0
    safe_magnitude = jnp.where(nonzero, magnitude, jnp.ones_like(magnitude))
    phase = jnp.where(
        nonzero,
        diagonal / safe_magnitude,
        jnp.ones_like(diagonal),
    )
    return q_block * phase[None, :], jnp.conj(phase)[:, None] * r_block
