from __future__ import annotations

import jax.numpy as jnp

from ..structure.layout import _blockstructure_items
from .storage import _require_jax_storage_data
from .tensor_map import TensorMap


def _jax_cotangent_to_riesz_gradient(cotangent: TensorMap) -> TensorMap:
    """Convert a JAX coordinate cotangent to a TensorMap Riesz gradient."""

    data = _require_jax_storage_data(
        cotangent.storage.data,
        "TensorMap gradient conversion",
    )
    if not jnp.issubdtype(data.dtype, jnp.inexact):
        raise TypeError("TensorMap gradients require real or complex storage")

    coordinate = jnp.conj(data) if jnp.issubdtype(
        data.dtype,
        jnp.complexfloating,
    ) else data
    blocks = tuple(_blockstructure_items(cotangent.space))
    if not blocks:
        return TensorMap(cotangent.space, coordinate)

    sector_type = cotangent.space.sector_spec
    weights = tuple(sector_type.quantum_dim(coupled) for coupled, _block in blocks)
    if all(weight == 1 for weight in weights):
        return TensorMap(cotangent.space, coordinate)

    real_dtype = jnp.real(coordinate).dtype
    pieces = tuple(
        coordinate[block.start : block.stop]
        / jnp.asarray(weight, dtype=real_dtype)
        for (_coupled, block), weight in zip(blocks, weights, strict=True)
    )
    return TensorMap(cotangent.space, jnp.concatenate(pieces))
