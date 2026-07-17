from __future__ import annotations

import math
from typing import SupportsFloat

import jax.numpy as jnp
from jax.typing import DTypeLike


def nonnegative_tolerance(value: SupportsFloat, argument_name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{argument_name} must be a non-negative float")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        raise TypeError(
            f"{argument_name} must be a non-negative float",
        ) from None
    if not math.isfinite(result) or result < 0:
        raise ValueError(
            f"{argument_name} must be a non-negative finite float",
        )
    return result


def default_pseudoinverse_rtol(
    real_dtype: DTypeLike,
    largest_block_dim: int,
) -> float:
    return float(10 * largest_block_dim * jnp.finfo(real_dtype).eps)
