"""Shared JAX operand and sharding boundary checks."""

from __future__ import annotations

from typing import Any, cast

from jax import Array
from jax.core import Tracer


def require_jax_array(value: object, argument_name: str) -> Array:
    """Require one JAX execution operand without silently transferring it."""

    if not isinstance(value, (Array, Tracer)):
        raise TypeError(
            f"{argument_name} must be a JAX Array or Tracer, "
            f"got {type(value).__name__}"
        )
    return cast(Array, value)


def batch_update_operands(
    arguments: tuple[Array, ...],
    dimensions: tuple[int | None, ...],
) -> tuple[Array, ...]:
    """Batch storage and coefficients without losing existing batch axes."""

    import jax.numpy as jnp
    from jax.interpreters import batching

    arguments = tuple(jnp.asarray(argument) for argument in arguments)
    batch_size = next(
        argument.shape[dimension]
        for argument, dimension in zip(arguments, dimensions, strict=True)
        if dimension is not None
    )
    values = [
        batching.bdim_at_front(argument, dimension, batch_size)
        for argument, dimension in zip(arguments[:2], dimensions[:2], strict=True)
    ]
    batch_shape = values[0].shape[:-1]
    for argument, dimension in zip(arguments[2:], dimensions[2:], strict=True):
        if dimension is None and argument.ndim == 0:
            values.append(argument)
            continue
        coefficient = batching.bdim_at_front(argument, dimension, batch_size)
        if coefficient.ndim == 1 and len(batch_shape) > 1:
            coefficient = jnp.broadcast_to(
                jnp.reshape(coefficient, (batch_size, *(1,) * (len(batch_shape) - 1))),
                batch_shape,
            )
        values.append(coefficient)
    return tuple(values)


def batch_only_named_sharding(shape: Any) -> Any:
    """Return sharding after proving the packed storage axis is replicated."""

    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec

    sharding = getattr(shape, "sharding", None)
    if not isinstance(sharding, NamedSharding):
        raise ValueError(
            "Tensor0 stride requires NamedSharding for multi-device fallback"
        )
    rank = len(shape.shape)
    specification = tuple(sharding.spec)
    if len(specification) > rank:
        raise ValueError("Tensor0 stride received an invalid sharding rank")
    normalized = specification + (None,) * (rank - len(specification))
    if normalized[-1] is not None:
        raise ValueError(
            "Tensor0 stride cannot shard the packed storage axis without an "
            "explicit collective"
        )
    return NamedSharding(sharding.mesh, PartitionSpec(*normalized))
