from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol, cast

from jax import Array
from jax.core import Tracer


@dataclass(frozen=True, eq=False, init=False)
class VectorStorage:
    data: Array

    def __init__(self, data: object) -> None:
        if isinstance(data, (str, bytes)):
            raise TypeError("storage data must not be a string or bytes")
        object.__setattr__(self, "data", data)


class _Sliceable(Protocol):
    def __getitem__(self, key: slice, /) -> object: ...


def _require_jax_storage_data(
    data: object, operation: str, *, allow_batch: bool = False,
) -> Array:
    """Require JAX-backed, unbatched storage at a numerical read boundary.

    Callers may delegate batch checks here if every successful path reaches
    this helper (directly or through a checked numerical operation). Keep
    existing argument validation order; raising paths need no batch check.
    Only genuinely batch-aware metric code may use ``allow_batch=True``.
    """

    if not allow_batch:
        _require_unbatched_storage(data, operation)
    if not isinstance(data, (Array, Tracer)):
        raise TypeError(
            f"{operation} requires JAX-backed storage, "
            f"got {type(data).__name__}"
        )
    return cast(Array, data)


def _validate_vector_storage_data(data: object, expected_total_dim: int) -> None:
    shape = cast(Iterable[object] | None, getattr(data, "shape", None))
    if shape is None:
        raise TypeError("storage data must have a shape")

    try:
        actual_shape = tuple(shape)
    except TypeError as exc:
        raise TypeError("storage data shape must be tuple-like") from exc

    expected_shape = (expected_total_dim,)
    if len(actual_shape) != 1:
        raise ValueError(
            f"storage data must be 1D; expected shape {expected_shape}, "
            f"actual shape {actual_shape}",
        )
    if actual_shape != expected_shape:
        raise ValueError(
            f"storage data length mismatch: expected shape {expected_shape}, "
            f"actual shape {actual_shape}",
        )

    if isinstance(data, Array):
        return

    try:
        empty_slice = cast(_Sliceable, data)[0:0]
    except Exception as exc:
        raise TypeError("storage data must support slicing") from exc

    reshape = getattr(empty_slice, "reshape", None)
    if not callable(reshape):
        raise TypeError("storage data slices must support reshape")


def _require_unbatched_storage(data: object, operation: str) -> None:
    """Reject batch containers without requiring JAX-backed storage.

    Check explicitly before direct array operations or successful early returns
    that bypass a checked numerical read, including identity/constant results
    and empty views. Delegating to an unconditional checked call is sufficient;
    do not add an entry check merely because later numerical reads also check.
    """
    shape = getattr(data, "shape", None)
    if shape is not None and len(shape) != 1:
        raise ValueError(
            f"{operation} requires unbatched storage; got shape {tuple(shape)}. "
            "Use jax.vmap to map over all leading batch axes."
        )


def _pytree_vector_storage(data: object, expected_dim: int) -> VectorStorage:
    """Rebuild a leaf with trailing storage and optional leading batch axes.

    JAX also reconstructs trees with opaque placeholders and axis specifications;
    these have no shape and are not numerical tensor values.
    """
    shape = getattr(data, "shape", None)
    if shape is not None and (len(shape) == 0 or shape[-1] != expected_dim):
        raise ValueError(
            "storage data length mismatch: pytree storage must have "
            f"trailing dimension {expected_dim}; "
            f"got shape {tuple(shape)}"
        )
    return VectorStorage(data)
