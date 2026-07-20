from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol, cast

from jax import Array


@dataclass(frozen=True, eq=False, init=False)
class VectorStorage:
    data: Array

    def __init__(self, data: object) -> None:
        if isinstance(data, (str, bytes)):
            raise TypeError("storage data must not be a string or bytes")
        object.__setattr__(self, "data", data)


class _Sliceable(Protocol):
    def __getitem__(self, key: slice, /) -> object: ...


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
