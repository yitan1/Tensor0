"""Shared materialize fixtures and references."""

from __future__ import annotations

import numpy as np

from tensor0._stride import StridedView


def reference_materialize(storage, sizes, strides, offset):
    indices = np.full(sizes, offset, dtype=np.int32)
    for axis, (size, stride) in enumerate(zip(sizes, strides, strict=True)):
        shape = (1,) * axis + (size,) + (1,) * (len(sizes) - axis - 1)
        indices += np.arange(size, dtype=np.int32).reshape(shape) * stride
    return storage[..., indices]


def _view(
    data,
    sizes: tuple[int, ...],
    strides: tuple[int, ...],
    offset: int,
) -> StridedView:
    return StridedView(data, sizes, strides, offset)
