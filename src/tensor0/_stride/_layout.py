"""Layout primitives independent of numerical operations and FFI encoding."""

from __future__ import annotations

from dataclasses import dataclass

UINT64_MAX = (1 << 64) - 1
INT64_MIN = -(1 << 63)
INT64_MAX = (1 << 63) - 1


@dataclass(frozen=True, slots=True)
class AffineRecord:
    logical_shape: tuple[int, ...]
    source_strides: tuple[int, ...]
    source_offset: int
    destination_strides: tuple[int, ...]
    destination_offset: int


def contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Return row-major strides while preserving empty-axis metadata."""

    stride = 1
    result = [0] * len(shape)
    for axis in range(len(shape) - 1, -1, -1):
        result[axis] = stride
        stride *= max(shape[axis], 1)
    return tuple(result)
