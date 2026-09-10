"""Data-bound immutable affine views for Tensor0 stride producers."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import TypeAlias

import jax
from jax import Array
from jax.core import Tracer
import jax.numpy as jnp

from ._plan import INT64_MAX, INT64_MIN, UINT64_MAX, contiguous_strides


StridedIndex: TypeAlias = int | slice


def _require_integer(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{field} must be an integer")
    return value


def _require_sizes(values: object) -> tuple[int, ...]:
    if not isinstance(values, tuple):
        raise TypeError("StridedView sizes must be a tuple")
    sizes = tuple(
        _require_integer(value, f"StridedView sizes[{axis}]")
        for axis, value in enumerate(values)
    )
    if any(value < 0 for value in sizes):
        raise ValueError("StridedView sizes must be non-negative")
    if any(value > UINT64_MAX for value in sizes):
        raise ValueError("StridedView size exceeds uint64")
    return sizes


def _require_strides(values: object) -> tuple[int, ...]:
    if not isinstance(values, tuple):
        raise TypeError("StridedView strides must be a tuple")
    strides = tuple(
        _require_integer(value, f"StridedView strides[{axis}]")
        for axis, value in enumerate(values)
    )
    if any(value < INT64_MIN or value > INT64_MAX for value in strides):
        raise ValueError("StridedView stride exceeds int64")
    return strides


def _reshape_strides(
    old_sizes: tuple[int, ...],
    old_strides: tuple[int, ...],
    new_sizes: tuple[int, ...],
) -> tuple[int, ...] | None:
    """Compute row-major logical reshape strides without moving elements."""

    if prod(old_sizes) != prod(new_sizes):
        return None
    element_count = prod(old_sizes)
    if element_count == 0:
        return contiguous_strides(new_sizes)
    if element_count == 1:
        return contiguous_strides(new_sizes)
    if not old_sizes or not new_sizes:
        return None

    new_strides = [0] * len(new_sizes)
    view_axis = len(new_sizes) - 1
    chunk_base_stride = old_strides[-1]
    old_chunk_elements = 1
    new_chunk_elements = 1

    for old_axis in range(len(old_sizes) - 1, -1, -1):
        old_chunk_elements *= old_sizes[old_axis]
        starts_new_chunk = old_axis == 0 or (
            old_sizes[old_axis - 1] != 1
            and old_strides[old_axis - 1]
            != old_chunk_elements * chunk_base_stride
        )
        if not starts_new_chunk:
            continue

        while view_axis >= 0 and (
            new_chunk_elements < old_chunk_elements
            or new_sizes[view_axis] == 1
        ):
            stride = new_chunk_elements * chunk_base_stride
            if stride < INT64_MIN or stride > INT64_MAX:
                return None
            new_strides[view_axis] = stride
            new_chunk_elements *= new_sizes[view_axis]
            view_axis -= 1
        if new_chunk_elements != old_chunk_elements:
            return None
        if old_axis > 0:
            chunk_base_stride = old_strides[old_axis - 1]
            old_chunk_elements = 1
            new_chunk_elements = 1

    if view_axis != -1:
        return None
    return tuple(new_strides)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, slots=True, eq=False)
class StridedView:
    """A JAX storage operand bound to immutable affine element metadata."""

    data: Array
    sizes: tuple[int, ...]
    strides: tuple[int, ...]
    offset: int

    __hash__ = None  # pyright: ignore[reportAssignmentType]

    @classmethod
    def from_dense(
        cls,
        data: Array,
        sizes: tuple[int, ...],
    ) -> StridedView:
        """Bind row-major logical elements after a metadata-only flatten."""

        new_sizes = _require_sizes(sizes)
        if not isinstance(data, (Array, Tracer)):
            raise TypeError("StridedView data must be a JAX Array")
        element_count = prod(new_sizes)
        rank = len(new_sizes)
        if rank == 0:
            batch_shape = data.shape
        elif data.ndim >= rank and tuple(data.shape[-rank:]) == new_sizes:
            batch_shape = data.shape[:-rank]
        elif data.ndim >= 1 and data.shape[-1] == element_count:
            batch_shape = data.shape[:-1]
        else:
            raise ValueError("dense data shape does not match StridedView sizes")
        flat = jnp.reshape(data, (*batch_shape, element_count))
        return cls(flat, new_sizes, contiguous_strides(new_sizes), 0)

    def __post_init__(self) -> None:
        if not isinstance(self.data, (Array, Tracer)):
            raise TypeError("StridedView data must be a JAX Array")
        if self.data.ndim < 1:
            raise ValueError("StridedView data requires at least one storage axis")
        sizes = _require_sizes(self.sizes)
        strides = _require_strides(self.strides)
        if len(sizes) != len(strides):
            raise ValueError("StridedView sizes and strides must have equal rank")
        offset = _require_integer(self.offset, "StridedView offset")
        if offset < 0 or offset > UINT64_MAX:
            raise ValueError("StridedView offset must fit uint64")
        object.__setattr__(self, "sizes", sizes)
        object.__setattr__(self, "strides", strides)
        object.__setattr__(self, "offset", offset)
        self._validate_storage_bounds()

    @property
    def rank(self) -> int:
        return len(self.sizes)

    @property
    def element_count(self) -> int:
        return prod(self.sizes)

    @property
    def storage_size(self) -> int:
        return self.data.shape[-1]

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self.data.shape[:-1]

    def _validate_storage_bounds(self) -> None:
        storage_size = self.storage_size
        if self.element_count == 0:
            if self.offset > storage_size:
                raise ValueError("empty StridedView offset exceeds storage size")
            return
        minimum = self.offset
        maximum = self.offset
        for size, stride in zip(self.sizes, self.strides, strict=True):
            delta = (size - 1) * stride
            minimum += min(delta, 0)
            maximum += max(delta, 0)
        if minimum < 0 or maximum >= storage_size:
            raise ValueError("StridedView addresses exceed storage bounds")

    def tree_flatten(
        self,
    ) -> tuple[tuple[Array], tuple[tuple[int, ...], tuple[int, ...], int]]:
        return (self.data,), (self.sizes, self.strides, self.offset)

    @classmethod
    def tree_unflatten(
        cls,
        metadata: tuple[tuple[int, ...], tuple[int, ...], int],
        children: tuple[Array],
    ) -> StridedView:
        sizes, strides, offset = metadata
        (data,) = children
        return cls(data, sizes, strides, offset)

    def _with_data(self, data: Array) -> StridedView:
        """Bind the same affine view metadata to replacement storage."""

        if not isinstance(data, (Array, Tracer)):
            raise TypeError("StridedView data must be a JAX Array")
        if data.shape != self.data.shape:
            raise ValueError("replacement storage shape must match StridedView data")
        return StridedView(data, self.sizes, self.strides, self.offset)

    def permute(self, permutation: tuple[int, ...]) -> StridedView:
        """Return the same storage with affine axes permuted."""

        if not isinstance(permutation, tuple):
            raise TypeError("StridedView permutation must be a tuple")
        if tuple(sorted(permutation)) != tuple(range(self.rank)):
            raise ValueError("StridedView permutation must be a rank permutation")
        return StridedView(
            self.data,
            tuple(self.sizes[axis] for axis in permutation),
            tuple(self.strides[axis] for axis in permutation),
            self.offset,
        )

    def subview(self, *indices: StridedIndex) -> StridedView:
        """Return an integer/slice subview without materializing its elements."""

        if len(indices) != self.rank:
            raise IndexError("StridedView subview requires one index per axis")
        sizes: list[int] = []
        strides: list[int] = []
        offset = self.offset
        for axis, (index, size, stride) in enumerate(
            zip(indices, self.sizes, self.strides, strict=True)
        ):
            if isinstance(index, bool):
                raise TypeError(f"StridedView index[{axis}] must be int or slice")
            if isinstance(index, int):
                coordinate = index + size if index < 0 else index
                if coordinate < 0 or coordinate >= size:
                    raise IndexError(f"StridedView index[{axis}] is out of bounds")
                offset += coordinate * stride
                continue
            if not isinstance(index, slice):
                raise TypeError(f"StridedView index[{axis}] must be int or slice")
            start, stop, step = index.indices(size)
            extent = len(range(start, stop, step))
            if extent:
                offset += start * stride
            sizes.append(extent)
            result_stride = stride * step
            if result_stride < INT64_MIN or result_stride > INT64_MAX:
                raise ValueError("StridedView subview stride exceeds int64")
            strides.append(result_stride)
        if 0 in sizes:
            offset = self.offset
        return StridedView(self.data, tuple(sizes), tuple(strides), offset)

    def reshape(self, sizes: tuple[int, ...]) -> StridedView:
        """Return an affine-compatible logical reshape of the same elements."""

        new_sizes = _require_sizes(sizes)
        new_strides = _reshape_strides(self.sizes, self.strides, new_sizes)
        if new_strides is None:
            raise ValueError("StridedView reshape is not affine-compatible")
        return StridedView(self.data, new_sizes, new_strides, self.offset)


__all__ = ["StridedView"]
