"""Functional affine algebra with zero/one short-circuit updates."""

from __future__ import annotations

from math import prod

from jax import Array
import jax.numpy as jnp
from jax.typing import DTypeLike

from ._ops._update import _execute_update
from ._ops._materialize import materialize
from ._ops._dot import _execute_dot
from ._plan import (
    AffineRecord,
    CompleteMode,
    StridedReductionKind,
    build_affine_plan,
    contiguous_strides,
)
from ._ops._reduction import lower_structured_reduction, _execute_reduction
from ._ops._selected_scale import _build_strided_scale_plan
from ._ops._update_support import _prepare_strided_update
from ._view import StridedView


def _require_view(value: object, name: str) -> StridedView:
    if not isinstance(value, StridedView):
        raise TypeError(f"{name} must be a StridedView")
    return value


def _require_matching_views(
    source: object,
    destination: object,
) -> tuple[StridedView, StridedView]:
    source_view = _require_view(source, "source")
    destination_view = _require_view(destination, "destination")
    if source_view.sizes != destination_view.sizes:
        raise ValueError("source and destination logical shapes must match")
    if source_view.batch_shape != destination_view.batch_shape:
        raise ValueError("source and destination batch shapes must match")
    return source_view, destination_view


def scale(view: StridedView, alpha: object) -> StridedView:
    """Scale selected elements, preserving the layout and unselected values.

    The coefficient must be scalar or match the batch shape. JAX promotion
    determines the product dtype; only the final result is cast to storage.
    Zero writes zero; one omits multiplication but retains the product cast.
    Neither the input nor its metadata is modified.
    """

    bound = _require_view(view, "view")
    return bound._with_data(
        _execute_update(
            bound.data,
            bound.data,
            source_factor=alpha,
            base_factor=0,
            plan=_build_strided_scale_plan(
                bound.sizes,
                bound.strides,
                bound.offset,
                bound.storage_size,
                bound.data.dtype.name,
            ),
        )
    )


def add(
    left: StridedView,
    right: StridedView,
    *,
    alpha: object = 1,
    beta: object = 1,
) -> StridedView:
    """Update selected elements with ``alpha * left + beta * right``.

    Preserve left's layout, storage shape, dtype, and unselected values without
    modifying either input. Logical and batch shapes must match. Coefficients
    must be scalar or match the batch shape. Each product and sum follows JAX
    promotion, including weak scalar typing; the final result is cast to left's
    dtype. Zero coefficients omit their terms; unit coefficients omit only the
    multiplication, not required casts. Two zero coefficients write zero,
    including for NaN and infinity inputs. Unsupported native dtype pairs fail
    before execution.
    """

    source_view, destination_view = _require_matching_views(right, left)
    base_data, source_data, plan = _prepare_strided_update(
        destination_view, source_view,
    )
    return destination_view._with_data(_execute_update(
        base_data, source_data, source_factor=beta, base_factor=alpha, plan=plan,
    ))


def _dot(
    source: object,
    destination: object,
    *,
    conjugate_source: bool,
    dtype: DTypeLike | None,
) -> Array:
    source_view, destination_view = _require_matching_views(source, destination)
    result_dtype = jnp.result_type(
        source_view.data.dtype,
        destination_view.data.dtype,
    )
    if dtype is not None:
        result_dtype = jnp.dtype(dtype)
    if (
        source_view.data.dtype == destination_view.data.dtype
        and result_dtype == source_view.data.dtype
        and result_dtype.name in ("float32", "complex64")
    ):
        return _execute_dot(
            source_view,
            destination_view,
            conjugate_source,
        )
    left = jnp.asarray(materialize(source_view), dtype=result_dtype)
    right = jnp.asarray(materialize(destination_view), dtype=result_dtype)
    if conjugate_source:
        left = jnp.conj(left)
    logical_axes = tuple(
        range(left.ndim - source_view.rank, left.ndim)
    )
    return jnp.sum(left * right, axis=logical_axes, dtype=result_dtype)


def dotu(
    left: StridedView,
    right: StridedView,
    *,
    dtype: DTypeLike | None = None,
) -> Array:
    """Return the unconjugated dot product, preserving batch axes."""

    return _dot(
        left,
        right,
        conjugate_source=False,
        dtype=dtype,
    )


def dotc(
    left: StridedView,
    right: StridedView,
    *,
    dtype: DTypeLike | None = None,
) -> Array:
    """Conjugate left and sum the product over logical axes, preserving batches."""

    return _dot(
        left,
        right,
        conjugate_source=True,
        dtype=dtype,
    )


def reduce_sum(
    view: StridedView,
    axes: tuple[int, ...] | None = None,
    *,
    dtype: DTypeLike | None = None,
) -> Array:
    """Reduce selected logical axes through the affine reduction primitive."""

    bound = _require_view(view, "view")
    if axes is None:
        logical_axes = tuple(range(bound.rank))
    else:
        normalized: list[int] = []
        for axis in axes:
            if not isinstance(axis, int) or isinstance(axis, bool):
                raise TypeError("reduction axes must be integers")
            logical_axis = axis + bound.rank if axis < 0 else axis
            if logical_axis < 0 or logical_axis >= bound.rank:
                raise ValueError("reduction axis is out of bounds")
            normalized.append(logical_axis)
        if len(set(normalized)) != len(normalized):
            raise ValueError("reduction axes must be unique")
        logical_axes = tuple(sorted(normalized))

    result_dtype = jnp.sum(
        jnp.zeros((), dtype=bound.data.dtype),
        dtype=dtype,
    ).dtype
    if not logical_axes:
        return jnp.asarray(materialize(bound), dtype=result_dtype)

    retained_axes = tuple(
        axis for axis in range(bound.rank) if axis not in logical_axes
    )
    output_shape = tuple(bound.sizes[axis] for axis in retained_axes)
    compact = contiguous_strides(output_shape)
    retained_strides = dict(zip(retained_axes, compact, strict=True))
    record = AffineRecord(
        logical_shape=bound.sizes,
        source_strides=bound.strides,
        source_offset=bound.offset,
        destination_strides=tuple(
            0 if axis in logical_axes else retained_strides[axis]
            for axis in range(bound.rank)
        ),
        destination_offset=0,
        reduction_axes=logical_axes,
    )
    plan = build_affine_plan(
        records=(record,),
        output_size=prod(output_shape),
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=bound.storage_size,
        source_dtype=bound.data.dtype,
        result_dtype=result_dtype,
        reduction_kind=StridedReductionKind.SUM,
    )
    try:
        lower_structured_reduction(plan)
    except ValueError:
        value = materialize(bound)
        physical_axes = tuple(
            value.ndim - bound.rank + axis for axis in logical_axes
        )
        return jnp.sum(value, axis=physical_axes, dtype=dtype)
    reduced = _execute_reduction(bound.data, plan=plan)
    return jnp.reshape(reduced, (*bound.batch_shape, *output_shape))


__all__ = [
    "add",
    "dotc",
    "dotu",
    "reduce_sum",
    "scale",
]
