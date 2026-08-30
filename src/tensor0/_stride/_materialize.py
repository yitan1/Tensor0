"""Explicit materialization of immutable affine views."""

from __future__ import annotations

from functools import lru_cache
from math import prod

from jax import Array
import jax.numpy as jnp
from jax.typing import DTypeLike

from ._ffi import strided_copy
from ._plan import (
    CompleteMode,
    StridedCopyPlan,
    StridedCopyRecord,
    _contiguous_strides,
    build_strided_copy_plan,
)
from ._view import StridedView


@lru_cache(maxsize=1_024)
def build_materialize_plan(
    *,
    sizes: tuple[int, ...],
    strides: tuple[int, ...],
    offset: int,
    source_size: int,
    source_dtype: DTypeLike,
    result_dtype: DTypeLike,
) -> StridedCopyPlan:
    """Build one certified affine map into a fresh compact destination."""

    destination_strides = _contiguous_strides(sizes)
    record = StridedCopyRecord(
        logical_shape=sizes,
        source_strides=strides,
        source_offset=offset,
        destination_strides=destination_strides,
        destination_offset=0,
    )
    return build_strided_copy_plan(
        records=(record,),
        output_size=prod(sizes),
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=source_size,
        source_dtype=source_dtype,
        result_dtype=result_dtype,
    )


def _execute_materialize(
    data: Array,
    *,
    sizes: tuple[int, ...],
    strides: tuple[int, ...],
    offset: int,
    result_dtype: DTypeLike,
) -> Array:
    plan = build_materialize_plan(
        sizes=sizes,
        strides=strides,
        offset=offset,
        source_size=data.shape[-1],
        source_dtype=data.dtype,
        result_dtype=result_dtype,
    )
    flat = strided_copy(data, plan=plan)
    return jnp.reshape(flat, (*data.shape[:-1], *sizes))


def materialize(
    view: StridedView,
    *,
    result_dtype: DTypeLike | None = None,
) -> Array:
    """Materialize one affine view as a fresh compact JAX Array."""

    if not isinstance(view, StridedView):
        raise TypeError("materialize requires a StridedView")
    return _execute_materialize(
        view.data,
        sizes=view.sizes,
        strides=view.strides,
        offset=view.offset,
        result_dtype=(view.data.dtype if result_dtype is None else result_dtype),
    )


__all__ = [
    "build_materialize_plan",
    "materialize",
]
