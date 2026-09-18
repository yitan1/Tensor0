"""StridedView operations through native CPU primitives."""

from math import prod

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import DTypeLike

from ._dtype import normalize_coefficient
from ._jax import copy_p, dot_p, reduction_p, update_p
from ._layout import AffineRecord, contiguous_strides
from ._view import StridedView


def materialize(view: StridedView, *, dtype: DTypeLike | None = None) -> Array:
    """Copy selected values into compact storage, converting at each write.

    Conversion uses native scalar rules; same-dtype copy preserves stored bits.
    AD supports floating/complex conversions, including real/complex crossings.
    Reverse mode projects onto the real component for real inputs, or embeds
    real cotangents with zero imaginary part for complex inputs, then converts
    each cotangent to source dtype before address accumulation.
    CPU NamedSharding supports batch partitioning with complete packed storage
    on each device. Forward and JVP use local Copy; reverse mode uses local
    address accumulation. Integer/bool results have float0 zero tangents.
    """
    if not isinstance(view, StridedView):
        raise TypeError("materialize requires a StridedView")
    result_dtype = view.data.dtype if dtype is None else jax.dtypes.canonicalize_dtype(dtype)
    result = copy_p.bind(
        view.data, records=(AffineRecord(
            view.sizes, view.strides, view.offset, contiguous_strides(view.sizes), 0,
        ),), output_size=view.element_count, dtype=result_dtype,
    )
    return result.reshape((*view.batch_shape, *view.sizes))


def scale(view: StridedView, alpha: object) -> StridedView:
    """Scale an injective selection, preserving layout and unselected storage.

    Coefficients must be scalar or match the batch shape. Weak coefficients are
    normalized before native zero/one branching; strong dtypes are preserved.
    The result keeps the storage dtype. Input AD supports floating/complex
    storage. Nonzero coefficient gradients require F16/BF16/F32/F64/C64/C128 storage and coefficient
    dtypes and differentiates the algebraic formula, including zero/one factors.
    Integer/bool results have float0 tangents and zero coefficient gradients.
    Batch axes support partitioning; packed storage stays local to each device.
    """
    if not isinstance(view, StridedView):
        raise TypeError("view must be a StridedView")
    coefficient = normalize_coefficient(view.data.dtype, alpha)
    if coefficient.shape not in ((), view.batch_shape):
        raise ValueError("scale coefficient must be scalar or match the batch shape")
    record = AffineRecord(
        logical_shape=view.sizes,
        source_strides=view.strides,
        source_offset=view.offset,
        destination_strides=view.strides,
        destination_offset=view.offset,
    )
    result = update_p.bind(view.data, view.data, coefficient, jnp.int32(0), records=(record,))
    return view._with_data(result)


def add(
    left: StridedView, right: StridedView, *, alpha: object = 1, beta: object = 1,
) -> StridedView:
    """Compute selected ``alpha * left + beta * right`` into left's layout.

    Logical and batch shapes must match. Coefficients are scalar or match the
    batch shape, and weak coefficients are normalized before native branching.
    Native validates supported source/storage dtype pairs. Floating/complex
    input AD supports these pairs and accumulates repeated
    right addresses, including zero- and nonzero-stride overlap, with fused
    scaling. Nonzero coefficient gradients require F16/BF16/F32/F64/C64/C128 storage and differentiated
    coefficient dtypes; fixed coefficients retain their dtypes. Integer/bool
    results have float0 tangents and zero coefficient gradients. Batch axes support
    partitioning; packed storage stays local to each device.
    Unselected left storage is preserved and neither input is modified.
    """
    for name, view in (("left", left), ("right", right)):
        if not isinstance(view, StridedView):
            raise TypeError(f"{name} must be a StridedView")
    if left.sizes != right.sizes:
        raise ValueError("source and destination logical shapes must match")
    if left.batch_shape != right.batch_shape:
        raise ValueError("source and destination batch shapes must match")
    left_factor = normalize_coefficient(left.data.dtype, alpha)
    right_factor = normalize_coefficient(right.data.dtype, beta)
    for coefficient in (left_factor, right_factor):
        if coefficient.shape not in ((), left.batch_shape):
            raise ValueError("add coefficients must be scalar or match the batch shape")
    record = AffineRecord(
        logical_shape=left.sizes,
        source_strides=right.strides,
        source_offset=right.offset,
        destination_strides=left.strides,
        destination_offset=left.offset,
    )
    result = update_p.bind(
        right.data, left.data, right_factor, left_factor, records=(record,),
    )
    return left._with_data(result)


def _execute_dot(
    left: StridedView, right: StridedView, *, conjugate_left: bool,
    dtype: DTypeLike | None,
) -> Array:
    for name, view in (("left", left), ("right", right)):
        if not isinstance(view, StridedView):
            raise TypeError(f"{name} must be a StridedView")
    if left.sizes != right.sizes:
        raise ValueError("dot logical shapes must match")
    result_dtype = jnp.result_type(left.data.dtype, right.data.dtype) if dtype is None else jax.dtypes.canonicalize_dtype(dtype)
    record = AffineRecord(left.sizes, left.strides, left.offset, right.strides, right.offset)
    return dot_p.bind(left.data, right.data, records=(record,), conjugate_left=conjugate_left, dtype=result_dtype)


def dotu(left: StridedView, right: StridedView, *, dtype: DTypeLike | None = None) -> Array:
    """Sum unconjugated products over all logical axes, preserving batch shape.

    Logical/batch shapes must match. Inputs and results support bool, signed and
    unsigned 8/16/32/64-bit integers, F16/BF16/F32/F64 and C64/C128.
    The default result follows concrete dtype promotion. Explicit dtype selects
    accumulator storage, not input conversion before multiplication. Integer
    products follow scalar promotion and wrap in that type before accumulation.
    Bool products and sums use AND and OR. Integer/bool inputs have float0
    tangents; integer/bool results have zero tangent spaces. First-order JVP/VJP
    also support low-precision floating inputs. Direct transpose has verified
    coverage for F32/F64/C64/C128 input/result combinations, projecting each
    product contribution to the input gradient's storage type at writeback.
    Higher derivatives through mixed coefficient AD support F16/BF16/F32/F64/C64/C128,
    including real/complex crossings.
    AD and vmap use native
    Dot and address accumulation, including repeated input addresses.
    CPU NamedSharding partitions batches while both storage axes stay complete.
    Matching batch placement uses local Dot and local input-gradient accumulation.
    Native traversal and row writeback determine rounding, not legacy partial sums.
    """
    return _execute_dot(left, right, conjugate_left=False, dtype=dtype)


def dotc(left: StridedView, right: StridedView, *, dtype: DTypeLike | None = None) -> Array:
    """Conjugate the left input before Dot, with the same contract as dotu."""
    return _execute_dot(left, right, conjugate_left=True, dtype=dtype)


def reduce_sum(
    view: StridedView, axes: tuple[int, ...] | None = None, *,
    dtype: DTypeLike | None = None,
) -> Array:
    """Sum logical axes, retaining storage batches and returning a dense array.

    Result dtype resolution follows JAX sum, including integer promotion.
    The dtype controls native accumulator storage, not a pre-add input cast.
    Native planning and row writeback determine accumulation and rounding order;
    this is not a bitwise substitute for JAX or the old stride reduction.
    Empty axes perform only materialization/conversion. Source AD supports
    floating-to-floating and complex-to-complex dtype changes, with gradients
    returned in source dtype. Real/complex crossings support F16/BF16/F32/F64/C64/C128.
    Batch axes support partitioning; packed storage remains local to each device.
    Integer/bool results have float0 zero tangents, including empty-axis casts.
    This independent entry does not route to the old backend.
    """
    if not isinstance(view, StridedView):
        raise TypeError("reduce_sum requires a StridedView")
    if axes is None:
        logical_axes = tuple(range(view.rank))
    else:
        normalized = []
        for axis in axes:
            if not isinstance(axis, int) or isinstance(axis, bool):
                raise TypeError("reduction axes must be integers")
            logical_axis = axis + view.rank if axis < 0 else axis
            if logical_axis < 0 or logical_axis >= view.rank:
                raise ValueError("reduction axis is out of bounds")
            normalized.append(logical_axis)
        if len(set(normalized)) != len(normalized):
            raise ValueError("reduction axes must be unique")
        logical_axes = tuple(sorted(normalized))
    result_dtype = jax.eval_shape(
        lambda value: jnp.sum(value, dtype=dtype),
        jax.ShapeDtypeStruct((), view.data.dtype),
    ).dtype
    if not logical_axes:
        return materialize(view, dtype=result_dtype)
    output_shape = tuple(
        extent for axis, extent in enumerate(view.sizes) if axis not in logical_axes
    )
    output_view_shape = tuple(
        1 if axis in logical_axes else extent for axis, extent in enumerate(view.sizes)
    )
    result = reduction_p.bind(
        view.data, records=(AffineRecord(
            view.sizes, view.strides, view.offset, contiguous_strides(output_view_shape), 0,
        ),), output_shapes=(output_view_shape,),
        reduction_axes=(tuple(axis in logical_axes for axis in range(view.rank)),),
        output_size=prod(output_shape), dtype=result_dtype,
    )
    return result.reshape((*view.batch_shape, *output_shape))
