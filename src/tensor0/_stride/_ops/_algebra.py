"""Selected-view algebra using normalized coefficients and native update."""

import jax.numpy as jnp

from .._dtype import normalize_coefficient
from .._jax import update_p
from .._layout import AffineRecord
from .._view import StridedView


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
