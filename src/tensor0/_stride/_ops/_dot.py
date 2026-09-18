"""Mixed-type view Dot through native, without materialized fallback."""

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import DTypeLike

from .._jax import dot_p
from .._layout import AffineRecord
from .._view import StridedView


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
