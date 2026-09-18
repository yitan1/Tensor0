"""Materialize a view through the native CPU copy/conversion path."""

import jax
from jax import Array
from jax.typing import DTypeLike

from .._jax import copy_p
from .._layout import AffineRecord, contiguous_strides
from .._view import StridedView


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
