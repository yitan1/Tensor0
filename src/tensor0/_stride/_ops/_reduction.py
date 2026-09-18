"""View reductions using native's explicit accumulation contract."""

from collections.abc import Iterable
from math import prod

import jax
from jax import Array
import jax.numpy as jnp
from jax.typing import DTypeLike

from ... import _native
from .._dtype import normalize_coefficient
from .._jax import reduction_p
from .._layout import AffineRecord, contiguous_strides
from .._view import StridedView
from ._materialize import materialize


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


def _strided_tensortrace(
    source: Array, *, destination_size: int,
    source_subblocks: tuple[_native.SubblockStructure, ...],
    destination_subblocks: tuple[_native.SubblockStructure, ...],
    entries: Iterable[tuple[int, int, object]], result_dtype: DTypeLike,
    permutation: tuple[int, ...], num_open_out: int, num_open_in: int, trace_count: int,
) -> Array:
    """Submit packed trace contributions through one native reduction call.

    Paired-axis strides are added as in the old trace producer. Layout bytes and
    scalar coefficient operands remain separate; None and one omit scaling,
    while zero skips the contribution. Weak coefficients use source-based JAX resolution.
    Native mixed arithmetic and cross-record writeback, not legacy preconversion
    or record-local partial sums, define the numerical contract. Source AD supports
    same-kind floating/complex changes and F16/BF16/F32/F64/C64/C128 real/complex crossings.
    Coefficient AD requires F16/BF16/F32/F64/C64/C128 results and differentiated
    coefficients; fixed source storage may also be integer/bool. Fixed coefficients
    retain their dtypes. Conjugation and production routing are not introduced
    by this adapter.
    """
    if not isinstance(source, (Array, jax.core.Tracer)):
        raise TypeError("trace source must be a JAX Array or Tracer")
    if source.ndim == 0:
        raise ValueError("trace source requires a storage dimension")
    counts = (num_open_out, num_open_in, trace_count)
    if any(type(count) is not int or count < 0 for count in counts):
        raise ValueError("trace axis counts must be nonnegative integers")
    rank = num_open_out + num_open_in + 2 * trace_count
    if permutation and (any(type(axis) is not int for axis in permutation)
                        or sorted(permutation) != list(range(rank))):
        raise ValueError("trace permutation must contain each source axis once")
    open_rank = num_open_out + num_open_in
    domain_open_start = num_open_out + trace_count
    trace_output_start = domain_open_start + num_open_in
    open_axes = (*range(num_open_out), *range(domain_open_start, trace_output_start))
    records = []
    output_shapes = []
    reduction_axes = []
    coefficients = []
    coefficient_records = []
    for record_index, (source_index, destination_index, coefficient) in enumerate(entries):
        if (type(source_index) is not int or not 0 <= source_index < len(source_subblocks)
                or type(destination_index) is not int
                or not 0 <= destination_index < len(destination_subblocks)):
            raise ValueError("trace subblock index is out of bounds")
        source_subblock = source_subblocks[source_index]
        destination_subblock = destination_subblocks[destination_index]
        source_sizes = tuple(source_subblock.sizes)
        source_strides = tuple(source_subblock.strides)
        if len(source_sizes) != rank or len(source_strides) != rank:
            raise ValueError("trace source subblock rank is inconsistent")
        if permutation:
            source_sizes = tuple(source_sizes[axis] for axis in permutation)
            source_strides = tuple(source_strides[axis] for axis in permutation)
        logical_shape = tuple(source_sizes[axis] for axis in open_axes)
        logical_source_strides = tuple(source_strides[axis] for axis in open_axes)
        for trace_index in range(trace_count):
            left_axis = num_open_out + trace_index
            right_axis = trace_output_start + trace_index
            if source_sizes[left_axis] != source_sizes[right_axis]:
                raise ValueError("trace source subblock axes have inconsistent sizes")
            logical_shape += (source_sizes[left_axis],)
            logical_source_strides += (source_strides[left_axis] + source_strides[right_axis],)
        destination_sizes = tuple(destination_subblock.sizes)
        if logical_shape[:open_rank] != destination_sizes:
            raise ValueError("trace destination subblock shape is inconsistent")
        records.append(AffineRecord(
            logical_shape, logical_source_strides, source_subblock.offset,
            tuple(destination_subblock.strides) + (1,) * trace_count,
            destination_subblock.offset,
        ))
        output_shapes.append(destination_sizes + (1,) * trace_count)
        reduction_axes.append((False,) * open_rank + (True,) * trace_count)
        if coefficient is not None:
            factor = normalize_coefficient(source.dtype, coefficient)
            if factor.ndim != 0:
                raise ValueError("trace coefficients must be scalars shared across storage batches")
            coefficients.append(factor)
            coefficient_records.append(record_index)
    return reduction_p.bind(
        source, *coefficients, coefficient_records=tuple(coefficient_records),
        records=tuple(records), output_shapes=tuple(output_shapes), reduction_axes=tuple(reduction_axes),
        output_size=destination_size, dtype=jax.dtypes.canonicalize_dtype(result_dtype),
    )
