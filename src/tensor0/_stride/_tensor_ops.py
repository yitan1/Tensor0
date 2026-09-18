"""Packed tensor transforms and trace with separate layout and coefficients."""

from collections.abc import Iterable
from math import prod

import jax
from jax import Array
import jax.numpy as jnp
from jax.typing import DTypeLike

from .. import _native
from ._dtype import normalize_coefficient
from ._jax import accumulation_p, copy_p, reduction_p, update_p
from ._layout import AffineRecord, contiguous_strides


def _strided_affine_transform(
    source: Array, *,
    source_subblocks: tuple[_native.SubblockStructure, ...],
    destination_subblocks: tuple[_native.SubblockStructure, ...],
    entries: Iterable[tuple[int, int, object]], permutation: tuple[int, ...],
    output_size: int, result_dtype: DTypeLike, shape_error: str,
) -> Array:
    """Transform packed subblocks into fresh storage, preserving storage batches.

    Producers supply injective, mutually disjoint destination subblocks;
    unselected storage stays zero and repeated source reads are allowed. Native
    checks address safety, not cross-record uniqueness. Coefficients are scalars
    shared across batches, normalized against source dtype without serialization.
    With no coefficients, Copy preserves identity bits. Otherwise accumulation
    initializes output once and binds None/zero/one/general scaling per record.
    Its mixed arithmetic and writeback rules apply, not legacy staged casts or
    bitwise copy semantics. Source AD supports same-kind floating/complex casts
    and F16/BF16/F32/F64/C64/C128 real/complex crossings. Coefficient AD requires
    F16/BF16/F32/F64/C64/C128 differentiated operands, including real/complex crossings.
    """
    if not isinstance(source, (Array, jax.core.Tracer)):
        raise TypeError("affine transform source must be a JAX Array or Tracer")
    if source.ndim == 0:
        raise ValueError("affine transform source requires a storage dimension")
    if permutation and (any(type(axis) is not int for axis in permutation)
                        or sorted(permutation) != list(range(len(permutation)))):
        raise ValueError("affine transform permutation must contain each source axis once")
    records = []
    coefficients = []
    coefficient_records = []
    for record_index, (source_index, destination_index, coefficient) in enumerate(entries):
        if (type(source_index) is not int or not 0 <= source_index < len(source_subblocks)
                or type(destination_index) is not int
                or not 0 <= destination_index < len(destination_subblocks)):
            raise ValueError("affine transform subblock index is out of bounds")
        source_subblock = source_subblocks[source_index]
        destination_subblock = destination_subblocks[destination_index]
        logical_shape = tuple(source_subblock.sizes)
        source_strides = tuple(source_subblock.strides)
        if permutation:
            if len(permutation) != len(logical_shape) or len(source_strides) != len(logical_shape):
                raise ValueError("affine transform permutation must match source subblock rank")
            logical_shape = tuple(logical_shape[axis] for axis in permutation)
            source_strides = tuple(source_strides[axis] for axis in permutation)
        if logical_shape != tuple(destination_subblock.sizes):
            raise ValueError(shape_error)
        records.append(AffineRecord(
            logical_shape, source_strides, source_subblock.offset,
            tuple(destination_subblock.strides), destination_subblock.offset,
        ))
        if coefficient is not None:
            factor = normalize_coefficient(source.dtype, coefficient)
            if factor.ndim != 0:
                raise ValueError("affine transform coefficients must be scalars shared across storage batches")
            coefficients.append(factor)
            coefficient_records.append(record_index)
    parameters = dict(records=tuple(records), output_size=output_size,
                      dtype=jax.dtypes.canonicalize_dtype(result_dtype))
    if not coefficients:
        return copy_p.bind(source, **parameters)
    return accumulation_p.bind(
        source, *coefficients, coefficient_records=tuple(coefficient_records), **parameters,
    )


def _strided_tree_transform(
    source: Array, *, source_layout: _native.DegeneracyStructure,
    destination_layout: _native.DegeneracyStructure, result_dtype: DTypeLike,
    permutation: tuple[int, ...], transformer: _native.TreeTransformer,
) -> Array:
    """Execute a tree transform through affine or grouped packed operations.

    Transformer entries provide subblock indices and separate coefficients.
    Metadata producers guarantee complete, disjoint destination writes. Numeric
    behavior and AD follow the selected adapter. Neither path falls back to the
    old backend.
    """
    if transformer.kind == "abelian":
        return _strided_affine_transform(
            source, source_subblocks=source_layout.subblockstructure,
            destination_subblocks=destination_layout.subblockstructure,
            entries=((entry.src, entry.dst, entry.coeff) for entry in transformer.abelian_data),
            permutation=permutation, output_size=destination_layout.total_dim,
            result_dtype=result_dtype,
            shape_error="Abelian tree transform subblock shapes are inconsistent",
        )
    if transformer.kind == "generic":
        return _strided_grouped_transform(
            source, source_layout, destination_layout, result_dtype, permutation,
            transformer.generic_data,
        )
    raise ValueError(f"unsupported tree transformer kind {transformer.kind!r}")


def _strided_grouped_transform(
    source: Array, source_layout: _native.DegeneracyStructure,
    destination_layout: _native.DegeneracyStructure, result_dtype: DTypeLike,
    permutation: tuple[int, ...], data: tuple[_native.GenericTransformData, ...],
) -> Array:
    """Map scalar groups directly and pack only multi-tree matrix groups.

    Scalar coefficients retain the caller's result dtype and use affine
    zero/one/general binding without preconverting source values. Multi-tree
    pack converts inputs before matrix multiplication. Producers guarantee
    valid group metadata and complete disjoint output selections. Both paths
    support same-kind floating/complex source AD and F16/BF16/F32/F64/C64/C128 real/complex
    crossings, using their own conversion contracts rather than matching each
    other's rounding steps.
    """
    result_dtype = jax.dtypes.canonicalize_dtype(result_dtype)
    source_subblocks = source_layout.subblockstructure
    destination_subblocks = destination_layout.subblockstructure
    pack_records = []
    unpack_records = []
    groups = []
    direct_entries = []
    pack_offset = 0
    unpack_offset = 0
    for entry in data:
        source_indices = tuple(entry.src_indices)
        destination_indices = tuple(entry.dst_indices)
        transform = jnp.asarray(entry.transform, dtype=result_dtype)
        if transform.shape == (1, 1):
            direct_entries.append((source_indices[0], destination_indices[0], transform.reshape(())))
            continue
        source_sizes = tuple(source_subblocks[source_indices[0]].sizes)
        block_size = prod(source_sizes)
        group_pack_offset = pack_offset
        source_contiguous_strides = contiguous_strides(source_sizes)
        for source_index in source_indices:
            subblock = source_subblocks[source_index]
            if tuple(subblock.sizes) != source_sizes:
                raise ValueError("generic tree transform source subblock shapes are inconsistent")
            pack_records.append(AffineRecord(
                source_sizes, tuple(subblock.strides), subblock.offset,
                source_contiguous_strides, pack_offset,
            ))
            pack_offset += block_size

        logical_shape = tuple(source_sizes[index] for index in permutation)
        permuted_strides = tuple(source_contiguous_strides[index] for index in permutation)
        for destination_index in destination_indices:
            subblock = destination_subblocks[destination_index]
            if tuple(subblock.sizes) != logical_shape:
                raise ValueError("generic tree transform destination subblock shapes are inconsistent")
            unpack_records.append(AffineRecord(
                logical_shape, permuted_strides, unpack_offset,
                tuple(subblock.strides), subblock.offset,
            ))
            unpack_offset += block_size
        groups.append((group_pack_offset, block_size, transform))

    if direct_entries:
        result = _strided_affine_transform(
            source, source_subblocks=source_subblocks, destination_subblocks=destination_subblocks,
            entries=direct_entries, permutation=permutation,
            output_size=destination_layout.total_dim, result_dtype=result_dtype,
            shape_error="generic tree transform destination subblock shapes are inconsistent",
        )
        if not groups:
            return result

    packed = copy_p.bind(source, records=tuple(pack_records), output_size=pack_offset, dtype=result_dtype)
    batch_shape = packed.shape[:-1]
    pieces = []
    for source_offset, block_size, transform in groups:
        destination_row_count, source_row_count = transform.shape
        start = source_offset
        stop = start + source_row_count * block_size
        source_rows = packed[..., start:stop].reshape((*batch_shape, source_row_count, block_size))
        destination_rows = transform @ source_rows
        pieces.append(destination_rows.reshape((*batch_shape, destination_row_count * block_size)))
    arena = jnp.concatenate(pieces, axis=-1) if pieces else jnp.zeros((*batch_shape, 0), dtype=result_dtype)
    if direct_entries:
        return update_p.bind(arena, result, jnp.int32(1), jnp.int32(0), records=tuple(unpack_records))
    return copy_p.bind(
        arena, records=tuple(unpack_records), output_size=destination_layout.total_dim, dtype=result_dtype,
    )


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
