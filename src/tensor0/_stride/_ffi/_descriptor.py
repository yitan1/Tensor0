"""Pure address encoding for native map, dot, accumulation, and reduction."""

import numpy as np

from .._layout import AffineRecord, INT64_MAX, INT64_MIN, UINT64_MAX


def encode_layout(
    records: tuple[AffineRecord, ...], *, source_size: int, output_size: int,
) -> np.ndarray:
    """Encode paired addresses; for Dot, destination fields describe the right input.

    Map, Dot, and accumulation share this representation, not their validation.
    The selected native handler determines the operation's address contract.
    """
    def size(value: int) -> int:
        if type(value) is not int or not 0 <= value <= INT64_MAX:
            raise ValueError("layout sizes and offsets must fit nonnegative int64")
        return value

    words = [1, size(source_size), size(output_size), len(records)]
    for record in records:
        rank = len(record.logical_shape)
        if len(record.source_strides) != rank or len(record.destination_strides) != rank:
            raise ValueError("layout shape and stride ranks must match")
        words.extend((rank, size(record.source_offset), size(record.destination_offset)))
        words.extend(size(extent) for extent in record.logical_shape)
        for stride in (*record.source_strides, *record.destination_strides):
            if type(stride) is not int or not INT64_MIN <= stride <= INT64_MAX:
                raise ValueError("layout strides must fit int64")
            words.append(stride)
    return np.asarray(words, dtype=np.int64)


def _reduction_size(value: int) -> int:
    if type(value) is not int or not 0 <= value <= UINT64_MAX:
        raise ValueError("reduction layout sizes must fit uint64")
    return value


def encode_reduction_layout(
    records: tuple[AffineRecord, ...], *, output_shapes: tuple[tuple[int, ...], ...],
    reduction_axes: tuple[tuple[bool, ...], ...], source_size: int, output_size: int,
) -> np.ndarray:
    """Encode all input/output layout pairs as little-endian protocol bytes.

    Only protocol representation is checked here; native validates true views
    and constructs shared traversal records. No coefficient data is encoded.
    Empty records emit a header for fresh output initialization.
    """
    def offset(value: int) -> int:
        if type(value) is not int or not 0 <= value <= INT64_MAX:
            raise ValueError("reduction layout offsets must fit nonnegative int64")
        return value

    def stride(value: int) -> int:
        if type(value) is not int or not INT64_MIN <= value <= INT64_MAX:
            raise ValueError("reduction layout strides must fit int64")
        return value & UINT64_MAX

    words = [1, _reduction_size(source_size), _reduction_size(output_size), len(records)]
    for record, output_shape, axes in zip(records, output_shapes, reduction_axes, strict=True):
        rank = len(record.logical_shape)
        if any(len(field) != rank for field in (
            record.source_strides, output_shape, record.destination_strides, axes,
        )):
            raise ValueError("reduction layout field ranks must match")
        if any(type(flag) is not bool for flag in axes):
            raise ValueError("reduction axis flags must be booleans")
        words.extend((rank, offset(record.source_offset), offset(record.destination_offset)))
        words.extend(_reduction_size(value) for value in record.logical_shape)
        words.extend(stride(value) for value in record.source_strides)
        words.extend(_reduction_size(value) for value in output_shape)
        words.extend(stride(value) for value in record.destination_strides)
        words.extend(int(flag) for flag in axes)
    return np.asarray(words, dtype="<u8").view(np.uint8)
