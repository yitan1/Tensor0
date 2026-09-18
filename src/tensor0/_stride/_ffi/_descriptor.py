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
    *, source_shape: tuple[int, ...], source_strides: tuple[int, ...], source_offset: int,
    output_shape: tuple[int, ...], output_strides: tuple[int, ...], output_offset: int,
    reduction_axes: tuple[bool, ...], source_size: int, output_size: int,
) -> np.ndarray:
    """Encode one real input/output layout pair as little-endian protocol bytes.

    Only protocol representation is checked here; native validates true views
    and constructs shared traversal records. No coefficient data is encoded.
    """
    def offset(value: int) -> int:
        if type(value) is not int or not 0 <= value <= INT64_MAX:
            raise ValueError("reduction layout offsets must fit nonnegative int64")
        return value

    def stride(value: int) -> int:
        if type(value) is not int or not INT64_MIN <= value <= INT64_MAX:
            raise ValueError("reduction layout strides must fit int64")
        return value & UINT64_MAX

    words = [1, _reduction_size(source_size), _reduction_size(output_size), 1]
    rank = len(source_shape)
    if any(len(field) != rank for field in (
        source_strides, output_shape, output_strides, reduction_axes,
    )):
        raise ValueError("reduction layout field ranks must match")
    if any(type(flag) is not bool for flag in reduction_axes):
        raise ValueError("reduction axis flags must be booleans")
    words.extend((rank, offset(source_offset), offset(output_offset)))
    words.extend(_reduction_size(value) for value in source_shape)
    words.extend(stride(value) for value in source_strides)
    words.extend(_reduction_size(value) for value in output_shape)
    words.extend(stride(value) for value in output_strides)
    words.extend(int(flag) for flag in reduction_axes)
    return np.asarray(words, dtype="<u8").view(np.uint8)


def merge_reduction_layouts(
    layouts: tuple[np.ndarray, ...], *, source_size: int, output_size: int,
) -> np.ndarray:
    """Join single-record encoder outputs in order, without changing their payloads.

    All records share storage sizes, not necessarily logical rank or output view.
    An empty tuple emits the existing zero-record protocol for fresh initialization.
    """
    header = np.asarray([1, _reduction_size(source_size), _reduction_size(output_size),
                         len(layouts)], dtype="<u8")
    for layout in layouts:
        words = layout.view("<u8")
        if not np.array_equal(words[:3], header[:3]) or words[3] != 1:
            raise ValueError("expected single-record layouts with matching storage sizes")
    return np.concatenate([header.view(np.uint8), *(layout[32:] for layout in layouts)])
