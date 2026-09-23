"""Copy/accumulation records, numerical reference and lowering checks."""

import numpy as np

from tensor0._stride._layout import AffineRecord


PARTITIONS = (AffineRecord((4, 2), (4, 1), 0, (4, 1), 2),
              AffineRecord((4, 2), (4, 1), 2, (4, 1), 0))


def reference_map(source, records, factors, output_size):
    result = np.zeros((*source.shape[:-1], output_size), dtype=source.dtype)
    for record, factor in zip(records, factors, strict=True):
        addresses = []
        for strides, offset in ((record.source_strides, record.source_offset),
                                (record.destination_strides, record.destination_offset)):
            indices = np.full(record.logical_shape, offset, dtype=np.int64)
            for axis, (size, stride) in enumerate(zip(record.logical_shape, strides, strict=True)):
                shape = (1,) * axis + (size,) + (1,) * (len(record.logical_shape) - axis - 1)
                indices += np.arange(size).reshape(shape) * stride
            addresses.append(indices.ravel())
        values = source[..., addresses[0]].astype(np.complex64 if np.iscomplexobj(source) else np.float32)
        result[..., addresses[1]] = (values * factor).astype(source.dtype)
    return result


def assert_single_native_call(lowered):
    text = lowered.as_text().lower()
    assert text.count("custom_call") == 1
    assert "tensor0_stride_" in text
    assert "gather" not in text and "scatter" not in text
