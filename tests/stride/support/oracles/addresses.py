"""Independent affine address enumeration."""

from itertools import product

import numpy as np


def addresses(shape, strides, offset):
    return np.asarray([offset + sum(index * stride for index, stride in zip(coordinates, strides, strict=True))
                       for coordinates in product(*(range(extent) for extent in shape))], dtype=np.int64)


def record_addresses(record):
    return (addresses(record.logical_shape, record.source_strides, record.source_offset),
            addresses(record.logical_shape, record.destination_strides, record.destination_offset))
