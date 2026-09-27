"""Small raw V1 descriptor fixtures shared by bridge and lowering tests."""

from tensor0._stride._ffi._descriptor import encode_layout, encode_reduction_layout
from tensor0._stride._layout import AffineRecord


OPERATIONS = ("copy", "update", "dot", "accumulation", "reduction")


def address_words(*, destination_strides=(3, 1)):
    return tuple(map(int, encode_layout(
        (AffineRecord((2, 3), (3, 1), 0, destination_strides, 0),),
        source_size=6, output_size=6)))


def reduction_words():
    layout = encode_reduction_layout(
        (AffineRecord((2, 3), (3, -1), 2, (1, -(1 << 63)), 0),),
        output_shapes=((2, 1),), reduction_axes=((False, True),),
        source_size=6, output_size=2)
    return tuple(map(int, layout.view("<i8")))
