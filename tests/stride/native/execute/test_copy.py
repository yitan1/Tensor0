"""Native copy execution through materialization."""

import jax.numpy as jnp
import numpy as np

from tensor0._stride._ffi._calls import execute_copy
from tensor0._stride._ffi._descriptor import encode_layout
from tensor0._stride._layout import AffineRecord


def test_materialize_conversion_partial_multirecord_conversion_and_zero_initialization():
    layout = encode_layout((
        AffineRecord((2,), (-1,), 2, (1,), 1),
        AffineRecord((1,), (1,), 0, (1,), 4),
    ), source_size=3, output_size=6)
    actual = execute_copy(jnp.asarray([5.5, 6.5, 7.5]), layout=layout, output_size=6, dtype="int16")
    np.testing.assert_array_equal(actual, [0, 7, 6, 0, 5, 0])
