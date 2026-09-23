"""Static affine layout metadata."""

from __future__ import annotations

import pytest

from tensor0._stride._layout import contiguous_strides


@pytest.mark.parametrize("shape,expected", [
    ((), ()), ((4,), (1,)), ((2, 3), (3, 1)),
    ((2, 0, 3), (3, 3, 1)), ((0, 0), (1, 1)), ((1,) * 12, (1,) * 12),
])
def test_contiguous_strides(shape, expected) -> None:
    assert contiguous_strides(shape) == expected
