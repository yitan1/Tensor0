"""Shared accumulation fixtures and references."""

from tensor0._stride._layout import AffineRecord


TWO_RECORDS = (
    AffineRecord((2, 2), (1, 1), 0, (1, 1), 1),
    AffineRecord((0,), (1,), 5, (1,), 5),
    AffineRecord((2, 2), (-1, -1), 3, (1, -1), 2),
)
