"""Address records shared by local and partitioning tests."""

from tensor0._stride._layout import AffineRecord


PARTITIONS = (AffineRecord((4, 2), (4, 1), 0, (4, 1), 2),
              AffineRecord((4, 2), (4, 1), 2, (4, 1), 0))


PARTIAL = (AffineRecord((16,), (2,), 1, (3,), 2),)


OVERLAP = AffineRecord((2, 2), (1, 2), 0, (1, 1), 1)
