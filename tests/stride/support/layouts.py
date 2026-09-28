"""Address records shared by local and partitioning tests."""

from tensor0._stride._layout import AffineRecord


MAPPING_PARTITIONS = (AffineRecord((4, 2), (4, 1), 0, (4, 1), 2),
                      AffineRecord((4, 2), (4, 1), 2, (4, 1), 0))


RAW_UPDATE_PARTIAL = (AffineRecord((16,), (2,), 1, (3,), 2),)


OVERLAP = AffineRecord((2, 2), (1, 2), 0, (1, 1), 1)

ACCUMULATION_RECORDS = (
    AffineRecord((2, 2), (2, 1), 0, (2, 1), 1),
    AffineRecord((0,), (1,), 5, (1,), 5),
    AffineRecord((2, 2), (-2, -1), 3, (2, -1), 1),
)

SINGLE_FACTOR_RECORDS = (AffineRecord((1,), (1,), 0, (1,), 0),)

PRODUCT_STAGE_PARTIAL = (AffineRecord((3,), (1,), 0, (2,), 1),)
