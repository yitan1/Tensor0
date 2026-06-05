from __future__ import annotations

from .. import _native

SectorKey = int | tuple[int, ...]
SectorType = _native.SectorSpec


def _normalize_sector_key(key: object) -> tuple[int, ...]:
    if isinstance(key, bool):
        raise TypeError("sector key must be an int or a tuple of ints")
    if isinstance(key, int):
        return (key,)
    if isinstance(key, tuple) and all(
        isinstance(value, int) and not isinstance(value, bool) for value in key
    ):
        return key
    raise TypeError("sector key must be an int or a tuple of ints")
