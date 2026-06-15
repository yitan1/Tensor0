from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Generic, TypeVar

from .sector_type import SectorKey, _normalize_sector_key

_Value = TypeVar("_Value")


@dataclass(frozen=True, init=False)
class SectorDict(Mapping[tuple[int, ...], _Value], Generic[_Value]):
    _items: tuple[tuple[tuple[int, ...], _Value], ...]
    _lookup: Mapping[tuple[int, ...], _Value]

    def __init__(
        self,
        items: Iterable[tuple[SectorKey, _Value]] | Mapping[SectorKey, _Value] = (),
    ) -> None:
        ordered_items: list[tuple[tuple[int, ...], _Value]] = []
        seen: set[tuple[int, ...]] = set()
        raw_items = items.items() if isinstance(items, Mapping) else items

        for raw_key, value in raw_items:
            key = _normalize_sector_key(raw_key)
            if key in seen:
                raise ValueError("sector appears multiple times")
            seen.add(key)
            ordered_items.append((key, value))

        item_tuple = tuple(ordered_items)
        object.__setattr__(self, "_items", item_tuple)
        object.__setattr__(
            self,
            "_lookup",
            MappingProxyType(dict(item_tuple)),
        )

    def __getitem__(self, key: object) -> _Value:
        normalized_key = _normalize_sector_key(key)
        return self._lookup[normalized_key]

    def __iter__(self) -> Iterator[tuple[int, ...]]:
        return (key for key, _value in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, key: object) -> bool:
        try:
            normalized_key = _normalize_sector_key(key)
        except TypeError:
            return False
        return normalized_key in self._lookup
