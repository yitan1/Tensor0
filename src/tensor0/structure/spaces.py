from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from .. import _native
from .sector_type import SectorType, _normalize_sector_key

SectorDims = Mapping[Any, int]
SectorValue = tuple[int, ...]
SectorDimItem = tuple[SectorValue, int]
SectorDimItems = tuple[SectorDimItem, ...]
ProductSpaceInput = Iterable[_native.ElementarySpace] | _native.ProductSpace


@dataclass(frozen=True)
class _VectBuilder:
    sector_type: SectorType

    def __call__(
        self,
        sector_dims: SectorDims,
        *,
        dual: bool = False,
    ) -> _native.ElementarySpace:
        return space(self.sector_type, sector_dims, dual=dual)


class _VectFactory:
    def __getitem__(self, sector_type: SectorType) -> _VectBuilder:
        if not isinstance(sector_type, SectorType):
            raise TypeError("Vect[...] requires a SectorType")
        return _VectBuilder(sector_type)


Vect = _VectFactory()


def space(
    sector_type: SectorType,
    sector_dims: SectorDims,
    *,
    dual: bool = False,
) -> _native.ElementarySpace:
    if not isinstance(sector_type, SectorType):
        raise TypeError("space() requires a SectorType")
    sector_items = _normalize_sector_dims(sector_dims)
    return _native.make_space(sector_type, sector_items, dual)


def hom(
    codomain: ProductSpaceInput,
    domain: ProductSpaceInput,
) -> _native.HomSpace:
    codomain_product = _product_space_or_none(codomain)
    domain_product = _product_space_or_none(domain)
    sector_type = _hom_sector_type(codomain_product, domain_product)

    if codomain_product is None:
        codomain_product = _native.make_product_space(sector_type, ())
    if domain_product is None:
        domain_product = _native.make_product_space(sector_type, ())

    return _native.make_hom_products(codomain_product, domain_product)


def _as_product_space_input(
    value: object,
    argument_name: str,
    function_name: str,
) -> _native.ProductSpace | tuple[_native.ElementarySpace, ...]:
    if isinstance(value, _native.ProductSpace):
        return value
    if isinstance(value, _native.ElementarySpace):
        return (value,)
    raise TypeError(
        f"{function_name}() requires {argument_name} to be an ElementarySpace "
        "or ProductSpace",
    )


def _as_hom_space(value: object, function_name: str) -> _native.HomSpace:
    if not isinstance(value, _native.HomSpace):
        raise TypeError(f"{function_name}() requires a HomSpace")
    return value


def _product_space_or_none(
    value: ProductSpaceInput,
) -> _native.ProductSpace | None:
    if isinstance(value, _native.ProductSpace):
        return value

    spaces = _space_tuple(value)
    if not spaces:
        return None

    return _native.make_product_space(spaces[0].sector_spec, spaces)


def _hom_sector_type(
    codomain: _native.ProductSpace | None,
    domain: _native.ProductSpace | None,
) -> SectorType:
    if codomain is not None:
        return codomain.sector_spec
    if domain is not None:
        return domain.sector_spec
    raise ValueError("hom() requires at least one space")


def _space_tuple(value: object) -> tuple[_native.ElementarySpace, ...]:
    try:
        spaces = tuple(value)  # type: ignore[arg-type]
    except TypeError:
        raise TypeError(
            "hom() requires ProductSpace or iterable of ElementarySpace",
        ) from None

    if not all(isinstance(space, _native.ElementarySpace) for space in spaces):
        raise TypeError("hom() requires ProductSpace or iterable of ElementarySpace")
    return spaces


def _normalize_sector_dims(sector_dims: SectorDims) -> SectorDimItems:
    if not isinstance(sector_dims, Mapping):
        raise TypeError("space() requires sector_dims to be a mapping")

    items: list[SectorDimItem] = []
    for key, dim in sector_dims.items():
        sector = _normalize_sector_key(key)
        if isinstance(dim, bool) or not isinstance(dim, int):
            raise TypeError("sector dimension must be an int")
        if dim < 0:
            raise ValueError("sector dimension must be non-negative")
        items.append((sector, dim))
    return tuple(items)
