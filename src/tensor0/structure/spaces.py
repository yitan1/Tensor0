from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import sys
from typing import Any

from .. import _native
from .layout import get_degeneracystructure
from .sector_type import SectorType, _normalize_sector_key

SectorDims = Mapping[Any, int]
SectorValue = tuple[int, ...]
SectorDimItem = tuple[SectorValue, int]
SectorDimItems = tuple[SectorDimItem, ...]
ProductSpaceInput = Iterable[_native.ElementarySpace] | _native.ProductSpace
_DimSpace = _native.ElementarySpace | _native.ProductSpace
_SectorSpace = _native.ElementarySpace | _native.ProductSpace | _native.HomSpace
_USIZE_MAX = 2 * sys.maxsize + 1


@dataclass(frozen=True)
class _VectBuilder:
    sector_type: SectorType

    def __call__(
        self,
        sector_dims: SectorDims | int,
        *,
        dual: bool = False,
    ) -> _native.ElementarySpace:
        if isinstance(sector_dims, Mapping):
            return space(self.sector_type, sector_dims, dual=dual)
        if self.sector_type != _native.Trivial:
            raise TypeError(
                "mapping-free Vect construction requires the Trivial sector type",
            )
        return Vect(sector_dims, dual=dual)


class _VectFactory:
    def __call__(
        self,
        dim: int = 0,
        *,
        dual: bool = False,
    ) -> _native.ElementarySpace:
        """Construct an ordinary vector space with no symmetry."""
        return space(_native.Trivial, {(): dim}, dual=dual)

    def __getitem__(self, sector_type: SectorType) -> _VectBuilder:
        if not isinstance(sector_type, SectorType):
            raise TypeError("Vect[...] requires a SectorType")
        return _VectBuilder(sector_type)


Vect = _VectFactory()


def ComplexSpace(
    dim: int = 0,
    *,
    dual: bool = False,
) -> _native.ElementarySpace:
    """Compatibility constructor equivalent to ``Vect(dim, dual=dual)``."""
    return Vect(dim, dual=dual)


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
    *,
    sector_type: SectorType | None = None,
) -> _native.HomSpace:
    if sector_type is not None and not isinstance(sector_type, SectorType):
        raise TypeError("hom() sector_type must be a SectorType")

    codomain_product = _product_space_or_none(codomain)
    domain_product = _product_space_or_none(domain)
    sector_type = _hom_sector_type(
        codomain_product,
        domain_product,
        explicit=sector_type,
    )

    if codomain_product is None:
        codomain_product = _native.make_product_space(sector_type, ())
    if domain_product is None:
        domain_product = _native.make_product_space(sector_type, ())

    return _native.make_hom_products(codomain_product, domain_product)


def dim(value: _DimSpace) -> int:
    """Return the physical dimension of an elementary or product space."""
    if not isinstance(value, (_native.ElementarySpace, _native.ProductSpace)):
        raise TypeError("dim() requires an ElementarySpace or ProductSpace")
    return _native.dim(value)


def sector_spec(value: _SectorSpace) -> SectorType:
    """Return the sector family carried by a typed space."""
    if isinstance(
        value,
        (_native.ElementarySpace, _native.ProductSpace, _native.HomSpace),
    ):
        return value.sector_spec
    raise TypeError(
        "sector_spec() requires an ElementarySpace, ProductSpace, or HomSpace",
    )


def reduced_dim(value: _native.ElementarySpace) -> int:
    """Return the sum of degeneracy dimensions of an elementary space."""
    if not isinstance(value, _native.ElementarySpace):
        raise TypeError("reduced_dim() requires an ElementarySpace")
    return _native.reduced_dim(value)


def storage_dim(value: _native.HomSpace) -> int:
    """Return the packed storage dimension of a morphism space."""
    if not isinstance(value, _native.HomSpace):
        raise TypeError("storage_dim() requires a HomSpace")
    if value.sector_spec != _native.Trivial:
        return get_degeneracystructure(value).total_dim

    dims = value.dims
    if 0 in dims:
        return 0

    result = 1
    for dimension in dims:
        if result > _USIZE_MAX // dimension:
            raise ValueError("degeneracy dimension overflowed")
        result *= dimension
    return result


def fuse(value: _native.ProductSpace) -> _native.ElementarySpace:
    """Fuse a typed product space into one elementary space."""
    if not isinstance(value, _native.ProductSpace):
        raise TypeError("fuse() requires a ProductSpace")
    return _native.fuse(value)


def unit_space(
    sector_type: SectorType,
    *,
    dual: bool = False,
) -> _native.ElementarySpace:
    """Return the canonical one-dimensional space for a sector family."""
    _require_sector_type(sector_type, "unit_space")
    return _native.unit_space(sector_type, dual)


def zero_space(
    sector_type: SectorType,
    *,
    dual: bool = False,
) -> _native.ElementarySpace:
    """Return the zero-dimensional space for a sector family."""
    _require_sector_type(sector_type, "zero_space")
    return _native.zero_space(sector_type, dual)


def infimum(
    left: _native.ElementarySpace,
    right: _native.ElementarySpace,
) -> _native.ElementarySpace:
    """Return the sector-wise infimum of two elementary spaces."""
    _require_elementary_pair(left, right, "infimum")
    return _native.infimum_space(left, right)


def supremum(
    left: _native.ElementarySpace,
    right: _native.ElementarySpace,
) -> _native.ElementarySpace:
    """Return the sector-wise supremum of two elementary spaces."""
    _require_elementary_pair(left, right, "supremum")
    return _native.supremum_space(left, right)


def direct_sum(
    left: _native.ElementarySpace,
    right: _native.ElementarySpace,
) -> _native.ElementarySpace:
    """Return the direct sum of two elementary spaces."""
    _require_elementary_pair(left, right, "direct_sum")
    return _native.direct_sum(left, right)


def is_isomorphic(
    left: _native.ElementarySpace,
    right: _native.ElementarySpace,
) -> bool:
    """Return whether two elementary spaces have equal visible dimensions."""
    _require_elementary_pair(left, right, "is_isomorphic")
    return _native.is_isomorphic(left, right)


def is_monomorphic(
    left: _native.ElementarySpace,
    right: _native.ElementarySpace,
) -> bool:
    """Return whether the left elementary space embeds into the right one."""
    _require_elementary_pair(left, right, "is_monomorphic")
    return _native.is_monomorphic(left, right)


def is_epimorphic(
    left: _native.ElementarySpace,
    right: _native.ElementarySpace,
) -> bool:
    """Return whether the left elementary space surjects onto the right one."""
    _require_elementary_pair(left, right, "is_epimorphic")
    return _native.is_epimorphic(left, right)


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
    *,
    explicit: SectorType | None,
) -> SectorType:
    if explicit is not None:
        for product in (codomain, domain):
            if product is not None and product.sector_spec != explicit:
                raise ValueError(
                    "hom() sector_type must match every non-empty or typed input",
                )
        return explicit

    if codomain is not None:
        return codomain.sector_spec
    if domain is not None:
        return domain.sector_spec
    raise ValueError(
        "hom() requires sector_type when both sides are untyped empty iterables",
    )


def _require_sector_type(value: object, function_name: str) -> None:
    if not isinstance(value, SectorType):
        raise TypeError(f"{function_name}() requires a SectorType")


def _require_elementary_pair(left: object, right: object, function_name: str) -> None:
    if not isinstance(left, _native.ElementarySpace) or not isinstance(
        right,
        _native.ElementarySpace,
    ):
        raise TypeError(f"{function_name}() requires two ElementarySpace values")


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
