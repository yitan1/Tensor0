from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, SupportsFloat, overload

from jax import Array
import jax.numpy as jnp

from .. import _native
from ..structure.layout import get_blockstructure, get_sectorstructure
from ..structure.sector_dict import SectorDict
from ..structure.spaces import _as_product_space_input, hom
from ._blocks import pack_blocks
from .diagonal import DiagonalTensorMap
from .tensor_map import TensorMap


def diagm(
    codomain: _native.ElementarySpace | _native.ProductSpace,
    domain: _native.ElementarySpace | _native.ProductSpace,
    values: Iterable[tuple[object, object]] | Mapping[Any, object],
) -> TensorMap:
    result_space = hom(
        _as_product_space_input(codomain, "codomain", "diagm"),
        _as_product_space_input(domain, "domain", "diagm"),
    )
    try:
        raw_value_blocks = SectorDict(values)
    except TypeError as error:
        raise TypeError("diagm() requires values to be a mapping or iterable") from error
    value_items = tuple(
        (coupled, jnp.asarray(value)) for coupled, value in raw_value_blocks.items()
    )
    dtype = (
        jnp.result_type(*(value for _coupled, value in value_items))
        if value_items
        else None
    )
    value_blocks = SectorDict(
        (coupled, jnp.asarray(value, dtype=dtype)) for coupled, value in value_items
    )
    blockstructures = get_blockstructure(result_space)
    block_arrays: list[tuple[tuple[int, ...], Array]] = []
    for coupled, block in blockstructures.items():
        sector_values = value_blocks.get(coupled)
        if sector_values is None:
            raise ValueError(f"missing data for diagonal block sector {coupled}")
        expected_shape = (min(block.row_dim, block.col_dim),)
        actual_shape = tuple(sector_values.shape)
        if actual_shape != expected_shape:
            raise ValueError(
                f"diagonal block sector {coupled} has shape {actual_shape}; "
                f"expected {expected_shape}",
            )
        block_array = jnp.zeros((block.row_dim, block.col_dim), dtype=dtype)
        row_indices = jnp.arange(sector_values.shape[0])
        block_array = block_array.at[row_indices, row_indices].set(sector_values)
        block_arrays.append((coupled, block_array))

    for coupled, sector_values in value_blocks.items():
        if coupled in blockstructures:
            continue
        if sector_values.size != 0:
            raise ValueError(f"unexpected diagonal block sector {coupled}")

    return TensorMap(
        result_space,
        pack_blocks(result_space, SectorDict(block_arrays), dtype=dtype),
    )


@overload
def zero_like(tensor: TensorMap) -> TensorMap: ...


@overload
def zero_like(tensor: DiagonalTensorMap) -> DiagonalTensorMap: ...


def zero_like(
    tensor: TensorMap | DiagonalTensorMap,
) -> TensorMap | DiagonalTensorMap:
    _require_tensor_like(tensor, "zero_like")
    return tensor.zero_like()


@overload
def scale(tensor: TensorMap, alpha: object) -> TensorMap: ...


@overload
def scale(tensor: DiagonalTensorMap, alpha: object) -> DiagonalTensorMap: ...


def scale(
    tensor: TensorMap | DiagonalTensorMap,
    alpha: object,
) -> TensorMap | DiagonalTensorMap:
    _require_tensor_like(tensor, "scale")
    return tensor.scale(alpha)


@overload
def add(
    t1: TensorMap,
    t2: TensorMap,
    alpha: object = 1,
    beta: object = 1,
) -> TensorMap: ...


@overload
def add(
    t1: DiagonalTensorMap,
    t2: DiagonalTensorMap,
    alpha: object = 1,
    beta: object = 1,
) -> DiagonalTensorMap: ...


def add(
    t1: TensorMap | DiagonalTensorMap,
    t2: TensorMap | DiagonalTensorMap,
    alpha: object = 1,
    beta: object = 1,
) -> TensorMap | DiagonalTensorMap:
    if isinstance(t1, TensorMap):
        if not isinstance(t2, TensorMap):
            raise TypeError("add() requires a TensorMap")
        return t1.add(t2, alpha=alpha, beta=beta)
    if isinstance(t1, DiagonalTensorMap):
        if not isinstance(t2, DiagonalTensorMap):
            raise TypeError("add() requires a DiagonalTensorMap")
        return t1.add(t2, alpha=alpha, beta=beta)
    raise TypeError("add() requires a TensorMap or DiagonalTensorMap")


def inner(t1: TensorMap, t2: TensorMap) -> Array:
    _require_tensor_map(t1, "inner")
    return t1.inner(t2)


def dot(t1: TensorMap, t2: TensorMap) -> Array:
    _require_tensor_map(t1, "dot")
    return t1.dot(t2)


def norm(
    tensor: TensorMap | DiagonalTensorMap,
    p: SupportsFloat = 2,
) -> Array:
    _require_tensor_like(tensor, "norm")
    return tensor.norm(p=p)


def normalize(tensor: TensorMap, p: SupportsFloat = 2) -> TensorMap:
    _require_tensor_map(tensor, "normalize")
    return tensor.normalize(p=p)


def scalar(tensor: TensorMap) -> Array:
    _require_tensor_map(tensor, "scalar")
    return tensor.scalar()


@overload
def adjoint(tensor: TensorMap) -> TensorMap: ...


@overload
def adjoint(tensor: DiagonalTensorMap) -> DiagonalTensorMap: ...


def adjoint(
    tensor: TensorMap | DiagonalTensorMap,
) -> TensorMap | DiagonalTensorMap:
    _require_tensor_like(tensor, "adjoint")
    return tensor.adjoint()


@overload
def real(tensor: TensorMap) -> TensorMap: ...


@overload
def real(tensor: DiagonalTensorMap) -> DiagonalTensorMap: ...


def real(
    tensor: TensorMap | DiagonalTensorMap,
) -> TensorMap | DiagonalTensorMap:
    _require_tensor_like(tensor, "real")
    return tensor.real()


@overload
def imag(tensor: TensorMap) -> TensorMap: ...


@overload
def imag(tensor: DiagonalTensorMap) -> DiagonalTensorMap: ...


def imag(
    tensor: TensorMap | DiagonalTensorMap,
) -> TensorMap | DiagonalTensorMap:
    _require_tensor_like(tensor, "imag")
    return tensor.imag()


@overload
def complex(tensor: TensorMap) -> TensorMap: ...


@overload
def complex(tensor: DiagonalTensorMap) -> DiagonalTensorMap: ...


def complex(
    tensor: TensorMap | DiagonalTensorMap,
) -> TensorMap | DiagonalTensorMap:
    _require_tensor_like(tensor, "complex")
    return tensor.complex()


def tr(tensor: TensorMap) -> Array:
    _require_tensor_map(tensor, "tr")
    return tensor.tr()


def diag(tensor: TensorMap) -> SectorDict[Array]:
    _require_tensor_map(tensor, "diag")
    return tensor.diag()


def is_diagonal(tensor: TensorMap | DiagonalTensorMap) -> Array:
    _require_tensor_like(tensor, "is_diagonal")
    return tensor.is_diagonal()


def _compose(left: TensorMap, right: TensorMap) -> TensorMap:
    """Compose two TensorMaps blockwise over their shared middle space."""
    if left.space.domain != right.space.codomain:
        raise ValueError(
            "TensorMap spaces are not composable: "
            "left domain must equal right codomain",
        )

    result_space = hom(left.space.codomain, right.space.domain)
    dtype = jnp.result_type(left.storage.data, right.storage.data)
    left_blocks = SectorDict(left.blocks())
    right_blocks = SectorDict(right.blocks())
    result_blocks: list[tuple[tuple[int, ...], Array]] = []

    for coupled in get_sectorstructure(result_space).blocksectors:
        left_block = left_blocks.get(coupled)
        right_block = right_blocks.get(coupled)
        if left_block is not None and right_block is not None:
            result_blocks.append((coupled, left_block @ right_block))

    return TensorMap(
        result_space,
        pack_blocks(
            result_space,
            SectorDict(result_blocks),
            dtype=dtype,
        ),
    )


def tensor_product(left: TensorMap, right: TensorMap) -> TensorMap:
    """Return the disconnected tensor-network product of two tensor maps."""
    if not isinstance(left, TensorMap) or not isinstance(right, TensorMap):
        raise TypeError(
            "tensor_product() requires left and right to be TensorMap instances",
        )

    output = (
        tuple((0, axis) for axis in left.codomainind)
        + tuple((1, axis) for axis in right.codomainind),
        tuple((0, axis) for axis in left.domainind)
        + tuple((1, axis) for axis in right.domainind),
    )
    from ..operations.contractions.primitives import tensorcontract

    return tensorcontract(
        left,
        right,
        axes=((), ()),
        output=output,
    )


def equal(
    t1: TensorMap | DiagonalTensorMap,
    t2: TensorMap | DiagonalTensorMap,
) -> Array:
    _require_tensor_like(t1, "equal")
    _require_tensor_like(t2, "equal")
    if t1.space != t2.space or t1.dtype != t2.dtype:
        return jnp.asarray(False)

    left, right = _comparison_storage(t1, t2)
    return jnp.array_equal(left, right)


def allclose(
    t1: TensorMap | DiagonalTensorMap,
    t2: TensorMap | DiagonalTensorMap,
    *,
    rtol: float,
    atol: float,
) -> Array:
    _require_tensor_like(t1, "allclose")
    _require_tensor_like(t2, "allclose")
    if t1.space != t2.space:
        return jnp.asarray(False)

    left, right = _comparison_storage(t1, t2)
    return jnp.allclose(left, right, rtol=rtol, atol=atol)


def _require_tensor_map(tensor: object, function_name: str) -> None:
    if not isinstance(tensor, TensorMap):
        raise TypeError(f"{function_name}() requires a TensorMap")


def _require_tensor_like(tensor: object, function_name: str) -> None:
    if not isinstance(tensor, (TensorMap, DiagonalTensorMap)):
        raise TypeError(
            f"{function_name}() requires a TensorMap or DiagonalTensorMap",
        )


def _comparison_storage(
    t1: TensorMap | DiagonalTensorMap,
    t2: TensorMap | DiagonalTensorMap,
) -> tuple[Array, Array]:
    if isinstance(t1, DiagonalTensorMap) == isinstance(t2, DiagonalTensorMap):
        return t1.storage.data, t2.storage.data

    left = t1.to_tensor_map() if isinstance(t1, DiagonalTensorMap) else t1
    right = t2.to_tensor_map() if isinstance(t2, DiagonalTensorMap) else t2
    return left.storage.data, right.storage.data
