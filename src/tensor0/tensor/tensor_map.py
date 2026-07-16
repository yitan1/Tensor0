from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, SupportsFloat, TypeAlias, overload

import jax.numpy as jnp
from jax import Array
from jax import tree_util as _tree_util
from jax.typing import DTypeLike

from .. import _native
from ..structure.layout import (
    get_blockstructure,
    get_degeneracystructure,
    get_sectorstructure,
)
from ..structure.sector_dict import SectorDict
from ..structure.sector_type import SectorKey
from ..structure.spaces import (
    _as_hom_space,
    _as_product_space_input,
    _normalize_sector_key,
    hom,
)
from ._blocks import (
    get_subblock as _get_subblock,
    normalize_fusiontree_pair_key as _normalize_fusiontree_pair_key,
    pack_blocks as _pack_blocks,
    pack_complete_blocks as _pack_complete_blocks,
)
from .diagonal import DiagonalTensorMap
from .storage import VectorStorage, _validate_vector_storage_data

_FusionTreePair: TypeAlias = tuple[_native.FusionTree, _native.FusionTree]
_VisibleSectorTuple: TypeAlias = tuple[SectorKey, ...]
_MISSING = object()


@dataclass(frozen=True, eq=False, init=False)
class TensorMap:
    space: _native.HomSpace
    storage: VectorStorage

    def __init__(self, space: _native.HomSpace, storage: object) -> None:
        if not isinstance(space, _native.HomSpace):
            raise TypeError("TensorMap requires a HomSpace")

        vector_storage = storage
        if not isinstance(vector_storage, VectorStorage):
            vector_storage = VectorStorage(vector_storage)

        degeneracystructure = get_degeneracystructure(space)
        _validate_vector_storage_data(
            vector_storage.data, degeneracystructure.total_dim
        )

        object.__setattr__(self, "space", space)
        object.__setattr__(self, "storage", vector_storage)

    def __repr__(self) -> str:
        dtype = getattr(self.storage.data, "dtype", "unknown")
        return (
            "TensorMap("
            f"numout={self.numout}, "
            f"numin={self.numin}, "
            f"dims={self.dims}, "
            f"blocks={len(self.blocksectors)}, "
            f"dtype={dtype}"
            ")"
        )

    def copy(self) -> TensorMap:
        return TensorMap(self.space, jnp.array(self.storage.data, copy=True))

    def astype(self, dtype: DTypeLike) -> TensorMap:
        return TensorMap(self.space, self.storage.data.astype(dtype))

    def similar(
        self,
        space: _native.HomSpace | None = None,
        dtype: DTypeLike | None = None,
    ) -> TensorMap:
        target_space = self.space if space is None else space
        target_dtype = self.storage.data.dtype if dtype is None else dtype
        return zeros(target_space, dtype=target_dtype)

    @property
    def codomain(self) -> _native.ProductSpace:
        return self.space.codomain

    @property
    def domain(self) -> _native.ProductSpace:
        return self.space.domain

    @property
    def numout(self) -> int:
        return self.space.numout

    @property
    def numin(self) -> int:
        return self.space.numin

    @property
    def numind(self) -> int:
        return self.space.numind

    @property
    def codomainind(self) -> tuple[int, ...]:
        return tuple(range(self.space.numout))

    @property
    def domainind(self) -> tuple[int, ...]:
        return tuple(range(self.space.numout, self.space.numind))

    @property
    def allind(self) -> tuple[int, ...]:
        return tuple(range(self.space.numind))

    @property
    def ndim(self) -> int:
        return self.numind

    @property
    def output_axes(self) -> tuple[int, ...]:
        return self.codomainind

    @property
    def input_axes(self) -> tuple[int, ...]:
        return self.domainind

    @property
    def axes(self) -> tuple[int, ...]:
        return self.allind

    @property
    def dim(self) -> int:
        return get_degeneracystructure(self.space).total_dim

    @property
    def dims(self) -> tuple[int, ...]:
        return tuple(_native.product_dims(self.space.codomain)) + tuple(
            _native.product_dims(self.space.domain),
        )

    @property
    def shape(self) -> tuple[int, ...]:
        return self.dims

    @property
    def dtype(self) -> jnp.dtype:
        return jnp.asarray(self.storage.data).dtype

    @property
    def blocksectors(self) -> tuple[tuple[int, ...], ...]:
        return get_sectorstructure(self.space).blocksectors

    @property
    def block_sectors(self) -> tuple[tuple[int, ...], ...]:
        return self.blocksectors

    @property
    def fusiontrees(
        self,
    ) -> tuple[tuple[_native.FusionTree, _native.FusionTree], ...]:
        return get_sectorstructure(self.space).fusiontree_pairs

    def hasblock(self, coupled: object) -> bool:
        return get_sectorstructure(self.space).blocksector_index(coupled) is not None

    @classmethod
    def from_dense(
        cls,
        space: _native.HomSpace,
        data: object,
        *,
        tol: float | None = None,
    ) -> TensorMap:
        from .dense import from_dense

        return from_dense(space, data, tol=tol)

    @classmethod
    def zeros(
        cls,
        space: _native.HomSpace,
        dtype: DTypeLike | None = None,
    ) -> TensorMap:
        return zeros(space, dtype=dtype)

    @classmethod
    def ones(
        cls,
        space: _native.HomSpace,
        dtype: DTypeLike | None = None,
    ) -> TensorMap:
        return ones(space, dtype=dtype)

    @classmethod
    def from_blocks(
        cls,
        space: _native.HomSpace,
        blocks: Iterable[tuple[object, object]] | Mapping[Any, object],
        dtype: DTypeLike | None = None,
    ) -> TensorMap:
        return from_blocks(space, blocks, dtype=dtype)

    def block(self, coupled: int | tuple[int, ...]) -> Array:
        key = _normalize_sector_key(coupled)
        try:
            block = get_blockstructure(self.space)[key]
        except KeyError:
            raise KeyError(key) from None
        return self.storage.data[block.start : block.stop].reshape(
            (block.row_dim, block.col_dim),
        )

    def blocks(self) -> tuple[tuple[tuple[int, ...], Array], ...]:
        return tuple(
            (
                coupled,
                self.storage.data[block.start : block.stop].reshape(
                    (block.row_dim, block.col_dim),
                ),
            )
            for coupled, block in get_blockstructure(self.space).items()
        )

    @overload
    def subblock(
        self,
        key: _native.FusionTree,
        col_tree: _native.FusionTree,
    ) -> Array: ...

    @overload
    def subblock(
        self,
        key: _FusionTreePair | _VisibleSectorTuple,
    ) -> Array: ...

    def subblock(self, key: object, col_tree: object = _MISSING) -> Array:
        if col_tree is not _MISSING:
            if not isinstance(key, _native.FusionTree) or not isinstance(
                col_tree,
                _native.FusionTree,
            ):
                raise TypeError(
                    "subblock() with two arguments requires a pair of FusionTree objects"
                )
            key = (key, col_tree)
        return self._subblock_at(_resolve_subblock_index(self.space, key))

    def _subblock_at(self, index: int) -> Array:
        subblock = get_degeneracystructure(self.space).subblock_at(index)
        if subblock is None:
            raise KeyError(index)
        return _get_subblock(self.storage.data, subblock)

    def subblocks(
        self,
    ) -> _SubblocksView:
        return _SubblocksView(
            self,
            get_sectorstructure(self.space),
            get_degeneracystructure(self.space),
        )

    def __getitem__(self, key: _FusionTreePair | _VisibleSectorTuple) -> Array:
        return self._subblock_at(_resolve_subblock_index(self.space, key))

    def to_dense(self) -> Array:
        from .dense import to_dense

        return to_dense(self)

    def scalar(self) -> Array:
        if self.space.numind != 0:
            raise ValueError("scalar() requires a TensorMap with no visible indices")
        return self.storage.data[0]

    def permute(self, p: tuple[tuple[int, ...], tuple[int, ...]]) -> TensorMap:
        from ..operations.transforms import permute

        return permute(self, p)

    def braid(
        self,
        p: tuple[tuple[int, ...], tuple[int, ...]],
        levels: tuple[int, ...],
    ) -> TensorMap:
        from ..operations.transforms import braid

        return braid(self, p, levels)

    def transpose(
        self,
        p: tuple[tuple[int, ...], tuple[int, ...]] | None = None,
    ) -> TensorMap:
        from ..operations.transforms import transpose

        return transpose(self, p)

    def repartition(self, nout: int, nin: int | None = None) -> TensorMap:
        from ..operations.transforms import repartition

        return repartition(self, nout, nin)

    def flip(
        self,
        indices: int | tuple[int, ...],
        inv: bool = False,
    ) -> TensorMap:
        from ..operations.transforms import flip

        return flip(self, indices, inv=inv)

    def twist(
        self,
        indices: int | tuple[int, ...],
        inv: bool = False,
    ) -> TensorMap:
        from ..operations.transforms import twist

        return twist(self, indices, inv=inv)

    def insertleftunit(
        self,
        position: int | None = None,
        *,
        dual: bool = False,
    ) -> TensorMap:
        from ..operations.transforms import insertleftunit

        return insertleftunit(self, position, dual=dual)

    def insertrightunit(
        self,
        position: int | None = None,
        *,
        dual: bool = False,
    ) -> TensorMap:
        from ..operations.transforms import insertrightunit

        return insertrightunit(self, position, dual=dual)

    def removeunit(self, index: int) -> TensorMap:
        from ..operations.transforms import removeunit

        return removeunit(self, index)

    def zero_like(self) -> TensorMap:
        return TensorMap(self.space, jnp.zeros_like(self.storage.data))

    def scale(self, alpha: object) -> TensorMap:
        scalar = _as_scalar_array(alpha, "alpha")
        dtype = jnp.result_type(self.storage.data, scalar)
        return TensorMap(
            self.space, jnp.asarray(self.storage.data * scalar, dtype=dtype)
        )

    def add(
        self,
        other: TensorMap,
        alpha: object = 1,
        beta: object = 1,
    ) -> TensorMap:
        _require_tensor_map(other, "add")
        if self.space != other.space:
            raise ValueError("TensorMap spaces are not compatible for add")

        alpha_scalar = _as_scalar_array(alpha, "alpha")
        beta_scalar = _as_scalar_array(beta, "beta")
        dtype = jnp.result_type(
            self.storage.data,
            other.storage.data,
            alpha_scalar,
            beta_scalar,
        )
        data = alpha_scalar * self.storage.data + beta_scalar * other.storage.data
        return TensorMap(self.space, jnp.asarray(data, dtype=dtype))

    def inner(self, other: TensorMap) -> Array:
        _require_tensor_map(other, "inner")
        if self.space != other.space:
            raise ValueError("TensorMap spaces are not compatible for inner")

        total = jnp.asarray(
            0, dtype=jnp.result_type(self.storage.data, other.storage.data)
        )
        sector_type = self.space.codomain.sector_spec
        for coupled, left in self.blocks():
            weight = sector_type.quantum_dim(coupled)
            total = total + weight * jnp.vdot(left, other.block(coupled))
        return total

    def dot(self, other: TensorMap) -> Array:
        return self.inner(other)

    def norm(self, p: SupportsFloat = 2) -> Array:
        if p == jnp.inf or p == float("inf"):
            return _max_abs_block_entry(self)

        p_value = float(p)
        if not math.isfinite(p_value) or p_value <= 0:
            raise ValueError("norm() requires positive finite p or inf")

        if p_value == 2.0:
            return jnp.sqrt(jnp.real(self.inner(self)))

        total = jnp.asarray(
            0,
            dtype=jnp.abs(jnp.asarray(self.storage.data).reshape(-1)[:0]).dtype,
        )
        sector_type = self.space.codomain.sector_spec
        for coupled, block in self.blocks():
            weight = sector_type.quantum_dim(coupled)
            total = total + weight * jnp.sum(jnp.abs(block) ** p_value)
        return total ** (1.0 / p_value)

    def normalized(self, p: SupportsFloat = 2) -> TensorMap:
        return self.normalize(p=p)

    def normalize(self, p: SupportsFloat = 2) -> TensorMap:
        return self.scale(1 / self.norm(p=p))

    def adjoint(self) -> TensorMap:
        result_space = hom(self.space.domain, self.space.codomain)
        block_arrays = SectorDict(
            (coupled, jnp.conj(block.T)) for coupled, block in self.blocks()
        )
        return TensorMap(
            result_space,
            _pack_blocks(
                result_space,
                block_arrays,
                dtype=None,
            ),
        )

    def real(self) -> TensorMap:
        if not jnp.issubdtype(self.storage.data.dtype, jnp.complexfloating):
            return self
        return TensorMap(self.space, jnp.real(self.storage.data))

    def imag(self) -> TensorMap:
        if not jnp.issubdtype(self.storage.data.dtype, jnp.complexfloating):
            return self.zero_like()
        return TensorMap(self.space, jnp.imag(self.storage.data))

    def complex(self) -> TensorMap:
        if jnp.issubdtype(self.storage.data.dtype, jnp.complexfloating):
            return self
        return TensorMap(
            self.space,
            self.storage.data.astype(jnp.result_type(self.storage.data, 1j)),
        )

    def trace(self) -> Array:
        return self.tr()

    def tr(self) -> Array:
        if self.space.domain != self.space.codomain:
            raise ValueError(
                "trace requires a square TensorMap with equal domain and codomain"
            )

        total = jnp.asarray(0, dtype=self.storage.data.dtype)
        sector_type = self.space.codomain.sector_spec
        for coupled, block in self.blocks():
            weight = sector_type.quantum_dim(coupled)
            total = total + weight * jnp.trace(block)
        return total

    def diag(self) -> SectorDict[Array]:
        return SectorDict((coupled, jnp.diag(block)) for coupled, block in self.blocks())

    def is_diagonal(self) -> Array:
        result = jnp.asarray(True)
        for _coupled, block in self.blocks():
            diagonal = jnp.zeros_like(block)
            diagonal_indices = jnp.arange(min(block.shape))
            diagonal = diagonal.at[diagonal_indices, diagonal_indices].set(
                block[diagonal_indices, diagonal_indices],
            )
            result = jnp.logical_and(result, jnp.all(block == diagonal))
        return result

    def __matmul__(self, other: object) -> TensorMap:
        if isinstance(other, DiagonalTensorMap):
            if self.space.domain != other.space.codomain:
                raise ValueError(
                    "TensorMap spaces are not composable: "
                    "left domain must equal right codomain",
                )
            return self @ other.to_tensor_map()

        if not isinstance(other, TensorMap):
            return NotImplemented

        if self.space.domain != other.space.codomain:
            raise ValueError(
                "TensorMap spaces are not composable: "
                "left domain must equal right codomain",
            )

        result_space = hom(self.space.codomain, other.space.domain)
        dtype = jnp.result_type(self.storage.data, other.storage.data)
        left_blocks = SectorDict(self.blocks())
        right_blocks = SectorDict(other.blocks())
        result_blocks: list[tuple[tuple[int, ...], Array]] = []

        for coupled in get_sectorstructure(result_space).blocksectors:
            left = left_blocks.get(coupled)
            right = right_blocks.get(coupled)
            if left is not None and right is not None:
                result_blocks.append((coupled, left @ right))

        return TensorMap(
            result_space,
            _pack_blocks(
                result_space,
                SectorDict(result_blocks),
                dtype=dtype,
            ),
        )

    def __neg__(self) -> TensorMap:
        return self.scale(-1)

    def __add__(self, other: object) -> TensorMap:
        if not isinstance(other, TensorMap):
            return NotImplemented
        return self.add(other)

    def __sub__(self, other: object) -> TensorMap:
        if not isinstance(other, TensorMap):
            return NotImplemented
        return self.add(other, alpha=1, beta=-1)

    def __mul__(self, other: object) -> TensorMap:
        if isinstance(other, TensorMap):
            return NotImplemented
        return self.scale(other)

    def __rmul__(self, other: object) -> TensorMap:
        if isinstance(other, TensorMap):
            return NotImplemented
        return self.scale(other)

    def __truediv__(self, other: object) -> TensorMap:
        return self.scale(1 / jnp.asarray(other))


@dataclass(frozen=True)
class _SubblocksView:
    tensor: TensorMap
    _sectorstructure: _native.SectorStructure
    _degeneracystructure: _native.DegeneracyStructure

    def __len__(self) -> int:
        return self._sectorstructure.fusiontree_pair_count

    @overload
    def __getitem__(self, index: int) -> tuple[_FusionTreePair, Array]: ...

    @overload
    def __getitem__(
        self,
        index: slice,
    ) -> tuple[tuple[_FusionTreePair, Array], ...]: ...

    def __getitem__(
        self,
        index: int | slice,
    ) -> tuple[_FusionTreePair, Array] | tuple[tuple[_FusionTreePair, Array], ...]:
        if isinstance(index, slice):
            return tuple(self[item] for item in range(*index.indices(len(self))))

        length = len(self)
        if index < 0:
            index += length
        if index < 0 or index >= length:
            raise IndexError("subblock index out of range")

        pair = self._sectorstructure.fusiontree_pair_at(index)
        subblock = self._degeneracystructure.subblock_at(index)
        if pair is None or subblock is None:
            raise RuntimeError("sector and degeneracy structures are inconsistent")
        return pair, _get_subblock(self.tensor.storage.data, subblock)

    def __iter__(
        self,
    ) -> Iterator[tuple[_FusionTreePair, Array]]:
        for index in range(len(self)):
            yield self[index]


def _resolve_subblock_index(
    space: _native.HomSpace,
    key: object,
) -> int:
    sectorstructure = get_sectorstructure(space)
    if isinstance(key, tuple) and any(
        isinstance(value, _native.FusionTree) for value in key
    ):
        row_tree, col_tree = _normalize_fusiontree_pair_key(key)
        index = sectorstructure.fusiontree_pair_index(row_tree, col_tree)
    else:
        if not isinstance(key, tuple):
            raise TypeError(
                "TensorMap indices must be a pair of FusionTree objects "
                "or a tuple of sectors"
            )
        index = _native.unique_fusiontree_pair_index(space, sectorstructure, key)

    if index is None:
        raise KeyError(key)
    return index


def zeros(space: _native.HomSpace, dtype: DTypeLike | None = None) -> TensorMap:
    space = _as_hom_space(space, "zeros")
    degeneracystructure = get_degeneracystructure(space)
    return TensorMap(
        space,
        jnp.zeros((degeneracystructure.total_dim,), dtype=dtype),
    )


def ones(space: _native.HomSpace, dtype: DTypeLike | None = None) -> TensorMap:
    space = _as_hom_space(space, "ones")
    degeneracystructure = get_degeneracystructure(space)
    return TensorMap(
        space,
        jnp.ones((degeneracystructure.total_dim,), dtype=dtype),
    )


def from_blocks(
    space: _native.HomSpace,
    blocks: Iterable[tuple[object, object]] | Mapping[Any, object],
    dtype: DTypeLike | None = None,
) -> TensorMap:
    space = _as_hom_space(space, "from_blocks")
    return TensorMap(
        space,
        _pack_complete_blocks(space, blocks, dtype=dtype),
    )


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
    dtype = jnp.result_type(*(value for _coupled, value in value_items)) if value_items else None
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
        _pack_blocks(result_space, SectorDict(block_arrays), dtype=dtype),
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


def isdiag(tensor: TensorMap | DiagonalTensorMap) -> Array:
    _require_tensor_like(tensor, "isdiag")
    return tensor.is_diagonal()


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


def _as_scalar_array(value: object, argument_name: str) -> Array:
    scalar = jnp.asarray(value)
    if scalar.shape != ():
        raise TypeError(f"{argument_name} must be a scalar")
    return scalar


def _max_abs_block_entry(tensor: TensorMap) -> Array:
    max_value = jnp.asarray(
        0,
        dtype=jnp.abs(jnp.asarray(tensor.storage.data).reshape(-1)[:0]).dtype,
    )
    for _coupled, block in tensor.blocks():
        block_abs = jnp.max(jnp.abs(block))
        max_value = jnp.maximum(max_value, block_abs)
    return max_value


def _tensormap_flatten(
    tensor: TensorMap,
) -> tuple[tuple[object, ...], _native.HomSpace]:
    return (tensor.storage.data,), tensor.space


def _tensormap_unflatten(
    aux_data: _native.HomSpace,
    children: tuple[object, ...],
) -> TensorMap:
    (data,) = children
    return TensorMap(aux_data, data)


_tree_util.register_pytree_node(
    TensorMap,
    _tensormap_flatten,
    _tensormap_unflatten,
)
