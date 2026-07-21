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
    _blockstructure_items,
    _find_blockstructure,
    get_degeneracystructure,
    get_sectorstructure,
)
from ..structure.sector_dict import SectorDict
from ..structure.sector_type import SectorKey
from ..structure.spaces import (
    _as_hom_space,
    _normalize_sector_key,
    hom,
    storage_dim,
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

        expected_total_dim = storage_dim(space)
        _validate_vector_storage_data(vector_storage.data, expected_total_dim)

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
        return storage_dim(self.space)

    @property
    def dims(self) -> tuple[int, ...]:
        return self.space.dims

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
        if self.space.sector_spec == _native.Trivial:
            key = _normalize_sector_key(coupled)
            self.space.sector_spec.quantum_dim(key)
            return self.storage.data.shape[0] != 0
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
        if self.space.sector_spec == _native.Trivial:
            if key != ():
                self.space.sector_spec.quantum_dim(key)
                raise KeyError(key)
            from .dense import _trivial_dense_array

            value = _trivial_dense_array(self)
            if value.size == 0:
                raise KeyError(key)
            row_dim = math.prod(value.shape[: self.numout])
            col_dim = math.prod(value.shape[self.numout :])
            return value.reshape((row_dim, col_dim))

        block = _find_blockstructure(self.space, key)
        if block is None:
            raise KeyError(key) from None
        return self.storage.data[block.start : block.stop].reshape(
            (block.row_dim, block.col_dim),
        )

    def blocks(self) -> tuple[tuple[tuple[int, ...], Array], ...]:
        if self.space.sector_spec == _native.Trivial:
            from .dense import _trivial_dense_array

            value = _trivial_dense_array(self)
            if value.size == 0:
                return ()
            row_dim = math.prod(value.shape[: self.numout])
            col_dim = math.prod(value.shape[self.numout :])
            return (((), value.reshape((row_dim, col_dim))),)

        return tuple(
            (
                coupled,
                self.storage.data[block.start : block.stop].reshape(
                    (block.row_dim, block.col_dim),
                ),
            )
            for coupled, block in _blockstructure_items(self.space)
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
        if self.space.sector_spec == _native.Trivial:
            if index != 0:
                raise KeyError(index)
            from .dense import _trivial_dense_array

            value = _trivial_dense_array(self)
            if value.size == 0:
                raise KeyError(index)
            return value

        subblock = get_degeneracystructure(self.space).subblock_at(index)
        if subblock is None:
            raise KeyError(index)
        return _get_subblock(self.storage.data, subblock)

    def subblocks(
        self,
    ) -> _SubblocksView:
        sectorstructure = get_sectorstructure(self.space)
        return _SubblocksView(
            self,
            sectorstructure,
            None
            if self.space.sector_spec == _native.Trivial
            else get_degeneracystructure(self.space),
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
        sector_type = self.space.sector_spec
        for (coupled, left), (_other_coupled, right) in zip(
            self.blocks(),
            other.blocks(),
            strict=True,
        ):
            weight = sector_type.quantum_dim(coupled)
            total = total + weight * jnp.vdot(left, right)
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
        sector_type = self.space.sector_spec
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
        sector_type = self.space.sector_spec
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

    def inverse(self) -> TensorMap:
        from .linalg import inverse

        return inverse(self)

    def pseudoinverse(
        self,
        *,
        atol: float = 0.0,
        rtol: float | None = None,
    ) -> TensorMap:
        from .linalg import pseudoinverse

        return pseudoinverse(self, atol=atol, rtol=rtol)

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
        from .linalg import _compose

        return _compose(self, other)

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
    _degeneracystructure: _native.DegeneracyStructure | None

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
        if self._degeneracystructure is None:
            if pair is None:
                raise RuntimeError("sector and degeneracy structures are inconsistent")
            from .dense import _trivial_dense_array

            return pair, _trivial_dense_array(self.tensor)

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
    return TensorMap(
        space,
        jnp.zeros((storage_dim(space),), dtype=dtype),
    )


def ones(space: _native.HomSpace, dtype: DTypeLike | None = None) -> TensorMap:
    space = _as_hom_space(space, "ones")
    return TensorMap(
        space,
        jnp.ones((storage_dim(space),), dtype=dtype),
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


def _require_tensor_map(tensor: object, function_name: str) -> None:
    if not isinstance(tensor, TensorMap):
        raise TypeError(f"{function_name}() requires a TensorMap")


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
