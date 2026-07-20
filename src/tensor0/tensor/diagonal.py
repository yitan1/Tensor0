from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, SupportsFloat

from jax import Array
from jax import tree_util as _tree_util
import jax.numpy as jnp
from jax.typing import DTypeLike

from .. import _native
from ..structure.layout import get_sectorstructure
from ..structure.spaces import dim as space_dim, hom, reduced_dim, storage_dim
from ._tolerances import default_pseudoinverse_rtol, nonnegative_tolerance
from .sector_vector import SectorVector
from .storage import VectorStorage, _validate_vector_storage_data

if TYPE_CHECKING:
    from .tensor_map import TensorMap


@dataclass(frozen=True, eq=False, init=False)
class DiagonalTensorMap:
    index_space: _native.ElementarySpace
    storage: VectorStorage

    def __init__(self, index_space: _native.ElementarySpace, storage: object) -> None:
        if not isinstance(index_space, _native.ElementarySpace):
            raise TypeError("DiagonalTensorMap requires an ElementarySpace")

        vector_storage = storage
        if not isinstance(vector_storage, VectorStorage):
            vector_storage = VectorStorage(vector_storage)

        _validate_vector_storage_data(vector_storage.data, reduced_dim(index_space))

        object.__setattr__(self, "index_space", index_space)
        object.__setattr__(self, "storage", vector_storage)

    def __repr__(self) -> str:
        return (
            "DiagonalTensorMap("
            f"dims={self.dims}, "
            f"blocks={len(self.blocksectors)}, "
            f"dtype={self.dtype}"
            ")"
        )

    def copy(self) -> DiagonalTensorMap:
        return DiagonalTensorMap(
            self.index_space,
            jnp.array(self.storage.data, copy=True),
        )

    def astype(self, dtype: DTypeLike) -> DiagonalTensorMap:
        return DiagonalTensorMap(
            self.index_space,
            jnp.asarray(self.storage.data).astype(dtype),
        )

    @property
    def space(self) -> _native.HomSpace:
        return hom((self.index_space,), (self.index_space,))

    @property
    def codomain(self) -> _native.ProductSpace:
        return self.space.codomain

    @property
    def domain(self) -> _native.ProductSpace:
        return self.space.domain

    @property
    def numout(self) -> int:
        return 1

    @property
    def numin(self) -> int:
        return 1

    @property
    def numind(self) -> int:
        return 2

    @property
    def codomainind(self) -> tuple[int, ...]:
        return (0,)

    @property
    def domainind(self) -> tuple[int, ...]:
        return (1,)

    @property
    def allind(self) -> tuple[int, ...]:
        return (0, 1)

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
        index_dim = space_dim(self.index_space)
        return (index_dim, index_dim)

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
        return self.diag().hasblock(coupled)

    def diag(self) -> SectorVector:
        return SectorVector(self.index_space, self.storage)

    def block(self, coupled: int | tuple[int, ...]) -> Array:
        return jnp.diag(self.diag().block(coupled))

    def blocks(self) -> tuple[tuple[tuple[int, ...], Array], ...]:
        return tuple((sector, jnp.diag(values)) for sector, values in self.diag().blocks())

    def to_tensor_map(self) -> TensorMap:
        from ._blocks import pack_blocks
        from .tensor_map import TensorMap

        block_arrays = dict(self.blocks())
        diagonal_space = self.space
        return TensorMap(
            diagonal_space,
            pack_blocks(diagonal_space, block_arrays, dtype=self.dtype),
        )

    def to_dense(self) -> Array:
        return self.to_tensor_map().to_dense()

    def zero_like(self) -> DiagonalTensorMap:
        return DiagonalTensorMap(
            self.index_space,
            jnp.zeros_like(self.storage.data),
        )

    def scale(self, alpha: object) -> DiagonalTensorMap:
        scalar = _as_scalar_array(alpha, "alpha")
        dtype = jnp.result_type(self.storage.data, scalar)
        return DiagonalTensorMap(
            self.index_space,
            jnp.asarray(self.storage.data * scalar, dtype=dtype),
        )

    def add(
        self,
        other: DiagonalTensorMap,
        alpha: object = 1,
        beta: object = 1,
    ) -> DiagonalTensorMap:
        if not isinstance(other, DiagonalTensorMap):
            raise TypeError("add() requires a DiagonalTensorMap")
        if self.index_space != other.index_space:
            raise ValueError("DiagonalTensorMap spaces are not compatible for add")

        alpha_scalar = _as_scalar_array(alpha, "alpha")
        beta_scalar = _as_scalar_array(beta, "beta")
        dtype = jnp.result_type(
            self.storage.data,
            other.storage.data,
            alpha_scalar,
            beta_scalar,
        )
        data = alpha_scalar * self.storage.data + beta_scalar * other.storage.data
        return DiagonalTensorMap(
            self.index_space,
            jnp.asarray(data, dtype=dtype),
        )

    def norm(self, p: SupportsFloat = 2) -> Array:
        p_value = float(p)
        if p_value == math.inf:
            return _max_abs_diagonal_entry(self)
        if not math.isfinite(p_value) or p_value <= 0:
            raise ValueError("norm() requires positive finite p or inf")

        total = jnp.asarray(
            0,
            dtype=jnp.abs(jnp.asarray(self.storage.data).reshape(-1)[:0]).dtype,
        )
        sector_type = self.index_space.sector_spec
        for coupled, values in self.diag().blocks():
            weight = sector_type.quantum_dim(coupled)
            total = total + weight * jnp.sum(jnp.abs(values) ** p_value)
        return total ** (1.0 / p_value)

    def adjoint(self) -> DiagonalTensorMap:
        if not jnp.issubdtype(self.dtype, jnp.complexfloating):
            return self
        return DiagonalTensorMap(
            self.index_space,
            jnp.conj(self.storage.data),
        )

    def real(self) -> DiagonalTensorMap:
        if not jnp.issubdtype(self.dtype, jnp.complexfloating):
            return self
        return DiagonalTensorMap(self.index_space, jnp.real(self.storage.data))

    def imag(self) -> DiagonalTensorMap:
        if not jnp.issubdtype(self.dtype, jnp.complexfloating):
            return self.zero_like()
        return DiagonalTensorMap(self.index_space, jnp.imag(self.storage.data))

    def complex(self) -> DiagonalTensorMap:
        if jnp.issubdtype(self.dtype, jnp.complexfloating):
            return self
        return DiagonalTensorMap(
            self.index_space,
            jnp.asarray(self.storage.data).astype(
                jnp.result_type(self.storage.data, 1j),
            ),
        )

    def inverse(self) -> DiagonalTensorMap:
        values = jnp.asarray(
            self.storage.data,
            dtype=jnp.result_type(self.storage.data, 1.0),
        )
        return DiagonalTensorMap(
            self.index_space,
            jnp.ones_like(values) / values,
        )

    def pseudoinverse(
        self,
        *,
        atol: float = 0.0,
        rtol: float | None = None,
    ) -> DiagonalTensorMap:
        """Return a cutoff pseudoinverse, rejecting values at or below tolerance.

        The cutoff is ``max(atol, rtol * max(abs(values)))``. When ``rtol`` is
        omitted it is ten times the largest reduced block dimension times
        machine epsilon for the promoted real dtype.
        Rejected entries, including exact zeros, map to zero.
        """
        atol_value = nonnegative_tolerance(atol, "atol")
        values = jnp.asarray(
            self.storage.data,
            dtype=jnp.result_type(self.storage.data, 1.0),
        )
        real_dtype = jnp.abs(values).dtype
        if rtol is None:
            largest_block_dim = max(
                (dimension for _sector, dimension in self.index_space.sectors),
                default=0,
            )
            rtol_value = default_pseudoinverse_rtol(
                real_dtype,
                largest_block_dim,
            )
        else:
            rtol_value = nonnegative_tolerance(rtol, "rtol")

        magnitudes = jnp.abs(values)
        max_magnitude = jnp.max(
            jnp.concatenate((jnp.zeros((1,), dtype=real_dtype), magnitudes)),
        )
        cutoff = jnp.maximum(
            jnp.asarray(atol_value, dtype=real_dtype),
            jnp.asarray(rtol_value, dtype=real_dtype) * max_magnitude,
        )
        accepted = magnitudes > cutoff
        safe_values = jnp.where(accepted, values, jnp.ones_like(values))
        data = jnp.where(
            accepted,
            jnp.ones_like(values) / safe_values,
            jnp.zeros_like(values),
        )
        return DiagonalTensorMap(self.index_space, data)

    def is_diagonal(self) -> Array:
        return jnp.asarray(True)

    def __matmul__(self, other: object):
        from .tensor_map import TensorMap

        if isinstance(other, DiagonalTensorMap):
            if self.domain != other.codomain:
                raise ValueError(
                    "TensorMap spaces are not composable: "
                    "left domain must equal right codomain",
                )
            dtype = jnp.result_type(self.storage.data, other.storage.data)
            return DiagonalTensorMap(
                self.index_space,
                jnp.asarray(self.storage.data * other.storage.data, dtype=dtype),
            )

        if not isinstance(other, TensorMap):
            return NotImplemented
        if self.domain != other.space.codomain:
            raise ValueError(
                "TensorMap spaces are not composable: "
                "left domain must equal right codomain",
            )
        return self.to_tensor_map() @ other

    def __neg__(self) -> DiagonalTensorMap:
        return self.scale(-1)

    def __add__(self, other: object) -> DiagonalTensorMap:
        if not isinstance(other, DiagonalTensorMap):
            return NotImplemented
        return self.add(other)

    def __sub__(self, other: object) -> DiagonalTensorMap:
        if not isinstance(other, DiagonalTensorMap):
            return NotImplemented
        return self.add(other, alpha=1, beta=-1)

    def __mul__(self, other: object) -> DiagonalTensorMap:
        if isinstance(other, DiagonalTensorMap):
            return NotImplemented
        return self.scale(other)

    def __rmul__(self, other: object) -> DiagonalTensorMap:
        if isinstance(other, DiagonalTensorMap):
            return NotImplemented
        return self.scale(other)

    def __truediv__(self, other: object) -> DiagonalTensorMap:
        scalar = _as_scalar_array(other, "divisor")
        return self.scale(jnp.ones_like(scalar) / scalar)


def _as_scalar_array(value: object, argument_name: str) -> Array:
    scalar = jnp.asarray(value)
    if scalar.shape != ():
        raise TypeError(f"{argument_name} must be a scalar")
    return scalar


def _max_abs_diagonal_entry(diagonal: DiagonalTensorMap) -> Array:
    dtype = jnp.abs(jnp.asarray(diagonal.storage.data).reshape(-1)[:0]).dtype
    max_value = jnp.asarray(0, dtype=dtype)
    for _coupled, values in diagonal.diag().blocks():
        if values.size > 0:
            max_value = jnp.maximum(max_value, jnp.max(jnp.abs(values)))
    return max_value


def _diagonal_flatten(
    diagonal: DiagonalTensorMap,
) -> tuple[tuple[object, ...], _native.ElementarySpace]:
    return (diagonal.storage.data,), diagonal.index_space


def _diagonal_unflatten(
    aux_data: _native.ElementarySpace,
    children: tuple[object, ...],
) -> DiagonalTensorMap:
    (data,) = children
    return DiagonalTensorMap(aux_data, data)


_tree_util.register_pytree_node(
    DiagonalTensorMap,
    _diagonal_flatten,
    _diagonal_unflatten,
)
