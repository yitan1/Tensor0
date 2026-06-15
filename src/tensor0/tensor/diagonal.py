from __future__ import annotations

from dataclasses import dataclass

from jax import tree_util as _tree_util
import jax.numpy as jnp

from .. import _native
from ..structure.spaces import hom
from .sector_vector import SectorVector, _space_dim
from .storage import VectorStorage, _validate_vector_storage_data


@dataclass(frozen=True, eq=False, init=False)
class DiagonalTensorMap:
    domain: _native.ElementarySpace
    storage: VectorStorage

    def __init__(self, domain: _native.ElementarySpace, storage: object) -> None:
        if not isinstance(domain, _native.ElementarySpace):
            raise TypeError("DiagonalTensorMap requires an ElementarySpace")

        vector_storage = storage
        if not isinstance(vector_storage, VectorStorage):
            vector_storage = VectorStorage(vector_storage)

        _validate_vector_storage_data(vector_storage.data, _space_dim(domain))

        object.__setattr__(self, "domain", domain)
        object.__setattr__(self, "storage", vector_storage)

    @property
    def space(self) -> _native.HomSpace:
        return hom((self.domain,), (self.domain,))

    def diag(self) -> SectorVector:
        return SectorVector(self.domain, self.storage)

    def block(self, coupled: int | tuple[int, ...]):
        return jnp.diag(self.diag().block(coupled))

    def blocks(self):
        return tuple((sector, jnp.diag(values)) for sector, values in self.diag().blocks())

    def to_tensor_map(self):
        from .tensor_map import TensorMap, _packed_vector_from_blocks

        diagonal_space = self.space
        block_arrays = dict(self.blocks())
        dtype = jnp.asarray(self.storage.data).dtype
        return TensorMap(
            diagonal_space,
            _packed_vector_from_blocks(diagonal_space, block_arrays, dtype=dtype),
        )


def _diagonal_flatten(
    diagonal: DiagonalTensorMap,
) -> tuple[tuple[object, ...], _native.ElementarySpace]:
    return (diagonal.storage.data,), diagonal.domain


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
