from __future__ import annotations

from math import prod
from numbers import Real

import jax.numpy as jnp
from jax import Array

from .. import _native
from ..structure.layout import get_degeneracystructure, get_sectorstructure
from ..structure.spaces import sector_spec
from ._blocks import (
    add_to_subblock as _add_to_subblock,
    get_subblock as _get_subblock,
)
from ._tolerances import nonnegative_tolerance
from .tensor_map import TensorMap


def to_dense(tensor: TensorMap) -> jnp.ndarray:
    if not isinstance(tensor, TensorMap):
        raise TypeError("to_dense() requires a TensorMap")
    if sector_spec(tensor.space) == _native.Trivial:
        return _trivial_dense_array(tensor)

    dense_shape = _dense_shape(tensor.space)
    storage = jnp.asarray(tensor.storage.data)
    dense = jnp.zeros(dense_shape, dtype=storage.dtype)
    sectorstructure = get_sectorstructure(tensor.space)
    degeneracystructure = get_degeneracystructure(tensor.space)

    for (row_tree, col_tree), subblock in zip(
        sectorstructure.fusiontree_pairs,
        degeneracystructure.subblockstructure,
    ):
        axes = _tree_pair_axes(tensor.space, row_tree, col_tree)
        coeff = _pair_coeff(row_tree, col_tree, storage)
        reduced = _get_subblock(storage, subblock)
        dense_block = _interleaved_product(reduced, coeff, axes)
        dense = dense.at[_dense_slices(axes)].add(dense_block)

    return dense


def from_dense(
    space: _native.HomSpace,
    data: object,
    *,
    tol: float | None = None,
) -> TensorMap:
    if not isinstance(space, _native.HomSpace):
        raise TypeError("from_dense() requires a HomSpace")

    dense = _normalize_dense_input(space, data)
    tolerance = _normalize_dense_tolerance(dense.dtype, tol)
    if sector_spec(space) == _native.Trivial:
        storage_dtype = jnp.result_type(dense, 0.0)
        storage = jnp.reshape(jnp.asarray(dense, dtype=storage_dtype), (-1,))
        return TensorMap(space, storage)

    sectorstructure = get_sectorstructure(space)
    degeneracystructure = get_degeneracystructure(space)
    storage = jnp.zeros(
        (degeneracystructure.total_dim,),
        dtype=jnp.result_type(dense, 0.0),
    )

    for (row_tree, col_tree), subblock in zip(
        sectorstructure.fusiontree_pairs,
        degeneracystructure.subblockstructure,
    ):
        axes = _tree_pair_axes(space, row_tree, col_tree)
        coeff = _pair_coeff(row_tree, col_tree, dense)
        dense_slice = dense[_dense_slices(axes)]
        reduced = _project_interleaved(dense_slice, coeff, axes)
        reduced = reduced / space.codomain.sector_spec.quantum_dim(row_tree.coupled)
        storage = _add_to_subblock(storage, subblock, reduced)

    result = TensorMap(space, storage)
    if not bool(jnp.allclose(to_dense(result), dense, rtol=tolerance, atol=tolerance)):
        raise ValueError("dense data does not match the target symmetry structure")
    return result


def _trivial_dense_array(tensor: TensorMap) -> Array:
    data = tensor.storage.data
    shape = tensor.dims
    if isinstance(data, Array):
        return jnp.reshape(data, shape)
    return jnp.asarray(data[0 : prod(shape)]).reshape(shape)


def _dense_shape(space: _native.HomSpace) -> tuple[int, ...]:
    codomain_dims, domain_dims = _product_dims(space)
    return codomain_dims + domain_dims


def _normalize_dense_input(space: _native.HomSpace, data: object) -> jnp.ndarray:
    dense = jnp.asarray(data)
    codomain_dims, domain_dims = _product_dims(space)
    full_shape = codomain_dims + domain_dims
    if tuple(dense.shape) == full_shape:
        return dense

    matrix_shape = (prod(codomain_dims), prod(domain_dims))
    if tuple(dense.shape) == matrix_shape:
        return jnp.reshape(dense, full_shape)

    raise ValueError(f"dense shape {tuple(dense.shape)} does not match expected {full_shape}")


def _product_dims(space: _native.HomSpace) -> tuple[tuple[int, ...], tuple[int, ...]]:
    return tuple(_native.product_dims(space.codomain)), tuple(
        _native.product_dims(space.domain),
    )


def _tree_pair_axes(
    space: _native.HomSpace,
    row_tree: _native.FusionTree,
    col_tree: _native.FusionTree,
) -> tuple[tuple[int, int, int, int], ...]:
    return tuple(_native.product_axes(space.codomain, row_tree.uncoupled)) + tuple(
        _native.product_axes(space.domain, col_tree.uncoupled),
    )


def _pair_coeff(
    row_tree: _native.FusionTree,
    col_tree: _native.FusionTree,
    reference: jnp.ndarray,
) -> jnp.ndarray:
    coeff = jnp.asarray(_native.fusiontree_pair_tensor(row_tree, col_tree))
    return coeff.astype(jnp.result_type(reference, coeff))


def _dense_slices(axes: tuple[tuple[int, int, int, int], ...]) -> tuple[slice, ...]:
    return tuple(slice(start, stop) for start, stop, _degeneracy_dim, _quantum_dim in axes)


def _interleaved_product(
    reduced: jnp.ndarray,
    coeff: jnp.ndarray,
    axes: tuple[tuple[int, int, int, int], ...],
) -> jnp.ndarray:
    reduced_shape = tuple(dim for axis in axes for dim in (axis[2], 1))
    coeff_shape = tuple(dim for axis in axes for dim in (1, axis[3]))
    dense_shape = tuple(stop - start for start, stop, _degeneracy_dim, _quantum_dim in axes)
    return jnp.reshape(
        jnp.reshape(reduced, reduced_shape) * jnp.reshape(coeff, coeff_shape),
        dense_shape,
    )


def _project_interleaved(
    dense_slice: jnp.ndarray,
    coeff: jnp.ndarray,
    axes: tuple[tuple[int, int, int, int], ...],
) -> jnp.ndarray:
    expanded_shape = tuple(dim for axis in axes for dim in (axis[2], axis[3]))
    expanded = jnp.reshape(dense_slice, expanded_shape)
    quantum_axes = tuple(range(1, 2 * len(axes), 2))
    coeff_axes = tuple(range(len(axes)))
    return jnp.tensordot(expanded, jnp.conj(coeff), axes=(quantum_axes, coeff_axes))


def _default_tol(dtype: jnp.dtype) -> float:
    real_dtype = jnp.zeros((), dtype=dtype).real.dtype
    if not jnp.issubdtype(real_dtype, jnp.floating):
        real_dtype = jnp.result_type(real_dtype, 0.0)
    return float(jnp.finfo(real_dtype).eps**0.5)


def _normalize_dense_tolerance(dtype: jnp.dtype, tol: float | None) -> float:
    if tol is None:
        return _default_tol(dtype)
    if isinstance(tol, Real) and not isinstance(tol, bool):
        return nonnegative_tolerance(tol, "tol")

    tol_shape = getattr(tol, "shape", None)
    tol_dtype = getattr(tol, "dtype", None)
    if (
        tol_shape is None
        or tuple(tol_shape) != ()
        or tol_dtype is None
        or not (
            jnp.issubdtype(tol_dtype, jnp.integer)
            or jnp.issubdtype(tol_dtype, jnp.floating)
        )
    ):
        raise TypeError("tol must be a finite non-negative real scalar")
    return nonnegative_tolerance(tol, "tol")
