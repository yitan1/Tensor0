"""Shared deterministic inputs for the standard benchmark suite."""

from __future__ import annotations

from math import ceil, exp, pi, prod, sqrt

import jax
import jax.numpy as jnp
from jax.typing import DTypeLike

from tensor0 import (
    ComplexSpace,
    ElementarySpace,
    HomSpace,
    SU2Irrep,
    TensorMap,
    U1Irrep,
    Z2Irrep,
    dim,
    from_dense,
    random_normal,
    space,
)
from tensor0.structure import get_degeneracystructure


def generate_space(
    sector: str,
    dimension: int,
    sigma: float | None = None,
) -> ElementarySpace:
    """Generate deterministic benchmark spaces for the supported sectors."""
    if dimension < 1:
        raise ValueError("benchmark dimensions must be positive")
    if sector == "trivial":
        return ComplexSpace(dimension)
    if sector == "z2":
        spread = 0.5 if sigma is None else sigma
        even = ceil(spread * dimension)
        odd = dimension - even
        sector_dims = {0: even}
        if odd > 0:
            sector_dims[1] = odd
        return space(Z2Irrep, sector_dims)
    if sector == "u1":
        spread = 0.5 if sigma is None else sigma
        if spread <= 0:
            raise ValueError("U1 benchmark sigma must be positive")
        remaining = dimension
        sector_dims: dict[int, int] = {}
        for magnitude in range(22):
            charges = (0,) if magnitude == 0 else (magnitude, -magnitude)
            for charge in charges:
                degeneracy = ceil(
                    dimension
                    * exp(-0.5 * (charge / spread) ** 2)
                    / (spread * sqrt(2 * pi))
                )
                sector_dims[charge] = degeneracy
                remaining -= degeneracy
                if remaining < 1:
                    return space(U1Irrep, sector_dims)
                if abs(charge) > 20:
                    raise ValueError("U1 benchmark space exceeded charge cutoff")
        raise AssertionError("unreachable U1 benchmark space generation")
    if sector == "su2":
        spread = 0.5 if sigma is None else sigma
        if spread <= 0:
            raise ValueError("SU2 benchmark sigma must be positive")
        remaining = dimension
        sector_dims: dict[int, int] = {}
        twice_spin = 0
        while remaining >= 1:
            spin = twice_spin / 2
            quantum_dimension = twice_spin + 1
            degeneracy = ceil(
                dimension
                * exp(-0.5 * (spin / spread) ** 2)
                / (spread * sqrt(2 * pi))
                / quantum_dimension
            )
            sector_dims[twice_spin] = degeneracy
            remaining -= degeneracy * quantum_dimension
            twice_spin += 1
        return space(SU2Irrep, sector_dims)
    raise ValueError(f"unsupported benchmark sector: {sector}")


def dtype_value(dtype: str) -> DTypeLike:
    if dtype == "float64":
        return jnp.float64
    if dtype == "complex128":
        return jnp.complex128
    raise ValueError(f"unsupported benchmark dtype: {dtype}")


def deterministic_array(
    shape: tuple[int, ...],
    *,
    dtype: str,
) -> jax.Array:
    size = prod(shape)
    values = jnp.arange(1, size + 1, dtype=jnp.float64).reshape(shape) / size
    if dtype == "float64":
        return values
    if dtype == "complex128":
        return values.astype(jnp.complex128) * (1.0 + 0.125j)
    raise ValueError(f"unsupported benchmark dtype: {dtype}")


def dense_tensor(target: HomSpace, *, dtype: str) -> TensorMap:
    shape = tuple(dim(item) for item in target.codomain.spaces) + tuple(
        dim(item) for item in target.domain.spaces
    )
    return from_dense(target, deterministic_array(shape, dtype=dtype))


def packed_tensor(
    target: HomSpace,
    *,
    dtype: str = "float64",
    scale: float = 1.0,
) -> TensorMap:
    total_dim = get_degeneracystructure(target).total_dim
    real = jnp.arange(1, total_dim + 1, dtype=jnp.float64) * scale
    if dtype == "float64":
        data = real
    elif dtype == "complex128":
        data = real.astype(jnp.complex128) * (1.0 + 0.1j)
    else:
        raise ValueError(f"unsupported benchmark dtype: {dtype}")
    return TensorMap(target, data)


def random_tensor(target: HomSpace, *, dtype: str, seed: int) -> TensorMap:
    return random_normal(
        jax.random.key(seed),
        target,
        dtype=dtype_value(dtype),
    )
