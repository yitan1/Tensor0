"""Array SVD backend shared by tensor factorizations and custom AD rules."""
from typing import Literal, overload

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike


def _validate_algorithm(algorithm, *, allow_callable=False):
    if allow_callable and callable(algorithm):
        return
    if algorithm is not None and algorithm not in ('qr', 'jacobi', 'divide_and_conquer'):
        raise ValueError('algorithm must be None, qr, jacobi or divide_and_conquer'
                         + (' or a compact matrix callable' if allow_callable else ''))


@overload
def matrix_svd(matrix: ArrayLike, *, full_matrices: bool = False,
               compute_uv: Literal[True] = True, algorithm=None) -> tuple[Array, Array, Array]: ...


@overload
def matrix_svd(matrix: ArrayLike, *, full_matrices: bool = False,
               compute_uv: Literal[False], algorithm=None) -> Array: ...


@overload
def matrix_svd(matrix: ArrayLike, *, full_matrices: bool = False,
               compute_uv: bool, algorithm=None) -> tuple[Array, Array, Array] | Array: ...


def matrix_svd(matrix: ArrayLike, *, full_matrices: bool = False,
               compute_uv: bool = True, algorithm=None):
    """Compute array SVD with a static algorithm selection.

    None preserves jnp.linalg.svd defaults. Explicit algorithms use JAX lax;
    unsupported device/algorithm combinations raise at lowering, never fall back.
    With compute_uv=True the result is (u, s, vh), otherwise singular values.
    """
    _validate_algorithm(algorithm)
    if algorithm is None:
        return jnp.linalg.svd(matrix, full_matrices=full_matrices, compute_uv=compute_uv)
    selected = {
        'qr': jax.lax.linalg.SvdAlgorithm.QR,
        'jacobi': jax.lax.linalg.SvdAlgorithm.JACOBI,
        'divide_and_conquer': jax.lax.linalg.SvdAlgorithm.DIVIDE_AND_CONQUER,
    }[algorithm]
    return jax.lax.linalg.svd(matrix, full_matrices=full_matrices,
                              compute_uv=compute_uv, algorithm=selected)
