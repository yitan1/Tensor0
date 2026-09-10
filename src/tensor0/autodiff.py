from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import wraps
from typing import Any

import jax

from .tensor.diagonal import DiagonalTensorMap
from .tensor._metric import _jax_cotangent_to_riesz_gradient
from .tensor.tensor_map import TensorMap


def _is_tensor_gradient(value: object) -> bool:
    return isinstance(value, (TensorMap, DiagonalTensorMap))


def _to_riesz_gradient_tree(gradient: Any, *, holomorphic: bool) -> Any:
    def convert(value: object) -> object:
        if isinstance(value, DiagonalTensorMap):
            raise TypeError(
                "tensor0.grad does not yet support DiagonalTensorMap arguments"
            )
        if not isinstance(value, TensorMap):
            return value
        if holomorphic:
            raise ValueError(
                "TensorMap Riesz gradients require a real-valued objective; "
                "holomorphic=True is not supported"
            )
        return _jax_cotangent_to_riesz_gradient(value)

    return jax.tree.map(convert, gradient, is_leaf=_is_tensor_gradient)


def grad(
    fun: Callable[..., Any],
    argnums: int | Sequence[int] = 0,
    has_aux: bool = False,
    holomorphic: bool = False,
    allow_int: bool = False,
    reduce_axes: Sequence[Any] = (),
) -> Callable[..., Any]:
    """Differentiate ``fun`` using the natural TensorMap inner product.

    Gradients for TensorMap arguments are returned as Riesz gradients. Gradients
    for ordinary JAX arrays retain the usual JAX coordinate convention.
    """

    coordinate_grad = jax.grad(
        fun,
        argnums=argnums,
        has_aux=has_aux,
        holomorphic=holomorphic,
        allow_int=allow_int,
        reduce_axes=reduce_axes,
    )

    @wraps(fun)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        result = coordinate_grad(*args, **kwargs)
        if has_aux:
            gradient, aux = result
            return _to_riesz_gradient_tree(
                gradient,
                holomorphic=holomorphic,
            ), aux
        return _to_riesz_gradient_tree(result, holomorphic=holomorphic)

    return wrapped


def value_and_grad(
    fun: Callable[..., Any],
    argnums: int | Sequence[int] = 0,
    has_aux: bool = False,
    holomorphic: bool = False,
    allow_int: bool = False,
    reduce_axes: Sequence[Any] = (),
) -> Callable[..., tuple[Any, Any]]:
    """Return the value of ``fun`` and its TensorMap Riesz gradient."""

    coordinate_value_and_grad = jax.value_and_grad(
        fun,
        argnums=argnums,
        has_aux=has_aux,
        holomorphic=holomorphic,
        allow_int=allow_int,
        reduce_axes=reduce_axes,
    )

    @wraps(fun)
    def wrapped(*args: Any, **kwargs: Any) -> tuple[Any, Any]:
        value, gradient = coordinate_value_and_grad(*args, **kwargs)
        return value, _to_riesz_gradient_tree(
            gradient,
            holomorphic=holomorphic,
        )

    return wrapped
