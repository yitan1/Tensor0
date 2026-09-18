"""Typed buffer calls to native, without old-backend fallback."""

from math import prod

import jax
from jax import Array
import numpy as np
import jax.numpy as jnp

from ._registration import operation_target


def execute_dot(left: Array, right: Array, *, layout: np.ndarray, conjugate_left: bool, dtype=None) -> Array:
    """Call Dot with flattened storage batches and one result per batch."""
    for name, value in (("left", left), ("right", right)):
        if not isinstance(value, (Array, jax.core.Tracer)):
            raise TypeError(f"{name} must be a JAX Array or Tracer")
        if value.ndim == 0:
            raise ValueError("dot storage must have at least one dimension")
    if left.shape[:-1] != right.shape[:-1]:
        raise ValueError("dot batch shapes must match")
    result_dtype = jnp.result_type(left.dtype, right.dtype) if dtype is None else jax.dtypes.canonicalize_dtype(dtype)
    batch_shape = left.shape[:-1]
    batch_count = prod(batch_shape)
    execute = jax.ffi.ffi_call(
        operation_target("dot", result_dtype),
        jax.ShapeDtypeStruct((batch_count,), result_dtype),
        vmap_method="sequential",
    )
    result = execute(
        left.reshape((batch_count, left.shape[-1])),
        right.reshape((batch_count, right.shape[-1])),
        layout=layout, conjugate_left=np.int64(conjugate_left),
    )
    return result.reshape(batch_shape)


def execute_copy(source: Array, *, layout: np.ndarray, output_size: int, dtype=None) -> Array:
    """Normalize storage batches and submit one native Copy call."""
    if not isinstance(source, (Array, jax.core.Tracer)):
        raise TypeError("copy source must be a JAX Array or Tracer")
    if source.ndim == 0:
        raise ValueError("copy storage must have at least one dimension")
    batch_shape = source.shape[:-1]
    batch_count = prod(batch_shape)
    result_dtype = source.dtype if dtype is None else jax.dtypes.canonicalize_dtype(dtype)
    execute = jax.ffi.ffi_call(
        operation_target("copy", result_dtype),
        jax.ShapeDtypeStruct((batch_count, output_size), result_dtype),
        vmap_method="sequential",
    )
    result = execute(source.reshape((batch_count, source.shape[-1])), layout=layout)
    return result.reshape((*batch_shape, output_size))


def execute_update(
    source: Array, base: Array, alpha: Array, beta: Array, *, layout: np.ndarray,
) -> Array:
    """Normalize storage batches and call the unified native update handler.

    Coefficients are scalars, shared length-one vectors, or arrays matching the
    storage batch shape. Reshaping changes neither dtype nor conversion order.
    Native owns zero/one branching, final casts, and unselected-base preservation.
    Layout producers must ensure disjoint destination address sets across records.
    Native validates each record's injectivity and address safety, not this
    cross-record precondition. Repeated source reads are supported; overlapping
    record writes are outside the contract, including for AD.
    Result aliases base at the FFI boundary; JAX protects live inputs. Native
    validates actual buffer relationships before writes and skips copying only
    for exact base/result reuse. Caller donation is an outer JIT decision.
    """
    for name, value in (("source", source), ("base", base), ("alpha", alpha), ("beta", beta)):
        if not isinstance(value, (Array, jax.core.Tracer)):
            raise TypeError(f"{name} must be a JAX Array or Tracer with an explicit dtype")
    if source.ndim == 0 or base.ndim == 0:
        raise ValueError("update storage buffers must have at least one dimension")
    batch_shape = base.shape[:-1]
    if source.shape[:-1] != batch_shape:
        raise ValueError("update batch shapes must match")
    for coefficient in (alpha, beta):
        if coefficient.ndim != 0 and coefficient.shape not in ((1,), batch_shape):
            raise ValueError("update coefficients must be scalar, length-one, or match the batch shape")
    execute = jax.ffi.ffi_call(
        operation_target("update", base.dtype),
        jax.ShapeDtypeStruct((prod(batch_shape), base.shape[-1]), base.dtype),
        vmap_method="sequential",
        input_output_aliases={1: 0},
    )
    result = execute(
        source.reshape((prod(batch_shape), source.shape[-1])),
        base.reshape((prod(batch_shape), base.shape[-1])),
        alpha if alpha.ndim == 0 else alpha.reshape((alpha.size,)),
        beta if beta.ndim == 0 else beta.reshape((beta.size,)),
        layout=layout,
    )
    return result.reshape(base.shape)


def execute_reduction(
    source: Array, coefficients: tuple[Array, ...] = (), *,
    coefficient_records: tuple[int, ...] = (), layout: np.ndarray,
    output_size: int, dtype=None,
) -> Array:
    """Fresh default mixed reduction, not the JAX sum/transpose binding.

    Each operand scales its listed original record; unlisted records have no
    scaling stage. Scalar and length-one coefficients are shared; other shapes
    must match the storage batch shape. Zero skips the contribution, one skips
    multiplication, and general factors keep their concrete dtypes.
    Output dtype determines accumulator storage, not a hidden
    input conversion. Native owns initialization, traversal, and writeback.
    """
    return _execute_reduction(
        "reduction", source, coefficients, coefficient_records=coefficient_records,
        layout=layout, output_size=output_size, dtype=dtype,
    )


def execute_accumulation(
    source: Array, coefficients: tuple[Array, ...] = (), *,
    coefficient_records: tuple[int, ...] = (), layout: np.ndarray,
    output_size: int, dtype=None,
) -> Array:
    """Fresh address sum using paired-address layout words, including overlap.

    Coefficients and writeback follow execute_reduction. Native clears the whole
    output once and sums records in order; nonzero destination stride overlap is
    legal. This raw FFI call has no AD rule and no old-backend fallback.
    """
    return _execute_reduction(
        "accumulation", source, coefficients, coefficient_records=coefficient_records,
        layout=layout, output_size=output_size, dtype=dtype,
    )


def _execute_reduction(
    operation: str, source: Array, coefficients: tuple[Array, ...], *,
    coefficient_records: tuple[int, ...], layout: np.ndarray, output_size: int, dtype,
) -> Array:
    if not isinstance(source, (Array, jax.core.Tracer)):
        raise TypeError("reduction source must be a JAX Array or Tracer")
    if source.ndim == 0:
        raise ValueError("reduction storage must have at least one dimension")
    batch_shape = source.shape[:-1]
    for coefficient in coefficients:
        if not isinstance(coefficient, (Array, jax.core.Tracer)):
            raise TypeError("reduction coefficients must be JAX Arrays or Tracers with explicit dtypes")
        if coefficient.ndim != 0 and coefficient.shape not in ((1,), batch_shape):
            raise ValueError("reduction coefficients must be scalar, length-one, or match the batch shape")
    if any(type(index) is not int or not 0 <= index <= np.iinfo(np.int64).max
           for index in coefficient_records):
        raise ValueError("coefficient record indices must fit nonnegative int64")
    result_dtype = source.dtype if dtype is None else jax.dtypes.canonicalize_dtype(dtype)
    execute = jax.ffi.ffi_call(
        operation_target(operation, result_dtype),
        jax.ShapeDtypeStruct((prod(batch_shape), output_size), result_dtype),
        vmap_method="sequential",
    )
    result = execute(
        source.reshape((prod(batch_shape), source.shape[-1])),
        *(value if value.ndim == 0 else value.reshape((value.size,)) for value in coefficients),
        layout=layout, coefficient_records=np.asarray(coefficient_records, dtype=np.int64),
    )
    return result.reshape((*batch_shape, output_size))
