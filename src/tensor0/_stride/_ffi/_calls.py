"""Typed buffer calls to native, without old-backend fallback."""

from functools import partial
from math import prod

import jax
from jax import Array
from jax.extend import core
from jax.interpreters import mlir, xla
import numpy as np
import jax.numpy as jnp

from ._registration import cuda_reduction_target, cuda_accumulation_target, cuda_copy_target, cuda_dot_target, cuda_update_target, operation_target


def execute_dot(left: Array, right: Array, *, layout: np.ndarray, conjugate_left: bool, dtype=None, platform="cpu") -> Array:
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
    if platform == "cuda":
        result = _cuda_dot_p.bind(
            left.reshape((batch_count, left.shape[-1])), right.reshape((batch_count, right.shape[-1])),
            layout=tuple(int(word) for word in layout), conjugate_left=conjugate_left, dtype=result_dtype,
        )
        return result.reshape(batch_shape)
    if platform != "cpu":
        raise NotImplementedError(f"native dot does not support platform {platform}")
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


def execute_copy(source: Array, *, layout: np.ndarray, output_size: int, dtype=None, platform="cpu") -> Array:
    """Normalize storage batches and submit one native Copy call.

    The caller supplies the compilation platform, including after partitioning.
    Layout producers must supply disjoint destination address sets across records;
    native validates injectivity within each record, not cross-record overlap.
    """
    if not isinstance(source, (Array, jax.core.Tracer)):
        raise TypeError("copy source must be a JAX Array or Tracer")
    if source.ndim == 0:
        raise ValueError("copy storage must have at least one dimension")
    batch_shape = source.shape[:-1]
    batch_count = prod(batch_shape)
    result_dtype = source.dtype if dtype is None else jax.dtypes.canonicalize_dtype(dtype)
    if platform == "cuda":
        result = _cuda_copy_p.bind(
            source.reshape((batch_count, source.shape[-1])),
            layout=tuple(int(word) for word in layout), output_size=output_size, dtype=result_dtype,
        )
        return result.reshape((*batch_shape, output_size))
    if platform != "cpu":
        raise NotImplementedError(f"native copy does not support platform {platform}")
    execute = jax.ffi.ffi_call(
        operation_target("copy", result_dtype),
        jax.ShapeDtypeStruct((batch_count, output_size), result_dtype),
        vmap_method="sequential",
    )
    result = execute(source.reshape((batch_count, source.shape[-1])), layout=layout)
    return result.reshape((*batch_shape, output_size))


def execute_update(
    source: Array, base: Array, alpha: Array, beta: Array, *, layout: np.ndarray, platform="cpu",
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
    if platform == "cuda":
        result = _cuda_update_p.bind(
            source.reshape((prod(batch_shape), source.shape[-1])),
            base.reshape((prod(batch_shape), base.shape[-1])),
            alpha if alpha.ndim == 0 else alpha.reshape((alpha.size,)),
            beta if beta.ndim == 0 else beta.reshape((beta.size,)),
            layout=tuple(int(word) for word in layout),
        )
        return result.reshape(base.shape)
    if platform != "cpu":
        raise NotImplementedError(f"native update does not support platform {platform}")
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
    output_size: int, dtype=None, platform="cpu",
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
        layout=layout, output_size=output_size, dtype=dtype, platform=platform,
    )


def execute_accumulation(
    source: Array, coefficients: tuple[Array, ...] = (), *,
    coefficient_records: tuple[int, ...] = (), layout: np.ndarray,
    output_size: int, dtype=None, platform="cpu",
) -> Array:
    """Fresh address sum using paired-address layout words, including overlap.

    Coefficients and writeback follow execute_reduction. Native clears the whole
    output once and sums records in order; nonzero destination stride overlap is
    legal. This raw FFI call has no AD rule and no old-backend fallback.
    """
    return _execute_reduction(
        "accumulation", source, coefficients, coefficient_records=coefficient_records,
        layout=layout, output_size=output_size, dtype=dtype, platform=platform,
    )


def _execute_reduction(
    operation: str, source: Array, coefficients: tuple[Array, ...], *,
    coefficient_records: tuple[int, ...], layout: np.ndarray, output_size: int, dtype, platform="cpu",
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
    if platform == "cuda":
        primitive = _cuda_accumulation_p if operation == "accumulation" else _cuda_reduction_p
        # Preserve every unsigned protocol bit, including with JAX x64 disabled.
        if operation == "reduction":
            layout = np.asarray(layout, dtype=np.uint8).view("<i8")
        result = primitive.bind(
            source.reshape((prod(batch_shape), source.shape[-1])),
            *(value if value.ndim == 0 else value.reshape((value.size,)) for value in coefficients),
            layout=tuple(int(word) for word in layout), output_size=output_size,
            dtype=result_dtype, coefficient_records=coefficient_records,
        )
        return result.reshape((*batch_shape, output_size))
    if platform != "cpu":
        raise NotImplementedError(f"native {operation} does not support platform {platform}")
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


def _cuda_copy_abstract(source, *, layout, output_size, dtype):
    return source.update(shape=(source.shape[0], output_size), dtype=dtype, weak_type=False)


def _cuda_copy_lowering(context, source, *, layout, output_size, dtype):
    target = cuda_copy_target(context.avals_in[0].dtype, dtype)
    words = np.asarray(layout, dtype=np.int64)
    # Emit i64 directly: tracing a JAX constant would narrow it when x64 is off.
    metadata = mlir.ir_constant(words)
    context = context.replace(avals_in=(*context.avals_in, jax.core.ShapedArray(words.shape, words.dtype)))
    return jax.ffi.ffi_lowering(
        target,
        operand_layouts=((1, 0), (0,)), result_layouts=((1, 0),),
    )(context, source, metadata, layout=words)


_cuda_copy_p = core.Primitive("tensor0_stride_cuda_copy_call")
_cuda_copy_p.def_impl(partial(xla.apply_primitive, _cuda_copy_p))
_cuda_copy_p.def_abstract_eval(_cuda_copy_abstract)
mlir.register_lowering(_cuda_copy_p, _cuda_copy_lowering, platform="cuda")


def _cuda_update_abstract(source, base, alpha, beta, *, layout):
    return base.update(weak_type=False)


def _cuda_update_lowering(context, source, base, alpha, beta, *, layout):
    target = cuda_update_target(*(value.dtype for value in context.avals_in))
    words = np.asarray(layout, dtype=np.int64)
    # Emit i64 directly even when JAX x64 is disabled.
    metadata = mlir.ir_constant(words)
    operand_layouts = tuple(tuple(reversed(range(value.ndim))) for value in context.avals_in)
    context = context.replace(avals_in=(*context.avals_in, jax.core.ShapedArray(words.shape, words.dtype)))
    return jax.ffi.ffi_lowering(
        target, operand_layouts=(*operand_layouts, (0,)), result_layouts=((1, 0),),
        operand_output_aliases={1: 0},
    )(context, source, base, alpha, beta, metadata, layout=words)


_cuda_update_p = core.Primitive("tensor0_stride_cuda_update_call")
_cuda_update_p.def_impl(partial(xla.apply_primitive, _cuda_update_p))
_cuda_update_p.def_abstract_eval(_cuda_update_abstract)
mlir.register_lowering(_cuda_update_p, _cuda_update_lowering, platform="cuda")


def _cuda_accumulation_abstract(source, *coefficients, layout, output_size, dtype, coefficient_records):
    return source.update(shape=(source.shape[0], output_size), dtype=dtype, weak_type=False)


def _cuda_accumulation_lowering(context, source, *coefficients, layout, output_size, dtype, coefficient_records):
    target = cuda_accumulation_target(context.avals_in[0].dtype, dtype,
                                      *(value.dtype for value in context.avals_in[1:]))
    words = np.asarray(layout, dtype=np.int64)
    metadata = mlir.ir_constant(words)
    avals = context.avals_in
    ordered_avals = (avals[0], jax.core.ShapedArray(words.shape, words.dtype), *avals[1:])
    context = context.replace(avals_in=ordered_avals)
    return jax.ffi.ffi_lowering(
        target, operand_layouts=tuple(tuple(reversed(range(value.ndim))) for value in ordered_avals),
        result_layouts=((1, 0),),
    )(context, source, metadata, *coefficients, layout=words,
      coefficient_records=np.asarray(coefficient_records, dtype=np.int64))


_cuda_accumulation_p = core.Primitive("tensor0_stride_cuda_accumulation_call")
_cuda_accumulation_p.def_impl(partial(xla.apply_primitive, _cuda_accumulation_p))
_cuda_accumulation_p.def_abstract_eval(_cuda_accumulation_abstract)
mlir.register_lowering(_cuda_accumulation_p, _cuda_accumulation_lowering, platform="cuda")


def _cuda_dot_abstract(left, right, *, layout, conjugate_left, dtype):
    return left.update(shape=(left.shape[0],), dtype=dtype, weak_type=False)


def _cuda_dot_lowering(context, left, right, *, layout, conjugate_left, dtype):
    target = cuda_dot_target(context.avals_in[0].dtype, context.avals_in[1].dtype, dtype)
    words = np.asarray(layout, dtype=np.int64)
    cursor, capacity = 4, 0
    for _ in range(layout[3]):
        rank = layout[cursor]
        count = prod(layout[cursor + 3:cursor + 3 + rank])
        capacity = max(capacity, (count + 1023) // 1024)
        cursor += 3 + 3 * rank
    batches = context.avals_in[0].shape[0]
    if capacity > np.iinfo(np.int64).max or batches * capacity > np.iinfo(np.int64).max // np.dtype(dtype).itemsize:
        raise ValueError("CUDA dot scratch storage size overflows")
    scratch = jax.core.ShapedArray((batches, capacity), dtype)
    metadata = mlir.ir_constant(words)
    context = context.replace(
        avals_in=(*context.avals_in, jax.core.ShapedArray(words.shape, words.dtype)),
        avals_out=(*context.avals_out, scratch),
    )
    results = jax.ffi.ffi_lowering(
        target, operand_layouts=((1, 0), (1, 0), (0,)), result_layouts=((0,), (1, 0)),
    )(context, left, right, metadata, layout=words, conjugate_left=np.int64(conjugate_left))
    return results[:1]


_cuda_dot_p = core.Primitive("tensor0_stride_cuda_dot_call")
_cuda_dot_p.def_impl(partial(xla.apply_primitive, _cuda_dot_p))
_cuda_dot_p.def_abstract_eval(_cuda_dot_abstract)
mlir.register_lowering(_cuda_dot_p, _cuda_dot_lowering, platform="cuda")


def _cuda_reduction_abstract(source, *coefficients, layout, output_size, dtype, coefficient_records):
    return source.update(shape=(source.shape[0], output_size), dtype=dtype, weak_type=False)


def _cuda_reduction_lowering(context, source, *coefficients, layout, output_size, dtype, coefficient_records):
    target = cuda_reduction_target(context.avals_in[0].dtype, dtype,
                                      *(value.dtype for value in context.avals_in[1:]))
    words = np.asarray(layout, dtype=np.int64)
    metadata = mlir.ir_constant(words)
    avals = context.avals_in
    ordered_avals = (avals[0], jax.core.ShapedArray(words.shape, words.dtype), *avals[1:])
    context = context.replace(avals_in=ordered_avals)
    return jax.ffi.ffi_lowering(
        target, operand_layouts=tuple(tuple(reversed(range(value.ndim))) for value in ordered_avals),
        result_layouts=((1, 0),),
    )(context, source, metadata, *coefficients, layout=words,
      coefficient_records=np.asarray(coefficient_records, dtype=np.int64))


_cuda_reduction_p = core.Primitive("tensor0_stride_cuda_reduction_call")
_cuda_reduction_p.def_impl(partial(xla.apply_primitive, _cuda_reduction_p))
_cuda_reduction_p.def_abstract_eval(_cuda_reduction_abstract)
mlir.register_lowering(_cuda_reduction_p, _cuda_reduction_lowering, platform="cuda")
