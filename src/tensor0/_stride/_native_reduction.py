"""Native output-owner structured sums for affine maps and transposes."""

from __future__ import annotations

from functools import lru_cache, partial
import struct
from typing import Any

import jax
from jax import Array
import jax.numpy as jnp

from . import _ffi as ffi_module
from ._compiler import (
    CompiledStridePlan,
    _descriptor_scale_words,
    _dtype_code,
    _project_cpu_execution,
    compile_plan,
)
from ._errors import raise_no_eligible_route
from ._native_lowering import NativeExecutionKind, NativeExecutionPlan
from ._plan import (
    MAXIMUM_RANK,
    UINT64_MAX,
    StridedCopyPlan,
    StridedOutputInit,
    StridedReductionKind,
    StridedScalarKind,
    StridedWriteKind,
    _exact_scalar_bytes,
    transpose_plan,
)


NATIVE_REDUCTION_ABI_VERSION = 2
NATIVE_REDUCTION_COMPILER_POLICY_VERSION = 3
NATIVE_REDUCTION_DESCRIPTOR_MAGIC = 0x3152444952543054
_SCALAR_POLICY_FORWARD_SCALE_CAST = 1
_SCALAR_POLICY_JAX_TRANSPOSE = 2


def _axis_mask(axes: tuple[int, ...]) -> int:
    return sum(1 << axis for axis in axes)


def _encode_descriptor(
    semantic: StridedCopyPlan,
    compiled: CompiledStridePlan,
    preferred_chunk_bytes: int,
    parallel_minimum_bytes: int,
) -> bytes:
    input_dtype = jnp.dtype(semantic.source_dtype)
    output_dtype = jnp.dtype(semantic.result_dtype)
    if semantic.scalar_kind is StridedScalarKind.STATIC_SCALE_CAST:
        mapped_dtype = output_dtype
        scalar_policy = _SCALAR_POLICY_FORWARD_SCALE_CAST
    elif semantic.scalar_kind is StridedScalarKind.JAX_TRANSPOSE:
        mapped_dtype = input_dtype
        scalar_policy = _SCALAR_POLICY_JAX_TRANSPOSE
    else:
        raise ValueError("native structured reduction scalar policy is unsupported")
    words: list[int] = [
        NATIVE_REDUCTION_DESCRIPTOR_MAGIC,
        NATIVE_REDUCTION_ABI_VERSION,
        0,
        NATIVE_REDUCTION_COMPILER_POLICY_VERSION,
        semantic.source_size,
        semantic.output_size,
        len(compiled.records),
        _dtype_code(input_dtype),
        _dtype_code(mapped_dtype),
        _dtype_code(output_dtype),
        scalar_policy,
        preferred_chunk_bytes,
        parallel_minimum_bytes,
    ]
    for raw_record, record in zip(
        semantic.records,
        compiled.records,
        strict=True,
    ):
        rank = record.rank
        input_order = tuple(
            sorted(
                range(rank),
                key=lambda axis: (abs(record.source_strides[axis]), axis),
            )
        )
        scale_real, scale_imaginary = _descriptor_scale_words(
            _exact_scalar_bytes(raw_record.scale, mapped_dtype),
            mapped_dtype,
        )
        words.extend(
            (
                rank,
                record.source_offset,
                record.destination_offset,
                scale_real,
                scale_imaginary,
                ((1 << rank) - 1) ^ _axis_mask(record.reduction_axes),
                _axis_mask(record.reduction_axes),
                record.output_count,
                record.reduction_count,
                *record.logical_shape,
                *(stride & UINT64_MAX for stride in record.source_strides),
                *(stride & UINT64_MAX for stride in record.destination_strides),
                *input_order,
                len(record.map_loop_order),
                *record.map_loop_order,
                len(record.reduction_loop_order),
                *record.reduction_loop_order,
            )
        )
    words[2] = len(words)
    if any(word < 0 or word > UINT64_MAX for word in words):
        raise AssertionError("native reduction descriptor word exceeds uint64")
    return b"".join(struct.pack("<Q", word) for word in words)


@lru_cache(maxsize=1_024)
def compile_native_structured_reduction(
    semantic: StridedCopyPlan,
) -> NativeExecutionPlan:
    """Project one semantic structured sum onto the retained CPU ABI."""

    if semantic.reduction_kind is not StridedReductionKind.SUM:
        raise ValueError("native structured reduction requires SUM semantics")
    if semantic.output_init is not StridedOutputInit.ZERO:
        raise ValueError("native structured reduction requires zero initialization")
    if (
        len(semantic.records) > 1
        and semantic.write_kind is not StridedWriteKind.ACCUMULATE
    ):
        raise ValueError("multi-record structured reduction requires accumulation")
    dtype_pair = (semantic.source_dtype, semantic.result_dtype)
    if semantic.scalar_kind is StridedScalarKind.JAX_TRANSPOSE:
        supported = dtype_pair in ffi_module._STRUCTURED_REDUCTION_TRANSPOSE_SUFFIXES
        reported_pair = tuple(reversed(dtype_pair))
    else:
        supported = dtype_pair in ffi_module._STRUCTURED_REDUCTION_FORWARD_SUFFIXES
        reported_pair = dtype_pair
    if not supported:
        raise ValueError(f"unsupported structured reduction dtype pair {reported_pair!r}")
    compiled = compile_plan(semantic)
    for index, record in enumerate(compiled.records):
        if record.rank > MAXIMUM_RANK:
            raise ValueError(
                f"record[{index}] rank exceeds native reduction limit {MAXIMUM_RANK}"
            )
    projection = _project_cpu_execution(compiled)
    if projection is None:
        raise ValueError("structured reduction has no native CPU projection")
    (
        cpu_records,
        preferred_chunk_bytes,
        parallel_minimum_bytes,
        chunks_per_batch,
    ) = projection
    return NativeExecutionPlan(
        kind=NativeExecutionKind.STRUCTURED_REDUCTION,
        descriptor=_encode_descriptor(
            semantic,
            compiled,
            preferred_chunk_bytes,
            parallel_minimum_bytes,
        ),
        compiled=compiled,
        records=cpu_records,
        preferred_chunk_bytes=preferred_chunk_bytes,
        parallel_minimum_bytes=parallel_minimum_bytes,
        chunks_per_batch=chunks_per_batch,
    )


@lru_cache(maxsize=1_024)
def compile_native_broadcast_transpose(
    forward: StridedCopyPlan,
) -> NativeExecutionPlan:
    """Compile the transpose of one explicit source-broadcast map."""

    if not forward.has_source_broadcast:
        raise ValueError("native broadcast transpose requires source broadcast")
    return compile_native_structured_reduction(transpose_plan(forward))


def _abstract_eval(source_aval: Any, *, plan: NativeExecutionPlan) -> Any:
    if plan.kind is not NativeExecutionKind.STRUCTURED_REDUCTION:
        raise ValueError("structured reduction primitive received an affine plan")
    if source_aval.dtype != jnp.dtype(plan.bound.source_dtype):
        raise TypeError("structured reduction input dtype mismatch")
    if not source_aval.shape or source_aval.shape[-1] != plan.bound.source_size:
        raise ValueError("structured reduction input size mismatch")
    return source_aval.update(
        shape=(*source_aval.shape[:-1], plan.bound.output_size),
        dtype=jnp.dtype(plan.bound.result_dtype),
    )


def _lowering(
    context: Any,
    source: Any,
    *,
    plan: NativeExecutionPlan,
) -> Any:
    from jax.interpreters import mlir

    device_count = getattr(context.module_context.axis_context, "num_devices", None)
    if device_count not in (None, 1):
        raise_no_eligible_route(("native_structured_reduction_sharded",))
    if not ffi_module.native_available():
        raise_no_eligible_route(("native_structured_reduction_unavailable",))
    function = lambda value: ffi_module._structured_reduction_ffi_call_v2(
        value,
        descriptor=plan.descriptor,
        output_size=plan.bound.output_size,
        output_dtype=jnp.dtype(plan.bound.result_dtype),
        scalar_kind=plan.bound.scalar_kind,
    )
    return mlir.lower_fun(function, multiple_results=False)(context, source)


def _non_cpu_lowering(
    context: Any,
    source: Any,
    *,
    plan: NativeExecutionPlan,
) -> Any:
    del context, source, plan
    raise_no_eligible_route(("native_structured_reduction_non_cpu",))


def _jvp(
    primals: tuple[Array],
    tangents: tuple[Any],
    *,
    plan: NativeExecutionPlan,
) -> tuple[Array, Any]:
    from jax.interpreters import ad

    (source,), (tangent,) = primals, tangents
    primal = _STRUCTURED_REDUCTION_PRIMITIVE.bind(source, plan=plan)
    if isinstance(tangent, ad.Zero):
        return primal, ad.Zero(jax.typeof(primal).to_tangent_aval())
    return primal, _STRUCTURED_REDUCTION_PRIMITIVE.bind(tangent, plan=plan)


def _transpose(
    cotangent: Any,
    source: Any,
    *,
    plan: NativeExecutionPlan,
) -> list[Any]:
    from jax.interpreters import ad

    if not ad.is_undefined_primal(source):
        return [None]
    if isinstance(cotangent, ad.Zero):
        return [ad.Zero(source.aval.to_ct_aval())]
    reverse = transpose_plan(plan.bound)
    if reverse.reduction_kind is StridedReductionKind.SUM:
        return [bind_native_structured_reduction(cotangent, reverse)]
    return [
        ffi_module.strided_copy(cotangent, plan=reverse, native_required=True)
    ]


def _batch(
    arguments: tuple[Array],
    dimensions: tuple[int | None],
    *,
    plan: NativeExecutionPlan,
) -> tuple[Array, int | None]:
    from jax.interpreters import batching

    (source,), (dimension,) = arguments, dimensions
    if dimension is None:
        return _STRUCTURED_REDUCTION_PRIMITIVE.bind(source, plan=plan), None
    source = batching.bdim_at_front(source, dimension, source.shape[dimension])
    return _STRUCTURED_REDUCTION_PRIMITIVE.bind(source, plan=plan), 0


def _create_primitive() -> Any:
    from jax._src import dispatch
    from jax.extend import core
    from jax.interpreters import ad, batching, mlir, xla

    primitive = core.Primitive("tensor0_stride_structured_reduction")
    primitive.def_impl(partial(xla.apply_primitive, primitive))
    primitive.def_abstract_eval(_abstract_eval)
    ad.primitive_jvps[primitive] = _jvp
    ad.primitive_transposes[primitive] = _transpose
    batching.primitive_batchers[primitive] = _batch
    mlir.register_lowering(primitive, _non_cpu_lowering)
    mlir.register_lowering(primitive, _lowering, platform="cpu")
    dispatch.prim_requires_devices_during_lowering.add(primitive)
    return primitive


_STRUCTURED_REDUCTION_PRIMITIVE = _create_primitive()


def bind_native_broadcast_transpose(
    cotangent: Array,
    forward: StridedCopyPlan,
) -> Array:
    """Bind the native structured transpose of one broadcast-read plan."""

    plan = compile_native_broadcast_transpose(forward)
    data = ffi_module._require_jax_array(cotangent, "broadcast cotangent")
    if data.dtype != jnp.dtype(plan.bound.source_dtype):
        raise TypeError("broadcast cotangent dtype does not match the plan")
    if not data.shape or data.shape[-1] != forward.output_size:
        raise ValueError("broadcast cotangent storage size does not match the plan")
    return _STRUCTURED_REDUCTION_PRIMITIVE.bind(data, plan=plan)


def bind_native_structured_reduction(
    source: Array,
    semantic: StridedCopyPlan,
) -> Array:
    """Bind one plan-owned native structured sum."""

    plan = compile_native_structured_reduction(semantic)
    data = ffi_module._require_jax_array(source, "structured reduction source")
    if data.dtype != jnp.dtype(plan.bound.source_dtype):
        raise TypeError("structured reduction source dtype does not match the plan")
    if not data.shape or data.shape[-1] != semantic.source_size:
        raise ValueError("structured reduction source size does not match the plan")
    return _STRUCTURED_REDUCTION_PRIMITIVE.bind(data, plan=plan)


def strided_reduce(source: object, *, plan: StridedCopyPlan) -> Array:
    """Execute one statically proved structured sum on the native CPU path."""

    data = ffi_module._require_jax_array(source, "strided_reduce source")
    return bind_native_structured_reduction(data, plan)


__all__ = [
    "NATIVE_REDUCTION_ABI_VERSION",
    "NATIVE_REDUCTION_COMPILER_POLICY_VERSION",
    "NATIVE_REDUCTION_DESCRIPTOR_MAGIC",
    "bind_native_broadcast_transpose",
    "bind_native_structured_reduction",
    "compile_native_broadcast_transpose",
    "compile_native_structured_reduction",
    "strided_reduce",
]
