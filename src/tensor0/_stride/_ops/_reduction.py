"""Structured affine reduction semantics and native execution boundary."""

from __future__ import annotations

from functools import lru_cache, partial
import struct
from typing import Any

import jax
from jax import Array
import jax.numpy as jnp

from .._native_descriptor import (
    NativeCallKind,
    NativeCallSpec,
    _descriptor_scale_words,
    _dtype_code,
)
from .._scalar import mapping_dtype
from .._errors import raise_no_eligible_route
from .._jax import require_jax_array
from .._map import _execute_map
from .._native import (
    _STRUCTURED_REDUCTION_FORWARD_SUFFIXES,
    _STRUCTURED_REDUCTION_TRANSPOSE_SUFFIXES,
    _structured_reduction_ffi_call,
    native_available,
)
from .._plan import (
    UINT64_MAX,
    AffinePlan,
    StridedOutputInit,
    StridedReductionKind,
    StridedScalarKind,
    StridedWriteKind,
    _exact_scalar_bytes,
    transpose_plan,
)


NATIVE_REDUCTION_ABI_VERSION = 4
NATIVE_REDUCTION_DESCRIPTOR_MAGIC = 0x3152444952543054
_SCALAR_POLICY_FORWARD_SCALE_CAST = 1
_SCALAR_POLICY_JAX_TRANSPOSE = 2
def _encode_descriptor(
    semantic: AffinePlan,
) -> bytes:
    input_dtype = jnp.dtype(semantic.source_dtype)
    output_dtype = jnp.dtype(semantic.result_dtype)
    if semantic.scalar_kind is StridedScalarKind.STATIC_SCALE_CAST:
        mapping_source_dtype = input_dtype
        scalar_policy = _SCALAR_POLICY_FORWARD_SCALE_CAST
    elif semantic.scalar_kind is StridedScalarKind.JAX_TRANSPOSE:
        mapping_source_dtype = output_dtype
        scalar_policy = _SCALAR_POLICY_JAX_TRANSPOSE
    else:
        raise ValueError("native structured reduction scalar policy is unsupported")
    words: list[int] = [
        NATIVE_REDUCTION_DESCRIPTOR_MAGIC,
        NATIVE_REDUCTION_ABI_VERSION,
        0,
        semantic.source_size,
        semantic.output_size,
        len(semantic.records),
        _dtype_code(input_dtype),
        _dtype_code(output_dtype),
        scalar_policy,
    ]
    for record in semantic.records:
        rank = len(record.logical_shape)
        mapped_dtype = mapping_dtype(mapping_source_dtype, record.scale)
        scale_real, scale_imaginary = _descriptor_scale_words(
            _exact_scalar_bytes(record.scale, mapped_dtype),
            mapped_dtype,
        )
        words.extend(
            (
                rank,
                record.source_offset,
                record.destination_offset,
                scale_real,
                scale_imaginary,
                int(record.scale is None),
                _dtype_code(mapped_dtype),
                *record.logical_shape,
                *(stride & UINT64_MAX for stride in record.source_strides),
                *(stride & UINT64_MAX for stride in record.destination_strides),
                *(
                    int(axis in record.reduction_axes)
                    for axis in range(rank)
                ),
            )
        )
    words[2] = len(words)
    if any(word < 0 or word > UINT64_MAX for word in words):
        raise AssertionError("native reduction descriptor word exceeds uint64")
    return b"".join(struct.pack("<Q", word) for word in words)


@lru_cache(maxsize=1_024)
def lower_structured_reduction(
    semantic: AffinePlan,
) -> NativeCallSpec:
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
        supported = dtype_pair in _STRUCTURED_REDUCTION_TRANSPOSE_SUFFIXES
        reported_pair = tuple(reversed(dtype_pair))
    else:
        supported = dtype_pair in _STRUCTURED_REDUCTION_FORWARD_SUFFIXES
        reported_pair = dtype_pair
    if not supported:
        raise ValueError(f"unsupported structured reduction dtype pair {reported_pair!r}")
    return NativeCallSpec(
        kind=NativeCallKind.STRUCTURED_REDUCTION,
        descriptor=_encode_descriptor(semantic),
        semantic=semantic,
    )


@lru_cache(maxsize=1_024)
def lower_broadcast_transpose(
    forward: AffinePlan,
) -> NativeCallSpec:
    """Lower the transpose of one explicit source-broadcast map."""

    if not forward.has_source_broadcast:
        raise ValueError("native broadcast transpose requires source broadcast")
    return lower_structured_reduction(transpose_plan(forward))


def _abstract_eval(source_aval: Any, *, plan: NativeCallSpec) -> Any:
    if plan.kind is not NativeCallKind.STRUCTURED_REDUCTION:
        raise ValueError("structured reduction primitive received an affine plan")
    if source_aval.dtype != jnp.dtype(plan.semantic.source_dtype):
        raise TypeError("structured reduction input dtype mismatch")
    if not source_aval.shape or source_aval.shape[-1] != plan.semantic.source_size:
        raise ValueError("structured reduction input size mismatch")
    return source_aval.update(
        shape=(*source_aval.shape[:-1], plan.semantic.output_size),
        dtype=jnp.dtype(plan.semantic.result_dtype),
    )


def _lowering(
    context: Any,
    source: Any,
    *,
    plan: NativeCallSpec,
    grouped_output_owner: bool = False,
) -> Any:
    from jax.interpreters import mlir

    device_count = getattr(context.module_context.axis_context, "num_devices", None)
    if device_count not in (None, 1):
        raise_no_eligible_route(("native_structured_reduction_sharded",))
    if not native_available():
        raise_no_eligible_route(("native_structured_reduction_unavailable",))
    function = lambda value: _structured_reduction_ffi_call(
        value,
        descriptor=plan.descriptor,
        output_size=plan.semantic.output_size,
        output_dtype=jnp.dtype(plan.semantic.result_dtype),
        scalar_kind=plan.semantic.scalar_kind,
        grouped_output_owner=grouped_output_owner,
    )
    return mlir.lower_fun(function, multiple_results=False)(context, source)


def _non_cpu_lowering(
    context: Any,
    source: Any,
    *,
    plan: NativeCallSpec,
) -> Any:
    del context, source, plan
    raise_no_eligible_route(("native_structured_reduction_non_cpu",))


def _jvp(
    primals: tuple[Array],
    tangents: tuple[Any],
    *,
    plan: NativeCallSpec,
    primitive: Any,
) -> tuple[Array, Any]:
    from jax.interpreters import ad

    (source,), (tangent,) = primals, tangents
    primal = primitive.bind(source, plan=plan)
    if isinstance(tangent, ad.Zero):
        return primal, ad.Zero(jax.typeof(primal).to_tangent_aval())
    return primal, primitive.bind(tangent, plan=plan)


def _transpose(
    cotangent: Any,
    source: Any,
    *,
    plan: NativeCallSpec,
) -> list[Any]:
    from jax.interpreters import ad

    if not ad.is_undefined_primal(source):
        return [None]
    if isinstance(cotangent, ad.Zero):
        return [ad.Zero(source.aval.to_ct_aval())]
    reverse = transpose_plan(plan.semantic)
    if reverse.reduction_kind is StridedReductionKind.SUM:
        return [bind_native_structured_reduction(cotangent, reverse)]
    return [
        _execute_map(cotangent, plan=reverse)
    ]


def _batch(
    arguments: tuple[Array],
    dimensions: tuple[int | None],
    *,
    plan: NativeCallSpec,
    primitive: Any,
) -> tuple[Array, int | None]:
    from jax.interpreters import batching

    (source,), (dimension,) = arguments, dimensions
    if dimension is None:
        return primitive.bind(source, plan=plan), None
    source = batching.bdim_at_front(source, dimension, source.shape[dimension])
    return primitive.bind(source, plan=plan), 0


def _create_primitive(*, grouped_output_owner: bool = False) -> Any:
    from jax._src import dispatch
    from jax.extend import core
    from jax.interpreters import ad, batching, mlir, xla

    primitive = core.Primitive(
        "tensor0_stride_structured_reduction"
        f"{'_grouped' if grouped_output_owner else ''}"
    )
    primitive.def_impl(partial(xla.apply_primitive, primitive))
    primitive.def_abstract_eval(_abstract_eval)
    ad.primitive_jvps[primitive] = partial(_jvp, primitive=primitive)
    ad.primitive_transposes[primitive] = _transpose
    batching.primitive_batchers[primitive] = partial(
        _batch,
        primitive=primitive,
    )
    mlir.register_lowering(primitive, _non_cpu_lowering)
    mlir.register_lowering(
        primitive,
        partial(_lowering, grouped_output_owner=grouped_output_owner),
        platform="cpu",
    )
    dispatch.prim_requires_devices_during_lowering.add(primitive)
    return primitive


_STRUCTURED_REDUCTION_PRIMITIVE = _create_primitive()
_GROUPED_REDUCTION_PRIMITIVE = _create_primitive(grouped_output_owner=True)


def bind_native_broadcast_transpose(
    cotangent: Array,
    forward: AffinePlan,
) -> Array:
    """Bind the native structured transpose of one broadcast-read plan."""

    plan = lower_broadcast_transpose(forward)
    data = require_jax_array(cotangent, "broadcast cotangent")
    if data.dtype != jnp.dtype(plan.semantic.source_dtype):
        raise TypeError("broadcast cotangent dtype does not match the plan")
    if not data.shape or data.shape[-1] != forward.output_size:
        raise ValueError("broadcast cotangent storage size does not match the plan")
    return _STRUCTURED_REDUCTION_PRIMITIVE.bind(data, plan=plan)


def bind_native_structured_reduction(
    source: Array,
    semantic: AffinePlan,
) -> Array:
    """Bind one plan-owned native structured sum."""

    plan = lower_structured_reduction(semantic)
    data = require_jax_array(source, "structured reduction source")
    if data.dtype != jnp.dtype(plan.semantic.source_dtype):
        raise TypeError("structured reduction source dtype does not match the plan")
    if not data.shape or data.shape[-1] != semantic.source_size:
        raise ValueError("structured reduction source size does not match the plan")
    return _STRUCTURED_REDUCTION_PRIMITIVE.bind(data, plan=plan)


def _bind_sequential_native_structured_reduction(
    source: Array,
    semantic: AffinePlan,
) -> Array:
    """Bind the ungrouped reduction target used by serial test comparators.

    The target itself does not force serial execution. Tests that require a
    serial result must disable fiber parallelism until execution completes.
    """

    plan = lower_structured_reduction(semantic)
    data = require_jax_array(source, "sequential reduction source")
    if data.dtype != jnp.dtype(plan.semantic.source_dtype):
        raise TypeError("sequential reduction source dtype does not match the plan")
    if not data.shape or data.shape[-1] != semantic.source_size:
        raise ValueError("sequential reduction source size does not match the plan")
    return _STRUCTURED_REDUCTION_PRIMITIVE.bind(data, plan=plan)


def _bind_grouped_native_structured_reduction(
    source: Array,
    semantic: AffinePlan,
) -> Array:
    """Bind the gated exact-output-map grouped reduction candidate."""

    plan = lower_structured_reduction(semantic)
    data = require_jax_array(source, "grouped reduction source")
    if data.dtype != jnp.dtype(plan.semantic.source_dtype):
        raise TypeError("grouped reduction source dtype does not match the plan")
    if not data.shape or data.shape[-1] != semantic.source_size:
        raise ValueError("grouped reduction source size does not match the plan")
    return _GROUPED_REDUCTION_PRIMITIVE.bind(data, plan=plan)


def _execute_reduction(source: object, *, plan: AffinePlan) -> Array:
    """Execute one statically proved structured sum on the native CPU path."""

    data = require_jax_array(source, "_execute_reduction source")
    return bind_native_structured_reduction(data, plan)


__all__ = [
    "NATIVE_REDUCTION_ABI_VERSION",
    "NATIVE_REDUCTION_DESCRIPTOR_MAGIC",
    "bind_native_broadcast_transpose",
    "bind_native_structured_reduction",
    "lower_broadcast_transpose",
    "lower_structured_reduction",
    "_execute_reduction",
]
