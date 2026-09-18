"""JAX differentiation, batching and batch partitioning for native operations."""

from functools import partial

import jax
import jax.numpy as jnp
from jax._src import dispatch
from jax.extend import core
from jax.experimental.custom_partitioning import custom_partitioning
from jax.interpreters import ad, batching, mlir, xla
from jax.sharding import NamedSharding, PartitionSpec, auto_axes

from ._ffi._calls import execute_accumulation, execute_copy, execute_dot, execute_reduction, execute_update
from ._ffi._descriptor import encode_layout, encode_reduction_layout, merge_reduction_layouts
from ._layout import AffineRecord


def _copy(source, *, records, output_size, dtype):
    layout = encode_layout(records, source_size=source.shape[-1], output_size=output_size)
    return execute_copy(source, layout=layout, output_size=output_size, dtype=dtype)


def _batch_storage_sharding(shape):
    sharding = shape.sharding
    if not isinstance(sharding, NamedSharding):
        raise ValueError("native requires NamedSharding for multi-device execution")
    specification = tuple(sharding.spec) + (None,) * (len(shape.shape) - len(sharding.spec))
    if specification[-1] is not None:
        raise ValueError("native cannot shard the packed storage axis")
    return NamedSharding(sharding.mesh, PartitionSpec(*specification))


@partial(custom_partitioning, static_argnums=(1, 2, 3))
def _partitioned_copy(source, records, output_size, dtype):
    return _copy(source, records=records, output_size=output_size, dtype=dtype)


def _partition_copy(records, output_size, dtype, mesh, argument_shapes, result_shape):
    source_sharding = _batch_storage_sharding(argument_shapes[0])
    _batch_storage_sharding(result_shape)
    return (mesh, partial(_copy, records=records, output_size=output_size, dtype=dtype),
            source_sharding, (source_sharding,))


def _infer_copy_sharding(records, output_size, dtype, mesh, argument_shapes, result_shape):
    return _batch_storage_sharding(argument_shapes[0])


def _propagate_copy_sharding(records, output_size, dtype, mesh, user_shape):
    return _batch_storage_sharding(user_shape)


_partitioned_copy.def_partition(
    partition=_partition_copy,
    infer_sharding_from_operands=_infer_copy_sharding,
    propagate_user_sharding=_propagate_copy_sharding,
    sharding_rule="... source -> ... result",
)


def _reduction(source, *coefficients, records, output_shapes, reduction_axes,
               output_size, dtype, coefficient_records=()):
    layouts = tuple(encode_reduction_layout(
        source_shape=record.logical_shape, source_strides=record.source_strides,
        source_offset=record.source_offset, output_shape=output_shape,
        output_strides=record.destination_strides, output_offset=record.destination_offset,
        reduction_axes=axes, source_size=source.shape[-1], output_size=output_size,
    ) for record, output_shape, axes in zip(records, output_shapes, reduction_axes, strict=True))
    layout = merge_reduction_layouts(
        layouts, source_size=source.shape[-1], output_size=output_size,
    )
    return execute_reduction(
        source, coefficients, coefficient_records=coefficient_records,
        layout=layout, output_size=output_size, dtype=dtype,
    )


@partial(custom_partitioning, static_argnums=(0, 1, 2, 3, 4, 5))
def _partitioned_reduction(records, output_shapes, reduction_axes, output_size, dtype,
                           coefficient_records, source, *coefficients):
    return _reduction(source, *coefficients, records=records, output_shapes=output_shapes,
                      reduction_axes=reduction_axes, output_size=output_size, dtype=dtype,
                      coefficient_records=coefficient_records)


def _partition_reduction(records, output_shapes, reduction_axes, output_size, dtype,
                         coefficient_records, mesh, argument_shapes, result_shape):
    source_sharding = _batch_storage_sharding(argument_shapes[0])
    _batch_storage_sharding(result_shape)
    coefficient_shardings = tuple(NamedSharding(mesh, PartitionSpec() if value.shape in ((), (1,))
        else PartitionSpec(*source_sharding.spec[:-1])) for value in argument_shapes[1:])
    return (mesh, partial(_reduction, records=records, output_shapes=output_shapes,
                          reduction_axes=reduction_axes, output_size=output_size, dtype=dtype,
                          coefficient_records=coefficient_records),
            source_sharding, (source_sharding, *coefficient_shardings))


def _infer_reduction_sharding(records, output_shapes, reduction_axes, output_size, dtype,
                              coefficient_records, mesh, argument_shapes, result_shape):
    return _batch_storage_sharding(argument_shapes[0])


def _propagate_reduction_sharding(records, output_shapes, reduction_axes, output_size, dtype,
                                  coefficient_records, mesh, user_shape):
    return _batch_storage_sharding(user_shape)


def _reduction_sharding_rule(records, output_shapes, reduction_axes, output_size, dtype,
                             coefficient_records, mesh, value_types, result_types):
    return _coefficient_sharding_rule(value_types)


_partitioned_reduction.def_partition(
    partition=_partition_reduction,
    infer_sharding_from_operands=_infer_reduction_sharding,
    propagate_user_sharding=_propagate_reduction_sharding,
    sharding_rule=_reduction_sharding_rule,
)


def _storage_abstract(source, *coefficients, output_size, dtype, **parameters):
    _batch_storage_sharding(source)
    return source.update(shape=(*source.shape[:-1], output_size), dtype=dtype, weak_type=False)


def _accumulation(source, *coefficients, records, output_size, dtype, coefficient_records=()):
    layout = encode_layout(records, source_size=source.shape[-1], output_size=output_size)
    return execute_accumulation(
        source, coefficients, coefficient_records=coefficient_records,
        layout=layout, output_size=output_size, dtype=dtype,
    )


@partial(custom_partitioning, static_argnums=(0, 1, 2, 3))
def _partitioned_accumulation(records, output_size, dtype, coefficient_records, source, *coefficients):
    return _accumulation(source, *coefficients, records=records, output_size=output_size,
                         dtype=dtype, coefficient_records=coefficient_records)


def _partition_accumulation(records, output_size, dtype, coefficient_records, mesh, argument_shapes, result_shape):
    source_sharding = _batch_storage_sharding(argument_shapes[0])
    _batch_storage_sharding(result_shape)
    coefficient_shardings = tuple(NamedSharding(mesh, PartitionSpec() if value.shape in ((), (1,))
        else PartitionSpec(*source_sharding.spec[:-1])) for value in argument_shapes[1:])
    return (mesh, partial(_accumulation, records=records, output_size=output_size,
                          dtype=dtype, coefficient_records=coefficient_records),
            source_sharding, (source_sharding, *coefficient_shardings))


def _infer_accumulation_sharding(records, output_size, dtype, coefficient_records, mesh, argument_shapes, result_shape):
    return _batch_storage_sharding(argument_shapes[0])


def _propagate_accumulation_sharding(records, output_size, dtype, coefficient_records, mesh, user_shape):
    return _batch_storage_sharding(user_shape)


def _accumulation_sharding_rule(records, output_size, dtype, coefficient_records, mesh, value_types, result_types):
    return _coefficient_sharding_rule(value_types)


def _coefficient_sharding_rule(value_types):
    from jaxlib.mlir import ir

    shapes = tuple(tuple(ir.RankedTensorType(value).shape) for value in value_types)
    batch_axes = tuple(f"batch{axis}" for axis in range(len(shapes[0]) - 1))
    operands = [" ".join((*batch_axes, "source"))]
    for index, shape in enumerate(shapes[1:]):
        operands.append("" if not shape else f"shared{index}" if shape == (1,) else " ".join(batch_axes))
    return f"{', '.join(operands)} -> {' '.join((*batch_axes, 'result'))}"


_partitioned_accumulation.def_partition(
    partition=_partition_accumulation,
    infer_sharding_from_operands=_infer_accumulation_sharding,
    propagate_user_sharding=_propagate_accumulation_sharding,
    sharding_rule=_accumulation_sharding_rule,
)


def _require_inexact_ad_dtypes(*dtypes):
    if any(value.name not in ("float16", "bfloat16", "float32", "float64", "complex64", "complex128") for value in dtypes):
        raise NotImplementedError("native AD supports F16/BF16/F32/F64/C64/C128 differentiated operands")


def _accumulation_jvp(primals, tangents, *, records, output_size, dtype, coefficient_records=()):
    source, *coefficients = primals
    source_tangent, *coefficient_tangents = tangents
    parameters = dict(records=records, output_size=output_size, dtype=dtype,
                      coefficient_records=coefficient_records)
    primal = accumulation_p.bind(*primals, **parameters)
    if not jnp.issubdtype(primal.dtype, jnp.inexact):
        return primal, ad.Zero(jax.typeof(primal).to_tangent_aval())
    if not isinstance(source_tangent, ad.Zero):
        _require_inexact_ad_dtypes(source.dtype, dtype)
    if any(not isinstance(value, ad.Zero) for value in coefficient_tangents):
        _require_inexact_ad_dtypes(dtype,
            *(value.dtype for value, direction in zip(coefficients, coefficient_tangents, strict=True)
              if not isinstance(direction, ad.Zero)))
    tangent = None
    if not isinstance(source_tangent, ad.Zero):
        tangent = accumulation_p.bind(source_tangent, *coefficients, **parameters)
    active = tuple((record, value) for record, value in zip(coefficient_records, coefficient_tangents, strict=True)
                   if not isinstance(value, ad.Zero))
    if active:
        contribution = accumulation_p.bind(
            source, *(value for _, value in active), records=tuple(records[index] for index, _ in active),
            coefficient_records=tuple(range(len(active))), output_size=output_size, dtype=dtype,
        )
        tangent = contribution if tangent is None else tangent + contribution
    if tangent is None:
        tangent = ad.Zero(jax.typeof(primal).to_tangent_aval())
    return primal, tangent


def _accumulation_transpose(cotangent, source, *coefficients, records, output_size, dtype,
                            coefficient_records=()):
    source_active = ad.is_undefined_primal(source)
    active = tuple(ad.is_undefined_primal(value) for value in coefficients)
    source_aval = source.aval if source_active else jax.typeof(source)
    coefficient_avals = tuple(value.aval if undefined else jax.typeof(value)
                             for value, undefined in zip(coefficients, active, strict=True))
    if source_active:
        _require_inexact_ad_dtypes(source_aval.dtype, dtype)
    if any(active):
        _require_inexact_ad_dtypes(dtype,
            *(value.dtype for value, undefined in zip(coefficient_avals, active, strict=True) if undefined))
    if isinstance(cotangent, ad.Zero):
        return [ad.Zero(source_aval.to_ct_aval()) if source_active else None,
                *(ad.Zero(value.to_ct_aval()) if undefined else None
                  for value, undefined in zip(coefficient_avals, active, strict=True))]
    if source_active and any(active):
        raise NotImplementedError("accumulation transpose requires a known source or known coefficients")
    source_gradient = None
    if source_active:
        reverse_records = tuple(AffineRecord(
            record.logical_shape, record.destination_strides, record.destination_offset,
            record.source_strides, record.source_offset,
        ) for record in records)
        cotangent = cotangent.astype(jnp.promote_types(cotangent.dtype, source_aval.dtype))
        source_gradient = accumulation_p.bind(
            cotangent, *coefficients, records=reverse_records, coefficient_records=coefficient_records,
            output_size=source_aval.shape[-1], dtype=source_aval.dtype,
        )
    coefficient_gradients = []
    for index, value, undefined in zip(coefficient_records, coefficient_avals, active, strict=True):
        if not undefined:
            coefficient_gradients.append(None)
            continue
        gradient = dot_p.bind(source, cotangent.astype(jnp.promote_types(cotangent.dtype, value.dtype)),
                              records=(records[index],), conjugate_left=False, dtype=value.dtype)
        if value.shape in ((), (1,)):
            gradient = jnp.sum(gradient).reshape(value.shape)
        coefficient_gradients.append(gradient)
    return [source_gradient, *coefficient_gradients]


def _update(source, base, alpha, beta, *, records):
    layout = encode_layout(records, source_size=source.shape[-1], output_size=base.shape[-1])
    return execute_update(source, base, alpha, beta, layout=layout)


def _update_abstract(source, base, alpha, beta, *, records):
    _batch_storage_sharding(source)
    _batch_storage_sharding(base)
    return base.update(weak_type=False)


@partial(custom_partitioning, static_argnums=(4,))
def _partitioned_update(source, base, alpha, beta, records):
    return _update(source, base, alpha, beta, records=records)


def _partition_update(records, mesh, argument_shapes, result_shape):
    _batch_storage_sharding(argument_shapes[0])
    storage_sharding = _batch_storage_sharding(argument_shapes[1])
    _batch_storage_sharding(result_shape)
    coefficient_shardings = tuple(NamedSharding(mesh, PartitionSpec() if value.shape in ((), (1,))
        else PartitionSpec(*storage_sharding.spec[:-1])) for value in argument_shapes[2:])
    return (mesh, partial(_update, records=records), storage_sharding,
            (storage_sharding, storage_sharding, *coefficient_shardings))


def _infer_update_sharding(records, mesh, argument_shapes, result_shape):
    _batch_storage_sharding(argument_shapes[0])
    return _batch_storage_sharding(argument_shapes[1])


def _propagate_update_sharding(records, mesh, user_shape):
    return _batch_storage_sharding(user_shape)


def _update_sharding_rule(records, mesh, value_types, result_types):
    from jaxlib.mlir import ir

    shapes = tuple(tuple(ir.RankedTensorType(value).shape) for value in value_types)
    batch_axes = tuple(f"batch{axis}" for axis in range(len(shapes[1]) - 1))
    operands = [" ".join((*batch_axes, storage)) for storage in ("source", "result")]
    for index, shape in enumerate(shapes[2:]):
        operands.append("" if not shape else f"shared{index}" if shape == (1,) else " ".join(batch_axes))
    return f"{', '.join(operands)} -> {' '.join((*batch_axes, 'result'))}"


_partitioned_update.def_partition(
    partition=_partition_update,
    infer_sharding_from_operands=_infer_update_sharding,
    propagate_user_sharding=_propagate_update_sharding,
    sharding_rule=_update_sharding_rule,
)


def _dot(left, right, *, records, conjugate_left, dtype=None):
    layout = encode_layout(records, source_size=left.shape[-1], output_size=right.shape[-1])
    return execute_dot(left, right, layout=layout, conjugate_left=conjugate_left, dtype=dtype)


def _dot_batch_sharding(shape):
    storage_sharding = _batch_storage_sharding(shape)
    return NamedSharding(storage_sharding.mesh, PartitionSpec(*storage_sharding.spec[:-1]))


@partial(custom_partitioning, static_argnums=(2, 3, 4))
def _partitioned_dot(left, right, records, conjugate_left, dtype=None):
    return _dot(left, right, records=records, conjugate_left=conjugate_left, dtype=dtype)


def _partition_dot(records, conjugate_left, dtype, mesh, argument_shapes, result_shape):
    result_sharding = _dot_batch_sharding(argument_shapes[0])
    _batch_storage_sharding(argument_shapes[1])
    storage_sharding = NamedSharding(mesh, PartitionSpec(*result_sharding.spec, None))
    return (mesh, partial(_dot, records=records, conjugate_left=conjugate_left, dtype=dtype),
            result_sharding, (storage_sharding, storage_sharding))


def _infer_dot_sharding(records, conjugate_left, dtype, mesh, argument_shapes, result_shape):
    return _dot_batch_sharding(argument_shapes[0])


_partitioned_dot.def_partition(
    partition=_partition_dot,
    infer_sharding_from_operands=_infer_dot_sharding,
    sharding_rule="... left, ... right -> ...",
)


def _dot_abstract(left, right, *, records, conjugate_left, dtype=None):
    if not left.shape or not right.shape:
        raise ValueError("dot storage must have at least one dimension")
    if left.shape[:-1] != right.shape[:-1]:
        raise ValueError("dot batch shapes must match")
    result_dtype = jnp.result_type(left.dtype, right.dtype) if dtype is None else jax.dtypes.canonicalize_dtype(dtype)
    for operand_dtype in (left.dtype, right.dtype, result_dtype):
        if operand_dtype.name not in (
                "bool", "int8", "int16", "int32", "int64",
                "uint8", "uint16", "uint32", "uint64", "float16", "bfloat16",
                "float32", "float64", "complex64", "complex128"):
            raise NotImplementedError(f"native dot does not support {operand_dtype}")
    result_sharding = _dot_batch_sharding(left)
    _batch_storage_sharding(right)
    return left.update(shape=left.shape[:-1], dtype=result_dtype, weak_type=False, sharding=result_sharding)


def _dot_jvp(primals, tangents, *, records, conjugate_left, dtype=None):
    left, right = primals
    left_tangent, right_tangent = tangents
    primal = dot_p.bind(left, right, records=records, conjugate_left=conjugate_left, dtype=dtype)
    left_zero = isinstance(left_tangent, ad.Zero)
    right_zero = isinstance(right_tangent, ad.Zero)
    if not jnp.issubdtype(primal.dtype, jnp.inexact) or (left_zero and right_zero):
        return primal, ad.Zero(jax.typeof(primal).to_tangent_aval())
    if left_zero:
        return primal, dot_p.bind(left, right_tangent, records=records, conjugate_left=conjugate_left, dtype=dtype)
    tangent = dot_p.bind(left_tangent, right, records=records, conjugate_left=conjugate_left, dtype=dtype)
    if not right_zero:
        tangent = tangent + dot_p.bind(left, right_tangent, records=records, conjugate_left=conjugate_left, dtype=dtype)
    return primal, tangent


def _dot_transpose(cotangent, left, right, *, records, conjugate_left, dtype=None):
    left_active = ad.is_undefined_primal(left)
    right_active = ad.is_undefined_primal(right)
    if isinstance(cotangent, ad.Zero):
        return [ad.Zero(left.aval.to_ct_aval()) if left_active else None,
                ad.Zero(right.aval.to_ct_aval()) if right_active else None]
    if left_active and right_active:
        raise NotImplementedError("dot transpose requires one known input")
    if not left_active and not right_active:
        return [None, None]
    if left_active:
        records = tuple(AffineRecord(
            record.logical_shape, record.destination_strides, record.destination_offset,
            record.source_strides, record.source_offset,
        ) for record in records)
    source = right if left_active else left
    result_aval = left.aval if left_active else right.aval
    cotangent = cotangent.astype(jnp.promote_types(cotangent.dtype, result_aval.dtype))
    factor = jnp.conj(cotangent) if conjugate_left and right_active else cotangent
    gradient = accumulation_p.bind(
        source, *((factor,) * len(records)), records=records,
        coefficient_records=tuple(range(len(records))),
        output_size=result_aval.shape[-1], dtype=result_aval.dtype,
    )
    if conjugate_left:
        gradient = jnp.conj(gradient)
    return [gradient, None] if left_active else [None, gradient]


def _dot_batch(arguments, dimensions, *, records, conjugate_left, dtype=None):
    if all(axis is None for axis in dimensions):
        return dot_p.bind(*arguments, records=records, conjugate_left=conjugate_left, dtype=dtype), None
    size = next(value.shape[axis] for value, axis in zip(arguments, dimensions, strict=True)
                if axis is not None)
    values = tuple(batching.bdim_at_front(value, axis, size)
                   for value, axis in zip(arguments, dimensions, strict=True))
    return dot_p.bind(*values, records=records, conjugate_left=conjugate_left, dtype=dtype), 0


def _require_linear_dtype(source_dtype, result_dtype):
    if not all(jnp.issubdtype(dtype, jnp.inexact) for dtype in (source_dtype, result_dtype)):
        raise NotImplementedError("native update AD requires floating or complex storage")


def _jvp(primitive, primals, tangents, **parameters):
    source, *coefficients = primals
    source_tangent, *coefficient_tangents = tangents
    primal = primitive.bind(source, *coefficients, **parameters)
    if not jnp.issubdtype(primal.dtype, jnp.inexact):
        return primal, ad.Zero(jax.typeof(primal).to_tangent_aval())
    if not isinstance(source_tangent, ad.Zero):
        _require_inexact_ad_dtypes(source.dtype, parameters["dtype"])
    active = tuple((index, value) for index, value in zip(
        parameters.get("coefficient_records", ()), coefficient_tangents, strict=True)
        if not isinstance(value, ad.Zero))
    if active:
        _require_inexact_ad_dtypes(parameters["dtype"],
            *(value.dtype for value, direction in zip(coefficients, coefficient_tangents, strict=True)
              if not isinstance(direction, ad.Zero)))
    tangent = None
    if not isinstance(source_tangent, ad.Zero):
        tangent = primitive.bind(source_tangent, *coefficients, **parameters)
    if active:
        contribution = primitive.bind(
            source, *(value for _, value in active),
            records=tuple(parameters["records"][index] for index, _ in active),
            output_shapes=tuple(parameters["output_shapes"][index] for index, _ in active),
            reduction_axes=tuple(parameters["reduction_axes"][index] for index, _ in active),
            coefficient_records=tuple(range(len(active))),
            output_size=parameters["output_size"], dtype=parameters["dtype"],
        )
        tangent = contribution if tangent is None else tangent + contribution
    if tangent is None:
        tangent = ad.Zero(jax.typeof(primal).to_tangent_aval())
    return primal, tangent


def _transpose(primitive, cotangent, source, *coefficients, records, output_size, dtype,
               output_shapes=None, reduction_axes=None, coefficient_records=()):
    source_active = ad.is_undefined_primal(source)
    active = tuple(ad.is_undefined_primal(value) for value in coefficients)
    if not source_active and not any(active):
        return [None] * (1 + len(coefficients))
    source_aval = source.aval if source_active else jax.typeof(source)
    coefficient_avals = tuple(value.aval if undefined else jax.typeof(value)
                             for value, undefined in zip(coefficients, active, strict=True))
    if source_active:
        _require_inexact_ad_dtypes(source_aval.dtype, dtype)
    if any(active):
        _require_inexact_ad_dtypes(dtype,
            *(value.dtype for value, undefined in zip(coefficient_avals, active, strict=True) if undefined))
    if isinstance(cotangent, ad.Zero):
        return [ad.Zero(source_aval.to_ct_aval()) if source_active else None,
                *(ad.Zero(value.to_ct_aval()) if undefined else None
                  for value, undefined in zip(coefficient_avals, active, strict=True))]
    if source_active and any(active):
        raise NotImplementedError("reduction transpose requires a known source or known coefficients")
    if reduction_axes is None:
        reduction_axes = tuple((False,) * len(record.logical_shape) for record in records)
    reverse_records = []
    for record, axes in zip(records, reduction_axes, strict=True):
        read_strides = tuple(0 if reduced else stride for reduced, stride in zip(
            axes, record.destination_strides, strict=True))
        reverse_records.append(AffineRecord(
            record.logical_shape, read_strides, record.destination_offset,
            record.source_strides, record.source_offset,
        ))
    source_gradient = None
    if source_active:
        if primitive is copy_p:
            if jnp.issubdtype(source_aval.dtype, jnp.floating):
                cotangent = jnp.real(cotangent)
            cotangent = cotangent.astype(source_aval.dtype)
        else:
            cotangent = cotangent.astype(jnp.promote_types(cotangent.dtype, source_aval.dtype))
        source_gradient = accumulation_p.bind(
            cotangent, *coefficients, coefficient_records=coefficient_records,
            records=tuple(reverse_records), output_size=source_aval.shape[-1], dtype=source_aval.dtype,
        )
    coefficient_gradients = []
    for index, value, undefined in zip(coefficient_records, coefficient_avals, active, strict=True):
        if not undefined:
            coefficient_gradients.append(None)
            continue
        gradient = dot_p.bind(cotangent.astype(jnp.promote_types(cotangent.dtype, value.dtype)), source,
                              records=(reverse_records[index],), conjugate_left=False, dtype=value.dtype)
        if value.shape in ((), (1,)):
            gradient = jnp.sum(gradient).reshape(value.shape)
        coefficient_gradients.append(gradient)
    return [source_gradient, *coefficient_gradients]


def _batch_coefficients(coefficients, dimensions, size, storage):
    batch_shape = storage.shape[:-1]
    sharding = jax.typeof(storage).sharding
    mesh_axis = sharding.spec[0] if sharding.mesh.explicit_axes else None
    coefficient_sharding = (sharding.update(spec=PartitionSpec(*sharding.spec[:-1]))
                            if sharding.mesh.explicit_axes else None)
    result = []
    for value, axis in zip(coefficients, dimensions, strict=True):
        if axis is None and value.shape in ((), (1,)):
            result.append(value)
        elif axis is not None and value.shape[:axis] + value.shape[axis + 1:] in ((), (1,)):
            coefficient = batching.bdim_at_front(value, axis, size).reshape((size,))
            result.append(jax.lax.broadcast_in_dim(
                coefficient, batch_shape, (0,), out_sharding=coefficient_sharding))
        else:
            result.append(batching.broadcast(value, size, 0, mesh_axis) if axis is None else
                          batching.bdim_at_front(value, axis, size))
    return result


def _batch(primitive, arguments, dimensions, **parameters):
    source, *coefficients = arguments
    dimension, *coefficient_dimensions = dimensions
    if all(axis is None for axis in dimensions):
        return primitive.bind(*arguments, **parameters), None
    if source.ndim == (dimension is not None):
        raise ValueError("storage must have at least one dimension")
    mapped_index = next(index for index, axis in enumerate(dimensions) if axis is not None)
    size = arguments[mapped_index].shape[dimensions[mapped_index]]
    mapped = batching.bdim_at_front(arguments[mapped_index], dimensions[mapped_index], size)
    sharding = jax.typeof(mapped).sharding
    mesh_axis = sharding.spec[0] if sharding.mesh.explicit_axes else None
    source = (batching.broadcast(source, size, 0, mesh_axis) if dimension is None else
              batching.bdim_at_front(source, dimension, size))
    coefficients = _batch_coefficients(coefficients, coefficient_dimensions, size, source)
    return primitive.bind(source, *coefficients, **parameters), 0


def _update_jvp(primals, tangents, *, records):
    source, base, alpha, beta = primals
    source_tangent, base_tangent, alpha_tangent, beta_tangent = tangents
    primal = update_p.bind(*primals, records=records)
    if not jnp.issubdtype(primal.dtype, jnp.inexact):
        return primal, ad.Zero(jax.typeof(primal).to_tangent_aval())
    _require_linear_dtype(source.dtype, base.dtype)
    active_coefficients = tuple(value.dtype for value, tangent in
                                ((alpha, alpha_tangent), (beta, beta_tangent))
                                if not isinstance(tangent, ad.Zero))
    if active_coefficients:
        _require_inexact_ad_dtypes(source.dtype, base.dtype, *active_coefficients)
    source_zero = isinstance(source_tangent, ad.Zero)
    base_zero = isinstance(base_tangent, ad.Zero)
    tangent = None
    if not (source_zero and base_zero):
        tangent = update_p.bind(
            ad.instantiate_zeros(source_tangent), ad.instantiate_zeros(base_tangent),
            jnp.int32(0) if source_zero else alpha, jnp.int32(0) if base_zero else beta,
            records=records,
        )
    if active_coefficients:
        base_records = tuple(AffineRecord(
            record.logical_shape, record.destination_strides, record.destination_offset,
            record.destination_strides, record.destination_offset,
        ) for record in records)
        for value, direction, layouts in ((source, alpha_tangent, records), (base, beta_tangent, base_records)):
            if isinstance(direction, ad.Zero):
                continue
            contribution = accumulation_p.bind(
                value, *((direction,) * len(layouts)), records=layouts,
                coefficient_records=tuple(range(len(layouts))),
                output_size=base.shape[-1], dtype=base.dtype,
            )
            tangent = contribution if tangent is None else tangent + contribution
    if tangent is None:
        tangent = ad.Zero(jax.typeof(primal).to_tangent_aval())
    return primal, tangent


def _update_transpose(cotangent, source, base, alpha, beta, *, records):
    arguments = (source, base, alpha, beta)
    active = tuple(ad.is_undefined_primal(value) for value in arguments)
    source_active, base_active, alpha_active, beta_active = active
    if not any(active):
        return [None, None, None, None]
    avals = tuple(value.aval if undefined else jax.typeof(value)
                  for value, undefined in zip(arguments, active, strict=True))
    source_aval, base_aval, alpha_aval, beta_aval = avals
    _require_linear_dtype(source_aval.dtype, base_aval.dtype)
    if alpha_active or beta_active:
        _require_inexact_ad_dtypes(source_aval.dtype, base_aval.dtype,
            *(value.dtype for value, undefined in zip(avals[2:], active[2:], strict=True) if undefined))
    if isinstance(cotangent, ad.Zero):
        return [ad.Zero(value.to_ct_aval()) if undefined else None
                for value, undefined in zip(avals, active, strict=True)]
    if (source_active and alpha_active) or (base_active and beta_active):
        raise NotImplementedError("update transpose requires a known operand in each product")
    base_records = tuple(AffineRecord(
        record.logical_shape, record.destination_strides, record.destination_offset,
        record.destination_strides, record.destination_offset,
    ) for record in records)
    source_gradient = None
    if source_active:
        reverse_records = tuple(AffineRecord(
            record.logical_shape, record.destination_strides, record.destination_offset,
            record.source_strides, record.source_offset,
        ) for record in records)
        source_gradient = accumulation_p.bind(
            cotangent, *((alpha,) * len(records)), coefficient_records=tuple(range(len(records))),
            records=reverse_records,
            output_size=source_aval.shape[-1], dtype=source_aval.dtype,
        )
    base_gradient = None
    if base_active:
        base_gradient = update_p.bind(
            cotangent, cotangent, beta, jnp.int32(0), records=base_records,
        )
    coefficient_gradients = []
    for value, layouts, aval, undefined in ((source, records, alpha_aval, alpha_active),
                                            (base, base_records, beta_aval, beta_active)):
        if not undefined:
            coefficient_gradients.append(None)
            continue
        gradient = dot_p.bind(value, cotangent.astype(jnp.promote_types(cotangent.dtype, aval.dtype)),
                              records=layouts, conjugate_left=False, dtype=aval.dtype)
        if aval.shape in ((), (1,)):
            gradient = jnp.sum(gradient).reshape(aval.shape)
        coefficient_gradients.append(gradient)
    return [source_gradient, base_gradient, *coefficient_gradients]


def _update_batch(arguments, dimensions, *, records):
    if all(axis is None for axis in dimensions):
        return update_p.bind(*arguments, records=records), None
    if any(value.ndim == (axis is not None)
           for value, axis in zip(arguments[:2], dimensions[:2], strict=True)):
        raise ValueError("update storage buffers must have at least one dimension")
    mapped_index = next(index for index, axis in enumerate(dimensions) if axis is not None)
    size = arguments[mapped_index].shape[dimensions[mapped_index]]
    mapped = batching.bdim_at_front(arguments[mapped_index], dimensions[mapped_index], size)
    sharding = jax.typeof(mapped).sharding
    mesh_axis = sharding.spec[0] if sharding.mesh.explicit_axes else None
    source, base = (batching.broadcast(value, size, 0, mesh_axis) if axis is None else
                    batching.bdim_at_front(value, axis, size)
                    for value, axis in zip(arguments[:2], dimensions[:2], strict=True))
    coefficients = _batch_coefficients(arguments[2:], dimensions[2:], size, base)
    return update_p.bind(source, base, *coefficients, records=records), 0


def _lower(implementation, context, *arguments, **parameters):
    return mlir.lower_fun(partial(implementation, **parameters), multiple_results=False)(context, *arguments)


def _lower_copy(context, source, **parameters):
    function = partial(_partitioned_copy, **parameters)
    if context.avals_in[0].sharding.mesh.explicit_axes:
        function = auto_axes(function, out_sharding=context.avals_out[0].sharding)
    return _lower(function, context, source)


def _lower_accumulation(context, *arguments, records, output_size, dtype, coefficient_records=()):
    function = partial(_partitioned_accumulation, records, output_size, dtype, coefficient_records)
    if context.avals_in[0].sharding.mesh.explicit_axes:
        function = auto_axes(function, out_sharding=context.avals_out[0].sharding)
    return _lower(function, context, *arguments)


def _lower_dot(context, left, right, **parameters):
    function = partial(_partitioned_dot, **parameters)
    if context.avals_in[0].sharding.mesh.explicit_axes:
        function = auto_axes(function, out_sharding=context.avals_out[0].sharding)
    return _lower(function, context, left, right)


def _lower_reduction(context, *arguments, records, output_shapes, reduction_axes,
                      output_size, dtype, coefficient_records=()):
    function = partial(_partitioned_reduction, records, output_shapes, reduction_axes,
                       output_size, dtype, coefficient_records)
    if context.avals_in[0].sharding.mesh.explicit_axes:
        function = auto_axes(function, out_sharding=context.avals_out[0].sharding)
    return _lower(function, context, *arguments)


def _lower_update(context, *arguments, records):
    function = partial(_partitioned_update, records=records)
    if context.avals_in[1].sharding.mesh.explicit_axes:
        function = auto_axes(function, out_sharding=context.avals_out[0].sharding)
    return _lower(function, context, *arguments)


copy_p = core.Primitive("tensor0_stride_copy")
reduction_p = core.Primitive("tensor0_stride_reduction")
update_p = core.Primitive("tensor0_stride_update")
dot_p = core.Primitive("tensor0_stride_dot")
accumulation_p = core.Primitive("tensor0_stride_accumulation")
for primitive, implementation in (
        (copy_p, _partitioned_copy), (reduction_p, _reduction), (update_p, _update), (dot_p, _dot),
        (accumulation_p, _accumulation)):
    primitive.def_impl(partial(xla.apply_primitive, primitive))
    if primitive is copy_p:
        lowering = _lower_copy
    elif primitive is accumulation_p:
        lowering = _lower_accumulation
    elif primitive is dot_p:
        lowering = _lower_dot
    elif primitive is reduction_p:
        lowering = _lower_reduction
    elif primitive is update_p:
        lowering = _lower_update
    else:
        lowering = partial(_lower, implementation)
    mlir.register_lowering(primitive, lowering, platform="cpu")
    dispatch.prim_requires_devices_during_lowering.add(primitive)

for primitive in (copy_p, reduction_p):
    primitive.def_abstract_eval(_storage_abstract)
    ad.primitive_jvps[primitive] = partial(_jvp, primitive)
    ad.primitive_transposes[primitive] = partial(_transpose, primitive)
    batching.primitive_batchers[primitive] = partial(_batch, primitive)

update_p.def_abstract_eval(_update_abstract)
ad.primitive_jvps[update_p] = _update_jvp
ad.primitive_transposes[update_p] = _update_transpose
batching.primitive_batchers[update_p] = _update_batch

dot_p.def_abstract_eval(_dot_abstract)
ad.primitive_jvps[dot_p] = _dot_jvp
ad.primitive_transposes[dot_p] = _dot_transpose
batching.primitive_batchers[dot_p] = _dot_batch

accumulation_p.def_abstract_eval(_storage_abstract)
ad.primitive_jvps[accumulation_p] = _accumulation_jvp
ad.primitive_transposes[accumulation_p] = _accumulation_transpose
batching.primitive_batchers[accumulation_p] = partial(_batch, accumulation_p)
