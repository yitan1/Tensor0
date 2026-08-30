"""Automatic StableHLO recipes for regular stride layouts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Never, TypeAlias, assert_never

import jax
from jax import Array
import jax.numpy as jnp

from ._compiler import CompiledStridePlan, LayoutKind, PlanKey
from ._plan import CompleteMode, UINT64_MAX, _stable_key_bytes


STABLEHLO_RECIPE_POLICY_VERSION = 2

_SUPPORTED_DTYPES = frozenset({"float16", "bfloat16", "float32", "complex64", "int32"})
_PORTABLE_DTYPES = frozenset(
    {
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float16",
        "bfloat16",
        "float32",
        "float64",
        "complex64",
        "complex128",
    }
)


class StableHloProofError(ValueError):
    """A stable, mechanically checkable recipe-ineligibility result."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True, slots=True)
class AffineIntervalProof:
    """Factor one positive affine map into a contiguous physical interval."""

    logical_shape: tuple[int, ...]
    logical_axes_slowest_first: tuple[int, ...]
    physical_shape: tuple[int, ...]
    mapped_physical_axes: tuple[int, ...]
    interval_elements: int

    @property
    def key(self) -> PlanKey:
        return _stable_key_bytes(
            "tensor0-stride-affine-interval-proof-v1",
            self.logical_shape,
            self.logical_axes_slowest_first,
            self.physical_shape,
            self.mapped_physical_axes,
            self.interval_elements,
        )


@dataclass(frozen=True, slots=True)
class CompactStableHloRecipe:
    """One verified flat compact-slice/scale recipe."""

    compiled: CompiledStridePlan = field(compare=False, hash=False, repr=False)
    source_offset: int
    element_count: int
    capability_key: PlanKey
    recipe_key: PlanKey
    policy_version: int = STABLEHLO_RECIPE_POLICY_VERSION


@dataclass(frozen=True, slots=True)
class CompactTransposeStableHloRecipe:
    """One verified compact transpose expressed as scale and zero padding."""

    compiled: CompiledStridePlan = field(compare=False, hash=False, repr=False)
    destination_offset: int
    element_count: int
    capability_key: PlanKey
    recipe_key: PlanKey
    policy_version: int = STABLEHLO_RECIPE_POLICY_VERSION


@dataclass(frozen=True, slots=True)
class CompactAffineInterval:
    """Certified contiguous source and destination intervals for one map."""

    compiled: CompiledStridePlan = field(compare=False, hash=False, repr=False)
    source_offset: int
    destination_offset: int
    element_count: int
    capability_key: PlanKey
    policy_version: int = STABLEHLO_RECIPE_POLICY_VERSION


@dataclass(frozen=True, slots=True)
class AffineSingleRecordStableHloRecipe:
    """One verified positive-affine read and dense destination permutation."""

    compiled: CompiledStridePlan = field(compare=False, hash=False, repr=False)
    source: AffineIntervalProof
    destination: AffineIntervalProof
    capability_key: PlanKey
    recipe_key: PlanKey
    policy_version: int = STABLEHLO_RECIPE_POLICY_VERSION


@dataclass(frozen=True, slots=True)
class AffineTransposeStableHloRecipe:
    """One verified dense-input map into a positive-affine zero-filled result."""

    compiled: CompiledStridePlan = field(compare=False, hash=False, repr=False)
    source: AffineIntervalProof
    destination: AffineIntervalProof
    capability_key: PlanKey
    recipe_key: PlanKey
    policy_version: int = STABLEHLO_RECIPE_POLICY_VERSION


@dataclass(frozen=True, slots=True)
class PortableAffineRecord:
    """One positive-affine record lowered without an address array."""

    record_index: int
    source: AffineIntervalProof
    destination: AffineIntervalProof

    @property
    def key(self) -> PlanKey:
        return _stable_key_bytes(
            "tensor0-stride-portable-affine-record-v1",
            self.record_index,
            self.source.key,
            self.destination.key,
        )


@dataclass(frozen=True, slots=True)
class PortableAffineStableHloRecipe:
    """A record-wise portable affine map with no element address array."""

    compiled: CompiledStridePlan = field(compare=False, hash=False, repr=False)
    records: tuple[PortableAffineRecord, ...]
    capability_key: PlanKey
    recipe_key: PlanKey
    policy_version: int = STABLEHLO_RECIPE_POLICY_VERSION


FreshMapStableHloRecipe: TypeAlias = (
    CompactStableHloRecipe
    | CompactTransposeStableHloRecipe
    | AffineSingleRecordStableHloRecipe
    | AffineTransposeStableHloRecipe
    | PortableAffineStableHloRecipe
)


def _reject(code: str, message: str) -> Never:
    raise StableHloProofError(code, message)


def _u64(value: object, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        _reject("invalid_integer", f"{field_name} must be an integer")
    if value < 0 or value > UINT64_MAX:
        _reject("integer_range", f"{field_name} must fit uint64")
    return value


def _checked_product(values: tuple[int, ...], field_name: str) -> int:
    result = 1
    for index, value in enumerate(values):
        item = _u64(value, f"{field_name}[{index}]")
        if result != 0 and item > UINT64_MAX // result:
            _reject("integer_overflow", f"{field_name} product overflows")
        result *= item
    return result


def _row_major_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    strides = [0] * len(shape)
    expected = 1
    for axis in reversed(range(len(shape))):
        strides[axis] = expected
        extent = shape[axis]
        if expected != 0 and extent > UINT64_MAX // expected:
            _reject("integer_overflow", "dense destination strides overflow")
        expected *= extent
    return tuple(strides)


def compile_compact_stablehlo_recipe(
    compiled: CompiledStridePlan,
) -> CompactStableHloRecipe:
    """Verify one bounded same-dtype compact slice/scale recipe."""

    bound = compiled.bound
    if bound.coverage is not CompleteMode.COMPLETE_UNIQUE:
        _reject("coverage", "compact recipe requires CompleteUnique")
    if len(bound.records) != 1 or len(compiled.records) != 1:
        _reject("record_count", "compact recipe requires exactly one record")
    if (
        bound.source_dtype not in _SUPPORTED_DTYPES
        or bound.result_dtype not in _SUPPORTED_DTYPES
        or bound.source_dtype != bound.result_dtype
    ):
        _reject("dtype", "compact recipe requires one supported same dtype")

    record = compiled.records[0]
    if record.layout_kind is not LayoutKind.COMPACT:
        _reject(
            "layout",
            "compact recipe requires the same dense source/destination mapping",
        )
    if record.destination_offset != 0:
        _reject("destination_offset", "destination offset must be zero")
    if record.logical_elements != bound.output_size:
        _reject("destination_size", "record must cover the complete output")

    source_limit = record.source_offset + record.logical_elements
    if source_limit > bound.source_size:
        _reject("source_bounds", "compact source slice exceeds source storage")
    capability_key = _stable_key_bytes(
        "tensor0-stride-compact-capability-v1",
        STABLEHLO_RECIPE_POLICY_VERSION,
        compiled.capability_key,
    )
    recipe_key = _stable_key_bytes(
        "tensor0-stride-compact-recipe-v1",
        STABLEHLO_RECIPE_POLICY_VERSION,
        compiled.canonical_execution_key,
        capability_key,
        record.source_offset,
        record.logical_elements,
    )
    return CompactStableHloRecipe(
        compiled=compiled,
        source_offset=record.source_offset,
        element_count=record.logical_elements,
        capability_key=capability_key,
        recipe_key=recipe_key,
    )


def compile_compact_affine_interval(
    compiled: CompiledStridePlan,
) -> CompactAffineInterval:
    """Verify one non-empty compact affine interval without fixing write policy."""

    bound = compiled.bound
    if len(bound.records) != 1 or len(compiled.records) != 1:
        _reject("record_count", "compact interval requires exactly one record")
    record = compiled.records[0]
    if record.layout_kind is not LayoutKind.COMPACT:
        _reject("layout", "compact interval requires one dense same mapping")
    if record.logical_elements == 0:
        _reject("empty", "compact interval excludes empty maps")
    source_limit = record.source_offset + record.logical_elements
    destination_limit = record.destination_offset + record.logical_elements
    if source_limit > bound.source_size:
        _reject("source_bounds", "compact interval exceeds source storage")
    if destination_limit > bound.output_size:
        _reject("destination_bounds", "compact interval exceeds destination storage")
    capability_key = _stable_key_bytes(
        "tensor0-stride-compact-affine-interval-v1",
        STABLEHLO_RECIPE_POLICY_VERSION,
        compiled.capability_key,
    )
    return CompactAffineInterval(
        compiled=compiled,
        source_offset=record.source_offset,
        destination_offset=record.destination_offset,
        element_count=record.logical_elements,
        capability_key=capability_key,
    )


def compile_affine_interval_proof(
    logical_shape: tuple[int, ...],
    strides: tuple[int, ...],
    *,
    offset: int,
    storage_size: int,
    field_name: str,
) -> AffineIntervalProof:
    """Prove a positive affine map as a sliced row-major interval."""

    if len(logical_shape) != len(strides):
        _reject("layout_rank", f"{field_name} shape/stride rank differs")
    if any(extent <= 0 for extent in logical_shape):
        _reject("empty", f"{field_name} interval excludes empty axes")
    if any(stride <= 0 for stride in strides):
        _reject("layout_stride", f"{field_name} strides must be positive")
    if not logical_shape:
        if offset >= storage_size:
            _reject("interval_bounds", f"{field_name} scalar is out of bounds")
        return AffineIntervalProof((), (), (1,), (), 1)

    axes = tuple(
        sorted(range(len(logical_shape)), key=strides.__getitem__, reverse=True)
    )
    physical_shape: list[int] = []
    mapped_physical_axes: list[int] = []
    for position, axis in enumerate(axes):
        mapped_physical_axes.append(len(physical_shape))
        physical_shape.append(logical_shape[axis])
        if position + 1 < len(axes):
            next_axis = axes[position + 1]
            denominator = logical_shape[next_axis] * strides[next_axis]
            if denominator == 0 or strides[axis] % denominator != 0:
                _reject(
                    "layout_factorization",
                    f"{field_name} strides are not a rectangular affine slice",
                )
            gap = strides[axis] // denominator
            if gap == 0:
                _reject(
                    "layout_factorization",
                    f"{field_name} affine axes overlap",
                )
            if gap > 1:
                physical_shape.append(gap)
        elif strides[axis] > 1:
            physical_shape.append(strides[axis])

    physical = tuple(physical_shape)
    physical_strides = _row_major_strides(physical)
    for axis, physical_axis in zip(axes, mapped_physical_axes, strict=True):
        if physical_strides[physical_axis] != strides[axis]:
            _reject(
                "layout_factorization",
                f"{field_name} physical strides do not match the affine map",
            )
    interval_elements = _checked_product(physical, f"{field_name}.physical_shape")
    if offset > storage_size or interval_elements > storage_size - offset:
        _reject(
            "interval_bounds",
            f"{field_name} physical interval exceeds storage",
        )
    return AffineIntervalProof(
        logical_shape=logical_shape,
        logical_axes_slowest_first=axes,
        physical_shape=physical,
        mapped_physical_axes=tuple(mapped_physical_axes),
        interval_elements=interval_elements,
    )


def compile_affine_single_record_stablehlo_recipe(
    compiled: CompiledStridePlan,
) -> AffineSingleRecordStableHloRecipe:
    """Verify one positive-affine source and complete dense destination map."""

    bound = compiled.bound
    if bound.coverage is not CompleteMode.COMPLETE_UNIQUE:
        _reject("coverage", "affine recipe requires CompleteUnique")
    if len(bound.records) != 1 or len(compiled.records) != 1:
        _reject("record_count", "affine recipe requires exactly one record")
    if (
        bound.source_dtype not in _SUPPORTED_DTYPES
        or bound.result_dtype not in _SUPPORTED_DTYPES
        or bound.source_dtype != bound.result_dtype
    ):
        _reject("dtype", "affine recipe requires one supported same dtype")
    record = compiled.records[0]
    if record.logical_elements == 0:
        _reject("empty", "affine recipe excludes empty maps")
    source = compile_affine_interval_proof(
        record.logical_shape,
        record.source_strides,
        offset=record.source_offset,
        storage_size=bound.source_size,
        field_name="source",
    )
    destination = compile_affine_interval_proof(
        record.logical_shape,
        record.destination_strides,
        offset=record.destination_offset,
        storage_size=bound.output_size,
        field_name="destination",
    )
    if (
        record.destination_offset != 0
        or destination.interval_elements != bound.output_size
        or destination.interval_elements != record.logical_elements
    ):
        _reject(
            "destination_layout",
            "affine recipe requires a complete dense destination permutation",
        )
    capability_key = _stable_key_bytes(
        "tensor0-stride-affine-stablehlo-capability-v1",
        STABLEHLO_RECIPE_POLICY_VERSION,
        compiled.capability_key,
        source.key,
        destination.key,
    )
    recipe_key = _stable_key_bytes(
        "tensor0-stride-affine-stablehlo-recipe-v1",
        STABLEHLO_RECIPE_POLICY_VERSION,
        compiled.canonical_execution_key,
        capability_key,
    )
    return AffineSingleRecordStableHloRecipe(
        compiled=compiled,
        source=source,
        destination=destination,
        capability_key=capability_key,
        recipe_key=recipe_key,
    )


def compile_affine_transpose_stablehlo_recipe(
    compiled: CompiledStridePlan,
) -> AffineTransposeStableHloRecipe:
    """Verify one complete dense input mapped into a zero-filled affine output."""

    bound = compiled.bound
    if bound.coverage is not CompleteMode.PARTIAL_UNIQUE_ZERO_FILL:
        _reject("coverage", "affine transpose requires partial zero-fill")
    if len(bound.records) != 1 or len(compiled.records) != 1:
        _reject("record_count", "affine transpose requires exactly one record")
    if (
        bound.source_dtype not in _SUPPORTED_DTYPES
        or bound.result_dtype not in _SUPPORTED_DTYPES
        or bound.source_dtype != bound.result_dtype
    ):
        _reject("dtype", "affine transpose requires one supported same dtype")
    record = compiled.records[0]
    if record.logical_elements == 0:
        _reject("empty", "affine transpose excludes empty maps")
    source = compile_affine_interval_proof(
        record.logical_shape,
        record.source_strides,
        offset=record.source_offset,
        storage_size=bound.source_size,
        field_name="source",
    )
    destination = compile_affine_interval_proof(
        record.logical_shape,
        record.destination_strides,
        offset=record.destination_offset,
        storage_size=bound.output_size,
        field_name="destination",
    )
    if (
        record.source_offset != 0
        or source.interval_elements != bound.source_size
        or source.interval_elements != record.logical_elements
    ):
        _reject(
            "source_layout",
            "affine transpose requires a complete dense source permutation",
        )
    capability_key = _stable_key_bytes(
        "tensor0-stride-affine-transpose-capability-v1",
        STABLEHLO_RECIPE_POLICY_VERSION,
        compiled.capability_key,
        source.key,
        destination.key,
    )
    recipe_key = _stable_key_bytes(
        "tensor0-stride-affine-transpose-recipe-v1",
        STABLEHLO_RECIPE_POLICY_VERSION,
        compiled.canonical_execution_key,
        capability_key,
    )
    return AffineTransposeStableHloRecipe(
        compiled=compiled,
        source=source,
        destination=destination,
        capability_key=capability_key,
        recipe_key=recipe_key,
    )


def compile_compact_transpose_stablehlo_recipe(
    compiled: CompiledStridePlan,
) -> CompactTransposeStableHloRecipe:
    """Verify one compact scale-and-zero-pad transpose recipe."""

    bound = compiled.bound
    if bound.coverage is not CompleteMode.PARTIAL_UNIQUE_ZERO_FILL:
        _reject("coverage", "compact transpose requires partial zero-fill")
    if len(bound.records) != 1 or len(compiled.records) != 1:
        _reject("record_count", "compact transpose requires exactly one record")
    if (
        bound.source_dtype not in _SUPPORTED_DTYPES
        or bound.result_dtype not in _SUPPORTED_DTYPES
        or bound.source_dtype != bound.result_dtype
    ):
        _reject("dtype", "compact transpose requires one supported same dtype")

    record = compiled.records[0]
    if record.layout_kind is not LayoutKind.COMPACT:
        _reject("layout", "compact transpose requires one dense mapping")
    if record.source_offset != 0:
        _reject("source_offset", "compact transpose source offset must be zero")
    if record.logical_elements != bound.source_size:
        _reject("source_size", "compact transpose must consume the complete input")
    destination_limit = record.destination_offset + record.logical_elements
    if destination_limit > bound.output_size:
        _reject("destination_bounds", "compact transpose exceeds output storage")

    capability_key = _stable_key_bytes(
        "tensor0-stride-compact-transpose-capability-v1",
        STABLEHLO_RECIPE_POLICY_VERSION,
        compiled.capability_key,
    )
    recipe_key = _stable_key_bytes(
        "tensor0-stride-compact-transpose-recipe-v1",
        STABLEHLO_RECIPE_POLICY_VERSION,
        compiled.canonical_execution_key,
        capability_key,
        record.destination_offset,
        record.logical_elements,
        bound.output_size,
    )
    return CompactTransposeStableHloRecipe(
        compiled=compiled,
        destination_offset=record.destination_offset,
        element_count=record.logical_elements,
        capability_key=capability_key,
        recipe_key=recipe_key,
    )


def compile_portable_affine_stablehlo_recipe(
    compiled: CompiledStridePlan,
) -> PortableAffineStableHloRecipe:
    """Verify a portable positive-affine map without materialized addresses."""

    bound = compiled.bound
    if bound.coverage not in {
        CompleteMode.COMPLETE_UNIQUE,
        CompleteMode.PARTIAL_UNIQUE_ZERO_FILL,
    }:
        _reject("coverage", "portable affine recipe requires unique destinations")
    if (
        bound.source_dtype not in _PORTABLE_DTYPES
        or bound.result_dtype not in _PORTABLE_DTYPES
    ):
        _reject("dtype", "portable affine recipe requires supported JAX dtypes")
    if bound.has_source_broadcast:
        _reject("source_broadcast", "portable affine recipe excludes broadcast reads")

    records: list[PortableAffineRecord] = []
    for record_index, record in enumerate(compiled.records):
        if record.logical_elements == 0:
            continue
        source = compile_affine_interval_proof(
            record.logical_shape,
            record.source_strides,
            offset=record.source_offset,
            storage_size=bound.source_size,
            field_name=f"source[{record_index}]",
        )
        destination = compile_affine_interval_proof(
            record.logical_shape,
            record.destination_strides,
            offset=record.destination_offset,
            storage_size=bound.output_size,
            field_name=f"destination[{record_index}]",
        )
        records.append(
            PortableAffineRecord(
                record_index=record_index,
                source=source,
                destination=destination,
            )
        )

    frozen_records = tuple(records)
    capability_key = _stable_key_bytes(
        "tensor0-stride-portable-affine-capability-v1",
        STABLEHLO_RECIPE_POLICY_VERSION,
        compiled.capability_key,
        tuple(record.key for record in frozen_records),
    )
    recipe_key = _stable_key_bytes(
        "tensor0-stride-portable-affine-recipe-v1",
        STABLEHLO_RECIPE_POLICY_VERSION,
        compiled.canonical_execution_key,
        capability_key,
    )
    return PortableAffineStableHloRecipe(
        compiled=compiled,
        records=frozen_records,
        capability_key=capability_key,
        recipe_key=recipe_key,
    )


def _extract_affine_interval(
    storage: object,
    *,
    offset: int,
    proof: AffineIntervalProof,
) -> Array:
    data = jnp.asarray(storage)
    batch_shape = data.shape[:-1]
    batch_rank = len(batch_shape)
    interval = jax.lax.slice_in_dim(
        data,
        offset,
        offset + proof.interval_elements,
        axis=data.ndim - 1,
    )
    physical = jnp.reshape(interval, (*batch_shape, *proof.physical_shape))
    mapped = frozenset(proof.mapped_physical_axes)
    limits = tuple(
        extent if axis in mapped else 1
        for axis, extent in enumerate(proof.physical_shape)
    )
    sliced = jax.lax.slice(
        physical,
        start_indices=(*((0,) * batch_rank), *((0,) * len(proof.physical_shape))),
        limit_indices=(*batch_shape, *limits),
    )
    axes = proof.logical_axes_slowest_first
    ordered_shape = tuple(proof.logical_shape[axis] for axis in axes)
    ordered = jnp.reshape(sliced, (*batch_shape, *ordered_shape))
    if not axes:
        return ordered
    permutation = (
        *range(batch_rank),
        *(batch_rank + axes.index(axis) for axis in range(len(axes))),
    )
    return jnp.transpose(ordered, permutation)


def _arrange_affine_interval_values(
    values: object,
    proof: AffineIntervalProof,
) -> Array:
    data = jnp.asarray(values)
    rank = len(proof.logical_shape)
    batch_shape = data.shape[:-rank] if rank else data.shape
    batch_rank = len(batch_shape)
    axes = proof.logical_axes_slowest_first
    if axes:
        permutation = (
            *range(batch_rank),
            *(batch_rank + axis for axis in axes),
        )
        ordered = jnp.transpose(data, permutation)
    else:
        ordered = data
    mapped_extent = {
        physical_axis: proof.logical_shape[logical_axis]
        for logical_axis, physical_axis in zip(
            axes,
            proof.mapped_physical_axes,
            strict=True,
        )
    }
    update_shape = tuple(
        mapped_extent.get(axis, 1) for axis in range(len(proof.physical_shape))
    )
    return jnp.reshape(ordered, (*batch_shape, *update_shape))


def _write_affine_interval(
    storage: object,
    values: object,
    *,
    offset: int,
    proof: AffineIntervalProof,
) -> Array:
    data = jnp.asarray(storage)
    batch_shape = data.shape[:-1]
    interval = jax.lax.slice_in_dim(
        data,
        offset,
        offset + proof.interval_elements,
        axis=data.ndim - 1,
    )
    physical = jnp.reshape(interval, (*batch_shape, *proof.physical_shape))
    update = _arrange_affine_interval_values(values, proof)
    updated = jax.lax.dynamic_update_slice(
        physical,
        update,
        (*((0,) * len(batch_shape)), *((0,) * len(proof.physical_shape))),
    )
    flat = jnp.reshape(updated, (*batch_shape, proof.interval_elements))
    return jax.lax.dynamic_update_slice_in_dim(
        data,
        flat,
        offset,
        axis=data.ndim - 1,
    )


def execute_affine_single_record_stablehlo_recipe(
    source: object,
    recipe: AffineSingleRecordStableHloRecipe,
) -> Array:
    """Execute a verified positive-affine read without address arrays."""

    bound = recipe.compiled.bound
    data = jnp.asarray(source)
    if data.ndim < 1 or data.shape[-1] != bound.source_size:
        raise ValueError(
            f"expected source shape (*batch, {bound.source_size}), got {data.shape}"
        )
    expected_dtype = jnp.dtype(bound.source_dtype)
    if data.dtype != expected_dtype:
        raise TypeError(
            f"expected source dtype {expected_dtype.name}, got {data.dtype.name}"
        )
    record = recipe.compiled.records[0]
    logical = _extract_affine_interval(
        data,
        offset=record.source_offset,
        proof=recipe.source,
    )
    converted = jnp.asarray(logical, dtype=jnp.dtype(bound.result_dtype))
    scale = jnp.asarray(bound.records[0].scale, dtype=converted.dtype)
    scaled = scale * converted
    arranged = _arrange_affine_interval_values(scaled, recipe.destination)
    return jnp.reshape(arranged, (*data.shape[:-1], bound.output_size))


def execute_affine_transpose_stablehlo_recipe(
    source: object,
    recipe: AffineTransposeStableHloRecipe,
) -> Array:
    """Execute a verified affine zero-fill map without address arrays."""

    bound = recipe.compiled.bound
    data = jnp.asarray(source)
    if data.ndim < 1 or data.shape[-1] != bound.source_size:
        raise ValueError(
            f"expected source shape (*batch, {bound.source_size}), got {data.shape}"
        )
    expected_dtype = jnp.dtype(bound.source_dtype)
    if data.dtype != expected_dtype:
        raise TypeError(
            f"expected source dtype {expected_dtype.name}, got {data.dtype.name}"
        )
    record = recipe.compiled.records[0]
    logical = _extract_affine_interval(
        data,
        offset=record.source_offset,
        proof=recipe.source,
    )
    converted = jnp.asarray(logical, dtype=jnp.dtype(bound.result_dtype))
    scale = jnp.asarray(bound.records[0].scale, dtype=converted.dtype)
    scaled = scale * converted
    result = jnp.zeros(
        (*data.shape[:-1], bound.output_size),
        dtype=converted.dtype,
    )
    return _write_affine_interval(
        result,
        scaled,
        offset=record.destination_offset,
        proof=recipe.destination,
    )


def execute_compact_stablehlo_recipe(
    source: object,
    recipe: CompactStableHloRecipe,
) -> Array:
    """Execute a verified compact recipe with bounded ordinary JAX operations."""

    bound = recipe.compiled.bound
    data = jnp.asarray(source)
    if data.ndim < 1 or data.shape[-1] != bound.source_size:
        raise ValueError(
            f"expected source shape (*batch, {bound.source_size}), got {data.shape}"
        )
    expected_dtype = jnp.dtype(bound.source_dtype)
    if data.dtype != expected_dtype:
        raise TypeError(
            f"expected source dtype {expected_dtype.name}, got {data.dtype.name}"
        )
    values = jax.lax.slice_in_dim(
        data,
        recipe.source_offset,
        recipe.source_offset + recipe.element_count,
        axis=data.ndim - 1,
    )
    scale = jnp.asarray(bound.records[0].scale, dtype=expected_dtype)
    return jnp.reshape(
        scale * values,
        (*data.shape[:-1], bound.output_size),
    )


def execute_compact_transpose_stablehlo_recipe(
    source: object,
    recipe: CompactTransposeStableHloRecipe,
) -> Array:
    """Execute a verified compact transpose with scale and zero padding."""

    bound = recipe.compiled.bound
    data = jnp.asarray(source)
    if data.ndim < 1 or data.shape[-1] != bound.source_size:
        raise ValueError(
            f"expected source shape (*batch, {bound.source_size}), got {data.shape}"
        )
    expected_dtype = jnp.dtype(bound.source_dtype)
    if data.dtype != expected_dtype:
        raise TypeError(
            f"expected source dtype {expected_dtype.name}, got {data.dtype.name}"
        )
    scale = jnp.asarray(bound.records[0].scale, dtype=expected_dtype)
    scaled = scale * data
    left = recipe.destination_offset
    right = bound.output_size - left - recipe.element_count
    padding = (*((0, 0) for _ in data.shape[:-1]), (left, right))
    return jnp.pad(scaled, padding)


def execute_portable_affine_stablehlo_recipe(
    source: object,
    recipe: PortableAffineStableHloRecipe,
) -> Array:
    """Execute a verified record-wise affine map on any JAX platform."""

    bound = recipe.compiled.bound
    data = jnp.asarray(source)
    if data.ndim < 1 or data.shape[-1] != bound.source_size:
        raise ValueError(
            f"expected source shape (*batch, {bound.source_size}), got {data.shape}"
        )
    expected_dtype = jnp.dtype(bound.source_dtype)
    if data.dtype != expected_dtype:
        raise TypeError(
            f"expected source dtype {expected_dtype.name}, got {data.dtype.name}"
        )

    result_dtype = jnp.dtype(bound.result_dtype)
    result = jnp.zeros(
        (*data.shape[:-1], bound.output_size),
        dtype=result_dtype,
    )
    for portable in recipe.records:
        record = recipe.compiled.records[portable.record_index]
        logical = _extract_affine_interval(
            data,
            offset=record.source_offset,
            proof=portable.source,
        )
        converted = jnp.asarray(logical, dtype=result_dtype)
        scale = jnp.asarray(
            bound.records[portable.record_index].scale,
            dtype=result_dtype,
        )
        result = _write_affine_interval(
            result,
            scale * converted,
            offset=record.destination_offset,
            proof=portable.destination,
        )
    return result


def try_compile_fresh_map_stablehlo(
    compiled: CompiledStridePlan,
) -> FreshMapStableHloRecipe | None:
    """Return the preferred verified fresh-map recipe, if one exists."""

    for compiler in (
        compile_compact_stablehlo_recipe,
        compile_affine_single_record_stablehlo_recipe,
        compile_compact_transpose_stablehlo_recipe,
        compile_affine_transpose_stablehlo_recipe,
        compile_portable_affine_stablehlo_recipe,
    ):
        try:
            return compiler(compiled)
        except StableHloProofError:
            pass
    return None


def execute_fresh_map_stablehlo(
    source: object,
    recipe: FreshMapStableHloRecipe,
) -> Array:
    """Execute any verified fresh-map recipe."""

    if isinstance(recipe, CompactStableHloRecipe):
        return execute_compact_stablehlo_recipe(source, recipe)
    if isinstance(recipe, CompactTransposeStableHloRecipe):
        return execute_compact_transpose_stablehlo_recipe(source, recipe)
    if isinstance(recipe, AffineSingleRecordStableHloRecipe):
        return execute_affine_single_record_stablehlo_recipe(source, recipe)
    if isinstance(recipe, AffineTransposeStableHloRecipe):
        return execute_affine_transpose_stablehlo_recipe(source, recipe)
    if isinstance(recipe, PortableAffineStableHloRecipe):
        return execute_portable_affine_stablehlo_recipe(source, recipe)
    assert_never(recipe)


__all__ = [
    "CompactStableHloRecipe",
    "CompactTransposeStableHloRecipe",
    "FreshMapStableHloRecipe",
    "PortableAffineStableHloRecipe",
    "STABLEHLO_RECIPE_POLICY_VERSION",
    "StableHloProofError",
    "compile_compact_stablehlo_recipe",
    "compile_compact_transpose_stablehlo_recipe",
    "execute_compact_stablehlo_recipe",
    "execute_compact_transpose_stablehlo_recipe",
    "execute_portable_affine_stablehlo_recipe",
    "execute_fresh_map_stablehlo",
    "try_compile_fresh_map_stablehlo",
]
