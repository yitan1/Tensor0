from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from itertools import product
import os
from pathlib import Path
import random
import subprocess
import sys
from typing import cast

import jax.numpy as jnp
from jax.typing import DTypeLike
import numpy as np
import pytest

from tensor0._stride import (
    StridedCopyPlan,
    CompleteMode,
    StridedCopyRecord,
)
from tensor0._stride._compiler import (
    CompiledRecord,
    CpuKernelKind,
    LayoutKind,
    ScaleKind,
    compile_plan,
)
from tensor0._stride._native_lowering import lower_compiled_plan, lower_plan
from tensor0._stride._plan import (
    build_strided_copy_plan,
    transpose_same_dtype_plan,
)

from ._fixtures import (
    empty_plan,
    heterogeneous_grid_fusion_plan,
    partial_mixed_plan,
    rank2_transpose_plan,
    rank4_avx2_shape_plan,
    rank4_two_pair_plan,
    rank_zero_plan,
    two_record_noncompact_plan,
    u1_three_record_plan,
    u1_two_record_plan,
)


_REPO_ROOT = Path(__file__).resolve().parents[2]


def _maximum_address(
    shape: tuple[int, ...],
    strides: tuple[int, ...],
    offset: int,
) -> int | None:
    if any(extent == 0 for extent in shape):
        return None
    return offset + sum(
        (extent - 1) * stride for extent, stride in zip(shape, strides, strict=True)
    )


def _single_bound(
    shape: tuple[int, ...],
    source_strides: tuple[int, ...],
    destination_strides: tuple[int, ...],
    *,
    source_order: tuple[int, ...],
    destination_order: tuple[int, ...],
    scale: int | float | complex = 1.25,
    dtype: DTypeLike = jnp.float32,
    coverage: CompleteMode = CompleteMode.COMPLETE_UNIQUE,
    source_offset: int = 0,
    destination_offset: int = 0,
    output_size: int | None = None,
) -> StridedCopyPlan:
    record = StridedCopyRecord(
        shape,
        source_strides,
        source_offset,
        destination_strides,
        destination_offset,
        scale,
    )
    source_maximum = _maximum_address(shape, source_strides, source_offset)
    destination_maximum = _maximum_address(
        shape,
        destination_strides,
        destination_offset,
    )
    source_size = source_offset if source_maximum is None else source_maximum + 1
    if output_size is None:
        output_size = (
            destination_offset
            if destination_maximum is None
            else destination_maximum + 1
    )
    return build_strided_copy_plan(
        records=(record,),
        output_size=output_size,
        coverage=coverage,
        source_size=source_size,
        source_dtype=dtype,
        result_dtype=dtype,
    )


def _raw_pairs(
    record: StridedCopyRecord,
    *,
    dtype: str,
) -> Counter[tuple[int, int, bytes]]:
    if any(extent == 0 for extent in record.logical_shape):
        return Counter()
    result_dtype = np.dtype(dtype).newbyteorder("<")
    scale_bytes = np.asarray(record.scale, dtype=result_dtype).tobytes()
    return Counter(
        (
            record.source_offset
            + sum(
                coordinate * stride
                for coordinate, stride in zip(
                    coordinates,
                    record.source_strides,
                    strict=True,
                )
            ),
            record.destination_offset
            + sum(
                coordinate * stride
                for coordinate, stride in zip(
                    coordinates,
                    record.destination_strides,
                    strict=True,
                )
            ),
            scale_bytes,
        )
        for coordinates in product(*(range(extent) for extent in record.logical_shape))
    )


def _compiled_pairs(
    record: CompiledRecord,
) -> Counter[tuple[int, int, bytes]]:
    shape = record.logical_shape
    if any(extent == 0 for extent in shape):
        return Counter()
    return Counter(
        (
            record.source_offset
            + sum(
                coordinate * stride
                for coordinate, stride in zip(
                    coordinates,
                    record.source_strides,
                    strict=True,
                )
            ),
            record.destination_offset
            + sum(
                coordinate * stride
                for coordinate, stride in zip(
                    coordinates,
                    record.destination_strides,
                    strict=True,
                )
            ),
            record.scale_bytes,
        )
        for coordinates in product(*(range(extent) for extent in shape))
    )


def _assert_pair_equivalent(bound: StridedCopyPlan) -> None:
    compiled = compile_plan(bound)
    assert len(compiled.records) == len(bound.records)
    for raw, normalized in zip(
        bound.records,
        compiled.records,
        strict=True,
    ):
        assert _raw_pairs(raw, dtype=bound.result_dtype) == _compiled_pairs(normalized)


def _dense_strides(
    shape: tuple[int, ...],
    axes_fastest_first: tuple[int, ...],
    rng: random.Random,
) -> tuple[int, ...]:
    strides = [0] * len(shape)
    expected = 1
    for axis in axes_fastest_first:
        extent = shape[axis]
        if extent <= 1:
            strides[axis] = rng.randrange(1, 100)
            continue
        strides[axis] = expected
        expected *= extent
    return tuple(strides)


@pytest.mark.parametrize(
    ("shape", "strides", "order", "expected_provenance"),
    [
        ((2, 3), (1, 2), (0, 1), ((0, 1),)),
        ((2, 3), (3, 1), (1, 0), ((1, 0),)),
        ((2, 1, 3), (3, 99, 1), (2, 0, 1), ((2, 0),)),
        ((2, 3, 4), (12, 4, 1), (2, 1, 0), ((2, 1, 0),)),
    ],
)
def test_normalization_removes_singletons_and_fuses_both_directions(
    shape: tuple[int, ...],
    strides: tuple[int, ...],
    order: tuple[int, ...],
    expected_provenance: tuple[tuple[int, ...], ...],
) -> None:
    bound = _single_bound(
        shape,
        strides,
        strides,
        source_order=order,
        destination_order=order,
    )

    compiled = compile_plan(bound)

    assert compiled.records[0].logical_shape == (int(np.prod(shape)),)
    assert compiled.records[0].source_strides == (1,)
    assert compiled.records[0].destination_strides == (1,)
    assert compiled.records[0].axis_provenance_fastest_first == expected_provenance
    assert compiled.records[0].layout_kind is LayoutKind.COMPACT
    _assert_pair_equivalent(bound)


def test_fusion_requires_source_and_destination_to_agree() -> None:
    bound = _single_bound(
        (2, 3),
        (1, 2),
        (3, 1),
        source_order=(0, 1),
        destination_order=(1, 0),
    )

    compiled = compile_plan(bound)

    assert compiled.records[0].logical_shape == (2, 3)
    assert compiled.records[0].axis_provenance_fastest_first == ((0,), (1,))
    assert compiled.records[0].layout_kind is LayoutKind.PERMUTATION_LIKE
    _assert_pair_equivalent(bound)


@pytest.mark.parametrize(
    "factory",
    [
        empty_plan,
        rank_zero_plan,
        partial_mixed_plan,
        rank2_transpose_plan,
        rank4_avx2_shape_plan,
        two_record_noncompact_plan,
        heterogeneous_grid_fusion_plan,
        u1_two_record_plan,
        u1_three_record_plan,
    ],
)
def test_normalized_records_preserve_paired_address_maps(
    factory: Callable[[], StridedCopyPlan],
) -> None:
    _assert_pair_equivalent(factory())


def test_random_valid_affine_plans_preserve_paired_address_maps() -> None:
    rng = random.Random(0x5A17C0DE)
    dtype_scales = (
        (jnp.float32, -1.25),
        (jnp.complex64, 0.75 - 0.5j),
        (jnp.int32, -3),
    )
    for _ in range(200):
        rank = rng.randrange(5)
        shape = tuple(rng.randrange(1, 5) for _ in range(rank))
        if rank and rng.randrange(8) == 0:
            zero_axis = rng.randrange(rank)
            shape = shape[:zero_axis] + (0,) + shape[zero_axis + 1 :]
        source_order_list = list(range(rank))
        destination_order_list = list(range(rank))
        rng.shuffle(source_order_list)
        rng.shuffle(destination_order_list)
        source_order = tuple(source_order_list)
        destination_order = tuple(destination_order_list)
        source_strides = _dense_strides(shape, source_order, rng)
        destination_strides = _dense_strides(
            shape,
            destination_order,
            rng,
        )
        dtype, scale = rng.choice(dtype_scales)
        bound = _single_bound(
            shape,
            source_strides,
            destination_strides,
            source_order=source_order,
            destination_order=destination_order,
            source_offset=0 if 0 in shape else rng.randrange(6),
            scale=scale,
            dtype=dtype,
        )

        _assert_pair_equivalent(bound)
        first = compile_plan(bound)
        second = compile_plan(bound)
        assert first == second
        assert first.canonical_execution_key == second.canonical_execution_key
        assert first.capability_key == second.capability_key


def test_heterogeneous_records_preserve_provenance_across_fusion() -> None:
    bound = heterogeneous_grid_fusion_plan()

    compiled = compile_plan(bound)

    assert tuple(record.logical_shape for record in compiled.records) == (
        (6,),
        (2, 2),
        (2, 2),
    )
    assert compiled.records[0].axis_provenance_fastest_first == ((1, 0),)
    _assert_pair_equivalent(bound)


def test_canonical_key_collapses_singletons_and_affine_factorizations() -> None:
    plans = (
        _single_bound(
            (6,),
            (1,),
            (1,),
            source_order=(0,),
            destination_order=(0,),
        ),
        _single_bound(
            (2, 3),
            (1, 2),
            (1, 2),
            source_order=(0, 1),
            destination_order=(0, 1),
        ),
        _single_bound(
            (2, 3),
            (3, 1),
            (3, 1),
            source_order=(1, 0),
            destination_order=(1, 0),
        ),
        _single_bound(
            (2, 1, 3),
            (3, 99, 1),
            (3, 17, 1),
            source_order=(2, 0, 1),
            destination_order=(2, 0, 1),
        ),
    )
    compiled = tuple(compile_plan(plan) for plan in plans)
    paired_maps = tuple(
        _raw_pairs(plan.records[0], dtype=plan.result_dtype) for plan in plans
    )

    assert all(paired_map == paired_maps[0] for paired_map in paired_maps)
    assert all(isinstance(plan.canonical_execution_key, bytes) for plan in compiled)
    assert all(isinstance(plan.capability_key, bytes) for plan in compiled)
    assert len({plan.bound.semantic_key for plan in compiled}) == len(compiled)
    assert len({plan.canonical_execution_key for plan in compiled}) == 1
    assert len({plan.capability_key for plan in compiled}) == 1


def test_canonical_keys_separate_execution_near_misses() -> None:
    base = _single_bound(
        (2, 3),
        (3, 1),
        (3, 1),
        source_order=(1, 0),
        destination_order=(1, 0),
    )
    near_misses = (
        _single_bound(
            (2, 3),
            (3, 1),
            (3, 1),
            source_order=(1, 0),
            destination_order=(1, 0),
            source_offset=1,
        ),
        _single_bound(
            (2, 3),
            (3, 1),
            (3, 1),
            source_order=(1, 0),
            destination_order=(1, 0),
            scale=-1.25,
        ),
        _single_bound(
            (2, 3),
            (3, 1),
            (3, 1),
            source_order=(1, 0),
            destination_order=(1, 0),
            dtype=jnp.complex64,
        ),
        _single_bound(
            (2, 3),
            (3, 1),
            (3, 1),
            source_order=(1, 0),
            destination_order=(1, 0),
            coverage=CompleteMode.PARTIAL_UNIQUE_ZERO_FILL,
            output_size=8,
        ),
    )
    compiled = compile_plan(base)
    assert all(
        compile_plan(candidate).canonical_execution_key
        != compiled.canonical_execution_key
        for candidate in near_misses
    )

def _float32_from_bits(bits: int) -> float:
    value = np.asarray(bits, dtype=np.uint32).view(np.float32)[()]
    return cast(float, value)


def _complex64_from_bits(real: int, imaginary: int) -> complex:
    value = np.asarray((real, imaginary), dtype=np.uint32).view(np.complex64)[0]
    return cast(complex, value)


def test_scale_bytes_and_keys_distinguish_signed_zero_and_nan_payloads() -> None:
    bit_patterns = (0x00000000, 0x80000000, 0x7FC00001, 0x7FC00002)
    plans = tuple(
        compile_plan(
            _single_bound(
                (4,),
                (1,),
                (1,),
                source_order=(0,),
                destination_order=(0,),
                scale=_float32_from_bits(bits),
            )
        )
        for bits in bit_patterns
    )

    assert (
        tuple(int.from_bytes(plan.records[0].scale_bytes, "little") for plan in plans)
        == bit_patterns
    )
    assert all(plan.records[0].scale_kind is ScaleKind.GENERAL for plan in plans)
    assert len({plan.bound.semantic_key for plan in plans}) == len(plans)
    assert len({plan.canonical_execution_key for plan in plans}) == len(plans)

    positive_zero = compile_plan(
        _single_bound(
            (4,),
            (1,),
            (1,),
            source_order=(0,),
            destination_order=(0,),
            dtype=jnp.complex64,
            scale=_complex64_from_bits(0x3F800000, 0x00000000),
        )
    )
    negative_zero = compile_plan(
        _single_bound(
            (4,),
            (1,),
            (1,),
            source_order=(0,),
            destination_order=(0,),
            dtype=jnp.complex64,
            scale=_complex64_from_bits(0x3F800000, 0x80000000),
        )
    )
    assert positive_zero.records[0].scale_kind is ScaleKind.UNIT
    assert negative_zero.records[0].scale_kind is ScaleKind.GENERAL
    assert positive_zero.records[0].scale_bytes != negative_zero.records[0].scale_bytes
    assert positive_zero.bound.semantic_key != negative_zero.bound.semantic_key
    assert (
        positive_zero.canonical_execution_key != negative_zero.canonical_execution_key
    )


def _zero_extent_plan() -> StridedCopyPlan:
    return _single_bound(
        (0, 3),
        (0, 1),
        (0, 1),
        source_order=(0, 1),
        destination_order=(0, 1),
        coverage=CompleteMode.PARTIAL_UNIQUE_ZERO_FILL,
        output_size=5,
    )


@pytest.mark.parametrize(
    ("factory", "expected"),
    [
        (_zero_extent_plan, (CpuKernelKind.EMPTY,)),
        (rank_zero_plan, (CpuKernelKind.COMPACT,)),
        (rank2_transpose_plan, (CpuKernelKind.RANK2_FORWARD,)),
        (rank4_avx2_shape_plan, (CpuKernelKind.RANK4_AXIS0,)),
        (rank4_two_pair_plan, (CpuKernelKind.RANK4_TWO_PAIR,)),
        (
            two_record_noncompact_plan,
            (CpuKernelKind.GENERIC, CpuKernelKind.GENERIC),
        ),
    ],
)
def test_cpu_projection_classifies_retained_work_kinds(
    factory: Callable[[], StridedCopyPlan],
    expected: tuple[CpuKernelKind, ...],
) -> None:
    compiled = compile_plan(factory())
    execution = lower_compiled_plan(compiled)

    assert tuple(record.kernel_kind for record in execution.records) == expected


def test_cpu_projection_classifies_reverse_rank2_independently() -> None:
    compiled = compile_plan(transpose_same_dtype_plan(rank2_transpose_plan()))
    execution = lower_compiled_plan(compiled)

    assert execution.records[0].kernel_kind is CpuKernelKind.RANK2_REVERSE


def test_cpu_projection_keeps_rank4_two_pair_tail_generic() -> None:
    compiled = compile_plan(rank4_two_pair_plan((8, 8, 7, 9)))
    execution = lower_compiled_plan(compiled)

    assert execution.records[0].kernel_kind is CpuKernelKind.GENERIC


def test_native_execution_accepts_mixed_dtype_and_rejects_excessive_rank() -> None:
    template = _single_bound(
        (4,),
        (1,),
        (1,),
        source_order=(0,),
        destination_order=(0,),
    )
    mixed_dtype = build_strided_copy_plan(
        records=template.records,
        output_size=template.output_size,
        coverage=template.coverage,
        source_size=template.source_size,
        source_dtype=jnp.float32,
        result_dtype=jnp.complex64,
    )
    generic_native_dtype = build_strided_copy_plan(
        records=template.records,
        output_size=template.output_size,
        coverage=template.coverage,
        source_size=template.source_size,
        source_dtype=jnp.uint32,
        result_dtype=jnp.uint32,
    )
    shape = (2,) * 9
    order = tuple(reversed(range(len(shape))))
    strides = tuple(2 ** (len(shape) - 1 - axis) for axis in range(len(shape)))
    excessive_raw_rank = _single_bound(
        shape,
        strides,
        strides,
        source_order=order,
        destination_order=order,
    )

    assert lower_plan(generic_native_dtype).records
    mixed_execution = lower_plan(mixed_dtype)
    assert mixed_execution.bound is mixed_dtype
    assert mixed_execution.records
    with pytest.raises(ValueError, match="rank exceeds native limit"):
        lower_plan(excessive_raw_rank)


def test_compiled_descriptor_size_is_independent_of_logical_element_count() -> None:
    small = lower_plan(
        _single_bound(
            (2, 3),
            (3, 1),
            (3, 1),
            source_order=(1, 0),
            destination_order=(1, 0),
        )
    )
    large = lower_plan(
        _single_bound(
            (200, 300),
            (300, 1),
            (300, 1),
            source_order=(1, 0),
            destination_order=(1, 0),
        )
    )

    assert len(small.words) == len(large.words)
    assert len(small.descriptor) == len(large.descriptor)
    assert len(small.descriptor) == 8 * len(small.words)


def test_lowered_plan_owns_compiled_execution_metadata() -> None:
    bound = rank2_transpose_plan()

    lowered = lower_plan(bound)

    assert lowered.compiled.bound is bound
    assert lowered.records
    assert lowered.words
    assert lowered.descriptor


def _interval_bound(record_count: int) -> StridedCopyPlan:
    record_size = 2
    records = tuple(
        StridedCopyRecord(
            (record_size,),
            (1,),
            index * record_size,
            (1,),
            index * record_size,
        )
        for index in range(record_count)
    )
    size = record_count * record_size
    return build_strided_copy_plan(
        records=records,
        output_size=size,
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=size,
        source_dtype=jnp.float32,
        result_dtype=jnp.float32,
    )


def test_compiled_descriptor_word_count_is_linear_in_records_and_rank() -> None:
    record_sizes = tuple(
        len(lower_plan(_interval_bound(count)).words)
        for count in range(1, 5)
    )
    record_deltas = tuple(
        right - left for left, right in zip(record_sizes, record_sizes[1:])
    )

    rank_sizes = []
    for rank in range(1, 5):
        shape = (2,) * rank
        order = tuple(reversed(range(rank)))
        strides = tuple(2 ** (rank - 1 - axis) for axis in range(rank))
        bound = _single_bound(
            shape,
            strides,
            strides,
            source_order=order,
            destination_order=order,
        )
        rank_sizes.append(len(lower_plan(bound).words))
    rank_deltas = tuple(right - left for left, right in zip(rank_sizes, rank_sizes[1:]))

    assert len(set(record_deltas)) == 1
    assert record_deltas[0] > 0
    assert len(set(rank_deltas)) == 1
    assert rank_deltas[0] > 0


def test_canonical_key_serialization_is_stable_across_fresh_processes() -> None:
    script = """
from tensor0._stride._compiler import compile_plan
from tests.stride._fixtures import rank2_transpose_plan
print(compile_plan(rank2_transpose_plan()).canonical_execution_key.hex())
"""
    outputs = []
    for seed in ("1", "2"):
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = seed
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=_REPO_ROOT,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        outputs.append(completed.stdout.strip())

    assert outputs[0]
    assert outputs[0] == outputs[1]
