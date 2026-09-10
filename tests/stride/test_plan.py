from __future__ import annotations

from collections.abc import Callable
import hashlib
from itertools import permutations
from math import prod
from pathlib import Path
import sys
import tomllib
from typing import Any

import jax.numpy as jnp
import pytest

from tensor0._stride._plan import (
    CompleteMode,
    AffinePlan,
    AffineRecord,
    StridedOutputInit,
    StridedReductionKind,
    StridedScalarKind,
    StridedWriteKind,
)
from tensor0._stride._map import _execute_map
from tensor0._stride._native_descriptor import (
    AFFINE_DESCRIPTOR_MAGIC,
    AFFINE_DESCRIPTOR_VERSION,
    lower_plan,
)
from tensor0._stride._plan import (
    PlanValidationError,
    _DTYPE_C64,
    _DTYPE_F32,
    build_affine_plan,
    transpose_plan,
)
from tensor0._stride._ops._selected_scale import build_selected_scale_plan
from tensor0._stride._ops._update_support import build_base_accumulate_plan, build_base_assign_plan

from ._fixtures import (
    contiguous_dtype_plan,
    two_record_noncompact_plan,
    u1_three_record_plan,
    u1_two_record_plan,
)
from ._oracle import enumerate_addresses


_REPO_ROOT = Path(__file__).resolve().parents[2]
_VENDORED_FFI_FILE_HASHES = {
    "include/xla/ffi/api/api.h": (
        "7f76572a80ed2097e5924e6d02d84891c725300172280bd009c0a7c9ac7961eb"
    ),
    "include/xla/ffi/api/c_api.h": (
        "85fc385c2d3a6b539a05b9cf4c3535aa24b4b41040f9e111c1f2c11b0e2fa539"
    ),
    "include/xla/ffi/api/ffi.h": (
        "4e4a1d8f9825e88e15a2bcbb7c08eb6233f020b952cab5bbbb8510e3017515c5"
    ),
    "LICENSE.txt": (
        "e3d8688a2c75d4e33641cc88046a8a1593ee79822411fa45a7bd32e37f847d29"
    ),
}


def _compact_strides(
    shape: tuple[int, ...],
    axes_fastest_first: tuple[int, ...],
) -> tuple[int, ...]:
    strides = [0] * len(shape)
    expected = 1
    for axis in axes_fastest_first:
        strides[axis] = expected
        expected *= shape[axis]
    return tuple(strides)


def _single_plan(
    shape: tuple[int, ...],
    source_order: tuple[int, ...],
    destination_order: tuple[int, ...],
) -> AffinePlan:
    size = prod(shape)
    record = AffineRecord(
        shape,
        _compact_strides(shape, source_order),
        0,
        _compact_strides(shape, destination_order),
        0,
        -1.25,
    )
    return build_affine_plan(
        records=(record,),
        output_size=size,
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=size,
        source_dtype=jnp.float32,
        result_dtype=jnp.float32,
    )


def test_noncompact_multi_record_plan_binds_and_lowers_compactly() -> None:
    bound = two_record_noncompact_plan()
    lowered = lower_plan(bound)

    assert bound.source_size == 16
    assert bound.output_size == 16
    assert bound.mapped_elements == 16
    assert len(lowered.descriptor) < 1_024
    assert len(lowered.descriptor) == 8 * len(lowered.words)


def test_lowering_emits_one_semantic_affine_descriptor() -> None:
    bound = contiguous_dtype_plan(jnp.float32, 1.25, size=6)

    lowered = lower_plan(bound)

    expected_words = (
        AFFINE_DESCRIPTOR_MAGIC,
        AFFINE_DESCRIPTOR_VERSION,
        22,
        6,
        6,
        6,
        1,
        1,
        1,
        1,
        1,
        0,
        1,
        0,
        0,
        0x3FA00000,
        0,
        0,
        1,
        6,
        1,
        1,
    )
    assert lowered.words == expected_words
    assert lowered.descriptor == b"".join(
        word.to_bytes(8, "little") for word in lowered.words
    )


def test_mixed_lowering_encodes_true_operand_and_scalar_policies() -> None:
    forward = build_affine_plan(
        records=(AffineRecord((4,), (1,), 0, (1,), 0, 1.0 + 2.0j),),
        output_size=4,
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=4,
        source_dtype=jnp.float32,
        result_dtype=jnp.complex64,
    )
    reverse = transpose_plan(forward)

    forward_execution = lower_plan(forward)
    reverse_execution = lower_plan(reverse)

    assert forward_execution.semantic is forward
    assert forward_execution.words[8:12] == (_DTYPE_F32, _DTYPE_C64, 1, 0)
    assert reverse_execution.semantic is reverse
    assert reverse_execution.words[8:12] == (_DTYPE_C64, _DTYPE_F32, 2, 0)


@pytest.mark.parametrize("dtype", [
    "bool", "int8", "uint8", "int16", "uint16", "int32", "uint32",
    "int64", "uint64", "float16", "bfloat16", "float32", "float64",
    "complex64", "complex128",
])
def test_scalar_descriptor_words_preserve_width_and_component_order(dtype):
    from tensor0._stride._native_descriptor import _descriptor_scale_words

    scalar_dtype = jnp.dtype(dtype)
    payload = bytes(range(scalar_dtype.itemsize))
    first, second = _descriptor_scale_words(payload, scalar_dtype)
    if jnp.issubdtype(scalar_dtype, jnp.complexfloating):
        width = scalar_dtype.itemsize // 2
        actual = first.to_bytes(width, "little") + second.to_bytes(width, "little")
    else:
        actual = first.to_bytes(scalar_dtype.itemsize, "little")
        assert second == 0
    assert actual == payload
    for invalid in (payload[:-1], payload + b"\x00"):
        with pytest.raises(ValueError, match="scale must contain"):
            _descriptor_scale_words(invalid, scalar_dtype)


def test_scalar_descriptor_words_reject_unsupported_dtype():
    from tensor0._stride._native_descriptor import _descriptor_scale_words

    with pytest.raises(ValueError, match="does not support"):
        _descriptor_scale_words(bytes(4), jnp.dtype("V4"))


def test_vendored_ffi_headers_and_license_match_frozen_hashes() -> None:
    vendor_root = (
        _REPO_ROOT / "crates" / "tensor0-py" / "vendor" / "jaxlib-0.10.1"
    )
    for relative_path, expected in _VENDORED_FFI_FILE_HASHES.items():
        actual = hashlib.sha256(
            (vendor_root / relative_path).read_bytes()
        ).hexdigest()
        assert actual == expected

    packaged_license = (
        _REPO_ROOT / "src" / "tensor0" / "_licenses" / "JAXLIB_FFI_LICENSE.txt"
    )
    assert hashlib.sha256(packaged_license.read_bytes()).hexdigest() == (
        _VENDORED_FFI_FILE_HASHES["LICENSE.txt"]
    )


def test_declared_jax_runtime_matches_vendored_ffi_version() -> None:
    project = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text())
    dependencies = set(project["project"]["dependencies"])
    build_script = (_REPO_ROOT / "crates" / "tensor0-py" / "build.rs").read_text()

    assert "jax==0.10.1" in dependencies
    assert "jaxlib==0.10.1" in dependencies
    assert 'const VENDORED_JAX_VERSION: &str = "0.10.1";' in build_script
    assert 'const VENDORED_JAXLIB_VERSION: &str = "0.10.1";' in build_script


def test_all_small_compact_axis_orders_preserve_exact_address_sets() -> None:
    for shape in ((), (2,), (2, 3), (2, 3, 2), (2, 2, 2, 2)):
        rank = len(shape)
        orders = tuple(permutations(range(rank)))
        for source_order in orders:
            for destination_order in orders:
                bound = _single_plan(shape, source_order, destination_order)
                source_addresses = enumerate_addresses(
                    bound.records[0],
                    "source",
                )
                destination_addresses = enumerate_addresses(
                    bound.records[0],
                    "destination",
                )
                expected = list(range(prod(shape)))
                assert sorted(source_addresses) == expected
                assert sorted(destination_addresses) == expected


def test_descriptor_size_depends_on_rank_not_logical_elements() -> None:
    small = _single_plan((2, 3), (1, 0), (0, 1))
    large = _single_plan((200, 300), (1, 0), (0, 1))

    small_lowered = lower_plan(small)
    large_lowered = lower_plan(large)

    assert len(small_lowered.words) == len(large_lowered.words)
    assert len(small_lowered.descriptor) == len(large_lowered.descriptor)
    assert lower_plan(small) == small_lowered


def test_semantic_and_native_call_keys_exclude_array_identity() -> None:
    first = _single_plan((2, 3, 2), (2, 0, 1), (1, 2, 0))
    second = _single_plan((2, 3, 2), (2, 0, 1), (1, 2, 0))

    assert first == second
    assert hash(first) == hash(second)
    assert lower_plan(first) == lower_plan(second)
    assert hash(lower_plan(first)) == hash(lower_plan(second))


def test_operation_policy_is_semantic_but_not_affine_execution_identity() -> None:
    fresh = contiguous_dtype_plan(jnp.float32, None, size=6)
    assign = build_base_assign_plan(fresh)
    accumulate = build_base_accumulate_plan(fresh)
    dynamic_scale = build_selected_scale_plan(fresh)

    assert fresh.output_init is StridedOutputInit.UNINITIALIZED
    assert fresh.write_kind is StridedWriteKind.ASSIGN
    assert fresh.scalar_kind is StridedScalarKind.STATIC_SCALE_CAST
    assert fresh.reduction_kind is StridedReductionKind.NONE
    assert assign.output_init is StridedOutputInit.PRESERVE_BASE
    assert assign.write_kind is StridedWriteKind.ASSIGN
    assert accumulate.output_init is StridedOutputInit.PRESERVE_BASE
    assert accumulate.write_kind is StridedWriteKind.ACCUMULATE
    assert dynamic_scale.output_init is StridedOutputInit.PRESERVE_BASE
    assert dynamic_scale.scalar_kind is StridedScalarKind.DYNAMIC_SCALE

    assert (
        len(
            {
                plan.semantic_key
                for plan in (fresh, assign, accumulate, dynamic_scale)
            }
        )
        == 4
    )
    assert len(
        {lower_plan(plan).descriptor for plan in (fresh, assign, accumulate)}
    ) == 1


def test_partial_fresh_map_defaults_to_zero_initialization() -> None:
    plan = build_affine_plan(
        records=(AffineRecord((2,), (1,), 0, (1,), 1),),
        output_size=4,
        coverage=CompleteMode.PARTIAL_UNIQUE_ZERO_FILL,
        source_size=2,
        source_dtype=jnp.float32,
        result_dtype=jnp.float32,
    )

    assert plan.output_init is StridedOutputInit.ZERO


def test_uninitialized_output_requires_complete_coverage() -> None:
    with pytest.raises(PlanValidationError, match="uninitialized output"):
        build_affine_plan(
            records=(AffineRecord((2,), (1,), 0, (1,), 1),),
            output_size=4,
            coverage=CompleteMode.PARTIAL_UNIQUE_ZERO_FILL,
            source_size=2,
            source_dtype=jnp.float32,
            result_dtype=jnp.float32,
            output_init=StridedOutputInit.UNINITIALIZED,
        )


def test_structured_sum_sets_zero_initialization_from_one_plan() -> None:
    bound = build_affine_plan(
        records=(
            AffineRecord(
                logical_shape=(4, 3),
                source_strides=(3, 1),
                source_offset=0,
                destination_strides=(0, 1),
                destination_offset=0,
                reduction_axes=(0,),
            ),
        ),
        output_size=3,
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=12,
        source_dtype=jnp.float32,
        result_dtype=jnp.float32,
        reduction_kind=StridedReductionKind.SUM,
    )

    assert bound.output_init is StridedOutputInit.ZERO
    assert bound.records[0].reduction_axes == (0,)


def test_empty_structured_sum_retains_semantic_reduction_axis() -> None:
    plan = build_affine_plan(
        records=(
            AffineRecord(
                (0, 3),
                (3, 1),
                0,
                (0, 1),
                0,
                reduction_axes=(0,),
            ),
        ),
        output_size=3,
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=0,
        source_dtype=jnp.float32,
        result_dtype=jnp.float32,
        reduction_kind=StridedReductionKind.SUM,
    )

    assert plan.records[0].reduction_axes == (0,)


@pytest.mark.parametrize(
    ("record", "output_size", "message"),
    (
        (
            AffineRecord((4, 3), (3, 1), 0, (1, 1), 0, reduction_axes=(0,)),
            3,
            "destination depends on reduction axis",
        ),
        (
            AffineRecord((4, 3), (3, 1), 0, (3, 1), 0),
            12,
            "requires a reduction axis",
        ),
    ),
)
def test_structured_sum_rejects_unstructured_output_collisions(
    record: AffineRecord,
    output_size: int,
    message: str,
) -> None:
    with pytest.raises(PlanValidationError, match=message):
        build_affine_plan(
            records=(record,),
            output_size=output_size,
            coverage=CompleteMode.COMPLETE_UNIQUE,
            source_size=12,
            source_dtype=jnp.float32,
            result_dtype=jnp.float32,
            reduction_kind=StridedReductionKind.SUM,
        )


def test_multi_record_structured_sum_requires_explicit_accumulation() -> None:
    records = tuple(
        AffineRecord(
            logical_shape=(4, 3),
            source_strides=(3, 1),
            source_offset=offset,
            destination_strides=(0, 1),
            destination_offset=0,
            reduction_axes=(0,),
        )
        for offset in (0, 12)
    )
    arguments: Any = dict(
        records=records,
        output_size=3,
        coverage=CompleteMode.PARTIAL_UNIQUE_ZERO_FILL,
        source_size=24,
        source_dtype=jnp.float32,
        result_dtype=jnp.float32,
        reduction_kind=StridedReductionKind.SUM,
    )

    with pytest.raises(PlanValidationError, match="requires accumulate"):
        build_affine_plan(**arguments)

    plan = build_affine_plan(
        **arguments,
        write_kind=StridedWriteKind.ACCUMULATE,
    )
    assert len(plan.records) == 2


def test_fresh_map_executor_rejects_reduction_operation_plan() -> None:
    plan = build_affine_plan(
        records=(
            AffineRecord(
                (4, 3),
                (3, 1),
                0,
                (0, 1),
                0,
                reduction_axes=(0,),
            ),
        ),
        output_size=3,
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=12,
        source_dtype=jnp.float32,
        result_dtype=jnp.float32,
        reduction_kind=StridedReductionKind.SUM,
    )

    with pytest.raises(ValueError, match="fresh affine map"):
        _execute_map(jnp.arange(12, dtype=jnp.float32), plan=plan)


@pytest.mark.parametrize(
    ("factory", "record_count", "size"),
    [
        (u1_two_record_plan, 2, 5),
        (u1_three_record_plan, 3, 4608),
    ],
)
def test_frozen_u1_address_fixtures_validate(
    factory: Callable[[], AffinePlan],
    record_count: int,
    size: int,
) -> None:
    bound = factory()

    assert len(bound.records) == record_count
    assert bound.source_size == size
    assert bound.output_size == size
    source_addresses = tuple(
        address
        for record in bound.records
        for address in enumerate_addresses(record, "source")
    )
    destination_addresses = tuple(
        address
        for record in bound.records
        for address in enumerate_addresses(record, "destination")
    )
    assert sorted(source_addresses) == list(range(size))
    assert sorted(destination_addresses) == list(range(size))


def test_positive_strides_that_repeat_source_addresses_are_allowed() -> None:
    record = AffineRecord((5, 5), (2, 2), 0, (5, 1), 0)

    plan = build_affine_plan(
        records=(record,),
        output_size=25,
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=25,
        source_dtype=jnp.float32,
        result_dtype=jnp.float32,
    )

    assert plan.required_source_size == 17


def test_positive_destination_strides_that_alias_are_rejected() -> None:
    record = AffineRecord((5, 5), (5, 1), 0, (2, 2), 0)

    with pytest.raises(PlanValidationError, match="noninjective_view"):
        build_affine_plan(
            records=(record,),
            output_size=25,
            coverage=CompleteMode.COMPLETE_UNIQUE,
            source_size=25,
            source_dtype=jnp.float32,
            result_dtype=jnp.float32,
        )


def test_host_pointer_domain_is_checked_when_binding() -> None:
    extent = sys.maxsize // jnp.dtype(jnp.float32).itemsize + 2
    record = AffineRecord((extent,), (1,), 0, (1,), 0)

    with pytest.raises(PlanValidationError, match="host_address_overflow"):
        build_affine_plan(
            records=(record,),
            output_size=extent,
            coverage=CompleteMode.COMPLETE_UNIQUE,
            source_size=extent,
            source_dtype=jnp.float32,
            result_dtype=jnp.float32,
        )
