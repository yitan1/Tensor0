"""Private native diagnostics used by the stride test suite."""

from __future__ import annotations

from jax import Array
import jax.numpy as jnp

from .. import _native
from ._native import (
    _affine_ffi_call,
    enable_threads,
    native_available,
    set_num_threads,
)


def _native_call_count_for_tests() -> int:
    if not native_available():
        return 0
    return int(_native._stride_native_call_count())


def _observe_native_leaf_kernels_for_tests(enabled: bool) -> None:
    if native_available():
        _native._observe_stride_leaf_kernels_for_tests(enabled)


def _native_leaf_kernel_masks_for_tests() -> tuple[int, int]:
    if not native_available():
        return 0, 0
    return _native._stride_leaf_kernel_masks_for_tests()


def _reset_native_call_count_for_tests() -> None:
    if native_available():
        _native._reset_stride_native_call_count()


def _native_worker_counts_for_tests() -> tuple[int, int]:
    if not native_available():
        return 0, 0
    return (
        int(_native._stride_last_worker_count()),
        int(_native._stride_last_available_worker_count()),
    )


def _native_reduction_fiber_chunks_for_tests() -> int:
    if not native_available():
        return 0
    return int(_native._stride_last_reduction_fiber_chunks())


def _native_grouped_output_owner_for_tests() -> bool:
    if not native_available():
        return False
    return bool(_native._stride_last_grouped_output_owner())


def _native_alias_pointers_for_tests() -> tuple[int, int, int, int]:
    if not native_available():
        return 0, 0, 0, 0
    base, source, factor, result = _native._stride_alias_pointers()
    return int(base), int(source), int(factor), int(result)


def _reset_native_alias_pointers_for_tests() -> None:
    if native_available():
        _native._reset_stride_alias_pointers()


def _set_native_worker_limit_for_tests(limit: int | None) -> None:
    if native_available():
        if limit is None:
            enable_threads()
        else:
            set_num_threads(limit)


def _set_native_force_generic_for_tests(enabled: bool) -> None:
    if native_available():
        _native._set_stride_force_generic_for_tests(enabled)


def _set_native_force_generated_baseline_for_tests(enabled: bool) -> None:
    if native_available():
        _native._set_stride_force_generated_baseline_for_tests(enabled)


def _set_native_reduction_fiber_parallel_mode_for_tests(
    mode: int | None,
) -> None:
    if native_available():
        _native._set_stride_reduction_fiber_parallel_mode_for_tests(
            (1 << 64) - 1 if mode is None else mode
        )


def _set_native_disable_f16_f32_contiguous_simd_for_tests(
    enabled: bool,
) -> None:
    if native_available():
        _native._set_stride_disable_f16_f32_contiguous_simd_for_tests(enabled)


def _set_native_disable_f16_f16_contiguous_simd_for_tests(
    enabled: bool,
) -> None:
    if native_available():
        _native._set_stride_disable_f16_f16_contiguous_simd_for_tests(enabled)


def _set_native_disable_f32_c64_contiguous_simd_for_tests(
    enabled: bool,
) -> None:
    if native_available():
        _native._set_stride_disable_f32_c64_contiguous_simd_for_tests(enabled)


def _prepared_native_call_for_tests(
    source: Array,
    *,
    descriptor: bytes,
    output_size: int,
    alias_source_result: bool = False,
) -> Array:
    return _affine_ffi_call(
        jnp.asarray(source),
        descriptor=descriptor,
        output_size=output_size,
        result_dtype=jnp.dtype(source.dtype),
        alias_source_result=alias_source_result,
    )
