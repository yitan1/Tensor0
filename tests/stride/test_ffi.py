from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import jax
import jax.numpy as jnp
from jax.typing import DTypeLike
import numpy as np
import pytest

from tensor0._stride import (
    CompleteMode,
    StridedCopyRecord,
    strided_copy,
    StridedCopyPlan,
)
from tensor0 import _native
import tensor0._stride._ffi as ffi_module
from tensor0._stride._compiler import COMPILED_DESCRIPTOR_VERSION
from tensor0._stride._ffi import (
    _ensure_registered_v7,
    _native_call_count_for_tests,
    _native_worker_counts_for_tests,
    _prepared_native_call_for_tests,
    _reset_native_call_count_for_tests,
    _set_native_worker_limit_for_tests,
    native_available,
)
from tensor0._stride._native_lowering import lower_plan
from tensor0._stride._plan import (
    build_strided_copy_plan,
    transpose_same_dtype_plan,
)

from ._fixtures import (
    contiguous_dtype_plan,
    dtype_transpose_plan,
    empty_plan,
    empty_partial_plan,
    gapped_interval_source_plan,
    gapped_single_source_plan,
    large_partial_transpose_plan,
    partial_mixed_plan,
    rank2_transpose_plan,
    rank4_avx2_shape_plan,
    rank4_tiled_plan,
    rank_zero_plan,
    two_record_noncompact_plan,
    u1_three_record_plan,
    u1_two_record_plan,
)
from ._oracle import execute_reference


pytestmark = pytest.mark.skipif(
    not native_available(),
    reason="the Tensor0 extension was built without JAX FFI headers",
)

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_native_handler_build_identity_matches_runtime() -> None:
    assert _native._stride_ffi_abi_version() == COMPILED_DESCRIPTOR_VERSION
    assert _native._stride_ffi_build_versions() == (
        jax.__version__,
        version("jaxlib"),
    )


def test_one_direction_is_canonicalized_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = contiguous_dtype_plan(jnp.int32, 1, size=32)
    source = jnp.arange(plan.source_size, dtype=jnp.int32)
    original = ffi_module.compile_plan
    calls: list[StridedCopyPlan] = []

    def counted_compile(bound: StridedCopyPlan):
        calls.append(bound)
        return original(bound)

    monkeypatch.setattr(ffi_module, "compile_plan", counted_compile)
    strided_copy(source, plan=plan, native_required=True).block_until_ready()

    assert calls == [plan]


def test_fresh_process_registration_and_repeated_module_import() -> None:
    script = textwrap.dedent(
        """
        import importlib
        import json

        import jax
        import jax.numpy as jnp

        import tensor0._stride._ffi as stride_ffi
        from tests.stride._fixtures import two_record_noncompact_plan

        plan = two_record_noncompact_plan()
        source = jnp.arange(plan.source_size, dtype=jnp.float32)

        def execute(module):
            result = jax.jit(
                lambda value: module.strided_copy(
                    value,
                    plan=plan,
                    native_required=True,
                )
            )(source)
            result.block_until_ready()

        execute(stride_ffi)
        stride_ffi = importlib.reload(stride_ffi)
        execute(stride_ffi)
        print(json.dumps({
            "available": stride_ffi.native_available(),
            "calls": stride_ffi._native_call_count_for_tests(),
        }))
        """
    )

    for _ in range(2):
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=_REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        assert json.loads(completed.stdout) == {
            "available": True,
            "calls": 2,
        }


@pytest.mark.parametrize("x64_enabled", [False, True])
def test_x64_mode_preserves_uint64_descriptor_words(
    x64_enabled: bool,
) -> None:
    script = textwrap.dedent(
        """
        import json

        import jax
        import jax.numpy as jnp

        from tensor0._stride import (
            CompleteMode,
            StridedCopyRecord,
            strided_copy,
        )
        from tensor0._stride._ffi import _native_call_count_for_tests
        from tensor0._stride._native_lowering import lower_plan
        from tensor0._stride._plan import (
            build_strided_copy_plan,
        )

        large_stride = 2**31 + 17
        record = StridedCopyRecord(
            (1,),
            (large_stride,),
            0,
            (large_stride + 2,),
            0,
            1,
        )
        plan = build_strided_copy_plan(
            records=(record,),
            output_size=1,
            coverage=CompleteMode.COMPLETE_UNIQUE,
            source_size=1,
            source_dtype=jnp.float32,
            result_dtype=jnp.float32,
        )
        lowered = lower_plan(plan)
        result = jax.jit(
            lambda value: strided_copy(
                value,
                plan=lowered,
                native_required=True,
            )
        )(jnp.asarray([3.5], dtype=jnp.float32))
        result.block_until_ready()
        print(json.dumps({
            "x64_enabled": jax.config.x64_enabled,
            "large_stride_count": lowered.words.count(large_stride),
            "larger_stride_count": lowered.words.count(large_stride + 2),
            "calls": _native_call_count_for_tests(),
        }))
        """
    )
    environment = os.environ.copy()
    environment["JAX_ENABLE_X64"] = "1" if x64_enabled else "0"
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=_REPO_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == {
        "x64_enabled": x64_enabled,
        "large_stride_count": 1,
        "larger_stride_count": 1,
        "calls": 1,
    }


def test_jit_cache_reuses_equal_plans_and_separates_unequal_plans() -> None:
    first = two_record_noncompact_plan()
    equal = two_record_noncompact_plan()
    unequal = rank2_transpose_plan(rows=4, columns=4)
    source = jnp.arange(first.source_size, dtype=jnp.float32)

    def execute(
        value: jax.Array,
        *,
        plan: StridedCopyPlan,
    ) -> jax.Array:
        return strided_copy(value, plan=plan, native_required=True)

    compiled = jax.jit(execute, static_argnames=("plan",))
    cache_size = getattr(compiled, "_cache_size")
    initial_size = int(cache_size())

    first_result = compiled(source, plan=first)
    first_result.block_until_ready()
    first_size = int(cache_size())

    equal_result = compiled(source, plan=equal)
    equal_result.block_until_ready()
    equal_size = int(cache_size())

    unequal_result = compiled(source, plan=unequal)
    unequal_result.block_until_ready()
    unequal_size = int(cache_size())

    np.testing.assert_array_equal(
        np.asarray(first_result),
        np.asarray(execute_reference(source, first)),
    )
    np.testing.assert_array_equal(
        np.asarray(equal_result),
        np.asarray(first_result),
    )
    np.testing.assert_array_equal(
        np.asarray(unequal_result),
        np.asarray(execute_reference(source, unequal)),
    )
    assert first_size == initial_size + 1
    assert equal_size == first_size
    assert unequal_size == equal_size + 1


def test_multi_device_cpu_uses_partitioned_native_without_implicit_all_gather() -> (
    None
):
    script = textwrap.dedent(
        """
        import json

        import jax
        import jax.numpy as jnp
        import numpy as np
        from jax.sharding import Mesh
        from jax.sharding import NamedSharding
        from jax.sharding import PartitionSpec as P

        from tensor0._stride import (
            CompleteMode,
            StridedCopyRecord,
            strided_copy,
        )
        from tensor0._stride._ffi import (
            _native_call_count_for_tests,
            _reset_native_call_count_for_tests,
        )
        from tensor0._stride._plan import (
            build_strided_copy_plan,
        )
        from tests.stride._fixtures import large_contiguous_plan
        from tests.stride._oracle import execute_reference

        plan = large_contiguous_plan()
        mesh = Mesh(np.asarray(jax.devices()), ("device",))
        batch_sharding = NamedSharding(mesh, P("device", None))
        host = np.arange(
            2 * plan.source_size,
            dtype=np.float32,
        ).reshape(2, plan.source_size)
        source = jax.device_put(host, batch_sharding)
        automatic = jax.jit(
            lambda value: strided_copy(value, plan=plan),
            in_shardings=batch_sharding,
            out_shardings=batch_sharding,
        ).lower(source).compile()
        optimized_hlo = automatic.as_text().lower()

        _reset_native_call_count_for_tests()
        actual = automatic(source)
        actual.block_until_ready()
        np.testing.assert_array_equal(
            np.asarray(actual),
            np.asarray(execute_reference(jnp.asarray(host), plan)),
        )
        batch_native_calls = _native_call_count_for_tests()

        replicated_sharding = NamedSharding(mesh, P())
        replicated_source = jax.device_put(host[0], replicated_sharding)
        replicated = jax.jit(
            lambda value: strided_copy(value, plan=plan),
            in_shardings=replicated_sharding,
            out_shardings=replicated_sharding,
        ).lower(replicated_source).compile()
        replicated_hlo = replicated.as_text().lower()
        _reset_native_call_count_for_tests()
        replicated_actual = replicated(replicated_source)
        replicated_actual.block_until_ready()
        np.testing.assert_array_equal(
            np.asarray(replicated_actual),
            np.asarray(execute_reference(jnp.asarray(host[0]), plan)),
        )
        replicated_native_calls = _native_call_count_for_tests()

        _reset_native_call_count_for_tests()
        mapped = jax.pmap(
            lambda value: strided_copy(value, plan=plan)
        )(jnp.asarray(host))
        mapped.block_until_ready()
        np.testing.assert_array_equal(np.asarray(mapped), np.asarray(actual))
        pmap_native_calls = _native_call_count_for_tests()

        storage_sharding = NamedSharding(mesh, P("device"))
        storage_source = jax.device_put(host[0], storage_sharding)
        try:
            jax.jit(
                lambda value: strided_copy(value, plan=plan),
                in_shardings=storage_sharding,
                out_shardings=storage_sharding,
            ).lower(storage_source).compile()
        except Exception as error:
            storage_error = str(error)
        else:
            storage_error = ""

        unsupported_shape = (2,) * 9
        unsupported_strides = tuple(2 ** (8 - axis) for axis in range(9))
        unsupported_plan = build_strided_copy_plan(
            records=(
                StridedCopyRecord(
                    unsupported_shape,
                    unsupported_strides,
                    0,
                    unsupported_strides,
                    0,
                    1.0,
                ),
            ),
            output_size=512,
            coverage=CompleteMode.COMPLETE_UNIQUE,
            source_size=512,
            source_dtype=jnp.float32,
            result_dtype=jnp.float32,
        )
        unsupported_host = np.arange(
            2 * unsupported_plan.source_size,
            dtype=np.float32,
        ).reshape(2, unsupported_plan.source_size)
        unsupported_source = jax.device_put(unsupported_host, batch_sharding)
        try:
            jax.jit(
                lambda value: strided_copy(value, plan=unsupported_plan),
                in_shardings=batch_sharding,
                out_shardings=batch_sharding,
            ).lower(unsupported_source).compile()
        except Exception as error:
            unsupported_error = str(error)
        else:
            unsupported_error = ""

        try:
            jax.jit(
                lambda value: strided_copy(
                    value,
                    plan=plan,
                    native_required=True,
                ),
                in_shardings=batch_sharding,
                out_shardings=batch_sharding,
            ).lower(source)
        except Exception as error:
            forced_error = str(error)
        else:
            forced_error = ""

        print(json.dumps({
            "devices": len(jax.devices()),
            "batch_native_calls": batch_native_calls,
            "batch_all_gather_count": optimized_hlo.count("all-gather"),
            "batch_native_target_count": optimized_hlo.count(
                "tensor0_stride_"
            ),
            "replicated_native_calls": replicated_native_calls,
            "replicated_all_gather_count": replicated_hlo.count(
                "all-gather"
            ),
            "replicated_native_target_count": replicated_hlo.count(
                "tensor0_stride_"
            ),
            "pmap_native_calls": pmap_native_calls,
            "storage_error": (
                "cannot shard the packed storage axis" in storage_error
            ),
            "unsupported_error": (
                "native_cpu_executor_unavailable" in unsupported_error
            ),
            "forced_error": (
                "only supports unsharded CPU input" in forced_error
            ),
        }))
        """
    )
    environment = os.environ.copy()
    environment["JAX_PLATFORMS"] = "cpu"
    environment["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=_REPO_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == {
        "devices": 2,
        "batch_native_calls": 2,
        "batch_all_gather_count": 0,
        "batch_native_target_count": 1,
        "replicated_native_calls": 2,
        "replicated_all_gather_count": 0,
        "replicated_native_target_count": 1,
        "pmap_native_calls": 2,
        "storage_error": True,
        "unsupported_error": True,
        "forced_error": True,
    }


def test_native_vertical_slice_is_one_observable_call_without_indices() -> None:
    bound = two_record_noncompact_plan()
    source = jnp.arange(bound.source_size, dtype=jnp.float32)
    expected = execute_reference(source, bound)
    compiled = jax.jit(
        lambda value: strided_copy(value, plan=bound, native_required=True)
    )

    _reset_native_call_count_for_tests()
    actual = compiled(source)
    actual.block_until_ready()

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected))
    assert _native_call_count_for_tests() == 1

    hlo = str(compiled.lower(source).compiler_ir()).lower()
    assert hlo.count("custom_call") == 1
    assert "gather" not in hlo
    assert "scatter" not in hlo


@pytest.mark.parametrize(
    ("factory", "expected"),
    [
        (
            gapped_single_source_plan,
            np.asarray([0.0, -4.0, -8.0], dtype=np.float32),
        ),
        (
            gapped_interval_source_plan,
            np.asarray([0.0, 1.5, -1.5, -2.0], dtype=np.float32),
        ),
    ],
)
def test_injective_source_with_storage_gaps_executes_natively(
    factory: Callable[[], StridedCopyPlan],
    expected: np.ndarray,
) -> None:
    bound = factory()
    source = jnp.arange(bound.source_size, dtype=jnp.float32)

    _reset_native_call_count_for_tests()
    actual = jax.jit(
        lambda value: strided_copy(value, plan=bound, native_required=True)
    )(source)
    actual.block_until_ready()

    np.testing.assert_array_equal(
        np.asarray(actual),
        expected,
    )
    assert _native_call_count_for_tests() == 1


@pytest.mark.parametrize("factory", [u1_two_record_plan, u1_three_record_plan])
def test_frozen_u1_multi_record_fixtures_execute_natively(
    factory: Callable[[], StridedCopyPlan],
) -> None:
    bound = factory()
    source = jnp.arange(bound.source_size, dtype=jnp.float32)
    expected = execute_reference(source, bound)

    _reset_native_call_count_for_tests()
    actual = jax.jit(
        lambda value: strided_copy(value, plan=bound, native_required=True)
    )(source)
    actual.block_until_ready()

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected))
    assert _native_call_count_for_tests() == 1


def test_native_handler_rejects_an_unknown_coverage_mode() -> None:
    bound = partial_mixed_plan()
    words = list(lower_plan(bound).words)
    words[21] = 99
    descriptor = b"".join(word.to_bytes(8, "little") for word in words)
    source = jnp.arange(bound.source_size, dtype=jnp.float32)

    with pytest.raises(Exception, match="coverage mode is invalid"):
        _prepared_native_call_for_tests(
            source,
            descriptor=descriptor,
            output_size=bound.output_size,
        ).block_until_ready()


def test_native_handler_revalidates_dtype_specific_scale_bits() -> None:
    bound = dtype_transpose_plan(jnp.float16, 1.25)
    words = list(lower_plan(bound).words)
    words[26] = 1 << 16
    descriptor = b"".join(word.to_bytes(8, "little") for word in words)
    source = jnp.arange(bound.source_size, dtype=jnp.float16)

    with pytest.raises(Exception, match="scale bits do not match"):
        _prepared_native_call_for_tests(
            source,
            descriptor=descriptor,
            output_size=bound.output_size,
        ).block_until_ready()


def test_native_handler_revalidates_mixed_operand_policy_header() -> None:
    template = partial_mixed_plan()
    bound = build_strided_copy_plan(
        records=template.records,
        output_size=template.output_size,
        coverage=template.coverage,
        source_size=template.source_size,
        source_dtype=jnp.float32,
        result_dtype=jnp.complex64,
    )
    execution = lower_plan(bound)
    words = list(execution.words)
    words[12] = words[11]
    forged = replace(
        execution,
        descriptor=b"".join(word.to_bytes(8, "little") for word in words),
    )
    source = jnp.arange(bound.source_size, dtype=jnp.float32)

    with pytest.raises(Exception, match="scalar dtype witness mismatch"):
        ffi_module._mixed_ffi_call_v2(source, forged).block_until_ready()


def test_native_handler_revalidates_records_and_actual_buffers() -> None:
    bound = two_record_noncompact_plan()
    lowered = lower_plan(bound)
    source = jnp.arange(bound.source_size, dtype=jnp.float32)

    words = list(lowered.words)
    first_destination_stride = 23 + 6 + 2 * len(bound.records[0].logical_shape)
    words[first_destination_stride + 1] = words[first_destination_stride]
    forged_record = b"".join(word.to_bytes(8, "little") for word in words)
    with pytest.raises(Exception, match="destination strides do not prove"):
        _prepared_native_call_for_tests(
            source,
            descriptor=forged_record,
            output_size=bound.output_size,
        ).block_until_ready()

    with pytest.raises(Exception, match="actual flat storage size"):
        _prepared_native_call_for_tests(
            source[:-1],
            descriptor=lowered.descriptor,
            output_size=bound.output_size,
        ).block_until_ready()
    with pytest.raises(Exception, match="actual flat storage size"):
        _prepared_native_call_for_tests(
            source,
            descriptor=lowered.descriptor,
            output_size=bound.output_size - 1,
        ).block_until_ready()


def test_native_handler_rejects_mismatched_flat_batch_counts() -> None:
    bound = rank2_transpose_plan(rows=2, columns=3)
    lowered = lower_plan(bound)
    source = jnp.arange(2 * bound.source_size, dtype=jnp.float32)
    descriptor = np.frombuffer(lowered.descriptor, dtype=np.uint8).copy()
    call = jax.ffi.ffi_call(
        _ensure_registered_v7("float32"),
        jax.ShapeDtypeStruct((3 * bound.output_size,), jnp.float32),
        input_layouts=((0,),),
        output_layouts=(0,),
        vmap_method="expand_dims",
        custom_call_api_version=4,
    )

    with pytest.raises(Exception, match="flat storage batch count"):
        jax.jit(lambda value: call(value, descriptor=descriptor))(
            source
        ).block_until_ready()


def test_native_handler_rejects_batched_input_output_alias() -> None:
    bound = rank2_transpose_plan(rows=2, columns=3)
    source = jnp.arange(2 * bound.source_size, dtype=jnp.float32).reshape(
        2,
        bound.source_size,
    )

    with pytest.raises(Exception, match="source and result buffers overlap"):
        _prepared_native_call_for_tests(
            source,
            descriptor=lower_plan(bound).descriptor,
            output_size=bound.output_size,
            alias_source_result=True,
        ).block_until_ready()


def test_multidimensional_batch_prefix_uses_one_flat_native_call() -> None:
    bound = rank2_transpose_plan(rows=2, columns=3)
    source = jnp.arange(
        2 * 3 * bound.source_size,
        dtype=jnp.float32,
    ).reshape(2, 3, bound.source_size)
    expected = jax.vmap(jax.vmap(lambda value: execute_reference(value, bound)))(source)
    compiled = jax.jit(
        lambda value: strided_copy(
            value,
            plan=bound,
            native_required=True,
        )
    )

    _reset_native_call_count_for_tests()
    actual = compiled(source)
    actual.block_until_ready()

    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    assert _native_call_count_for_tests() == 1
    hlo = str(compiled.lower(source).compiler_ir()).lower()
    assert hlo.count("custom_call") == 1
    assert f"tensor<{source.size}xf32>" in hlo


def _assert_parallel_matches_pool_size_one(
    bound: StridedCopyPlan,
) -> np.ndarray:
    dtype = jnp.dtype(bound.source_dtype)
    if jnp.issubdtype(dtype, jnp.integer):
        source = jnp.arange(bound.source_size, dtype=dtype)
    elif dtype == jnp.dtype(jnp.complex64):
        indices = jnp.arange(bound.source_size, dtype=jnp.int32)
        real = (indices % 2048).astype(jnp.float32) / 256.0 - 4.0
        source = (real + 1j * (1.0 - real)).astype(dtype)
    else:
        indices = jnp.arange(bound.source_size, dtype=jnp.int32)
        source = ((indices % 2048).astype(jnp.float32) / 256.0 - 4.0).astype(dtype)
    compiled = (
        jax.jit(
            lambda value: strided_copy(
                value,
                plan=bound,
                native_required=True,
            )
        )
        .lower(source)
        .compile()
    )

    try:
        _set_native_worker_limit_for_tests(1)
        _reset_native_call_count_for_tests()
        single_thread = compiled(source)
        single_thread.block_until_ready()
        single_workers, available_workers = _native_worker_counts_for_tests()
        assert _native_call_count_for_tests() == 1
        assert single_workers == 1

        _set_native_worker_limit_for_tests(None)
        _reset_native_call_count_for_tests()
        parallel = compiled(source)
        parallel.block_until_ready()
        parallel_workers, parallel_available = _native_worker_counts_for_tests()
        assert _native_call_count_for_tests() == 1
    finally:
        _set_native_worker_limit_for_tests(None)

    parallel_host = np.asarray(parallel)
    assert parallel_host.tobytes() == np.asarray(single_thread).tobytes()
    assert parallel_available == available_workers
    assert 1 <= parallel_workers <= max(parallel_available, 1)
    if parallel_available > 1:
        assert parallel_workers > 1
    return parallel_host


def test_large_transpose_matches_pool_size_one_and_uses_bounded_workers() -> None:
    forward = rank2_transpose_plan(rows=1024, columns=768)
    for bound in (forward, transpose_same_dtype_plan(forward)):
        _assert_parallel_matches_pool_size_one(bound)


def test_large_compact_and_rank4_chunks_match_pool_size_one() -> None:
    plans = (
        contiguous_dtype_plan(jnp.float32, -0.75, size=2_097_152),
        rank4_avx2_shape_plan((64, 32, 32, 64)),
    )
    for bound in plans:
        _assert_parallel_matches_pool_size_one(bound)


def test_large_scalar_dtype_chunks_match_pool_size_one() -> None:
    plans = (
        dtype_transpose_plan(
            jnp.float16,
            1.3,
            rows=1024,
            columns=512,
        ),
        dtype_transpose_plan(
            jnp.bfloat16,
            -0.7,
            rows=1024,
            columns=512,
        ),
        dtype_transpose_plan(
            jnp.complex64,
            0.75 - 0.5j,
            rows=512,
            columns=256,
        ),
        dtype_transpose_plan(
            jnp.complex64,
            -0.25 + 0.75j,
            rows=517,
            columns=263,
        ),
        dtype_transpose_plan(
            jnp.int32,
            -3,
            rows=512,
            columns=512,
        ),
    )
    for bound in plans:
        _assert_parallel_matches_pool_size_one(bound)


def test_large_partial_zero_fill_matches_pool_size_one() -> None:
    bound = large_partial_transpose_plan()
    actual = _assert_parallel_matches_pool_size_one(bound)
    padding = 1024
    np.testing.assert_array_equal(actual[:padding], 0)
    np.testing.assert_array_equal(actual[-padding:], 0)


def test_concurrent_parallel_native_invocations_are_race_free() -> None:
    bound = rank2_transpose_plan(rows=512, columns=512)
    compiled = jax.jit(
        lambda value: strided_copy(
            value,
            plan=bound,
            native_required=True,
        )
    )
    reference = jax.jit(lambda value: execute_reference(value, bound))
    base = jnp.arange(bound.source_size, dtype=jnp.float32)
    compiled(base).block_until_ready()
    reference(base).block_until_ready()

    def execute(seed: int) -> np.ndarray:
        result = compiled(base + seed)
        result.block_until_ready()
        return np.asarray(result)

    _reset_native_call_count_for_tests()
    with ThreadPoolExecutor(max_workers=4) as executor:
        actual = tuple(executor.map(execute, range(4)))

    for seed, result in enumerate(actual):
        expected = reference(base + seed)
        expected.block_until_ready()
        np.testing.assert_array_equal(result, np.asarray(expected))
    assert _native_call_count_for_tests() == 4


@pytest.mark.parametrize("factory", [partial_mixed_plan, empty_partial_plan])
def test_partial_unique_zero_fill_executes_natively(
    factory: Callable[[], StridedCopyPlan],
) -> None:
    bound = factory()
    source = jnp.arange(bound.source_size, dtype=jnp.float32)
    expected = execute_reference(source, bound)

    _reset_native_call_count_for_tests()
    compiled = jax.jit(
        lambda value: strided_copy(value, plan=bound, native_required=True)
    )
    actual = compiled(source)
    actual.block_until_ready()

    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    assert _native_call_count_for_tests() == 1
    hlo = str(compiled.lower(source).compiler_ir()).lower()
    assert hlo.count("custom_call") == 1
    assert "gather" not in hlo
    assert "scatter" not in hlo


def test_batched_empty_partial_plan_derives_batch_from_flat_result() -> None:
    bound = empty_partial_plan()
    source = jnp.empty((3, 0), dtype=jnp.float32)

    _reset_native_call_count_for_tests()
    actual = jax.jit(
        lambda value: strided_copy(
            value,
            plan=bound,
            native_required=True,
        )
    )(source)
    actual.block_until_ready()

    np.testing.assert_array_equal(np.asarray(actual), np.zeros((3, 8)))
    assert _native_call_count_for_tests() == 1


@pytest.mark.parametrize(
    ("dtype", "scale"),
    [
        (jnp.float16, 1.3),
        (jnp.bfloat16, -0.7),
        (jnp.complex64, 0.75 - 0.5j),
        (jnp.int32, -1),
    ],
)
def test_enabled_same_dtype_kernels_match_reference(
    dtype: DTypeLike,
    scale: int | float | complex,
) -> None:
    bound = dtype_transpose_plan(dtype, scale)
    if jnp.dtype(dtype) == jnp.dtype(jnp.complex64):
        source = (
            jnp.linspace(-2.0, 3.0, bound.source_size, dtype=jnp.float32)
            + 1j * jnp.linspace(4.0, -1.0, bound.source_size, dtype=jnp.float32)
        ).astype(jnp.complex64)
    elif jnp.dtype(dtype) == jnp.dtype(jnp.int32):
        source = jnp.arange(bound.source_size, dtype=jnp.int32)
        source = source.at[0].set(np.iinfo(np.int32).min)
        source = source.at[1].set(np.iinfo(np.int32).max)
    else:
        source = jnp.linspace(-3.0, 2.0, bound.source_size, dtype=dtype)
    expected = execute_reference(source, bound)

    _reset_native_call_count_for_tests()
    compiled = jax.jit(
        lambda value: strided_copy(value, plan=bound, native_required=True)
    )
    actual = compiled(source)
    actual.block_until_ready()

    if jnp.dtype(dtype) == jnp.dtype(jnp.complex64):
        np.testing.assert_allclose(
            np.asarray(actual),
            np.asarray(expected),
            rtol=1e-6,
            atol=1e-6,
        )
    else:
        assert np.asarray(actual).tobytes() == np.asarray(expected).tobytes()
    assert _native_call_count_for_tests() == 1
    hlo = str(compiled.lower(source).compiler_ir()).lower()
    assert hlo.count("custom_call") == 1
    assert "gather" not in hlo
    assert "scatter" not in hlo


@pytest.mark.parametrize(
    ("dtype", "scale"),
    [(jnp.float16, 1.3), (jnp.bfloat16, -0.7)],
)
def test_narrow_float_kernels_cover_every_storage_bit_pattern(
    dtype: DTypeLike,
    scale: float,
) -> None:
    bits = np.arange(1 << 16, dtype=np.uint16)
    host_values = bits.view(np.dtype(jnp.dtype(dtype)))
    source = jnp.asarray(host_values)
    bound = contiguous_dtype_plan(dtype, scale, size=source.size)
    expected = execute_reference(source, bound)

    _reset_native_call_count_for_tests()
    actual = jax.jit(
        lambda value: strided_copy(value, plan=bound, native_required=True)
    )(source)
    actual.block_until_ready()

    actual_host = np.asarray(actual)
    expected_host = np.asarray(expected)
    actual_nan = np.isnan(actual_host.astype(np.float32))
    expected_nan = np.isnan(expected_host.astype(np.float32))
    np.testing.assert_array_equal(actual_nan, expected_nan)
    np.testing.assert_array_equal(
        actual_host.view(np.uint16)[~expected_nan],
        expected_host.view(np.uint16)[~expected_nan],
    )
    assert _native_call_count_for_tests() == 1


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16, jnp.complex64])
def test_partial_zero_fill_supports_enabled_inexact_dtypes(
    dtype: DTypeLike,
) -> None:
    template = partial_mixed_plan()
    bound = build_strided_copy_plan(
        records=template.records,
        output_size=template.output_size,
        coverage=template.coverage,
        source_size=template.source_size,
        source_dtype=dtype,
        result_dtype=dtype,
    )
    source = jnp.linspace(-2.0, 3.0, bound.source_size, dtype=dtype)
    expected = execute_reference(source, bound)

    _reset_native_call_count_for_tests()
    actual = jax.jit(
        lambda value: strided_copy(value, plan=bound, native_required=True)
    )(source)
    actual.block_until_ready()

    if jnp.dtype(dtype) == jnp.dtype(jnp.complex64):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected))
    else:
        assert np.asarray(actual).tobytes() == np.asarray(expected).tobytes()
    assert _native_call_count_for_tests() == 1


@pytest.mark.parametrize("dtype", [jnp.complex64, jnp.int32])
def test_enabled_dtype_batching_remains_one_native_call(
    dtype: DTypeLike,
) -> None:
    bound = dtype_transpose_plan(dtype, -1)
    source = (
        jnp.arange(3 * bound.source_size, dtype=jnp.int32)
        .reshape(
            3,
            bound.source_size,
        )
        .astype(dtype)
    )
    run = lambda value: strided_copy(
        value,
        plan=bound,
        native_required=True,
    )

    _reset_native_call_count_for_tests()
    actual = jax.jit(jax.vmap(run))(source)
    actual.block_until_ready()
    expected = jax.vmap(lambda value: execute_reference(value, bound))(source)

    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    assert _native_call_count_for_tests() == 1


@pytest.mark.parametrize(
    "factory",
    [rank2_transpose_plan, rank4_tiled_plan, rank4_avx2_shape_plan],
)
def test_measured_specialized_kernel_shapes_match_reference(
    factory: Callable[[], StridedCopyPlan],
) -> None:
    bound = factory()
    source = jnp.linspace(-3.0, 4.0, bound.source_size, dtype=jnp.float32)
    expected = execute_reference(source, bound)

    _reset_native_call_count_for_tests()
    actual = jax.jit(
        lambda value: strided_copy(value, plan=bound, native_required=True)
    )(source)
    actual.block_until_ready()

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected))
    assert _native_call_count_for_tests() == 1


@pytest.mark.parametrize(
    ("factory", "expected", "expected_calls"),
    [
        (rank_zero_plan, np.asarray([7.5], dtype=np.float32), 1),
        (empty_plan, np.asarray([], dtype=np.float32), 0),
    ],
)
def test_rank_zero_and_empty_plans_execute_natively(
    factory: Callable[[], StridedCopyPlan],
    expected: np.ndarray,
    expected_calls: int,
) -> None:
    bound = factory()
    source = jnp.asarray([3.0], dtype=jnp.float32)[: bound.source_size]

    _reset_native_call_count_for_tests()
    actual = strided_copy(source, plan=bound, native_required=True)
    actual.block_until_ready()

    np.testing.assert_array_equal(np.asarray(actual), expected)
    assert _native_call_count_for_tests() == expected_calls


def test_leading_and_nonleading_vmap_each_emit_one_native_call() -> None:
    bound = two_record_noncompact_plan()
    leading = jnp.arange(3 * bound.source_size, dtype=jnp.float32).reshape(
        3, bound.source_size
    )
    run = lambda value: strided_copy(
        value,
        plan=bound,
        native_required=True,
    )

    leading_compiled = jax.jit(jax.vmap(run))
    _reset_native_call_count_for_tests()
    leading_actual = leading_compiled(leading)
    leading_actual.block_until_ready()
    leading_expected = jax.vmap(lambda value: execute_reference(value, bound))(leading)
    np.testing.assert_allclose(
        np.asarray(leading_actual),
        np.asarray(leading_expected),
    )
    assert _native_call_count_for_tests() == 1
    leading_hlo = str(leading_compiled.lower(leading).compiler_ir()).lower()
    assert leading_hlo.count("custom_call") == 1
    assert "stablehlo.transpose" not in leading_hlo

    nonleading = jnp.moveaxis(leading, 0, 1)
    nonleading_compiled = jax.jit(jax.vmap(run, in_axes=1, out_axes=1))
    _reset_native_call_count_for_tests()
    nonleading_actual = nonleading_compiled(nonleading)
    nonleading_actual.block_until_ready()
    nonleading_expected = jax.vmap(
        lambda value: execute_reference(value, bound),
        in_axes=1,
        out_axes=1,
    )(nonleading)
    np.testing.assert_allclose(
        np.asarray(nonleading_actual),
        np.asarray(nonleading_expected),
    )
    assert _native_call_count_for_tests() == 1
    nonleading_hlo = str(nonleading_compiled.lower(nonleading).compiler_ir()).lower()
    assert nonleading_hlo.count("custom_call") == 1
    assert nonleading_hlo.count("stablehlo.transpose") == 2


def test_large_leading_vmap_uses_one_call_and_bounded_workers() -> None:
    bound = rank2_transpose_plan(rows=256, columns=256)
    source = jnp.arange(
        4 * bound.source_size,
        dtype=jnp.float32,
    ).reshape(4, bound.source_size)
    run = lambda value: strided_copy(
        value,
        plan=bound,
        native_required=True,
    )
    compiled = jax.jit(jax.vmap(run))

    _reset_native_call_count_for_tests()
    actual = compiled(source)
    actual.block_until_ready()
    workers, available_workers = _native_worker_counts_for_tests()
    expected = jax.vmap(lambda value: execute_reference(value, bound))(source)

    np.testing.assert_array_equal(
        np.asarray(actual),
        np.asarray(expected),
    )
    assert _native_call_count_for_tests() == 1
    assert 1 <= workers <= max(available_workers, 1)
    if available_workers > 1:
        assert workers > 1


def test_native_required_cannot_silently_bypass_rank_limit() -> None:
    shape = (2,) * 9
    strides = tuple(2 ** (8 - axis) for axis in range(9))
    unsupported = build_strided_copy_plan(
        records=(StridedCopyRecord(shape, strides, 0, strides, 0, 1.0),),
        output_size=512,
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=512,
        source_dtype=jnp.float32,
        result_dtype=jnp.float32,
    )
    source = jnp.arange(unsupported.source_size, dtype=jnp.float32)

    with pytest.raises(ValueError, match="rank exceeds native limit"):
        strided_copy(source, plan=unsupported, native_required=True)
