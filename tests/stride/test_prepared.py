from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import textwrap

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensor0 import _native
from tensor0._stride._plan import CompleteMode, AffineRecord
from tensor0._stride._map import _execute_map
import tensor0._stride._native as native_module
from tensor0._stride._testing import (
    _native_worker_counts_for_tests,
    _set_native_worker_limit_for_tests,
    native_available,
)
from tensor0._stride._native_descriptor import lower_plan
from tensor0._stride._plan import (
    build_affine_plan,
)

from ._fixtures import (
    dtype_transpose_plan,
    many_tiny_balanced_plan,
    rank2_transpose_plan,
    rank4_two_pair_plan,
    selected_scale_plan,
    two_record_noncompact_plan,
)
from ._oracle import execute_reference


pytestmark = pytest.mark.skipif(
    not native_available(),
    reason="the Tensor0 extension was built without JAX FFI headers",
)

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _affine_target(dtype_name: str = "float32") -> str:
    return native_module._ensure_affine_registered(dtype_name)


def _raw_prepared_call(
    source: jax.Array,
    *,
    descriptor: bytes,
    output_shape: tuple[int, ...],
    alias_source_result: bool = False,
    dtype_name: str = "float32",
) -> jax.Array:
    call = jax.ffi.ffi_call(
        _affine_target(dtype_name),
        jax.ShapeDtypeStruct(output_shape, jnp.float32),
        input_layouts=(tuple(reversed(range(source.ndim))),),
        output_layouts=tuple(reversed(range(len(output_shape)))),
        input_output_aliases={0: 0} if alias_source_result else None,
        vmap_method="expand_dims",
        custom_call_api_version=4,
    )
    attribute = np.frombuffer(descriptor, dtype=np.uint8).copy()
    return call(source, descriptor=attribute)


def test_prepared_operation_target_mismatch_is_rejected() -> None:
    plan = selected_scale_plan()
    source = jnp.arange(plan.source_size, dtype=jnp.float32)
    native_module._ensure_affine_registered("float32")
    registration = _native._stride_prepared_registration()
    target = "tensor0_stride_test_prepared_operation_mismatch"
    jax.ffi.register_ffi_target(
        target,
        {
            "instantiate": registration["instantiate_selected_scale"],
            "execute": registration["execute_f32"],
        },
        platform="cpu",
        api_version=1,
    )
    call = jax.ffi.ffi_call(
        target,
        jax.ShapeDtypeStruct((plan.output_size,), jnp.float32),
        input_layouts=((0,),),
        output_layouts=(0,),
        vmap_method="expand_dims",
        custom_call_api_version=4,
    )
    descriptor = np.frombuffer(
        lower_plan(plan).descriptor,
        dtype=np.uint8,
    ).copy()
    with pytest.raises(Exception, match="prepared operation does not match"):
        call(source, descriptor=descriptor).block_until_ready()


@pytest.mark.parametrize("scale", [1.0, -0.75])
def test_rank4_two_pair_prepared_batch_matches_reference(scale: float) -> None:
    plan = rank4_two_pair_plan(scale=scale)
    source = jnp.arange(3 * plan.source_size, dtype=jnp.float32).reshape(
        3, plan.source_size
    )

    actual = jax.jit(
        lambda value: _execute_map(value, plan=plan)
    )(source)
    expected = jax.vmap(lambda value: execute_reference(value, plan))(source)

    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def test_rank4_two_pair_prepared_parallel_range_matches_reference() -> None:
    plan = rank4_two_pair_plan((32, 16, 16, 16), scale=-0.75)
    source = jnp.arange(plan.source_size, dtype=jnp.float32)
    _set_native_worker_limit_for_tests(4)
    try:
        actual = jax.jit(
            lambda value: _execute_map(value, plan=plan)
        )(source)
        workers, available = _native_worker_counts_for_tests()
    finally:
        _set_native_worker_limit_for_tests(None)

    np.testing.assert_array_equal(
        np.asarray(actual), np.asarray(execute_reference(source, plan))
    )
    if available >= 2:
        assert workers >= 2


@pytest.mark.parametrize(
    ("field", "word_index"),
    [
        ("scalar_policy", 10),
        ("shape", 19),
        ("source_stride", 21),
        ("destination_stride", 23),
    ],
)
def test_affine_v7_instantiate_revalidates_semantic_descriptor(
    field: str,
    word_index: int,
) -> None:
    plan = rank2_transpose_plan()
    lowered = lower_plan(plan)
    words = list(lowered.words)
    words[word_index] ^= 1
    descriptor = b"".join(word.to_bytes(8, "little") for word in words)
    source = jnp.arange(plan.source_size, dtype=jnp.float32)

    with pytest.raises(Exception):
        _raw_prepared_call(
            source,
            descriptor=descriptor,
            output_shape=(plan.output_size,),
        ).block_until_ready()


@pytest.mark.parametrize(
    "mutation", ["version", "execution_flags", "truncated", "extended"]
)
def test_affine_v8_instantiate_rejects_malformed_descriptor(mutation: str) -> None:
    plan = two_record_noncompact_plan()
    words = list(lower_plan(plan).words)
    if mutation == "version":
        words[1] += 1
        descriptor = b"".join(word.to_bytes(8, "little") for word in words)
    elif mutation == "execution_flags":
        words[11] = 2
        descriptor = b"".join(word.to_bytes(8, "little") for word in words)
    else:
        descriptor = b"".join(word.to_bytes(8, "little") for word in words)
        descriptor = (
            descriptor[:-1] if mutation == "truncated" else descriptor + bytes(8)
        )
    source = jnp.arange(plan.source_size, dtype=jnp.float32)

    with pytest.raises(Exception):
        _raw_prepared_call(
            source,
            descriptor=descriptor,
            output_shape=(plan.output_size,),
        ).block_until_ready()


def test_affine_v8_handler_rejects_rank_two_physical_buffers() -> None:
    plan = rank2_transpose_plan(rows=2, columns=3)
    source = jnp.arange(2 * plan.source_size, dtype=jnp.float32).reshape(
        2,
        plan.source_size,
    )

    with pytest.raises(Exception, match="rank-one physical buffers"):
        _raw_prepared_call(
            source,
            descriptor=lower_plan(plan).descriptor,
            output_shape=(2, plan.output_size),
        ).block_until_ready()


def test_affine_v8_execute_rejects_descriptor_dtype_target_mismatch() -> None:
    plan = dtype_transpose_plan(jnp.float16, 0.75)
    source = jnp.arange(plan.source_size, dtype=jnp.float32)

    with pytest.raises(
        Exception,
        match="descriptor dtype does not match prepared typed target",
    ):
        _raw_prepared_call(
            source,
            descriptor=lower_plan(plan).descriptor,
            output_shape=(plan.output_size,),
        ).block_until_ready()


def test_affine_v8_execute_rejects_explicit_alias() -> None:
    plan = two_record_noncompact_plan()
    source = jnp.arange(plan.source_size, dtype=jnp.float32)

    with pytest.raises(Exception, match="source and result buffers overlap"):
        _raw_prepared_call(
            source,
            descriptor=lower_plan(plan).descriptor,
            output_shape=(plan.output_size,),
            alias_source_result=True,
        ).block_until_ready()


def test_affine_v8_instantiate_enforces_absolute_state_limit() -> None:
    record_count = 12_000
    records = tuple(
        AffineRecord((1,), (1,), index, (1,), index)
        for index in range(record_count)
    )
    plan = build_affine_plan(
        records=records,
        output_size=record_count,
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=record_count,
        source_dtype=jnp.float32,
        result_dtype=jnp.float32,
    )
    source = jnp.arange(record_count, dtype=jnp.float32)

    with pytest.raises(Exception, match="prepared state exceeds size limit"):
        _raw_prepared_call(
            source,
            descriptor=lower_plan(plan).descriptor,
            output_shape=(record_count,),
        ).block_until_ready()


def test_prepared_compiled_calls_are_concurrent_and_independent() -> None:
    plan = many_tiny_balanced_plan()
    compiled = jax.jit(
        lambda value: _execute_map(
            value,
            plan=plan,
        )
    )
    sources = [
        jnp.arange(plan.source_size, dtype=jnp.float32) + offset for offset in range(4)
    ]
    compiled(sources[0]).block_until_ready()

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(compiled, sources))

    for source, result in zip(sources, results, strict=True):
        np.testing.assert_array_equal(np.asarray(result), np.asarray(source))


@pytest.mark.parametrize(
    "destruction",
    ["compiled.clear_cache()", "jax.clear_caches()"],
)
def test_prepared_lifecycle_and_reload_in_fresh_process(
    destruction: str,
) -> None:
    script = textwrap.dedent(
        """
        import gc
        import importlib
        import json

        import jax
        import jax.numpy as jnp

        from tensor0 import _native
        import tensor0._stride._map as stride_primitive
        import tensor0._stride._native as stride_native
        from tests.stride._fixtures import two_record_noncompact_plan

        metric_names = (
            "instantiate_count",
            "execute_count",
            "live_state_count",
            "destroyed_state_count",
            "live_bytes",
            "last_state_bytes",
            "last_descriptor_bytes",
        )
        metrics = lambda: dict(
            zip(metric_names, _native._stride_prepared_metrics(), strict=True)
        )
        plan = two_record_noncompact_plan()
        source = jnp.arange(plan.source_size, dtype=jnp.float32)
        compiled = jax.jit(
            lambda value: stride_primitive._execute_map(
                value,
                plan=plan,
            )
        )
        compiled(source).block_until_ready()
        compiled(source).block_until_ready()
        before = metrics()

        stride_native = importlib.reload(stride_native)
        stride_native._ensure_affine_registered("float32")
        after_reload = metrics()

        DESTRUCTION
        gc.collect()
        after_clear = metrics()
        if not _native._stride_prepared_reset_metrics():
            raise RuntimeError("cannot reset prepared metrics with live states")
        after_reset = metrics()
        print(json.dumps({
            "before": before,
            "after_reload": after_reload,
            "after_clear": after_clear,
            "after_reset": after_reset,
        }))
        """
    ).replace("DESTRUCTION", destruction)

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)
    assert result["before"]["instantiate_count"] == 1
    assert result["before"]["execute_count"] == 2
    assert result["before"]["live_state_count"] == 1
    assert result["before"]["live_bytes"] > 0
    assert result["after_reload"] == result["before"]
    assert result["after_clear"]["live_state_count"] == 0
    assert result["after_clear"]["destroyed_state_count"] == 1
    assert result["after_clear"]["live_bytes"] == 0
    assert not any(result["after_reset"].values())


def test_prepared_process_exit_with_live_state_is_clean() -> None:
    script = textwrap.dedent(
        """
        import jax
        import jax.numpy as jnp

        import tensor0._stride._map as stride_primitive
        from tests.stride._fixtures import two_record_noncompact_plan

        plan = two_record_noncompact_plan()
        source = jnp.arange(plan.source_size, dtype=jnp.float32)
        compiled = jax.jit(
            lambda value: stride_primitive._execute_map(
                value,
                plan=plan,
            )
        )
        compiled(source).block_until_ready()
        print("prepared-state-live-at-normal-process-exit")
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == "prepared-state-live-at-normal-process-exit"


def test_mixed_affine_prepared_states_reuse_and_destroy_in_fresh_process() -> None:
    script = textwrap.dedent(
        """
        import gc
        import json

        import jax
        import jax.numpy as jnp

        from tensor0 import _native
        from tensor0._stride._map import _execute_map
        from tensor0._stride._plan import (
            build_affine_plan,
        )
        from tests.stride._fixtures import two_record_noncompact_plan

        template = two_record_noncompact_plan()
        plan = build_affine_plan(
            records=template.records,
            output_size=template.output_size,
            coverage=template.coverage,
            source_size=template.source_size,
            source_dtype=jnp.float32,
            result_dtype=jnp.complex64,
        )
        source = jnp.arange(plan.source_size, dtype=jnp.float32)
        cotangent = jnp.ones(plan.output_size, dtype=jnp.complex64) * (1 + 2j)
        apply = lambda value: _execute_map(
            value, plan=plan
        )
        compiled = jax.jit(
            lambda value, cot: (
                apply(value),
                jax.vjp(apply, value)[1](cot)[0],
            )
        )
        compiled(source, cotangent)[1].block_until_ready()
        compiled(source, cotangent)[1].block_until_ready()
        before = _native._stride_prepared_metrics()
        compiled.clear_cache()
        gc.collect()
        after = _native._stride_prepared_metrics()
        print(json.dumps({"before": before, "after": after}))
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)
    before = result["before"]
    after = result["after"]
    assert before[0] == 2
    assert before[1] == 4
    assert before[2] == 2
    assert before[4] > 0
    assert after[2] == 0
    assert after[3] == 2
    assert after[4] == 0
