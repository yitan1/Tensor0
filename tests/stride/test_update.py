from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensor0._stride._plan import CompleteMode, AffineRecord
from tensor0._stride._testing import (
    _native_call_count_for_tests,
    _native_worker_counts_for_tests,
    _reset_native_call_count_for_tests,
    _set_native_worker_limit_for_tests,
    native_available,
)
from tests.stride._fixtures import execute_update_accumulate, execute_update_assign
from tensor0._stride._ops._update_support import build_base_accumulate_plan, build_base_assign_plan
from tensor0._stride._plan import (
    build_affine_plan,
)

from ._fixtures import (
    contiguous_dtype_plan,
    noncompact_identity_plan,
    partial_mixed_plan,
    rank2_transpose_plan,
    two_record_noncompact_plan,
)
from ._oracle import (
    assert_bitwise_equal,
    execute_base_accumulate_reference,
    execute_base_assign_reference,
)


_REPO_ROOT = Path(__file__).resolve().parents[2]
_LARGE_ROUTE_SIZE = 1_100_000


@pytest.mark.parametrize("shape,dtype,exception", [
    ((), "float32", ValueError),
    ((3,), "float32", ValueError),
    ((2, 4), "float32", ValueError),
    ((4,), "int32", TypeError),
])
@pytest.mark.parametrize("operand", ["base", "source"])
def test_assignment_abstract_and_array_validation_enforce_same_contract(shape, dtype, exception, operand):
    from tensor0._stride._ops._update_support import (
        _base_assign_abstract_eval, _validate_operands,
    )

    plan = build_base_assign_plan(contiguous_dtype_plan(jnp.float32, 1., size=4))
    arguments = [jnp.zeros((4,), dtype=jnp.float32), jnp.zeros((4,), dtype=jnp.float32)]
    arguments[0 if operand == "base" else 1] = jnp.zeros(shape, dtype=dtype)
    with pytest.raises(exception):
        _validate_operands(arguments[0], arguments[1], plan)
    with pytest.raises(exception):
        _base_assign_abstract_eval(*(jax.typeof(value) for value in arguments), plan=plan)


def test_assignment_abstract_validation_preserves_base_aval():
    from tensor0._stride._ops._update_support import _base_assign_abstract_eval

    plan = build_base_assign_plan(contiguous_dtype_plan(jnp.float32, 1., size=4))
    base = jax.typeof(jnp.zeros((2, 4), dtype=jnp.float32))
    source = jax.typeof(jnp.ones((2, 4), dtype=jnp.float32))
    assert _base_assign_abstract_eval(base, source, plan=plan) is base


@pytest.mark.parametrize(
    ("dtype", "scale"),
    (
        (jnp.float16, -1.25),
        (jnp.bfloat16, -1.25),
        (jnp.float32, -1.25),
        (jnp.complex64, 1.25 - 0.75j),
        (jnp.int32, -3),
    ),
)
@pytest.mark.parametrize(
    ("compile_operation", "execute_operation", "reference_operation", "target"),
    (
        (
            build_base_assign_plan,
            execute_update_assign,
            execute_base_assign_reference,
            "base_assign",
        ),
        (
            build_base_accumulate_plan,
            execute_update_accumulate,
            execute_base_accumulate_reference,
            "base_accumulate",
        ),
    ),
)
@pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")
def test_native_base_updates_cover_all_same_dtype_scalar_policies(
    dtype,
    scale,
    compile_operation,
    execute_operation,
    reference_operation,
    target: str,
) -> None:
    bound = contiguous_dtype_plan(dtype, scale, size=17)
    plan = compile_operation(bound)
    source = jnp.arange(17, dtype=dtype) - 3
    base = jnp.arange(17, dtype=dtype) + 5
    execute = lambda old, value: execute_operation(old, value, plan=plan)

    _reset_native_call_count_for_tests()
    actual = jax.jit(execute)(base, source)
    actual.block_until_ready()
    assert _native_call_count_for_tests() == 1
    np.testing.assert_array_equal(
        np.asarray(actual),
        np.asarray(reference_operation(base, source, plan)),
    )
    hlo = str(jax.jit(execute).lower(base, source).compiler_ir("stablehlo")).lower()
    assert f"tensor0_stride_update_{base.dtype.name}_cpu_v1" in hlo


@pytest.mark.parametrize(
    ("source_dtype", "result_dtype"),
    (
        (jnp.float16, jnp.float32),
        (jnp.float32, jnp.complex64),
        (jnp.complex64, jnp.float32),
    ),
)
@pytest.mark.parametrize(
    ("compile_operation", "execute_operation", "reference_operation"),
    (
        (build_base_assign_plan, execute_update_assign, execute_base_assign_reference),
        (
            build_base_accumulate_plan,
            execute_update_accumulate,
            execute_base_accumulate_reference,
        ),
    ),
)
@pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")
def test_native_base_updates_cover_mixed_dtype_scalar_policies(
    source_dtype,
    result_dtype,
    compile_operation,
    execute_operation,
    reference_operation,
) -> None:
    template = partial_mixed_plan()
    bound = build_affine_plan(
        records=template.records,
        output_size=template.output_size,
        coverage=template.coverage,
        source_size=template.source_size,
        source_dtype=source_dtype,
        result_dtype=result_dtype,
    )
    plan = compile_operation(bound)
    real = jnp.linspace(-2, 3, bound.source_size, dtype=jnp.float32)
    source = jnp.asarray(real, dtype=source_dtype)
    if source_dtype == jnp.complex64:
        source = source + 1j * source[::-1]
    base_real = jnp.linspace(3, -2, bound.output_size, dtype=jnp.float32)
    base = jnp.asarray(base_real, dtype=result_dtype)
    if result_dtype == jnp.complex64:
        base = base + 1j * base[::-1]
    execute = lambda old, value: execute_operation(old, value, plan=plan)

    actual = jax.jit(execute)(base, source)
    expected = reference_operation(base, source, plan)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


@pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")
def test_native_base_accumulate_signed_specialized_ranges_are_parallel() -> None:
    rows, columns = 1024, 512
    elements = rows * columns
    record = AffineRecord(
        (rows, columns),
        (-columns, 1),
        (rows - 1) * columns,
        (1, rows),
        0,
        1.25,
    )
    bound = build_affine_plan(
        records=(record,),
        output_size=elements,
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=elements,
        source_dtype=jnp.float32,
        result_dtype=jnp.float32,
    )
    plan = build_base_accumulate_plan(bound)
    base = jnp.arange(elements, dtype=jnp.float32) * 0.01
    source = jnp.arange(elements, dtype=jnp.float32) * 0.02
    _set_native_worker_limit_for_tests(4)
    try:
        actual = jax.jit(
            lambda old, value: execute_update_accumulate(old, value, plan=plan)
        )(base, source)
        actual.block_until_ready()
        workers, available = _native_worker_counts_for_tests()
    finally:
        _set_native_worker_limit_for_tests(None)

    np.testing.assert_array_equal(
        np.asarray(actual),
        np.asarray(execute_base_accumulate_reference(base, source, plan)),
    )
    if available >= 2:
        assert workers >= 2


@pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")
def test_base_accumulate_broadcast_row_keeps_per_element_add_semantics() -> None:
    outputs = 257
    copies = 7
    bound = build_affine_plan(
        records=(
            AffineRecord(
                (outputs, copies),
                (1, 0),
                0,
                (copies, 1),
                0,
                1.25,
                source_broadcast_axes=(1,),
            ),
        ),
        output_size=outputs * copies,
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=outputs,
        source_dtype=jnp.float32,
        result_dtype=jnp.float32,
    )
    plan = build_base_accumulate_plan(bound)
    source = jnp.linspace(-2, 3, outputs, dtype=jnp.float32)
    base = jnp.linspace(1, -1, outputs * copies, dtype=jnp.float32)
    actual = jax.jit(
        lambda old, value: execute_update_accumulate(old, value, plan=plan)
    )(base, source)

    assert_bitwise_equal(
        actual,
        execute_base_accumulate_reference(base, source, plan),
    )


@pytest.mark.parametrize(
    ("compile_operation", "execute_operation", "reference_operation"),
    (
        (build_base_assign_plan, execute_update_assign, execute_base_assign_reference),
        (
            build_base_accumulate_plan,
            execute_update_accumulate,
            execute_base_accumulate_reference,
        ),
    ),
)
@pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")
def test_native_complex_base_updates_match_finite_values(
    compile_operation,
    execute_operation,
    reference_operation,
) -> None:
    plan = compile_operation(
        contiguous_dtype_plan(jnp.complex64, 1.25 - 0.75j, size=8)
    )
    base = jnp.asarray(
        [
            0 + 0j,
            complex(-0.0, 0.0),
            complex(1.25, 2.0),
            complex(-1.25, -3.0),
            complex(.125, 1.0),
            complex(jnp.float32(1e10), -jnp.float32(1e10)),
            1.25 - 0.75j,
            -2.5 + 4j,
        ],
        dtype=jnp.complex64,
    )
    source = base[::-1]
    actual = jax.jit(
        lambda old, value: execute_operation(old, value, plan=plan)
    )(base, source)
    expected = reference_operation(base, source, plan)

    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-6)


def test_base_assign_preserves_partial_base_and_replaces_complete_base() -> None:
    partial = build_base_assign_plan(partial_mixed_plan())
    partial_base = jnp.arange(50, dtype=jnp.float32) + 100
    partial_source = jnp.arange(32, dtype=jnp.float32)
    partial_actual = execute_update_assign(
        partial_base,
        partial_source,
        plan=partial,
    )
    partial_expected = execute_base_assign_reference(
        partial_base,
        partial_source,
        partial,
    )

    complete = build_base_assign_plan(two_record_noncompact_plan())
    complete_base = jnp.full(16, -7, dtype=jnp.float32)
    complete_source = jnp.arange(16, dtype=jnp.float32)
    complete_actual = execute_update_assign(
        complete_base,
        complete_source,
        plan=complete,
    )

    np.testing.assert_array_equal(partial_actual, partial_expected)
    untouched = np.ones(50, dtype=bool)
    untouched[2:50:3] = False
    np.testing.assert_array_equal(
        np.asarray(partial_actual)[untouched],
        np.asarray(partial_base)[untouched],
    )
    np.testing.assert_array_equal(
        complete_actual,
        execute_base_assign_reference(
            complete_base,
            complete_source,
            complete,
        ),
    )
    np.testing.assert_array_equal(
        complete_actual,
        execute_update_assign(
            jnp.full(16, 99, dtype=jnp.float32),
            complete_source,
            plan=complete,
        ),
    )


def test_base_assign_jit_jvp_vjp_and_linear_transpose_match_oracle() -> None:
    plan = build_base_assign_plan(partial_mixed_plan())
    base = jnp.linspace(-3, 4, 50, dtype=jnp.float32)
    source = jnp.linspace(-2, 2, 32, dtype=jnp.float32)
    base_tangent = jnp.linspace(1, 2, 50, dtype=jnp.float32)
    source_tangent = jnp.linspace(-1, 1, 32, dtype=jnp.float32)
    cotangent = jnp.linspace(-4, 3, 50, dtype=jnp.float32)

    migrated = lambda old, value: execute_update_assign(old, value, plan=plan)
    oracle = lambda old, value: execute_base_assign_reference(
        old,
        value,
        plan,
    )
    migrated_primal, migrated_tangent = jax.jvp(
        migrated,
        (base, source),
        (base_tangent, source_tangent),
    )
    oracle_primal, oracle_tangent = jax.jvp(
        oracle,
        (base, source),
        (base_tangent, source_tangent),
    )
    migrated_vjp = jax.vjp(migrated, base, source)[1](cotangent)
    oracle_vjp = jax.vjp(oracle, base, source)[1](cotangent)
    migrated_transpose = jax.linear_transpose(
        migrated,
        jnp.zeros_like(base),
        jnp.zeros_like(source),
    )(cotangent)
    oracle_transpose = jax.linear_transpose(
        oracle,
        jnp.zeros_like(base),
        jnp.zeros_like(source),
    )(cotangent)

    np.testing.assert_array_equal(
        jax.jit(migrated)(base, source),
        oracle_primal,
    )
    np.testing.assert_array_equal(migrated_primal, oracle_primal)
    np.testing.assert_array_equal(migrated_tangent, oracle_tangent)
    for actual, expected in zip(migrated_vjp, oracle_vjp, strict=True):
        np.testing.assert_array_equal(actual, expected)
    for actual, expected in zip(migrated_transpose, oracle_transpose, strict=True):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(
        np.asarray(migrated_vjp[0])[2:50:3],
        np.zeros(16, dtype=np.float32),
    )


def test_base_assign_mixed_dtype_transpose_matches_jax_oracle() -> None:
    template = partial_mixed_plan()
    mixed = build_affine_plan(
        records=template.records,
        output_size=template.output_size,
        coverage=template.coverage,
        source_size=template.source_size,
        source_dtype=jnp.float32,
        result_dtype=jnp.complex64,
    )
    plan = build_base_assign_plan(mixed)
    base = (
        jnp.linspace(-2, 2, 50, dtype=jnp.float32)
        + 1j * jnp.linspace(3, -1, 50, dtype=jnp.float32)
    ).astype(jnp.complex64)
    source = jnp.linspace(-1, 2, 32, dtype=jnp.float32)
    cotangent = (
        jnp.linspace(-3, 4, 50, dtype=jnp.float32)
        + 1j * jnp.linspace(2, -5, 50, dtype=jnp.float32)
    ).astype(jnp.complex64)
    migrated = lambda old, value: execute_update_assign(old, value, plan=plan)
    oracle = lambda old, value: execute_base_assign_reference(old, value, plan)

    actual = jax.vjp(migrated, base, source)[1](cotangent)
    expected = jax.vjp(oracle, base, source)[1](cotangent)

    np.testing.assert_array_equal(migrated(base, source), oracle(base, source))
    for actual_value, expected_value in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(actual_value, expected_value)


def test_base_assign_vmap_supports_joint_and_single_operand_batching() -> None:
    plan = build_base_assign_plan(partial_mixed_plan())
    base = jnp.arange(50, dtype=jnp.float32) + 100
    source = jnp.arange(32, dtype=jnp.float32)
    base_batch = jnp.stack((base, 2 * base, -base))
    source_batch = jnp.stack((source, 3 * source, -2 * source))
    migrated = lambda old, value: execute_update_assign(old, value, plan=plan)
    oracle = lambda old, value: execute_base_assign_reference(old, value, plan)

    for in_axes in ((0, 0), (0, None), (None, 0)):
        actual = jax.jit(jax.vmap(migrated, in_axes=in_axes))(
            base_batch if in_axes[0] == 0 else base,
            source_batch if in_axes[1] == 0 else source,
        )
        expected = jax.jit(jax.vmap(oracle, in_axes=in_axes))(
            base_batch if in_axes[0] == 0 else base,
            source_batch if in_axes[1] == 0 else source,
        )
        np.testing.assert_array_equal(actual, expected)


def test_plan_factory_does_not_reprove_cross_record_uniqueness() -> None:
    records = (
        AffineRecord((2,), (1,), 0, (1,), 0),
        AffineRecord((2,), (1,), 2, (1,), 0),
    )
    plan = build_affine_plan(
        records=records,
        output_size=4,
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=4,
        source_dtype=jnp.float32,
        result_dtype=jnp.float32,
    )

    assert plan.records == records


def test_base_assign_compact_route_uses_native_without_addresses() -> None:
    bound = contiguous_dtype_plan(
        jnp.float32,
        1,
        size=_LARGE_ROUTE_SIZE,
    )
    plan = build_base_assign_plan(bound)
    base = jax.ShapeDtypeStruct((bound.output_size,), jnp.float32)
    source = jax.ShapeDtypeStruct((bound.source_size,), jnp.float32)
    function = jax.jit(lambda old, value: execute_update_assign(old, value, plan=plan))
    hlo = str(function.lower(base, source).compiler_ir("stablehlo"))

    assert "tensor0_stride_update_float32_cpu_v1" in hlo
    assert "stablehlo.custom_call" in hlo
    assert "stablehlo.gather" not in hlo
    assert "stablehlo.scatter" not in hlo


def test_base_assign_large_noncompact_route_uses_native_without_addresses() -> None:
    bound = noncompact_identity_plan(_LARGE_ROUTE_SIZE)
    plan = build_base_assign_plan(bound)
    base = jax.ShapeDtypeStruct((bound.output_size,), jnp.float32)
    source = jax.ShapeDtypeStruct((bound.source_size,), jnp.float32)
    function = jax.jit(lambda old, value: execute_update_assign(old, value, plan=plan))

    hlo = str(function.lower(base, source).compiler_ir("stablehlo")).lower()
    assert "tensor0_stride_update_float32_cpu_v1" in hlo
    assert "gather" not in hlo
    assert "scatter" not in hlo


@pytest.mark.parametrize(
    ("compile_operation", "execute_operation", "reference_operation"),
    (
        (
            build_base_assign_plan,
            execute_update_assign,
            execute_base_assign_reference,
        ),
        (
            build_base_accumulate_plan,
            execute_update_accumulate,
            execute_base_accumulate_reference,
        ),
    ),
)
@pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")
def test_large_rank2_base_update_leaf_matches_reference_sequential_and_parallel(
    compile_operation,
    execute_operation,
    reference_operation,
) -> None:
    bound = rank2_transpose_plan(rows=512, columns=512)
    plan = compile_operation(bound)
    special = jnp.asarray(
        [0.0, -0.0, jnp.inf, -jnp.inf, jnp.nan, 1.5, -2.0],
        dtype=jnp.float32,
    )
    source = jnp.tile(special, (bound.source_size + special.size - 1) // special.size)[
        : bound.source_size
    ]
    base = jnp.linspace(-3.0, 2.0, bound.output_size, dtype=jnp.float32)
    expected = reference_operation(base, source, plan)
    compiled = jax.jit(
        lambda old, value: execute_operation(old, value, plan=plan)
    )

    worker_counts: list[tuple[int, int]] = []
    for worker_limit in (1, 4):
        _set_native_worker_limit_for_tests(worker_limit)
        try:
            actual = compiled(base, source)
            actual.block_until_ready()
            worker_counts.append(_native_worker_counts_for_tests())
        finally:
            _set_native_worker_limit_for_tests(None)
        assert_bitwise_equal(actual, expected)

    assert worker_counts[0][0] == 1
    parallel_workers, available_workers = worker_counts[1]
    if available_workers >= 2:
        assert parallel_workers >= 2


@pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")
def test_rank2_base_assign_leaf_preserves_jvp_and_vjp() -> None:
    bound = rank2_transpose_plan(rows=17, columns=13)
    plan = build_base_assign_plan(bound)
    source = jnp.linspace(-2.0, 3.0, bound.source_size, dtype=jnp.float32)
    base = jnp.linspace(4.0, -1.0, bound.output_size, dtype=jnp.float32)
    source_tangent = jnp.linspace(1.0, -1.0, source.size, dtype=source.dtype)
    base_tangent = jnp.linspace(-0.5, 0.5, base.size, dtype=base.dtype)
    cotangent = jnp.linspace(-3.0, 2.0, bound.output_size, dtype=jnp.float32)
    native = lambda old, value: execute_update_assign(old, value, plan=plan)
    reference = lambda old, value: execute_base_assign_reference(
        old, value, plan
    )

    actual_primal, actual_tangent = jax.jvp(
        native,
        (base, source),
        (base_tangent, source_tangent),
    )
    expected_primal, expected_tangent = jax.jvp(
        reference,
        (base, source),
        (base_tangent, source_tangent),
    )
    actual_vjp = jax.vjp(native, base, source)[1](cotangent)
    expected_vjp = jax.vjp(reference, base, source)[1](cotangent)

    assert_bitwise_equal(actual_primal, expected_primal)
    assert_bitwise_equal(actual_tangent, expected_tangent)
    assert_bitwise_equal(actual_vjp[0], expected_vjp[0])
    assert_bitwise_equal(actual_vjp[1], expected_vjp[1])


def test_base_update_non_cpu_lowering_fails_closed() -> None:
    bound = two_record_noncompact_plan()
    plan = build_base_accumulate_plan(bound)
    base = jax.ShapeDtypeStruct((bound.output_size,), jnp.float32)
    source = jax.ShapeDtypeStruct((bound.source_size,), jnp.float32)
    traced = jax.jit(
        lambda old, value: execute_update_accumulate(old, value, plan=plan)
    ).trace(base, source)

    with pytest.raises(RuntimeError, match="native_update_executor_non_cpu"):
        traced.lower(lowering_platforms=("tpu",))


def test_base_assign_has_no_donation_or_native_alias_contract() -> None:
    plan = build_base_assign_plan(partial_mixed_plan())
    base = jnp.zeros(50, dtype=jnp.float32)
    source = jnp.arange(32, dtype=jnp.float32)
    lowered = jax.jit(lambda old, value: execute_update_assign(old, value, plan=plan)).lower(
        base, source
    )
    hlo = str(lowered.compiler_ir("stablehlo")).lower()
    memory = lowered.compile().memory_analysis()

    assert "tensor0_stride_update_float32_cpu_v1" in hlo
    assert memory is not None
    assert memory.alias_size_in_bytes == 0


def test_base_assign_multi_device_batch_sharding_and_storage_rejection() -> None:
    script = textwrap.dedent(
        """
        import json

        import jax
        import jax.numpy as jnp
        import numpy as np
        from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

        from tests.stride._fixtures import execute_update_accumulate, execute_update_assign
        from tensor0._stride._ops._update_support import build_base_accumulate_plan, build_base_assign_plan
        from tests.stride._fixtures import partial_mixed_plan
        from tests.stride._oracle import execute_base_accumulate_reference

        plan = build_base_assign_plan(partial_mixed_plan())
        mesh = Mesh(np.asarray(jax.devices()), ("device",))
        batch_sharding = NamedSharding(mesh, P("device", None))
        base_host = np.arange(100, dtype=np.float32).reshape(2, 50)
        source_host = np.arange(64, dtype=np.float32).reshape(2, 32)
        base = jax.device_put(base_host, batch_sharding)
        source = jax.device_put(source_host, batch_sharding)
        executable = jax.jit(
            lambda old, value: execute_update_assign(old, value, plan=plan),
            in_shardings=(batch_sharding, batch_sharding),
            out_shardings=batch_sharding,
        ).lower(base, source).compile()
        hlo = executable.as_text().lower()
        actual = executable(base, source)
        actual.block_until_ready()

        accumulate_plan = build_base_accumulate_plan(partial_mixed_plan())
        accumulate_executable = jax.jit(
            lambda old, value: execute_update_accumulate(
                old,
                value,
                plan=accumulate_plan,
            ),
            in_shardings=(batch_sharding, batch_sharding),
            out_shardings=batch_sharding,
        ).lower(base, source).compile()
        accumulate_hlo = accumulate_executable.as_text().lower()
        accumulate_actual = accumulate_executable(base, source)
        accumulate_actual.block_until_ready()
        np.testing.assert_array_equal(
            np.asarray(accumulate_actual),
            np.asarray(
                execute_base_accumulate_reference(
                    jnp.asarray(base_host),
                    jnp.asarray(source_host),
                    accumulate_plan,
                )
            ),
        )

        storage_sharding = NamedSharding(mesh, P("device"))
        storage_base = jax.device_put(base_host[0], storage_sharding)
        replicated = NamedSharding(mesh, P())
        replicated_source = jax.device_put(source_host[0], replicated)
        try:
            jax.jit(
                lambda old, value: execute_update_assign(old, value, plan=plan),
                in_shardings=(storage_sharding, replicated),
                out_shardings=storage_sharding,
            ).lower(storage_base, replicated_source).compile()
        except Exception as error:
            storage_error = "cannot shard the packed storage axis" in str(error)
        else:
            storage_error = False

        print(json.dumps({
            "devices": len(jax.devices()),
            "shape": list(actual.shape),
            "all_gather": hlo.count("all-gather"),
            "all_reduce": hlo.count("all-reduce"),
            "accumulate_all_gather": accumulate_hlo.count("all-gather"),
            "accumulate_all_reduce": accumulate_hlo.count("all-reduce"),
            "accumulate_shape": list(accumulate_actual.shape),
            "storage_error": storage_error,
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
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr

    assert json.loads(completed.stdout) == {
        "devices": 2,
        "shape": [2, 50],
        "all_gather": 0,
        "all_reduce": 0,
        "accumulate_all_gather": 0,
        "accumulate_all_reduce": 0,
        "accumulate_shape": [2, 50],
        "storage_error": True,
    }


def test_base_accumulate_preserves_base_and_adds_each_unique_update_once() -> None:
    plan = build_base_accumulate_plan(partial_mixed_plan())
    base = jnp.arange(50, dtype=jnp.float32) + 100
    source = jnp.arange(32, dtype=jnp.float32)

    actual = execute_update_accumulate(base, source, plan=plan)
    expected = execute_base_accumulate_reference(base, source, plan)

    np.testing.assert_array_equal(actual, expected)
    untouched = np.ones(50, dtype=bool)
    untouched[2:50:3] = False
    np.testing.assert_array_equal(
        np.asarray(actual)[untouched],
        np.asarray(base)[untouched],
    )
    np.testing.assert_array_equal(
        np.asarray(actual)[2:50:3],
        np.asarray(base)[2:50:3] + 0.5 * np.asarray(source)[1:32:2],
    )


def test_base_accumulate_jvp_vjp_and_linear_transpose_match_oracle() -> None:
    template = partial_mixed_plan()
    mixed = build_affine_plan(
        records=template.records,
        output_size=template.output_size,
        coverage=template.coverage,
        source_size=template.source_size,
        source_dtype=jnp.float32,
        result_dtype=jnp.complex64,
    )
    plan = build_base_accumulate_plan(mixed)
    base = (
        jnp.linspace(-2, 2, 50, dtype=jnp.float32)
        + 1j * jnp.linspace(3, -1, 50, dtype=jnp.float32)
    ).astype(jnp.complex64)
    source = jnp.linspace(-1, 2, 32, dtype=jnp.float32)
    base_tangent = (0.5 - 0.25j) * jnp.ones_like(base)
    source_tangent = jnp.linspace(2, -1, 32, dtype=jnp.float32)
    cotangent = (
        jnp.linspace(-3, 4, 50, dtype=jnp.float32)
        + 1j * jnp.linspace(2, -5, 50, dtype=jnp.float32)
    ).astype(jnp.complex64)
    migrated = lambda old, value: execute_update_accumulate(old, value, plan=plan)
    oracle = lambda old, value: execute_base_accumulate_reference(
        old,
        value,
        plan,
    )

    actual_jvp = jax.jvp(
        migrated,
        (base, source),
        (base_tangent, source_tangent),
    )
    expected_jvp = jax.jvp(
        oracle,
        (base, source),
        (base_tangent, source_tangent),
    )
    actual_vjp = jax.vjp(migrated, base, source)[1](cotangent)
    expected_vjp = jax.vjp(oracle, base, source)[1](cotangent)
    actual_transpose = jax.linear_transpose(
        migrated,
        jnp.zeros_like(base),
        jnp.zeros_like(source),
    )(cotangent)
    expected_transpose = jax.linear_transpose(
        oracle,
        jnp.zeros_like(base),
        jnp.zeros_like(source),
    )(cotangent)

    for actual, expected in zip(actual_jvp, expected_jvp, strict=True):
        np.testing.assert_array_equal(actual, expected)
    for actual, expected in zip(actual_vjp, expected_vjp, strict=True):
        np.testing.assert_array_equal(actual, expected)
    for actual, expected in zip(actual_transpose, expected_transpose, strict=True):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual_vjp[0], cotangent)


def test_base_accumulate_jit_vmap_and_allocation_match_oracle() -> None:
    plan = build_base_accumulate_plan(two_record_noncompact_plan())
    base = jnp.arange(16, dtype=jnp.float32)
    source = jnp.linspace(-2, 3, 16, dtype=jnp.float32)
    base_batch = jnp.stack((base, 2 * base, -base))
    source_batch = jnp.stack((source, 3 * source, -2 * source))
    migrated = lambda old, value: execute_update_accumulate(old, value, plan=plan)
    oracle = lambda old, value: execute_base_accumulate_reference(
        old,
        value,
        plan,
    )
    lowered = jax.jit(migrated).lower(base, source)
    memory = lowered.compile().memory_analysis()

    np.testing.assert_array_equal(
        jax.jit(migrated)(base, source),
        oracle(base, source),
    )
    np.testing.assert_array_equal(
        jax.jit(jax.vmap(migrated))(base_batch, source_batch),
        jax.vmap(oracle)(base_batch, source_batch),
    )
    assert memory is not None
    assert memory.alias_size_in_bytes == 0
    assert "tensor0_stride_update_float32_cpu_v1" in str(
        lowered.compiler_ir("stablehlo")
    ).lower()


def test_base_accumulate_compact_route_uses_native_without_addresses() -> None:
    bound = contiguous_dtype_plan(
        jnp.float32,
        1,
        size=_LARGE_ROUTE_SIZE,
    )
    plan = build_base_accumulate_plan(bound)
    base = jax.ShapeDtypeStruct((bound.output_size,), jnp.float32)
    source = jax.ShapeDtypeStruct((bound.source_size,), jnp.float32)
    function = jax.jit(lambda old, value: execute_update_accumulate(old, value, plan=plan))
    hlo = str(function.lower(base, source).compiler_ir("stablehlo"))

    assert "tensor0_stride_update_float32_cpu_v1" in hlo
    assert "stablehlo.custom_call" in hlo
    assert "stablehlo.gather" not in hlo
    assert "stablehlo.scatter" not in hlo


def test_base_accumulate_large_noncompact_route_uses_native_without_addresses() -> (
    None
):
    bound = noncompact_identity_plan(_LARGE_ROUTE_SIZE)
    plan = build_base_accumulate_plan(bound)
    base = jax.ShapeDtypeStruct((bound.output_size,), jnp.float32)
    source = jax.ShapeDtypeStruct((bound.source_size,), jnp.float32)
    function = jax.jit(lambda old, value: execute_update_accumulate(old, value, plan=plan))

    hlo = str(function.lower(base, source).compiler_ir("stablehlo")).lower()
    assert "tensor0_stride_update_float32_cpu_v1" in hlo
    assert "gather" not in hlo
    assert "scatter" not in hlo
