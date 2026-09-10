from __future__ import annotations

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

from tensor0._stride._testing import (
    _native_call_count_for_tests,
    _reset_native_call_count_for_tests,
    native_available,
)
from tensor0._stride._ops._selected_scale import build_selected_scale_plan
from tests.stride._fixtures import execute_update_scale
from tensor0._stride._plan import AffineRecord, CompleteMode, build_affine_plan

from ._fixtures import (
    contiguous_dtype_plan,
    noncompact_identity_plan,
    partial_mixed_plan,
    selected_scale_plan,
)
from ._oracle import assert_bitwise_equal, execute_selected_scale_reference


_REPO_ROOT = Path(__file__).resolve().parents[2]
_LARGE_ROUTE_SIZE = 1_100_000


@pytest.mark.parametrize(
    ("dtype", "factor"),
    (
        (jnp.float16, -1.25),
        (jnp.bfloat16, -1.25),
        (jnp.float32, -1.25),
        (jnp.complex64, 1.25 - 0.75j),
        (jnp.int32, -3),
    ),
)
@pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")
def test_native_selected_scale_covers_all_same_dtype_scalar_policies(
    dtype: DTypeLike,
    factor: float | complex,
) -> None:
    plan = build_selected_scale_plan(selected_scale_plan(dtype))
    base = jnp.arange(50, dtype=dtype) - 7
    factor_array = jnp.asarray(factor, dtype=dtype)
    execute = lambda old, value: execute_update_scale(old, value, plan=plan)

    _reset_native_call_count_for_tests()
    actual = jax.jit(execute)(base, factor_array)
    actual.block_until_ready()
    assert _native_call_count_for_tests() == 1
    np.testing.assert_array_equal(
        np.asarray(actual),
        np.asarray(execute_selected_scale_reference(base, factor_array, plan)),
    )
    hlo = str(
        jax.jit(execute).lower(base, factor_array).compiler_ir("stablehlo")
    ).lower()
    assert f"tensor0_stride_update_{base.dtype.name}_cpu_v1" in hlo


def test_selected_scale_preserves_unselected_base_and_validates_plan() -> None:
    plan = build_selected_scale_plan(selected_scale_plan())
    base = jnp.arange(50, dtype=jnp.float32) - 7

    actual = execute_update_scale(base, -2, plan=plan)
    expected = execute_selected_scale_reference(base, -2, plan)

    np.testing.assert_array_equal(actual, expected)
    unselected = np.ones(50, dtype=bool)
    unselected[2:50:3] = False
    np.testing.assert_array_equal(
        np.asarray(actual)[unselected],
        np.asarray(base)[unselected],
    )
    mismatched = partial_mixed_plan()
    same_storage_size = build_affine_plan(
        records=mismatched.records,
        output_size=mismatched.output_size,
        coverage=mismatched.coverage,
        source_size=mismatched.output_size,
        source_dtype=mismatched.source_dtype,
        result_dtype=mismatched.result_dtype,
    )
    with pytest.raises(ValueError, match="addresses must be identical"):
        build_selected_scale_plan(same_storage_size)
    with pytest.raises(ValueError, match="identity mapping"):
        build_selected_scale_plan(contiguous_dtype_plan(jnp.float32, 2, size=8))


@pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")
def test_native_selected_scale_complete_coverage_initializes_every_element() -> None:
    plan = build_selected_scale_plan(
        contiguous_dtype_plan(jnp.float32, None, size=32)
    )
    base = jnp.arange(64, dtype=jnp.float32).reshape(2, 32) - 13
    factor = jnp.asarray([2.0, -0.5], dtype=jnp.float32)

    _reset_native_call_count_for_tests()
    actual = jax.jit(lambda old, value: execute_update_scale(old, value, plan=plan))(
        base,
        factor,
    )
    actual.block_until_ready()

    assert _native_call_count_for_tests() == 1
    np.testing.assert_array_equal(
        np.asarray(actual),
        np.asarray(execute_selected_scale_reference(base, factor, plan)),
    )


@pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")
def test_selected_scale_factor_vjp_uses_one_multirecord_dot_call() -> None:
    plan = build_selected_scale_plan(
        build_affine_plan(
            records=(
                AffineRecord((3,), (2,), 0, (2,), 0),
                AffineRecord((2,), (3,), 6, (3,), 6),
            ),
            output_size=12,
            coverage=CompleteMode.PARTIAL_UNIQUE_ZERO_FILL,
            source_size=12,
            source_dtype=jnp.float32,
            result_dtype=jnp.float32,
        )
    )
    base = jnp.linspace(-2, 3, 12, dtype=jnp.float32)
    factor = jnp.asarray(1.25, dtype=jnp.float32)
    cotangent = jnp.linspace(4, -1, 12, dtype=jnp.float32)

    primal, pullback = jax.vjp(
        lambda value: execute_update_scale(base, value, plan=plan),
        factor,
    )
    primal.block_until_ready()
    _reset_native_call_count_for_tests()
    (actual,) = pullback(cotangent)
    actual.block_until_ready()

    assert _native_call_count_for_tests() == 1
    indices = jnp.asarray([0, 2, 4, 6, 9])
    expected = jnp.sum(cotangent[indices] * base[indices])
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize(
    ("dtype", "factor", "factor_tangent"),
    (
        (jnp.float16, -1.25, 0.75),
        (jnp.bfloat16, -1.25, 0.75),
        (jnp.float32, -1.25, 0.75),
        (jnp.float64, -1.25, 0.75),
        (jnp.complex64, 1.25 - 0.5j, 0.75 + 0.25j),
        (jnp.complex128, 1.25 - 0.5j, 0.75 + 0.25j),
    ),
)
@pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")
def test_selected_scale_factor_jvp_uses_one_axpby_call(
    dtype: DTypeLike,
    factor: float | complex,
    factor_tangent: float | complex,
) -> None:
    with jax.enable_x64():
        plan = build_selected_scale_plan(selected_scale_plan(dtype))
        base = jnp.arange(50, dtype=jnp.float32).astype(dtype) - 7
        factor_array = jnp.asarray(factor, dtype=dtype)
        tangent_array = jnp.asarray(factor_tangent, dtype=dtype)

        @jax.jit
        def tangent(value: jax.Array, dvalue: jax.Array) -> jax.Array:
            return jax.jvp(
                lambda coefficient: execute_update_scale(
                    base,
                    coefficient,
                    plan=plan,
                ),
                (value,),
                (dvalue,),
            )[1]

        _reset_native_call_count_for_tests()
        actual = tangent(factor_array, tangent_array)
        actual.block_until_ready()

        assert _native_call_count_for_tests() == 1
        expected = jax.jvp(
            lambda coefficient: execute_selected_scale_reference(
                base,
                coefficient,
                plan,
            ),
            (factor_array,),
            (tangent_array,),
        )[1]
        np.testing.assert_allclose(actual, expected, rtol=5e-3, atol=5e-3)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.complex64])
def test_selected_scale_finite_values_match_direct_oracle(
    dtype: DTypeLike,
) -> None:
    plan = build_selected_scale_plan(selected_scale_plan(dtype))
    if dtype == jnp.float32:
        pattern = jnp.asarray(
            [
                0.0,
                -0.0,
                1.25,
                -1.25,
                .125,
                jnp.float32(1e10),
                -jnp.float32(1e10),
                1.25,
            ],
            dtype=jnp.float32,
        )
        factors = (
            0.0,
            -0.0,
            1.0,
            -2.0,
            1.25,
            -1.25,
            .125,
            3.5,
        )
    else:
        pattern = jnp.asarray(
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
        factors = (
            0 + 0j,
            complex(-0.0, 0.0),
            1 + 0j,
            -2 + 3j,
            complex(1.25, 1.0),
            complex(.125, -2.0),
            complex(3.5, 3.5),
        )
    base = jnp.resize(pattern, (50,))
    migrated = jax.jit(lambda old, factor: execute_update_scale(old, factor, plan=plan))
    oracle = jax.jit(
        lambda old, factor: execute_selected_scale_reference(old, factor, plan)
    )

    for factor in factors:
        factor_array = jnp.asarray(factor, dtype=dtype)
        np.testing.assert_allclose(
            migrated(base, factor_array),
            oracle(base, factor_array),
        )


def test_selected_scale_ad_and_higher_order_compositions_match_oracle() -> None:
    plan = build_selected_scale_plan(selected_scale_plan())
    base = jnp.linspace(-3, 4, 50, dtype=jnp.float32)
    factor = jnp.asarray(-1.25, dtype=jnp.float32)
    base_tangent = jnp.linspace(2, -1, 50, dtype=jnp.float32)
    factor_tangent = jnp.asarray(0.75, dtype=jnp.float32)
    cotangent = jnp.linspace(-4, 3, 50, dtype=jnp.float32)
    migrated = lambda old, value: execute_update_scale(old, value, plan=plan)
    oracle = lambda old, value: execute_selected_scale_reference(old, value, plan)

    actual_primal, actual_tangent = jax.jvp(
        migrated,
        (base, factor),
        (base_tangent, factor_tangent),
    )
    expected_primal, expected_tangent = jax.jvp(
        oracle,
        (base, factor),
        (base_tangent, factor_tangent),
    )
    np.testing.assert_array_equal(actual_primal, expected_primal)
    np.testing.assert_allclose(
        actual_tangent,
        expected_tangent,
        rtol=1e-6,
        atol=1e-6,
    )
    actual_base_vjp, actual_factor_vjp = jax.vjp(migrated, base, factor)[1](
        cotangent
    )
    expected_base_vjp, expected_factor_vjp = jax.vjp(oracle, base, factor)[1](
        cotangent
    )
    np.testing.assert_array_equal(actual_base_vjp, expected_base_vjp)
    np.testing.assert_allclose(
        actual_factor_vjp,
        expected_factor_vjp,
        rtol=1e-6,
        atol=1e-6,
    )

    actual_base_transpose = jax.linear_transpose(
        lambda old: migrated(old, factor),
        jnp.zeros_like(base),
    )(cotangent)
    expected_base_transpose = jax.linear_transpose(
        lambda old: oracle(old, factor),
        jnp.zeros_like(base),
    )(cotangent)
    actual_factor_transpose = jax.linear_transpose(
        lambda value: migrated(base, value),
        jnp.zeros_like(factor),
    )(cotangent)
    expected_factor_transpose = jax.vjp(
        lambda value: oracle(base, value),
        jnp.zeros_like(factor),
    )[1](cotangent)
    np.testing.assert_array_equal(actual_base_transpose, expected_base_transpose)
    np.testing.assert_allclose(
        actual_factor_transpose,
        expected_factor_transpose,
        rtol=1e-6,
        atol=1e-6,
    )

    migrated_pullback = lambda ct: jax.vjp(lambda old: migrated(old, factor), base)[1](
        ct
    )[0]
    oracle_pullback = lambda ct: jax.vjp(lambda old: oracle(old, factor), base)[1](ct)[
        0
    ]
    actual_nested = jax.jvp(
        migrated_pullback,
        (jnp.zeros_like(cotangent),),
        (jnp.ones_like(cotangent),),
    )
    expected_nested = jax.jvp(
        oracle_pullback,
        (jnp.zeros_like(cotangent),),
        (jnp.ones_like(cotangent),),
    )
    for actual, expected in zip(actual_nested, expected_nested, strict=True):
        np.testing.assert_array_equal(actual, expected)

    migrated_factor_gradient = lambda old: jax.grad(
        lambda value: jnp.sum(migrated(old, value))
    )(factor)
    oracle_factor_gradient = lambda old: jax.grad(
        lambda value: jnp.sum(oracle(old, value))
    )(factor)
    actual_second = jax.jvp(
        migrated_factor_gradient,
        (base,),
        (base_tangent,),
    )
    expected_second = jax.jvp(
        oracle_factor_gradient,
        (base,),
        (base_tangent,),
    )
    for actual, expected in zip(actual_second, expected_second, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_selected_scale_complex_jvp_vjp_and_transposes_match_oracle() -> None:
    plan = build_selected_scale_plan(selected_scale_plan(jnp.complex64))
    base = (
        jnp.linspace(-2, 3, 50, dtype=jnp.float32)
        + 1j * jnp.linspace(4, -1, 50, dtype=jnp.float32)
    ).astype(jnp.complex64)
    factor = jnp.asarray(1.25 - 0.75j, dtype=jnp.complex64)
    base_tangent = jnp.full_like(base, -0.5 + 2j)
    factor_tangent = jnp.asarray(-2 + 0.25j, dtype=jnp.complex64)
    cotangent = (
        jnp.linspace(3, -4, 50, dtype=jnp.float32)
        + 1j * jnp.linspace(-1, 2, 50, dtype=jnp.float32)
    ).astype(jnp.complex64)
    migrated = lambda old, value: execute_update_scale(old, value, plan=plan)
    oracle = lambda old, value: execute_selected_scale_reference(old, value, plan)

    actual_jvp = jax.jvp(
        migrated,
        (base, factor),
        (base_tangent, factor_tangent),
    )
    expected_jvp = jax.jvp(
        oracle,
        (base, factor),
        (base_tangent, factor_tangent),
    )
    actual_vjp = jax.vjp(migrated, base, factor)[1](cotangent)
    expected_vjp = jax.vjp(oracle, base, factor)[1](cotangent)
    actual_base_transpose = jax.linear_transpose(
        lambda old: migrated(old, factor),
        jnp.zeros_like(base),
    )(cotangent)
    expected_base_transpose = jax.linear_transpose(
        lambda old: oracle(old, factor),
        jnp.zeros_like(base),
    )(cotangent)
    actual_factor_transpose = jax.linear_transpose(
        lambda value: migrated(base, value),
        jnp.zeros_like(factor),
    )(cotangent)
    expected_factor_transpose = jax.vjp(
        lambda value: oracle(base, value),
        jnp.zeros_like(factor),
    )[1](cotangent)

    for actual, expected in (
        *zip(actual_jvp, expected_jvp, strict=True),
        (actual_vjp[0], expected_vjp[0]),
        (actual_base_transpose[0], expected_base_transpose[0]),
    ):
        assert_bitwise_equal(actual, expected)
    np.testing.assert_allclose(actual_vjp[1], expected_vjp[1], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        actual_factor_transpose[0],
        expected_factor_transpose[0],
        rtol=1e-6,
        atol=1e-6,
    )


def test_selected_scale_supports_batched_factors_and_vmap_axes() -> None:
    plan = build_selected_scale_plan(selected_scale_plan())
    base = jnp.linspace(-3, 4, 50, dtype=jnp.float32)
    factors = jnp.asarray([2.0, -0.5, 3.0], dtype=jnp.float32)
    bases = jnp.stack((base, 2 * base, -base))
    migrated = lambda old, value: execute_update_scale(old, value, plan=plan)
    oracle = lambda old, value: execute_selected_scale_reference(old, value, plan)

    np.testing.assert_array_equal(
        jax.jit(migrated)(bases, factors),
        jax.jit(oracle)(bases, factors),
    )
    for in_axes in ((0, 0), (0, None), (None, 0)):
        actual = jax.jit(jax.vmap(migrated, in_axes=in_axes))(
            bases if in_axes[0] == 0 else base,
            factors if in_axes[1] == 0 else factors[0],
        )
        expected = jax.jit(jax.vmap(oracle, in_axes=in_axes))(
            bases if in_axes[0] == 0 else base,
            factors if in_axes[1] == 0 else factors[0],
        )
        np.testing.assert_array_equal(actual, expected)


def test_selected_scale_dynamic_operand_hlo_and_allocation_contract() -> None:
    plan = build_selected_scale_plan(selected_scale_plan())
    base = jnp.arange(50, dtype=jnp.float32)
    function = jax.jit(lambda old, factor: execute_update_scale(old, factor, plan=plan))
    lowered = function.lower(base, jnp.asarray(2, dtype=jnp.float32))
    hlo = str(lowered.compiler_ir("stablehlo")).lower()
    executable = lowered.compile()
    memory = executable.memory_analysis()

    first = executable(base, jnp.asarray(2, dtype=jnp.float32))
    second = executable(base, jnp.asarray(-3, dtype=jnp.float32))
    assert not np.array_equal(np.asarray(first), np.asarray(second))
    assert "%arg1" in hlo
    assert "tensor0_stride_update_float32_cpu_v1" in hlo
    assert memory is not None
    assert memory.alias_size_in_bytes == 0


def test_selected_scale_compact_route_uses_native_without_addresses() -> None:
    bound = contiguous_dtype_plan(
        jnp.float32,
        None,
        size=_LARGE_ROUTE_SIZE,
    )
    plan = build_selected_scale_plan(bound)
    base = jax.ShapeDtypeStruct((bound.output_size,), jnp.float32)
    factor = jax.ShapeDtypeStruct((), jnp.float32)
    function = jax.jit(lambda old, value: execute_update_scale(old, value, plan=plan))
    hlo = str(function.lower(base, factor).compiler_ir("stablehlo"))

    assert "tensor0_stride_update_float32_cpu_v1" in hlo
    assert "stablehlo.custom_call" in hlo
    assert "stablehlo.gather" not in hlo
    assert "stablehlo.scatter" not in hlo


def test_selected_scale_large_noncompact_route_uses_native_without_addresses() -> None:
    bound = noncompact_identity_plan(_LARGE_ROUTE_SIZE)
    plan = build_selected_scale_plan(bound)
    base = jax.ShapeDtypeStruct((bound.output_size,), jnp.float32)
    factor = jax.ShapeDtypeStruct((), jnp.float32)
    function = jax.jit(lambda old, value: execute_update_scale(old, value, plan=plan))

    hlo = str(function.lower(base, factor).compiler_ir("stablehlo")).lower()
    assert "tensor0_stride_update_float32_cpu_v1" in hlo
    assert "gather" not in hlo
    assert "scatter" not in hlo


def test_selected_scale_non_cpu_lowering_fails_closed() -> None:
    bound = selected_scale_plan()
    plan = build_selected_scale_plan(bound)
    base = jax.ShapeDtypeStruct((bound.output_size,), jnp.float32)
    factor = jax.ShapeDtypeStruct((), jnp.float32)
    traced = jax.jit(
        lambda old, value: execute_update_scale(old, value, plan=plan)
    ).trace(base, factor)

    with pytest.raises(RuntimeError, match="native_update_executor_non_cpu"):
        traced.lower(lowering_platforms=("tpu",))


def test_selected_scale_multi_device_batch_sharding_and_storage_rejection() -> None:
    script = textwrap.dedent(
        """
        import json

        import jax
        import jax.numpy as jnp
        import numpy as np
        from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

        from tensor0._stride._ops._selected_scale import build_selected_scale_plan
        from tests.stride._fixtures import execute_update_scale
        from tests.stride._fixtures import selected_scale_plan
        from tests.stride._oracle import execute_selected_scale_reference

        plan = build_selected_scale_plan(selected_scale_plan())
        mesh = Mesh(np.asarray(jax.devices()), ("device",))
        base_sharding = NamedSharding(mesh, P("device", None))
        factor_sharding = NamedSharding(mesh, P("device"))
        base_host = np.arange(100, dtype=np.float32).reshape(2, 50)
        factor_host = np.asarray([2.0, -3.0], dtype=np.float32)
        base = jax.device_put(base_host, base_sharding)
        factor = jax.device_put(factor_host, factor_sharding)
        executable = jax.jit(
            lambda old, value: execute_update_scale(old, value, plan=plan),
            in_shardings=(base_sharding, factor_sharding),
            out_shardings=base_sharding,
        ).lower(base, factor).compile()
        hlo = executable.as_text().lower()
        actual = executable(base, factor)
        actual.block_until_ready()
        expected = execute_selected_scale_reference(
            jnp.asarray(base_host),
            jnp.asarray(factor_host),
            plan,
        )
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))

        for coefficient in (factor, jax.device_put(
            np.float32(2), NamedSharding(mesh, P()),
        )):
            derivative = jax.jit(
                lambda old, value: jax.jvp(
                    lambda data: execute_update_scale(data, value, plan=plan),
                    (old,), (jnp.ones_like(old),),
                )[1],
                in_shardings=(base_sharding, coefficient.sharding),
                out_shardings=base_sharding,
            ).lower(base, coefficient).compile()
            differential_hlo = derivative.as_text().lower()
            assert "tensor0_stride_scale_tangent" in differential_hlo
            assert "all-gather" not in differential_hlo
            tangent = derivative(base, coefficient)
            expected_tangent = execute_update_scale(
                jnp.ones_like(base_host), jnp.asarray(coefficient), plan=plan,
            )
            np.testing.assert_array_equal(tangent, expected_tangent)

        storage_sharding = NamedSharding(mesh, P("device"))
        storage_base = jax.device_put(base_host[0], storage_sharding)
        replicated_factor = jax.device_put(
            factor_host[0],
            NamedSharding(mesh, P()),
        )
        try:
            jax.jit(
                lambda old, value: execute_update_scale(old, value, plan=plan),
                in_shardings=(storage_sharding, NamedSharding(mesh, P())),
                out_shardings=storage_sharding,
            ).lower(storage_base, replicated_factor).compile()
        except Exception as error:
            storage_error = "cannot shard the packed storage axis" in str(error)
        else:
            storage_error = False

        print(json.dumps({
            "devices": len(jax.devices()),
            "shape": list(actual.shape),
            "all_gather": hlo.count("all-gather"),
            "all_reduce": hlo.count("all-reduce"),
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
        "storage_error": True,
    }
