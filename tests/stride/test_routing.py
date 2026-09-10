from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from tensor0._stride._map import _execute_map
from tensor0._stride._native import native_available
from tensor0._stride._map_routing import (
    RouteKind,
    RouteSharding,
    build_affine_route_features,
    decide_fresh_map_route,
)
from tensor0._stride._testing import (
    _native_call_count_for_tests,
    _reset_native_call_count_for_tests,
)

from ._fixtures import (
    contiguous_dtype_plan,
    many_tiny_balanced_plan,
    two_record_noncompact_plan,
)
from ._oracle import execute_reference


def _decision(
    plan,
    *,
    platform: str = "cpu",
    available: bool = True,
    device_count: int | None = 1,
    sharding: RouteSharding = RouteSharding.UNSHARDED,
):
    features = build_affine_route_features(
        plan,
        source_shape=(plan.source_size,),
        platform=platform,
        native_runtime_available=available,
        device_count=device_count,
        sharding=sharding,
    )
    return decide_fresh_map_route(features)


def test_cpu_native_primary_ignores_recipe_size_and_record_count() -> None:
    plans = (
        two_record_noncompact_plan(),
        contiguous_dtype_plan(jnp.float32, 1.0, size=4_096),
        many_tiny_balanced_plan(),
    )

    for plan in plans:
        decision = _decision(plan)
        assert decision.route is RouteKind.NATIVE
        assert decision.reason == "native_cpu_primary"


def test_route_boundaries() -> None:
    plan = two_record_noncompact_plan()
    cases = (
        (
            _decision(
                plan,
                platform="tpu",
                available=False,
            ),
            RouteKind.NO_ROUTE,
            "native_device_executor_unavailable",
        ),
        (
            _decision(plan, available=False),
            RouteKind.NO_ROUTE,
            "native_cpu_executor_unavailable",
        ),
        (
            _decision(
                plan,
                device_count=2,
                sharding=RouteSharding.BATCH_ONLY,
            ),
            RouteKind.NATIVE,
            "batch_sharded_native",
        ),
        (
            _decision(
                plan,
                device_count=2,
                sharding=RouteSharding.PACKED_STORAGE,
            ),
            RouteKind.NO_ROUTE,
            "packed_storage_axis",
        ),
        (
            _decision(
                plan,
                device_count=2,
                sharding=RouteSharding.UNKNOWN,
            ),
            RouteKind.NO_ROUTE,
            "unknown_sharding",
        ),
    )

    for decision, expected_route, expected_reason in cases:
        assert decision.route is expected_route
        assert decision.reason == expected_reason


@pytest.mark.skipif(not native_available(), reason="native CPU stride unavailable")
def test_cpu_route_executes_one_native_call() -> None:
    plan = contiguous_dtype_plan(jnp.float32, 0.75, size=4_096)
    source = jnp.arange(plan.source_size, dtype=jnp.float32)
    function = jax.jit(lambda value: _execute_map(value, plan=plan))

    _reset_native_call_count_for_tests()
    actual = function(source)
    actual.block_until_ready()

    assert _native_call_count_for_tests() == 1
    assert jnp.array_equal(actual, execute_reference(source, plan))
