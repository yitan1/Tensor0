from __future__ import annotations

from dataclasses import FrozenInstanceError

import jax.numpy as jnp
import pytest

from tensor0._stride._ffi import native_available
from tensor0._stride._routing import (
    RouteKind,
    RouteSharding,
    StableHloCapability,
    compile_affine_route_features,
    decide_fresh_map_route,
)

from ._fixtures import (
    composite_pack_plan,
    contiguous_dtype_plan,
    many_tiny_balanced_plan,
    rank2_transpose_plan,
    rank4_two_pair_plan,
    two_record_noncompact_plan,
    u1_three_record_plan,
)


def _features(
    plan,
    source_shape: tuple[int, ...],
    *,
    platform: str = "cpu",
    capability: StableHloCapability = StableHloCapability.NONE,
    available: bool | None = None,
    device_count: int | None = 1,
    sharding: RouteSharding = RouteSharding.UNSHARDED,
    automatic: bool = True,
):
    return compile_affine_route_features(
        plan,
        source_shape=source_shape,
        platform=platform,
        stablehlo_capability=capability,
        native_runtime_available=(
            native_available() if available is None else available
        ),
        device_count=device_count,
        sharding=sharding,
        automatic=automatic,
    )


@pytest.mark.skipif(not native_available(), reason="native CPU stride is unavailable")
@pytest.mark.parametrize(
    ("fixture", "expected_reason", "expected_kernels"),
    [
        (
            lambda: (u1_three_record_plan(), (4_608,)),
            "retained_affine_map_class",
            (("generic", 3),),
        ),
        (
            lambda: (rank2_transpose_plan(rows=64, columns=64), (4_096,)),
            "retained_affine_map_class",
            (("rank2_forward", 1),),
        ),
    ],
    ids=("u1", "fermion"),
)
def test_shadow_retains_large_abelian_native_route(
    fixture,
    expected_reason: str,
    expected_kernels: tuple[tuple[str, int], ...],
) -> None:
    plan, source_shape = fixture()
    features = _features(plan, source_shape)
    decision = decide_fresh_map_route(features)

    assert decision.route is RouteKind.NATIVE
    assert decision.reason == expected_reason
    assert features.kernel_histogram == expected_kernels


@pytest.mark.skipif(not native_available(), reason="native CPU stride is unavailable")
def test_shadow_classifies_composite_pack_and_unpack_independently() -> None:
    pack_plan = composite_pack_plan()
    unpack_plan = rank4_two_pair_plan((8, 8, 8, 64))
    pack = _features(pack_plan, (pack_plan.source_size,))
    unpack = _features(
        unpack_plan,
        (unpack_plan.source_size,),
    )

    assert decide_fresh_map_route(pack).route is RouteKind.NATIVE
    assert decide_fresh_map_route(pack).reason == "retained_pack_class"
    assert pack.kernel_histogram == (("compact", 1), ("generic", 1))
    assert decide_fresh_map_route(unpack).route is RouteKind.NATIVE
    assert decide_fresh_map_route(unpack).reason == "retained_rank4_two_pair_class"
    assert unpack.kernel_histogram == (("rank4_two_pair", 1),)
    assert pack.feature_key != unpack.feature_key


def test_shadow_feature_and_decision_keys_are_stable_and_separate() -> None:
    plan = two_record_noncompact_plan()
    first = _features(plan, (plan.source_size,), available=False)
    second = _features(plan, (plan.source_size,), available=False)
    batched = _features(plan, (3, plan.source_size), available=False)
    first_decision = decide_fresh_map_route(first)
    second_decision = decide_fresh_map_route(second)

    assert first == second
    assert first.feature_key == second.feature_key
    assert first_decision == second_decision
    assert first_decision.key == second_decision.key
    assert first.feature_key != first.canonical_execution_key
    assert first.feature_key != first.capability_key
    assert first_decision.key != first.feature_key
    assert batched.feature_key != first.feature_key
    with pytest.raises(FrozenInstanceError):
        first.work_bytes = 0  # type: ignore[misc]


def test_shadow_cost_capability_and_many_record_routes() -> None:
    small = two_record_noncompact_plan()
    compact = contiguous_dtype_plan(jnp.float32, 1.0, size=4_096)
    many = many_tiny_balanced_plan()

    small_decision = decide_fresh_map_route(
        _features(small, (small.source_size,), available=True)
    )
    compact_decision = decide_fresh_map_route(
        _features(
            compact,
            (compact.source_size,),
            capability=StableHloCapability.COMPACT_AUTOMATIC,
            available=True,
        )
    )
    many_decision = decide_fresh_map_route(
        _features(many, (many.source_size,), available=True)
    )

    assert small_decision.route is RouteKind.NATIVE
    assert small_decision.reason == "general_native"
    assert compact_decision.route is RouteKind.STABLEHLO
    assert compact_decision.reason == "compact_automatic"
    assert many_decision.route is RouteKind.NATIVE
    assert many_decision.reason == "general_native"


@pytest.mark.skipif(
    not native_available(),
    reason="native CPU stride is unavailable",
)
def test_affine_recipe_preserves_retained_native_priority() -> None:
    plan = rank2_transpose_plan(rows=64, columns=64)
    decision = decide_fresh_map_route(
        _features(
            plan,
            (plan.source_size,),
            capability=StableHloCapability.AFFINE_AUTOMATIC,
        )
    )

    assert decision.route is RouteKind.NATIVE
    assert decision.reason == "retained_affine_map_class"


def test_shadow_non_cpu_forced_and_sharding_boundaries() -> None:
    plan = two_record_noncompact_plan()
    shape = (plan.source_size,)

    non_cpu = decide_fresh_map_route(
        _features(
            plan,
            shape,
            platform="tpu",
            capability=StableHloCapability.PORTABLE_AUTOMATIC,
            available=False,
        )
    )
    assert non_cpu.route is RouteKind.STABLEHLO
    assert non_cpu.reason == "portable_automatic"
    assert (
        decide_fresh_map_route(
            _features(plan, shape, available=True, automatic=False)
        ).route
        is RouteKind.NATIVE
    )
    assert (
        decide_fresh_map_route(
            _features(plan, shape, available=False, automatic=False)
        ).route
        is RouteKind.NO_ROUTE
    )
    assert (
        decide_fresh_map_route(
            _features(
                plan,
                shape,
                available=True,
                device_count=2,
                sharding=RouteSharding.BATCH_ONLY,
            )
        ).route
        is RouteKind.NATIVE
    )
    packed = decide_fresh_map_route(
        _features(
            plan,
            shape,
            available=True,
            device_count=2,
            sharding=RouteSharding.PACKED_STORAGE,
        )
    )
    assert packed.route is RouteKind.NO_ROUTE
    assert packed.reason == "packed_storage_axis"


def test_non_cpu_automatic_recipe_is_selected_without_native_executor() -> None:
    plan = contiguous_dtype_plan(
        jnp.float32,
        1.0,
        size=4_096,
    )
    features = _features(
        plan,
        (plan.source_size,),
        platform="tpu",
        capability=StableHloCapability.COMPACT_AUTOMATIC,
        available=False,
    )
    decision = decide_fresh_map_route(features)

    assert decision.route is RouteKind.STABLEHLO
    assert decision.reason == "portable_automatic"
