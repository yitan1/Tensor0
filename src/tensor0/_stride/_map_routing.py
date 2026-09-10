"""Side-effect-free route features and central policy for affine maps."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ._native_descriptor import (
    NativeCallKind,
    NativeCallSpec,
    lower_plan,
)
from ._plan import AffinePlan


class RouteKind(str, Enum):
    """One execution tier selected by the native route policy."""

    NATIVE = "native"
    NO_ROUTE = "no_route"


class RouteSharding(str, Enum):
    """Static sharding class relevant to the packed storage axis."""

    UNSHARDED = "unsharded"
    REPLICATED = "replicated"
    BATCH_ONLY = "batch_only"
    PACKED_STORAGE = "packed_storage"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class FreshMapRouteFeatures:
    """Immutable `O(T * R)` facts consumed by the central route policy."""

    platform: str
    native_capable: bool
    device_count: int | None
    sharding: RouteSharding


@dataclass(frozen=True, slots=True)
class RouteDecision:
    """Central execution-tier decision plus an explanatory reason."""

    route: RouteKind
    reason: str


def build_affine_route_features(
    semantic: AffinePlan,
    *,
    source_shape: tuple[int, ...],
    platform: str,
    native_runtime_available: bool,
    execution_plan: NativeCallSpec | None = None,
    native_projection_available: bool = True,
    device_count: int | None = 1,
    sharding: RouteSharding = RouteSharding.UNSHARDED,
) -> FreshMapRouteFeatures:
    """Build route facts from semantic and optional native projections."""

    if not source_shape or source_shape[-1] != semantic.source_size:
        raise ValueError("route source shape does not match plan storage")
    if execution_plan is not None:
        if execution_plan.kind is not NativeCallKind.AFFINE_MAP:
            raise ValueError("route execution plan is not an affine map")
        if execution_plan.semantic != semantic:
            raise ValueError("route native plan does not match semantic plan")
        native_execution = execution_plan
    else:
        try:
            native_execution = lower_plan(semantic)
        except ValueError:
            native_execution = None

    return FreshMapRouteFeatures(
        platform=platform,
        native_capable=(
            native_runtime_available
            and native_projection_available
            and native_execution is not None
        ),
        device_count=device_count,
        sharding=sharding,
    )


def decide_fresh_map_route(features: FreshMapRouteFeatures) -> RouteDecision:
    """Return the central execution-tier decision."""

    def decision(route: RouteKind, reason: str) -> RouteDecision:
        return RouteDecision(route, reason)

    if features.platform != "cpu":
        return decision(RouteKind.NO_ROUTE, "native_device_executor_unavailable")

    if features.device_count in (None, 1):
        if not features.native_capable:
            return decision(RouteKind.NO_ROUTE, "native_cpu_executor_unavailable")
        return decision(RouteKind.NATIVE, "native_cpu_primary")

    if features.sharding is RouteSharding.PACKED_STORAGE:
        return decision(RouteKind.NO_ROUTE, "packed_storage_axis")
    if features.sharding is RouteSharding.UNKNOWN:
        return decision(RouteKind.NO_ROUTE, "unknown_sharding")
    if features.native_capable:
        return decision(RouteKind.NATIVE, "batch_sharded_native")
    return decision(RouteKind.NO_ROUTE, "native_cpu_executor_unavailable")


__all__ = [
    "FreshMapRouteFeatures",
    "RouteDecision",
    "RouteKind",
    "RouteSharding",
    "build_affine_route_features",
    "decide_fresh_map_route",
]
