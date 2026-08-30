"""Side-effect-free route features and central policy for affine maps."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum
from math import prod

import jax.numpy as jnp

from ._compiler import (
    CompiledStridePlan,
    CpuKernelKind,
    ScaleKind,
    compile_plan,
)
from ._native_lowering import (
    NativeExecutionKind,
    NativeExecutionPlan,
    lower_compiled_plan,
)
from ._plan import (
    CompleteMode,
    StridedCopyPlan,
    _stable_key_bytes,
)


ROUTE_POLICY_VERSION = 7
AUTO_NATIVE_MINIMUM_COPIED_BYTES = 131_072 * 4
AUTO_NATIVE_MANY_RECORD_MINIMUM_RECORDS = 1_024
RETAINED_AFFINE_NATIVE_MINIMUM_ELEMENTS = 4_096
RETAINED_COMPOSITE_NATIVE_MINIMUM_ELEMENTS = 32_768


class RouteKind(str, Enum):
    """One execution tier selected by the shadow route policy."""

    STABLEHLO = "stablehlo"
    NATIVE = "native"
    NO_ROUTE = "no_route"


class StableHloCapability(str, Enum):
    """How an already certified StableHLO recipe may be selected."""

    NONE = "none"
    COMPACT_AUTOMATIC = "compact_automatic"
    AFFINE_AUTOMATIC = "affine_automatic"
    PORTABLE_AUTOMATIC = "portable_automatic"


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

    canonical_execution_key: bytes
    capability_key: bytes
    platform: str
    source_dtype: str
    result_dtype: str
    batch_shape: tuple[int, ...]
    copied_elements: int
    source_size: int
    output_size: int
    work_bytes: int
    record_count: int
    coverage: str
    source_complete: bool
    layout_histogram: tuple[tuple[str, int], ...]
    kernel_histogram: tuple[tuple[str, int], ...]
    scale_histogram: tuple[tuple[str, int], ...]
    stablehlo_capability: StableHloCapability
    native_capable: bool
    device_count: int | None
    sharding: RouteSharding
    automatic: bool = True
    policy_version: int = ROUTE_POLICY_VERSION

    @property
    def feature_key(self) -> bytes:
        """Return a route-feature key distinct from executable/plan keys."""

        return _stable_key_bytes(
            "tensor0-stride-route-features-v7",
            self.policy_version,
            self.canonical_execution_key,
            self.capability_key,
            self.platform,
            self.source_dtype,
            self.result_dtype,
            self.batch_shape,
            self.copied_elements,
            self.source_size,
            self.output_size,
            self.work_bytes,
            self.record_count,
            self.coverage,
            self.source_complete,
            self.layout_histogram,
            self.kernel_histogram,
            self.scale_histogram,
            self.stablehlo_capability.value,
            self.native_capable,
            self.device_count,
            self.sharding.value,
            self.automatic,
        )


@dataclass(frozen=True, slots=True)
class RouteDecision:
    """Versioned central decision plus a stable explanatory reason."""

    route: RouteKind
    reason: str
    feature_key: bytes
    policy_version: int = ROUTE_POLICY_VERSION

    @property
    def key(self) -> bytes:
        return _stable_key_bytes(
            "tensor0-stride-route-decision-v5",
            self.policy_version,
            self.feature_key,
            self.route.value,
            self.reason,
        )


def _histogram(values: tuple[str, ...]) -> tuple[tuple[str, int], ...]:
    return tuple(sorted(Counter(values).items()))


def _work_bytes(
    bound: StridedCopyPlan,
    source_shape: tuple[int, ...],
) -> int:
    batch_count = prod(source_shape[:-1])
    per_batch_elements = max(
        bound.copied_elements,
        (
            bound.output_size
            if bound.coverage is CompleteMode.PARTIAL_UNIQUE_ZERO_FILL
            else 0
        ),
    )
    return batch_count * per_batch_elements * jnp.dtype(bound.result_dtype).itemsize


def compile_affine_route_features(
    semantic: StridedCopyPlan,
    *,
    source_shape: tuple[int, ...],
    platform: str,
    stablehlo_capability: StableHloCapability = StableHloCapability.NONE,
    native_runtime_available: bool,
    compiled_plan: CompiledStridePlan | None = None,
    execution_plan: NativeExecutionPlan | None = None,
    native_projection_available: bool = True,
    device_count: int | None = 1,
    sharding: RouteSharding = RouteSharding.UNSHARDED,
    automatic: bool = True,
) -> FreshMapRouteFeatures:
    """Compile route facts from semantic and optional native projections."""

    if not source_shape or source_shape[-1] != semantic.source_size:
        raise ValueError("route source shape does not match plan storage")
    compiled: CompiledStridePlan
    if execution_plan is not None:
        if execution_plan.kind is not NativeExecutionKind.AFFINE_MAP:
            raise ValueError("route execution plan is not an affine map")
        compiled = execution_plan.compiled
        if compiled.bound != semantic:
            raise ValueError("route native plan does not match semantic plan")
        native_execution = execution_plan
    elif compiled_plan is not None:
        if compiled_plan.bound != semantic:
            raise ValueError("route canonical plan does not match semantic plan")
        compiled = compiled_plan
        try:
            native_execution = lower_compiled_plan(compiled)
        except ValueError:
            native_execution = None
    else:
        compiled = compile_plan(semantic)
        try:
            native_execution = lower_compiled_plan(compiled)
        except ValueError:
            native_execution = None

    return FreshMapRouteFeatures(
        canonical_execution_key=compiled.canonical_execution_key,
        capability_key=compiled.capability_key,
        platform=platform,
        source_dtype=semantic.source_dtype,
        result_dtype=semantic.result_dtype,
        batch_shape=source_shape[:-1],
        copied_elements=semantic.copied_elements,
        source_size=semantic.source_size,
        output_size=semantic.output_size,
        work_bytes=_work_bytes(semantic, source_shape),
        record_count=len(compiled.records),
        coverage=semantic.coverage.value,
        source_complete=(
            semantic.copied_elements == semantic.source_size
            and semantic.required_source_size == semantic.source_size
        ),
        layout_histogram=_histogram(
            tuple(record.layout_kind.value for record in compiled.records)
        ),
        kernel_histogram=_histogram(
            ()
            if native_execution is None
            else tuple(
                record.kernel_kind.value for record in native_execution.records
            )
        ),
        scale_histogram=_histogram(
            tuple(record.scale_kind.value for record in compiled.records)
        ),
        stablehlo_capability=stablehlo_capability,
        native_capable=(
            native_runtime_available
            and native_projection_available
            and native_execution is not None
        ),
        device_count=device_count,
        sharding=sharding,
        automatic=automatic,
    )


def _kernel_count(features: FreshMapRouteFeatures, kind: CpuKernelKind) -> int:
    return dict(features.kernel_histogram).get(kind.value, 0)


def _prefers_small_structured_hlo(features: FreshMapRouteFeatures) -> bool:
    return (
        features.record_count < AUTO_NATIVE_MANY_RECORD_MINIMUM_RECORDS
        and features.work_bytes < AUTO_NATIVE_MINIMUM_COPIED_BYTES
    )


def _retained_native_reason(
    features: FreshMapRouteFeatures,
) -> str | None:
    """Recognize retained performance classes from generic compiler facts."""

    if (
        features.source_dtype != "float32"
        or features.result_dtype != "float32"
        or features.coverage != CompleteMode.COMPLETE_UNIQUE.value
        or not features.source_complete
        or not features.native_capable
    ):
        return None

    kernels = dict(features.kernel_histogram)
    if (
        features.copied_elements >= RETAINED_AFFINE_NATIVE_MINIMUM_ELEMENTS
        and kernels
        and set(kernels)
        <= {
            CpuKernelKind.GENERIC.value,
            CpuKernelKind.RANK2_FORWARD.value,
            CpuKernelKind.RANK2_REVERSE.value,
        }
    ):
        return "retained_affine_map_class"

    if features.copied_elements < RETAINED_COMPOSITE_NATIVE_MINIMUM_ELEMENTS:
        return None
    if kernels and set(kernels) <= {
        CpuKernelKind.GENERIC.value,
        CpuKernelKind.COMPACT.value,
    }:
        return "retained_pack_class"
    if (
        _kernel_count(features, CpuKernelKind.RANK4_TWO_PAIR) == features.record_count
        and features.record_count > 0
    ):
        return "retained_rank4_two_pair_class"
    return None


def decide_fresh_map_route(features: FreshMapRouteFeatures) -> RouteDecision:
    """Return the central execution-tier decision."""

    def decision(route: RouteKind, reason: str) -> RouteDecision:
        return RouteDecision(route, reason, features.feature_key)

    if not features.automatic:
        if features.platform != "cpu":
            return decision(RouteKind.NO_ROUTE, "forced_native_non_cpu")
        if (
            features.device_count != 1
            or features.sharding is not RouteSharding.UNSHARDED
        ):
            return decision(RouteKind.NO_ROUTE, "forced_native_sharded")
        if features.native_capable:
            return decision(RouteKind.NATIVE, "forced_native")
        return decision(RouteKind.NO_ROUTE, "forced_native_unavailable")

    if features.platform != "cpu":
        if (
            features.stablehlo_capability is not StableHloCapability.NONE
            and features.device_count in (None, 1)
        ):
            return decision(RouteKind.STABLEHLO, "portable_automatic")
        return decision(RouteKind.NO_ROUTE, "native_device_executor_unavailable")

    if features.device_count in (None, 1):
        if not features.native_capable:
            if features.stablehlo_capability is not StableHloCapability.NONE:
                return decision(RouteKind.STABLEHLO, "portable_automatic")
            return decision(RouteKind.NO_ROUTE, "native_cpu_executor_unavailable")
        retained_reason = _retained_native_reason(features)
        if (
            features.stablehlo_capability is StableHloCapability.AFFINE_AUTOMATIC
            and retained_reason is not None
        ):
            return decision(RouteKind.NATIVE, retained_reason)
        if features.stablehlo_capability in {
            StableHloCapability.COMPACT_AUTOMATIC,
            StableHloCapability.AFFINE_AUTOMATIC,
        } and (
            _prefers_small_structured_hlo(features)
            or (
                features.stablehlo_capability is StableHloCapability.COMPACT_AUTOMATIC
                and features.source_dtype == "complex64"
                and bool(features.batch_shape)
            )
        ):
            reason = (
                "compact_automatic"
                if features.stablehlo_capability
                is StableHloCapability.COMPACT_AUTOMATIC
                else "affine_automatic"
            )
            return decision(RouteKind.STABLEHLO, reason)

        if retained_reason is not None:
            return decision(RouteKind.NATIVE, retained_reason)
        return decision(RouteKind.NATIVE, "general_native")

    if features.sharding is RouteSharding.PACKED_STORAGE:
        return decision(RouteKind.NO_ROUTE, "packed_storage_axis")
    if features.sharding is RouteSharding.UNKNOWN:
        return decision(RouteKind.NO_ROUTE, "unknown_sharding")
    if features.native_capable:
        return decision(RouteKind.NATIVE, "batch_sharded_native")
    return decision(RouteKind.NO_ROUTE, "native_cpu_executor_unavailable")


__all__ = [
    "AUTO_NATIVE_MANY_RECORD_MINIMUM_RECORDS",
    "AUTO_NATIVE_MINIMUM_COPIED_BYTES",
    "FreshMapRouteFeatures",
    "RETAINED_AFFINE_NATIVE_MINIMUM_ELEMENTS",
    "RETAINED_COMPOSITE_NATIVE_MINIMUM_ELEMENTS",
    "ROUTE_POLICY_VERSION",
    "RouteDecision",
    "RouteKind",
    "RouteSharding",
    "StableHloCapability",
    "compile_affine_route_features",
    "decide_fresh_map_route",
]
