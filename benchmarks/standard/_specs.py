"""Declarative benchmark workload, execution, and profile specifications."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import jax

from ._runner import (
    Operation,
    Scenario,
    block_until_ready,
)


PreparedOperation = tuple[Callable[..., object], tuple[object, ...]]
PreparedFactory = Callable[[], PreparedOperation]
ExecutionMode = Literal[
    "eager",
    "cold",
    "raw",
    "jit_compile",
    "jit_cached",
    "gradient_cached",
]
ScenarioProfile = Literal["quick", "full-only", "explicit-only"]

QUICK: ScenarioProfile = "quick"
FULL: ScenarioProfile = "full-only"
EXPLICIT: ScenarioProfile = "explicit-only"


@dataclass(frozen=True)
class WorkloadSpec:
    """Describe one operation/topology independently of how it executes."""

    base_id: str
    group: str
    description: str
    dtype: str
    size_label: str
    factory: PreparedFactory
    gradient_factory: PreparedFactory | None = None
    cache_clear: Callable[[], None] | None = None


@dataclass(frozen=True)
class ExecutionSpec:
    """Describe execution and cache behavior independently of a workload."""

    id_suffix: str
    execution: str
    cache_policy: str
    mode: ExecutionMode
    before_each: Callable[[], None] | None = None
    use_workload_cache_clear: bool = False


@dataclass(frozen=True)
class ScenarioSpec:
    """Bind a workload, execution mode, and selection profile."""

    workload: WorkloadSpec
    execution: ExecutionSpec
    profile: ScenarioProfile


def _clear_jax_caches() -> None:
    jax.clear_caches()


EAGER = ExecutionSpec(
    "eager",
    "eager",
    "warmed_metadata",
    "eager",
)
JIT_COMPILE = ExecutionSpec(
    "jit_compile_and_run",
    "jit_compile_and_run",
    "clear_jax_caches",
    "jit_compile",
    before_each=_clear_jax_caches,
)
JIT_CACHED = ExecutionSpec(
    "jit_cached_run",
    "jit_cached_run",
    "compiled_once",
    "jit_cached",
)
GRADIENT_CACHED = ExecutionSpec(
    "value_and_grad_cached",
    "value_and_grad_cached",
    "compiled_once",
    "gradient_cached",
)


def eager_execution(
    *,
    id_suffix: str = "eager",
    execution: str = "eager",
    cache_policy: str = "warmed_metadata",
) -> ExecutionSpec:
    return ExecutionSpec(
        id_suffix,
        execution,
        cache_policy,
        "eager",
    )


def cold_execution(
    *,
    id_suffix: str = "cold",
    execution: str = "eager",
    cache_policy: str,
) -> ExecutionSpec:
    return ExecutionSpec(
        id_suffix,
        execution,
        cache_policy,
        "cold",
        use_workload_cache_clear=True,
    )


def raw_execution(
    *,
    id_suffix: str = "",
    execution: str = "eager",
    cache_policy: str,
) -> ExecutionSpec:
    return ExecutionSpec(
        id_suffix,
        execution,
        cache_policy,
        "raw",
    )


def prepared_from_operation(
    factory: Callable[[], Operation],
) -> PreparedFactory:
    def prepare() -> PreparedOperation:
        return factory(), ()

    return prepare


def _eager_operation(workload: WorkloadSpec, *, warm: bool) -> Operation:
    function, arguments = workload.factory()
    if warm:
        block_until_ready(function(*arguments))

    def run() -> object:
        return function(*arguments)

    return run


def _jit_compile_operation(workload: WorkloadSpec) -> Operation:
    function, arguments = workload.factory()
    block_until_ready(function(*arguments))
    compiled = jax.jit(function)

    def run() -> object:
        return compiled(*arguments)

    return run


def _jit_cached_operation(workload: WorkloadSpec) -> Operation:
    function, arguments = workload.factory()
    compiled = jax.jit(function)
    block_until_ready(compiled(*arguments))

    def run() -> object:
        return compiled(*arguments)

    return run


def _gradient_cached_operation(workload: WorkloadSpec) -> Operation:
    if workload.gradient_factory is None:
        raise ValueError(
            f"workload {workload.base_id!r} has no gradient factory"
        )
    function, arguments = workload.gradient_factory()
    compiled = jax.jit(jax.value_and_grad(function))
    block_until_ready(compiled(*arguments))

    def run() -> object:
        return compiled(*arguments)

    return run


def _operation_factory(spec: ScenarioSpec) -> Callable[[], Operation]:
    workload = spec.workload
    mode = spec.execution.mode
    if mode == "eager":
        return lambda: _eager_operation(workload, warm=True)
    if mode in {"cold", "raw"}:
        return lambda: _eager_operation(workload, warm=False)
    if mode == "jit_compile":
        return lambda: _jit_compile_operation(workload)
    if mode == "jit_cached":
        return lambda: _jit_cached_operation(workload)
    if mode == "gradient_cached":
        return lambda: _gradient_cached_operation(workload)
    raise AssertionError(f"unsupported execution mode: {mode}")


def _before_each(spec: ScenarioSpec) -> Callable[[], None] | None:
    execution = spec.execution
    if execution.use_workload_cache_clear:
        if spec.workload.cache_clear is None:
            raise ValueError(
                f"workload {spec.workload.base_id!r} has no cache-clear hook"
            )
        return spec.workload.cache_clear
    return execution.before_each


def materialize_scenario(spec: ScenarioSpec) -> Scenario:
    suffix = spec.execution.id_suffix
    scenario_id = (
        f"{spec.workload.base_id}.{suffix}"
        if suffix
        else spec.workload.base_id
    )
    return Scenario(
        id=scenario_id,
        group=spec.workload.group,
        description=spec.workload.description,
        scenario_profile=spec.profile,
        dtype=spec.workload.dtype,
        size_label=spec.workload.size_label,
        execution=spec.execution.execution,
        cache_policy=spec.execution.cache_policy,
        factory=_operation_factory(spec),
        before_each=_before_each(spec),
    )


def materialize_scenarios(
    specs: tuple[ScenarioSpec, ...],
) -> tuple[Scenario, ...]:
    return tuple(materialize_scenario(spec) for spec in specs)
