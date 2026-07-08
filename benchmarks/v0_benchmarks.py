from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import platform
import shlex
import statistics
import time
from typing import Any

import jax
import jax.numpy as jnp

import tensor0
import tensor0.structure.layout as layout_module
import tensor0.transforms as transforms_module
from tensor0 import (
    FermionParity,
    HomSpace,
    SectorType,
    SU2Irrep,
    TensorMap,
    U1Irrep,
    get_degeneracystructure,
    get_sectorstructure,
    hom,
    permute,
    repartition,
    space,
    svd_compact,
)
from tensor0.tensor import _blocks as block_module


jax.config.update("jax_enable_x64", True)

Operation = Callable[[], object]

QUICK_WARMUP = 1
QUICK_REPEAT = 2
FULL_WARMUP = 2
FULL_REPEAT = 7


@dataclass(frozen=True)
class Scenario:
    id: str
    group: str
    description: str
    scenario_profile: str
    dtype: str
    size_label: str
    execution: str
    cache_policy: str
    factory: Callable[[], Operation]


@dataclass(frozen=True)
class BenchmarkResult:
    id: str
    group: str
    description: str
    scenario_profile: str
    dtype: str
    size_label: str
    execution: str
    cache_policy: str
    warmup: int
    repeat: int
    min_ms: float
    median_ms: float
    iqr_ms: float
    max_ms: float
    times_ms: list[float]


def _iqr(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    quartiles = statistics.quantiles(values, n=4, method="inclusive")
    return quartiles[2] - quartiles[0]


def _data_for(hom_space: HomSpace, *, dtype: str, scale: float = 0.1):
    total_dim = get_degeneracystructure(hom_space).total_dim
    real = jnp.arange(1, total_dim + 1, dtype=jnp.float64) * scale
    if dtype == "complex128":
        return real.astype(jnp.complex128) + (0.01j * real).astype(jnp.complex128)
    if dtype == "float64":
        return real
    raise ValueError(f"unsupported benchmark dtype: {dtype}")


def _block_until_ready(value: Any) -> None:
    if hasattr(value, "storage") and hasattr(value.storage, "data"):
        _block_until_ready(value.storage.data)
        return
    if hasattr(value, "block_until_ready"):
        value.block_until_ready()
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            _block_until_ready(item)
        return
    if isinstance(value, dict):
        for item in value.values():
            _block_until_ready(item)


def _clear_jax_caches() -> bool:
    clear_caches = getattr(jax, "clear_caches", None)
    if clear_caches is None:
        return False
    clear_caches()
    return True


def _clear_layout_caches() -> None:
    layout_module._clear_layout_caches_for_tests()


def _clear_braider_cache() -> None:
    transforms_module._TREE_BRAIDER_CACHE.clear()


def _clear_transposer_cache() -> None:
    transforms_module._TREE_TRANSPOSER_CACHE.clear()


def _strided_indices_builder_operation(
    *,
    sizes: tuple[int, ...],
    strides: tuple[int, ...],
    offset: int,
) -> Operation:
    index_dtype = jnp.result_type(offset)

    def run() -> object:
        return block_module._build_strided_indices(sizes, strides, offset, index_dtype)

    return run


def _layout_u1_two_factor_hom() -> HomSpace:
    a = space(U1Irrep, {0: 2, -1: 5, 1: 3})
    b = space(U1Irrep, {0: 7, -1: 13, 1: 11})
    c = space(U1Irrep, {0: 19, -1: 17, 1: 23})
    d = space(U1Irrep, {0: 31, -1: 29, 1: 37})
    return hom((a, b), (c, d))


def _layout_su2_four_half_hom() -> HomSpace:
    half = space(SU2Irrep, {1: 1})
    return hom((half, half, half, half), ())


def _layout_operation(hom_space: HomSpace, *, clear_each_run: bool) -> Operation:
    if not clear_each_run:
        _block_until_ready((get_sectorstructure(hom_space), get_degeneracystructure(hom_space)))

    def run() -> object:
        if clear_each_run:
            _clear_layout_caches()
        return (get_sectorstructure(hom_space), get_degeneracystructure(hom_space))

    return run


def _layout_u1_two_factor_cold() -> Operation:
    return _layout_operation(_layout_u1_two_factor_hom(), clear_each_run=True)


def _layout_u1_two_factor_cached() -> Operation:
    return _layout_operation(_layout_u1_two_factor_hom(), clear_each_run=False)


def _layout_su2_four_half_cold() -> Operation:
    return _layout_operation(_layout_su2_four_half_hom(), clear_each_run=True)


def _layout_su2_four_half_cached() -> Operation:
    return _layout_operation(_layout_su2_four_half_hom(), clear_each_run=False)


def _composition_tensors(
    sector_type: SectorType,
    left_dims: dict[int, int],
    mid_dims: dict[int, int],
    right_dims: dict[int, int],
    *,
    dtype: str,
) -> tuple[TensorMap, TensorMap]:
    left_space = space(sector_type, left_dims)
    mid_space = space(sector_type, mid_dims)
    right_space = space(sector_type, right_dims)
    left_hom = hom((left_space,), (mid_space,))
    right_hom = hom((mid_space,), (right_space,))
    return (
        TensorMap(left_hom, _data_for(left_hom, dtype=dtype)),
        TensorMap(right_hom, _data_for(right_hom, dtype=dtype)),
    )


def _u1_composition_tensors(*, dtype: str, size_label: str) -> tuple[TensorMap, TensorMap]:
    if size_label == "small":
        return _composition_tensors(
            U1Irrep,
            {0: 2, 1: 3},
            {0: 5, 1: 7},
            {0: 11, 1: 13},
            dtype=dtype,
        )
    if size_label == "medium":
        return _composition_tensors(
            U1Irrep,
            {0: 4, 1: 6},
            {0: 8, 1: 10},
            {0: 12, 1: 14},
            dtype=dtype,
        )
    if size_label == "large":
        return _composition_tensors(
            U1Irrep,
            {0: 8, 1: 10},
            {0: 12, 1: 14},
            {0: 16, 1: 18},
            dtype=dtype,
        )
    raise ValueError(f"unsupported U1 composition size: {size_label}")


def _composition_u1_eager(*, dtype: str = "float64", size_label: str = "small") -> Operation:
    left, right = _u1_composition_tensors(dtype=dtype, size_label=size_label)

    def run() -> object:
        return left @ right

    return run


def _composition_fermion_parity_eager() -> Operation:
    left, right = _composition_tensors(
        FermionParity,
        {0: 2, 1: 3},
        {0: 5, 1: 7},
        {0: 11, 1: 13},
        dtype="float64",
    )

    def run() -> object:
        return left @ right

    return run


def _svd_tensor(
    sector_type: SectorType,
    left_dims: dict[int, int],
    right_dims: dict[int, int],
    *,
    dtype: str,
) -> TensorMap:
    left = space(sector_type, left_dims)
    right = space(sector_type, right_dims)
    hom_space = hom((left,), (right,))
    return TensorMap(hom_space, _data_for(hom_space, dtype=dtype))


def _svd_u1_compact_eager(*, dtype: str = "float64", size_label: str = "small") -> Operation:
    if size_label == "small":
        tensor = _svd_tensor(U1Irrep, {0: 2, 1: 4}, {0: 3, 1: 2}, dtype=dtype)
    elif size_label == "medium":
        tensor = _svd_tensor(U1Irrep, {0: 4, 1: 6}, {0: 5, 1: 3}, dtype=dtype)
    else:
        raise ValueError(f"unsupported U1 SVD size: {size_label}")

    def run() -> object:
        return svd_compact(tensor)

    return run


def _svd_fermion_parity_compact_eager() -> Operation:
    tensor = _svd_tensor(FermionParity, {0: 2, 1: 3}, {0: 4, 1: 2}, dtype="float64")

    def run() -> object:
        return svd_compact(tensor)

    return run


def _u1_permute_tensor(*, dtype: str = "float64", size_label: str = "small") -> TensorMap:
    if size_label == "small":
        v = space(U1Irrep, {0: 2, 1: 1})
        w = space(U1Irrep, {0: 1, 1: 2})
        x = space(U1Irrep, {1: 1})
    elif size_label == "large":
        v = space(U1Irrep, {0: 16, 1: 16})
        w = space(U1Irrep, {0: 12, 1: 12})
        x = space(U1Irrep, {0: 8, 1: 8})
    else:
        raise ValueError(f"unsupported U1 permute size: {size_label}")
    hom_space = hom((v, w), (x,))
    return TensorMap(hom_space, _data_for(hom_space, dtype=dtype, scale=1.0))


def _transform_u1_permute(*, clear_each_run: bool, size_label: str = "small") -> Operation:
    tensor = _u1_permute_tensor(size_label=size_label)
    permutation = ((1,), (0, 2))
    if not clear_each_run:
        _block_until_ready(permute(tensor, permutation))

    def run() -> object:
        if clear_each_run:
            _clear_braider_cache()
        return permute(tensor, permutation)

    return run


def _transform_u1_permute_cold() -> Operation:
    return _transform_u1_permute(clear_each_run=True)


def _transform_u1_permute_cached() -> Operation:
    return _transform_u1_permute(clear_each_run=False)


def _transform_u1_permute_large_cold() -> Operation:
    return _transform_u1_permute(clear_each_run=True, size_label="large")


def _transform_u1_permute_large_cached() -> Operation:
    return _transform_u1_permute(clear_each_run=False, size_label="large")


def _transform_u1_repartition(*, clear_each_run: bool) -> Operation:
    v = space(U1Irrep, {0: 2})
    w = space(U1Irrep, {0: 3})
    x = space(U1Irrep, {0: 5})
    hom_space = hom((v,), (w, x))
    tensor = TensorMap(hom_space, _data_for(hom_space, dtype="float64", scale=1.0))
    if not clear_each_run:
        _block_until_ready(repartition(tensor, 2))

    def run() -> object:
        if clear_each_run:
            _clear_transposer_cache()
        return repartition(tensor, 2)

    return run


def _transform_u1_repartition_cold() -> Operation:
    return _transform_u1_repartition(clear_each_run=True)


def _transform_u1_repartition_cached() -> Operation:
    return _transform_u1_repartition(clear_each_run=False)


def _transform_su2_permute(*, clear_each_run: bool) -> Operation:
    half = space(SU2Irrep, {1: 1})
    hom_space = hom((half, half, half), (half,))
    tensor = TensorMap(hom_space, _data_for(hom_space, dtype="float64", scale=1.0))
    permutation = ((1, 2), (0, 3))
    if not clear_each_run:
        _block_until_ready(permute(tensor, permutation))

    def run() -> object:
        if clear_each_run:
            _clear_braider_cache()
        return permute(tensor, permutation)

    return run


def _transform_su2_permute_cold() -> Operation:
    return _transform_su2_permute(clear_each_run=True)


def _transform_su2_permute_cached() -> Operation:
    return _transform_su2_permute(clear_each_run=False)


def _jax_composition_jit_compile_and_run() -> Operation:
    left, right = _u1_composition_tensors(dtype="float64", size_label="small")

    def run() -> object:
        _clear_jax_caches()

        @jax.jit
        def compose(left_value: TensorMap, right_value: TensorMap) -> TensorMap:
            return left_value @ right_value

        return compose(left, right)

    return run


def _jax_composition_jit_cached_run() -> Operation:
    left, right = _u1_composition_tensors(dtype="float64", size_label="small")

    @jax.jit
    def compose(left_value: TensorMap, right_value: TensorMap) -> TensorMap:
        return left_value @ right_value

    _block_until_ready(compose(left, right))

    def run() -> object:
        return compose(left, right)

    return run


def _jax_composition_value_and_grad_cached() -> Operation:
    left, right = _u1_composition_tensors(dtype="float64", size_label="small")

    def loss(candidate: TensorMap) -> object:
        composed = candidate @ right
        return jnp.sum(composed.storage.data * composed.storage.data)

    compiled = jax.jit(jax.value_and_grad(loss))
    _block_until_ready(compiled(left))

    def run() -> object:
        return compiled(left)

    return run


def _scenario(
    id: str,
    group: str,
    description: str,
    scenario_profile: str,
    dtype: str,
    size_label: str,
    execution: str,
    cache_policy: str,
    factory: Callable[[], Operation],
) -> Scenario:
    return Scenario(
        id=id,
        group=group,
        description=description,
        scenario_profile=scenario_profile,
        dtype=dtype,
        size_label=size_label,
        execution=execution,
        cache_policy=cache_policy,
        factory=factory,
    )


def _scenarios() -> tuple[Scenario, ...]:
    quick = (
        _scenario(
            "layout.u1_two_factor.cold",
            "layout",
            "U1 two-factor sector and degeneracy structure cold construction",
            "quick",
            "none",
            "small",
            "metadata",
            "cold_layout",
            _layout_u1_two_factor_cold,
        ),
        _scenario(
            "layout.su2_four_half.cold",
            "layout",
            "SU2 four spin-half layout cold construction",
            "quick",
            "none",
            "small",
            "metadata",
            "cold_layout",
            _layout_su2_four_half_cold,
        ),
        _scenario(
            "composition.u1.eager",
            "tensor",
            "Eager U1 blockwise TensorMap composition",
            "quick",
            "float64",
            "small",
            "eager",
            "warmed_metadata",
            _composition_u1_eager,
        ),
        _scenario(
            "svd.u1_compact.eager",
            "factorization",
            "Eager compact SVD on a rectangular U1 TensorMap",
            "quick",
            "float64",
            "small",
            "eager",
            "warmed_metadata",
            _svd_u1_compact_eager,
        ),
        _scenario(
            "transform.u1_permute.cold",
            "transform",
            "U1 permute with cold tree braider construction",
            "quick",
            "float64",
            "small",
            "eager",
            "cold_transform",
            _transform_u1_permute_cold,
        ),
        _scenario(
            "transform.u1_repartition.cold",
            "transform",
            "U1 repartition with cold tree transposer construction",
            "quick",
            "float64",
            "small",
            "eager",
            "cold_transform",
            _transform_u1_repartition_cold,
        ),
        _scenario(
            "transform.su2_permute.cold",
            "transform",
            "SU2 permute with cold generic tree transformer construction",
            "quick",
            "float64",
            "small",
            "eager",
            "cold_transform",
            _transform_su2_permute_cold,
        ),
        _scenario(
            "jax.composition.jit_compile_and_run",
            "jax",
            "JAX jit compile plus first run for U1 composition",
            "quick",
            "float64",
            "small",
            "jit_compile_and_run",
            "clear_jax_caches",
            _jax_composition_jit_compile_and_run,
        ),
        _scenario(
            "jax.composition.jit_cached_run",
            "jax",
            "JAX jit cached run for U1 composition",
            "quick",
            "float64",
            "small",
            "jit_cached_run",
            "compiled_once",
            _jax_composition_jit_cached_run,
        ),
        _scenario(
            "jax.composition.value_and_grad_cached",
            "jax",
            "JAX value_and_grad cached run through U1 composition loss",
            "quick",
            "float64",
            "small",
            "value_and_grad_cached",
            "compiled_once",
            _jax_composition_value_and_grad_cached,
        ),
    )
    full_only = (
        _scenario(
            "layout.u1_two_factor.cached",
            "layout",
            "Cached U1 two-factor sector and degeneracy structure lookup",
            "full-only",
            "none",
            "small",
            "metadata",
            "cached_layout",
            _layout_u1_two_factor_cached,
        ),
        _scenario(
            "layout.su2_four_half.cached",
            "layout",
            "Cached SU2 four spin-half layout lookup",
            "full-only",
            "none",
            "small",
            "metadata",
            "cached_layout",
            _layout_su2_four_half_cached,
        ),
        _scenario(
            "internal.strided_indices.rank2_noncontiguous",
            "internal",
            "Private rank-2 non-contiguous strided index construction cache miss path",
            "explicit-only",
            "int64",
            "large",
            "eager",
            "cache_miss_builder",
            lambda: _strided_indices_builder_operation(
                sizes=(192, 128),
                strides=(1, 256),
                offset=7,
            ),
        ),
        _scenario(
            "internal.strided_indices.rank3_noncontiguous",
            "internal",
            "Private rank-3 non-contiguous strided index construction cache miss path",
            "explicit-only",
            "int64",
            "large",
            "eager",
            "cache_miss_builder",
            lambda: _strided_indices_builder_operation(
                sizes=(32, 16, 8),
                strides=(1, 256, 16),
                offset=5,
            ),
        ),
        _scenario(
            "composition.u1.eager.float64.medium",
            "tensor",
            "Medium real U1 blockwise TensorMap composition",
            "full-only",
            "float64",
            "medium",
            "eager",
            "warmed_metadata",
            lambda: _composition_u1_eager(dtype="float64", size_label="medium"),
        ),
        _scenario(
            "composition.u1.eager.float64.large",
            "tensor",
            "Large real U1 blockwise TensorMap composition",
            "full-only",
            "float64",
            "large",
            "eager",
            "warmed_metadata",
            lambda: _composition_u1_eager(dtype="float64", size_label="large"),
        ),
        _scenario(
            "composition.u1.eager.complex128.small",
            "tensor",
            "Small complex U1 blockwise TensorMap composition",
            "full-only",
            "complex128",
            "small",
            "eager",
            "warmed_metadata",
            lambda: _composition_u1_eager(dtype="complex128", size_label="small"),
        ),
        _scenario(
            "composition.u1.eager.complex128.medium",
            "tensor",
            "Medium complex U1 blockwise TensorMap composition",
            "full-only",
            "complex128",
            "medium",
            "eager",
            "warmed_metadata",
            lambda: _composition_u1_eager(dtype="complex128", size_label="medium"),
        ),
        _scenario(
            "composition.fermion_parity.eager.float64.small",
            "tensor",
            "Small real FermionParity blockwise TensorMap composition",
            "full-only",
            "float64",
            "small",
            "eager",
            "warmed_metadata",
            _composition_fermion_parity_eager,
        ),
        _scenario(
            "svd.u1_compact.eager.float64.medium",
            "factorization",
            "Medium real compact SVD on a U1 TensorMap",
            "full-only",
            "float64",
            "medium",
            "eager",
            "warmed_metadata",
            lambda: _svd_u1_compact_eager(dtype="float64", size_label="medium"),
        ),
        _scenario(
            "svd.u1_compact.eager.complex128.small",
            "factorization",
            "Small complex compact SVD on a U1 TensorMap",
            "full-only",
            "complex128",
            "small",
            "eager",
            "warmed_metadata",
            lambda: _svd_u1_compact_eager(dtype="complex128", size_label="small"),
        ),
        _scenario(
            "svd.u1_compact.eager.complex128.medium",
            "factorization",
            "Medium complex compact SVD on a U1 TensorMap",
            "full-only",
            "complex128",
            "medium",
            "eager",
            "warmed_metadata",
            lambda: _svd_u1_compact_eager(dtype="complex128", size_label="medium"),
        ),
        _scenario(
            "svd.fermion_parity_compact.eager.float64.small",
            "factorization",
            "Small real compact SVD on a FermionParity TensorMap",
            "full-only",
            "float64",
            "small",
            "eager",
            "warmed_metadata",
            _svd_fermion_parity_compact_eager,
        ),
        _scenario(
            "transform.u1_permute.cached",
            "transform",
            "U1 permute with cached tree braider construction",
            "full-only",
            "float64",
            "small",
            "eager",
            "cached_transform",
            _transform_u1_permute_cached,
        ),
        _scenario(
            "transform.u1_repartition.cached",
            "transform",
            "U1 repartition with cached tree transposer construction",
            "full-only",
            "float64",
            "small",
            "eager",
            "cached_transform",
            _transform_u1_repartition_cached,
        ),
        _scenario(
            "transform.su2_permute.cached",
            "transform",
            "SU2 permute with cached generic tree transformer construction",
            "full-only",
            "float64",
            "small",
            "eager",
            "cached_transform",
            _transform_su2_permute_cached,
        ),
        _scenario(
            "transform.u1_permute.large.cold",
            "transform",
            "Large U1 permute with cold tree braider construction",
            "full-only",
            "float64",
            "large",
            "eager",
            "cold_transform",
            _transform_u1_permute_large_cold,
        ),
        _scenario(
            "transform.u1_permute.large.cached",
            "transform",
            "Large U1 permute with cached tree braider construction",
            "full-only",
            "float64",
            "large",
            "eager",
            "cached_transform",
            _transform_u1_permute_large_cached,
        ),
    )
    return quick + full_only


def _selected_scenarios(selected_ids: Iterable[str] | None, *, quick: bool) -> tuple[Scenario, ...]:
    scenarios = _scenarios()
    if selected_ids is None:
        if quick:
            return tuple(scenario for scenario in scenarios if scenario.scenario_profile == "quick")
        return tuple(scenario for scenario in scenarios if scenario.scenario_profile != "explicit-only")

    by_id = {scenario.id: scenario for scenario in scenarios}
    selected: list[Scenario] = []
    for scenario_id in selected_ids:
        try:
            selected.append(by_id[scenario_id])
        except KeyError:
            valid = ", ".join(sorted(by_id))
            raise SystemExit(f"unknown scenario {scenario_id!r}; valid scenarios: {valid}")
    return tuple(selected)


def _measure_scenario(scenario: Scenario, *, warmup: int, repeat: int) -> BenchmarkResult:
    operation = scenario.factory()

    for _ in range(warmup):
        _block_until_ready(operation())

    times_ms: list[float] = []
    for _ in range(repeat):
        start = time.perf_counter_ns()
        _block_until_ready(operation())
        stop = time.perf_counter_ns()
        times_ms.append((stop - start) / 1_000_000.0)

    return BenchmarkResult(
        id=scenario.id,
        group=scenario.group,
        description=scenario.description,
        scenario_profile=scenario.scenario_profile,
        dtype=scenario.dtype,
        size_label=scenario.size_label,
        execution=scenario.execution,
        cache_policy=scenario.cache_policy,
        warmup=warmup,
        repeat=repeat,
        min_ms=min(times_ms),
        median_ms=statistics.median(times_ms),
        iqr_ms=_iqr(times_ms),
        max_ms=max(times_ms),
        times_ms=times_ms,
    )


def _environment() -> dict[str, str | bool]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "jax": getattr(jax, "__version__", "unknown"),
        "jax_backend": jax.default_backend(),
        "jax_clear_caches_available": getattr(jax, "clear_caches", None) is not None,
        "tensor0": getattr(tensor0, "__version__", "unknown"),
    }


def _run_payload(
    scenarios: tuple[Scenario, ...],
    *,
    profile: str,
    warmup: int,
    repeat: int,
) -> dict[str, Any]:
    return {
        "environment": _environment(),
        "config": {
            "profile": profile,
            "warmup": warmup,
            "repeat": repeat,
            "scenario_count": len(scenarios),
        },
        "results": [
            asdict(_measure_scenario(scenario, warmup=warmup, repeat=repeat))
            for scenario in scenarios
        ],
    }


def _render_markdown(payload: dict[str, Any], *, command: str) -> str:
    results = payload["results"]
    ranked = sorted(results, key=lambda result: result["median_ms"], reverse=True)
    top = ranked[:5]
    first = top[0] if top else None

    lines = [
        "# Tensor0 v0 Benchmark Baseline",
        "",
        "This report is a local planning artifact. It ranks Tensor0 v0",
        "investigation targets and does not define a public performance commitment.",
        "",
        "## Measurement Scope",
        "",
        "- Measures Tensor0 wall-clock elapsed time with standard-library timing.",
        "- Records warmup, repeated measurements, min, median, IQR, max, and raw samples.",
        "- Covers metadata/layout construction, blockwise linalg, transforms, and JAX gates.",
        "- Excludes optimization work, CI thresholds, and cross-library performance claims.",
        "",
        "## TensorKit Reference Alignment",
        "",
        "- Mirrors TensorKit benchmark discipline through explicit parameter matrices.",
        "- Treats results as local regression and profiling evidence for Tensor0.",
        "- Limits parity to current Tensor0 public APIs.",
        "",
        "## Command",
        "",
        f"`{command}`",
        "",
        "## Environment",
        "",
    ]
    for key, value in payload["environment"].items():
        lines.append(f"- {key}: `{value}`")

    lines.extend(["", "## Config", ""])
    for key, value in payload["config"].items():
        lines.append(f"- {key}: `{value}`")

    lines.extend(["", "## Results", ""])
    for result in results:
        lines.extend(
            [
                f"### `{result['id']}`",
                "",
                f"- Group: `{result['group']}`",
                f"- Description: {result['description']}",
                f"- Scenario profile: `{result['scenario_profile']}`",
                f"- Dtype: `{result['dtype']}`",
                f"- Size label: `{result['size_label']}`",
                f"- Execution: `{result['execution']}`",
                f"- Cache policy: `{result['cache_policy']}`",
                f"- Warmup: `{result['warmup']}`",
                f"- Repeat: `{result['repeat']}`",
                f"- Min ms: `{result['min_ms']:.3f}`",
                f"- Median ms: `{result['median_ms']:.3f}`",
                f"- IQR ms: `{result['iqr_ms']:.3f}`",
                f"- Max ms: `{result['max_ms']:.3f}`",
                "- Times ms: `"
                + ", ".join(f"{sample:.3f}" for sample in result["times_ms"])
                + "`",
                "",
            ],
        )

    lines.extend(["## Highest Median Scenarios", ""])
    for index, result in enumerate(top, start=1):
        lines.append(
            f"{index}. `{result['id']}`: median `{result['median_ms']:.3f}` ms",
        )

    lines.extend(["", "## Recommendation", ""])
    if first is None:
        lines.append("- No scenario was measured, so Phase 7 has no optimization evidence.")
    else:
        lines.append(
            f"- Start follow-up investigation with `{first['id']}` because it has "
            "the highest local median time in this run.",
        )
        lines.append(
            "- Treat the ranking as a planning signal only; confirm with a focused "
            "profile before implementing an optimization.",
        )

    lines.extend(
        [
            "",
            "## Deferred TensorKit Parity",
            "",
            "- Exact `Trivial` sector parity is deferred.",
            "- Broad `Z2Irrep` parity is deferred; FermionParity is only a stable analogue.",
            "- Mutating TensorKit-style operations such as `mul!`, `tsvd!`, and `permute!` are deferred.",
            "- TensorNetwork contraction scenarios such as MPO, PEPO, and MERA are deferred.",
            "- Cross-library performance comparisons are deferred.",
            "",
        ],
    )
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Tensor0 v0 benchmarks.")
    parser.add_argument("--quick", action="store_true", help="Use quick smoke timing and scenario selection.")
    parser.add_argument("--json", action="store_true", help="Write JSON payload to stdout.")
    parser.add_argument(
        "--markdown",
        type=Path,
        help="Write a Markdown report to this path.",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        help="Run one scenario id; repeat this flag to select multiple scenarios.",
    )
    parser.add_argument(
        "--list-scenarios",
        action="store_true",
        help="Print every registered scenario id and exit.",
    )
    parser.add_argument("--repeat", type=int, help="Override measured iteration count.")
    parser.add_argument("--warmup", type=int, help="Override warmup iteration count.")
    return parser.parse_args()


def _command_from_args(args: argparse.Namespace) -> str:
    parts = ["uv", "run", "python", "benchmarks/v0_benchmarks.py"]
    if args.quick:
        parts.append("--quick")
    for scenario_id in args.scenario or ():
        parts.extend(("--scenario", scenario_id))
    if args.json:
        parts.append("--json")
    if args.markdown is not None:
        parts.extend(("--markdown", str(args.markdown)))
    if args.repeat is not None:
        parts.extend(("--repeat", str(args.repeat)))
    if args.warmup is not None:
        parts.extend(("--warmup", str(args.warmup)))
    return " ".join(shlex.quote(part) for part in parts)


def main() -> None:
    args = _parse_args()

    if args.list_scenarios:
        for scenario in _scenarios():
            print(scenario.id)
        return

    profile = "quick" if args.quick else "full"
    warmup = args.warmup if args.warmup is not None else (QUICK_WARMUP if args.quick else FULL_WARMUP)
    repeat = args.repeat if args.repeat is not None else (QUICK_REPEAT if args.quick else FULL_REPEAT)
    if warmup < 0:
        raise SystemExit("--warmup must be non-negative")
    if repeat < 1:
        raise SystemExit("--repeat must be at least 1")

    scenarios = _selected_scenarios(args.scenario, quick=args.quick)
    payload = _run_payload(scenarios, profile=profile, warmup=warmup, repeat=repeat)

    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(
            _render_markdown(payload, command=_command_from_args(args)),
            encoding="utf-8",
        )
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
