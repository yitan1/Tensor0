from __future__ import annotations

import jax.numpy as jnp

from .._specs import (
    EXPLICIT,
    PreparedOperation,
    ScenarioSpec,
    WorkloadSpec,
    raw_execution,
)


def _build_strided_indices(
    sizes: tuple[int, ...],
    strides: tuple[int, ...],
    offset: int,
    index_dtype: jnp.dtype,
) -> object:
    indices = jnp.zeros(sizes, dtype=index_dtype)
    for axis, (size, stride) in enumerate(zip(sizes, strides, strict=True)):
        shape = (1,) * axis + (size,) + (1,) * (len(sizes) - axis - 1)
        indices = indices + jnp.arange(size, dtype=index_dtype).reshape(shape) * stride
    return (indices + offset).reshape(-1)


def _strided_indices_prepared(
    *,
    sizes: tuple[int, ...],
    strides: tuple[int, ...],
    offset: int,
) -> PreparedOperation:
    index_dtype = jnp.result_type(offset)

    def run() -> object:
        return _build_strided_indices(
            sizes,
            strides,
            offset,
            index_dtype,
        )

    return run, ()


_UNCACHED_BUILDER = raw_execution(
    id_suffix="",
    execution="eager",
    cache_policy="uncached_builder",
)

STRIDED_INDICES_SCENARIO_SPECS: tuple[ScenarioSpec, ...] = (
    ScenarioSpec(
        WorkloadSpec(
            "internal.strided_indices.rank2_noncontiguous",
            "diagnostics",
            (
                "Private rank-2 non-contiguous strided index construction "
                "uncached builder"
            ),
            "int64",
            "large",
            lambda: _strided_indices_prepared(
                sizes=(192, 128),
                strides=(1, 256),
                offset=7,
            ),
        ),
        _UNCACHED_BUILDER,
        EXPLICIT,
    ),
    ScenarioSpec(
        WorkloadSpec(
            "internal.strided_indices.rank3_noncontiguous",
            "diagnostics",
            (
                "Private rank-3 non-contiguous strided index construction "
                "uncached builder"
            ),
            "int64",
            "large",
            lambda: _strided_indices_prepared(
                sizes=(32, 16, 8),
                strides=(1, 256, 16),
                offset=5,
            ),
        ),
        _UNCACHED_BUILDER,
        EXPLICIT,
    ),
)
