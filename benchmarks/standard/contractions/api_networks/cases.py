from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from functools import partial
from pathlib import Path
import tomllib
from typing import cast

import jax.numpy as jnp

from tensor0 import TensorMap, U1Irrep, contract, hom, idx, ncon, space

from ..._inputs import packed_tensor
from ..._runner import block_until_ready as _block_until_ready
from ..._specs import (
    EAGER,
    FULL,
    GRADIENT_CACHED,
    JIT_CACHED,
    JIT_COMPILE,
    QUICK,
    PreparedOperation,
    ScenarioSpec,
    WorkloadSpec,
)
_PARAMETERS_PATH = Path(__file__).with_name("params.toml")
with _PARAMETERS_PATH.open("rb") as stream:
    _PARAMETERS = tomllib.load(stream)

def _chain_dimensions(size_label: str) -> tuple[int, int, int, int]:
    raw_chain = _PARAMETERS.get("chain")
    if not isinstance(raw_chain, dict):
        raise ValueError("API network parameters must define a chain table")
    raw_dimensions = raw_chain.get(size_label)
    if (
        not isinstance(raw_dimensions, list)
        or len(raw_dimensions) != 4
        or not all(isinstance(value, int) and value > 0 for value in raw_dimensions)
    ):
        raise ValueError(f"invalid API network chain size: {size_label}")
    values = tuple(cast(list[int], raw_dimensions))
    return values[0], values[1], values[2], values[3]


def _chain_tensors(*, size_label: str) -> tuple[TensorMap, TensorMap, TensorMap]:
    a_dim, x_dim, y_dim, b_dim = _chain_dimensions(size_label)
    a = space(U1Irrep, {0: a_dim})
    x = space(U1Irrep, {0: x_dim})
    y = space(U1Irrep, {0: y_dim})
    b = space(U1Irrep, {0: b_dim})
    return (
        packed_tensor(hom((a,), (x,)), scale=0.01),
        packed_tensor(hom((x,), (y,)), scale=0.01),
        packed_tensor(hom((y,), (b,)), scale=0.01),
    )


def _named_network_prepared(
    *,
    custom_order: bool,
    size_label: str,
) -> PreparedOperation:
    left, middle, right = _chain_tensors(size_label=size_label)
    order = ("y", "x") if custom_order else None
    if custom_order:
        default = contract(
            idx(left, "a,x"),
            idx(middle, "x,y"),
            idx(right, "y,b"),
            output=("a", "b"),
        )
        custom = contract(
            idx(left, "a,x"),
            idx(middle, "x,y"),
            idx(right, "y,b"),
            output=("a", "b"),
            order=order,
        )
        _block_until_ready((default, custom))
        if not bool(jnp.allclose(default.storage.data, custom.storage.data)):
            raise AssertionError("named default/custom benchmark networks differ")

    def run(
        left_value: TensorMap,
        middle_value: TensorMap,
        right_value: TensorMap,
    ) -> object:
        return contract(
            idx(left_value, "a,x"),
            idx(middle_value, "x,y"),
            idx(right_value, "y,b"),
            output=("a", "b"),
            order=order,
        )

    return run, (left, middle, right)


def _ncon_network_prepared(
    *,
    custom_order: bool,
    size_label: str,
) -> PreparedOperation:
    tensors = _chain_tensors(size_label=size_label)
    labels = ((-1, 1), (1, 2), (2, -2))
    order = (2, 1) if custom_order else None
    if custom_order:
        default = ncon(tensors, labels)
        custom = ncon(tensors, labels, order=order)
        _block_until_ready((default, custom))
        if not bool(jnp.allclose(default.storage.data, custom.storage.data)):
            raise AssertionError("ncon default/custom benchmark networks differ")

    def run(*values: TensorMap) -> object:
        return ncon(values, labels, order=order)

    return run, tensors


def _disconnected_network_prepared() -> PreparedOperation:
    a = space(U1Irrep, {0: 16})
    b = space(U1Irrep, {0: 8})
    left = packed_tensor(hom((a,), ()), scale=0.01)
    right = packed_tensor(hom((b,), ()), scale=0.01)

    def run(left_value: TensorMap, right_value: TensorMap) -> object:
        return contract(
            idx(left_value, "a"),
            idx(right_value, "b"),
            output=("a,b", ""),
        )

    return run, (left, right)


def _network_gradient_prepared() -> PreparedOperation:
    left, middle, right = _chain_tensors(size_label="small")

    def loss(data: object) -> object:
        candidate = TensorMap(left.space, data)
        result = contract(
            idx(candidate, "a,x"),
            idx(middle, "x,y"),
            idx(right, "y,b"),
            output=("a", "b"),
            order=("y", "x"),
        )
        return jnp.sum(result.storage.data**2)

    return loss, (left.storage.data,)


_EAGER_NO_SUFFIX = replace(EAGER, id_suffix="")
_NETWORK_JIT_COMPILE = replace(JIT_COMPILE, id_suffix="compile_and_run")
_NETWORK_JIT_CACHED = replace(JIT_CACHED, id_suffix="cached")
_NETWORK_GRADIENT = replace(GRADIENT_CACHED, id_suffix="value_and_grad")


def _workload(
    id: str,
    description: str,
    size_label: str,
    factory: Callable[[], PreparedOperation],
) -> WorkloadSpec:
    return WorkloadSpec(
        id,
        "contractions",
        description,
        "float64",
        size_label,
        factory,
    )


_JAX_NETWORK_WORKLOAD = WorkloadSpec(
    "jax.network",
    "contractions",
    "Named three-tensor custom-order network",
    "float64",
    "small",
    partial(
        _named_network_prepared,
        custom_order=True,
        size_label="small",
    ),
    gradient_factory=_network_gradient_prepared,
)

API_NETWORK_SCENARIO_SPECS: tuple[ScenarioSpec, ...] = (
    ScenarioSpec(
        _workload(
            "network.named.default",
            "Named three-tensor default-order contraction",
            "small",
            partial(
                _named_network_prepared,
                custom_order=False,
                size_label="small",
            ),
        ),
        _EAGER_NO_SUFFIX,
        QUICK,
    ),
    ScenarioSpec(
        _workload(
            "network.ncon.default",
            "Integer-label three-tensor default-order contraction",
            "small",
            partial(
                _ncon_network_prepared,
                custom_order=False,
                size_label="small",
            ),
        ),
        _EAGER_NO_SUFFIX,
        QUICK,
    ),
    ScenarioSpec(
        _workload(
            "network.disconnected",
            "Disconnected named tensor product",
            "small",
            _disconnected_network_prepared,
        ),
        _EAGER_NO_SUFFIX,
        QUICK,
    ),
    ScenarioSpec(_JAX_NETWORK_WORKLOAD, _NETWORK_JIT_COMPILE, QUICK),
    ScenarioSpec(_JAX_NETWORK_WORKLOAD, _NETWORK_JIT_CACHED, QUICK),
    ScenarioSpec(_JAX_NETWORK_WORKLOAD, _NETWORK_GRADIENT, QUICK),
    ScenarioSpec(
        _workload(
            "network.named.default.medium",
            "Named default-order medium contraction",
            "medium",
            partial(
                _named_network_prepared,
                custom_order=False,
                size_label="medium",
            ),
        ),
        _EAGER_NO_SUFFIX,
        FULL,
    ),
    ScenarioSpec(
        _workload(
            "network.named.custom.medium",
            "Named custom-order medium contraction",
            "medium",
            partial(
                _named_network_prepared,
                custom_order=True,
                size_label="medium",
            ),
        ),
        _EAGER_NO_SUFFIX,
        FULL,
    ),
    ScenarioSpec(
        _workload(
            "network.ncon.default.medium",
            "Integer-label default-order medium contraction",
            "medium",
            partial(
                _ncon_network_prepared,
                custom_order=False,
                size_label="medium",
            ),
        ),
        _EAGER_NO_SUFFIX,
        FULL,
    ),
    ScenarioSpec(
        _workload(
            "network.ncon.custom.medium",
            "Integer-label custom-order medium contraction",
            "medium",
            partial(
                _ncon_network_prepared,
                custom_order=True,
                size_label="medium",
            ),
        ),
        _EAGER_NO_SUFFIX,
        FULL,
    ),
)


def api_network_metadata() -> dict[str, object]:
    return {
        "scenario_count": len(API_NETWORK_SCENARIO_SPECS),
        "chain_dimensions": {
            size: list(_chain_dimensions(size))
            for size in ("small", "medium")
        },
    }
