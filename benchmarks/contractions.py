from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import subprocess
from typing import Any

import jax
import jax.numpy as jnp

import tensor0
from tensor0 import (
    FermionParity,
    HomSpace,
    SU2Irrep,
    TensorMap,
    U1Irrep,
    contract,
    hom,
    idx,
    ncon,
    space,
    tensorcontract,
    tensortrace,
    twist,
)
from tensor0.structure import get_degeneracystructure

from _runner import (
    Operation,
    Scenario,
    block_until_ready as _block_until_ready,
    run_cli as _run_cli,
    scenario as _scenario,
)


jax.config.update("jax_enable_x64", True)

def _data_for(hom_space: HomSpace, *, dtype: str, scale: float = 0.01):
    total_dim = get_degeneracystructure(hom_space).total_dim
    real = jnp.arange(1, total_dim + 1, dtype=jnp.float64) * scale
    if dtype == "float64":
        return real
    if dtype == "complex128":
        return real.astype(jnp.complex128) * (1.0 + 0.1j)
    raise ValueError(f"unsupported benchmark dtype: {dtype}")


def _tensor(hom_space: HomSpace, *, dtype: str = "float64") -> TensorMap:
    return TensorMap(hom_space, _data_for(hom_space, dtype=dtype))


def _clear_jax_caches() -> None:
    clear_caches = getattr(jax, "clear_caches", None)
    if clear_caches is not None:
        clear_caches()


def _twist_u1_identity() -> Operation:
    factor = space(U1Irrep, {0: 8, 1: 8})
    tensor = _tensor(hom((factor,), (factor,)))

    def run() -> object:
        return twist(tensor, 0)

    return run


def _twist_fermion_nontrivial() -> Operation:
    factor = space(FermionParity, {0: 16, 1: 16})
    tensor = _tensor(hom((factor,), (factor,)))

    def run() -> object:
        return twist(tensor, 0)

    return run


def _trace_u1_partial() -> Operation:
    open_space = space(U1Irrep, {0: 8, 1: 4})
    traced = space(U1Irrep, {0: 6, 1: 3})
    tensor = _tensor(hom((open_space, traced), (open_space, traced)))

    def run() -> object:
        return tensortrace(
            tensor,
            axes=((1,), (3,)),
            output=((0,), (2,)),
        )

    return run


def _trace_su2_partial() -> Operation:
    half = space(SU2Irrep, {1: 2})
    tensor = _tensor(hom((half, half), (half, half)))

    def run() -> object:
        return tensortrace(
            tensor,
            axes=((1,), (3,)),
            output=((0,), (2,)),
        )

    return run


def _trace_fermion_full() -> Operation:
    factor = space(FermionParity, {0: 16, 1: 16})
    tensor = _tensor(hom((factor,), (factor,)))

    def run() -> object:
        return tensortrace(
            tensor,
            axes=((0,), (1,)),
            output=((), ()),
        )

    return run


def _contract_u1_partial(*, dtype: str = "float64") -> Operation:
    a = space(U1Irrep, {0: 8, 1: 4})
    x = space(U1Irrep, {0: 6, 1: 3})
    b = space(U1Irrep, {0: 7, 1: 5})
    c = space(U1Irrep, {0: 4, 1: 2})
    d = space(U1Irrep, {0: 5, 1: 3})
    left = _tensor(hom((a, x), (c,)), dtype=dtype)
    right = _tensor(hom((x.dual(), b), (d,)), dtype=dtype)

    def run() -> object:
        return tensorcontract(
            left,
            right,
            axes=((1,), (0,)),
            output=(((0, 0), (1, 1)), ((0, 2), (1, 2))),
        )

    return run


def _contract_su2_fusion_basis() -> Operation:
    half = space(SU2Irrep, {1: 1})
    left = _tensor(hom((half, half, half), (half,)))
    right = _tensor(hom((half.dual(), half, half), (half,)))

    def run() -> object:
        return tensorcontract(
            left,
            right,
            axes=((2,), (0,)),
            output=(
                ((0, 0), (0, 1), (1, 1), (1, 2)),
                ((0, 3), (1, 3)),
            ),
        )

    return run


def _contract_fermion_twist() -> Operation:
    odd = space(FermionParity, {1: 16})
    odd_dual = odd.dual()
    left = _tensor(hom((odd,), (odd_dual,)))
    right = _tensor(hom((odd_dual,), (odd,)))

    def run() -> object:
        return tensorcontract(
            left,
            right,
            axes=((1,), (0,)),
            output=(((0, 0),), ((1, 1),)),
        )

    return run


def _chain_tensors(*, size_label: str) -> tuple[TensorMap, TensorMap, TensorMap]:
    if size_label == "small":
        dimensions = (4, 3, 5, 4)
    elif size_label == "medium":
        dimensions = (16, 4, 16, 4)
    else:
        raise ValueError(f"unsupported chain size: {size_label}")
    a_dim, x_dim, y_dim, b_dim = dimensions
    a = space(U1Irrep, {0: a_dim})
    x = space(U1Irrep, {0: x_dim})
    y = space(U1Irrep, {0: y_dim})
    b = space(U1Irrep, {0: b_dim})
    return (
        _tensor(hom((a,), (x,))),
        _tensor(hom((x,), (y,))),
        _tensor(hom((y,), (b,))),
    )


def _named_network(*, custom_order: bool, size_label: str) -> Operation:
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

    def run() -> object:
        return contract(
            idx(left, "a,x"),
            idx(middle, "x,y"),
            idx(right, "y,b"),
            output=("a", "b"),
            order=order,
        )

    return run


def _ncon_network(*, custom_order: bool, size_label: str) -> Operation:
    tensors = _chain_tensors(size_label=size_label)
    labels = ((-1, 1), (1, 2), (2, -2))
    order = (2, 1) if custom_order else None
    if custom_order:
        default = ncon(tensors, labels)
        custom = ncon(tensors, labels, order=order)
        _block_until_ready((default, custom))
        if not bool(jnp.allclose(default.storage.data, custom.storage.data)):
            raise AssertionError("ncon default/custom benchmark networks differ")

    def run() -> object:
        return ncon(tensors, labels, order=order)

    return run


def _disconnected_network() -> Operation:
    a = space(U1Irrep, {0: 16})
    b = space(U1Irrep, {0: 8})
    left = _tensor(hom((a,), ()))
    right = _tensor(hom((b,), ()))

    def run() -> object:
        return contract(
            idx(left, "a"),
            idx(right, "b"),
            output=("a,b", ""),
        )

    return run


def _jax_network_compile_and_run() -> Operation:
    tensors = _chain_tensors(size_label="small")

    def run() -> object:
        @jax.jit
        def network(left: TensorMap, middle: TensorMap, right: TensorMap):
            return contract(
                idx(left, "a,x"),
                idx(middle, "x,y"),
                idx(right, "y,b"),
                output=("a", "b"),
                order=("y", "x"),
            )

        return network(*tensors)

    return run


def _compiled_network():
    @jax.jit
    def network(left: TensorMap, middle: TensorMap, right: TensorMap):
        return contract(
            idx(left, "a,x"),
            idx(middle, "x,y"),
            idx(right, "y,b"),
            output=("a", "b"),
            order=("y", "x"),
        )

    return network


def _jax_network_cached() -> Operation:
    tensors = _chain_tensors(size_label="small")
    network = _compiled_network()
    _block_until_ready(network(*tensors))

    def run() -> object:
        return network(*tensors)

    return run


def _jax_network_value_and_grad() -> Operation:
    left, middle, right = _chain_tensors(size_label="small")

    def loss(data):
        candidate = TensorMap(left.space, data)
        result = contract(
            idx(candidate, "a,x"),
            idx(middle, "x,y"),
            idx(right, "y,b"),
            output=("a", "b"),
            order=("y", "x"),
        )
        return jnp.sum(result.storage.data**2)

    compiled = jax.jit(jax.value_and_grad(loss))
    _block_until_ready(compiled(left.storage.data))

    def run() -> object:
        return compiled(left.storage.data)

    return run


def _scenarios() -> tuple[Scenario, ...]:
    quick = (
        _scenario(
            "twist.u1.identity", "primitive", "U1 identity twist", "quick",
            "float64", "small", "eager", "warmed_metadata", _twist_u1_identity,
        ),
        _scenario(
            "twist.fermion.nontrivial", "primitive",
            "FermionParity nontrivial twist", "quick", "float64", "small",
            "eager", "warmed_metadata", _twist_fermion_nontrivial,
        ),
        _scenario(
            "trace.u1.partial", "primitive", "U1 partial tensor trace",
            "quick", "float64", "small", "eager", "warmed_metadata",
            _trace_u1_partial,
        ),
        _scenario(
            "contract.u1.partial", "primitive", "U1 partial binary contraction",
            "quick", "float64", "small", "eager", "warmed_metadata",
            _contract_u1_partial,
        ),
        _scenario(
            "network.named.default", "network",
            "Named three-tensor default-order contraction", "quick", "float64",
            "small", "eager", "warmed_metadata",
            lambda: _named_network(custom_order=False, size_label="small"),
        ),
        _scenario(
            "network.ncon.default", "network",
            "Integer-label three-tensor default-order contraction", "quick",
            "float64", "small", "eager", "warmed_metadata",
            lambda: _ncon_network(custom_order=False, size_label="small"),
        ),
        _scenario(
            "network.disconnected", "network", "Disconnected named tensor product",
            "quick", "float64", "small", "eager", "warmed_metadata",
            _disconnected_network,
        ),
        _scenario(
            "jax.network.compile_and_run", "jax",
            "Named network JIT compile plus first run", "quick", "float64",
            "small", "jit_compile_and_run", "clear_jax_caches",
            _jax_network_compile_and_run,
            before_each=_clear_jax_caches,
        ),
        _scenario(
            "jax.network.cached", "jax", "Named network cached JIT execution",
            "quick", "float64", "small", "jit_cached_run", "compiled_once",
            _jax_network_cached,
        ),
        _scenario(
            "jax.network.value_and_grad", "jax",
            "Named network cached reverse-mode value and gradient", "quick",
            "float64", "small", "value_and_grad_cached", "compiled_once",
            _jax_network_value_and_grad,
        ),
    )
    additional = (
        _scenario(
            "trace.su2.partial", "primitive", "SU2 partial tensor trace",
            "full-only", "float64", "small", "eager", "warmed_metadata",
            _trace_su2_partial,
        ),
        _scenario(
            "trace.fermion.full", "primitive", "FermionParity full tensor trace",
            "full-only", "float64", "small", "eager", "warmed_metadata",
            _trace_fermion_full,
        ),
        _scenario(
            "contract.su2.fusion_basis", "primitive",
            "SU2 fusion-basis binary contraction", "full-only", "float64",
            "small", "eager", "warmed_metadata", _contract_su2_fusion_basis,
        ),
        _scenario(
            "contract.fermion.twist", "primitive",
            "FermionParity contraction with right twist", "full-only",
            "float64", "small", "eager", "warmed_metadata",
            _contract_fermion_twist,
        ),
        _scenario(
            "network.named.default.medium", "network",
            "Named default-order medium contraction", "full-only", "float64",
            "medium", "eager", "warmed_metadata",
            lambda: _named_network(custom_order=False, size_label="medium"),
        ),
        _scenario(
            "network.named.custom.medium", "network",
            "Named custom-order medium contraction", "full-only", "float64",
            "medium", "eager", "warmed_metadata",
            lambda: _named_network(custom_order=True, size_label="medium"),
        ),
        _scenario(
            "network.ncon.default.medium", "network",
            "Integer-label default-order medium contraction", "full-only",
            "float64", "medium", "eager", "warmed_metadata",
            lambda: _ncon_network(custom_order=False, size_label="medium"),
        ),
        _scenario(
            "network.ncon.custom.medium", "network",
            "Integer-label custom-order medium contraction", "full-only",
            "float64", "medium", "eager", "warmed_metadata",
            lambda: _ncon_network(custom_order=True, size_label="medium"),
        ),
        _scenario(
            "contract.u1.partial.complex", "primitive",
            "Complex U1 partial binary contraction", "explicit-only",
            "complex128", "medium", "eager", "warmed_metadata",
            lambda: _contract_u1_partial(dtype="complex128"),
        ),
    )
    return quick + additional


def _git_state(path: Path) -> dict[str, str | bool]:
    revision = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ("git", "status", "--short"),
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {"revision": revision, "dirty": bool(status)}


def _environment() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "jax": getattr(jax, "__version__", "unknown"),
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "tensor0": getattr(tensor0, "__version__", "unknown"),
        "thread_settings": {
            name: os.environ.get(name, "unset")
            for name in (
                "XLA_FLAGS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
            )
        },
        "public_repository": _git_state(root),
    }


def _render_markdown(payload: dict[str, Any], *, command: str) -> str:
    lines = [
        "# Tensor0 Contraction Benchmark Baseline",
        "",
        "This local report records Tensor0 measurements for release planning;",
        "it is not a cross-library or hardware-general performance claim.",
        "",
        "## Command",
        "",
        f"`{command}`",
        "",
        "## Environment",
        "",
        "```json",
        json.dumps(payload["environment"], indent=2, sort_keys=True),
        "```",
        "",
        "## Configuration",
        "",
    ]
    for key, value in payload["config"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Results", ""])
    for result in payload["results"]:
        lines.extend(
            [
                f"### `{result['id']}`",
                "",
                f"- Group: `{result['group']}`",
                f"- Description: {result['description']}",
                f"- Profile: `{result['scenario_profile']}`",
                f"- Dtype: `{result['dtype']}`",
                f"- Size: `{result['size_label']}`",
                f"- Execution: `{result['execution']}`",
                f"- Cache policy: `{result['cache_policy']}`",
                f"- Warmup/repeat: `{result['warmup']}/{result['repeat']}`",
                f"- Min/median/IQR/max ms: `{result['min_ms']:.3f}` / "
                f"`{result['median_ms']:.3f}` / `{result['iqr_ms']:.3f}` / "
                f"`{result['max_ms']:.3f}`",
                "- Times ms: `"
                + ", ".join(f"{sample:.3f}" for sample in result["times_ms"])
                + "`",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation Boundary",
            "",
            "- Setup and metadata construction are outside eager timed regions.",
            "- JIT compile, cached execution, and cached gradient are separate scenarios.",
            "- Default/custom order fixtures are checked for equal results before timing.",
            "- Optimization decisions require focused profiling beyond median ranking.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    _run_cli(
        description="Run Tensor0 contraction benchmarks.",
        script_path="benchmarks/contractions.py",
        scenarios=_scenarios(),
        environment=_environment,
        render_markdown=_render_markdown,
    )


if __name__ == "__main__":
    main()
