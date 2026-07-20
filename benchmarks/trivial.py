from __future__ import annotations

from collections.abc import Callable
import hashlib
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
    ComplexSpace,
    HomSpace,
    SU2Irrep,
    TensorMap,
    U1Irrep,
    contract,
    dim,
    from_dense,
    hom,
    idx,
    permute,
    space,
    tensorcontract,
    tensortrace,
    to_dense,
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

PreparedOperation = tuple[Callable[..., object], tuple[object, ...]]
PreparedFactory = Callable[[], PreparedOperation]


def _clear_jax_caches() -> None:
    clear_caches = getattr(jax, "clear_caches", None)
    if clear_caches is not None:
        clear_caches()


def _array(shape: tuple[int, ...], *, dtype: str) -> jax.Array:
    size = 1
    for dimension in shape:
        size *= dimension
    values = jnp.arange(1, size + 1, dtype=jnp.float64).reshape(shape) / size
    if dtype == "float64":
        return values
    if dtype == "complex128":
        return values.astype(jnp.complex128) * (1.0 + 0.125j)
    raise ValueError(f"unsupported benchmark dtype: {dtype}")


def _tensor(space_value: HomSpace, *, dtype: str) -> TensorMap:
    shape = tuple(dim(item) for item in space_value.codomain.spaces) + tuple(
        dim(item) for item in space_value.domain.spaces
    )
    return from_dense(space_value, _array(shape, dtype=dtype))


def _packed_tensor(space_value: HomSpace, *, dtype: str) -> TensorMap:
    size = get_degeneracystructure(space_value).total_dim
    return TensorMap(space_value, _array((size,), dtype=dtype))


def _eager(factory: PreparedFactory) -> Operation:
    function, arguments = factory()
    _block_until_ready(function(*arguments))

    def run() -> object:
        return function(*arguments)

    return run


def _jit_compile_and_run(factory: PreparedFactory) -> Operation:
    function, arguments = factory()
    _block_until_ready(function(*arguments))
    compiled = jax.jit(function)

    def run() -> object:
        return compiled(*arguments)

    return run


def _jit_cached(factory: PreparedFactory) -> Operation:
    function, arguments = factory()
    compiled = jax.jit(function)
    _block_until_ready(compiled(*arguments))

    def run() -> object:
        return compiled(*arguments)

    return run


def _from_dense_prepared(*, dtype: str, size_label: str) -> PreparedOperation:
    dimensions = {"small": (16, 12), "medium": (64, 48)}[size_label]
    codomain_dim, domain_dim = dimensions
    space_value = hom(
        (ComplexSpace(codomain_dim),),
        (ComplexSpace(domain_dim),),
    )
    dense = _array(dimensions, dtype=dtype)

    def run(data: object) -> object:
        return from_dense(space_value, data)

    return run, (dense,)


def _to_dense_prepared(*, dtype: str, size_label: str) -> PreparedOperation:
    dimensions = {
        "small": (4, 3, 5, 2),
        "medium": (12, 8, 10, 6),
    }[size_label]
    a_dim, b_dim, c_dim, d_dim = dimensions
    space_value = hom(
        (ComplexSpace(a_dim), ComplexSpace(b_dim)),
        (ComplexSpace(c_dim), ComplexSpace(d_dim)),
    )
    tensor = _tensor(space_value, dtype=dtype)
    return to_dense, (tensor,)


def _permute_prepared(*, dtype: str, size_label: str) -> PreparedOperation:
    dimensions = {
        "small": (4, 3, 5, 2),
        "medium": (12, 8, 10, 6),
    }[size_label]
    a_dim, b_dim, c_dim, d_dim = dimensions
    space_value = hom(
        (ComplexSpace(a_dim), ComplexSpace(b_dim)),
        (ComplexSpace(c_dim), ComplexSpace(d_dim)),
    )
    tensor = _tensor(space_value, dtype=dtype)

    def run(value: TensorMap) -> object:
        return permute(value, ((1, 0), (3, 2)))

    return run, (tensor,)


def _trace_prepared(*, dtype: str, size_label: str) -> PreparedOperation:
    dimensions = {
        "small": (8, 4, 6),
        "medium": (24, 12, 20),
    }[size_label]
    out_dim, traced_dim, in_dim = dimensions
    traced = ComplexSpace(traced_dim)
    space_value = hom(
        (ComplexSpace(out_dim), traced),
        (ComplexSpace(in_dim), traced),
    )
    tensor = _tensor(space_value, dtype=dtype)

    def run(value: TensorMap) -> object:
        return tensortrace(
            value,
            axes=((1,), (3,)),
            output=((0,), (2,)),
        )

    return run, (tensor,)


def _contract_prepared(*, dtype: str, size_label: str) -> PreparedOperation:
    dimensions = {
        "small": (6, 4, 7, 5, 3),
        "medium": (16, 12, 18, 10, 8),
    }[size_label]
    a_dim, x_dim, b_dim, c_dim, d_dim = dimensions
    x = ComplexSpace(x_dim)
    left_space = hom(
        (ComplexSpace(a_dim), x),
        (ComplexSpace(c_dim),),
    )
    right_space = hom(
        (x.dual(), ComplexSpace(b_dim)),
        (ComplexSpace(d_dim),),
    )
    left = _tensor(left_space, dtype=dtype)
    right = _tensor(right_space, dtype=dtype)

    def run(left_value: TensorMap, right_value: TensorMap) -> object:
        return tensorcontract(
            left_value,
            right_value,
            axes=((1,), (0,)),
            output=(((0, 0), (1, 1)), ((0, 2), (1, 2))),
        )

    return run, (left, right)


def _composition_prepared(*, dtype: str, size_label: str) -> PreparedOperation:
    dimensions = {"small": (16, 12, 14), "medium": (64, 48, 56)}[size_label]
    out_dim, mid_dim, in_dim = dimensions
    middle = ComplexSpace(mid_dim)
    left = _tensor(
        hom((ComplexSpace(out_dim),), (middle,)),
        dtype=dtype,
    )
    right = _tensor(
        hom((middle,), (ComplexSpace(in_dim),)),
        dtype=dtype,
    )

    def run(left_value: TensorMap, right_value: TensorMap) -> object:
        return left_value @ right_value

    return run, (left, right)


def _network_prepared(*, dtype: str, size_label: str) -> PreparedOperation:
    dimensions = {"small": (6, 4, 5, 7), "medium": (32, 24, 28, 36)}[
        size_label
    ]
    a_dim, x_dim, y_dim, b_dim = dimensions
    a = ComplexSpace(a_dim)
    x = ComplexSpace(x_dim)
    y = ComplexSpace(y_dim)
    b = ComplexSpace(b_dim)
    left = _tensor(hom((a,), (x,)), dtype=dtype)
    middle = _tensor(hom((x,), (y,)), dtype=dtype)
    right = _tensor(hom((y,), (b,)), dtype=dtype)

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
            order=("y", "x"),
        )

    return run, (left, middle, right)


def _protected_u1_contract() -> PreparedOperation:
    a = space(U1Irrep, {0: 4, 1: 3})
    x = space(U1Irrep, {0: 3, 1: 2})
    b = space(U1Irrep, {0: 5, 1: 2})
    c = space(U1Irrep, {0: 3, 1: 2})
    d = space(U1Irrep, {0: 4, 1: 2})
    left = _packed_tensor(hom((a, x), (c,)), dtype="float64")
    right = _packed_tensor(hom((x.dual(), b), (d,)), dtype="float64")

    def run(left_value: TensorMap, right_value: TensorMap) -> object:
        return tensorcontract(
            left_value,
            right_value,
            axes=((1,), (0,)),
            output=(((0, 0), (1, 1)), ((0, 2), (1, 2))),
        )

    return run, (left, right)


def _protected_u1_from_dense() -> PreparedOperation:
    factor = space(U1Irrep, {-1: 2, 0: 4, 1: 3})
    space_value = hom((factor,), (factor,))
    dense = to_dense(_packed_tensor(space_value, dtype="float64"))

    def run(data: object) -> object:
        return from_dense(space_value, data)

    return run, (dense,)


def _protected_u1_composition() -> PreparedOperation:
    output = space(U1Irrep, {-1: 2, 0: 4, 1: 3})
    middle = space(U1Irrep, {-1: 3, 0: 3, 1: 2})
    input_space = space(U1Irrep, {-1: 2, 0: 5, 1: 2})
    left = _packed_tensor(hom((output,), (middle,)), dtype="float64")
    right = _packed_tensor(hom((middle,), (input_space,)), dtype="float64")

    def run(left_value: TensorMap, right_value: TensorMap) -> object:
        return left_value @ right_value

    return run, (left, right)


def _protected_su2_trace() -> PreparedOperation:
    half = space(SU2Irrep, {1: 2})
    tensor = _packed_tensor(hom((half, half), (half, half)), dtype="float64")

    def run(value: TensorMap) -> object:
        return tensortrace(
            value,
            axes=((1,), (3,)),
            output=((0,), (2,)),
        )

    return run, (tensor,)


def _protected_su2_permute() -> PreparedOperation:
    half = space(SU2Irrep, {1: 3})
    one = space(SU2Irrep, {2: 2})
    tensor = _packed_tensor(hom((half, one), (half, one)), dtype="float64")

    def run(value: TensorMap) -> object:
        return permute(value, ((1, 0), (3, 2)))

    return run, (tensor,)


def _execution_scenarios(
    *,
    base_id: str,
    group: str,
    description: str,
    dtype: str,
    size_label: str,
    factory: PreparedFactory,
    eager_profile: str,
    include_jit: bool,
    jit_profile: str = "full-only",
) -> tuple[Scenario, ...]:
    scenarios = [
        _scenario(
            f"{base_id}.eager",
            group,
            description,
            eager_profile,
            dtype,
            size_label,
            "eager",
            "warmed_metadata",
            lambda: _eager(factory),
        )
    ]
    if include_jit:
        scenarios.extend(
            (
                _scenario(
                    f"{base_id}.jit_compile_and_run",
                    group,
                    description,
                    jit_profile,
                    dtype,
                    size_label,
                    "jit_compile_and_run",
                    "clear_jax_caches",
                    lambda: _jit_compile_and_run(factory),
                    before_each=_clear_jax_caches,
                ),
                _scenario(
                    f"{base_id}.jit_cached_run",
                    group,
                    description,
                    jit_profile,
                    dtype,
                    size_label,
                    "jit_cached_run",
                    "compiled_once",
                    lambda: _jit_cached(factory),
                ),
            )
        )
    return tuple(scenarios)


def _scenarios() -> tuple[Scenario, ...]:
    scenarios: list[Scenario] = []
    small_operations = (
        (
            "dense.trivial.from_dense.rank2.float64.small",
            "dense",
            "Trivial rank-2 from_dense with shape (16, 12)",
            lambda: _from_dense_prepared(dtype="float64", size_label="small"),
            False,
            "full-only",
        ),
        (
            "dense.trivial.to_dense.rank4.float64.small",
            "dense",
            "Trivial rank-4 to_dense with shape (4, 3, 5, 2)",
            lambda: _to_dense_prepared(dtype="float64", size_label="small"),
            True,
            "full-only",
        ),
        (
            "permute.trivial.rank4.float64.small",
            "permutation",
            "Trivial rank-4 nonidentity permutation with shape (4, 3, 5, 2)",
            lambda: _permute_prepared(dtype="float64", size_label="small"),
            True,
            "full-only",
        ),
        (
            "trace.trivial.partial.rank4.float64.small",
            "trace",
            "Trivial partial trace with source shape (8, 4, 6, 4)",
            lambda: _trace_prepared(dtype="float64", size_label="small"),
            True,
            "full-only",
        ),
        (
            "contract.trivial.partial.float64.small",
            "contraction",
            "Trivial partial contraction with dimensions (6, 4, 7, 5, 3)",
            lambda: _contract_prepared(dtype="float64", size_label="small"),
            True,
            "full-only",
        ),
        (
            "composition.trivial.float64.small",
            "composition",
            "Trivial matrix composition with dimensions (16, 12, 14)",
            lambda: _composition_prepared(dtype="float64", size_label="small"),
            True,
            "full-only",
        ),
        (
            "network.trivial.three_tensor.float64.small",
            "network",
            "Trivial three-tensor chain with dimensions (6, 4, 5, 7)",
            lambda: _network_prepared(dtype="float64", size_label="small"),
            True,
            "quick",
        ),
    )
    for base_id, group, description, factory, include_jit, jit_profile in small_operations:
        scenarios.extend(
            _execution_scenarios(
                base_id=base_id,
                group=group,
                description=description,
                dtype="float64",
                size_label="small",
                factory=factory,
                eager_profile="quick",
                include_jit=include_jit,
                jit_profile=jit_profile,
            )
        )

    medium_operations = (
        (
            "dense.trivial.from_dense.rank2.complex128.medium",
            "dense",
            "Trivial rank-2 from_dense with shape (64, 48)",
            lambda: _from_dense_prepared(dtype="complex128", size_label="medium"),
        ),
        (
            "dense.trivial.to_dense.rank4.complex128.medium",
            "dense",
            "Trivial rank-4 to_dense with shape (12, 8, 10, 6)",
            lambda: _to_dense_prepared(dtype="complex128", size_label="medium"),
        ),
        (
            "permute.trivial.rank4.complex128.medium",
            "permutation",
            "Trivial rank-4 nonidentity permutation with shape (12, 8, 10, 6)",
            lambda: _permute_prepared(dtype="complex128", size_label="medium"),
        ),
        (
            "trace.trivial.partial.rank4.complex128.medium",
            "trace",
            "Trivial partial trace with source shape (24, 12, 20, 12)",
            lambda: _trace_prepared(dtype="complex128", size_label="medium"),
        ),
        (
            "contract.trivial.partial.complex128.medium",
            "contraction",
            "Trivial partial contraction with dimensions (16, 12, 18, 10, 8)",
            lambda: _contract_prepared(dtype="complex128", size_label="medium"),
        ),
        (
            "composition.trivial.complex128.medium",
            "composition",
            "Trivial matrix composition with dimensions (64, 48, 56)",
            lambda: _composition_prepared(dtype="complex128", size_label="medium"),
        ),
        (
            "network.trivial.three_tensor.complex128.medium",
            "network",
            "Trivial three-tensor chain with dimensions (32, 24, 28, 36)",
            lambda: _network_prepared(dtype="complex128", size_label="medium"),
        ),
    )
    for base_id, group, description, factory in medium_operations:
        scenarios.extend(
            _execution_scenarios(
                base_id=base_id,
                group=group,
                description=description,
                dtype="complex128",
                size_label="medium",
                factory=factory,
                eager_profile="full-only",
                include_jit=False,
            )
        )

    scenarios.extend(
        (
            _scenario(
                "protect.u1.from_dense.float64.small.eager",
                "protected",
                "Protected multi-sector U1 dense construction",
                "quick",
                "float64",
                "small",
                "eager",
                "warmed_metadata",
                lambda: _eager(_protected_u1_from_dense),
            ),
            _scenario(
                "protect.su2.permute.float64.small.eager",
                "protected",
                "Protected SU2 nonidentity permutation",
                "quick",
                "float64",
                "small",
                "eager",
                "warmed_metadata",
                lambda: _eager(_protected_su2_permute),
            ),
            _scenario(
                "protect.u1.contract.float64.small.eager",
                "protected",
                "Protected multi-sector U1 partial contraction",
                "quick",
                "float64",
                "small",
                "eager",
                "warmed_metadata",
                lambda: _eager(_protected_u1_contract),
            ),
            _scenario(
                "protect.su2.trace.float64.small.eager",
                "protected",
                "Protected SU2 partial trace",
                "quick",
                "float64",
                "small",
                "eager",
                "warmed_metadata",
                lambda: _eager(_protected_su2_trace),
            ),
            _scenario(
                "protect.u1.composition.float64.small.eager",
                "protected",
                "Protected multi-sector U1 composition",
                "quick",
                "float64",
                "small",
                "eager",
                "warmed_metadata",
                lambda: _eager(_protected_u1_composition),
            ),
        )
    )
    return tuple(scenarios)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_bytes(root: Path, *arguments: str) -> bytes:
    return subprocess.run(
        ("git", *arguments),
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout


def _source_state(root: Path) -> dict[str, Any]:
    revision = _git_bytes(root, "rev-parse", "HEAD").decode().strip()
    status = _git_bytes(root, "status", "--short")
    diff = _git_bytes(root, "diff", "--binary", "--no-ext-diff", "HEAD")
    untracked = _git_bytes(
        root,
        "ls-files",
        "--others",
        "--exclude-standard",
        "-z",
    )
    manifest = []
    for encoded_path in untracked.split(b"\0"):
        if not encoded_path:
            continue
        relative_path = encoded_path.decode()
        path = root / relative_path
        if path.is_file():
            manifest.append(
                {"path": relative_path, "sha256": _sha256(path.read_bytes())}
            )
    return {
        "revision": revision,
        "dirty": bool(status),
        "binary_diff_sha256": _sha256(diff),
        "untracked_files": manifest,
    }


def _environment() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    script = Path(__file__).resolve()
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "jax": getattr(jax, "__version__", "unknown"),
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "jax_enable_x64": True,
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
        "benchmark_script_sha256": _sha256(script.read_bytes()),
        "public_repository": _source_state(root),
    }


def _render_markdown(payload: dict[str, Any], *, command: str) -> str:
    lines = [
        "# Tensor0 Trivial Fast-Path Benchmark",
        "",
        "This report records reproducible local measurements. It does not make",
        "cross-machine, cross-backend, or cross-library performance claims.",
        "",
        "## Command",
        "",
        f"`{command}`",
        "",
        "## Environment And Source State",
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
                f"- Dtype/size: `{result['dtype']}` / `{result['size_label']}`",
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
            "## Measurement Boundary",
            "",
            "- Input generation and metadata warmup occur outside timed regions.",
            "- Eager, compile-and-first-run, and cached JIT scenarios have distinct IDs.",
            "- Compile scenarios clear JAX caches before each timed iteration.",
            "- Cached JIT scenarios compile and synchronize once in their factory.",
            "- The shared runner synchronizes every returned JAX array before timing stops.",
            "",
            "## Interpretation Boundary",
            "",
            "- Task 4 and Task 5 gates must be locked from repeated private baseline runs.",
            "- This public report contains measurements only; it does not choose a gate outcome.",
            "- TensorKit source structure is design evidence, not a timing comparator.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    _run_cli(
        description="Run Tensor0 Trivial fast-path benchmarks.",
        script_path="benchmarks/trivial.py",
        scenarios=_scenarios(),
        environment=_environment,
        render_markdown=_render_markdown,
    )


if __name__ == "__main__":
    main()
