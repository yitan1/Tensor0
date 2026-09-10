# Tensor0 Standard Benchmarks

This suite contains reproducible Tensor0 performance-regression benchmarks.
`python -m benchmarks.standard` is its scenario-runner entry point; workload
ownership follows Tensor0 operation domains:

- `linalg/` owns TensorMap composition, compact SVD, and their focused JAX and
  Trivial-path execution variants.
- `transforms/` owns permutation, repartition, and twist, including rank-4
  Trivial paths and cold/cached transformer diagnostics.
- `contractions/` owns every contraction workload:
  - `primitives/` covers `tensortrace` and `tensorcontract`, including their
    Trivial eager and JIT paths.
  - `api_networks/` covers small named, `ncon`, disconnected, JIT, and gradient
    workloads.
  - `tensor_networks/` keeps the fixed MPO, PEPO, and MERA matrix.
- `diagnostics/` owns cache-controlled layout construction and private
  strided-index builders.
- `_inputs.py` provides shared deterministic space and tensor construction.
- `_specs.py` defines workloads, execution modes, profiles, and their
  materialization into runnable scenarios.
- `_runner.py` provides scenario selection, timing, CLI, and result
  serialization.

Parameter-driven workload modules keep definitions in `cases.py`. Scalable
matrices live in the adjacent `params.toml`; fixed implementation diagnostics
stay in code so they cannot be mistaken for standard scaling cases. The runner
retains `tensor_networks` as a separate selection group even though it is
physically nested under `contractions`, allowing macro workloads to run
independently.
Parameter files use Tensor0-native lowercase values such as `float64`,
`complex128`, `trivial`, `z2`, `u1`, and `su2`, with `dtype`/`dtypes`,
`sector`, and `dimensions` as the common field vocabulary.

Every registered scenario is assembled through one model:

```text
WorkloadSpec(operation/topology, sector, dtype, dimensions)
    × ExecutionSpec(eager, cold, JIT compile, cached JIT, gradient)
    × ScenarioProfile(quick, full-only, explicit-only)
    → Scenario
```

Domain modules describe the operation and inputs. `_specs.py` alone owns
warmup, cold-cache, JIT compilation, cached execution, and gradient execution
behavior, so adding an execution axis does not duplicate domain workloads.

## Running benchmarks

Use the project environment through `uv`:

```sh
uv run python -m benchmarks.standard --list-scenarios
uv run python -m benchmarks.standard --quick --json
```

Without `--quick`, the default run includes `quick` and `full-only` scenarios.
Scenarios marked `explicit-only` run only when selected by id. An explicit
`--scenario` selection overrides profile-based scenario selection:

```sh
uv run python -m benchmarks.standard \
  --scenario internal.strided_indices.rank2_noncontiguous \
  --warmup 0 \
  --repeat 1 \
  --json
```

The suite contains 132 standard eager cases: 40 linalg, 4 transform, and 88
tensor-network cases. Focused sector, cache, contraction, JIT, and diagnostic
extensions bring the registry to 203 scenarios. `--quick` selects 32 smoke
scenarios; the default selects 105 quick and full scenarios; `--all` selects
the complete registry. The standard functional permutation inputs are 32 MiB;
because the input stays live while a distinct output is allocated, their
minimum logical live-data boundary is 64 MiB before backend temporaries. Large
tensor-network cases can require substantially more memory, so check available
memory before using `--all`:

```sh
uv run python -m benchmarks.standard --all \
  --json-output benchmark-results/standard/tensor0-suite.json
```

Use `--group` to select a benchmark domain without spelling every scenario id:

```sh
uv run python -m benchmarks.standard --all --group linalg --json
uv run python -m benchmarks.standard --all --group transforms --json
uv run python -m benchmarks.standard --all --group contractions --json
uv run python -m benchmarks.standard --all --group tensor_networks --json
uv run python -m benchmarks.standard --all --group diagnostics --json
```

Write reproducible reports with `--json-output` and `--markdown`:

```sh
uv run python -m benchmarks.standard \
  --all \
  --group contractions \
  --json-output benchmark-results/standard/contractions.json \
  --markdown benchmark-results/standard/contractions.md
```

## Measurement boundary

Scenario factories construct inputs and operations before timing. Optional
per-iteration setup, such as clearing a metadata or JAX cache for a cold
scenario, also runs before the timer starts. The timed region contains only the
operation and synchronization needed to make asynchronous work complete.

Cold compile scenarios therefore include compilation and execution, but not the
cache-clearing call. Cached JAX scenarios compile and synchronize once in the
factory before measurements begin.

MPO, PEPO, and MERA use fixed index graphs, space-distribution rules, and
expression contraction trees. Space and seeded random-tensor construction are
outside timing. The fully contracted result is kept as a rank-zero `TensorMap`;
scalar extraction is not timed. Exact dimensions, sigmas, contraction orders,
and support-file hashes are recorded in the result metadata.

Tensor0's `@`, `svd_compact`, and `permute` operations are functional, so result
allocation remains inside the timed region.

Benchmark results are local profiling and regression evidence. They are not
cross-library or hardware-independent performance claims.

## Focused Affine Update Check

Run `uv run python -m benchmarks.standard.diagnostics.stride_update --size 512 --repeats 30`
to compare contiguous and transposed conversion, assignment, accumulation,
weighted update, scaling, and JVP execution. It also compares fresh maps and
updates across contiguous, transposed, broadcast, and rank-4 layouts with
complete and partial coverage, identity mappings, and explicit static factors.
The layout matrix covers float16, float32, and complex64. Compare a fresh
static single scale with an identity-mapped dynamic scale to check equivalent
kernel performance; the leaf-dispatch tests separately observe actual kernel
selection, including zero/one factors and non-aligned tile tails.
The JSON output includes native
call counts and compiled allocation statistics; compare timing on the same
idle machine after warming both builds. This diagnostic is independent of the
standard scenario runner.
