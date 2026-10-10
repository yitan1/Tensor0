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

## Retained snapshot

See the [retained report](results/report.md) and [raw results and samples](results/results.json).
The current default CPU data is an **exploratory snapshot on a non-isolated
workstation**, not an interference-free formal baseline. It records the loaded
`333de65` native artifact, not unbuilt workspace native changes; successful
timing is not numerical validation.

Use these fixed filenames and keep date, revision, loaded artifact identity,
environment, worker/affinity policy and limitations inside the report and JSON.
Publication-only annotations belong in top-level `publication_metadata`;
runner fields, samples and as-run provenance remain unchanged. No timestamp
directories or results index are required. Commands below writing these paths
overwrite the retained snapshot; review metadata before publishing a replacement.
Historical comparisons and diagnostics belong under `local/benchmarks/`,
without public report dependencies on private files. See the
[shared publication convention](../README.md#retained-results).

## Running benchmarks

Use the project environment through `uv`:

```sh
uv run python -m benchmarks.standard --list-scenarios
uv run python -m benchmarks.standard --quick --json
```

### Strict single-worker CPU runs

BLAS/OpenMP environment variables and XLA Eigen flags do **not** constrain
Tensor0's independent native stride worker limit. The standard CLI does not
set that limit. To reproduce a strict CPU run without adding a new CLI option,
use the existing API in the same process before running the module:

```sh
env JAX_PLATFORMS=cpu \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  JULIA_NUM_THREADS=1 CROSS_BACKEND_THREADS=1 \
  OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE \
  XLA_FLAGS=--xla_cpu_multi_thread_eigen=false \
  taskset -c 2 .venv/bin/python -c \
  'import runpy; from tensor0._stride import set_num_threads, get_num_threads; set_num_threads(1); assert get_num_threads() == 1; runpy.run_module("benchmarks.standard", run_name="__main__")' \
  --json-output benchmarks/standard/results/results.json \
  --markdown benchmarks/standard/results/report.md
```

CPU2 is an example, not a portable default: check the allowed cpuset and that
CPU's SMT sibling for occupation before pinning. Runtime helper OS threads may
still exist; the native limit is a worker upper bound. This CPU recipe does not
change CUDA execution. Record the queried native limit and actual loaded
extension identity alongside results; successful timing is not numerical
validation.

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
  --json-output benchmarks/standard/results/results.json
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
  --json-output benchmarks/standard/results/results.json \
  --markdown benchmarks/standard/results/report.md
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
complete and partial coverage, identity mappings, and explicit factors.
The layout matrix covers float16, float32, and complex64. The diagnostic uses
native Copy, address accumulation, and Update; coefficients are separate from
layout records. It measures operation performance, not specialized kernel
selection or parity with the old backend.
The JSON output includes synchronized timing and compiled allocation
statistics, not legacy native-call counters; compare timing on the same
idle machine after warming both builds. This diagnostic is independent of the
standard scenario runner.
