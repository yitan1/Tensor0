# Benchmark Suites

`benchmarks/` is a container for independent, reproducible benchmark suites:

- `standard/` measures Tensor0 performance regressions, execution modes, cache
  behavior, gradients, and implementation diagnostics.
- `cross_backend/` compares independent libraries on frozen backend-neutral
  workloads. Tensor0 is one adapter rather than the owner of that suite.

Standard is the primary suite for Tensor0's own regressions. Cross-backend
comparisons are auxiliary, run as needed; the costly full matrix is not a normal
CI or release gate. Retained results are selected snapshots, not measurements
of current HEAD.

The suites do not share workload definitions, inputs, runners, parameters, or
result models:

```text
standard:
    WorkloadSpec × ExecutionSpec × ScenarioProfile → Scenario

cross_backend:
    NeutralWorkload × Backend × MeasurementPolicy → ComparisonResult
```

Run them as Python modules from the repository root:

```sh
uv run python -m benchmarks.standard --quick --json
uv run python -m benchmarks.cross_backend --profile smoke --json
uv run --group bench python -m benchmarks.cross_backend \
  --profile full --plot local/benchmarks/cross-full/latency.svg
```

## Retained results

Suites publish retained snapshots at stable configuration paths:

- [Standard report](standard/results/report.md) and
  [raw results](standard/results/results.json).
- [Cross-backend configuration overview](cross_backend/results/report.md), with
  separate [CPU one-worker](cross_backend/results/cpu-1t/report.md) and
  [CPU four-worker](cross_backend/results/cpu-4t/report.md) and
  [CPU eight-worker](cross_backend/results/cpu-8t/report.md) reports, raw samples
  and latency plots.

The cross-backend overview documents the selected full CPU snapshots, sampling
policy, environment and limitations. Public JSON retains all raw timings,
round statistics and scalar-check summaries; complete as-run responses and
per-round scalar pairs are retained privately. Standard results remain an
independent local Ryzen snapshot, unchanged.

Standard uses `results/report.md` and `results/results.json`. Cross-backend uses
`results/{cpu-1t,cpu-4t,cpu-8t}/{report.md,results.json,latency.svg}` and a concise
`results/report.md` configuration overview. Do not add dated/source-SHA
subdirectories. Separate configurations are not controlled scalability comparisons
unless remeasured with a paired policy. Keep date, source revision,
dirty-tree state, loaded native artifact identity, environment, worker/affinity
policy, validation and limitations inside the report and JSON. Publication-only
machine-readable annotations belong in top-level `publication_metadata`;
original runner fields, samples, statistics and as-run provenance stay intact.
Replacing a retained snapshot requires reviewing its metadata and limitations;
running a command alone does not make it a validated publication.

Temporary outputs, historical comparisons and internal diagnostics belong under
`local/benchmarks/`. Public reports and reproduction commands must not depend on
private files. There is no shared root-level results directory.

A bounded CUDA stride owner/fiber diagnostic is available separately from the
standard suite runner:

```sh
mkdir -p local/benchmarks/cuda-owner-fiber
uv run python -m benchmarks.standard.diagnostics.cuda_owner_fiber > local/benchmarks/cuda-owner-fiber/results.json
```

Use a CUDA-enabled Tensor0 installation and JAX CUDA device; this diagnostic
fails instead of falling back to CPU. It checks results against independent NumPy
references, then records compiled-call-and-synchronize latency (not pure kernel
time). Run it separately for each candidate build to compare versions; its output
identifies the imported native extension. See
[`standard/README.md`](standard/README.md) and
[`cross_backend/README.md`](cross_backend/README.md) for suite-specific
contracts.
