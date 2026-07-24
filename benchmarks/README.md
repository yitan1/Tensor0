# Benchmark Suites

`benchmarks/` is a container for independent, reproducible benchmark suites:

- `standard/` measures Tensor0 performance regressions, execution modes, cache
  behavior, gradients, and implementation diagnostics.
- `cross_backend/` compares independent libraries on frozen backend-neutral
  workloads. Tensor0 is one adapter rather than the owner of that suite.

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
  --profile full --plot benchmark-results/cross_backend/full.svg
```

Generated results belong under the ignored `benchmark-results/standard/` and
`benchmark-results/cross_backend/` directories. Large results retained for
internal analysis belong under `local/benchmarks/`.

See [`standard/README.md`](standard/README.md) and
[`cross_backend/README.md`](cross_backend/README.md) for suite-specific
contracts.
