# Tensor0 Benchmarks

This directory contains reproducible benchmarks that run from a public Tensor0
checkout:

- `core.py` covers layout construction, tensor composition, factorization,
  transforms, and JAX execution modes.
- `contractions.py` covers contraction primitives, named and `ncon` networks,
  and JAX execution modes.
- `trivial.py` covers Trivial dense conversion, permutation, trace,
  contraction, composition, and network fast-path candidates together with
  protected U1 and SU2 scenarios.
- `_runner.py` provides the shared scenario registry, timing, CLI, and result
  serialization.

## Running benchmarks

Use the project environment through `uv`:

```sh
uv run python benchmarks/core.py --list-scenarios
uv run python benchmarks/core.py --quick --json
uv run python benchmarks/contractions.py --quick --json
uv run python benchmarks/trivial.py --quick --json
```

Without `--quick`, the default run includes `quick` and `full-only` scenarios.
Scenarios marked `explicit-only` run only when selected by id. An explicit
`--scenario` selection overrides profile-based scenario selection:

```sh
uv run python benchmarks/core.py \
  --scenario internal.strided_indices.rank2_noncontiguous \
  --warmup 0 \
  --repeat 1 \
  --json
```

Write reproducible reports with `--json-output` and `--markdown`:

```sh
uv run python benchmarks/contractions.py \
  --quick \
  --json-output benchmark-results/contractions.json \
  --markdown benchmark-results/contractions.md
```

## Measurement boundary

Scenario factories construct inputs and operations before timing. Optional
per-iteration setup, such as clearing a metadata or JAX cache for a cold
scenario, also runs before the timer starts. The timed region contains only the
operation and synchronization needed to make asynchronous work complete.

Cold compile scenarios therefore include compilation and execution, but not the
cache-clearing call. Cached JAX scenarios compile and synchronize once in the
factory before measurements begin.

Benchmark results are local profiling and regression evidence. They are not
cross-library or hardware-independent performance claims.
