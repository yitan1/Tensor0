# Cross-Backend Benchmarks

This suite compares independent tensor-library implementations on the same
frozen, backend-neutral workloads. It is separate from Tensor0's standard
performance-regression suite; Tensor0 is only one backend adapter.

The comparison model is:

```text
NeutralWorkload × Backend × MeasurementPolicy → ComparisonResult
```

## Ownership

- `workloads/*.toml` is the only source of mathematical workload definitions.
- `profiles.toml` selects official workload/backend matrices and timing counts.
- `protocol.schema.json` defines the versioned cross-language message format.
- `_protocol.py` normalizes and validates workloads and backend messages.
- `_orchestrator.py` runs isolated backend processes and pairs results.
- `_report.py` renders backend-neutral reports.
- `_plot.py` renders backend-neutral small-multiple latency plots.
- `backends/` translates normalized requests into each library's public API.

The suite does not import any module from `benchmarks.standard`. In particular,
its Tensor0 adapter constructs inputs independently and calls only Tensor0's
public API.

## Running

List the stable workload, profile, and backend names:

```sh
uv run python -m benchmarks.cross_backend --list-workloads
uv run python -m benchmarks.cross_backend --list-profiles
uv run python -m benchmarks.cross_backend --list-backends
```

Run the paired smoke profile:

```sh
uv run python -m benchmarks.cross_backend \
  --profile smoke \
  --json-output local/benchmarks/cross-smoke/results.json \
  --markdown local/benchmarks/cross-smoke/report.md
```

Run the cost-aware half matrix with the default single-thread policy:

```sh
uv run --group bench python -m benchmarks.cross_backend \
  --profile medium --threads 1 \
  --json-output benchmarks/cross_backend/results/cpu-1t/results.json \
  --markdown benchmarks/cross_backend/results/cpu-1t/report.md \
  --plot benchmarks/cross_backend/results/cpu-1t/latency.svg
```

For the retained four-worker medium policy, check core/sibling occupancy and
allowed cpuset first, then pin the whole orchestrator (CPU numbers are
host-specific):

```sh
env JAX_PLATFORMS=cpu JULIA_PKG_OFFLINE=true JULIA_PKG_PRECOMPILE_AUTO=0 \
  taskset -c 1,2,4,7 .venv/bin/python -m benchmarks.cross_backend \
  --profile medium --threads 4 --warmup 2 --repeat 7 --rounds 3 \
  --json-output benchmarks/cross_backend/results/cpu-4t/results.json \
  --markdown benchmarks/cross_backend/results/cpu-4t/report.md \
  --plot benchmarks/cross_backend/results/cpu-4t/latency.svg
```

This overwrites that configuration, not the one-worker snapshot. See its
[report](results/cpu-4t/report.md) for artifact identity and interference limits.

Run the complete matrix with eight CPU threads per backend process:

```sh
uv run --group bench python -m benchmarks.cross_backend \
  --profile full \
  --threads 8 \
  --json-output local/benchmarks/cross-full-8t/results.json \
  --markdown local/benchmarks/cross-full-8t/report.md \
  --plot local/benchmarks/cross-full-8t/latency.svg
```

The `medium` profile retains 36 of 69 workloads. It keeps four of eight
dimension cases per Abelian MPO/PEPO sector, three of six per Abelian MERA
sector, and every SU2 extension.

The `full` profile includes the complete comparison matrices: eight dimensions
per MPO/PEPO Abelian sector and six per MERA Abelian sector, followed by one SU2
extension per topology. Some cases intentionally reproduce the original
large-memory regime; the largest cases may require 32–64 GB of RAM.

The plot groups all dimensions for one topology/sector in the same panel.
Every x-axis group is one workload, and the adjacent colored boxplots are the
backends executing that exact workload. Steady-state latency is the logarithmic
vertical axis, so lower is faster. Rows group MPO, PEPO, and MERA; columns group
Trivial, Z2, U1, and SU2. Each dimension group labels the faster backend and
median speedup directly when numerical validation passes. Structural-only or
failed validation and unavailable backend results are explicit and do not
declare a faster backend. `--plot` accepts `.svg` and `.png`; SVG is recommended
for reports.

Run Tensor0 alone when developing or validating the protocol:

```sh
uv run python -m benchmarks.cross_backend \
  --profile smoke \
  --backend tensor0 \
  --warmup 0 \
  --repeat 1 \
  --rounds 1 \
  --json
```

The TensorKit adapter uses the pinned Julia environment in
`backends/tensorkit/`. Set `CROSS_BACKEND_JULIA` to an explicit Julia
executable when the desired runtime is not named `julia` on `PATH`.

## Workload contract

`workloads/tensor_networks.toml` defines MPO, PEPO, and MERA:

- tensor product-space signatures;
- operand order, index labels, conjugation, and contraction order;
- dtype and sector;
- nominal comparison dimension matrices and versioned space policies;
- explicit one-off spaces for workloads outside those matrices.

The protocol loader expands each matrix policy once into explicit sector
degeneracies before constructing the normalized JSON workload. Adapters receive
that same expanded workload and must not regenerate spaces from nominal
dimensions. `trivial_v1` uses one unit sector, `z2_equal_v1` splits the nominal
dimension evenly, and `u1_poisson_v1` reproduces the comparison suite's
charge-degeneracy distribution. Because that U1 policy rounds every charge
sector upward, an expanded space can be slightly larger than its nominal label
(for example, nominal dimension 10 expands to dimension 12). The explicit
expanded space—not the nominal label—is what both backends construct.

The TensorKit adapter validates each normalized contraction graph and order
against its corresponding fixed `@tensor` kernel before measurement; a graph
change cannot silently run through a stale compiled kernel.

Tensor values use a workload-owned reduced-data policy:

- `uniform_v1` fills every reduced coefficient with the tensor scale.
- `fusion_tree_v1` assigns each SU2 fusion-tree subblock and degeneracy
  coordinate a deterministic, nonuniform coefficient.

`fusion_tree_v1` is defined independently of backend storage order:

```text
mix(seed, value) = (257 * seed + value + 1) % 65521

seed = mix(17, number of UTF-8 tensor-name bytes)
seed = mix(seed, each tensor-name byte)
for (marker, tree) in ((11, row_tree), (29, column_tree)):
    seed = mix(seed, marker)
    seed = mix(seed, rank)
    seed = mix(seed, each uncoupled twice-spin label)
    seed = mix(seed, coupled twice-spin label)
    seed = mix(seed, each dual flag as 0 or 1)
    seed = mix(seed, number of inner lines)
    seed = mix(seed, each inner-line twice-spin label)

code = seed
for each one-based (axis, coordinate):
    code = (code + axis * 1009 * coordinate) % 65521
coefficient = tensor_scale * (1 + ((code % 1021) - 510) / 2048)
```

SU2 vertices are omitted because SU2 fusion is multiplicity-free. The adapters
assemble the same tree-keyed coefficients into their own block layout.

The second policy makes fusion-tree ordering, recoupling signs, and basis
conventions observable instead of allowing a uniform fill to hide them. Full
contractions return a scalar inside the timed operation, so output boundaries
match across backends.

## Measurement contract

Official comparisons use only `steady_state`:

- input and space construction are outside timing;
- contraction planning, tracing or macro expansion, compilation, and one
  mandatory compilation warmup are outside timing;
- configured warmups are outside timing;
- every timed invocation executes the already-prepared contraction kernel;
- result completion or synchronization is inside timing;
- the final timed invocation's scalar is retained for validation after timing;
- numerical validation never invokes the full contraction an extra time;
- public-API dispatch performed by that kernel remains inside timing;
- process startup is outside timing;
- backend and workload order alternate between paired rounds;
- raw samples, median, IQR, environment, versions, and workload hashes are
  retained.

The orchestrator defaults to a one-worker CPU budget. `--threads N` sets
BLAS/OpenMP and Julia to `N`, and the Tensor0 CPU adapter calls the existing
`tensor0._stride.set_num_threads(N)` before preparation or measurement. This
limits native stride workers independently of XLA: at `N=1`, Eigen
multithreading is disabled; at larger `N`, XLA receives an intra-operation
thread setting. Tensor0 metadata records the queried `native_worker_limit`,
CPU affinity where available, and thread environment; TensorKit records its
queried Julia and BLAS counts. Julia runs with `--startup-file=no`. This is a
worker upper bound, not an assertion of exactly `N` active workers or total OS
threads. CUDA execution does not change the native CPU worker limit (the
metadata field is `null`); `--threads` is not a GPU tuning parameter.

The runner does not pin CPUs. For strict single-logical-CPU comparisons on
Linux, first check the allowed cpuset and occupation of the selected CPU and
its SMT sibling, then run the entire orchestrator with `taskset -c <cpu>` so
both backend processes inherit the same affinity. Runs with different native
limits, affinities, or thread counts are different measurement policies and
must not be compared as if only the library implementation changed. Results
from the earlier runner that set only environment/XLA flags did not constrain
Tensor0's native stride workers and are not strict one-worker baselines.

`contender_speedup` means:

```text
baseline median / contender median
```

Values greater than one mean the contender was faster.

## Correctness

- All official workloads compare the scalar already returned by each backend's
  timed execution.
- SU2 uses nonuniform `fusion_tree_v1` data and therefore validates
  fusion-basis-sensitive results rather than structure alone.
- A backend failure is retained as structured result data and makes the command
  exit unsuccessfully.

## Retained configurations

See the [configuration overview](results/report.md) for retained medium CPU
snapshots: [one worker](results/cpu-1t/report.md),
[four workers](results/cpu-4t/report.md) and
[eight workers](results/cpu-8t/report.md). All use the shared Slurm node `n008`,
Xeon E5-2680 v3 (Haswell, no SMT), with verified native/Julia/BLAS budgets.
The new 4t/8t snapshots use the same committed `e5bd1f7` production source,
release artifact and environment, sequentially on same-socket physical cores;
pinned-process throughput and native/BLAS per-thread gates verified usable
resources before timing. No quota throttling or steal was observed. The node
was not exclusive and frequency was not locked; gate success does not imply
full occupancy of every workload. **The 1t snapshot uses older `333de65`
source, so these are not same-version 1→4→8 scaling measurements.** The 4t/8t
comparison is descriptive, not a randomized scaling trial. No dirty production
changes were measured. The standard suite remains local and independent.

Retain `report.md`, `results.json` and `latency.svg` under stable configuration
paths `results/cpu-1t/`, `results/cpu-4t/` and `results/cpu-8t/`, with date, revision, loaded artifact
identity, environment, worker/affinity policy, validation and limitations inside
the files. `results/report.md` only links these configurations. Publication-only
metadata belongs in JSON's top-level `publication_metadata`, without changing
numeric runner fields or samples. Convert sensitive absolute home paths to
repository-relative paths or explicit placeholders and document any conversion;
never change timings for privacy. As-run snapshot paths/hashes are provenance, not current
links. Do not add timestamp directories. Examples writing public paths overwrite
that configuration; review metadata before retaining a replacement. Historical
comparisons and internal experiments belong under `local/benchmarks/`, without
public reports depending on private files. See the
[shared publication convention](../README.md#retained-results).

### Explicit workload sampling plans

By default, the profile or `--repeat` count still applies to every workload in
all rounds. For a preflight-budgeted run, `--sampling-plan plan.json` accepts a
JSON list with exactly one entry for **every selected workload**:

```json
[
  {"workload_id": "tensor_networks.mpo.trivial.float64.d10x4x3", "repeat": 100},
  {"workload_id": "tensor_networks.mpo.trivial.float64.d2560x4x3", "repeat": 7}
]
```

Select the matching workloads with `--workload`, or provide a complete plan for
the chosen profile. Counts must be positive integers; unknown, missing and
duplicate workloads are errors. The plan overrides `--repeat` for selected
workloads, without adaptive sampling during measurement. Use the **same frozen
plan** for both backends and for each thread-count run. Preflight samples are
not formal measurement samples.

The optional protocol-v1 `measurement.repeat_by_workload` map carries the actual
counts to both adapters; requests without that field retain the original
uniform-repeat behavior. Result JSON retains the effective map, each record's
`repeat_per_round`, and the CLI plan's path and SHA256. `config.repeat` retains
the profile/CLI fallback and is not the actual workload count when a plan is
present.

Pooled min/median/IQR/max and speedup definitions are unchanged. New
`round_statistics` entries retain the original zero-based round index, actual
sample count and within-round statistics. `paired_rounds` compares medians only
for round indices completed by both backends. The Markdown report displays
round medians and these ratios, labeling `direction inconsistent` when ratios
occur both below and above one (exact ties are not reversals). A ratio above one
favors the contender. These diagnostics describe observed round stability;
they are **not significance tests**, and a lack of observed reversal is not a
claim of statistical significance or stability beyond the measured rounds.
