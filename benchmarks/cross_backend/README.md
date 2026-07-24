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
  --json-output benchmark-results/cross_backend/smoke.json \
  --markdown benchmark-results/cross_backend/smoke.md
```

Run the cost-aware half matrix with the default single-thread policy:

```sh
uv run --group bench python -m benchmarks.cross_backend \
  --profile medium \
  --json-output benchmark-results/cross_backend/medium-1t.json \
  --markdown benchmark-results/cross_backend/medium-1t.md \
  --plot benchmark-results/cross_backend/medium-1t.svg
```

Run the complete matrix with eight CPU threads per backend process:

```sh
uv run --group bench python -m benchmarks.cross_backend \
  --profile full \
  --threads 8 \
  --json-output benchmark-results/cross_backend/full-8t.json \
  --markdown benchmark-results/cross_backend/full-8t.md \
  --plot benchmark-results/cross_backend/full-8t.svg
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

The orchestrator defaults to one CPU thread. `--threads N` gives both backends
the same explicit thread budget: BLAS/OpenMP and Julia use `N`, while Tensor0's
XLA CPU Eigen pool receives `N` intra-operation threads. Backend metadata and
the result configuration record the effective settings. Runs with different
thread counts are different measurement policies and must not be compared as
if only the library implementation changed.

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

Raw results are generated under `benchmark-results/cross_backend/`, which is
ignored by Git. Large retained experiments belong under `local/benchmarks/`.
