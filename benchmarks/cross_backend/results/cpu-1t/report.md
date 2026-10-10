# Cross-Backend Medium CPU Benchmark — 2026-10-09

**Single-core snapshot on a shared Slurm Haswell node, not an exclusive-node
or frequency-locked baseline.** [Raw results and all samples](results.json) ·
[Latency plot](latency.svg) · [Configuration overview](../report.md)

Tensor0 has the lower pooled median on **31/36 workloads**, TensorKit on 5/36.
TensorKit/Tensor0 median ratios have geometric mean **1.608×**, median **1.467×**
and range **0.742–4.643×**. These are descriptive observations, not significance
tests. Near-parity cases should not be interpreted as robust wins.

## Configuration and executable identity

Run ID: `ea043a01ff6c40e8b53f836eb0efa23c`.

- Medium: 36 workloads, two backends, warmup 2 / repeat 7 / three alternating
  paired rounds; 21 samples per record. All 72 records and 108 per-round scalar
  pairs passed, with no backend failures. All **1,512 samples** are retained.
- Intel Xeon E5-2680 v3 @ 2.50GHz (Haswell), node `n008`, two sockets × 12
  physical cores, SMT disabled (CPU0's sibling list is CPU0 only).
- Slurm allocated four physical CPUs `[0,1,2,4]` and 24 GiB. The orchestrator
  and all backend children were pinned to **CPU0**; a lightweight five-second
  sampler used CPU4. Native worker limit, Julia threads and BLAS threads were
  queried as **1** in all three backend rounds. Worker budgets do not imply
  exactly one total OS thread. XLA Eigen multithreading and OMP/MKL dynamic
  threading were disabled. Builds/precompilation did not overlap benchmarks.
- Linux 4.18.0-240.el8.x86_64 / glibc 2.28; Python 3.14.6, NumPy 2.4.6,
  JAX/JAXlib 0.10.1; Julia 1.12.5, TensorKit 0.16.5, TensorOperations 5.6.1,
  using the committed Julia Manifest. No CUDA execution.
- Source: **`333de65d10bbf4381b8189f0433927c8286c8729` plus benchmark-only
  runner changes** (native worker setup/limit and affinity metadata; Julia
  `--startup-file=no`). All 160 production source hashes matched the HEAD
  archive. Uncommitted production changes were not transferred or measured.
- Loaded extension: `src/tensor0/_native.cpython-314-x86_64-linux-gnu.so`, SHA256
  **`8b06c9daf63570fb35e1461b10323fbf3832d9d1b28fc268640735b6b7d4a45d`**.
  Release build with Rust/Cargo 1.89.0, GCC/G++ 15.3.0 and maturin 1.13.1;
  explicit final root-crate linker flags below, **no LD_PRELOAD**.

The source snapshot had no `.git`, so original runner revision/dirty fields are
`unknown`; `publication_metadata` supplies verified archived source/build
identity, dependency-lock hashes and monitoring summaries. Auxiliary evidence's
private home/run paths are omitted: the native import is repository-relative,
and the compiler prefix is a placeholder. The original medium JSON payload,
including all numeric values and samples, is unchanged; only publication
metadata was added.

## Timing and correctness

Input/space construction, planning, tracing, compilation, mandatory first call,
configured warmups and process startup are excluded. Prepared steady-state
public dispatch, execution, synchronization and scalar return are timed.
Backend/workload order alternates between paired rounds. Medians and inclusive
IQRs pool all 21 samples per record; ratios divide pooled medians, not medians
of same-round ratios. No GC-tail removal, interference filtering or selective
benchmark reruns were applied. Total orchestrator wall time **2740.34 s** includes
preparation/JIT/startup and is not kernel latency. See the
[suite timing contract](../../README.md#measurement-contract).

Each round's final timed scalar is checked without an extra contraction:
real and imaginary components separately use `rtol=atol=1e-9`. All 108 pairs
passed; maximum aggregate relative scalar difference was **4.34e-12**. SU2
uses nonuniform fusion-tree data. Publication checks verified exact pooled
concatenation, positive finite samples, workload hashes, medians and inclusive
IQRs. Scalar agreement is not exhaustive intermediate/elementwise or
adversarial validation.

## Reproduction requirements

Use the recorded source revision with the benchmark-only worker policy described
above (the public suite documents the native-worker setup), the recorded Python,
Julia and toolchain versions, and the pinned Julia Manifest. Compare support-file
hashes and loaded native identity in JSON. Python/Cargo lockfiles used for this
run were existing, unchanged files absent from the HEAD archive; their SHA256s
are recorded in JSON. A fresh dependency resolution or build on another machine
is a policy reproduction, **not guaranteed byte-identical artifact reproduction**.

The successful GCC15 build required explicit **root-crate** flags after `--`,
not just environment `RUSTFLAGS`. From a clean source checkout with an existing
CPython 3.14.6 virtual environment and maturin 1.13.1, substitute your GCC15
installation prefix (no dependency on private scripts or directories):

```sh
GCC15_PREFIX=/path/to/gcc-15.3.0
export CC="$GCC15_PREFIX/bin/gcc" CXX="$GCC15_PREFIX/bin/g++"
export CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER="$CXX"
export LD_LIBRARY_PATH="$GCC15_PREFIX/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TENSOR0_CUDA=0 CARGO_BUILD_JOBS=4 VIRTUAL_ENV="$PWD/.venv"
export RUSTFLAGS='-C link-arg=-lstdc++ -C link-arg=-lgcc'
unset LD_PRELOAD PYTHONHOME PYTHONPATH
.venv/bin/maturin develop --release --locked -- \
  -C link-arg=-Wl,--no-as-needed \
  -C link-arg=-Wl,--undefined=__cpu_features2 \
  -C link-arg=-lstdc++ -C link-arg=-lgcc
```

Select Rust/Cargo 1.89.0 on `PATH` before building. Ensure the Python runtime's
shared library is available if needed; verify `ldd` resolves `libstdc++` and
`libgcc_s` to GCC15. The flags retain the C++ runtime and extract GCC cpuinfo.
Instantiate/precompile the committed Julia environment before measuring. In a
Slurm allocation, check the allowed cpuset, topology and CPU occupancy; CPU0
is the recorded node-specific choice, not a portable allocation assumption.
Use an existing Python/Julia environment and a Julia executable on `PATH`
(or set `CROSS_BACKEND_JULIA`), then run from the repository root:

```sh
env JAX_PLATFORMS=cpu JULIA_PKG_OFFLINE=true JULIA_PKG_PRECOMPILE_AUTO=0 \
  JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  CROSS_BACKEND_THREADS=1 OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE \
  XLA_FLAGS=--xla_cpu_multi_thread_eigen=false PYTHONDONTWRITEBYTECODE=1 \
  taskset -c 0 .venv/bin/python -m benchmarks.cross_backend \
  --profile medium --threads 1 --warmup 2 --repeat 7 --rounds 3 \
  --json-output benchmarks/cross_backend/results/cpu-1t/results.json \
  --markdown benchmarks/cross_backend/results/cpu-1t/report.md \
  --plot benchmarks/cross_backend/results/cpu-1t/latency.svg
```

This command **overwrites the retained files** and reproduces the timing policy,
not historical host occupancy. The as-run wrapper recorded backend responses
and monitored the node outside timed calls without changing operations or
aggregation. This publication only reused evidence; no benchmark was rerun.

## Monitoring and limitations

572 observations over 2900.3 s showed unlimited CPU quota at all observed cgroup
levels, zero throttling deltas and zero steal time. CPU0 was 99.86% busy;
whole-node utilization was 8.42%. No competing non-system application was
observed on CPU0, but small kernel housekeeping activity was present. Peak
observed job cgroup memory was 4.43 GiB, below the 24 GiB allocation.

The node was **shared with another CPU workload**, so shared-cache/memory and
package-level interference cannot be excluded. Process snapshots are not an
exhaustive scheduler trace. CPU0 frequency snapshots ranged **2.99–3.26 GHz**
(median 3.23 GHz); frequency was not locked. There is no SMT sibling contention,
but this is still a single older Haswell node, not an interference-free formal
baseline. Plot winner labels describe pooled medians only.

The retained [four-worker result](../cpu-4t/report.md) remains a local Ryzen
workstation run with observed interference. **Different hardware and conditions
make these snapshots unsuitable for one/four-thread scaling measurements** or
causal implementation-improvement claims. The standard suite remains local
and independent.

## Results

| Workload | Backend | Status | Median ms | IQR ms | Validation |
| --- | --- | --- | ---: | ---: | --- |
| `tensor_networks.mpo.trivial.float64.d10x4x3` | `tensor0` | `ok` | 0.080076 | 0.004353 | `passed` |
| `tensor_networks.mpo.trivial.float64.d10x4x3` | `tensorkit` | `ok` | 0.098467 | 0.020384 | `passed` |
| `tensor_networks.mpo.trivial.float64.d40x4x3` | `tensor0` | `ok` | 0.252275 | 0.031195 | `passed` |
| `tensor_networks.mpo.trivial.float64.d40x4x3` | `tensorkit` | `ok` | 0.407376 | 0.291438 | `passed` |
| `tensor_networks.mpo.trivial.float64.d160x4x3` | `tensor0` | `ok` | 6.760656 | 0.852868 | `passed` |
| `tensor_networks.mpo.trivial.float64.d160x4x3` | `tensorkit` | `ok` | 11.615323 | 3.597008 | `passed` |
| `tensor_networks.mpo.trivial.float64.d100x10x10` | `tensor0` | `ok` | 21.572276 | 0.305548 | `passed` |
| `tensor_networks.mpo.trivial.float64.d100x10x10` | `tensorkit` | `ok` | 27.769457 | 6.749460 | `passed` |
| `tensor_networks.mpo.z2.float64.d10x4x4` | `tensor0` | `ok` | 0.113920 | 0.009518 | `passed` |
| `tensor_networks.mpo.z2.float64.d10x4x4` | `tensorkit` | `ok` | 0.084578 | 0.007342 | `passed` |
| `tensor_networks.mpo.z2.float64.d40x4x4` | `tensor0` | `ok` | 0.291785 | 0.020963 | `passed` |
| `tensor_networks.mpo.z2.float64.d40x4x4` | `tensorkit` | `ok` | 0.294312 | 0.141260 | `passed` |
| `tensor_networks.mpo.z2.float64.d160x4x4` | `tensor0` | `ok` | 4.916049 | 0.183972 | `passed` |
| `tensor_networks.mpo.z2.float64.d160x4x4` | `tensorkit` | `ok` | 7.216797 | 2.419562 | `passed` |
| `tensor_networks.mpo.z2.float64.d100x10x10` | `tensor0` | `ok` | 8.741745 | 0.362596 | `passed` |
| `tensor_networks.mpo.z2.float64.d100x10x10` | `tensorkit` | `ok` | 9.769693 | 0.815555 | `passed` |
| `tensor_networks.mpo.u1.float64.d40x5x3` | `tensor0` | `ok` | 0.202502 | 0.012278 | `passed` |
| `tensor_networks.mpo.u1.float64.d40x5x3` | `tensorkit` | `ok` | 0.238306 | 0.052792 | `passed` |
| `tensor_networks.mpo.u1.float64.d160x5x3` | `tensor0` | `ok` | 1.265979 | 0.037055 | `passed` |
| `tensor_networks.mpo.u1.float64.d160x5x3` | `tensorkit` | `ok` | 1.856609 | 0.637000 | `passed` |
| `tensor_networks.mpo.u1.float64.d640x5x3` | `tensor0` | `ok` | 27.398943 | 0.548083 | `passed` |
| `tensor_networks.mpo.u1.float64.d640x5x3` | `tensorkit` | `ok` | 34.452700 | 7.945791 | `passed` |
| `tensor_networks.mpo.u1.float64.d200x20x20` | `tensor0` | `ok` | 64.320834 | 4.975751 | `passed` |
| `tensor_networks.mpo.u1.float64.d200x20x20` | `tensorkit` | `ok` | 57.600335 | 11.624770 | `passed` |
| `tensor_networks.mpo.su2.float64.d40x5x3` | `tensor0` | `ok` | 0.234576 | 0.015169 | `passed` |
| `tensor_networks.mpo.su2.float64.d40x5x3` | `tensorkit` | `ok` | 0.370386 | 0.061260 | `passed` |
| `tensor_networks.pepo.trivial.float64.d3x2x2x50` | `tensor0` | `ok` | 18.100807 | 0.776411 | `passed` |
| `tensor_networks.pepo.trivial.float64.d3x2x2x50` | `tensorkit` | `ok` | 36.130416 | 15.607393 | `passed` |
| `tensor_networks.pepo.trivial.float64.d4x2x2x50` | `tensor0` | `ok` | 88.899121 | 0.565250 | `passed` |
| `tensor_networks.pepo.trivial.float64.d4x2x2x50` | `tensorkit` | `ok` | 207.649210 | 364.249003 | `passed` |
| `tensor_networks.pepo.trivial.float64.d5x2x2x50` | `tensor0` | `ok` | 262.999948 | 6.253615 | `passed` |
| `tensor_networks.pepo.trivial.float64.d5x2x2x50` | `tensorkit` | `ok` | 716.349008 | 345.379283 | `passed` |
| `tensor_networks.pepo.trivial.float64.d6x2x2x50` | `tensor0` | `ok` | 579.334166 | 7.907041 | `passed` |
| `tensor_networks.pepo.trivial.float64.d6x2x2x50` | `tensorkit` | `ok` | 1166.676822 | 138.953723 | `passed` |
| `tensor_networks.pepo.z2.float64.d4x2x2x50` | `tensor0` | `ok` | 68.083025 | 0.784799 | `passed` |
| `tensor_networks.pepo.z2.float64.d4x2x2x50` | `tensorkit` | `ok` | 62.337088 | 98.153446 | `passed` |
| `tensor_networks.pepo.z2.float64.d5x2x2x50` | `tensor0` | `ok` | 211.264222 | 2.390701 | `passed` |
| `tensor_networks.pepo.z2.float64.d5x2x2x50` | `tensorkit` | `ok` | 166.938073 | 362.693800 | `passed` |
| `tensor_networks.pepo.z2.float64.d6x2x2x50` | `tensor0` | `ok` | 458.064356 | 5.005035 | `passed` |
| `tensor_networks.pepo.z2.float64.d6x2x2x50` | `tensorkit` | `ok` | 721.244027 | 373.817626 | `passed` |
| `tensor_networks.pepo.z2.float64.d8x2x2x50` | `tensor0` | `ok` | 1485.005387 | 3.063757 | `passed` |
| `tensor_networks.pepo.z2.float64.d8x2x2x50` | `tensorkit` | `ok` | 1740.088884 | 749.660109 | `passed` |
| `tensor_networks.pepo.u1.float64.d4x2x2x100` | `tensor0` | `ok` | 96.507012 | 0.689116 | `passed` |
| `tensor_networks.pepo.u1.float64.d4x2x2x100` | `tensorkit` | `ok` | 75.558584 | 112.314929 | `passed` |
| `tensor_networks.pepo.u1.float64.d6x2x2x100` | `tensor0` | `ok` | 518.020202 | 3.188762 | `passed` |
| `tensor_networks.pepo.u1.float64.d6x2x2x100` | `tensorkit` | `ok` | 909.518153 | 103.026617 | `passed` |
| `tensor_networks.pepo.u1.float64.d8x2x2x100` | `tensor0` | `ok` | 1405.448637 | 6.569491 | `passed` |
| `tensor_networks.pepo.u1.float64.d8x2x2x100` | `tensorkit` | `ok` | 1582.028735 | 79.102474 | `passed` |
| `tensor_networks.pepo.u1.float64.d10x2x2x50` | `tensor0` | `ok` | 1881.019109 | 7.982443 | `passed` |
| `tensor_networks.pepo.u1.float64.d10x2x2x50` | `tensorkit` | `ok` | 2136.988783 | 480.415503 | `passed` |
| `tensor_networks.pepo.su2.float64.d4x2x2x100` | `tensor0` | `ok` | 25.950499 | 0.696871 | `passed` |
| `tensor_networks.pepo.su2.float64.d4x2x2x100` | `tensorkit` | `ok` | 30.161443 | 11.510791 | `passed` |
| `tensor_networks.mera.trivial.float64.d2` | `tensor0` | `ok` | 0.089531 | 0.004160 | `passed` |
| `tensor_networks.mera.trivial.float64.d2` | `tensorkit` | `ok` | 0.338983 | 0.124512 | `passed` |
| `tensor_networks.mera.trivial.float64.d3` | `tensor0` | `ok` | 0.110373 | 0.005757 | `passed` |
| `tensor_networks.mera.trivial.float64.d3` | `tensorkit` | `ok` | 0.512469 | 0.039698 | `passed` |
| `tensor_networks.mera.trivial.float64.d4` | `tensor0` | `ok` | 0.174799 | 0.007426 | `passed` |
| `tensor_networks.mera.trivial.float64.d4` | `tensorkit` | `ok` | 0.749024 | 0.114226 | `passed` |
| `tensor_networks.mera.z2.float64.d2` | `tensor0` | `ok` | 0.152102 | 0.005145 | `passed` |
| `tensor_networks.mera.z2.float64.d2` | `tensorkit` | `ok` | 0.444416 | 0.032492 | `passed` |
| `tensor_networks.mera.z2.float64.d4` | `tensor0` | `ok` | 0.342842 | 0.017831 | `passed` |
| `tensor_networks.mera.z2.float64.d4` | `tensorkit` | `ok` | 0.704519 | 0.054707 | `passed` |
| `tensor_networks.mera.z2.float64.d8` | `tensor0` | `ok` | 12.630751 | 0.244109 | `passed` |
| `tensor_networks.mera.z2.float64.d8` | `tensorkit` | `ok` | 17.719407 | 7.748599 | `passed` |
| `tensor_networks.mera.u1.float64.d4` | `tensor0` | `ok` | 0.396472 | 0.028864 | `passed` |
| `tensor_networks.mera.u1.float64.d4` | `tensorkit` | `ok` | 1.550412 | 0.097863 | `passed` |
| `tensor_networks.mera.u1.float64.d8` | `tensor0` | `ok` | 5.918574 | 0.214817 | `passed` |
| `tensor_networks.mera.u1.float64.d8` | `tensorkit` | `ok` | 20.153201 | 0.476663 | `passed` |
| `tensor_networks.mera.u1.float64.d12` | `tensor0` | `ok` | 96.392043 | 3.291547 | `passed` |
| `tensor_networks.mera.u1.float64.d12` | `tensorkit` | `ok` | 126.457500 | 23.057727 | `passed` |
| `tensor_networks.mera.su2.float64.d4` | `tensor0` | `ok` | 1.007948 | 0.028747 | `passed` |
| `tensor_networks.mera.su2.float64.d4` | `tensorkit` | `ok` | 2.099529 | 0.349734 | `passed` |

## Paired Comparison

| Workload | Baseline | Contender | Contender speedup | Validation |
| --- | --- | --- | ---: | --- |
| `tensor_networks.mpo.trivial.float64.d10x4x3` | `tensorkit` | `tensor0` | 1.230× | `passed` |
| `tensor_networks.mpo.trivial.float64.d40x4x3` | `tensorkit` | `tensor0` | 1.615× | `passed` |
| `tensor_networks.mpo.trivial.float64.d160x4x3` | `tensorkit` | `tensor0` | 1.718× | `passed` |
| `tensor_networks.mpo.trivial.float64.d100x10x10` | `tensorkit` | `tensor0` | 1.287× | `passed` |
| `tensor_networks.mpo.z2.float64.d10x4x4` | `tensorkit` | `tensor0` | 0.742× | `passed` |
| `tensor_networks.mpo.z2.float64.d40x4x4` | `tensorkit` | `tensor0` | 1.009× | `passed` |
| `tensor_networks.mpo.z2.float64.d160x4x4` | `tensorkit` | `tensor0` | 1.468× | `passed` |
| `tensor_networks.mpo.z2.float64.d100x10x10` | `tensorkit` | `tensor0` | 1.118× | `passed` |
| `tensor_networks.mpo.u1.float64.d40x5x3` | `tensorkit` | `tensor0` | 1.177× | `passed` |
| `tensor_networks.mpo.u1.float64.d160x5x3` | `tensorkit` | `tensor0` | 1.467× | `passed` |
| `tensor_networks.mpo.u1.float64.d640x5x3` | `tensorkit` | `tensor0` | 1.257× | `passed` |
| `tensor_networks.mpo.u1.float64.d200x20x20` | `tensorkit` | `tensor0` | 0.896× | `passed` |
| `tensor_networks.mpo.su2.float64.d40x5x3` | `tensorkit` | `tensor0` | 1.579× | `passed` |
| `tensor_networks.pepo.trivial.float64.d3x2x2x50` | `tensorkit` | `tensor0` | 1.996× | `passed` |
| `tensor_networks.pepo.trivial.float64.d4x2x2x50` | `tensorkit` | `tensor0` | 2.336× | `passed` |
| `tensor_networks.pepo.trivial.float64.d5x2x2x50` | `tensorkit` | `tensor0` | 2.724× | `passed` |
| `tensor_networks.pepo.trivial.float64.d6x2x2x50` | `tensorkit` | `tensor0` | 2.014× | `passed` |
| `tensor_networks.pepo.z2.float64.d4x2x2x50` | `tensorkit` | `tensor0` | 0.916× | `passed` |
| `tensor_networks.pepo.z2.float64.d5x2x2x50` | `tensorkit` | `tensor0` | 0.790× | `passed` |
| `tensor_networks.pepo.z2.float64.d6x2x2x50` | `tensorkit` | `tensor0` | 1.575× | `passed` |
| `tensor_networks.pepo.z2.float64.d8x2x2x50` | `tensorkit` | `tensor0` | 1.172× | `passed` |
| `tensor_networks.pepo.u1.float64.d4x2x2x100` | `tensorkit` | `tensor0` | 0.783× | `passed` |
| `tensor_networks.pepo.u1.float64.d6x2x2x100` | `tensorkit` | `tensor0` | 1.756× | `passed` |
| `tensor_networks.pepo.u1.float64.d8x2x2x100` | `tensorkit` | `tensor0` | 1.126× | `passed` |
| `tensor_networks.pepo.u1.float64.d10x2x2x50` | `tensorkit` | `tensor0` | 1.136× | `passed` |
| `tensor_networks.pepo.su2.float64.d4x2x2x100` | `tensorkit` | `tensor0` | 1.162× | `passed` |
| `tensor_networks.mera.trivial.float64.d2` | `tensorkit` | `tensor0` | 3.786× | `passed` |
| `tensor_networks.mera.trivial.float64.d3` | `tensorkit` | `tensor0` | 4.643× | `passed` |
| `tensor_networks.mera.trivial.float64.d4` | `tensorkit` | `tensor0` | 4.285× | `passed` |
| `tensor_networks.mera.z2.float64.d2` | `tensorkit` | `tensor0` | 2.922× | `passed` |
| `tensor_networks.mera.z2.float64.d4` | `tensorkit` | `tensor0` | 2.055× | `passed` |
| `tensor_networks.mera.z2.float64.d8` | `tensorkit` | `tensor0` | 1.403× | `passed` |
| `tensor_networks.mera.u1.float64.d4` | `tensorkit` | `tensor0` | 3.911× | `passed` |
| `tensor_networks.mera.u1.float64.d8` | `tensorkit` | `tensor0` | 3.405× | `passed` |
| `tensor_networks.mera.u1.float64.d12` | `tensorkit` | `tensor0` | 1.312× | `passed` |
| `tensor_networks.mera.su2.float64.d4` | `tensorkit` | `tensor0` | 2.083× | `passed` |

## Environment

```json
{
  "backends": {
    "tensor0": {
      "cpu_affinity": [
        0
      ],
      "jax": "0.10.1",
      "jax_backend": "cpu",
      "jax_devices": [
        "cpu:0"
      ],
      "name": "tensor0",
      "native_worker_limit": 1,
      "runtime": "Python 3.14.6",
      "thread_settings": {
        "CROSS_BACKEND_THREADS": "1",
        "MKL_DYNAMIC": "FALSE",
        "MKL_NUM_THREADS": "1",
        "OMP_DYNAMIC": "FALSE",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false"
      },
      "version": "0.0.0"
    },
    "tensorkit": {
      "blas_threads": 1,
      "name": "tensorkit",
      "runtime": "Julia 1.12.5",
      "thread_settings": {
        "CROSS_BACKEND_THREADS": "1",
        "JULIA_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1"
      },
      "threads": 1,
      "version": "0.16.5"
    }
  },
  "host": {
    "orchestrator_python": "3.14.6",
    "platform": "Linux-4.18.0-240.el8.x86_64-x86_64-with-glibc2.28",
    "processor": "x86_64",
    "repository": {
      "dirty": "unknown",
      "revision": "unknown"
    },
    "support_files_sha256": {
      "benchmarks/cross_backend/__init__.py": "8383f52bcbcc8128ca3860f612c351a39d5d90a9853f3a50360c7b0afd8f2daa",
      "benchmarks/cross_backend/__main__.py": "713eb4ea5994acb79003076333421e32b61d9ff5d8da1ce218473ada3976ad9c",
      "benchmarks/cross_backend/_orchestrator.py": "2384b0558db2bfe4a4e05b1a85ee3483a0dfcd84507b6edb799c0b29e6fe6ea0",
      "benchmarks/cross_backend/_plot.py": "e032a04019dffa6c1e5204ba4df5725f027a92cbb3267063afbe1fcb895a7ab0",
      "benchmarks/cross_backend/_protocol.py": "afcfce90a63752be8197dba1d9a23342d676f63dbc38c75eacd89585b9f64af1",
      "benchmarks/cross_backend/_report.py": "4359bb3c05029253ce32af5c60689bb0d2dee81c3c7c7079f5b2ade57eb6ec1e",
      "benchmarks/cross_backend/backends/__init__.py": "40682fd3994c4eca0dc4592d542883812d28e9b8f8cc58405720207ca2470b2f",
      "benchmarks/cross_backend/backends/tensor0/__init__.py": "db421ffa05a1a28f2b5fdc11f3fe2aebe9a0f6856edb6573c134492acdf5517b",
      "benchmarks/cross_backend/backends/tensor0/runner.py": "8bbe02b442bde1ce0dec4b6780f6bbf8e40872d1ed3b641c1f9f05ba3334bb3d",
      "benchmarks/cross_backend/backends/tensorkit/Manifest.toml": "6ecc3a543d037268c0bd6a68bbc63d40ada152f87ca1eb3bfe201001a81d38d3",
      "benchmarks/cross_backend/backends/tensorkit/Project.toml": "138eaf04c8667a26271bfdf1d46d0dc0b8c44a1667938f04ba1a6e4e8fdcf894",
      "benchmarks/cross_backend/backends/tensorkit/runner.jl": "17df93170c7eb3544ba21fde90de935e8ce8bc984c1139644aaaceca99a55f20",
      "benchmarks/cross_backend/profiles.toml": "b844397b57458fb3679f249b7a9c233f2d9d197ae97e9ce557a374c20bc7dae2",
      "benchmarks/cross_backend/protocol.schema.json": "86c0e4f4dcf40b7fbfaeb6a442897c4f482cffd37a0085b7ada4afef6c7680d3",
      "benchmarks/cross_backend/workloads/tensor_networks.toml": "e82ca16fba65dc6a4e8571c3397de7edaef7df7ad37289c1cb06d3d7c05346e0"
    },
    "thread_policy": {
      "CROSS_BACKEND_THREADS": "1",
      "JULIA_NUM_THREADS": "1",
      "MKL_DYNAMIC": "FALSE",
      "MKL_NUM_THREADS": "1",
      "OMP_DYNAMIC": "FALSE",
      "OMP_NUM_THREADS": "1",
      "OPENBLAS_NUM_THREADS": "1",
      "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false"
    }
  }
}
```

## Interpretation

- Inputs, space construction, compilation, and warmup are outside timing.
- Each backend consumes the same normalized workload and contraction order.
- Validation reuses a timed invocation's returned scalar; it does not execute the contraction again.
- Raw samples are retained in the JSON result; the table reports medians.
- `structural_only` does not assert cross-library fusion-basis equivalence.
