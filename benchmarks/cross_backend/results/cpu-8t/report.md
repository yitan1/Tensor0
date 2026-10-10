# Cross-Backend Medium CPU — 8 Workers

[Raw results and all samples](results.json) · [Latency plot](latency.svg) · [Overview](../report.md)

Tensor0 wins **34/36** pooled medians, TensorKit 2/36.
TensorKit/Tensor0 median ratio: geometric mean **2.990×**,
median **2.884×**, range 0.860–13.984×.
These are descriptive medians, not significance claims.

| Topology | Workloads | Tensor0 wins | Geometric mean ratio |
| --- | ---: | ---: | ---: |
| MPO | 13 | 11 | 2.277× |
| PEPO | 13 | 13 | 3.132× |
| MERA | 10 | 10 | 4.011× |

## Identity and policy

- Date: 2026-10-09 UTC; Slurm job 991188, partition 128G24c, node n008.
- Committed production source **`e5bd1f7383ac6675ef8d864ca7b7b7418691703c`**; all 160 production file hashes match its git archive. Only benchmark-only native worker/affinity metadata and Julia startup suppression were patched (SHA256 `526dc80ae060e25d475db552bb94c1296c184384fabe6913ee0f82bc3f97cfec`). No dirty production code was used.
- Same loaded release extension for 4t and 8t: `src/tensor0/_native.cpython-314-x86_64-linux-gnu.so`, SHA256 **`42c0e10e539d98c4fad85389ee787aab8e6d8b9afd16c4b89d34fa2bd92aec9a`**.
- Xeon E5-2680 v3, two 12-core sockets, SMT off. Allocation: 9 physical cores / 32 GiB; orchestrator and all backend children inherit **[0, 1, 2, 4, 5, 6, 7, 8]**, a same-socket subset; monitor CPU9. 4t precedes 8t, no overlap with build/tests/probes.
- Queried native worker/Julia/BLAS budgets **8** in every backend round. XLA flags `--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=8 inter_op_parallelism_threads=1`; OMP/MKL dynamic threading disabled. Budgets are upper bounds, not total OS-thread counts.
- Python 3.14.6, NumPy 2.4.6, JAX/JAXlib 0.10.1; Julia 1.12.5, TensorKit 0.16.5, TensorOperations 5.6.1; Linux 4.18 / glibc 2.28. Independent environment, target and Julia depot overlay; offline dependencies unchanged.
- Rust/Cargo 1.89.0, GCC/G++ 15.3.0, maturin 1.13.1. GCC15 linker, `RUSTFLAGS='-C link-arg=-lstdc++ -C link-arg=-lgcc'`; final root-crate flags `-C link-arg=-Wl,--no-as-needed -C link-arg=-Wl,--undefined=__cpu_features2 -C link-arg=-lstdc++ -C link-arg=-lgcc`. C++/GCC libraries resolved to GCC15; **no LD_PRELOAD**.

## Actual resource gate and monitoring

Before formal sampling, all readable cgroup ancestors' CPU quota/cpuset/memory limits were captured. Allocation contains eight distinct same-socket physical cores. The 8-second synchronous independently pinned CPU-bound probe delivered **7.981 core-equivalents**, each worker>=99.7% CPU/wall (threshold 90% per worker and aggregate). No throttling or steal was observed.

Tensor0 public native independent-output 4096×4096 F64 reduction delivered **5.685 CPU/wall** with 8 substantial worker-thread CPU deltas; compiled HLO confirms native execution. TensorMap MERA SU2 smoke also passed. TensorKit 1536×1536 F64 multiplication delivered **7.938 CPU/wall**, with queried Julia/BLAS counts 8. Native/BLAS gate threshold was 1.5 CPU/wall and >=2 threads each>0.1 CPU seconds. Full per-thread measurements are retained in JSON publication metadata. Related new-scheduling/FFI/AD regressions: **142 passed**.

A lightweight 2-second sampler ran on allocated CPU9. Across smoke+medium, observed measurement-core busy fractions: cpu0: 11.7%; cpu1: 16.5%; cpu2: 24.2%; cpu4: 17.8%; cpu5: 21.7%; cpu6: 26.6%; cpu7: 28.8%; cpu8: 15.6%. No quota throttling or steal observed; peak job cgroup memory **4.09 GiB**. Frequency snapshots 2.80–3.26 GHz, not locked. JSON retains backend CPU time/elapsed including startup/JIT; these are **not** steady-state kernel occupancies. Shared cache/memory interference remains possible.

Native schedules proven independent disjoint outputs, shares immutable generated metadata and conservatively falls back for unknown/overlapping layouts. Small work remains serial. Memory-bandwidth and task granularity limit occupancy; passing gate does not imply every medium case reaches 100% utilization.

## Measurement and validation

Medium: 36 workloads × 2 backends; warmup 2 / repeat 7 / rounds 3 with alternating backend/workload order. Mandatory compilation call, input construction and startup are excluded; prepared execution and synchronization are timed. All **72 records**, **108 per-round scalar pairs**, **1,512 unfiltered samples** passed (`rtol=atol=1e-9`). Maximum aggregate relative scalar error **4.34e-12**. All 21 samples per record and pooled medians/inclusive IQRs were checked exactly against raw responses. Smoke: 4 records/24 samples/4 scalar pairs also passed. No selective reruns or outlier filtering. Medium orchestrator elapsed **2347.54 s**, including preparation, is not operation latency.

## Reproduction and limits

From an environment matching the versions above, with Julia available and the committed TensorKit manifest instantiated:

```sh
# Within a Slurm allocation >=8 distinct physical cores, 32GiB, >=3h:
# first verify all cgroup ancestors and topology, then run bounded pinned
# CPU-throughput and native/BLAS per-thread utilization gates as described above.
taskset -c 0,1,2,4,5,6,7,8 python -m benchmarks.cross_backend \
  --profile smoke --threads 8 --json-output smoke.json
taskset -c 0,1,2,4,5,6,7,8 python -m benchmarks.cross_backend \
  --profile medium --threads 8 --warmup 2 --repeat 7 --rounds 3 \
  --json-output results.json --markdown report.md --plot latency.svg
```

Use your allocation's verified CPU IDs, not these IDs blindly. Build the revision above with the release/compiler settings listed, and use the suite's native worker setup/metadata and Julia `--startup-file=no` behavior. The recorded benchmark-only patch hash distinguishes the runner changes from committed production code. See [suite contract](../../README.md#measurement-contract). Public reproduction does not depend on private scripts or paths.

The 4t/8t snapshots use the same revision, node, artifact and environment, but are sequential rather than randomized scaling trials. The old 1t result is **333de65**, not this revision: do not infer 1→4→8 same-version scaling. The standard suite is unchanged. The source archive lacks `.git`, hence raw revision fields are unknown; publication metadata provides verified identity. Private absolute paths were omitted only from auxiliary publication provenance; all runner numeric fields and samples are unchanged.

## Benchmark-only patch for the archived revision

Apply with `git apply` before the release build and comparison commands above.

```diff
diff --git a/benchmarks/cross_backend/_orchestrator.py b/benchmarks/cross_backend/_orchestrator.py
index 85ae0ee..f773355 100644
--- a/benchmarks/cross_backend/_orchestrator.py
+++ b/benchmarks/cross_backend/_orchestrator.py
@@ -60,6 +60,7 @@ def _backend_command(name: str) -> list[str]:
             raise FileNotFoundError("julia executable was not found")
         return [
             julia,
+            "--startup-file=no",
             f"--project={_TENSORKIT_ROOT}",
             str(_TENSORKIT_ROOT / "runner.jl"),
         ]
diff --git a/benchmarks/cross_backend/backends/tensor0/runner.py b/benchmarks/cross_backend/backends/tensor0/runner.py
index 94aab57..27b7d3d 100644
--- a/benchmarks/cross_backend/backends/tensor0/runner.py
+++ b/benchmarks/cross_backend/backends/tensor0/runner.py
@@ -33,6 +33,8 @@ from tensor0 import (
     space,
 )

+from tensor0._stride import get_num_threads, set_num_threads
+
 from ..._protocol import SCHEMA_VERSION, validate_request


@@ -291,6 +293,13 @@ def _backend_metadata() -> dict[str, Any]:
         "jax": getattr(jax, "__version__", "unknown"),
         "jax_backend": jax.default_backend(),
         "jax_devices": [str(device) for device in jax.devices()],
+        "native_worker_limit": (
+            get_num_threads() if jax.default_backend() == "cpu" else None
+        ),
+        "cpu_affinity": (
+            sorted(os.sched_getaffinity(0))
+            if hasattr(os, "sched_getaffinity") else None
+        ),
         "thread_settings": {
             name: os.environ.get(name, "unset")
             for name in (
@@ -308,6 +317,8 @@ def _backend_metadata() -> dict[str, Any]:

 def execute(request: dict[str, Any]) -> dict[str, Any]:
     validate_request(request)
+    if jax.default_backend() == "cpu":
+        set_num_threads(int(os.environ.get("CROSS_BACKEND_THREADS", "1")))
     measurement = request["measurement"]
     results = []
     with jax.enable_x64(True):
```

## Per-workload timings

| Workload | Backend | Status | Median ms | IQR ms | Validation |
| --- | --- | --- | ---: | ---: | --- |
| `tensor_networks.mpo.trivial.float64.d10x4x3` | `tensor0` | `ok` | 0.032853 | 0.003580 | `passed` |
| `tensor_networks.mpo.trivial.float64.d10x4x3` | `tensorkit` | `ok` | 0.097129 | 0.027393 | `passed` |
| `tensor_networks.mpo.trivial.float64.d40x4x3` | `tensor0` | `ok` | 0.230948 | 0.042283 | `passed` |
| `tensor_networks.mpo.trivial.float64.d40x4x3` | `tensorkit` | `ok` | 0.518292 | 0.163968 | `passed` |
| `tensor_networks.mpo.trivial.float64.d160x4x3` | `tensor0` | `ok` | 1.913555 | 0.069118 | `passed` |
| `tensor_networks.mpo.trivial.float64.d160x4x3` | `tensorkit` | `ok` | 7.945964 | 14.762957 | `passed` |
| `tensor_networks.mpo.trivial.float64.d100x10x10` | `tensor0` | `ok` | 5.064349 | 0.078325 | `passed` |
| `tensor_networks.mpo.trivial.float64.d100x10x10` | `tensorkit` | `ok` | 24.101810 | 37.309266 | `passed` |
| `tensor_networks.mpo.z2.float64.d10x4x4` | `tensor0` | `ok` | 0.061766 | 0.002887 | `passed` |
| `tensor_networks.mpo.z2.float64.d10x4x4` | `tensorkit` | `ok` | 0.079431 | 0.008913 | `passed` |
| `tensor_networks.mpo.z2.float64.d40x4x4` | `tensor0` | `ok` | 0.243516 | 0.011673 | `passed` |
| `tensor_networks.mpo.z2.float64.d40x4x4` | `tensorkit` | `ok` | 0.376763 | 0.041725 | `passed` |
| `tensor_networks.mpo.z2.float64.d160x4x4` | `tensor0` | `ok` | 2.660997 | 0.025214 | `passed` |
| `tensor_networks.mpo.z2.float64.d160x4x4` | `tensorkit` | `ok` | 2.289367 | 0.471564 | `passed` |
| `tensor_networks.mpo.z2.float64.d100x10x10` | `tensor0` | `ok` | 3.717059 | 0.089046 | `passed` |
| `tensor_networks.mpo.z2.float64.d100x10x10` | `tensorkit` | `ok` | 18.916389 | 5.917593 | `passed` |
| `tensor_networks.mpo.u1.float64.d40x5x3` | `tensor0` | `ok` | 0.146086 | 0.005244 | `passed` |
| `tensor_networks.mpo.u1.float64.d40x5x3` | `tensorkit` | `ok` | 0.218480 | 0.013833 | `passed` |
| `tensor_networks.mpo.u1.float64.d160x5x3` | `tensor0` | `ok` | 1.519301 | 0.033936 | `passed` |
| `tensor_networks.mpo.u1.float64.d160x5x3` | `tensorkit` | `ok` | 1.309220 | 1.156331 | `passed` |
| `tensor_networks.mpo.u1.float64.d640x5x3` | `tensor0` | `ok` | 11.523907 | 1.089202 | `passed` |
| `tensor_networks.mpo.u1.float64.d640x5x3` | `tensorkit` | `ok` | 38.455776 | 51.508283 | `passed` |
| `tensor_networks.mpo.u1.float64.d200x20x20` | `tensor0` | `ok` | 24.856264 | 1.723692 | `passed` |
| `tensor_networks.mpo.u1.float64.d200x20x20` | `tensorkit` | `ok` | 105.482116 | 96.398417 | `passed` |
| `tensor_networks.mpo.su2.float64.d40x5x3` | `tensor0` | `ok` | 0.163585 | 0.012075 | `passed` |
| `tensor_networks.mpo.su2.float64.d40x5x3` | `tensorkit` | `ok` | 0.346516 | 0.030362 | `passed` |
| `tensor_networks.pepo.trivial.float64.d3x2x2x50` | `tensor0` | `ok` | 5.033100 | 0.074932 | `passed` |
| `tensor_networks.pepo.trivial.float64.d3x2x2x50` | `tensorkit` | `ok` | 70.382641 | 123.388992 | `passed` |
| `tensor_networks.pepo.trivial.float64.d4x2x2x50` | `tensor0` | `ok` | 22.926417 | 0.181152 | `passed` |
| `tensor_networks.pepo.trivial.float64.d4x2x2x50` | `tensorkit` | `ok` | 205.380842 | 170.864074 | `passed` |
| `tensor_networks.pepo.trivial.float64.d5x2x2x50` | `tensor0` | `ok` | 67.459483 | 0.747149 | `passed` |
| `tensor_networks.pepo.trivial.float64.d5x2x2x50` | `tensorkit` | `ok` | 362.454175 | 69.005361 | `passed` |
| `tensor_networks.pepo.trivial.float64.d6x2x2x50` | `tensor0` | `ok` | 140.780102 | 1.091162 | `passed` |
| `tensor_networks.pepo.trivial.float64.d6x2x2x50` | `tensorkit` | `ok` | 542.072512 | 143.349353 | `passed` |
| `tensor_networks.pepo.z2.float64.d4x2x2x50` | `tensor0` | `ok` | 20.031910 | 4.178871 | `passed` |
| `tensor_networks.pepo.z2.float64.d4x2x2x50` | `tensorkit` | `ok` | 32.700381 | 9.426627 | `passed` |
| `tensor_networks.pepo.z2.float64.d5x2x2x50` | `tensor0` | `ok` | 73.219714 | 0.578852 | `passed` |
| `tensor_networks.pepo.z2.float64.d5x2x2x50` | `tensorkit` | `ok` | 269.013894 | 97.009561 | `passed` |
| `tensor_networks.pepo.z2.float64.d6x2x2x50` | `tensor0` | `ok` | 160.947696 | 1.420329 | `passed` |
| `tensor_networks.pepo.z2.float64.d6x2x2x50` | `tensorkit` | `ok` | 452.550141 | 80.275967 | `passed` |
| `tensor_networks.pepo.z2.float64.d8x2x2x50` | `tensor0` | `ok` | 518.741534 | 1.965334 | `passed` |
| `tensor_networks.pepo.z2.float64.d8x2x2x50` | `tensorkit` | `ok` | 907.174240 | 62.690277 | `passed` |
| `tensor_networks.pepo.u1.float64.d4x2x2x100` | `tensor0` | `ok` | 47.063360 | 0.494355 | `passed` |
| `tensor_networks.pepo.u1.float64.d4x2x2x100` | `tensorkit` | `ok` | 98.448077 | 111.092888 | `passed` |
| `tensor_networks.pepo.u1.float64.d6x2x2x100` | `tensor0` | `ok` | 261.641812 | 1.629608 | `passed` |
| `tensor_networks.pepo.u1.float64.d6x2x2x100` | `tensorkit` | `ok` | 562.497656 | 96.516751 | `passed` |
| `tensor_networks.pepo.u1.float64.d8x2x2x100` | `tensor0` | `ok` | 522.992066 | 4.887981 | `passed` |
| `tensor_networks.pepo.u1.float64.d8x2x2x100` | `tensorkit` | `ok` | 974.252312 | 84.267209 | `passed` |
| `tensor_networks.pepo.u1.float64.d10x2x2x50` | `tensor0` | `ok` | 652.840704 | 4.022312 | `passed` |
| `tensor_networks.pepo.u1.float64.d10x2x2x50` | `tensorkit` | `ok` | 1234.423047 | 52.714547 | `passed` |
| `tensor_networks.pepo.su2.float64.d4x2x2x100` | `tensor0` | `ok` | 14.708215 | 1.270502 | `passed` |
| `tensor_networks.pepo.su2.float64.d4x2x2x100` | `tensorkit` | `ok` | 33.967500 | 57.231338 | `passed` |
| `tensor_networks.mera.trivial.float64.d2` | `tensor0` | `ok` | 0.039537 | 0.004055 | `passed` |
| `tensor_networks.mera.trivial.float64.d2` | `tensorkit` | `ok` | 0.335633 | 0.065967 | `passed` |
| `tensor_networks.mera.trivial.float64.d3` | `tensor0` | `ok` | 0.073257 | 0.013244 | `passed` |
| `tensor_networks.mera.trivial.float64.d3` | `tensorkit` | `ok` | 0.486060 | 0.027831 | `passed` |
| `tensor_networks.mera.trivial.float64.d4` | `tensor0` | `ok` | 0.135093 | 0.015899 | `passed` |
| `tensor_networks.mera.trivial.float64.d4` | `tensorkit` | `ok` | 0.723665 | 0.078433 | `passed` |
| `tensor_networks.mera.z2.float64.d2` | `tensor0` | `ok` | 0.110822 | 0.005758 | `passed` |
| `tensor_networks.mera.z2.float64.d2` | `tensorkit` | `ok` | 0.414244 | 0.021077 | `passed` |
| `tensor_networks.mera.z2.float64.d4` | `tensor0` | `ok` | 0.255722 | 0.012811 | `passed` |
| `tensor_networks.mera.z2.float64.d4` | `tensorkit` | `ok` | 0.713152 | 0.020147 | `passed` |
| `tensor_networks.mera.z2.float64.d8` | `tensor0` | `ok` | 4.791991 | 0.059259 | `passed` |
| `tensor_networks.mera.z2.float64.d8` | `tensorkit` | `ok` | 11.302292 | 9.255326 | `passed` |
| `tensor_networks.mera.u1.float64.d4` | `tensor0` | `ok` | 0.352042 | 0.047798 | `passed` |
| `tensor_networks.mera.u1.float64.d4` | `tensorkit` | `ok` | 1.566411 | 0.023150 | `passed` |
| `tensor_networks.mera.u1.float64.d8` | `tensor0` | `ok` | 4.253516 | 0.427693 | `passed` |
| `tensor_networks.mera.u1.float64.d8` | `tensorkit` | `ok` | 20.899356 | 0.988339 | `passed` |
| `tensor_networks.mera.u1.float64.d12` | `tensor0` | `ok` | 34.225485 | 0.517676 | `passed` |
| `tensor_networks.mera.u1.float64.d12` | `tensorkit` | `ok` | 112.768356 | 55.861694 | `passed` |
| `tensor_networks.mera.su2.float64.d4` | `tensor0` | `ok` | 1.039811 | 0.029601 | `passed` |
| `tensor_networks.mera.su2.float64.d4` | `tensorkit` | `ok` | 2.097091 | 0.207602 | `passed` |

## Paired Comparison

| Workload | Baseline | Contender | Contender speedup | Validation |
| --- | --- | --- | ---: | --- |
| `tensor_networks.mpo.trivial.float64.d10x4x3` | `tensorkit` | `tensor0` | 2.956× | `passed` |
| `tensor_networks.mpo.trivial.float64.d40x4x3` | `tensorkit` | `tensor0` | 2.244× | `passed` |
| `tensor_networks.mpo.trivial.float64.d160x4x3` | `tensorkit` | `tensor0` | 4.152× | `passed` |
| `tensor_networks.mpo.trivial.float64.d100x10x10` | `tensorkit` | `tensor0` | 4.759× | `passed` |
| `tensor_networks.mpo.z2.float64.d10x4x4` | `tensorkit` | `tensor0` | 1.286× | `passed` |
| `tensor_networks.mpo.z2.float64.d40x4x4` | `tensorkit` | `tensor0` | 1.547× | `passed` |
| `tensor_networks.mpo.z2.float64.d160x4x4` | `tensorkit` | `tensor0` | 0.860× | `passed` |
| `tensor_networks.mpo.z2.float64.d100x10x10` | `tensorkit` | `tensor0` | 5.089× | `passed` |
| `tensor_networks.mpo.u1.float64.d40x5x3` | `tensorkit` | `tensor0` | 1.496× | `passed` |
| `tensor_networks.mpo.u1.float64.d160x5x3` | `tensorkit` | `tensor0` | 0.862× | `passed` |
| `tensor_networks.mpo.u1.float64.d640x5x3` | `tensorkit` | `tensor0` | 3.337× | `passed` |
| `tensor_networks.mpo.u1.float64.d200x20x20` | `tensorkit` | `tensor0` | 4.244× | `passed` |
| `tensor_networks.mpo.su2.float64.d40x5x3` | `tensorkit` | `tensor0` | 2.118× | `passed` |
| `tensor_networks.pepo.trivial.float64.d3x2x2x50` | `tensorkit` | `tensor0` | 13.984× | `passed` |
| `tensor_networks.pepo.trivial.float64.d4x2x2x50` | `tensorkit` | `tensor0` | 8.958× | `passed` |
| `tensor_networks.pepo.trivial.float64.d5x2x2x50` | `tensorkit` | `tensor0` | 5.373× | `passed` |
| `tensor_networks.pepo.trivial.float64.d6x2x2x50` | `tensorkit` | `tensor0` | 3.850× | `passed` |
| `tensor_networks.pepo.z2.float64.d4x2x2x50` | `tensorkit` | `tensor0` | 1.632× | `passed` |
| `tensor_networks.pepo.z2.float64.d5x2x2x50` | `tensorkit` | `tensor0` | 3.674× | `passed` |
| `tensor_networks.pepo.z2.float64.d6x2x2x50` | `tensorkit` | `tensor0` | 2.812× | `passed` |
| `tensor_networks.pepo.z2.float64.d8x2x2x50` | `tensorkit` | `tensor0` | 1.749× | `passed` |
| `tensor_networks.pepo.u1.float64.d4x2x2x100` | `tensorkit` | `tensor0` | 2.092× | `passed` |
| `tensor_networks.pepo.u1.float64.d6x2x2x100` | `tensorkit` | `tensor0` | 2.150× | `passed` |
| `tensor_networks.pepo.u1.float64.d8x2x2x100` | `tensorkit` | `tensor0` | 1.863× | `passed` |
| `tensor_networks.pepo.u1.float64.d10x2x2x50` | `tensorkit` | `tensor0` | 1.891× | `passed` |
| `tensor_networks.pepo.su2.float64.d4x2x2x100` | `tensorkit` | `tensor0` | 2.309× | `passed` |
| `tensor_networks.mera.trivial.float64.d2` | `tensorkit` | `tensor0` | 8.489× | `passed` |
| `tensor_networks.mera.trivial.float64.d3` | `tensorkit` | `tensor0` | 6.635× | `passed` |
| `tensor_networks.mera.trivial.float64.d4` | `tensorkit` | `tensor0` | 5.357× | `passed` |
| `tensor_networks.mera.z2.float64.d2` | `tensorkit` | `tensor0` | 3.738× | `passed` |
| `tensor_networks.mera.z2.float64.d4` | `tensorkit` | `tensor0` | 2.789× | `passed` |
| `tensor_networks.mera.z2.float64.d8` | `tensorkit` | `tensor0` | 2.359× | `passed` |
| `tensor_networks.mera.u1.float64.d4` | `tensorkit` | `tensor0` | 4.450× | `passed` |
| `tensor_networks.mera.u1.float64.d8` | `tensorkit` | `tensor0` | 4.913× | `passed` |
| `tensor_networks.mera.u1.float64.d12` | `tensorkit` | `tensor0` | 3.295× | `passed` |
| `tensor_networks.mera.su2.float64.d4` | `tensorkit` | `tensor0` | 2.017× | `passed` |
