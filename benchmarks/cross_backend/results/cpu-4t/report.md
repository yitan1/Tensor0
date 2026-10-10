# Cross-Backend Medium CPU — 4 Workers

[Raw results and all samples](results.json) · [Latency plot](latency.svg) · [Overview](../report.md)

Tensor0 wins **36/36** pooled medians, TensorKit 0/36.
TensorKit/Tensor0 median ratio: geometric mean **2.553×**,
median **2.271×**, range 1.098–9.135×.
These are descriptive medians, not significance claims.

| Topology | Workloads | Tensor0 wins | Geometric mean ratio |
| --- | ---: | ---: | ---: |
| MPO | 13 | 13 | 1.962× |
| PEPO | 13 | 13 | 2.333× |
| MERA | 10 | 10 | 4.042× |

## Identity and policy

- Date: 2026-10-09 UTC; Slurm job 991188, partition 128G24c, node n008.
- Committed production source **`e5bd1f7383ac6675ef8d864ca7b7b7418691703c`**; all 160 production file hashes match its git archive. Only benchmark-only native worker/affinity metadata and Julia startup suppression were patched (SHA256 `526dc80ae060e25d475db552bb94c1296c184384fabe6913ee0f82bc3f97cfec`). No dirty production code was used.
- Same loaded release extension for 4t and 8t: `src/tensor0/_native.cpython-314-x86_64-linux-gnu.so`, SHA256 **`42c0e10e539d98c4fad85389ee787aab8e6d8b9afd16c4b89d34fa2bd92aec9a`**.
- Xeon E5-2680 v3, two 12-core sockets, SMT off. Allocation: 9 physical cores / 32 GiB; orchestrator and all backend children inherit **[0, 1, 2, 4]**, a same-socket subset; monitor CPU9. 4t precedes 8t, no overlap with build/tests/probes.
- Queried native worker/Julia/BLAS budgets **4** in every backend round. XLA flags `--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=4 inter_op_parallelism_threads=1`; OMP/MKL dynamic threading disabled. Budgets are upper bounds, not total OS-thread counts.
- Python 3.14.6, NumPy 2.4.6, JAX/JAXlib 0.10.1; Julia 1.12.5, TensorKit 0.16.5, TensorOperations 5.6.1; Linux 4.18 / glibc 2.28. Independent environment, target and Julia depot overlay; offline dependencies unchanged.
- Rust/Cargo 1.89.0, GCC/G++ 15.3.0, maturin 1.13.1. GCC15 linker, `RUSTFLAGS='-C link-arg=-lstdc++ -C link-arg=-lgcc'`; final root-crate flags `-C link-arg=-Wl,--no-as-needed -C link-arg=-Wl,--undefined=__cpu_features2 -C link-arg=-lstdc++ -C link-arg=-lgcc`. C++/GCC libraries resolved to GCC15; **no LD_PRELOAD**.

## Actual resource gate and monitoring

Before formal sampling, all readable cgroup ancestors' CPU quota/cpuset/memory limits were captured. Allocation contains eight distinct same-socket physical cores. The 8-second synchronous independently pinned CPU-bound probe delivered **3.991 core-equivalents**, each worker>=99.7% CPU/wall (threshold 90% per worker and aggregate). No throttling or steal was observed.

Tensor0 public native independent-output 4096×4096 F64 reduction delivered **3.096 CPU/wall** with 4 substantial worker-thread CPU deltas; compiled HLO confirms native execution. TensorMap MERA SU2 smoke also passed. TensorKit 1536×1536 F64 multiplication delivered **3.976 CPU/wall**, with queried Julia/BLAS counts 4. Native/BLAS gate threshold was 1.5 CPU/wall and >=2 threads each>0.1 CPU seconds. Full per-thread measurements are retained in JSON publication metadata. Related new-scheduling/FFI/AD regressions: **142 passed**.

A lightweight 2-second sampler ran on allocated CPU9. Across smoke+medium, observed measurement-core busy fractions: cpu0: 37.5%; cpu1: 19.1%; cpu2: 21.1%; cpu4: 54.8%. No quota throttling or steal observed; peak job cgroup memory **4.07 GiB**. Frequency snapshots 2.80–3.26 GHz, not locked. JSON retains backend CPU time/elapsed including startup/JIT; these are **not** steady-state kernel occupancies. Shared cache/memory interference remains possible.

Native schedules proven independent disjoint outputs, shares immutable generated metadata and conservatively falls back for unknown/overlapping layouts. Small work remains serial. Memory-bandwidth and task granularity limit occupancy; passing gate does not imply every medium case reaches 100% utilization.

## Measurement and validation

Medium: 36 workloads × 2 backends; warmup 2 / repeat 7 / rounds 3 with alternating backend/workload order. Mandatory compilation call, input construction and startup are excluded; prepared execution and synchronization are timed. All **72 records**, **108 per-round scalar pairs**, **1,512 unfiltered samples** passed (`rtol=atol=1e-9`). Maximum aggregate relative scalar error **4.34e-12**. All 21 samples per record and pooled medians/inclusive IQRs were checked exactly against raw responses. Smoke: 4 records/24 samples/4 scalar pairs also passed. No selective reruns or outlier filtering. Medium orchestrator elapsed **2420.96 s**, including preparation, is not operation latency.

## Reproduction and limits

From an environment matching the versions above, with Julia available and the committed TensorKit manifest instantiated:

```sh
# Within a Slurm allocation >=8 distinct physical cores, 32GiB, >=3h:
# first verify all cgroup ancestors and topology, then run bounded pinned
# CPU-throughput and native/BLAS per-thread utilization gates as described above.
taskset -c 0,1,2,4 python -m benchmarks.cross_backend \
  --profile smoke --threads 4 --json-output smoke.json
taskset -c 0,1,2,4 python -m benchmarks.cross_backend \
  --profile medium --threads 4 --warmup 2 --repeat 7 --rounds 3 \
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
| `tensor_networks.mpo.trivial.float64.d10x4x3` | `tensor0` | `ok` | 0.032435 | 0.004660 | `passed` |
| `tensor_networks.mpo.trivial.float64.d10x4x3` | `tensorkit` | `ok` | 0.091694 | 0.032351 | `passed` |
| `tensor_networks.mpo.trivial.float64.d40x4x3` | `tensor0` | `ok` | 0.224351 | 0.019545 | `passed` |
| `tensor_networks.mpo.trivial.float64.d40x4x3` | `tensorkit` | `ok` | 0.467118 | 0.203427 | `passed` |
| `tensor_networks.mpo.trivial.float64.d160x4x3` | `tensor0` | `ok` | 2.902158 | 0.080610 | `passed` |
| `tensor_networks.mpo.trivial.float64.d160x4x3` | `tensorkit` | `ok` | 7.702536 | 2.257104 | `passed` |
| `tensor_networks.mpo.trivial.float64.d100x10x10` | `tensor0` | `ok` | 7.737901 | 0.079550 | `passed` |
| `tensor_networks.mpo.trivial.float64.d100x10x10` | `tensorkit` | `ok` | 17.965415 | 44.255733 | `passed` |
| `tensor_networks.mpo.z2.float64.d10x4x4` | `tensor0` | `ok` | 0.061354 | 0.003161 | `passed` |
| `tensor_networks.mpo.z2.float64.d10x4x4` | `tensorkit` | `ok` | 0.090165 | 0.012448 | `passed` |
| `tensor_networks.mpo.z2.float64.d40x4x4` | `tensor0` | `ok` | 0.240570 | 0.013057 | `passed` |
| `tensor_networks.mpo.z2.float64.d40x4x4` | `tensorkit` | `ok` | 0.336156 | 0.130971 | `passed` |
| `tensor_networks.mpo.z2.float64.d160x4x4` | `tensor0` | `ok` | 2.672580 | 0.036553 | `passed` |
| `tensor_networks.mpo.z2.float64.d160x4x4` | `tensorkit` | `ok` | 2.933245 | 0.737497 | `passed` |
| `tensor_networks.mpo.z2.float64.d100x10x10` | `tensor0` | `ok` | 4.486240 | 0.095751 | `passed` |
| `tensor_networks.mpo.z2.float64.d100x10x10` | `tensorkit` | `ok` | 14.170552 | 1.421975 | `passed` |
| `tensor_networks.mpo.u1.float64.d40x5x3` | `tensor0` | `ok` | 0.144708 | 0.003411 | `passed` |
| `tensor_networks.mpo.u1.float64.d40x5x3` | `tensorkit` | `ok` | 0.251932 | 0.032116 | `passed` |
| `tensor_networks.mpo.u1.float64.d160x5x3` | `tensor0` | `ok` | 1.541602 | 0.048611 | `passed` |
| `tensor_networks.mpo.u1.float64.d160x5x3` | `tensorkit` | `ok` | 2.034164 | 0.551572 | `passed` |
| `tensor_networks.mpo.u1.float64.d640x5x3` | `tensor0` | `ok` | 14.918032 | 1.120243 | `passed` |
| `tensor_networks.mpo.u1.float64.d640x5x3` | `tensorkit` | `ok` | 32.918384 | 10.330621 | `passed` |
| `tensor_networks.mpo.u1.float64.d200x20x20` | `tensor0` | `ok` | 30.949925 | 2.529988 | `passed` |
| `tensor_networks.mpo.u1.float64.d200x20x20` | `tensorkit` | `ok` | 68.102144 | 46.093095 | `passed` |
| `tensor_networks.mpo.su2.float64.d40x5x3` | `tensor0` | `ok` | 0.156329 | 0.020877 | `passed` |
| `tensor_networks.mpo.su2.float64.d40x5x3` | `tensorkit` | `ok` | 0.347197 | 0.044065 | `passed` |
| `tensor_networks.pepo.trivial.float64.d3x2x2x50` | `tensor0` | `ok` | 7.602404 | 0.153216 | `passed` |
| `tensor_networks.pepo.trivial.float64.d3x2x2x50` | `tensorkit` | `ok` | 29.453908 | 44.441227 | `passed` |
| `tensor_networks.pepo.trivial.float64.d4x2x2x50` | `tensor0` | `ok` | 35.167068 | 0.231559 | `passed` |
| `tensor_networks.pepo.trivial.float64.d4x2x2x50` | `tensorkit` | `ok` | 243.242703 | 130.561227 | `passed` |
| `tensor_networks.pepo.trivial.float64.d5x2x2x50` | `tensor0` | `ok` | 101.311484 | 0.563061 | `passed` |
| `tensor_networks.pepo.trivial.float64.d5x2x2x50` | `tensorkit` | `ok` | 374.647749 | 65.324871 | `passed` |
| `tensor_networks.pepo.trivial.float64.d6x2x2x50` | `tensor0` | `ok` | 218.261660 | 1.407212 | `passed` |
| `tensor_networks.pepo.trivial.float64.d6x2x2x50` | `tensorkit` | `ok` | 645.739794 | 116.070223 | `passed` |
| `tensor_networks.pepo.z2.float64.d4x2x2x50` | `tensor0` | `ok` | 31.287665 | 0.606935 | `passed` |
| `tensor_networks.pepo.z2.float64.d4x2x2x50` | `tensorkit` | `ok` | 36.259192 | 8.636147 | `passed` |
| `tensor_networks.pepo.z2.float64.d5x2x2x50` | `tensor0` | `ok` | 93.750095 | 1.092448 | `passed` |
| `tensor_networks.pepo.z2.float64.d5x2x2x50` | `tensorkit` | `ok` | 234.679672 | 165.049036 | `passed` |
| `tensor_networks.pepo.z2.float64.d6x2x2x50` | `tensor0` | `ok` | 199.956700 | 1.068304 | `passed` |
| `tensor_networks.pepo.z2.float64.d6x2x2x50` | `tensorkit` | `ok` | 504.125755 | 66.850247 | `passed` |
| `tensor_networks.pepo.z2.float64.d8x2x2x50` | `tensor0` | `ok` | 641.436929 | 4.313718 | `passed` |
| `tensor_networks.pepo.z2.float64.d8x2x2x50` | `tensorkit` | `ok` | 1031.869541 | 56.421152 | `passed` |
| `tensor_networks.pepo.u1.float64.d4x2x2x100` | `tensor0` | `ok` | 57.630598 | 0.377463 | `passed` |
| `tensor_networks.pepo.u1.float64.d4x2x2x100` | `tensorkit` | `ok` | 121.369071 | 141.430016 | `passed` |
| `tensor_networks.pepo.u1.float64.d6x2x2x100` | `tensor0` | `ok` | 352.427646 | 2.550930 | `passed` |
| `tensor_networks.pepo.u1.float64.d6x2x2x100` | `tensorkit` | `ok` | 635.319470 | 57.481092 | `passed` |
| `tensor_networks.pepo.u1.float64.d8x2x2x100` | `tensor0` | `ok` | 653.370024 | 4.545873 | `passed` |
| `tensor_networks.pepo.u1.float64.d8x2x2x100` | `tensorkit` | `ok` | 1166.545408 | 85.973276 | `passed` |
| `tensor_networks.pepo.u1.float64.d10x2x2x50` | `tensor0` | `ok` | 836.663685 | 6.195552 | `passed` |
| `tensor_networks.pepo.u1.float64.d10x2x2x50` | `tensorkit` | `ok` | 1408.763344 | 85.870548 | `passed` |
| `tensor_networks.pepo.su2.float64.d4x2x2x100` | `tensor0` | `ok` | 17.294573 | 0.427729 | `passed` |
| `tensor_networks.pepo.su2.float64.d4x2x2x100` | `tensorkit` | `ok` | 26.574782 | 34.003141 | `passed` |
| `tensor_networks.mera.trivial.float64.d2` | `tensor0` | `ok` | 0.038621 | 0.003314 | `passed` |
| `tensor_networks.mera.trivial.float64.d2` | `tensorkit` | `ok` | 0.352805 | 0.058635 | `passed` |
| `tensor_networks.mera.trivial.float64.d3` | `tensor0` | `ok` | 0.063565 | 0.002271 | `passed` |
| `tensor_networks.mera.trivial.float64.d3` | `tensorkit` | `ok` | 0.505456 | 0.037849 | `passed` |
| `tensor_networks.mera.trivial.float64.d4` | `tensor0` | `ok` | 0.128414 | 0.005533 | `passed` |
| `tensor_networks.mera.trivial.float64.d4` | `tensorkit` | `ok` | 0.746991 | 0.072623 | `passed` |
| `tensor_networks.mera.z2.float64.d2` | `tensor0` | `ok` | 0.102113 | 0.003156 | `passed` |
| `tensor_networks.mera.z2.float64.d2` | `tensorkit` | `ok` | 0.422485 | 0.027802 | `passed` |
| `tensor_networks.mera.z2.float64.d4` | `tensor0` | `ok` | 0.252104 | 0.010753 | `passed` |
| `tensor_networks.mera.z2.float64.d4` | `tensorkit` | `ok` | 0.706988 | 0.034666 | `passed` |
| `tensor_networks.mera.z2.float64.d8` | `tensor0` | `ok` | 6.108318 | 0.129583 | `passed` |
| `tensor_networks.mera.z2.float64.d8` | `tensorkit` | `ok` | 11.268101 | 6.080066 | `passed` |
| `tensor_networks.mera.u1.float64.d4` | `tensor0` | `ok` | 0.337326 | 0.042337 | `passed` |
| `tensor_networks.mera.u1.float64.d4` | `tensorkit` | `ok` | 1.581942 | 0.047679 | `passed` |
| `tensor_networks.mera.u1.float64.d8` | `tensor0` | `ok` | 4.346134 | 0.144822 | `passed` |
| `tensor_networks.mera.u1.float64.d8` | `tensorkit` | `ok` | 21.089997 | 0.763430 | `passed` |
| `tensor_networks.mera.u1.float64.d12` | `tensor0` | `ok` | 43.361213 | 0.653547 | `passed` |
| `tensor_networks.mera.u1.float64.d12` | `tensorkit` | `ok` | 111.904125 | 24.363657 | `passed` |
| `tensor_networks.mera.su2.float64.d4` | `tensor0` | `ok` | 0.975539 | 0.046297 | `passed` |
| `tensor_networks.mera.su2.float64.d4` | `tensorkit` | `ok` | 2.135618 | 16.904512 | `passed` |

## Paired Comparison

| Workload | Baseline | Contender | Contender speedup | Validation |
| --- | --- | --- | ---: | --- |
| `tensor_networks.mpo.trivial.float64.d10x4x3` | `tensorkit` | `tensor0` | 2.827× | `passed` |
| `tensor_networks.mpo.trivial.float64.d40x4x3` | `tensorkit` | `tensor0` | 2.082× | `passed` |
| `tensor_networks.mpo.trivial.float64.d160x4x3` | `tensorkit` | `tensor0` | 2.654× | `passed` |
| `tensor_networks.mpo.trivial.float64.d100x10x10` | `tensorkit` | `tensor0` | 2.322× | `passed` |
| `tensor_networks.mpo.z2.float64.d10x4x4` | `tensorkit` | `tensor0` | 1.470× | `passed` |
| `tensor_networks.mpo.z2.float64.d40x4x4` | `tensorkit` | `tensor0` | 1.397× | `passed` |
| `tensor_networks.mpo.z2.float64.d160x4x4` | `tensorkit` | `tensor0` | 1.098× | `passed` |
| `tensor_networks.mpo.z2.float64.d100x10x10` | `tensorkit` | `tensor0` | 3.159× | `passed` |
| `tensor_networks.mpo.u1.float64.d40x5x3` | `tensorkit` | `tensor0` | 1.741× | `passed` |
| `tensor_networks.mpo.u1.float64.d160x5x3` | `tensorkit` | `tensor0` | 1.320× | `passed` |
| `tensor_networks.mpo.u1.float64.d640x5x3` | `tensorkit` | `tensor0` | 2.207× | `passed` |
| `tensor_networks.mpo.u1.float64.d200x20x20` | `tensorkit` | `tensor0` | 2.200× | `passed` |
| `tensor_networks.mpo.su2.float64.d40x5x3` | `tensorkit` | `tensor0` | 2.221× | `passed` |
| `tensor_networks.pepo.trivial.float64.d3x2x2x50` | `tensorkit` | `tensor0` | 3.874× | `passed` |
| `tensor_networks.pepo.trivial.float64.d4x2x2x50` | `tensorkit` | `tensor0` | 6.917× | `passed` |
| `tensor_networks.pepo.trivial.float64.d5x2x2x50` | `tensorkit` | `tensor0` | 3.698× | `passed` |
| `tensor_networks.pepo.trivial.float64.d6x2x2x50` | `tensorkit` | `tensor0` | 2.959× | `passed` |
| `tensor_networks.pepo.z2.float64.d4x2x2x50` | `tensorkit` | `tensor0` | 1.159× | `passed` |
| `tensor_networks.pepo.z2.float64.d5x2x2x50` | `tensorkit` | `tensor0` | 2.503× | `passed` |
| `tensor_networks.pepo.z2.float64.d6x2x2x50` | `tensorkit` | `tensor0` | 2.521× | `passed` |
| `tensor_networks.pepo.z2.float64.d8x2x2x50` | `tensorkit` | `tensor0` | 1.609× | `passed` |
| `tensor_networks.pepo.u1.float64.d4x2x2x100` | `tensorkit` | `tensor0` | 2.106× | `passed` |
| `tensor_networks.pepo.u1.float64.d6x2x2x100` | `tensorkit` | `tensor0` | 1.803× | `passed` |
| `tensor_networks.pepo.u1.float64.d8x2x2x100` | `tensorkit` | `tensor0` | 1.785× | `passed` |
| `tensor_networks.pepo.u1.float64.d10x2x2x50` | `tensorkit` | `tensor0` | 1.684× | `passed` |
| `tensor_networks.pepo.su2.float64.d4x2x2x100` | `tensorkit` | `tensor0` | 1.537× | `passed` |
| `tensor_networks.mera.trivial.float64.d2` | `tensorkit` | `tensor0` | 9.135× | `passed` |
| `tensor_networks.mera.trivial.float64.d3` | `tensorkit` | `tensor0` | 7.952× | `passed` |
| `tensor_networks.mera.trivial.float64.d4` | `tensorkit` | `tensor0` | 5.817× | `passed` |
| `tensor_networks.mera.z2.float64.d2` | `tensorkit` | `tensor0` | 4.137× | `passed` |
| `tensor_networks.mera.z2.float64.d4` | `tensorkit` | `tensor0` | 2.804× | `passed` |
| `tensor_networks.mera.z2.float64.d8` | `tensorkit` | `tensor0` | 1.845× | `passed` |
| `tensor_networks.mera.u1.float64.d4` | `tensorkit` | `tensor0` | 4.690× | `passed` |
| `tensor_networks.mera.u1.float64.d8` | `tensorkit` | `tensor0` | 4.853× | `passed` |
| `tensor_networks.mera.u1.float64.d12` | `tensorkit` | `tensor0` | 2.581× | `passed` |
| `tensor_networks.mera.su2.float64.d4` | `tensorkit` | `tensor0` | 2.189× | `passed` |
