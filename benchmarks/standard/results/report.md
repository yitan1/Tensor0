# Standard Default CPU Benchmark — 2026-10-09

**Status: exploratory snapshot on a non-isolated workstation, not an
interference-free formal baseline.** The native worker limit was 1 and the
processes were pinned to CPU2; neither setting guarantees exclusive CPU access.

[Raw results and samples](results.json)

Configuration ID: `20261009T051330Z-standard-default`. The default selection
completed 105 scenarios with warmup 2 / repeat 7, retaining 735 positive finite
samples. Neither `--quick` nor `--all` was used; the JSON profile label `full`
does not mean the complete registry. Execution counts: 87 eager, 6 JIT compile
and run, 6 cached JIT, 2 cached value-and-gradient, and 4 metadata scenarios.
Wall time was 17.804s, after the cross-backend run; it is not kernel latency.

## Reproduction and worker policy

Run from the repository root using the existing Python and Julia environments.
Check the allowed cpuset and the selected CPU's SMT sibling before pinning;
CPU2 and its sibling CPU10 are machine-specific examples. The commands below
write the fixed publication paths and overwrite the retained snapshot if run.
They reproduce the public timing policy, not the original uncontrolled host
conditions or the recorded artifact unless that artifact is loaded.

```sh
env JAX_PLATFORMS=cpu JULIA_PKG_OFFLINE=true JULIA_PKG_PRECOMPILE_AUTO=0 \
  JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  CROSS_BACKEND_THREADS=1 OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE \
  XLA_FLAGS=--xla_cpu_multi_thread_eigen=false PYTHONDONTWRITEBYTECODE=1 \
  taskset -c 2 .venv/bin/python -c \
  'import runpy; from tensor0._stride import set_num_threads, get_num_threads; set_num_threads(1); assert get_num_threads() == 1; runpy.run_module("benchmarks.standard", run_name="__main__"); assert get_num_threads() == 1' \
  --json-output benchmarks/standard/results/results.json \
  --markdown benchmarks/standard/results/report.md
```

The native limit must be set and queried in the **same process** before the
standard module runs; BLAS/OpenMP variables and XLA flags alone do not constrain
native stride workers. The actual launch additionally asserted CPU2 affinity,
loaded artifact SHA and native limit before/after execution using a recording
wrapper. The recorded native limit was 1 and affinity `[2]`. This is a worker
upper bound, not a count of all runtime OS threads; no CUDA run was performed.

## Environment and executable identity

AMD Ryzen 7 7840HS, 8 physical cores / 16 logical CPUs, about 46 GiB RAM;
Linux 7.2.6-arch2-1 / glibc 2.44. CPU governor `powersave`, boost enabled;
clocks and temperatures were not locked. Python 3.14.4, JAX/JAXlib 0.10.1,
NumPy 2.4.6; Julia 1.12.5 / TensorKit 0.16.5 using the pinned Manifest.
No dependency installation, native build, or binary replacement was performed.

The measured executable is **revision
`333de65d10bbf4381b8189f0433927c8286c8729` plus benchmark-only runner changes**,
not the modified native sources currently on disk. The loaded extension was
`src/tensor0/_native.cpython-314-x86_64-linux-gnu.so`, 85,932,624 bytes,
SHA256 `04dfed3079e7737dc5427a8390c672fdf33333a60ddc18217db37e268a41ca57`.
It matches the previously build-verified `333de65` artifact; no rebuild was
performed for this measurement. Tensor0 Python production sources were unchanged.

The as-run workspace contained four pre-existing, **unbuilt** native edits in
`execute/reduction.h`, `execute/scheduling.cc`, `execute/scheduling.h`, and
`ffi/reduction_impl.h` under `crates/tensor0-py/native/`, with related test/doc
edits. Those edits are not included in the loaded artifact: this snapshot does
not measure or validate them. The JSON's `publication_metadata.identity` records
executable identity separately from dirty source status. Original runner
metadata retains versions, support/workload hashes and as-run repository state.
Historical paths in that metadata describe the original snapshot, not current
publication links or a final publication manifest.

## Measurement and correctness boundary

Factories construct spaces, seeded random tensors and operations before timing.
Per-iteration setup/cache clearing is untimed; each operation and completion
synchronization are timed. Eager/cached modes use their declared warmups;
cold/raw modes follow their scenario policy. Compile-and-run intentionally
includes JIT compilation and execution; cached JIT/gradient preparation is
outside timing. Network outputs are rank-zero TensorMaps; scalar extraction is
excluded. Functional result allocation remains timed. Execution/cache labels,
all samples, and original statistics are retained unchanged in JSON.

Completion and finite timing samples are timing evidence, **not numerical
correctness validation**. Cross-backend scalar checks do not validate these
separate seeded-random standard workloads. These are Tensor0 profiling results,
not cross-library or hardware-independent claims. See the [suite contract](../README.md).

## Limitations

Standard ran after the medium comparison, in one registry-ordered process and
one timing round on a non-isolated workstation. CPU2 affinity and native limit 1
do not exclude SMT contention or shared-resource interference. The preceding
comparison observed CPU10 up to 100% busy and external tests on CPU0/CPU3;
see the [cross-backend report](../../cross_backend/results/report.md).
That observation is not evidence of CPU10 occupancy during each standard call.
Unlocked clocks, caches, thermals and background activity remain uncontrolled.
This task's own tests/builds did not overlap measurement. Do not infer causal
improvements or robust regression bounds from this snapshot. No samples were
filtered and no selective reruns were applied.

## Results

### `linalg.mul.trivial.float64.d2x2x2.eager`

- Group: `linalg`
- Description: Tensor0 mul workload via TensorMap @; sector=trivial, dimensions=(2, 2, 2)
- Profile: `quick`
- Dtype/size: `float64` / `d2x2x2`
- Execution/cache: `eager` / `warmed_metadata`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `0.070` / `0.074` / `0.006` / `0.087`
- Times ms: `0.087, 0.081, 0.074, 0.072, 0.070, 0.071, 0.075`

### `linalg.mul.trivial.float64.d8x8x8.eager`

- Group: `linalg`
- Description: Tensor0 mul workload via TensorMap @; sector=trivial, dimensions=(8, 8, 8)
- Profile: `full-only`
- Dtype/size: `float64` / `d8x8x8`
- Execution/cache: `eager` / `warmed_metadata`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `0.087` / `0.116` / `0.026` / `0.126`
- Times ms: `0.116, 0.121, 0.126, 0.091, 0.087, 0.097, 0.120`

### `linalg.mul.trivial.float64.d32x32x32.eager`

- Group: `linalg`
- Description: Tensor0 mul workload via TensorMap @; sector=trivial, dimensions=(32, 32, 32)
- Profile: `full-only`
- Dtype/size: `float64` / `d32x32x32`
- Execution/cache: `eager` / `warmed_metadata`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `0.085` / `0.095` / `0.014` / `0.122`
- Times ms: `0.122, 0.103, 0.103, 0.095, 0.092, 0.087, 0.085`

### `linalg.mul.trivial.float64.d64x64x64.eager`

- Group: `linalg`
- Description: Tensor0 mul workload via TensorMap @; sector=trivial, dimensions=(64, 64, 64)
- Profile: `full-only`
- Dtype/size: `float64` / `d64x64x64`
- Execution/cache: `eager` / `warmed_metadata`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `0.104` / `0.112` / `0.015` / `0.144`
- Times ms: `0.125, 0.112, 0.111, 0.144, 0.107, 0.121, 0.104`

### `linalg.mul.trivial.float64.d128x128x128.eager`

- Group: `linalg`
- Description: Tensor0 mul workload via TensorMap @; sector=trivial, dimensions=(128, 128, 128)
- Profile: `full-only`
- Dtype/size: `float64` / `d128x128x128`
- Execution/cache: `eager` / `warmed_metadata`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `0.184` / `0.199` / `0.032` / `0.274`
- Times ms: `0.202, 0.193, 0.274, 0.244, 0.199, 0.189, 0.184`

### `linalg.mul.trivial.complex128.d2x2x2.eager`

- Group: `linalg`
- Description: Tensor0 mul workload via TensorMap @; sector=trivial, dimensions=(2, 2, 2)
- Profile: `full-only`
- Dtype/size: `complex128` / `d2x2x2`
- Execution/cache: `eager` / `warmed_metadata`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `0.066` / `0.069` / `0.007` / `0.083`
- Times ms: `0.083, 0.075, 0.069, 0.067, 0.068, 0.074, 0.066`

### `linalg.mul.trivial.complex128.d8x8x8.eager`

- Group: `linalg`
- Description: Tensor0 mul workload via TensorMap @; sector=trivial, dimensions=(8, 8, 8)
- Profile: `full-only`
- Dtype/size: `complex128` / `d8x8x8`
- Execution/cache: `eager` / `warmed_metadata`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `0.084` / `0.085` / `0.005` / `0.096`
- Times ms: `0.096, 0.092, 0.087, 0.085, 0.085, 0.084, 0.084`

### `linalg.mul.trivial.complex128.d32x32x32.eager`

- Group: `linalg`
- Description: Tensor0 mul workload via TensorMap @; sector=trivial, dimensions=(32, 32, 32)
- Profile: `full-only`
- Dtype/size: `complex128` / `d32x32x32`
- Execution/cache: `eager` / `warmed_metadata`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `0.096` / `0.107` / `0.014` / `0.117`
- Times ms: `0.117, 0.102, 0.111, 0.107, 0.098, 0.117, 0.096`

### `linalg.mul.trivial.complex128.d64x64x64.eager`

- Group: `linalg`
- Description: Tensor0 mul workload via TensorMap @; sector=trivial, dimensions=(64, 64, 64)
- Profile: `full-only`
- Dtype/size: `complex128` / `d64x64x64`
- Execution/cache: `eager` / `warmed_metadata`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `0.143` / `0.154` / `0.017` / `0.187`
- Times ms: `0.187, 0.154, 0.148, 0.167, 0.146, 0.143, 0.160`

### `linalg.mul.trivial.complex128.d128x128x128.eager`

- Group: `linalg`
- Description: Tensor0 mul workload via TensorMap @; sector=trivial, dimensions=(128, 128, 128)
- Profile: `full-only`
- Dtype/size: `complex128` / `d128x128x128`
- Execution/cache: `eager` / `warmed_metadata`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `0.470` / `0.490` / `0.030` / `0.547`
- Times ms: `0.485, 0.490, 0.492, 0.547, 0.472, 0.470, 0.524`

### `linalg.mul.z2.float64.d2x2x2.eager`

- Group: `linalg`
- Description: Tensor0 mul workload via TensorMap @; sector=z2, dimensions=(2, 2, 2)
- Profile: `full-only`
- Dtype/size: `float64` / `d2x2x2`
- Execution/cache: `eager` / `warmed_metadata`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `0.263` / `0.280` / `0.013` / `0.306`
- Times ms: `0.306, 0.286, 0.271, 0.272, 0.283, 0.280, 0.263`

### `linalg.mul.z2.float64.d8x8x8.eager`

- Group: `linalg`
- Description: Tensor0 mul workload via TensorMap @; sector=z2, dimensions=(8, 8, 8)
- Profile: `full-only`
- Dtype/size: `float64` / `d8x8x8`
- Execution/cache: `eager` / `warmed_metadata`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `0.274` / `0.299` / `0.040` / `0.329`
- Times ms: `0.319, 0.299, 0.315, 0.279, 0.276, 0.329, 0.274`

### `linalg.mul.z2.float64.d32x32x32.eager`

- Group: `linalg`
- Description: Tensor0 mul workload via TensorMap @; sector=z2, dimensions=(32, 32, 32)
- Profile: `full-only`
- Dtype/size: `float64` / `d32x32x32`
- Execution/cache: `eager` / `warmed_metadata`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `0.325` / `0.343` / `0.011` / `0.364`
- Times ms: `0.364, 0.344, 0.340, 0.343, 0.328, 0.325, 0.347`

### `linalg.mul.z2.float64.d64x64x64.eager`

- Group: `linalg`
- Description: Tensor0 mul workload via TensorMap @; sector=z2, dimensions=(64, 64, 64)
- Profile: `full-only`
- Dtype/size: `float64` / `d64x64x64`
- Execution/cache: `eager` / `warmed_metadata`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `0.332` / `0.351` / `0.014` / `0.365`
- Times ms: `0.365, 0.351, 0.355, 0.344, 0.359, 0.342, 0.332`

### `linalg.mul.z2.float64.d128x128x128.eager`

- Group: `linalg`
- Description: Tensor0 mul workload via TensorMap @; sector=z2, dimensions=(128, 128, 128)
- Profile: `full-only`
- Dtype/size: `float64` / `d128x128x128`
- Execution/cache: `eager` / `warmed_metadata`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `0.365` / `0.376` / `0.013` / `0.404`
- Times ms: `0.404, 0.380, 0.376, 0.392, 0.370, 0.376, 0.365`

### `linalg.mul.z2.complex128.d2x2x2.eager`

- Group: `linalg`
- Description: Tensor0 mul workload via TensorMap @; sector=z2, dimensions=(2, 2, 2)
- Profile: `full-only`
- Dtype/size: `complex128` / `d2x2x2`
- Execution/cache: `eager` / `warmed_metadata`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `0.281` / `0.308` / `0.027` / `0.322`
- Times ms: `0.322, 0.308, 0.290, 0.290, 0.314, 0.321, 0.281`

### `linalg.mul.z2.complex128.d8x8x8.eager`

- Group: `linalg`
- Description: Tensor0 mul workload via TensorMap @; sector=z2, dimensions=(8, 8, 8)
- Profile: `full-only`
- Dtype/size: `complex128` / `d8x8x8`
- Execution/cache: `eager` / `warmed_metadata`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `0.281` / `0.293` / `0.029` / `0.322`
- Times ms: `0.316, 0.322, 0.286, 0.287, 0.281, 0.315, 0.293`

### `linalg.mul.z2.complex128.d32x32x32.eager`

- Group: `linalg`
- Description: Tensor0 mul workload via TensorMap @; sector=z2, dimensions=(32, 32, 32)
- Profile: `full-only`
- Dtype/size: `complex128` / `d32x32x32`
- Execution/cache: `eager` / `warmed_metadata`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `0.331` / `0.364` / `0.030` / `0.397`
- Times ms: `0.397, 0.368, 0.364, 0.380, 0.346, 0.342, 0.331`

### `linalg.mul.z2.complex128.d64x64x64.eager`

- Group: `linalg`
- Description: Tensor0 mul workload via TensorMap @; sector=z2, dimensions=(64, 64, 64)
- Profile: `full-only`
- Dtype/size: `complex128` / `d64x64x64`
- Execution/cache: `eager` / `warmed_metadata`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `0.354` / `0.373` / `0.022` / `0.394`
- Times ms: `0.394, 0.364, 0.382, 0.373, 0.355, 0.380, 0.354`

### `linalg.mul.z2.complex128.d128x128x128.eager`

- Group: `linalg`
- Description: Tensor0 mul workload via TensorMap @; sector=z2, dimensions=(128, 128, 128)
- Profile: `full-only`
- Dtype/size: `complex128` / `d128x128x128`
- Execution/cache: `eager` / `warmed_metadata`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `0.450` / `0.465` / `0.022` / `0.508`
- Times ms: `0.494, 0.465, 0.465, 0.508, 0.468, 0.450, 0.453`

### `linalg.svd.trivial.float64.d2x2.eager`

- Group: `linalg`
- Description: Tensor0 svd workload via svd_compact; sector=trivial, dimensions=(2, 2)
- Profile: `quick`
- Dtype/size: `float64` / `d2x2`
- Execution/cache: `eager` / `warmed_metadata`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `0.113` / `0.128` / `0.013` / `0.140`
- Times ms: `0.140, 0.134, 0.128, 0.126, 0.133, 0.117, 0.113`

### `linalg.svd.trivial.float64.d8x8.eager`

- Group: `linalg`
- Description: Tensor0 svd workload via svd_compact; sector=trivial, dimensions=(8, 8)
- Profile: `full-only`
- Dtype/size: `float64` / `d8x8`
- Execution/cache: `eager` / `warmed_metadata`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `0.124` / `0.130` / `0.008` / `0.140`
- Times ms: `0.140, 0.132, 0.137, 0.126, 0.130, 0.127, 0.124`

### `linalg.svd.trivial.float64.d32x32.eager`

- Group: `linalg`
- Description: Tensor0 svd workload via svd_compact; sector=trivial, dimensions=(32, 32)
- Profile: `full-only`
- Dtype/size: `float64` / `d32x32`
- Execution/cache: `eager` / `warmed_metadata`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `0.242` / `0.253` / `0.011` / `0.264`
- Times ms: `0.262, 0.264, 0.253, 0.248, 0.246, 0.255, 0.242`

### `linalg.svd.trivial.float64.d64x64.eager`

- Group: `linalg`
- Description: Tensor0 svd workload via svd_compact; sector=trivial, dimensions=(64, 64)
- Profile: `full-only`
- Dtype/size: `float64` / `d64x64`
- Execution/cache: `eager` / `warmed_metadata`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `0.535` / `0.550` / `0.008` / `0.570`
- Times ms: `0.570, 0.557, 0.550, 0.553, 0.535, 0.543, 0.550`

### `linalg.svd.trivial.float64.d128x128.eager`

- Group: `linalg`
- Description: Tensor0 svd workload via svd_compact; sector=trivial, dimensions=(128, 128)
- Profile: `full-only`
- Dtype/size: `float64` / `d128x128`
- Execution/cache: `eager` / `warmed_metadata`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `1.657` / `1.695` / `0.034` / `1.720`
- Times ms: `1.657, 1.666, 1.676, 1.695, 1.720, 1.705, 1.705`

### `linalg.svd.trivial.complex128.d2x2.eager`

- Group: `linalg`
- Description: Tensor0 svd workload via svd_compact; sector=trivial, dimensions=(2, 2)
- Profile: `full-only`
- Dtype/size: `complex128` / `d2x2`
- Execution/cache: `eager` / `warmed_metadata`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `0.120` / `0.123` / `0.010` / `0.147`
- Times ms: `0.147, 0.139, 0.123, 0.120, 0.123, 0.122, 0.120`

### `linalg.svd.trivial.complex128.d8x8.eager`

- Group: `linalg`
- Description: Tensor0 svd workload via svd_compact; sector=trivial, dimensions=(8, 8)
- Profile: `full-only`
- Dtype/size: `complex128` / `d8x8`
- Execution/cache: `eager` / `warmed_metadata`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `0.126` / `0.136` / `0.014` / `0.149`
- Times ms: `0.144, 0.142, 0.149, 0.136, 0.132, 0.126, 0.127`

### `linalg.svd.trivial.complex128.d32x32.eager`

- Group: `linalg`
- Description: Tensor0 svd workload via svd_compact; sector=trivial, dimensions=(32, 32)
- Profile: `full-only`
- Dtype/size: `complex128` / `d32x32`
- Execution/cache: `eager` / `warmed_metadata`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `0.293` / `0.303` / `0.014` / `0.328`
- Times ms: `0.328, 0.309, 0.301, 0.314, 0.303, 0.294, 0.293`

### `linalg.svd.trivial.complex128.d64x64.eager`

- Group: `linalg`
- Description: Tensor0 svd workload via svd_compact; sector=trivial, dimensions=(64, 64)
- Profile: `full-only`
- Dtype/size: `complex128` / `d64x64`
- Execution/cache: `eager` / `warmed_metadata`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `0.692` / `0.709` / `0.010` / `0.758`
- Times ms: `0.709, 0.758, 0.692, 0.710, 0.709, 0.716, 0.698`

### `linalg.svd.trivial.complex128.d128x128.eager`

- Group: `linalg`
- Description: Tensor0 svd workload via svd_compact; sector=trivial, dimensions=(128, 128)
- Profile: `full-only`
- Dtype/size: `complex128` / `d128x128`
- Execution/cache: `eager` / `warmed_metadata`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `3.089` / `3.107` / `0.023` / `3.130`
- Times ms: `3.130, 3.122, 3.093, 3.089, 3.107, 3.121, 3.104`

### `linalg.svd.z2.float64.d2x2.eager`

- Group: `linalg`
- Description: Tensor0 svd workload via svd_compact; sector=z2, dimensions=(2, 2)
- Profile: `full-only`
- Dtype/size: `float64` / `d2x2`
- Execution/cache: `eager` / `warmed_metadata`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `0.328` / `0.343` / `0.017` / `0.364`
- Times ms: `0.344, 0.350, 0.329, 0.364, 0.343, 0.328, 0.332`

### `linalg.svd.z2.float64.d8x8.eager`

- Group: `linalg`
- Description: Tensor0 svd workload via svd_compact; sector=z2, dimensions=(8, 8)
- Profile: `full-only`
- Dtype/size: `float64` / `d8x8`
- Execution/cache: `eager` / `warmed_metadata`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `0.348` / `0.356` / `0.021` / `0.388`
- Times ms: `0.358, 0.356, 0.385, 0.388, 0.348, 0.351, 0.349`

### `linalg.svd.z2.float64.d32x32.eager`

- Group: `linalg`
- Description: Tensor0 svd workload via svd_compact; sector=z2, dimensions=(32, 32)
- Profile: `full-only`
- Dtype/size: `float64` / `d32x32`
- Execution/cache: `eager` / `warmed_metadata`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `0.391` / `0.411` / `0.033` / `0.454`
- Times ms: `0.444, 0.413, 0.454, 0.397, 0.395, 0.411, 0.391`

### `linalg.svd.z2.float64.d64x64.eager`

- Group: `linalg`
- Description: Tensor0 svd workload via svd_compact; sector=z2, dimensions=(64, 64)
- Profile: `full-only`
- Dtype/size: `float64` / `d64x64`
- Execution/cache: `eager` / `warmed_metadata`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `0.585` / `0.631` / `0.018` / `0.663`
- Times ms: `0.645, 0.585, 0.663, 0.632, 0.622, 0.631, 0.618`

### `linalg.svd.z2.float64.d128x128.eager`

- Group: `linalg`
- Description: Tensor0 svd workload via svd_compact; sector=z2, dimensions=(128, 128)
- Profile: `full-only`
- Dtype/size: `float64` / `d128x128`
- Execution/cache: `eager` / `warmed_metadata`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `1.142` / `1.157` / `0.007` / `1.162`
- Times ms: `1.155, 1.157, 1.158, 1.158, 1.147, 1.142, 1.162`

### `linalg.svd.z2.complex128.d2x2.eager`

- Group: `linalg`
- Description: Tensor0 svd workload via svd_compact; sector=z2, dimensions=(2, 2)
- Profile: `full-only`
- Dtype/size: `complex128` / `d2x2`
- Execution/cache: `eager` / `warmed_metadata`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `0.329` / `0.353` / `0.029` / `0.379`
- Times ms: `0.373, 0.353, 0.365, 0.379, 0.337, 0.344, 0.329`

### `linalg.svd.z2.complex128.d8x8.eager`

- Group: `linalg`
- Description: Tensor0 svd workload via svd_compact; sector=z2, dimensions=(8, 8)
- Profile: `full-only`
- Dtype/size: `complex128` / `d8x8`
- Execution/cache: `eager` / `warmed_metadata`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `0.334` / `0.357` / `0.028` / `0.381`
- Times ms: `0.373, 0.381, 0.380, 0.352, 0.334, 0.357, 0.344`

### `linalg.svd.z2.complex128.d32x32.eager`

- Group: `linalg`
- Description: Tensor0 svd workload via svd_compact; sector=z2, dimensions=(32, 32)
- Profile: `full-only`
- Dtype/size: `complex128` / `d32x32`
- Execution/cache: `eager` / `warmed_metadata`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `0.406` / `0.416` / `0.016` / `0.478`
- Times ms: `0.434, 0.417, 0.478, 0.412, 0.416, 0.406, 0.409`

### `linalg.svd.z2.complex128.d64x64.eager`

- Group: `linalg`
- Description: Tensor0 svd workload via svd_compact; sector=z2, dimensions=(64, 64)
- Profile: `full-only`
- Dtype/size: `complex128` / `d64x64`
- Execution/cache: `eager` / `warmed_metadata`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `0.674` / `0.703` / `0.029` / `0.729`
- Times ms: `0.716, 0.674, 0.721, 0.697, 0.729, 0.682, 0.703`

### `linalg.svd.z2.complex128.d128x128.eager`

- Group: `linalg`
- Description: Tensor0 svd workload via svd_compact; sector=z2, dimensions=(128, 128)
- Profile: `full-only`
- Dtype/size: `complex128` / `d128x128`
- Execution/cache: `eager` / `warmed_metadata`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `1.527` / `1.551` / `0.028` / `1.569`
- Times ms: `1.529, 1.530, 1.561, 1.569, 1.527, 1.555, 1.551`

### `linalg.mul.u1.float64.small.eager`

- Group: `linalg`
- Description: Focused blockwise mul workload; sector=u1, size=small
- Profile: `quick`
- Dtype/size: `float64` / `small`
- Execution/cache: `eager` / `warmed_metadata`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `0.276` / `0.289` / `0.026` / `0.329`
- Times ms: `0.329, 0.315, 0.289, 0.281, 0.301, 0.283, 0.276`

### `linalg.mul.u1.float64.medium.eager`

- Group: `linalg`
- Description: Focused blockwise mul workload; sector=u1, size=medium
- Profile: `full-only`
- Dtype/size: `float64` / `medium`
- Execution/cache: `eager` / `warmed_metadata`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `0.336` / `0.344` / `0.013` / `0.398`
- Times ms: `0.398, 0.356, 0.350, 0.344, 0.336, 0.336, 0.343`

### `linalg.mul.u1.float64.large.eager`

- Group: `linalg`
- Description: Focused blockwise mul workload; sector=u1, size=large
- Profile: `full-only`
- Dtype/size: `float64` / `large`
- Execution/cache: `eager` / `warmed_metadata`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `0.325` / `0.332` / `0.013` / `0.358`
- Times ms: `0.358, 0.337, 0.332, 0.349, 0.330, 0.325, 0.330`

### `linalg.mul.u1.complex128.small.eager`

- Group: `linalg`
- Description: Focused blockwise mul workload; sector=u1, size=small
- Profile: `full-only`
- Dtype/size: `complex128` / `small`
- Execution/cache: `eager` / `warmed_metadata`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `0.289` / `0.307` / `0.018` / `0.322`
- Times ms: `0.307, 0.289, 0.301, 0.298, 0.321, 0.322, 0.314`

### `linalg.mul.u1.complex128.medium.eager`

- Group: `linalg`
- Description: Focused blockwise mul workload; sector=u1, size=medium
- Profile: `full-only`
- Dtype/size: `complex128` / `medium`
- Execution/cache: `eager` / `warmed_metadata`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `0.312` / `0.327` / `0.011` / `0.363`
- Times ms: `0.363, 0.327, 0.327, 0.331, 0.314, 0.312, 0.321`

### `linalg.mul.fermion_parity.float64.small.eager`

- Group: `linalg`
- Description: Focused blockwise mul workload; sector=fermion_parity, size=small
- Profile: `full-only`
- Dtype/size: `float64` / `small`
- Execution/cache: `eager` / `warmed_metadata`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `0.295` / `0.305` / `0.009` / `0.317`
- Times ms: `0.305, 0.307, 0.296, 0.306, 0.317, 0.299, 0.295`

### `linalg.svd.u1.float64.small.eager`

- Group: `linalg`
- Description: Focused blockwise svd workload; sector=u1, size=small
- Profile: `quick`
- Dtype/size: `float64` / `small`
- Execution/cache: `eager` / `warmed_metadata`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `0.357` / `0.377` / `0.022` / `0.401`
- Times ms: `0.378, 0.377, 0.357, 0.401, 0.357, 0.361, 0.386`

### `linalg.svd.u1.float64.medium.eager`

- Group: `linalg`
- Description: Focused blockwise svd workload; sector=u1, size=medium
- Profile: `full-only`
- Dtype/size: `float64` / `medium`
- Execution/cache: `eager` / `warmed_metadata`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `0.338` / `0.356` / `0.019` / `0.375`
- Times ms: `0.367, 0.358, 0.347, 0.375, 0.356, 0.338, 0.340`

### `linalg.svd.u1.complex128.small.eager`

- Group: `linalg`
- Description: Focused blockwise svd workload; sector=u1, size=small
- Profile: `full-only`
- Dtype/size: `complex128` / `small`
- Execution/cache: `eager` / `warmed_metadata`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `0.354` / `0.375` / `0.026` / `0.390`
- Times ms: `0.385, 0.381, 0.375, 0.390, 0.354, 0.358, 0.356`

### `linalg.svd.u1.complex128.medium.eager`

- Group: `linalg`
- Description: Focused blockwise svd workload; sector=u1, size=medium
- Profile: `full-only`
- Dtype/size: `complex128` / `medium`
- Execution/cache: `eager` / `warmed_metadata`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `0.356` / `0.375` / `0.024` / `0.396`
- Times ms: `0.396, 0.370, 0.383, 0.395, 0.356, 0.375, 0.360`

### `linalg.svd.fermion_parity.float64.small.eager`

- Group: `linalg`
- Description: Focused blockwise svd workload; sector=fermion_parity, size=small
- Profile: `full-only`
- Dtype/size: `float64` / `small`
- Execution/cache: `eager` / `warmed_metadata`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `0.341` / `0.361` / `0.022` / `0.395`
- Times ms: `0.391, 0.361, 0.361, 0.395, 0.348, 0.360, 0.341`

### `linalg.mul.u1.float64.small.jit_compile_and_run`

- Group: `linalg`
- Description: Focused blockwise mul workload; sector=u1, size=small
- Profile: `quick`
- Dtype/size: `float64` / `small`
- Execution/cache: `jit_compile_and_run` / `clear_jax_caches`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `51.442` / `57.370` / `0.650` / `59.850`
- Times ms: `57.541, 57.998, 59.850, 57.047, 57.370, 57.192, 51.442`

### `linalg.mul.u1.float64.small.jit_cached_run`

- Group: `linalg`
- Description: Focused blockwise mul workload; sector=u1, size=small
- Profile: `quick`
- Dtype/size: `float64` / `small`
- Execution/cache: `jit_cached_run` / `compiled_once`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `0.008` / `0.008` / `0.002` / `0.011`
- Times ms: `0.010, 0.011, 0.008, 0.008, 0.008, 0.009, 0.008`

### `linalg.mul.u1.float64.small.value_and_grad_cached`

- Group: `linalg`
- Description: Focused blockwise mul workload; sector=u1, size=small
- Profile: `quick`
- Dtype/size: `float64` / `small`
- Execution/cache: `value_and_grad_cached` / `compiled_once`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `0.021` / `0.022` / `0.001` / `0.025`
- Times ms: `0.025, 0.023, 0.023, 0.022, 0.022, 0.022, 0.021`

### `composition.trivial.float64.small.eager`

- Group: `linalg`
- Description: Trivial matrix composition with dimensions (16, 12, 14)
- Profile: `quick`
- Dtype/size: `float64` / `small`
- Execution/cache: `eager` / `warmed_metadata`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `0.089` / `0.096` / `0.009` / `0.118`
- Times ms: `0.102, 0.100, 0.118, 0.096, 0.092, 0.092, 0.089`

### `composition.trivial.float64.small.jit_compile_and_run`

- Group: `linalg`
- Description: Trivial matrix composition with dimensions (16, 12, 14)
- Profile: `full-only`
- Dtype/size: `float64` / `small`
- Execution/cache: `jit_compile_and_run` / `clear_jax_caches`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `7.929` / `8.195` / `0.410` / `8.769`
- Times ms: `8.769, 8.304, 8.556, 7.929, 7.943, 8.195, 8.097`

### `composition.trivial.float64.small.jit_cached_run`

- Group: `linalg`
- Description: Trivial matrix composition with dimensions (16, 12, 14)
- Profile: `full-only`
- Dtype/size: `float64` / `small`
- Execution/cache: `jit_cached_run` / `compiled_once`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `0.017` / `0.017` / `0.004` / `0.025`
- Times ms: `0.020, 0.025, 0.022, 0.017, 0.017, 0.017, 0.017`

### `composition.trivial.complex128.medium.eager`

- Group: `linalg`
- Description: Trivial matrix composition with dimensions (64, 48, 56)
- Profile: `full-only`
- Dtype/size: `complex128` / `medium`
- Execution/cache: `eager` / `warmed_metadata`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `0.122` / `0.132` / `0.009` / `0.139`
- Times ms: `0.139, 0.136, 0.132, 0.126, 0.122, 0.133, 0.125`

### `transforms.permute.trivial.float64.d64x48.p2x1_to_empty.eager`

- Group: `transforms`
- Description: Tensor0 functional permute workload; sector=trivial, dimensions=(64, 48)
- Profile: `quick`
- Dtype/size: `float64` / `d64x48`
- Execution/cache: `eager` / `warmed_metadata`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `0.043` / `0.047` / `0.005` / `0.056`
- Times ms: `0.056, 0.051, 0.047, 0.045, 0.043, 0.045, 0.050`

### `transforms.permute.u1.float64.small.cold`

- Group: `transforms`
- Description: U1 permute with explicit tree-braider cache state
- Profile: `quick`
- Dtype/size: `float64` / `small`
- Execution/cache: `eager` / `cold_transform`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `0.144` / `0.146` / `0.021` / `0.176`
- Times ms: `0.176, 0.176, 0.157, 0.146, 0.145, 0.146, 0.144`

### `transforms.permute.u1.float64.small.cached`

- Group: `transforms`
- Description: U1 permute with explicit tree-braider cache state
- Profile: `full-only`
- Dtype/size: `float64` / `small`
- Execution/cache: `eager` / `cached_transform`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `0.127` / `0.132` / `0.014` / `0.162`
- Times ms: `0.134, 0.131, 0.162, 0.152, 0.132, 0.127, 0.128`

### `transforms.repartition.u1.float64.small.cold`

- Group: `transforms`
- Description: U1 repartition with explicit tree-transposer cache state
- Profile: `quick`
- Dtype/size: `float64` / `small`
- Execution/cache: `eager` / `cold_transform`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `0.090` / `0.101` / `0.019` / `0.116`
- Times ms: `0.116, 0.107, 0.101, 0.115, 0.094, 0.091, 0.090`

### `transforms.repartition.u1.float64.small.cached`

- Group: `transforms`
- Description: U1 repartition with explicit tree-transposer cache state
- Profile: `full-only`
- Dtype/size: `float64` / `small`
- Execution/cache: `eager` / `cached_transform`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `0.080` / `0.084` / `0.007` / `0.097`
- Times ms: `0.084, 0.097, 0.091, 0.086, 0.084, 0.080, 0.080`

### `transforms.permute.su2.float64.small.cold`

- Group: `transforms`
- Description: SU2 permute with explicit generic transformer cache state
- Profile: `quick`
- Dtype/size: `float64` / `small`
- Execution/cache: `eager` / `cold_transform`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `0.196` / `0.203` / `0.016` / `0.228`
- Times ms: `0.228, 0.220, 0.212, 0.203, 0.198, 0.196, 0.202`

### `transforms.permute.su2.float64.small.cached`

- Group: `transforms`
- Description: SU2 permute with explicit generic transformer cache state
- Profile: `full-only`
- Dtype/size: `float64` / `small`
- Execution/cache: `eager` / `cached_transform`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `0.163` / `0.174` / `0.012` / `0.187`
- Times ms: `0.187, 0.183, 0.174, 0.170, 0.168, 0.163, 0.180`

### `transforms.permute.u1.float64.large.cold`

- Group: `transforms`
- Description: Large U1 permute with explicit tree-braider cache state
- Profile: `full-only`
- Dtype/size: `float64` / `large`
- Execution/cache: `eager` / `cold_transform`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `0.193` / `0.204` / `0.012` / `0.238`
- Times ms: `0.238, 0.211, 0.204, 0.196, 0.194, 0.204, 0.193`

### `transforms.permute.u1.float64.large.cached`

- Group: `transforms`
- Description: Large U1 permute with explicit tree-braider cache state
- Profile: `full-only`
- Dtype/size: `float64` / `large`
- Execution/cache: `eager` / `cached_transform`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `0.176` / `0.178` / `0.002` / `0.181`
- Times ms: `0.177, 0.179, 0.178, 0.176, 0.181, 0.176, 0.178`

### `transforms.twist.u1.identity.eager`

- Group: `transforms`
- Description: U1 identity twist
- Profile: `quick`
- Dtype/size: `float64` / `small`
- Execution/cache: `eager` / `warmed_metadata`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `0.002` / `0.002` / `0.000` / `0.002`
- Times ms: `0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002`

### `transforms.twist.fermion_parity.nontrivial.eager`

- Group: `transforms`
- Description: FermionParity nontrivial twist
- Profile: `quick`
- Dtype/size: `float64` / `small`
- Execution/cache: `eager` / `warmed_metadata`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `0.115` / `0.122` / `0.014` / `0.134`
- Times ms: `0.134, 0.132, 0.133, 0.119, 0.115, 0.118, 0.122`

### `protect.su2.permute.float64.small.eager`

- Group: `transforms`
- Description: SU2 mixed-spin rank-4 permutation coverage
- Profile: `quick`
- Dtype/size: `float64` / `small`
- Execution/cache: `eager` / `warmed_metadata`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `0.171` / `0.181` / `0.008` / `0.194`
- Times ms: `0.190, 0.181, 0.182, 0.194, 0.181, 0.174, 0.171`

### `permute.trivial.rank4.float64.small.eager`

- Group: `transforms`
- Description: Trivial rank-4 nonidentity permutation with shape (4, 3, 5, 2)
- Profile: `quick`
- Dtype/size: `float64` / `small`
- Execution/cache: `eager` / `warmed_metadata`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `0.043` / `0.045` / `0.001` / `0.050`
- Times ms: `0.050, 0.045, 0.044, 0.046, 0.045, 0.045, 0.043`

### `permute.trivial.rank4.float64.small.jit_compile_and_run`

- Group: `transforms`
- Description: Trivial rank-4 nonidentity permutation with shape (4, 3, 5, 2)
- Profile: `full-only`
- Dtype/size: `float64` / `small`
- Execution/cache: `jit_compile_and_run` / `clear_jax_caches`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `18.733` / `19.474` / `0.626` / `20.258`
- Times ms: `19.474, 18.733, 18.971, 19.498, 20.258, 19.722, 18.996`

### `permute.trivial.rank4.float64.small.jit_cached_run`

- Group: `transforms`
- Description: Trivial rank-4 nonidentity permutation with shape (4, 3, 5, 2)
- Profile: `full-only`
- Dtype/size: `float64` / `small`
- Execution/cache: `jit_cached_run` / `compiled_once`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `0.006` / `0.006` / `0.001` / `0.008`
- Times ms: `0.008, 0.007, 0.007, 0.006, 0.006, 0.006, 0.006`

### `permute.trivial.rank4.complex128.medium.eager`

- Group: `transforms`
- Description: Trivial rank-4 nonidentity permutation with shape (12, 8, 10, 6)
- Profile: `full-only`
- Dtype/size: `complex128` / `medium`
- Execution/cache: `eager` / `warmed_metadata`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `0.050` / `0.053` / `0.006` / `0.063`
- Times ms: `0.063, 0.058, 0.055, 0.053, 0.050, 0.051, 0.050`

### `trace.u1.partial`

- Group: `contractions`
- Description: U1 partial tensor trace
- Profile: `quick`
- Dtype/size: `float64` / `small`
- Execution/cache: `eager` / `warmed_metadata`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `0.255` / `0.274` / `0.012` / `0.294`
- Times ms: `0.294, 0.274, 0.264, 0.275, 0.255, 0.286, 0.274`

### `contract.u1.partial`

- Group: `contractions`
- Description: U1 partial binary contraction
- Profile: `quick`
- Dtype/size: `float64` / `small`
- Execution/cache: `eager` / `warmed_metadata`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `1.151` / `1.207` / `0.076` / `1.344`
- Times ms: `1.344, 1.252, 1.151, 1.205, 1.201, 1.207, 1.306`

### `trace.su2.partial`

- Group: `contractions`
- Description: SU2 partial tensor trace
- Profile: `quick`
- Dtype/size: `float64` / `small`
- Execution/cache: `eager` / `warmed_metadata`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `0.177` / `0.187` / `0.016` / `0.212`
- Times ms: `0.194, 0.187, 0.183, 0.184, 0.206, 0.212, 0.177`

### `trace.fermion.full`

- Group: `contractions`
- Description: FermionParity full tensor trace
- Profile: `full-only`
- Dtype/size: `float64` / `small`
- Execution/cache: `eager` / `warmed_metadata`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `0.163` / `0.168` / `0.005` / `0.180`
- Times ms: `0.180, 0.172, 0.168, 0.172, 0.166, 0.168, 0.163`

### `contract.su2.fusion_basis`

- Group: `contractions`
- Description: SU2 fusion-basis binary contraction
- Profile: `full-only`
- Dtype/size: `float64` / `small`
- Execution/cache: `eager` / `warmed_metadata`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `0.790` / `0.794` / `0.010` / `0.850`
- Times ms: `0.850, 0.794, 0.794, 0.792, 0.801, 0.790, 0.805`

### `contract.fermion.twist`

- Group: `contractions`
- Description: FermionParity contraction with right twist
- Profile: `full-only`
- Dtype/size: `float64` / `small`
- Execution/cache: `eager` / `warmed_metadata`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `0.270` / `0.280` / `0.011` / `0.295`
- Times ms: `0.295, 0.285, 0.287, 0.280, 0.275, 0.274, 0.270`

### `trace.trivial.partial.rank4.float64.small.eager`

- Group: `contractions`
- Description: Trivial partial trace with source shape (8, 4, 6, 4)
- Profile: `quick`
- Dtype/size: `float64` / `small`
- Execution/cache: `eager` / `warmed_metadata`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `0.070` / `0.077` / `0.005` / `0.084`
- Times ms: `0.084, 0.080, 0.080, 0.077, 0.075, 0.075, 0.070`

### `trace.trivial.partial.rank4.float64.small.jit_compile_and_run`

- Group: `contractions`
- Description: Trivial partial trace with source shape (8, 4, 6, 4)
- Profile: `full-only`
- Dtype/size: `float64` / `small`
- Execution/cache: `jit_compile_and_run` / `clear_jax_caches`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `23.650` / `23.784` / `0.186` / `24.309`
- Times ms: `23.938, 23.650, 23.784, 23.983, 23.765, 24.309, 23.784`

### `trace.trivial.partial.rank4.float64.small.jit_cached_run`

- Group: `contractions`
- Description: Trivial partial trace with source shape (8, 4, 6, 4)
- Profile: `full-only`
- Dtype/size: `float64` / `small`
- Execution/cache: `jit_cached_run` / `compiled_once`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `0.015` / `0.016` / `0.001` / `0.019`
- Times ms: `0.019, 0.018, 0.016, 0.017, 0.016, 0.016, 0.015`

### `trace.trivial.partial.rank4.complex128.medium.eager`

- Group: `contractions`
- Description: Trivial partial trace with source shape (24, 12, 20, 12)
- Profile: `full-only`
- Dtype/size: `complex128` / `medium`
- Execution/cache: `eager` / `warmed_metadata`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `0.159` / `0.170` / `0.021` / `0.195`
- Times ms: `0.180, 0.195, 0.191, 0.170, 0.163, 0.159, 0.167`

### `contract.trivial.partial.float64.small.eager`

- Group: `contractions`
- Description: Trivial partial contraction with dimensions (6, 4, 7, 5, 3)
- Profile: `quick`
- Dtype/size: `float64` / `small`
- Execution/cache: `eager` / `warmed_metadata`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `0.124` / `0.152` / `0.015` / `0.159`
- Times ms: `0.159, 0.155, 0.152, 0.159, 0.142, 0.142, 0.124`

### `contract.trivial.partial.float64.small.jit_compile_and_run`

- Group: `contractions`
- Description: Trivial partial contraction with dimensions (6, 4, 7, 5, 3)
- Profile: `full-only`
- Dtype/size: `float64` / `small`
- Execution/cache: `jit_compile_and_run` / `clear_jax_caches`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `30.662` / `31.217` / `0.939` / `33.884`
- Times ms: `30.662, 31.485, 30.726, 31.217, 31.061, 32.179, 33.884`

### `contract.trivial.partial.float64.small.jit_cached_run`

- Group: `contractions`
- Description: Trivial partial contraction with dimensions (6, 4, 7, 5, 3)
- Profile: `full-only`
- Dtype/size: `float64` / `small`
- Execution/cache: `jit_cached_run` / `compiled_once`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `0.019` / `0.022` / `0.005` / `0.025`
- Times ms: `0.025, 0.025, 0.023, 0.022, 0.020, 0.019, 0.019`

### `contract.trivial.partial.complex128.medium.eager`

- Group: `contractions`
- Description: Trivial partial contraction with dimensions (16, 12, 18, 10, 8)
- Profile: `full-only`
- Dtype/size: `complex128` / `medium`
- Execution/cache: `eager` / `warmed_metadata`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `0.213` / `0.221` / `0.010` / `0.246`
- Times ms: `0.228, 0.227, 0.220, 0.215, 0.246, 0.213, 0.221`

### `network.named.default`

- Group: `contractions`
- Description: Named three-tensor default-order contraction
- Profile: `quick`
- Dtype/size: `float64` / `small`
- Execution/cache: `eager` / `warmed_metadata`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `0.351` / `0.370` / `0.022` / `0.399`
- Times ms: `0.392, 0.370, 0.384, 0.399, 0.364, 0.368, 0.351`

### `network.ncon.default`

- Group: `contractions`
- Description: Integer-label three-tensor default-order contraction
- Profile: `quick`
- Dtype/size: `float64` / `small`
- Execution/cache: `eager` / `warmed_metadata`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `0.379` / `0.390` / `0.011` / `0.412`
- Times ms: `0.392, 0.412, 0.388, 0.389, 0.390, 0.379, 0.407`

### `network.disconnected`

- Group: `contractions`
- Description: Disconnected named tensor product
- Profile: `quick`
- Dtype/size: `float64` / `small`
- Execution/cache: `eager` / `warmed_metadata`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `0.397` / `0.423` / `0.020` / `0.439`
- Times ms: `0.439, 0.416, 0.424, 0.435, 0.397, 0.423, 0.403`

### `jax.network.compile_and_run`

- Group: `contractions`
- Description: Named three-tensor custom-order network
- Profile: `quick`
- Dtype/size: `float64` / `small`
- Execution/cache: `jit_compile_and_run` / `clear_jax_caches`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `20.269` / `20.610` / `0.636` / `21.242`
- Times ms: `20.665, 20.269, 21.242, 20.269, 20.271, 21.149, 20.610`

### `jax.network.cached`

- Group: `contractions`
- Description: Named three-tensor custom-order network
- Profile: `quick`
- Dtype/size: `float64` / `small`
- Execution/cache: `jit_cached_run` / `compiled_once`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `0.007` / `0.007` / `0.000` / `0.008`
- Times ms: `0.008, 0.008, 0.007, 0.007, 0.007, 0.007, 0.007`

### `jax.network.value_and_grad`

- Group: `contractions`
- Description: Named three-tensor custom-order network
- Profile: `quick`
- Dtype/size: `float64` / `small`
- Execution/cache: `value_and_grad_cached` / `compiled_once`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `0.005` / `0.006` / `0.001` / `0.007`
- Times ms: `0.007, 0.007, 0.006, 0.006, 0.005, 0.006, 0.006`

### `network.named.default.medium`

- Group: `contractions`
- Description: Named default-order medium contraction
- Profile: `full-only`
- Dtype/size: `float64` / `medium`
- Execution/cache: `eager` / `warmed_metadata`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `0.403` / `0.428` / `0.029` / `0.463`
- Times ms: `0.463, 0.428, 0.455, 0.429, 0.419, 0.407, 0.403`

### `network.named.custom.medium`

- Group: `contractions`
- Description: Named custom-order medium contraction
- Profile: `full-only`
- Dtype/size: `float64` / `medium`
- Execution/cache: `eager` / `warmed_metadata`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `0.374` / `0.395` / `0.029` / `0.449`
- Times ms: `0.408, 0.382, 0.413, 0.381, 0.395, 0.449, 0.374`

### `network.ncon.default.medium`

- Group: `contractions`
- Description: Integer-label default-order medium contraction
- Profile: `full-only`
- Dtype/size: `float64` / `medium`
- Execution/cache: `eager` / `warmed_metadata`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `0.424` / `0.433` / `0.011` / `0.446`
- Times ms: `0.433, 0.440, 0.425, 0.438, 0.432, 0.424, 0.446`

### `network.ncon.custom.medium`

- Group: `contractions`
- Description: Integer-label custom-order medium contraction
- Profile: `full-only`
- Dtype/size: `float64` / `medium`
- Execution/cache: `eager` / `warmed_metadata`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `0.382` / `0.395` / `0.027` / `0.450`
- Times ms: `0.450, 0.389, 0.409, 0.416, 0.382, 0.395, 0.382`

### `tensor_networks.mpo.trivial.float64.d10x4x3.eager`

- Group: `tensor_networks`
- Description: Tensor0 MPO topology; sector=trivial, dimensions=(10, 4, 3)
- Profile: `quick`
- Dtype/size: `float64` / `d10x4x3`
- Execution/cache: `eager` / `warmed_metadata`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `0.694` / `0.705` / `0.011` / `0.721`
- Times ms: `0.710, 0.721, 0.705, 0.712, 0.699, 0.694, 0.701`

### `tensor_networks.pepo.trivial.float64.d3x2x2x50.eager`

- Group: `tensor_networks`
- Description: Tensor0 PEPO topology; sector=trivial, dimensions=(3, 2, 2, 50)
- Profile: `quick`
- Dtype/size: `float64` / `d3x2x2x50`
- Execution/cache: `eager` / `warmed_metadata`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `24.453` / `25.203` / `1.435` / `26.464`
- Times ms: `26.386, 24.453, 26.464, 26.064, 24.467, 25.203, 25.113`

### `tensor_networks.mera.trivial.float64.d2.eager`

- Group: `tensor_networks`
- Description: Tensor0 MERA topology; sector=trivial, dimensions=(2,)
- Profile: `quick`
- Dtype/size: `float64` / `d2`
- Execution/cache: `eager` / `warmed_metadata`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `1.939` / `1.976` / `0.033` / `1.997`
- Times ms: `1.976, 1.984, 1.997, 1.983, 1.944, 1.958, 1.939`

### `layout.u1_two_factor.cold`

- Group: `diagnostics`
- Description: U1 two-factor sector and degeneracy structure construction
- Profile: `quick`
- Dtype/size: `none` / `small`
- Execution/cache: `metadata` / `cold_layout`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `0.007` / `0.008` / `0.001` / `0.009`
- Times ms: `0.009, 0.008, 0.008, 0.008, 0.008, 0.007, 0.007`

### `layout.su2_four_half.cold`

- Group: `diagnostics`
- Description: SU2 four spin-half sector and degeneracy structure construction
- Profile: `quick`
- Dtype/size: `none` / `small`
- Execution/cache: `metadata` / `cold_layout`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `0.003` / `0.003` / `0.000` / `0.003`
- Times ms: `0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003`

### `layout.u1_two_factor.cached`

- Group: `diagnostics`
- Description: U1 two-factor sector and degeneracy structure construction
- Profile: `full-only`
- Dtype/size: `none` / `small`
- Execution/cache: `metadata` / `cached_layout`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `0.001` / `0.001` / `0.000` / `0.001`
- Times ms: `0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001`

### `layout.su2_four_half.cached`

- Group: `diagnostics`
- Description: SU2 four spin-half sector and degeneracy structure construction
- Profile: `full-only`
- Dtype/size: `none` / `small`
- Execution/cache: `metadata` / `cached_layout`
- Warmup/repeat: `2/7`
- Min/median/IQR/max ms: `0.001` / `0.001` / `0.000` / `0.001`
- Times ms: `0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001`

