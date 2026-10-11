# Full Cross-Backend CPU — 8 Workers

[All raw samples](results.json) · [Latency distributions](latency.svg) · [Shared sampling plan](../sampling-plan.json) · [Overview](../report.md)

**69 workloads (66 reference dimension cases + 3 nonuniform SU2 extensions), same committed source and native build at 1/4/8t.** Equal-workload geometric mean TK/T0 latency ratios: **median 1.918×; mean 2.257×**. Values >1 favor Tensor0. These are workload-weighted descriptive ratios, not total time, throughput or significance.

| Round (zero-based) | Tensor0 lower median | Geometric mean TK/T0 |
| --- | ---: | ---: |
| 0 | 56/69 | 1.975× |
| 1 | 56/69 | 1.913× |
| 2 | 57/69 | 1.950× |

Observed direction reversals: **3/69**, flagged below; raw within-round medians, pooled min/median/inclusive IQR/max and all samples are in JSON. No observed reversal is not a claim of significance or future stability.

## Identity, resources and measurement

- Source `387b2b8c4eeebeb092a96158236315cc2e8a1129`, committed archive, all 160 production file hashes verified. Fresh release native SHA256 `a5e0c34f4179a94e8a8552a16441237c9dc81a68b7e9c8e06f30b4440d017638`. As-run public runner unchanged; no dirty production, GEMM strategy edits or benchmark patches. Rust 1.89/GCC 15.3/maturin 1.13.1; approved offline dependencies unchanged. GCC 15 g++ linker; `RUSTFLAGS="-C link-arg=-lstdc++ -C link-arg=-lgcc"` and final root-crate flags `-C link-arg=-Wl,--no-as-needed -C link-arg=-Wl,--undefined=__cpu_features2 -C link-arg=-lstdc++ -C link-arg=-lgcc`; no LD_PRELOAD. Distribution 0.1.0 still exports literal 0.0.0 runtime version in this source; the commit/hash, not that exported string, identifies the build.
- Python 3.14.6, JAX/JAXlib 0.10.1, NumPy 2.4.6; Julia 1.12.5, TensorKit 0.16.5, TensorOperations 5.6.1, Strided 2.5.0. Actual Julia BLAS configuration `LBTConfig([ILP64] libopenblas64_.so)`; library hashes retained in JSON.
- Slurm node **n002**, E5-2680v3 (Haswell-EP), 24 physical cores/two sockets, SMT off. Exclusive 24-core allocation, 104 GiB job memory on 128 GiB-class hardware with 20 GiB OS reservation. Measurement affinity **[0, 1, 2, 3, 4, 5, 6, 7]**, nested 1/4/8 same-socket distinct physical cores; monitor on CPU 8. Three sequential exclusive round jobs **993086, 993087, 993088**, thread order 1/4/8 then 4/8/1 then 8/1/4; backend/workload forward/reverse/forward. Allocation identities and gates checked each time.
- Before timing, independent pinned 4/8-process gates reached approximately 4/8 core-equivalents; native reduction and Julia/BLAS multiplication showed actual worker-thread CPU time with queried 4/8 limits. Gates finished before samples and prove usable resources/representative parallelism, not saturation of every workload. Queried runtime limits match every request.
- Per-workload mandatory first invocation + **two warmups**, then frozen **7–100 timed calls per round** according to the [shared plan](../sampling-plan.json), identical for both backends and all thread counts. Slowest of six complete preflight medians >=2s gets 7; other counts are clamp(ceil(2s/t_slow),15,100). The threshold selects a cost tier; the 2s numerator is a bounded target, not a guarantee of precision or sample duration. Preflight is separate (one round, 1/3 timed calls by risk), never pooled with formal samples. Counts are not adapted to speedup, winners or observed formal results.
- Prepared contraction execution and completion are timed. Inputs, planning, compilation, mandatory call and warmups are outside timing; scalar validation reuses the final timed output without extra contractions. Natural GC tails are retained: no sample filtering or selective reruns. The [pinned upstream network-comparison script](https://github.com/QuantumKitHub/TensorKit.jl/blob/cfaa073e4d1e3eb2167edcbdc3be9872f41e7d91/benchmark/comparisons/run_tensorkit.jl) instead sets **K=100** outer samples, with [one initial compilation call, inner=1](https://github.com/QuantumKitHub/TensorKit.jl/blob/cfaa073e4d1e3eb2167edcbdc3be9872f41e7d91/benchmark/comparisons/benchtools.jl), reports averages and separately records GC time. We add two explicit warmups, use fixed 7–100 repeats per round and three paired rounds, report pooled medians (now supplemented by derived unfiltered means), and retain GC in wall samples without subtracting it. Thus heavy cases have 21 rather than 100 timed calls, and 100-repeat cases have 300 across three processes, not an identical sampling policy. The separate upstream BenchmarkTools suite (20s/10000-sample limits) is not that network-comparison script.
- **138 aggregate records, 207 per-round scalar pairs, 16,290 raw samples passed**, including basis-sensitive nonuniform SU2, unchanged component-wise rtol=atol=1e-9. Maximum aggregate relative error 4.01e-10. Sample counts, hashes, original round indices and pooled statistics checked exactly. Fresh native regressions 142 passed; benchmark tests 48 passed.

## Monitoring and limitations

Continuous 2s process/thread CPU, RSS, cgroup quota/throttle, per-core busy/steal, frequency and process observations ran off target cores. Peak sampled process RSS **39.19 GiB**; no memory/timeout guard fired. Observed CPU quota unlimited, maximum throttle-counter delta **0**, steal ticks **0** across allocated cores. Frequency snapshots **2.67–3.30 GHz**, not locked; no governor changed. Exclusive Slurm placement does not eliminate OS activity, thermal behavior or cache/history differences across allocations.

Measurement interval: **2026-10-10T21:45:15.272747+00:00–2026-10-11T02:18:11.590269+00:00**. Job cgroup high-water memory observed during these intervals: **40.47 GiB** (the counter may include earlier thread configurations in the same allocation).

Preparation failures occurred before formal sampling. The three formal jobs used immutable scripts and all completed successfully; no formal sample was omitted or replaced. Detailed preparation diagnostics are retained privately.

Backend-process wall total **1.99h**, timed samples **0.93h**, remainder **1.06h** (startup, input/setup, planning/compilation, mandatory call, warmups, validation and teardown). Unchanged adapters do not expose separate compile versus input/setup timers; the remainder is not a compile-only measurement. Process CPU/wall and core busy fractions likewise include untimed work, not steady-kernel occupancy.

Same-version scaling in the [overview](../report.md) is descriptive: different thread counts are measured sequentially, and allocation boundaries remain a limitation. Medians alone are not confidence intervals or a general framework ranking. Larger thread budgets can hurt small or memory-bound cases.

## Reproduction

On equivalent approved dependencies, build this committed revision in release mode. Under an exclusive Slurm allocation with >=96 GiB plus margin and >=8 distinct same-socket cores, first verify quota/topology and bounded independent/native/BLAS gates. Use your allocation's CPU IDs:

```sh
taskset -c 0,1,2,3,4,5,6,7 python -m benchmarks.cross_backend \
  --profile full --threads 8 --warmup 2 --repeat 7 --rounds 3 \
  --sampling-plan benchmarks/cross_backend/results/sampling-plan.json \
  --json-output results.json --markdown report.md --plot latency.svg
```

This public CLI preserves the measurement contract but runs one thread configuration at a time; the retained run interleaved configurations by true round as described above. The shared plan SHA256 is `e8879fce6ef36c9bbe547b015e4a69d25deab49beb670f354d5357b10e6acd91`. Auxiliary private paths are omitted; raw numeric fields/scalars are unchanged.

## Per-workload pooled latency and round ratios

Median / mean values are milliseconds; means include all recorded timings without filtering.
Repeat is per backend/round (three rounds); all records and scalar comparisons passed validation.
Ratios are TK/T0; r0/r1/r2 are paired within-round **median** ratios. `*` marks ratios on both sides of 1, not a confidence interval.
Pooled min/median/inclusive IQR/max, exact sample counts and all samples remain in [JSON](results.json).

Adjacent colored boxes compare Tensor0 and TensorKit for each workload. Rows are MPO/PEPO/MERA; columns are Trivial/Z2/U1/SU2, with the original dtype and dimension labels. The logarithmic latency axis is in milliseconds (lower is faster). Boxes show Q1–Q3 with median lines; whiskers use 1.5× IQR and all outliers remain visible. Open diamonds show the arithmetic mean of all recorded timings, without filtering. The median remains the primary distribution summary; the mean describes tails, not throughput or GC causality.

![Latency distributions with medians and arithmetic means](latency.svg)

### MPO

| Workload | Repeat | T0 median / mean ms | TK median / mean ms | Median ratio | Mean ratio | r0 / r1 / r2 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `tensor_networks.mpo.trivial.float64.d10x4x3` | 100 | 0.028581 / 0.029842 | 0.091745 / 0.103575 | 3.210 | 3.471 | 3.032 / 4.487 / 3.325 |
| `tensor_networks.mpo.trivial.float64.d40x4x3` | 100 | 0.201884 / 0.206016 | 0.612605 / 0.951764 | 3.034 | 4.620 | 3.286 / 3.197 / 2.056 |
| `tensor_networks.mpo.trivial.float64.d160x4x3` | 100 | 1.948784 / 1.951852 | 6.261966 / 11.876530 | 3.213 | 6.085 | 4.848 / 2.586 / 4.279 |
| `tensor_networks.mpo.trivial.float64.d640x4x3` | 15 | 93.694062 / 94.634161 | 162.607234 / 181.235482 | 1.736 | 1.915 | 2.025 / 1.428 / 2.173 |
| `tensor_networks.mpo.trivial.float64.d2560x4x3` | 7 | 4664.905895 / 4649.379226 | 3274.959013 / 3297.805380 | 0.702 | 0.709 | 0.698 / 0.734 / 0.698 |
| `tensor_networks.mpo.trivial.float64.d100x10x10` | 67 | 5.130431 / 5.132359 | 15.973460 / 30.179340 | 3.113 | 5.880 | 3.721 / 2.448 / 3.164 |
| `tensor_networks.mpo.trivial.float64.d200x10x10` | 15 | 33.381863 / 33.425035 | 90.852483 / 118.123861 | 2.722 | 3.534 | 2.725 / 1.507 / 4.145 |
| `tensor_networks.mpo.trivial.float64.d300x20x20` | 7 | 459.482764 / 460.030543 | 701.585554 / 686.058390 | 1.527 | 1.491 | 1.408 / 1.558 / 1.496 |
| `tensor_networks.mpo.z2.float64.d10x4x4` | 100 | 0.053656 / 0.054817 | 0.076985 / 0.082613 | 1.435 | 1.507 | 1.341 / 1.844 / 1.443 |
| `tensor_networks.mpo.z2.float64.d40x4x4` | 100 | 0.222519 / 0.225432 | 0.268096 / 0.344281 | 1.205 | 1.527 | 1.182 / 1.906 / 1.193 |
| `tensor_networks.mpo.z2.float64.d160x4x4` | 100 | 2.716017 / 2.722974 | 2.217612 / 4.530248 | 0.816 | 1.664 | 0.874 / 0.821 / 0.790 |
| `tensor_networks.mpo.z2.float64.d640x4x4` | 15 | 50.353397 / 50.352541 | 99.481364 / 120.812288 | 1.976 | 2.399 | 3.154 / 1.457 / 2.157 |
| `tensor_networks.mpo.z2.float64.d2560x4x4` | 7 | 1870.236565 / 1870.912812 | 1578.636770 / 1589.709799 | 0.844 | 0.850 | 0.821 / 0.889 / 0.841 |
| `tensor_networks.mpo.z2.float64.d100x10x10` | 100 | 3.743933 / 3.759074 | 11.964883 / 16.787008 | 3.196 | 4.466 | 3.658 / 3.132 / 3.062 |
| `tensor_networks.mpo.z2.float64.d200x10x10` | 33 | 16.494624 / 16.515593 | 33.078883 / 51.088991 | 2.005 | 3.093 | 2.222 / 1.879 / 2.010 |
| `tensor_networks.mpo.z2.float64.d300x20x20` | 15 | 217.048496 / 217.336843 | 422.065420 / 420.081731 | 1.945 | 1.933 | 2.026 / 1.953 / 1.878 |
| `tensor_networks.mpo.u1.float64.d40x5x3` | 100 | 0.142354 / 0.143441 | 0.219582 / 0.239955 | 1.543 | 1.673 | 1.517 / 1.997 / 1.530 |
| `tensor_networks.mpo.u1.float64.d160x5x3` | 100 | 1.524179 / 1.528359 | 1.271052 / 2.137220 | 0.834 | 1.398 | 0.858 / 0.916 / 0.815 |
| `tensor_networks.mpo.u1.float64.d640x5x3` | 63 | 11.579512 / 11.525525 | 25.611455 / 39.556573 | 2.212 | 3.432 | 2.696 / 1.725 / 2.645 |
| `tensor_networks.mpo.u1.float64.d2560x5x3` | 15 | 391.806512 / 392.926655 | 564.388619 / 560.948142 | 1.440 | 1.428 | 1.441 / 1.451 / 1.345 |
| `tensor_networks.mpo.u1.float64.d6120x5x3` | 7 | 3628.015019 / 3642.907380 | 2745.654083 / 2751.760086 | 0.757 | 0.755 | 0.757 / 0.784 / 0.738 |
| `tensor_networks.mpo.u1.float64.d200x20x20` | 15 | 24.411274 / 23.601575 | 51.683037 / 71.060300 | 2.117 | 3.011 | 1.912 / 2.487 / 2.227 |
| `tensor_networks.mpo.u1.float64.d400x20x20` | 15 | 133.596632 / 133.687822 | 242.541894 / 249.286757 | 1.815 | 1.865 | 2.081 / 1.483 / 1.901 |
| `tensor_networks.mpo.u1.float64.d400x40x40` | 15 | 537.970125 / 537.512272 | 795.125594 / 775.283353 | 1.478 | 1.442 | 1.488 / 1.513 / 1.403 |
| `tensor_networks.mpo.su2.float64.d40x5x3` | 100 | 0.141427 / 0.146031 | 0.352055 / 1.075321 | 2.489 | 7.364 | 2.744 / 2.573 / 2.214 |
### PEPO

| Workload | Repeat | T0 median / mean ms | TK median / mean ms | Median ratio | Mean ratio | r0 / r1 / r2 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `tensor_networks.pepo.trivial.float64.d3x2x2x50` | 16 | 5.140492 / 5.140296 | 30.597663 / 57.644290 | 5.952 | 11.214 | 5.861 / 5.952 / 10.521 |
| `tensor_networks.pepo.trivial.float64.d3x3x3x100` | 15 | 109.813195 / 109.892457 | 361.208507 / 370.207888 | 3.289 | 3.369 | 3.449 / 3.052 / 3.428 |
| `tensor_networks.pepo.trivial.float64.d4x2x2x50` | 15 | 22.902716 / 22.973151 | 153.651598 / 162.957516 | 6.709 | 7.093 | 6.986 / 5.171 / 8.300 |
| `tensor_networks.pepo.trivial.float64.d4x3x3x100` | 7 | 314.453759 / 314.675830 | 784.600466 / 798.025868 | 2.495 | 2.536 | 2.494 / 2.617 / 2.488 |
| `tensor_networks.pepo.trivial.float64.d5x2x2x50` | 15 | 67.124277 / 67.149337 | 275.224009 / 288.453643 | 4.100 | 4.296 | 4.686 / 3.519 / 3.997 |
| `tensor_networks.pepo.trivial.float64.d5x2x3x100` | 7 | 369.445952 / 369.626770 | 822.254743 / 825.541916 | 2.226 | 2.233 | 2.225 / 2.333 / 2.225 |
| `tensor_networks.pepo.trivial.float64.d6x2x2x50` | 15 | 140.541763 / 140.750372 | 443.970744 / 444.575332 | 3.159 | 3.159 | 3.108 / 3.363 / 2.922 |
| `tensor_networks.pepo.trivial.float64.d6x3x2x100` | 7 | 1391.547567 / 1392.573955 | 2066.342935 / 2099.291087 | 1.485 | 1.507 | 1.469 / 1.540 / 1.470 |
| `tensor_networks.pepo.z2.float64.d4x2x2x50` | 31 | 19.702849 / 19.779073 | 35.600223 / 44.247734 | 1.807 | 2.237 | 1.846 / 1.815 / 1.394 |
| `tensor_networks.pepo.z2.float64.d4x4x4x100` | 7 | 921.543749 / 922.733175 | 1140.114707 / 1150.242746 | 1.237 | 1.247 | 1.272 / 1.284 / 1.208 |
| `tensor_networks.pepo.z2.float64.d5x2x2x50` | 15 | 72.736724 / 72.779746 | 201.797379 / 192.815857 | 2.774 | 2.649 | 2.782 / 2.245 / 3.102 |
| `tensor_networks.pepo.z2.float64.d5x3x4x100` | 7 | 1300.385849 / 1299.581873 | 1311.802823 / 1315.574402 | 1.009 | 1.012 | 1.028 / 1.028 / 0.972 * |
| `tensor_networks.pepo.z2.float64.d6x2x2x50` | 15 | 159.720022 / 159.902252 | 377.664849 / 375.553691 | 2.365 | 2.349 | 2.238 / 2.146 / 2.485 |
| `tensor_networks.pepo.z2.float64.d6x2x4x100` | 7 | 1152.623827 / 1153.796829 | 1383.889499 / 1392.328826 | 1.201 | 1.207 | 1.201 / 1.207 / 1.191 |
| `tensor_networks.pepo.z2.float64.d8x2x2x50` | 15 | 515.145477 / 515.236668 | 791.223400 / 803.087813 | 1.536 | 1.559 | 1.642 / 1.417 / 1.511 |
| `tensor_networks.pepo.z2.float64.d8x3x2x100` | 7 | 4815.412008 / 4819.730569 | 3512.388666 / 3504.827231 | 0.729 | 0.727 | 0.733 / 0.729 / 0.736 |
| `tensor_networks.pepo.u1.float64.d4x2x2x100` | 22 | 46.947485 / 47.408598 | 78.561552 / 109.185143 | 1.673 | 2.303 | 1.922 / 1.411 / 1.611 |
| `tensor_networks.pepo.u1.float64.d4x4x4x200` | 7 | 1559.348516 / 1573.460293 | 1568.500684 / 1579.132868 | 1.006 | 1.004 | 0.996 / 0.961 / 1.029 * |
| `tensor_networks.pepo.u1.float64.d6x2x2x100` | 15 | 256.888830 / 257.437648 | 524.742916 / 512.136999 | 2.043 | 1.989 | 2.054 / 2.026 / 2.042 |
| `tensor_networks.pepo.u1.float64.d6x3x4x200` | 7 | 3960.148677 / 4005.041064 | 3383.678950 / 3401.574609 | 0.854 | 0.849 | 0.868 / 0.799 / 0.891 |
| `tensor_networks.pepo.u1.float64.d8x2x2x100` | 15 | 519.102586 / 519.357691 | 977.382965 / 992.479123 | 1.883 | 1.911 | 1.857 / 1.872 / 1.966 |
| `tensor_networks.pepo.u1.float64.d8x2x4x200` | 7 | 4697.547966 / 4745.725955 | 3647.553338 / 3663.906548 | 0.776 | 0.772 | 0.775 / 0.768 / 0.794 |
| `tensor_networks.pepo.u1.float64.d10x2x2x50` | 15 | 651.509358 / 651.655039 | 1225.272428 / 1227.053222 | 1.881 | 1.883 | 1.899 / 1.835 / 1.910 |
| `tensor_networks.pepo.u1.float64.d10x3x2x100` | 7 | 6183.819247 / 6259.423532 | 6757.892078 / 6764.431660 | 1.093 | 1.081 | 1.093 / 1.029 / 1.135 |
| `tensor_networks.pepo.su2.float64.d4x2x2x100` | 61 | 14.412817 / 14.740870 | 20.985765 / 39.038157 | 1.456 | 2.648 | 1.222 / 1.959 / 1.631 |
### MERA

| Workload | Repeat | T0 median / mean ms | TK median / mean ms | Median ratio | Mean ratio | r0 / r1 / r2 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `tensor_networks.mera.trivial.float64.d2` | 100 | 0.035660 / 0.037485 | 0.321855 / 0.336796 | 9.026 | 8.985 | 9.132 / 10.561 / 8.487 |
| `tensor_networks.mera.trivial.float64.d3` | 100 | 0.059897 / 0.062714 | 0.484388 / 0.484084 | 8.087 | 7.719 | 8.468 / 8.286 / 7.439 |
| `tensor_networks.mera.trivial.float64.d4` | 100 | 0.120260 / 0.122176 | 0.706746 / 1.232121 | 5.877 | 10.085 | 5.464 / 7.662 / 5.821 |
| `tensor_networks.mera.trivial.float64.d8` | 54 | 9.135534 / 9.171304 | 21.920626 / 44.408317 | 2.399 | 4.842 | 2.679 / 2.389 / 2.940 |
| `tensor_networks.mera.trivial.float64.d12` | 15 | 223.001658 / 223.237811 | 483.967261 / 492.152095 | 2.170 | 2.205 | 2.265 / 2.144 / 2.061 |
| `tensor_networks.mera.trivial.float64.d16` | 7 | 3188.718135 / 3184.760479 | 2401.439028 / 2386.192859 | 0.753 | 0.749 | 0.731 / 0.745 / 0.753 |
| `tensor_networks.mera.z2.float64.d2` | 100 | 0.098891 / 0.099530 | 0.409939 / 0.415726 | 4.145 | 4.177 | 4.035 / 4.286 / 4.341 |
| `tensor_networks.mera.z2.float64.d4` | 100 | 0.241406 / 0.245565 | 0.681324 / 0.720921 | 2.822 | 2.936 | 2.800 / 3.405 / 2.773 |
| `tensor_networks.mera.z2.float64.d8` | 100 | 4.819263 / 4.828014 | 9.212779 / 13.837029 | 1.912 | 2.866 | 1.877 / 1.927 / 1.925 |
| `tensor_networks.mera.z2.float64.d12` | 15 | 120.166359 / 120.237469 | 321.056110 / 327.789644 | 2.672 | 2.726 | 2.670 / 2.692 / 2.612 |
| `tensor_networks.mera.z2.float64.d16` | 7 | 1049.790982 / 1049.751508 | 1151.867538 / 1161.603472 | 1.097 | 1.107 | 1.098 / 1.147 / 1.065 |
| `tensor_networks.mera.z2.float64.d20` | 7 | 5881.293595 / 5880.238412 | 5102.836840 / 5133.845635 | 0.868 | 0.873 | 0.864 / 0.869 / 0.872 |
| `tensor_networks.mera.u1.float64.d4` | 100 | 0.348127 / 0.357716 | 1.478022 / 1.497411 | 4.246 | 4.186 | 4.272 / 4.411 / 4.015 |
| `tensor_networks.mera.u1.float64.d8` | 95 | 4.017479 / 4.107849 | 20.123849 / 21.220821 | 5.009 | 5.166 | 4.961 / 5.292 / 4.852 |
| `tensor_networks.mera.u1.float64.d12` | 18 | 34.043990 / 34.103671 | 97.420691 / 103.966374 | 2.862 | 3.049 | 2.836 / 3.045 / 2.846 |
| `tensor_networks.mera.u1.float64.d16` | 15 | 259.652890 / 259.717744 | 402.504653 / 441.062394 | 1.550 | 1.698 | 1.546 / 1.602 / 1.444 |
| `tensor_networks.mera.u1.float64.d22` | 7 | 4797.311873 / 4802.060595 | 4698.917392 / 4720.234095 | 0.979 | 0.983 | 0.980 / 0.969 / 1.006 * |
| `tensor_networks.mera.u1.float64.d28` | 7 | 12601.573476 / 12611.177202 | 11668.018529 / 11684.829301 | 0.926 | 0.927 | 0.931 / 0.915 / 0.931 |
| `tensor_networks.mera.su2.float64.d4` | 100 | 1.061786 / 1.062585 | 1.885178 / 4.853267 | 1.775 | 4.567 | 1.810 / 1.619 / 1.751 |
