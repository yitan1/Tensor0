# Full Cross-Backend CPU — 1 Worker

[All raw samples](results.json) · [Latency distributions](latency.svg) · [Shared sampling plan](../sampling-plan.json) · [Overview](../report.md)

**69 workloads (66 reference dimension cases + 3 nonuniform SU2 extensions), same committed source and native build at 1/4/8t.** Equal-workload geometric mean TK/T0 latency ratios: **median 1.185×; mean 1.465×**. Values >1 favor Tensor0. These are workload-weighted descriptive ratios, not total time, throughput or significance.

| Round (zero-based) | Tensor0 lower median | Geometric mean TK/T0 |
| --- | ---: | ---: |
| 0 | 38/69 | 1.192× |
| 1 | 38/69 | 1.198× |
| 2 | 38/69 | 1.179× |

Observed direction reversals: **6/69**, flagged below; raw within-round medians, pooled min/median/inclusive IQR/max and all samples are in JSON. No observed reversal is not a claim of significance or future stability.

## Identity, resources and measurement

- Source `387b2b8c4eeebeb092a96158236315cc2e8a1129`, committed archive, all 160 production file hashes verified. Fresh release native SHA256 `a5e0c34f4179a94e8a8552a16441237c9dc81a68b7e9c8e06f30b4440d017638`. As-run public runner unchanged; no dirty production, GEMM strategy edits or benchmark patches. Rust 1.89/GCC 15.3/maturin 1.13.1; approved offline dependencies unchanged. GCC 15 g++ linker; `RUSTFLAGS="-C link-arg=-lstdc++ -C link-arg=-lgcc"` and final root-crate flags `-C link-arg=-Wl,--no-as-needed -C link-arg=-Wl,--undefined=__cpu_features2 -C link-arg=-lstdc++ -C link-arg=-lgcc`; no LD_PRELOAD. Distribution 0.1.0 still exports literal 0.0.0 runtime version in this source; the commit/hash, not that exported string, identifies the build.
- Python 3.14.6, JAX/JAXlib 0.10.1, NumPy 2.4.6; Julia 1.12.5, TensorKit 0.16.5, TensorOperations 5.6.1, Strided 2.5.0. Actual Julia BLAS configuration `LBTConfig([ILP64] libopenblas64_.so)`; library hashes retained in JSON.
- Slurm node **n002**, E5-2680v3 (Haswell-EP), 24 physical cores/two sockets, SMT off. Exclusive 24-core allocation, 104 GiB job memory on 128 GiB-class hardware with 20 GiB OS reservation. Measurement affinity **[0]**, nested 1/4/8 same-socket distinct physical cores; monitor on CPU 8. Three sequential exclusive round jobs **993086, 993087, 993088**, thread order 1/4/8 then 4/8/1 then 8/1/4; backend/workload forward/reverse/forward. Allocation identities and gates checked each time.
- Before timing, independent pinned 4/8-process gates reached approximately 4/8 core-equivalents; native reduction and Julia/BLAS multiplication showed actual worker-thread CPU time with queried 4/8 limits. Gates finished before samples and prove usable resources/representative parallelism, not saturation of every workload. Queried runtime limits match every request.
- Per-workload mandatory first invocation + **two warmups**, then frozen **7–100 timed calls per round** according to the [shared plan](../sampling-plan.json), identical for both backends and all thread counts. Slowest of six complete preflight medians >=2s gets 7; other counts are clamp(ceil(2s/t_slow),15,100). The threshold selects a cost tier; the 2s numerator is a bounded target, not a guarantee of precision or sample duration. Preflight is separate (one round, 1/3 timed calls by risk), never pooled with formal samples. Counts are not adapted to speedup, winners or observed formal results.
- Prepared contraction execution and completion are timed. Inputs, planning, compilation, mandatory call and warmups are outside timing; scalar validation reuses the final timed output without extra contractions. Natural GC tails are retained: no sample filtering or selective reruns. The [pinned upstream network-comparison script](https://github.com/QuantumKitHub/TensorKit.jl/blob/cfaa073e4d1e3eb2167edcbdc3be9872f41e7d91/benchmark/comparisons/run_tensorkit.jl) instead sets **K=100** outer samples, with [one initial compilation call, inner=1](https://github.com/QuantumKitHub/TensorKit.jl/blob/cfaa073e4d1e3eb2167edcbdc3be9872f41e7d91/benchmark/comparisons/benchtools.jl), reports averages and separately records GC time. We add two explicit warmups, use fixed 7–100 repeats per round and three paired rounds, report pooled medians (now supplemented by derived unfiltered means), and retain GC in wall samples without subtracting it. Thus heavy cases have 21 rather than 100 timed calls, and 100-repeat cases have 300 across three processes, not an identical sampling policy. The separate upstream BenchmarkTools suite (20s/10000-sample limits) is not that network-comparison script.
- **138 aggregate records, 207 per-round scalar pairs, 16,290 raw samples passed**, including basis-sensitive nonuniform SU2, unchanged component-wise rtol=atol=1e-9. Maximum aggregate relative error 4.01e-10. Sample counts, hashes, original round indices and pooled statistics checked exactly. Fresh native regressions 142 passed; benchmark tests 48 passed.

## Monitoring and limitations

Continuous 2s process/thread CPU, RSS, cgroup quota/throttle, per-core busy/steal, frequency and process observations ran off target cores. Peak sampled process RSS **38.41 GiB**; no memory/timeout guard fired. Observed CPU quota unlimited, maximum throttle-counter delta **0**, steal ticks **0** across allocated cores. Frequency snapshots **2.87–3.30 GHz**, not locked; no governor changed. Exclusive Slurm placement does not eliminate OS activity, thermal behavior or cache/history differences across allocations.

Measurement interval: **2026-10-10T19:15:15.280461+00:00–2026-10-11T03:58:34.411786+00:00**. Job cgroup high-water memory observed during these intervals: **40.47 GiB** (the counter may include earlier thread configurations in the same allocation).

Preparation failures occurred before formal sampling. The three formal jobs used immutable scripts and all completed successfully; no formal sample was omitted or replaced. Detailed preparation diagnostics are retained privately.

Backend-process wall total **5.03h**, timed samples **3.07h**, remainder **1.96h** (startup, input/setup, planning/compilation, mandatory call, warmups, validation and teardown). Unchanged adapters do not expose separate compile versus input/setup timers; the remainder is not a compile-only measurement. Process CPU/wall and core busy fractions likewise include untimed work, not steady-kernel occupancy.

Same-version scaling in the [overview](../report.md) is descriptive: different thread counts are measured sequentially, and allocation boundaries remain a limitation. Medians alone are not confidence intervals or a general framework ranking. Larger thread budgets can hurt small or memory-bound cases.

## Reproduction

On equivalent approved dependencies, build this committed revision in release mode. Under an exclusive Slurm allocation with >=96 GiB plus margin and >=8 distinct same-socket cores, first verify quota/topology and bounded independent/native/BLAS gates. Use your allocation's CPU IDs:

```sh
taskset -c 0 python -m benchmarks.cross_backend \
  --profile full --threads 1 --warmup 2 --repeat 7 --rounds 3 \
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
| `tensor_networks.mpo.trivial.float64.d10x4x3` | 100 | 0.076287 / 0.078142 | 0.097306 / 0.105684 | 1.276 | 1.352 | 1.298 / 1.656 / 1.134 |
| `tensor_networks.mpo.trivial.float64.d40x4x3` | 100 | 0.246719 / 0.249192 | 0.417034 / 2.005377 | 1.690 | 8.048 | 1.712 / 2.531 / 1.643 |
| `tensor_networks.mpo.trivial.float64.d160x4x3` | 100 | 6.531921 / 6.549184 | 8.310325 / 13.097652 | 1.272 | 2.000 | 1.659 / 1.246 / 1.376 |
| `tensor_networks.mpo.trivial.float64.d640x4x3` | 15 | 545.933357 / 545.436644 | 394.427495 / 474.421367 | 0.722 | 0.870 | 0.733 / 0.706 / 0.829 |
| `tensor_networks.mpo.trivial.float64.d2560x4x3` | 7 | 32423.588623 / 32491.925718 | 20992.575968 / 21164.219484 | 0.647 | 0.651 | 0.638 / 0.668 / 0.647 |
| `tensor_networks.mpo.trivial.float64.d100x10x10` | 67 | 21.741362 / 22.009760 | 29.482442 / 35.444899 | 1.356 | 1.610 | 1.304 / 1.143 / 1.394 |
| `tensor_networks.mpo.trivial.float64.d200x10x10` | 15 | 163.705202 / 163.468514 | 156.080490 / 214.714358 | 0.953 | 1.313 | 0.983 / 0.938 / 0.943 |
| `tensor_networks.mpo.trivial.float64.d300x20x20` | 7 | 2713.057198 / 2748.212686 | 2566.942366 / 2605.176175 | 0.946 | 0.948 | 0.909 / 1.055 / 0.947 * |
| `tensor_networks.mpo.z2.float64.d10x4x4` | 100 | 0.104539 / 0.105282 | 0.077020 / 0.082892 | 0.737 | 0.787 | 0.734 / 0.886 / 0.720 |
| `tensor_networks.mpo.z2.float64.d40x4x4` | 100 | 0.278431 / 0.279515 | 0.274620 / 1.514765 | 0.986 | 5.419 | 0.975 / 1.507 / 0.997 * |
| `tensor_networks.mpo.z2.float64.d160x4x4` | 100 | 4.668016 / 4.773938 | 3.983674 / 6.403971 | 0.853 | 1.341 | 0.793 / 0.866 / 0.854 |
| `tensor_networks.mpo.z2.float64.d640x4x4` | 15 | 213.020704 / 215.835390 | 155.053676 / 199.000925 | 0.728 | 0.922 | 0.692 / 0.741 / 0.720 |
| `tensor_networks.mpo.z2.float64.d2560x4x4` | 7 | 11933.906847 / 11913.923274 | 8027.235970 / 8098.907232 | 0.673 | 0.680 | 0.671 / 0.679 / 0.671 |
| `tensor_networks.mpo.z2.float64.d100x10x10` | 100 | 8.594289 / 8.624152 | 9.309287 / 14.894913 | 1.083 | 1.727 | 1.084 / 1.078 / 1.087 |
| `tensor_networks.mpo.z2.float64.d200x10x10` | 33 | 61.400994 / 61.594198 | 48.862902 / 71.951869 | 0.796 | 1.168 | 0.802 / 0.785 / 0.784 |
| `tensor_networks.mpo.z2.float64.d300x20x20` | 15 | 1004.838792 / 992.328144 | 968.203171 / 932.383675 | 0.964 | 0.940 | 1.032 / 0.804 / 1.027 * |
| `tensor_networks.mpo.u1.float64.d40x5x3` | 100 | 0.193437 / 0.195579 | 0.219638 / 0.240455 | 1.135 | 1.229 | 1.130 / 1.495 / 1.097 |
| `tensor_networks.mpo.u1.float64.d160x5x3` | 100 | 1.202533 / 1.208465 | 1.387685 / 2.460230 | 1.154 | 2.036 | 1.068 / 1.922 / 1.046 |
| `tensor_networks.mpo.u1.float64.d640x5x3` | 63 | 26.588278 / 26.963947 | 27.789989 / 47.203048 | 1.045 | 1.751 | 1.059 / 1.019 / 1.063 |
| `tensor_networks.mpo.u1.float64.d2560x5x3` | 15 | 1681.974669 / 1687.112528 | 1487.419088 / 1434.894706 | 0.884 | 0.851 | 0.891 / 0.764 / 0.841 |
| `tensor_networks.mpo.u1.float64.d6120x5x3` | 7 | 20718.513172 / 20711.677006 | 13401.995211 / 13593.487287 | 0.647 | 0.656 | 0.644 / 0.676 / 0.647 |
| `tensor_networks.mpo.u1.float64.d200x20x20` | 15 | 63.439729 / 63.583162 | 48.003207 / 50.726204 | 0.757 | 0.798 | 0.743 / 0.872 / 0.758 |
| `tensor_networks.mpo.u1.float64.d400x20x20` | 15 | 355.330463 / 355.788699 | 313.156520 / 374.291117 | 0.881 | 1.052 | 0.879 / 0.764 / 0.904 |
| `tensor_networks.mpo.u1.float64.d400x40x40` | 15 | 1858.668350 / 1859.118705 | 1770.648997 / 1798.680202 | 0.953 | 0.967 | 0.954 / 1.114 / 0.950 * |
| `tensor_networks.mpo.su2.float64.d40x5x3` | 100 | 0.214822 / 0.217293 | 0.377534 / 1.228791 | 1.757 | 5.655 | 1.749 / 1.701 / 1.798 |
### PEPO

| Workload | Repeat | T0 median / mean ms | TK median / mean ms | Median ratio | Mean ratio | r0 / r1 / r2 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `tensor_networks.pepo.trivial.float64.d3x2x2x50` | 16 | 17.584968 / 17.581274 | 34.640009 / 75.742884 | 1.970 | 4.308 | 2.187 / 1.481 / 2.034 |
| `tensor_networks.pepo.trivial.float64.d3x3x3x100` | 15 | 446.977482 / 447.646609 | 962.259863 / 897.565807 | 2.153 | 2.005 | 2.223 / 1.538 / 2.130 |
| `tensor_networks.pepo.trivial.float64.d4x2x2x50` | 15 | 86.900435 / 86.867047 | 128.994187 / 203.623442 | 1.484 | 2.344 | 1.519 / 1.491 / 1.356 |
| `tensor_networks.pepo.trivial.float64.d4x3x3x100` | 7 | 1369.986210 / 1370.522764 | 2434.449102 / 2505.794647 | 1.777 | 1.828 | 1.683 / 1.773 / 1.950 |
| `tensor_networks.pepo.trivial.float64.d5x2x2x50` | 15 | 256.093394 / 256.258147 | 436.886122 / 519.864437 | 1.706 | 2.029 | 1.778 / 1.707 / 1.692 |
| `tensor_networks.pepo.trivial.float64.d5x2x3x100` | 7 | 1538.375353 / 1536.415120 | 2658.688725 / 2641.559009 | 1.728 | 1.719 | 1.599 / 1.712 / 1.764 |
| `tensor_networks.pepo.trivial.float64.d6x2x2x50` | 15 | 562.257977 / 562.150428 | 1155.924297 / 1117.265151 | 2.056 | 1.987 | 2.058 / 2.237 / 2.004 |
| `tensor_networks.pepo.trivial.float64.d6x3x2x100` | 7 | 6001.574458 / 6018.547265 | 8414.805957 / 8561.672464 | 1.402 | 1.423 | 1.378 / 1.500 / 1.403 |
| `tensor_networks.pepo.z2.float64.d4x2x2x50` | 31 | 66.286939 / 66.383011 | 58.956914 / 101.889492 | 0.889 | 1.535 | 0.933 / 0.823 / 0.886 |
| `tensor_networks.pepo.z2.float64.d4x4x4x100` | 7 | 3417.508558 / 3416.288798 | 2858.281634 / 2808.157749 | 0.836 | 0.822 | 0.838 / 0.765 / 0.837 |
| `tensor_networks.pepo.z2.float64.d5x2x2x50` | 15 | 206.219111 / 206.235183 | 156.171111 / 263.311573 | 0.757 | 1.277 | 0.763 / 0.762 / 0.757 |
| `tensor_networks.pepo.z2.float64.d5x3x4x100` | 7 | 4106.468306 / 4102.759677 | 3473.211050 / 3551.366131 | 0.846 | 0.866 | 0.847 / 0.911 / 0.834 |
| `tensor_networks.pepo.z2.float64.d6x2x2x50` | 15 | 445.063346 / 444.917730 | 457.098291 / 552.524471 | 1.027 | 1.242 | 1.112 / 1.018 / 1.028 |
| `tensor_networks.pepo.z2.float64.d6x2x4x100` | 7 | 3549.200429 / 3544.536878 | 3215.619713 / 3200.313111 | 0.906 | 0.903 | 0.911 / 0.854 / 0.908 |
| `tensor_networks.pepo.z2.float64.d8x2x2x50` | 15 | 1451.811092 / 1452.465725 | 1610.650935 / 1636.280130 | 1.109 | 1.127 | 1.109 / 1.203 / 1.080 |
| `tensor_networks.pepo.z2.float64.d8x3x2x100` | 7 | 14972.696970 / 14956.750261 | 11839.272950 / 11955.782535 | 0.791 | 0.799 | 0.789 / 0.812 / 0.791 |
| `tensor_networks.pepo.u1.float64.d4x2x2x100` | 22 | 91.900250 / 91.964720 | 80.518895 / 142.561369 | 0.876 | 1.550 | 0.883 / 0.763 / 0.888 |
| `tensor_networks.pepo.u1.float64.d4x4x4x200` | 7 | 3297.470993 / 3294.159634 | 3139.374253 / 3050.705101 | 0.952 | 0.926 | 0.960 / 0.834 / 0.954 |
| `tensor_networks.pepo.u1.float64.d6x2x2x100` | 15 | 494.739579 / 495.322286 | 538.428439 / 655.083242 | 1.088 | 1.323 | 1.108 / 0.817 / 1.089 * |
| `tensor_networks.pepo.u1.float64.d6x3x4x200` | 7 | 9060.278650 / 9057.822569 | 6904.537081 / 6736.461915 | 0.762 | 0.744 | 0.765 / 0.720 / 0.764 |
| `tensor_networks.pepo.u1.float64.d8x2x2x100` | 15 | 1343.912255 / 1343.670903 | 1629.922609 / 1633.696675 | 1.213 | 1.216 | 1.208 / 1.241 / 1.200 |
| `tensor_networks.pepo.u1.float64.d8x2x4x200` | 7 | 11054.051628 / 11046.385469 | 8465.251915 / 8499.124974 | 0.766 | 0.769 | 0.765 / 0.775 / 0.761 |
| `tensor_networks.pepo.u1.float64.d10x2x2x50` | 15 | 1799.099265 / 1799.277900 | 2018.388475 / 2013.150609 | 1.122 | 1.119 | 1.104 / 1.124 / 1.105 |
| `tensor_networks.pepo.u1.float64.d10x3x2x100` | 7 | 15565.998526 / 15556.622652 | 12659.017809 / 12644.897509 | 0.813 | 0.813 | 0.808 / 0.815 / 0.814 |
| `tensor_networks.pepo.su2.float64.d4x2x2x100` | 61 | 24.564713 / 24.754855 | 32.117836 / 55.919188 | 1.307 | 2.259 | 1.176 / 1.311 / 1.129 |
### MERA

| Workload | Repeat | T0 median / mean ms | TK median / mean ms | Median ratio | Mean ratio | r0 / r1 / r2 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `tensor_networks.mera.trivial.float64.d2` | 100 | 0.087411 / 0.087756 | 0.326792 / 0.342634 | 3.739 | 3.904 | 3.739 / 4.456 / 3.658 |
| `tensor_networks.mera.trivial.float64.d3` | 100 | 0.108405 / 0.109491 | 0.498414 / 0.499043 | 4.598 | 4.558 | 4.688 / 4.634 / 4.591 |
| `tensor_networks.mera.trivial.float64.d4` | 100 | 0.169402 / 0.170109 | 0.813621 / 0.820786 | 4.803 | 4.825 | 4.811 / 5.189 / 4.552 |
| `tensor_networks.mera.trivial.float64.d8` | 54 | 32.729096 / 33.887668 | 46.095063 / 65.269638 | 1.408 | 1.926 | 1.488 / 1.268 / 1.530 |
| `tensor_networks.mera.trivial.float64.d12` | 15 | 1120.100971 / 1132.679735 | 1338.167336 / 1242.994924 | 1.195 | 1.097 | 1.404 / 1.148 / 1.011 |
| `tensor_networks.mera.trivial.float64.d16` | 7 | 16229.933901 / 16381.851301 | 11431.665318 / 11446.900430 | 0.704 | 0.699 | 0.714 / 0.665 / 0.716 |
| `tensor_networks.mera.z2.float64.d2` | 100 | 0.145963 / 0.147581 | 0.410137 / 0.411899 | 2.810 | 2.791 | 2.798 / 2.847 / 2.767 |
| `tensor_networks.mera.z2.float64.d4` | 100 | 0.314204 / 0.318796 | 0.668702 / 0.697348 | 2.128 | 2.187 | 2.127 / 2.400 / 2.086 |
| `tensor_networks.mera.z2.float64.d8` | 100 | 11.934006 / 11.983194 | 14.955475 / 22.524567 | 1.253 | 1.880 | 1.246 / 1.259 / 1.255 |
| `tensor_networks.mera.z2.float64.d12` | 15 | 468.732251 / 468.840724 | 371.317668 / 528.075641 | 0.792 | 1.126 | 0.796 / 0.772 / 0.831 |
| `tensor_networks.mera.z2.float64.d16` | 7 | 4314.003383 / 4378.150958 | 4300.297923 / 4260.362065 | 0.997 | 0.973 | 1.005 / 0.905 / 1.002 * |
| `tensor_networks.mera.z2.float64.d20` | 7 | 29107.110309 / 29367.476011 | 21589.048827 / 21233.960796 | 0.742 | 0.723 | 0.758 / 0.720 / 0.765 |
| `tensor_networks.mera.u1.float64.d4` | 100 | 0.366322 / 0.368488 | 1.525123 / 1.532152 | 4.163 | 4.158 | 4.152 / 4.162 / 4.163 |
| `tensor_networks.mera.u1.float64.d8` | 95 | 5.208359 / 5.225677 | 19.557408 / 21.524371 | 3.755 | 4.119 | 3.777 / 3.629 / 3.819 |
| `tensor_networks.mera.u1.float64.d12` | 18 | 90.654840 / 90.717783 | 113.402844 / 125.204473 | 1.251 | 1.380 | 1.265 / 1.285 / 1.239 |
| `tensor_networks.mera.u1.float64.d16` | 15 | 727.718152 / 727.900888 | 935.158285 / 911.420775 | 1.285 | 1.252 | 1.260 / 1.289 / 1.291 |
| `tensor_networks.mera.u1.float64.d22` | 7 | 15943.027916 / 15950.624379 | 13068.019890 / 12748.711635 | 0.820 | 0.799 | 0.820 / 0.823 / 0.784 |
| `tensor_networks.mera.u1.float64.d28` | 7 | 44959.552260 / 44894.159711 | 33946.462201 / 34155.980914 | 0.755 | 0.761 | 0.756 / 0.735 / 0.762 |
| `tensor_networks.mera.su2.float64.d4` | 100 | 0.970698 / 0.976728 | 1.765493 / 4.969406 | 1.819 | 5.088 | 1.777 / 1.943 / 1.794 |
