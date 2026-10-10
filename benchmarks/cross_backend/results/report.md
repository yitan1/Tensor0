# Cross-Backend CPU Results

Retained **medium** configurations: 36 workloads, two backends, warmup 2 /
repeat 7 / three alternating paired rounds; 21 samples per backend/workload.

| Configuration | Source | Host / affinity | Report | Raw samples | Plot |
| --- | --- | --- | --- | --- | --- |
| One worker | `333de65` | Slurm n008, Xeon E5-2680 v3, CPU0 | [Report](cpu-1t/report.md) | [JSON](cpu-1t/results.json) | [SVG](cpu-1t/latency.svg) |
| Four workers | `e5bd1f7` | Slurm n008, same-socket CPUs 0,1,2,4 | [Report](cpu-4t/report.md) | [JSON](cpu-4t/results.json) | [SVG](cpu-4t/latency.svg) |
| Eight workers | `e5bd1f7` | Slurm n008, same-socket CPUs 0,1,2,4,5,6,7,8 | [Report](cpu-8t/report.md) | [JSON](cpu-8t/results.json) | [SVG](cpu-8t/latency.svg) |

All configurations passed 72 aggregate records and 108 per-round scalar pairs;
all **1,512 samples per configuration** are retained without filtering. They
use committed production snapshots plus benchmark-only worker/affinity metadata
and Julia startup suppression. See each report for full source and artifact
hashes, environment, resource gates, validation and limitations.

| Workers | Tensor0 lower pooled median | TensorKit/Tensor0 geometric mean ratio |
| --- | ---: | ---: |
| 1 | 31/36 | 1.608× |
| 4 | 36/36 | 2.553× |
| 8 | 34/36 | 2.990× |

## Four/eight-worker resource verification

The new 4t/8t measurements ran sequentially in one Slurm allocation, using the
same node, committed source, release native artifact and environment. SMT is
disabled; eight distinct physical cores on one socket were verified within the
allocation's cpuset. Before timing, synchronous pinned CPU-bound processes
achieved **3.991/7.981 core-equivalents** for 4/8 workers. Tensor0 native
independent-output reduction achieved **3.096/5.685 CPU/wall**, with four/eight
substantial worker CPU-time deltas; TensorKit multiplication achieved
**3.976/7.938 CPU/wall** with queried Julia/BLAS counts 4/8. All readable cgroup
ancestors were inspected; no quota throttling or steal was observed in gates
or measurement. These checks demonstrate available resources and real runtime
parallelism, not full occupancy of every workload.

Descriptively, the 4t median divided by the 8t median has geometric mean
**1.163× for Tensor0** (23/36 workloads faster at 8t) and **0.993× for
TensorKit** (22/36 faster). The sequential runs are not randomized scaling
trials. Small tasks, conservative scheduling, memory bandwidth, preparation and
serial work limit utilization; more threads do not guarantee lower latency.

**The one-worker snapshot uses the older `333de65` revision. It is not a
same-version 1→4→8 scaling baseline**, even though the node model is the same.
All snapshots use a shared node and unlocked frequency; cache/memory
interference cannot be excluded. Winner labels are descriptive pooled medians,
not significance tests. The former local Ryzen four-worker snapshot has been
replaced; the standard suite remains an independent local snapshot, unchanged.

See the [suite documentation](../README.md) for the measurement contract.
