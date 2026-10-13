# Support and compatibility

Tensor0 is experimental, pre-1.0 software. Use it with pinned dependencies and
validate the numerical paths needed by your application; production readiness,
performance and exhaustive coverage are not promised.

## Published 0.2.0 baseline

| Area | Policy |
| --- | --- |
| Released numerical runtime | Linux x86_64 CPU (glibc **>=2.28**) and macOS **15+ ARM64 CPU**, with JAX **0.10.1** and JAXLIB **0.10.1** |
| Python | **CPython 3.11–3.14** tested CPU wheel matrix on both platforms; **3.11** numerical baseline |
| Published wheels | [**0.2.0** on PyPI](https://pypi.org/project/tensor0/0.2.0/), released **2026-10-12**; eight core wheels, including hosted macOS 15 acceptance; no Rust/C++ compiler needed |
| Native source build | Rust **>=1.87**, C++20 compiler; see [Installation](installation.md) |
| CUDA | Published, experimental Linux x86_64 `tensor0-cuda12` **0.2.0** companion; bounded A100 (`sm_80`) [operation/dtype/AD matrix](cuda.md), not full API support; no macOS CUDA |
| Other platforms/backends or JAX versions | Not supported numerical execution targets |

The released 0.2.0 package's Python >=3.11 metadata is an eligibility range, not proof that all
future interpreters work. CI results establish only the versions, platforms and
paths actually executed. Skipped device tests are not acceptance evidence.

## Current source-development target: 0.3.0 (unreleased)

| Area | Target policy (not acceptance evidence) |
| --- | --- |
| Runtime | Exact JAX **0.11.2** and JAXLIB **0.11.2**, matching vendored FFI headers and rebuilt native binaries |
| Python | Python **>=3.12**; ordinary CPython **3.12–3.15** wheel targets; no free-threaded support |
| CPU platforms | Linux **x86_64**, glibc **>=2.28**; macOS **15+ ARM64 CPU** |
| Verification | Daily Linux CPU CI endpoints **3.12/3.15**; all eight release wheels require installed-artifact acceptance |
| Native build tools | Rust **>=1.87**, C++20, PyO3/Rust `numpy` **0.29**, maturin **>=1.15.0,<2** (fixed validation tool **1.15.0**) |
| CUDA 12 | Existing experimental Linux x86_64 operation/dtype/AD scope only; freshly rebuilt exact-version core/addon pair requires CPU/GPU acceptance |

The new matrix is a target, not a statement that builds, hosted macOS jobs or
CUDA execution have passed. Published 0.2.0 evidence does not validate 0.3.0 or
JAX 0.11.2. Python 3.11 users must stay on the pinned 0.2.0 runtime or migrate
to Python 3.12+; see [source installation and migration](installation.md#source-install).
Python >=3.12 metadata does not promise arbitrary future interpreter support.
Rebuild native binaries with the exact headers/runtime; upgrading JAX alone is
not a supported migration.

Unavailable native stride routes fail explicitly; Tensor0 does not silently
substitute element-address gather/scatter or transfer unsupported CUDA work to
CPU. Native FFI operations appear as `stablehlo.custom_call` in compiler IR;
there is no separate pure-StableHLO stride backend.

JAX x64 and matmul precision settings affect dtypes and numerical accuracy.
Numerical acceptance uses operation-appropriate tolerances, not general bitwise
agreement across devices, compiler versions or worker counts. See the
[contraction numerical contract](contractions.md#single-tensor-trace),
[usage boundaries](usage.md#current-boundaries) and CUDA matrix for restrictions.

## Public API before 1.0

The [API inventory](api.md) identifies public exports from `tensor0` and
`tensor0.structure`; the usage and contraction guides describe their contracts.
Underscore-prefixed modules, native registration/FFI interfaces and internal
layouts not documented as public are implementation details. Do not depend on
their import paths, binary ABI or serialization across versions.

Before 1.0, patch releases (`0.x.y` to `0.x.(y+1)`) preserve backward
compatibility of the documented public API. Breaking public API changes require
a minor release (`0.x` to `0.(x+1)`) and migration guidance in the
[changelog](https://github.com/yitan1/Tensor0/blob/main/CHANGELOG.md). This policy
covers documented names, signatures and behavior, not internal layouts, binary
ABI, execution plans or performance. Bug fixes may correct behavior that
violates the documented contract; numerical results remain subject to the
stated tolerances rather than bitwise stability.

There is no guaranteed deprecation window or backport/LTS commitment. Pin the
Tensor0 artifact or source commit as well as its runtime dependencies for
reproducibility. A future 1.0 compatibility policy must be stated explicitly;
it is not implied here.

## Reporting problems

Open a [GitHub issue](https://github.com/yitan1/Tensor0/issues) with a minimal
reproducer, expected and actual behavior, Tensor0 version and source commit,
Python/JAX/JAXLIB versions, OS/architecture, device/backend, storage and
coefficient dtypes, x64/matmul precision settings, and whether JIT/AD is used.
For builds, include Rust/compiler versions, build command, job count and memory
limit. Unsupported environments can be reported, but a report is not a promise
to expand support or provide a fix on a deadline.
