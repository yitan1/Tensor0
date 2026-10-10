# Support and compatibility

Tensor0 is experimental, pre-1.0 software. Use it with pinned dependencies and
validate the numerical paths needed by your application; production readiness,
performance and exhaustive coverage are not promised.

## Execution baseline

| Area | Policy |
| --- | --- |
| Numerical runtime | Linux CPU with JAX **0.10.1** and JAXLIB **0.10.1** |
| Python | **3.11** baseline; other versions are CI-tested only when explicitly recorded for the candidate |
| Native source build | Rust **>=1.87**, C++20 compiler; see [Installation](installation.md) |
| CUDA | Experimental opt-in build with a bounded [operation/dtype/AD matrix](cuda.md); not full API support |
| Other platforms/backends or JAX versions | Not supported numerical execution targets |

The package's Python >=3.11 metadata is an eligibility range, not proof that all
future interpreters work. CI results establish only the versions, platforms and
paths actually executed. Skipped device tests are not acceptance evidence.

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

Before 1.0, public names, signatures, storage conventions and numerical policies
may change between releases. There is no guaranteed deprecation window or
backport/LTS commitment. Changes affecting public use should be recorded in the
[changelog](https://github.com/yitan1/Tensor0/blob/main/CHANGELOG.md), with
migration guidance for breaking changes. Pin the Tensor0 artifact or source
commit as well as its runtime dependencies for reproducibility. A future 1.0
compatibility policy must be stated explicitly; it is not implied here.

## Reporting problems

Open a [GitHub issue](https://github.com/yitan1/Tensor0/issues) with a minimal
reproducer, expected and actual behavior, Tensor0 version and source commit,
Python/JAX/JAXLIB versions, OS/architecture, device/backend, storage and
coefficient dtypes, x64/matmul precision settings, and whether JIT/AD is used.
For builds, include Rust/compiler versions, build command, job count and memory
limit. Unsupported environments can be reported, but a report is not a promise
to expand support or provide a fix on a deadline.
