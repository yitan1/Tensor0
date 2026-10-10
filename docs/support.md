# Support and compatibility

Tensor0 is experimental, pre-1.0 software. Use it with pinned dependencies and
validate the numerical paths needed by your application; production readiness,
performance and exhaustive coverage are not promised.

## Execution baseline

| Area | Policy |
| --- | --- |
| Numerical runtime | Linux CPU with JAX **0.10.1** and JAXLIB **0.10.1** |
| Python | **CPython 3.11–3.14** tested CPU wheel matrix; **3.11** numerical baseline |
| Published wheels | [**0.1.0** on PyPI](https://pypi.org/project/tensor0/0.1.0/), released **2026-10-10**; Linux **x86_64 CPU**, **glibc >=2.28**; no Rust/C++ compiler needed |
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
