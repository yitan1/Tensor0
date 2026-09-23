# Development

Tensor0 is developed from a source checkout with Python, Rust, uv, maturin, and
pytest.

Building the Linux Native backend requires a C++20-capable compiler. The build
uses `/usr/bin/c++` by default; set `CXX` to select another compiler.
C++ optimization follows Cargo's profile optimization level: ordinary
`maturin develop` uses `-O0`, while `maturin develop --release` uses `-O3`
with the default profiles. Size levels `s` and `z` both use `-Os`.
Set `TENSOR0_CXX_OPT_LEVEL` to `0`, `1`, `2`, `3`, `s`, or `z` to override
only the C++ optimization level, leaving Rust's profile unchanged. For example,
to compare C++ O2 with the default O3 release build:

```bash
TENSOR0_CXX_OPT_LEVEL=2 uv run maturin develop --release
```

Changing or removing the override invalidates the native object cache when the
compiler command changes. The build script does not read `CXXFLAGS`.
Benchmark representative kernels before choosing a lower optimization level for
production; O2 is not guaranteed to preserve O3 performance.
The Native backend compiles ordinary layout, descriptor/prepared-state, validation,
and scheduling implementations as separate objects. Numerical templates are
instantiated in four operation-owned units: `native/ffi/copy.cc`,
`update.cc`, `reduction.cc` (including Accumulation), and `dot.cc`.
Their typed inner kernels remain visible through headers for inlining.
Shared lifecycle state and ordinary support implementations have one definition.
Builds are serial and reuse each object independently using compiler-generated
dependency files, dependency contents,
compiler identity/command and build-script inputs. Implementation-only edits to an
ordinary `.cc` do not rebuild the numerical objects. Operation-specific `.cc` or
execution-header edits rebuild only that operation; shared kernel headers can
rebuild multiple operations.
Missing objects/dependency files and failed compilations invalidate the affected
cache entry. Compiler paths and discovered dependencies are watched by Cargo.
Use a release build for performance measurements.

## Local Setup

```bash
uv sync --group dev
uv run maturin develop
```

## Verification

```bash
cargo test
uv run pytest tests -q
uv run python examples/basic_usage.py
uv run python examples/contractions.py
```

Native regression tests live in `tests/stride`, including the standalone C++
sources in `tests/stride/cpp`. On Linux, `native/test_cpp.py` compiles them with a C++20
compiler and UndefinedBehaviorSanitizer; set `CXX` to select the compiler.
The tests cover scalar promotion, expression stages, address planning, SIMD
tails, skipped reads, asynchronous task ownership, and validation before writes.
The FFI boundary unit compiles the current binding functions without expanding
every exported dtype handler; registration is checked separately in Python.

Standalone C++ objects and executables are cached in pytest's cache directory
(`.pytest_cache/d/stride-cpp` by default). Every invocation still runs all selected
executables, including UBSan checks and the JAX promotion oracle. Native sources,
vendored headers, shared test headers, compiler identity, compile flags, and
compiler search-path environment changes invalidate the cached builds. Individual
C++ test source changes rebuild that contract. Failed builds are not cached.
Use `uv run pytest tests/stride/native/test_cpp.py --cache-clear -q` for a clean build,
including after changes to system headers or libraries outside the checkout.

### Stride coverage policy

Stride uses fixed, explicit representative combinations alongside broad type-rule
checks and dedicated numerical regressions. It does not exhaustively cross every
dtype, layout, batch shape, coefficient form and derivative order. The parameter
tables in each test owner define the exact coverage; there is no random sampler
or default fast/slow exclusion. `uv run pytest tests -q` remains the complete
Python-suite command, including root TensorMap integration.

| Layer | Coverage strategy |
| --- | --- |
| Metadata and C++ numeric rules | Broad dtype-resolution, promotion and conversion tables, with separate rounding, overflow and special-value checks |
| Native coefficient execution | Representative source/coefficient/result triples; all mixed-storage zero/unit/general branch patterns unbatched, with selected nonempty and empty batches |
| Reduction FFI | Every source/result pair without a coefficient; every coefficient dtype when source equals result or either storage dtype is float32 |
| Public Dot and Materialize | Every same-type pair plus selected mixed pairs in both directions; separate default/explicit dtype, layout, conversion-boundary and lowering checks |
| JAX and public API AD | Representative precision, real/complex, layout, coefficient-form and batch combinations, preserving each surviving case's derivative checks |
| Packed trace adapters | Typed-coefficient and mixed-storage representatives, repeated-read and higher-derivative checks; separate real TensorMap integration tests |

The Reduction FFI rule retains complete source/coefficient and result/coefficient
pair coverage, but **not** every typed source/result pair or three-way interaction.
Missing coefficients and typed coefficients are distinct ABI paths. The separate
FFI batch matrix keeps float32 source/result storage while varying coefficient
dtype, scalar/singleton/batched form and empty/nonempty batch shapes.

Coefficient AD uses boolean, signed and unsigned representatives rather than a
full integer-width cross product. Wider precisions and complex storage do not
repeat every layout or empty-batch combination. Public discrete-result tests
retain all result widths across their operations, but not every source/result
width combination. Fixed integer coefficients in packed trace source AD do not
constitute active-integer or float0 coverage.

Dedicated low-precision, projection, rounding-order, overflow, zero/one,
nonfinite, aliasing and concurrency regressions remain separate. Selected-argument
AD checks are not replaced by joint checks; JVP, VJP, direct transpose, higher
orders and symbolic-zero/float0 contracts are distinct. References, cotangents
and tolerances also remain specific to their owning tests.

Representative coverage deliberately leaves some precision-by-layout, batch,
coefficient and higher-order interactions untested. Lower-level type tables do
not prove omitted public-wrapper or adapter paths correct. When fixing a defect,
add its specific regression rather than assuming a neighboring representative
covers the same behavior.

Two-device tests focus on sharding rather than repeating the single-device dtype
and layout matrices. Both automatic and explicit mesh modes retain forward,
JVP and VJP checks, local-shard values, inferred sharding, collective requirements,
and packed-storage rejection. Dot covers both conjugation modes and four numeric
directions with representative ranks/layouts. Public transforms retain real flat
vmap and complex nested-vmap cases, higher derivatives and empty batches.
Mapped coefficients retain every existing mapped/shared axis pattern with a real
or complex representative, including both local and all-reduce gradients.

Stride tests separate metadata (`tests/stride/metadata/`), native arithmetic and
execution (`native/`), FFI boundaries (`ffi/`), JAX transformations (`jax/`), public view algebra (`api/`), packed tensor
adapters (`tensor_ops/`) and build infrastructure (`build_checks/`). The latter
paths are also relative to `tests/stride/`. JAX AD and partitioning checks have
their own subdirectories; abstract evaluation, lowering, batching and donation
checks are at the JAX test root. Root TensorMap integration tests remain separate
from the packed-adapter fixtures. `build_checks` avoids pytest's default exclusion
of directories named `build`. All these directories are part of the normal
`pytest tests/stride` collection.

Shared test helpers live under `tests/stride/support/`; test modules do not
import one another. Dtype-only tables in `support/data.py` do not initialize JAX,
while differentiable sample arrays and operation-specific references have separate
modules. Isolated probes live under `support/workers/`; their launching tests set
the subprocess environment before the child imports JAX.

Shared coefficient AD checks compile JVP, joint VJP, and coefficient-only
transpose together, with primal buffers, tangents derived from those buffers,
and cotangents supplied dynamically. Cached operation factories reuse function
identities for identical static layouts; distinct local factories remain separate.
The shared checker's oracle remains eager and independent of native execution.
Accumulation/Reduction references cast mapped contributions before scatter-add;
Update references preserve their separate base/overwrite semantics. Rounding-order
and nonfinite-value regressions retain their separate explicit expectations.
A trace-reuse regression checks that changing input data and zero/one/general
coefficients does not retrace or freeze the inputs into the compiled check.

### Focused runs

Use explicit paths or test names for local iteration, for example:

```bash
uv run pytest tests/stride/jax/ad/test_update.py -q
uv run pytest tests/stride -q
```

A selected subset is not equivalent to the full suite. Run complete affected
owners as needed, and run `uv run pytest tests -q` before merging. A faster subset
shortens feedback by doing less work; it does not demonstrate a speedup at equal
coverage. All selected C++ executables still execute, even with a warm build cache.

To run native compilation checks separately from the rest of Stride, run both
groups below. Neither group alone covers the entire Stride suite, and together
they still omit external TensorMap and other project tests:

```bash
uv run pytest tests/stride/native/test_cpp.py tests/stride/build_checks/test_native_build.py -q
uv run pytest tests/stride -q --ignore=tests/stride/native/test_cpp.py --ignore=tests/stride/build_checks/test_native_build.py
```

## Documentation Site

Build the documentation site locally with:

```bash
uv run --with mkdocs==1.6.1 mkdocs build --strict
```

The public documentation site is deployed to GitHub Pages by
`.github/workflows/docs.yml` on pushes to `main` and by manual workflow
dispatch.

The documentation site is public-facing. Internal planning notes under `local/`
are not part of the public navigation.

## Production Stride Routing

TensorMap readers, dense projection writes, index transforms and tensor traces
use `tensor0._stride` and the native CPU handlers. Only native is compiled
and linked. The previous backend and operation adapters have been removed.

Native uses the migrated AVX2/F16C kernels for contiguous F16-to-F32
conversion and F16 input scaling (F16 coefficients/results or F32
coefficients/results), with runtime CPU detection and the scalar expression
for tails and unsupported CPUs. The internal map dispatcher also supports
F32 scaling followed by F16 storage conversion using the migrated eight-element
kernel, rounding after the F32 multiply, not before it. Update accepts F32
source with F16 storage through the same FFI target and execution path,
including shared/per-batch coefficients and base-buffer reuse. Rank-two F16 transposes with contiguous rows
also use 8-by-8 SIMD tiles for identity mapping or F16-coefficient scaling,
including rectangular subblocks with retained row pitches; edge tiles remain
scalar. Contiguous C64 input scaling by a C64 coefficient uses the migrated
AVX2/FMA kernel with scalar tails; real coefficients retain componentwise
real multiplication. C64 identity/scaling transposes retain the old 16-by-16
blocking with 4-by-4 SIMD tiles and scalar edges, including pitched subblocks.
F32 input scaled by a C64 coefficient also uses the migrated eight-element
AVX2/FMA-targeted kernel, multiplying the two coefficient components separately.
Pure F32-to-C64 conversion is not replaced by multiplication by complex one.
F32 scaling followed by conversion to C64 uses the migrated eight-element
real-product embedding kernel: its imaginary output is zero, including for
NaN/Inf real products. This remains distinct from scaling by a C64 coefficient.
Contiguous C64-to-F32 mappings retain two distinct SIMD formulas: complex
multiplication followed by real projection, and real projection followed by
F32 scaling. C64 scaling by an F32 coefficient followed by real projection
also selects the real-scaling kernel, without moving the expression's final
Cast. Final storage conversion after complex multiplication selects the
complex-product kernel. Conjugation and wider intermediate
types retain their existing expression paths.
F32 four-dimensional two-pair layouts use the migrated AVX2 8-by-8 transpose
kernel for identity or F32 scaling. Matching resolves axis roles from strides
so locality reordering does not hide eligible layouts. The old packing and
tile-divisibility conditions remain; generated blocks that no longer satisfy
them use the general path. Identity tiles preserve their input bits.
Other SIMD specializations remain to be migrated.

Native Copy, Update, Dot, Reduction and address accumulation can schedule independent
storage batches on XLA's CPU FFI thread pool. Copy and Update can also partition
a single batch into disjoint layout subdomains after initializing its output
once. Multi-batch calls do not nest subdomain tasks. Small workloads remain
serial. Dot can split one batch into partial sums. Reduction and address
accumulation can do the same when all nonempty records contribute to one
storage address. Partial-sum execution currently uses F32/F64/C64/C128 result
storage; integer, boolean and F16/BF16 single-output results retain the serial
path. Partials are merged
in layout order after all workers succeed, not in completion order. Parallel
grouping can change floating-point rounding; numerical tolerance, rather than
bitwise agreement with serial execution, is the acceptance criterion.
Multiple-output reductions can instead partition independent output coordinates
for all supported dtypes, without splitting any output's sum. Records with the
same output map stay in original order within a task. Distinct maps require
disjoint address bounds to use this optimization; otherwise execution remains
serial. This conservative eligibility check does not reject valid interleaved
layouts or change the address-accumulation contract.
Arithmetic and buffer alias contracts are unchanged. These scheduled operations
complete through an FFI Future, including error completion.

`tensor0._stride` exports `set_num_threads`, `get_num_threads`,
`enable_threads` and `disable_threads`. The process-wide native limit bounds
both layout partitioning and task scheduling; XLA pool capacity and workload
size may further reduce parallelism. Disabling threads selects one worker,
and enabling threads removes the additional limit. Settings are read at native
execution time, not JAX tracing time, so existing compiled executables observe
changes. Wait for outstanding work before changing the setting.

The extension's shared JAX build-version query reads native's version exports.
The extension provides native registration, prepared lifecycle statistics,
worker limits, and the shared availability/build-version queries.

## Focused Stride Benchmark

Run the focused continuous/transposed-layout regression benchmark with
`uv run python -m benchmarks.standard.diagnostics.stride_update --size 512 --repeats 30`.
This diagnostic uses native and reports synchronized execution time and
compiled output, temporary, and alias bytes for conversion, updates, and a
wide-coefficient JVP. It no longer reports legacy native-call counters.
