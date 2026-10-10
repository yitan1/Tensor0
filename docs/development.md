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
instantiated in seven operation-owned units: `native/ffi/copy.cc`,
`update_a.cc`, `update_b.cc`, `reduction_a.cc`, `reduction_b.cc`, `dot_a.cc`, and `dot_b.cc`.
Update, Reduction/Accumulation, and Dot each use two fixed output-dtype groups
sharing `native/ffi/update_impl.h`, `native/ffi/reduction_impl.h`, and
`native/ffi/dot_impl.h`, respectively; the implementation is not duplicated.
Reduction and Accumulation for the same output dtype remain in the same unit.
Their typed inner kernels remain visible through headers for inlining.
Shared lifecycle state and ordinary support implementations have one definition.
Native translation units compile in parallel, bounded by Cargo's `NUM_JOBS`
(and the number of source files). The build script uses its implicit Cargo job
slot for one task and acquires shared jobserver tokens for additional tasks, so
C++ and Rust compilation share Cargo's concurrency budget. Tokens are returned
when no longer needed; pending token requests are cancelled on completion or
failure without waiting for other Cargo jobs to finish.
Without an explicit job setting, builds follow Cargo's default concurrency budget,
which is based on available logical CPUs. Tensor0 does not impose a fixed job count
or independently probe the machine to create another concurrency budget. Use `-j N`
or Cargo's `CARGO_BUILD_JOBS` environment variable to override the budget:

```bash
# Follow Cargo's default concurrency budget.
uv run maturin build --release

# Limit concurrency explicitly; use -j 1 for serial compilation.
uv run maturin build --release -j 4

# Alternatively, configure Cargo through the environment.
CARGO_BUILD_JOBS=4 uv run maturin build --release
```

The values above are examples, not recommended defaults for every machine. Heavy
numerical units can each use several GiB of memory; choose the job count for the
memory available to the build, not just CPU count. CPU-based defaults do not
guarantee sufficient memory, especially on high-core-count machines or in
memory-limited containers. The jobserver shares available slots dynamically; it
does not adjust the overall budget in response to free memory. Configure the budget
through Cargo rather than setting the build-script input `NUM_JOBS` directly.
Build concurrency is separate from runtime execution: `tensor0._stride.set_num_threads`
controls native operation workers, not compiler jobs.

Without an inherited jobserver, or without `NUM_JOBS`, the script uses one worker.
After observing a failure it stops assigning new units; assigned units finish
before the build exits. Successful
objects remain cached for a retry. All workers finish before linking, and linker
input order remains fixed. Changing only the job count does not invalidate
object fingerprints. Use `maturin build --release -j 2 -vv` to display build-script
stderr, including per-unit cache hits, compilation starts/results and elapsed wall
times. The final native-object time includes cache checks and compilation, but not
Rust compilation or linking. Per-unit times overlap in parallel builds and must
not be summed as total wall time. When Cargo reuses the entire build-script result,
the script does not run and no new per-unit messages are emitted.
Builds reuse each object independently using compiler-generated
dependency files, dependency contents,
compiler identity/command and build-script inputs. Implementation-only edits to an
ordinary `.cc` do not rebuild the numerical objects. Operation-specific `.cc` or
execution-header edits rebuild only that operation; shared kernel headers can
rebuild multiple operations. Editing `update_impl.h`, `reduction_impl.h`, or
`dot_impl.h` rebuilds both units of that operation; editing an entry `.cc`
rebuilds only that unit.
Missing objects/dependency files and failed compilations invalidate the affected
cache entry. Build-script tests compile a small Cargo harness offline; run
`cargo fetch` first if the build dependencies are not already cached. Compiler paths and discovered dependencies are watched by Cargo.
Use a release build for performance measurements.

Both CPU instantiate and CUDA lowering use the same device-independent native
layout preparation: full semantic validation, stable locality sorting, compatible
axis fusion, then re-sorting and removal of nonempty singleton axes. This promotes
the original CPU sorting/fusion policy to the shared layer rather than limiting CPU
to CUDA's former order-preserving adjacent fusion. Copy/Update use injective map
validation; Dot permits repeated read addresses, whereas Accumulation requires
provably injective output owners after zero-stride fiber axes are separated.
Reduction moves explicit axis roles with every permutation, fuses only like
roles, then projects its address
records for CPU execution. Empty records retain their original axis representation.
Preparation repeats after a rank reduction until the canonical layout is stable:
removing singleton/fused axes can change locality ranks. CPU and CUDA both consume
this canonical layout; CPU execution may still choose different row/block/partial-
sum schedules, so shared metadata does not imply bitwise-identical arithmetic.

CPU Accumulation caches immutable generated traversal programs in its prepared
state, keyed by source and result item sizes, independently of the worker count.
Both serial and parallel execution reuse these programs. Parallel task planning
remains per invocation; output-task planning checks group intervals before copying
programs, so rejected layouts do not allocate unused task programs. Partial-reduction
plans copy the programs before rewriting or splitting them. Asynchronous batch tasks retain shared ownership of the cached
programs. Input/output buffers, coefficients, output initialization and partial
results are not cached. Reuse therefore benefits repeated execution of the same
prepared instance even when tensor values change, but does not eliminate first-use
preparation or parallel scheduling costs.

Weighted affine producers also preserve an internal `outputs_disjoint` guarantee
when selected destination subblocks from their disjoint layout are unique.
Repeated destination selections and raw calls remain generic accumulations.
The CPU executor skips global output-geometry planning for these injective maps:
workers process ranges of shared immutable traversal programs, while independent
batches retain batch scheduling. A single large record uses the existing map
subdomain splitter; multiple records are not individually split or load-balanced
by work in this initial path. Output initialization, scaling, mixed arithmetic,
and writeback still use the accumulation kernel. Source and coefficient JVP keep the proof;
transpose drops it because repeated source reads can become gradient write
conflicts. CUDA ignores this CPU scheduling metadata and retains generic sums.
The worker granularity remains 32K; this is not a GPU performance optimization.

## Optional CUDA stride build

Linux builds are CPU-only by default and do not discover or link CUDA. To enable
CUDA affine Copy, Update, Accumulation, Dot and Reduction for F32, F64, C64 and C128, use an existing CUDA toolkit
with a C++20-capable `nvcc`:

```bash
TENSOR0_CUDA=1 CUDA_HOME=/usr/local/cuda TENSOR0_CUDA_ARCH=sm_80 \
    uv run maturin develop --release
```

`TENSOR0_CUDA` accepts `0` (also the unset default) or `1`. An explicitly requested
CUDA build fails rather than silently producing a CPU-only extension when the
compiler or required headers are unavailable. `NVCC` selects the compiler;
otherwise it is `$CUDA_HOME/bin/nvcc`, or `nvcc` from `PATH` when `CUDA_HOME` is
unset. `CUDA_HOME` selects the toolkit root for runtime-library lookup (`lib64`
or `lib`); by default it is inferred from the resolved NVCC path. The toolkit must
provide `libcudart_static.a`. Opted-in builds link this static CUDA runtime plus
`dl`, `rt` and `pthread`, avoiding a runtime search path for `libcudart.so`.
Running CUDA operations still requires a compatible NVIDIA driver. CPU-only
builds do not link the CUDA runtime.

`CXX` selects both the CPU compiler and NVCC's host compiler (passed explicitly as
`-ccbin`); it defaults to `/usr/bin/c++`. `NVCC_CCBIN` is cleared for the NVCC
invocation. The selected host compiler path, resolved executable contents and
version are included in the CUDA fingerprint, with the same Cargo file watches
used for CPU compilation. Replacing the host compiler at the same path therefore
invalidates both CPU and CUDA objects.

`TENSOR0_CUDA_ARCH` is passed as NVCC's single `-arch` argument and defaults to
`sm_80`; select an architecture supported by both your toolkit and target GPU.
Changing the compiler, architecture, CUDA environment inputs, source or discovered
headers invalidates the CUDA object cache. The selected `libcudart_static.a` is
watched separately: replacing it triggers relinking without recompiling unchanged
native objects. Other toolkit internals, such as an independently replaced
`ptxas`, are not individually fingerprinted; perform a clean rebuild after such
updates. CUDA optimization follows the selected C++ optimization level, except
size levels `s` and `z` map to NVCC `-O2`.
CUDA compiles Copy, Update, Accumulation, Dot and Reduction serially after all CPU workers have finished and
returned their jobserver tokens, using the build script's implicit Cargo slot.
It does not start a separate worker pool.

The private `_stride_cuda_available()` export reports build-time support, not
whether a usable GPU is present. `_stride_cuda_registration()` returns twenty
execution-handler capsules plus Copy, Update, Accumulation, Dot and Reduction instantiate handlers
and their independent state type ids/type infos (35 entries), or an empty dictionary
in CPU-only builds. These five operations own immutable host metadata per executable;
Reduction owns explicit-role map/fiber schedules, descriptor cursors and sparse coefficient mapping. `_stride_cuda_copy_prepared_stats()`,
`_stride_cuda_update_prepared_stats()`, `_stride_cuda_accumulation_prepared_stats()`,
`_stride_cuda_dot_prepared_stats()` and `_stride_cuda_reduction_prepared_stats()` expose separate creation/destruction counts for lifecycle tests, or `None` without
CUDA. No state owns device resources.
This backend supports
same-type Copy, Update, Accumulation, Dot and Reduction, not dtype
conversion or complete reverse-mode differentiation. See [CUDA support](cuda.md)
for the supported coefficient types and limited AD paths.

## Local Setup

```bash
uv sync --group dev
uv run maturin develop
```

## Verification

Include the existing plotting dependencies with
`uv sync --group dev --group bench` for benchmark plotting tests and type
checking. Without the `bench` group, plotting tests skip rather than execute.

```bash
cargo test
uv run pytest tests -q
uv run python examples/basic_usage.py
uv run python examples/contractions.py
uv run pyright src examples benchmarks
```

Linux CPU CI is configured to run the build, tests and examples on CPython
3.11–3.14 with a freshly built extension. Each matrix job must pass before a
candidate is accepted; configuration alone is not verification. The source,
examples and benchmarks type check runs on Python 3.11 and is blocking. The default
`uv run pyright` also includes tests; it currently reports test typing debt,
including dynamic test doubles and intentionally invalid inputs. CI retains
this full check as an explicitly advisory report with its diagnostics uploaded,
not as evidence that the complete typing scope passes. Do not widen production
API types or disable diagnostic rules to accommodate negative tests.

Native regression tests live in `tests/stride`, including the standalone C++
sources in `tests/stride/native/cpp`. On Linux, `native/cpp/test_contracts.py` compiles them with a C++20
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
Use `uv run pytest tests/stride/native/cpp/test_contracts.py --cache-clear -q` for a clean build,
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
paths are also relative to `tests/stride/`. JAX AD, partitioning and CUDA GPU
integration checks have their own `jax/ad/`, `jax/partitioning/` and `jax/cuda/`
subdirectories; abstract evaluation, lowering, batching and donation checks
remain at the JAX test root. `ffi/test_cuda_lowering.py` checks the host-to-MLIR
boundary without running a GPU kernel; `native/cuda/` owns standalone CUDA
contract sources and binaries, separate from the JAX GPU integration tests.
Root TensorMap integration tests remain separate from the packed-adapter fixtures.
`jax/ad/test_support.py` owns the coefficient AD harness trace-reuse regression.
`build_checks/` covers production `build.rs` object caching, optimization,
jobserver concurrency and CUDA linkage; the standalone C++ test executable cache;
and vendored FFI header/license and runtime-version integrity. Most build-script
cases compile a small real Cargo harness but use synthetic native sources and a
mock compiler/NVCC to inspect invalidation and scheduling. Selected cases actually
compile/link small C++ and Cargo fixtures (including CUDA-linkage simulation with
C++ stubs), not the production CUDA kernels. `test_cpp_cache.py` uses a mock
compiler; `test_vendor.py` checks files and declarations without compilation.
`build_checks` avoids pytest's default exclusion of directories named `build`.
All these directories are part of the normal `pytest tests/stride` collection.

Shared test helpers live under `tests/stride/support/`; test modules do not
import one another. Directory ownership follows the contract being asserted, not
which helper, compiler, primitive or execution API the test happens to call:
standalone native C++/CUDA executables belong under `native/`, host-side FFI
contracts under `ffi/`, and device execution under `jax/cuda/`. Device tests
request the local `cuda_device` fixture explicitly so CPU-only checks in that
directory still run without a GPU.

Dtype-only tables in `support/data.py` do not initialize JAX,
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

Select the layer and environment that matches the contract:

| Contract | Prerequisite | Focused command |
| --- | --- | --- |
| Host metadata, FFI lowering and CPU native execution | Built CPU extension and project test environment; no GPU required | `uv run pytest tests/stride/metadata tests/stride/ffi/test_cuda_lowering.py tests/stride/native -q` |
| Standalone C++ executable contracts | Linux C++20 compiler and UBSan; selected binaries execute even when cached | `uv run pytest tests/stride/native/cpp/test_contracts.py -q` |
| CUDA native executable contracts | CUDA toolkit/NVCC and compatible GPU | `uv run pytest tests/stride/native/cuda -q` |
| JAX GPU primitives and lifecycle | CUDA-enabled extension, compatible driver/GPU and JAX GPU backend | `uv run pytest tests/stride/jax/cuda -q` |
| Isolated JAX workers | Required devices (two for partitioning) and subprocess-accessible environment | `uv run pytest tests/stride/jax/partitioning -q` |
| Coefficient AD harness trace reuse | Built CPU extension and JAX test environment; no GPU required (skips without native extension) | `uv run pytest tests/stride/jax/ad/test_support.py -q` |
| Build-script cache, jobserver and CUDA-linkage checks | Linux, Cargo with cached offline `jobserver` dependency (`cargo fetch` first if needed); C++20 compiler and `ar` for real compile/link cases; no CUDA toolkit/GPU required | `uv run pytest tests/stride/build_checks/test_native_build.py -q` |
| Standalone C++ test cache and vendor integrity | Linux for cache simulations (Python mock compiler); repository vendored files for integrity checks; no native extension/toolkit/GPU required | `uv run pytest tests/stride/build_checks/test_cpp_cache.py tests/stride/build_checks/test_vendor.py -q` |

GPU tests use explicit device fixtures and may skip when unavailable; successful
CPU-only collection or a skipped run is not evidence of GPU execution. Some host
and C++ tests require a built extension or compiler even without CUDA. Partitioning
workers set their own subprocess environment before importing JAX; do not replace
them with in-process checks. A selected subset is not equivalent to the full suite.
Run complete affected owners as needed, and run `uv run pytest tests -q` before merging. A faster subset
shortens feedback by doing less work; it does not demonstrate a speedup at equal
coverage. All selected C++ executables still execute, even with a warm build cache.

To run native compilation checks separately from the rest of Stride, run both
groups below. Neither group alone covers the entire Stride suite, and together
they still omit external TensorMap and other project tests:

```bash
uv run pytest tests/stride/native/cpp/test_contracts.py tests/stride/build_checks/test_native_build.py -q
uv run pytest tests/stride -q --ignore=tests/stride/native/cpp/test_contracts.py --ignore=tests/stride/build_checks/test_native_build.py
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
use `tensor0._stride` and, by default, the native CPU handlers. The optional
[CUDA backend](cuda.md) is selected by compilation platform for supported
Copy, Update, Accumulation, Dot and Reduction; unsupported dtype combinations fail explicitly. Only native is
compiled and linked. The previous backend and operation adapters have been removed.

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

Native Copy avoids zero-filling a nonempty output only when one record proves dense,
complete destination coverage, including signed-stride permutations and scalar
records. Sparse or multiple records retain the prefill. The generic CPU affine-row
traversal avoids a coordinate vector for rank-zero and rank-one records; higher
ranks still track outer coordinates. Neither change alters the CPU batch scheduler.

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
serial. This conservative CPU scheduling eligibility check does not reject otherwise
valid interleaved layouts. Independently, the common Accumulation preparation
contract accepts only output owner maps proven injective; zero-stride fibers and
ordered overlap across records remain legal. Reverse AD for nonzero-stride
repeated source addresses is unsupported, while forward reads remain legal.
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
