# Development

Tensor0 is developed from a source checkout with Python, Rust, uv, maturin, and
pytest.

Building the Linux Native backend requires a C++20-capable compiler. The build
uses `/usr/bin/c++` by default; set `CXX` to select another compiler.
C++ optimization follows Cargo's profile optimization level: ordinary
`maturin develop` uses `-O0`, while `maturin develop --release` uses `-O3`
with the default profiles. Size levels `s` and `z` both use `-Os`.
The Native backend reuses its object file when its declared source contents,
vendored FFI headers, compiler command/version, and build script are unchanged.
After changing system headers or replacing a compiler in place, clean the
extension build with `cargo clean -p tensor0-py` before rebuilding.
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
sources in `tests/stride/cpp`. On Linux, `test_cpp.py` compiles them with a C++20
compiler and UndefinedBehaviorSanitizer; set `CXX` to select the compiler.
The tests cover scalar promotion, expression stages, address planning, SIMD
tails, skipped reads, asynchronous task ownership, and validation before writes.
The FFI boundary unit compiles the current binding functions without expanding
every exported dtype handler; registration is checked separately in Python.

```bash
uv run pytest tests/stride/test_cpp.py tests/stride/test_build.py -q
uv run pytest tests/stride -q --ignore=tests/stride/test_cpp.py --ignore=tests/stride/test_build.py
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
