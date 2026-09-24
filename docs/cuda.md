# Experimental CUDA stride backend

Tensor0 can optionally build CUDA implementations of stride Copy, Update,
Accumulation, Dot and Reduction through XLA FFI. CPU-only builds remain the default and do not require a
CUDA toolkit or CUDA runtime. See [development.md](development.md) for source-build
switches. This is a correctness-oriented backend, not a performance recommendation.

## Supported operations

Copy, Update, Accumulation, Dot and Reduction support **same-dtype float32, float64, complex64 and complex128
storage**. Enable JAX x64 when using float64 or complex128. Integer, boolean,
half/bfloat16 storage and source/result dtype conversion are not supported by
these CUDA paths. Unsupported operations and dtype combinations raise errors
rather than falling back to CPU.

Supported layouts include affine offsets, positive and negative strides, legal
source broadcasting, scalar records, singleton dimensions, multiple records,
empty records, empty storage and batches. Copy/Update destinations must be injective within
each record, and layout producers must supply disjoint destinations across records,
as on CPU. Native validation does not check cross-record disjointness.

### Copy

The output is independent of the source; unwritten regions are zero-initialized.
Copied values preserve their storage bits, including signed zeros and NaN payloads.

### Update

For selected addresses, Update evaluates `alpha * source + beta * base`; unselected
storage preserves base exactly. Alpha and beta independently accept these types:

| Source/base/result storage | Allowed coefficient types |
| --- | --- |
| float32 | int32, float32 |
| float64 | int32, float64 |
| complex64 | int32, float32, complex64 |
| complex128 | int32, float64, complex128 |

Each coefficient can be scalar, a shared length-one operand, or match the storage
batch shape. Weak Python literals use the existing API's normalization against
storage dtype. Strong coefficients retain their type; other combinations,
including int64 and wider-precision coefficients with narrow storage, are rejected.
The types actually presented to the primitive follow JAX's x64 configuration.

Int32 coefficients are converted on the device to the storage's real component
precision, matching CPU Update. Large integers can round during this conversion;
this is not support for integer storage arithmetic. Real coefficients remain
real when scaling complex storage, rather than introducing zero-imaginary
cross-products that would change NaN/Inf behavior.

A zero coefficient skips its term, and a unit coefficient skips multiplication.
Both zero coefficients write positive zero only to selected elements. An empty
selection preserves base regardless of coefficient values. Arithmetic uses
separate real products/additions and the established explicit full-complex FMA
formula; it does not enable fast-math. Ordinary finite results follow dtype
accuracy expectations, not a blanket bitwise CPU/GPU equivalence promise.

The internal FFI result may alias base; JAX protects live inputs, with donation
an explicit outer-JIT choice. Native allows exact source/base/result reuse only
for matching per-element addresses, rejects unsafe partial overlap, and requires
coefficients to be disjoint from output. Zero coefficients do not exempt invalid
buffer relationships from validation.

### Accumulation

Accumulation clears independent output to arithmetic positive zero, then adds
scaled source contributions in original record order. Both within-record and
cross-record destination overlap are legal, including nonzero-stride collisions.
Holes stay zero. Optional coefficients address original record indices, including
empty records, and use the same type matrix and scalar/shared/batched forms as
Update. Zero skips source contribution arithmetic; one skips multiplication.
Real coefficients remain real. Each contribution is multiplied before addition,
not factored out of a sum. Arithmetic accumulation is not bit-preserving Copy.

Scheduling follows the CPU's safety principle without executing on CPU:

- Proven injective destination maps have parallel output owners.
- Zero destination-stride axes are reduced by one owner per remaining map
  coordinate if that map is proven injective. Each owner starts from existing
  output and traverses its fiber in fixed logical order.
- Other affine maps, including injective maps not established by the sufficient
  stride-span proof, use one GPU thread per batch and sequential record traversal.

Records launch in order on the XLA stream. No atomics, sorting, per-element index
arrays or extra device allocation are used. The general branch and long fibers
can be slow, especially for a single large batch; no performance claim is made.
Fixed traversal does not promise CPU bitwise equality or reproducibility across
architectures. Source, descriptors and coefficients must be disjoint from output;
zero coefficients and empty selections do not exempt invalid metadata or aliases.

### Dot

Dot supports same-dtype left/right/result F32/F64/C64/C128, with one scalar per
batch on GPU. `dotu` multiplies directly; `dotc` conjugates only the left input.
All legal affine read layouts, including repeated addresses, are supported.
Left and right may alias each other; neither may overlap result or scratch.
Empty selections return arithmetic positive zero.

Each record uses fixed 1024-contribution chunks and 256-thread tree reductions,
followed by a second reduction stage adding its result in record order. An
internal second FFI result provides XLA-owned temporary partials, sized to the
largest record and reused in stream order. No atomics, native per-call device
allocation, synchronization or per-element address table is used. Real products
and additions are separate RN operations; complex products retain the explicit
FMA expression. Reduction order differs from CPU; finite results are compared
with dtype-appropriate tolerances, not bitwise equivalence.

### Reduction

Reduction uses same-type F32/F64/C64/C128 storage and the Update coefficient
matrix. Each record separates map axes from reduction axes. Output shapes must
match those roles, and the remaining output map must pass the same injectivity
proof as CPU; repeated source reads are legal. Reduction-axis output strides are
ignored. Output bounds are validated even for empty fibers. Records may overlap
one another and execute in original order; optional coefficients retain original
record indices, including empty records.

Independent output is zero-initialized, including holes. One GPU thread owns
each output map coordinate and traverses its fiber in fixed logical order, starting
from the existing output. Every contribution is multiplied before addition, with
the same zero/one and real/full-complex branches as Accumulation. There are no
atomics or temporary partials. Long fibers can be slow; this is not a performance
claim. Unsigned 64-bit descriptor extents retain their full protocol range even
when JAX x64 is disabled. Invalid role/shape/map/count/address/alias metadata is
rejected before output writes.

### Remaining limitations

All five native stride primitives have CUDA paths in the limited dtype scope above.
Existing JAX linear algebra is unchanged.
At the **stride primitive** layer, same-dtype Dot JVP/reverse, Copy reverse (including repeated source addresses),
fixed-coefficient Update input reverse, Reduction source reverse and matching-type
Reduction JVP, and Update coefficient JVP use the existing
AD rules. Update/Accumulation/Reduction coefficient VJPs are supported only when the
resulting Dot operands and result all have the same supported dtype. In
particular **real coefficients with complex storage still fail in coefficient
VJP**, because that requests complex Dot inputs and a real result. Wider/mixed
precision paths also remain unsupported; no implicit cast expands this scope.
High-level structural plans can also produce unsupported coefficient dtypes:
for example, an SU2 grouped trace with JAX x64 enabled can generate float64
coefficients even for float32/complex64 storage. That path is rejected rather
than silently narrowing the plan's coefficients. The bounded high-level acceptance matrix below does not establish general
TensorMap GPU support. Primitive restrictions do not automatically apply to
high-level operations that use only JAX arithmetic.
Existing complex AD conventions are retained without extra conjugation.
**Full GPU reverse AD/grad/VJP support is not provided.** Multi-GPU execution,
CUDA Graph support and performance improvements are not claimed.

## Tested high-level TensorMap subset

The public-API tests in `tests/tensor/test_cuda_integration.py` run on an A100
with JAX/JAXLIB 0.10.1. They compare explicit CPU and CUDA placements, check
returned arrays' device residency and dtypes, and inspect CUDA native targets for
nontrivial transforms and trace. Small dense/sign/reconstruction oracles supplement
the CPU comparisons. This is a representative acceptance matrix, **not support
for every sector, shape, contraction topology or AD composition**.

All positive cases below use **`jax.default_matmul_precision("highest")`**.
Tensor0 does not set this option for users. JAX's default GPU matrix-product
precision can produce larger errors, including in complex64 composition and
batched SU2 recoupling; pure-JAX matrix multiplication reproduces this behavior
without native stride execution. Choose the precision policy appropriate to your
application rather than assuming GPU default matmul has storage-precision accuracy:

```python
with jax.enable_x64(False), jax.default_matmul_precision("highest"):
    result = jax.jit(operation)(tensor)  # tensor storage already on CUDA
```

Use `jax.enable_x64(True)` for the wide-storage cases; disabling x64 is not a
valid float64/complex128 test.

| Test group | Sector/space sample | Storage and x64 configuration |
| --- | --- | --- |
| Main forward and AD matrix | Trivial dimension 2; U1 charges 0 and 1 (multiplicity 1); SU2 spin-half (multiplicity 1). Two outgoing and two incoming legs | F32/C64 with x64=False; F64/C128 with x64=True |
| Fermionic signs | FermionParity: two outgoing odd legs | F32 with x64=False; C128 with x64=True |
| Scale coefficient VJP | U1 charges 0 and 1, one outgoing/incoming leg | Same four storage/x64 pairs as the main matrix |
| Linear algebra forward reconstruction | U1 charges 0 and 1, or SU2 spins 0 and 1/2, multiplicities 2 and 1; one outgoing/incoming leg | F64/C128 with x64=True only |

For the main matrix, the crossing permutation is `((1, 2), (0, 3))`; the SU2
case includes multi-tree recoupling. Trace joins legs 1 and 3, leaving legs 0 and
2. The contraction joins both input legs of one tensor to both output legs of
another and is checked against composition; arbitrary networks and partial
contraction topologies are not covered by this matrix.

| Public operation | Verified execution / AD within those samples |
| --- | --- |
| `permute`, `transpose`, `adjoint` | JIT forward for all three; `permute` additionally has JVP and external `vmap` over packed storage |
| `scale`, `add`, `inner` (also the implementation of `dot`), `norm` | JIT forward; separate `scale` coefficient VJP checks described below |
| Composition (`@`) and `tensorcontract` | JIT forward; input `grad` of the real composition loss `real(inner(t @ t, t @ t))` |
| `tensortrace` | JIT forward and input VJP, including complex cotangents |
| Fermionic `permute`, `twist`, `flip` | JIT odd swap/twist signs, double-forward flip sign and forward/inverse flip roundtrip; input grad of the real odd-swap loss only |
| `qr_compact`, `svd_compact`, `eigh_full`, `left_solve` | Small JIT forward reconstruction/residual checks only; no decomposition/solve AD claim |

AD comparisons use JAX's existing complex cotangent convention. A real-valued
loss on complex storage is **not** a holomorphic differentiation claim. Vmap and
AD in one row must not be inferred for the other operations.

### High-level versus primitive coefficient gradients

`TensorMap.scale` and `add` use JAX elementwise arithmetic; `inner`/`dot` use
weighted blockwise `jnp.vdot`. They do **not** call stride Update or Dot just
because the operation names resemble them. Public `scale` coefficient VJP is
verified for matching-type coefficients **and for real coefficients with complex
storage** in the four storage/x64 configurations above. This does not remove the
native stride coefficient-VJP limitation: Update/Accumulation/Reduction with
complex storage and real coefficient gradients still request an unsupported
mixed-result native Dot.

The SU2 trace sample with **F32/C64 storage and x64=True** explicitly rejects
strong F64 structural coefficients on CUDA. It is not silently narrowed or
skipped. This does not imply that every SU2 operation with that configuration
fails; the restriction is on the coefficient types actually produced by a plan.

Other sector families/products, broader duality/braiding/network cases, additional
batching/AD combinations, decomposition AD and mixed storage/result dtypes remain
outside this high-level acceptance matrix. Multi-GPU, CUDA Graph, sanitizer,
other toolchains/architectures and performance remain unverified, not silently
classified as passing.

## Execution and ownership

Compilation platform determines the CPU or CUDA target, including through custom
partitioning. Installing GPU JAX does not enable native CUDA execution in a
CPU-only Tensor0 extension; rebuild with CUDA explicitly enabled. JAX and JAXLIB
must match the pinned native FFI header versions (0.10.1).

CUDA lowering emits compact i64 metadata as an XLA-owned constant operand, even
when JAX x64 is disabled. The same encoding is a host attribute for validation. Reduction bitcasts its
little-endian unsigned-word byte protocol to i64 without changing its bits.
There is no per-element index table or native device cache. The internal FFI ABI
requires operand contents to match the immutable attribute; use stride primitives
rather than hand-crafted FFI calls.

The native CUDA implementation separates three responsibilities:

- `native/cuda/*.cu` adapts the XLA FFI call, validates metadata and invocation
  buffers, and contains exceptions at the call boundary.
- `native/cuda/execute/` owns CUDA scheduling, output initialization, coefficient
  dispatch and ordered kernel launches, including Dot scratch reuse.
- `native/cuda/kernels/` implements device address traversal and numerical work.

Both backends use `native/layout/descriptor.{h,cc}` for descriptor decoding and
semantic validation, without CPU axis sorting or merging. CPU preparation applies
its own record optimization before storing immutable `PreparedState`; CUDA keeps
per-call host preparation and derives its schedules from the original metadata.
Reduction semantic records canonicalize ignored reduction-axis output strides,
so they are not a replacement for the original protocol's axis flags and cursors.
Buffer-byte and alias checks in `native/ffi/buffers.h` remain invocation-boundary
checks, separate from static descriptor semantics.

CUDA executors use the XLA stream, independently of CPU executors and their
thread pool. No persistent CUDA prepared state or device cache is introduced.
Neither semantic records nor CPU-optimized records authorize reinterpreting an
original device descriptor, especially for Reduction axis roles.

After validating host metadata and buffer relationships, Copy, Accumulation and Reduction zero output; Update
copies base to output when they differ. Copy/Update/Accumulation/Reduction launch one generic kernel per nonempty record; Dot
launches two stages per nonempty record, in order on the XLA-provided stream. Update, Accumulation and Reduction
read coefficients on device. No handler reads device data on host or allocates
device memory per
call, and none synchronizes. Immediate launch errors are reported by the handler;
asynchronous errors may surface at a later JAX synchronization. Ordered launches
preserve overlapping-record Accumulation/Reduction semantics, but do not relax Copy/Update destination disjointness requirements.

## Testing

The bounded public TensorMap matrix is in `tests/tensor/test_cuda_integration.py`.
It has explicit CPU/CUDA parametrization, so CPU-only runs execute substantive
CPU cases rather than skipping the entire module. The matrix fixes JAX matmul
precision to `highest` without changing Tensor0 runtime defaults.

Primitive GPU integration tests are in `tests/stride/jax/test_cuda_copy.py`,
`tests/stride/jax/test_cuda_update.py`,
`tests/stride/jax/test_cuda_accumulation.py`, `tests/stride/jax/test_cuda_dot.py`,
`tests/stride/jax/test_cuda_reduction.py`, and `tests/stride/cuda/`. Tests requiring
native GPU support skip without a CUDA-enabled extension/device. The standalone
Update/Accumulation/Dot/Reduction boundary tests instead need an existing toolkit/device, accept
`NVCC`,
`CUDA_HOME` and `TENSOR0_CUDA_ARCH` (default `sm_80`), and exercise actual buffer
alias rejection and stream dependency ordering. A skip is not a validation pass.

Run against an isolated CUDA-enabled candidate, and run CPU stride regressions
with CPU placement separately. Registration/build tests cover CPU-only and
requested-but-unavailable configurations. No speedup, sanitizer validation or
universal toolkit compatibility is implied by the optional build.
