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
scaled source contributions in original record order. Zero-stride fibers may
have multiple contributions per output, and cross-record overlap stays ordered.
Within one record, the nonzero destination-stride map axes must pass the native
sufficient injectivity proof; otherwise preparation rejects the layout even if
its addresses happen to be unique. Nonzero-stride owner collisions are unsupported.
Holes stay zero. Optional coefficients address original record indices, including
empty records, and use the same type matrix and scalar/shared/batched forms as
Update. Zero skips source contribution arithmetic; one skips multiplication.
Real coefficients remain real. Each contribution is multiplied before addition,
not factored out of a sum. Arithmetic accumulation is not bit-preserving Copy.

Scheduling follows the CPU's safety principle without executing on CPU:

- Proven injective destination maps have parallel output owners.
- Zero destination-stride axes are reduced by one owner per remaining map
  coordinate if that map is proven injective. Short or resource-limited fibers
  are traversed by one owner thread in logical order; selected long fibers use
  multiple threads per owner and a tree reduction.
- Layouts whose map-axis injectivity cannot be proven are rejected before
  launch, even with zero coefficients or an empty batch.

Records launch in order on the XLA stream. Parallel fibers use an XLA-owned
internal partial buffer, not a native per-call allocation; each owner combines
its partials with the existing output exactly once. No atomics or per-element
index arrays are used. Parallel summation can change floating-point grouping;
neither strategy promises CPU bitwise equality or reproducibility across
architectures. Source, descriptors and coefficients must be disjoint from output;
zero coefficients and empty selections do not exempt invalid metadata or aliases.

### Dot

Dot supports same-dtype left/right/result F32/F64/C64/C128, with one scalar per
batch on GPU. `dotu` multiplies directly; `dotc` conjugates only the left input.
All legal affine read layouts, including repeated addresses, are supported.
Left and right may alias each other; neither may overlap the result.
Empty selections return arithmetic positive zero.

Dot uses one output owner per batch. Single-contribution records use the
common sequential owner/fiber kernel. Longer fibers use 1024-contribution
chunks with a 256-thread tree per chunk. A single chunk writes directly to the
output; multiple chunks use a second kernel to combine their partials with the
previous output once. Records complete in order on the XLA stream. The internal partial buffer is an XLA-owned second result; it is
not returned to the caller. Both strategies share the packed owner/fiber
address representation and static Dot calculation policy with the common
map/reduce kernel used by the other operations. No atomics, native per-call
device allocation, synchronization or per-element address table is used. Real
products and additions are separate RN operations; complex products retain
the explicit FMA expression. Parallel reduction changes the sum grouping from
sequential traversal and does not guarantee bitwise equality with CPU.

### Reduction

Reduction uses same-type F32/F64/C64/C128 storage and the Update coefficient
matrix. Each record separates map axes from reduction axes. Output shapes must
match those roles, and the remaining output map must pass the same injectivity
proof as CPU; repeated source reads are legal. Reduction-axis output strides are
ignored. Output bounds are validated even for empty fibers. Records may overlap
one another and execute in original order; optional coefficients retain original
record indices, including empty records.

Independent output is zero-initialized, including holes. Each output map
coordinate has one owner. Short or resource-limited fibers run sequentially;
selected long fibers use shared chunk partials and an XLA-owned internal
buffer. For Accumulation and Reduction, parallel execution requires at least
256 contributions per owner and at most 2^20 partial slots per batch
(`owners * ceil(contributions / 1024)`); chunks remain 1024 contributions.
This static threshold is based on limited A100 measurements, not a universal
performance guarantee. A finish task combines each owner's existing output exactly once.
Every contribution is multiplied before addition, with the same zero/one and
real/full-complex branches as Accumulation. No atomics are used, and changed
sum grouping does not imply bitwise equality. Unsigned 64-bit descriptor extents retain their full protocol range even
when JAX x64 is disabled. Invalid role/shape/map/count/address/alias metadata is
rejected before output writes.

### Remaining limitations

All five native stride primitives have CUDA paths in the limited dtype scope above.
Existing JAX linear algebra is unchanged.
At the **stride primitive** layer, same-dtype Dot JVP/reverse, Copy reverse,
fixed-coefficient Update input reverse, Reduction source reverse and matching-type
Reduction JVP, and Update coefficient JVP use the existing AD rules. Input reverse
AD requires the differentiated view's nonzero-stride map axes to pass the
injectivity check; zero-stride broadcast repetition remains supported. Forward
repeated reads remain legal, but nonzero-stride overlapping input reverse AD is
unsupported. Update/Accumulation/Reduction coefficient VJPs are supported only when the
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
general CUDA Graph support and performance improvements are not claimed. Native
Copy and Update capture/replay contracts and an explicitly forced JAX Update
CUDA Graph path have been tested; default JAX execution is not asserted to use
CUDA Graphs.

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
outside this high-level acceptance matrix. Multi-GPU, high-level TensorMap CUDA
Graph integration, sanitizer, other toolchains/architectures and performance remain
unverified, not silently classified as passing.

## Execution and ownership

Compilation platform determines the CPU or CUDA target, including through custom
partitioning. Installing GPU JAX does not enable native CUDA execution in a
CPU-only Tensor0 extension; rebuild with CUDA explicitly enabled. JAX and JAXLIB
must match the pinned native FFI header versions (0.10.1).

CUDA lowering calls native semantic preparation before emitting compact i64
metadata. All five operations use the validated semantic result as the host
attribute and a mechanically packed owner/fiber view of those same prepared
records as the XLA-owned device constant. Reduction retains explicit roles in its host protocol;
its unsigned extents and signed strides retain their i64 bit patterns in the
device view. Constants remain i64 even when JAX x64 is disabled.
There is no per-element index table or native device cache. The internal FFI ABI
requires device operand contents to match the semantic host attribute through
the native packer; use stride primitives rather than hand-crafted FFI calls.

The native CUDA implementation separates three responsibilities:

- `native/cuda/*.cu` adapts the XLA FFI call, validates metadata and invocation
  buffers, and contains exceptions at the call boundary.
- `native/cuda/execute/` owns CUDA scheduling, output initialization, coefficient
  dispatch and ordered kernel launches.
- `native/cuda/kernels/` implements device address traversal and numerical work.

Both backends use `native/layout/descriptor.{h,cc}` for descriptor decoding and
semantic validation. Reduction records compose an address layout with explicit
axis roles; roles are never inferred from zero strides. Native lowering preparation
validates the complete original protocol before transforming or encoding it.
Preparation uses the previous CPU stable locality order, fuses adjacent axes
contiguous in both address views in first-axis-fastest order, then sorts again
and removes nonempty singleton axes. Repeated preparation after rank reduction
makes the canonical descriptor idempotent when removed axes change stride ranks.
Reduction permutes explicit roles with each axis and fuses only equal roles;
roles are never inferred from zero destination strides. Empty records, storage
sizes, offsets, semantic indices, record order and coefficient mapping remain
intact. Address fused extents stay within the signed V1 protocol; Reduction
retains unsigned extents. Ignored reduction-axis output strides are canonicalized in the host semantic
metadata and the derived device execution view. Sorting can change the
floating-point contribution order; correctness is checked against the numerical
contract, not a former GPU bitwise traversal order.
CPU preparation calls the same shared functions and stores their normalized
records in immutable `PreparedState`; Reduction fuses role-aware records before
projecting their address layouts. CPU and CUDA therefore share canonical prepared
records, though CPU row traversal, blocking and partial-sum trees still differ
from CUDA execution and do not guarantee bitwise equality. CPU Copy/Update may
reorder independent map iterations within private execution plans without changing
the canonical prepared records. CUDA Copy, Update,
Accumulation and Dot validate the supplied semantic host attribute at FFI
instantiate and store decoded records, element counts, packed owner/fiber
schedules and the expected device operand length in separate immutable host
states owned by the executable. Update stores each record's identity-mapping
predicate; Accumulation stores the static mapping from records to coefficient
operands and scratch capacity, retaining empty records. Dot also stores its
conjugation mode and static scratch capacity. Reduction stores its scratch
capacity alongside its explicit-role map/fiber schedule. Actual aliases,
coefficient buffer types/shapes and operand counts remain invocation checks.
Reduction keeps explicit-role map/fiber schedules and sparse coefficient mapping in immutable executable-owned host state. Neither
backend independently transforms the canonical prepared layout. CPU Update
alias validation compares logical addresses and ignores unused singleton strides
for nonempty records, as CUDA Update does.
Buffer-byte and alias checks in `native/ffi/buffers.h` remain invocation-boundary
checks, separate from static descriptor semantics.

CUDA executors use the XLA stream, independently of CPU executors and their
thread pool. All five CUDA host states contain no device pointers or stream
resources; their destructors neither free device memory nor synchronize. Launched kernels use
value arguments and the XLA-owned device descriptor, not host state pointers.
Actual buffer shapes, batch products, byte sizes and alias relationships are still
checked on every call before any writes. No native device cache is introduced.
Every layout transform must update the authoritative preparation result
before host and device encoding; changing host records alone is insufficient.

After validating host metadata and buffer relationships, Dot,
Accumulation and Reduction zero output. Copy skips zeroing when one injective
record fully covers output; Update skips copying base to a distinct output in
that case. Otherwise Copy zeroes output and Update copies base when needed.
For multiple Copy or Update records, the executor can launch their packed owner
blocks in one grid (up to 65535 blocks); a larger combined grid, one record,
zero batches or no work retains the original per-record route. Copy threads
scan the packed block-to-record mapping; Update resolves it once per block and
broadcasts the selected record to its threads. Both use the existing XLA-owned
descriptor, without an additional device allocation or host-to-device operand.
This grouping requires disjoint destinations across Copy/Update records, as
specified by the producer contract; native does not validate cross-record
disjointness. Update still copies the base before writes when needed and retains
its typed per-batch coefficient and zero/no-read semantics. Accumulation and
Reduction choose sequential owner or shared parallel partial/finish execution
per record. Dot uses the same sequential kernel for a single contribution, and
the parallel fiber strategy for longer records; single-chunk fibers write
directly to output. Accumulation, Reduction and Dot retain ordered per-record
launches; Copy/Update's grouped blocks need no ordering across disjoint records.
All launches use the XLA-provided stream.
Update, Accumulation and Reduction read coefficients on device. No handler reads device data on host or allocates
device memory per
call, and none synchronizes. Immediate launch errors are reported by the handler;
asynchronous errors may surface at a later JAX synchronization. Ordered launches
preserve overlapping-record Accumulation/Reduction semantics, but do not relax
Copy/Update destination disjointness requirements. The native packer maps
Copy/Update axes to output owners, Dot axes to a single owner's fiber, and
Accumulation/Reduction axes to their validated owner/fiber groups. The two
device address lanes denote source/output for Copy, Update, Accumulation and
Reduction and left/right inputs for Dot; Dot's output is the batch scalar.
Reduction retains explicit roles in its semantic descriptor and zeros ignored
fiber output strides before packing. The CUDA device descriptor format is
internal to the lowering/FFI boundary.

## Reproducible CUDA execution diagnostic

The bounded script described in [benchmarks/README.md](../benchmarks/README.md)
measures Copy, Update, Dot, Accumulation and Reduction on the installed CUDA
backend. It covers short and long fibers, single and multiple owners, multiple
Dot records and conjugated complex Dot. It checks an independent NumPy result,
preloads input buffers, compiles before warmup, and synchronizes every timed
call. Reported latency includes JAX invocation and synchronization, not just
kernel time; compile time is reported separately. Results from different builds
must use the same GPU, input cases and sampling settings. This is a regression
diagnostic, not a cross-backend or cross-device performance ranking.

## Testing

The bounded public TensorMap matrix is in `tests/tensor/test_cuda_integration.py`.
It has explicit CPU/CUDA parametrization, so CPU-only runs execute substantive
CPU cases rather than skipping the entire module. The matrix fixes JAX matmul
precision to `highest` without changing Tensor0 runtime defaults.

Primitive GPU integration tests are in `tests/stride/jax/cuda/test_copy.py`,
`tests/stride/jax/cuda/test_update.py`,
`tests/stride/jax/cuda/test_accumulation.py`, `tests/stride/jax/cuda/test_dot.py`,
`tests/stride/jax/cuda/test_reduction.py`. Standalone native CUDA contracts live
in `tests/stride/native/cuda/`, separate from JAX GPU integration. Tests requiring
native GPU support skip without a CUDA-enabled extension/device. The standalone
Copy/Update/Accumulation/Dot/Reduction contracts instead need an existing toolkit/device, accept
`NVCC`,
`CUDA_HOME` and `TENSOR0_CUDA_ARCH` (default `sm_80`), and exercise actual buffer
alias rejection and stream dependency ordering. A skip is not a validation pass.

Run against an isolated CUDA-enabled candidate, and run CPU stride regressions
with CPU placement separately. Registration/build tests cover CPU-only and
requested-but-unavailable configurations. No speedup, sanitizer validation or
universal toolkit compatibility is implied by the optional build.
