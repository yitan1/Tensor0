# Tensor0

Tensor0 is an experimental symmetric tensor library with Rust-backed structural
metadata and JAX-backed tensor storage.

Use this documentation for the current checkout. The public API is still
evolving, and Tensor0 is not yet a production-ready tensor-network framework.
[Tensor0 0.3.0 is available on PyPI](https://pypi.org/project/tensor0/0.3.0/),
published on 2026-10-14 UTC. Install with `pip install tensor0==0.3.0` in a virtual
environment; supported prebuilt wheels require no Rust or C++ compiler.

## Runtime Compatibility

Release **0.3.0** requires Python **>=3.12** and exact JAX/JAXLIB **0.11.2**.
All eight ordinary CPython **3.12–3.15** core wheel jobs passed installed-artifact
acceptance on Linux x86_64 (manylinux_2_28, glibc >=2.28) and macOS **15+ ARM64**.
The wheels share one core sdist; daily Linux CPU CI covers **3.12/3.15**.
macOS is CPU-only and free-threaded Python is unsupported. The matching CUDA
pair passed bounded A100 acceptance; its experimental scope is unchanged.
Python metadata does not imply support for arbitrary future interpreters.

Historical **0.2.0** supports CPython **3.11–3.14** with JAX/JAXLIB **0.10.1**.
Python 3.11 users must keep that pinned runtime or migrate. See
[Installation](installation.md#source-install) for migration and native rebuild
requirements, [Support and compatibility](support.md) for platform scope, and
[Releasing](releasing.md) for per-wheel acceptance.

The native stride handler is compiled against its exact XLA FFI header version
and rejects mismatched runtimes. Missing native routes are reported explicitly
rather than hidden by an element-address fallback.
Native stride operations lower through JAX as `stablehlo.custom_call`
operations. Tensor0 does not ship a separate pure-StableHLO stride backend.

Experimental [CUDA support](cuda.md) covers selected operations, dtypes and AD
paths, not the full TensorMap API or full reverse AD.

## Start Here

- [Installation](installation.md): PyPI wheels, source builds and prerequisites.
- [Support and compatibility](support.md): runtime scope and pre-1.0 API policy.
- [Usage Guide](usage.md): current public API examples and boundaries.
- [Contractions](contractions.md): primitive and tensor-network contraction
  interfaces.
- [Examples](examples.md): executable ordinary, symmetric and contraction scripts.
- [Development](development.md): contributor setup and detailed verification.
- [Release checklist](releasing.md): installed-artifact acceptance before publication.

## Current Scope

Tensor0 currently covers typed sector metadata, graded spaces, `TensorMap`
storage, block access, composition, SVD helpers, transforms, symmetry-aware
contractions, deterministic morphism constructors, explicit tensor products
and random construction, inverse/pseudoinverse, direct solves, QR/LQ- and
SVD-backed orthogonalization, full/truncated Hermitian eigendecomposition,
numerical structure predicates, monoidal-unit insertion and removal, correctness-first
dense conversion for symmetric sectors, direct immutable array execution for
Trivial tensors, and JAX `jit` / `grad` smoke gates.
