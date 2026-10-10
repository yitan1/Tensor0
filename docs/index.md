# Tensor0

Tensor0 is an experimental symmetric tensor library with Rust-backed structural
metadata and JAX-backed tensor storage.

Use this documentation for the current checkout. The public API is still
evolving, and Tensor0 is not yet a production-ready tensor-network framework.
Source installation is the current distribution route; these docs do not assume
published PyPI packages or prebuilt wheels.

## Runtime Compatibility

The current numerical execution baseline supports Linux CPU with JAX and
JAXLIB 0.10.1, with Python 3.11 as the baseline interpreter. Other Python
versions are CI-tested only where explicitly recorded, not universally
supported by virtue of the Python >=3.11 metadata. The native stride handler is compiled against that exact XLA FFI
header version and rejects mismatched runtimes. Other platforms and versions
are not currently supported execution targets, and missing native routes are
reported explicitly rather than hidden by an element-address fallback.
Native stride operations lower through JAX as `stablehlo.custom_call`
operations. Tensor0 does not ship a separate pure-StableHLO stride backend.

Experimental [CUDA support](cuda.md) covers selected operations, dtypes and AD
paths, not the full TensorMap API or full reverse AD.

## Start Here

- [Installation](installation.md): source builds, prerequisites and memory limits.
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
