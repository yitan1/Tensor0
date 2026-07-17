# Tensor0

Tensor0 is an experimental symmetric tensor library with Rust-backed structural
metadata and JAX-backed tensor storage.

Use this documentation for the current checkout. The public API is still
evolving, and Tensor0 is not yet a production-ready tensor-network framework.

## Start Here

- [Usage Guide](usage.md): current public API examples and boundaries.
- [Contractions](contractions.md): primitive and tensor-network contraction
  interfaces.
- `examples/basic_usage.py`: executable public example covered by pytest.
- `examples/contractions.py`: executable contraction and JAX example.

## Current Scope

Tensor0 currently covers typed sector metadata, graded spaces, `TensorMap`
storage, block access, composition, SVD helpers, transforms, symmetry-aware
contractions, deterministic morphism constructors, explicit tensor products
and random construction, inverse/pseudoinverse, direct solves, QR/LQ- and
SVD-backed orthogonalization, full/truncated Hermitian eigendecomposition,
numerical structure predicates, monoidal-unit insertion and removal, correctness-first
dense conversion, and JAX `jit` / `grad` smoke gates.
