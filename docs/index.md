# Tensor0

Tensor0 is an experimental symmetric tensor library with Rust-backed structural
metadata and JAX-backed tensor storage.

Use this documentation for the current checkout. The public API is still
evolving, and Tensor0 is not yet a production-ready tensor-network framework.

## Start Here

- [Usage Guide](usage.md): current public API examples and boundaries.
- `examples/basic_usage.py`: executable public example covered by pytest.

## Current Scope

Tensor0 currently covers typed sector metadata, graded spaces, `TensorMap`
storage, block access, composition, SVD helpers, transforms, correctness-first
dense conversion, and JAX `jit` / `grad` smoke gates.
