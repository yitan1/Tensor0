---
name: Bug report
about: Report a build failure or incorrect Tensor0 behavior
---

## Expected and actual behavior

Describe the expected result and what happened instead. Include the complete
traceback or compiler error as text. For numerical differences, give a reference
result and tolerances rather than only saying that values differ.

## Minimal reproducer

Provide a small, runnable script and exact command (or the source-build command
for installation failures). Include sector/space definitions, shapes, explicit
random seeds/keys, and whether JIT, vmap, JVP, grad or VJP is involved.

```python
# Paste a standalone reproducer here.
```

## Versions and environment (required)

- Tensor0 installed version and source commit, if built from source:
- Installation method (source, editable, or local wheel) and build command:
- Python version:
- JAX and JAXLIB versions (both):
- NumPy version:
- OS, architecture and CPU:
- Backend/device used by the failing arrays (CPU or GPU, model and device count):
- Storage/input/result and coefficient dtypes:
- JAX x64 and matmul precision settings:

For build failures, also include Rust/Cargo and C++ compiler versions, `CXX`,
`CARGO_BUILD_JOBS`/`-j`, available RAM and container/job memory limit. For CUDA,
include whether Tensor0 was built with `TENSOR0_CUDA=1`, toolkit/NVCC and driver
versions, `TENSOR0_CUDA_ARCH`, and the GPU-enabled JAX installation details.

The supported numerical baseline is Linux CPU with JAX/JAXLIB 0.10.1; Python
3.11 is the baseline interpreter and builds require Rust >=1.87 and C++20.
Other Python versions are CI-tested only where explicitly recorded; CUDA is
experimental, not full API support. Reports outside this scope are welcome but
do not imply a support commitment.

## Additional evidence

If relevant, attach compiler logs, import locations (`tensor0.__file__` and
`tensor0._native.__file__`), results with/without JIT, and synchronized device
results. State which checks actually ran and which skipped. Remove credentials,
private paths/data and unrelated output before posting.
