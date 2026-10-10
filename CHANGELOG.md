# Changelog

User-facing changes are recorded here. Before 1.0, patch releases preserve the
documented public API; breaking changes require a minor release and migration
notes. See [Support and compatibility](docs/support.md).

## Unreleased

## 0.1.0 — Release candidate (not published)

This entry summarizes the initial public feature set, not a published release
or a dated release announcement.

### Features

- Rust-backed sector, graded-space and fusion-tree metadata with built-in
  Trivial, U(1), cyclic, SU(2), fermionic and selected product sector families.
- JAX-backed ordinary and diagonal tensor maps, symmetry-allowed block access,
  composition, transforms, traces and symmetry-aware contractions, including
  named and integer-label tensor networks.
- Blockwise solves, inverse/pseudoinverse, QR/LQ, SVD, Hermitian eigendecomposition
  and truncation; morphism and explicit-key random constructors.
- JAX pytree, `jit` and differentiation integration, including TensorMap Riesz
  gradient wrappers, within the documented operation-specific boundaries.
- Dense conversion for correctness checks and direct dense-array paths for
  Trivial symmetry; executable examples and reproducible benchmark suites.

### Packaging and compatibility

- Set Python, Rust workspace and citation versions to **0.1.0**.
- Target Linux x86_64 CPU wheels for **CPython 3.11–3.14**, pending successful
  candidate matrix verification. Source installation remains the documented
  route; no available wheels or PyPI publication are announced.
- Pin JAX/JAXLIB to **0.10.1**; source builds require Rust **>=1.87** and C++20.
- Provide CPU CI, typing checks, sdist-to-wheel inspection and isolated installed
  checks, with a manually approved, exact-artifact publication policy.

### Limits

- Experimental, pre-1.0 tensor primitives, not a complete simulation framework
  or a production-readiness/performance guarantee.
- No GenericFusion, anyonic braiding, arbitrary dynamic product sector families,
  automatic contraction-order optimization or mutable public block views.
- Nontrivial dense conversion does not enable generic dense numerical execution.
  Value-dependent truncation runs eagerly; differentiation has documented
  rank/spectrum and operation-specific restrictions.
- CUDA is opt-in and experimental, with a bounded operation/dtype/AD subset;
  CPU verification does not establish CUDA correctness. Other platforms,
  architectures and JAX versions are not supported execution targets.

See [Usage](docs/usage.md), [Contractions](docs/contractions.md) and
[CUDA support](docs/cuda.md) for detailed contracts and restrictions.
