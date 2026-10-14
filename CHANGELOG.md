# Changelog

User-facing changes are recorded here. Before 1.0, patch releases preserve the
documented public API; breaking changes require a minor release and migration
notes. See [Support and compatibility](docs/support.md).

## 0.3.0 — 2026-10-14

Available on [PyPI](https://pypi.org/project/tensor0/0.3.0/), with the optional
[tensor0-cuda12 companion](https://pypi.org/project/tensor0-cuda12/0.3.0/),
and as [GitHub Release v0.3.0](https://github.com/yitan1/Tensor0/releases/tag/v0.3.0).
All eleven published distributions match the original accepted bytes: eight
core wheels, one core sdist, one repaired Linux addon wheel and one addon sdist.

### Fixes

- Fix [issue #1](https://github.com/yitan1/Tensor0/issues/1): homogeneous
  `TensorMap`, `DiagonalTensorMap` and `SectorVector` pytree batches can be
  stacked and returned from `jax.vmap`, with a validated trailing storage
  dimension and leading `batch_shape`. Public constructors still take 1D
  single-tensor storage; metadata describes one tensor. Use (nested) `vmap` to
  consume every batch axis before single-tensor numerical operations, which
  reject residual batches. Whole-batch scalar-loss `grad`/`value_and_grad`
  applies Riesz weights per tensor. Tensor-valued Jacobian tree products remain
  unsupported; see [homogeneous pytree batches](docs/usage.md#homogeneous-pytree-batches).

### Runtime and migration

- Move to Python **>=3.12** and exact JAX/JAXLIB **0.11.2**, with matching
  vendored XLA FFI headers and native build/runtime guards. Upgrade PyO3 and
  Rust `numpy` to **0.29**; source builds require maturin **>=1.15.0,<2**,
  with **1.15.0** as the fixed validation tool.
- Publish validated ordinary CPython **3.12–3.15** wheels on Linux **x86_64**,
  **glibc >=2.28** (manylinux_2_28), and macOS **15+ ARM64 CPU**. All eight wheel
  jobs and same-commit CI passed, including hosted macOS 15 acceptance. Daily
  Linux CPU CI covers **3.12/3.15**. Free-threaded Python is not supported.
- **Python 3.11 migration:** create a Python 3.12+ environment for this release;
  retain pinned 0.2.0/JAX 0.10.1 when Python 3.11 is required. Install the 0.3.0
  wheels with their pinned runtime rather than independently upgrading JAX in
  an old environment; CUDA users should install the matching 0.3.0 core/addon
  pair. Source users must rebuild the native core against 0.11.2 headers and,
  for CUDA, rebuild the exact-version companion from matching source and repeat
  exact-byte CPU/GPU validation. Do not mix 0.2.0 binaries with 0.3.0 source or
  either runtime pin.
- CUDA 12 remains experimental with the existing bounded operation/dtype/AD
  scope. All four CUDA ABI jobs and exact-pair A100 acceptance passed, with
  accepted core bytes unchanged after addon installation. Production CPU
  regression passed **7,318 tests** (**865 skipped**); GPU primitives passed
  **790 tests** (**10 allowed CPU-only skips**), plus **18 boundary** and
  **72 high-level TensorMap** tests. These results do not expand support or
  promise performance. See [Installation](docs/installation.md),
  [Development](docs/development.md) and [Releasing](docs/releasing.md).

## 0.2.0 — 2026-10-12

Available on [PyPI](https://pypi.org/project/tensor0/0.2.0/), with the optional
[tensor0-cuda12 companion](https://pypi.org/project/tensor0-cuda12/0.2.0/),
and as [GitHub Release v0.2.0](https://github.com/yitan1/Tensor0/releases/tag/v0.2.0).

### Packaging and compatibility

- Add macOS **15+ ARM64 CPU** wheels alongside Linux **x86_64**, **glibc >=2.28**,
  each for CPython **3.11–3.14**. All eight core wheels are built from one shared
  sdist and pass installed-wheel acceptance, including hosted macOS 15 checks.
  Prebuilt wheels require no Rust or C++ compiler.
- Publish the optional, version-paired `tensor0-cuda12` companion. The
  `tensor0[cuda12]` extra installs the addon and GPU JAX; the unified CPU core
  remains usable without CUDA and is not rebuilt or replaced by addon installation.
- Use fixed, SDK-typed local forwarders to the external
  `nvidia-cuda-runtime-cu12>=12.9.79,<13` runtime; no NVIDIA runtime or driver
  libraries are bundled. Tensor0's own code remains MIT, with third-party
  license notices included in the distributions.
- Publish eleven original, accepted distributions without rebuild: eight core
  wheels, one core sdist, one repaired Linux addon wheel and one addon sdist.
  All published file hashes match the accepted artifacts.
- Retain JAX/JAXLIB **0.10.1**, Rust **>=1.87** and C++20 source-build requirements.
  Existing CPU installation commands remain valid; CUDA users need the new
  matching companion, not GPU JAX alone or a 0.1.0 core.

### CUDA scope

- Provide experimental Linux x86_64 CUDA execution for Copy, Update,
  Accumulation, Dot and Reduction, with bounded dtype and differentiation support.
  Exact-pair acceptance covers the documented A100 (`sm_80`) primitive and
  high-level TensorMap subset, not general GPU API coverage, full reverse AD,
  multi-GPU support or a performance guarantee. macOS remains CPU-only;
  Windows and other architectures remain outside the supported scope.

See [Installation](docs/installation.md), [Support and compatibility](docs/support.md)
and [CUDA support](docs/cuda.md) for requirements and restrictions.

## 0.1.0 — 2026-10-10

Initial public release, available on [PyPI](https://pypi.org/project/tensor0/0.1.0/)
and as [GitHub Release v0.1.0](https://github.com/yitan1/Tensor0/releases/tag/v0.1.0).

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
- Publish tested Linux x86_64 CPU wheels for **CPython 3.11–3.14**,
  **glibc >=2.28** (manylinux_2_28). Wheel installation requires no Rust/C++
  compiler. All five PyPI distribution files (four wheels and one sdist) have
  hashes matching the accepted candidate.
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
