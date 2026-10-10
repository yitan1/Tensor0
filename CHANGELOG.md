# Changelog

User-facing changes are recorded here. Tensor0 is pre-1.0 and may make breaking
API changes without a guaranteed deprecation window; see
[Support and compatibility](docs/support.md). This file is not a retrospective
release history for all existing functionality.

## Unreleased

### Packaging and verification

- Add Linux CPU build/test CI, blocking type checks for source/examples/benchmarks,
  and a separate advisory report for existing test typing debt.
- Add candidate sdist-to-manylinux-wheel builds, archive/license inspection and
  isolated installed-package checks. These workflows do not publish packages.
- Add project links, explicit license-file metadata and software citation metadata.
- Make local source annotations precise without changing public API signatures,
  update the accumulation partition callback test for its current signature,
  and isolate the CUDA-linkage test crate from unrelated parent Cargo workspaces.

### Documentation

- Add source-install instructions, C++20/Rust 1.87 build prerequisites and native
  compilation memory guidance.
- Clarify the Linux CPU numerical baseline with pinned JAX/JAXLIB 0.10.1,
  Python 3.11 baseline versus other explicitly CI-tested interpreters, and the
  bounded experimental CUDA scope.
- Document pre-1.0 public API compatibility and installed wheel/sdist acceptance
  criteria without implying publication or unverified portability guarantees.
- Add contributor guidance and a bug-report template; streamline the README and
  documentation navigation.

No versioned release or package publication is announced by these entries.
