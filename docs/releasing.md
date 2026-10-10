# Release and artifact acceptance

This is a maintainer checklist, not a statement that packages have been published
or that a release has passed. Tensor0 currently uses version **0.0.0**; keep it
until an intentional release is coordinated. Do not tag or publish merely to
exercise packaging. Source installation remains the documented user route.

## Candidate preparation

- [ ] Select and record a clean source commit, intended distribution version,
  and CPU-only or explicitly experimental CUDA scope. For an intentional
  release, coordinate Python, Cargo and citation metadata rather than inferring
  a release number from `Unreleased`.
- [ ] Review public API changes, migration notes and known limitations in
  [CHANGELOG.md](https://github.com/yitan1/Tensor0/blob/main/CHANGELOG.md).
  Move entries out of `Unreleased` only when the release is actually prepared.
- [ ] Verify dependency pins and vendored XLA FFI headers agree with JAX/JAXLIB
  0.10.1, and check licenses/notices and package metadata.
- [ ] Run the [development verification](development.md#verification) and strict
  [documentation build](development.md#documentation-site). Record exact
  commands, toolchain/runtime versions, outcomes and skips. Do not claim a
  hosted Python matrix unless those jobs actually executed successfully.

## Build and inspect distributions

The [artifact workflow](https://github.com/yitan1/Tensor0/blob/main/.github/workflows/artifacts.yml)
is an artifact-only validation route, not PyPI publishing. Its configured target
is Linux x86_64 CPU, CPython 3.11 and manylinux_2_28 (glibc >=2.28); it builds a
wheel from the extracted sdist, inspects distributions and runs an isolated
installed-wheel public smoke check. This describes the workflow's intent, not
an assertion that a hosted run has passed. Consult the actual run logs and
artifacts. Its smoke check does not replace the full CPU suite or validate CUDA,
other architectures or other Python wheel ABIs.

The custom archive check only covers Tensor0-specific contents: native build
inputs, Python/type files, declared license files and excluded private/build
paths. It reads license patterns from `pyproject.toml` rather than duplicating
package version or dependency constraints. Maturin generates metadata and wheel
tags, auditwheel checks binary compatibility, and isolated installation plus
`uv pip check` verifies dependency consistency. License-file presence is not a
legal compliance review; metadata still needs maintainer review before release.

In a clean candidate checkout with the existing build tools available:

```bash
CARGO_BUILD_JOBS=1 maturin build --release --out dist
maturin sdist --out dist
```

Use an empty candidate output directory, not a mixture of old artifacts. Memory
limits still apply; see [Installation](installation.md). These commands produce
local artifacts, not uploads. Platform/interpreter tags describe the build;
they do not themselves establish portability or manylinux compliance.

- [ ] Inspect wheel and sdist contents: Python package, compiled extension in
  the wheel, native C++ sources/headers and required build inputs in the sdist,
  license files and correct metadata. Check that private files, caches and
  local benchmark outputs are absent.
- [ ] Record artifact filenames and SHA-256 hashes, source commit, Python,
  Rust/C++ toolchains, Linux architecture and build settings.
- [ ] Rebuild a wheel **from the unpacked sdist**, in isolation from the original
  checkout, to prove the source archive is sufficient. Do not reuse the original
  checkout's extension or Cargo build outputs as acceptance evidence.

## Accept the installed artifacts

Test both the checkout-built wheel and the sdist-rebuilt wheel in **separate,
fresh non-editable virtual environments**, with pinned runtime dependencies.
For example, replace the placeholder with one exact candidate filename:

```bash
python3.11 -m venv /tmp/tensor0-candidate
/tmp/tensor0-candidate/bin/python -m pip install /absolute/path/to/candidate.whl
/tmp/tensor0-candidate/bin/python -m pip check
```

- [ ] Run outside the checkout with no source-tree `PYTHONPATH`. Inspect
  `tensor0.__file__` and `tensor0._native.__file__` to confirm both resolve into
  that environment, and check the installed distribution version.
- [ ] Run the [installed CPU smoke check](installation.md#check-the-installed-package)
  using the candidate environment's interpreter. Import alone is insufficient:
  compile and synchronize a native numerical operation.
- [ ] Execute both public example scripts using that interpreter and explicitly
  CPU placement (`JAX_PLATFORMS=cpu`). They may be copied into a neutral working
  directory; do not substitute `uv run`, which could select the checkout's
  development environment.
- [ ] Run the applicable Python suite against the installed package, with test
  tooling provisioned separately. Verify import origins again, including in
  subprocess tests. Retain the complete checkout for source-dependent tests;
  those checks supplement, rather than replace, installed numerical acceptance.
- [ ] Check declared Python 3.11 baseline acceptance explicitly. Other Python
  versions count only as tested versions with recorded results. Package metadata
  or a local run on a different interpreter is not a 3.11 validation pass.
- [ ] If a CUDA artifact is proposed, separately execute the bounded CUDA tests
  on a compatible GPU and record toolkit/driver/architecture and skips. A CPU
  pass does not establish CUDA support. Do not generalize beyond the documented
  [CUDA matrix](cuda.md).

## Publish only after approval

- [ ] Resolve failures and disclose untested configurations. Preserve candidate
  evidence and hashes; an artifact-building CI job is not automatically a full
  numerical or portability certification.
- [ ] Confirm maintainer approval, destination, version and artifact selection
  before any tag or upload. Never upload test artifacts under an intended release
  version without coordinating that release.
- [ ] Only after publication, update installation links to artifacts that really
  exist and describe their verified platform/interpreter scope. Recheck a fresh
  install of the published artifact, not just the local build.

No signing, reproducible-byte builds, universal wheel portability or release
automation guarantees are implied by this checklist.
