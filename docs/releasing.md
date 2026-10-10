# Release and artifact acceptance

This is a maintainer checklist, not a statement that packages have been published
or that a release has passed. Python, Rust workspace and citation metadata use
**0.1.0** for the release candidate; no publication or release date is announced.
"Candidate" describes acceptance status: the distribution version is `0.1.0`,
not a PEP 440 prerelease such as `0.1.0rc1`.
Do not tag or publish merely to exercise packaging. Source installation remains
the documented user route.

## Candidate preparation

- [ ] Select and record a clean source commit, intended distribution version,
  and CPU-only or explicitly experimental CUDA scope. For an intentional
  release, coordinate Python, Cargo and citation metadata rather than inferring
  a release number from `Unreleased`.
- [ ] Review public API changes, migration notes and known limitations in
  [CHANGELOG.md](https://github.com/yitan1/Tensor0/blob/main/CHANGELOG.md).
  Keep the 0.1.0 entry marked as a candidate until publication is confirmed.
- [ ] Verify dependency pins and vendored XLA FFI headers agree with JAX/JAXLIB
  0.10.1, and check licenses/notices and package metadata.
- [ ] Run the [development verification](development.md#verification) and strict
  [documentation build](development.md#documentation-site). Record exact
  commands, toolchain/runtime versions, outcomes and skips. Do not claim a
  hosted Python matrix unless those jobs actually executed successfully.

## Build and inspect distributions

The [artifact workflow](https://github.com/yitan1/Tensor0/blob/main/.github/workflows/artifacts.yml)
is an artifact-only validation route, not PyPI publishing. The 0.1.0 candidate
target is Linux x86_64 CPU, CPython **3.11–3.14**, manylinux_2_28 (glibc >=2.28),
pending successful matrix verification. Wheels must be built from the extracted
sdist, inspected and tested in isolated installed environments. Consult the
actual workflow configuration, run logs and artifacts: a target matrix is not
an assertion that hosted jobs have passed. Smoke checks do not replace the full
CPU suite or validate CUDA, other architectures or untested Python wheel ABIs.

Candidate preparation is manual on `main`. The workflow assembles one sdist,
four interpreter-specific wheels and `SHA256SUMS` into
`tensor0-dist-<sha>`, where `<sha>` is the full candidate commit hash. It does
not create tags or publish.

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
- [ ] Verify every candidate target interpreter, CPython **3.11, 3.12, 3.13 and
  3.14**, with recorded build and installed CPU results. Until the matrix passes,
  describe it as a target, not verified coverage. Package metadata or a pass on
  one interpreter is not evidence for another.
- [ ] If a CUDA artifact is proposed, separately execute the bounded CUDA tests
  on a compatible GPU and record toolkit/driver/architecture and skips. A CPU
  pass does not establish CUDA support. Do not generalize beyond the documented
  [CUDA matrix](cuda.md).

## Publish only after approval

Configure PyPI Trusted Publishing for the following exact GitHub identity:

| Publisher field | Value |
| --- | --- |
| Owner / repository | `yitan1` / `Tensor0` |
| Workflow filename | `publish.yml` |
| GitHub environment | `pypi` |

Before the first publication, the intended PyPI project owner's account must
configure a pending Trusted Publisher for the new project with the identity
above. For an existing project, its owner must add the publisher in that
project's settings. A missing public project page does not reserve its name or
establish ownership.

Create the GitHub `pypi` environment, restrict deployments to `main`, and
configure a required reviewer for manual maintainer approval before upload. Referencing `environment: pypi` in workflow
YAML does **not** configure reviewer protection or guarantee approval. Trusted
Publishing uses OIDC rather than a stored PyPI API token. These are outstanding
setup requirements, not claims of a configured publisher or an upload.

- [ ] Resolve failures and disclose untested configurations. Preserve candidate
  evidence and hashes; an artifact-building CI job is not automatically a full
  numerical or portability certification.
- [ ] Select `tensor0-dist-<sha>` from a successful manual `artifacts.yml` run
  on this repository's `main`, recording its run ID, source commit and hashes.
  The commit must remain an ancestor of `main`.
- [ ] Confirm successful same-commit **push** CI on `main`: Linux x86_64 CPU
  jobs for Python 3.11–3.14 and the blocking public-package Pyright job. Advisory
  test typing reports do not replace the blocking check.
- [ ] Confirm the existing release tag (for example, `v0.1.0`) points at exactly
  that candidate commit and matches both its Python and Rust source versions.
- [ ] Confirm maintainer approval, destination, version and artifact selection
  before any tag or upload. Never upload test artifacts under an intended release
  version without coordinating that release.
- [ ] Manually dispatch [publish.yml](https://github.com/yitan1/Tensor0/blob/main/.github/workflows/publish.yml)
  on `main` with `candidate_run_id` and `release_tag`. It verifies the candidate,
  tag, source versions and same-commit CI, then checks the exact distribution
  set and `SHA256SUMS`. The approved `pypi` job uploads those verified bytes.
  **Do not rebuild for publication** or substitute artifacts from another run.
- [ ] Only after publication, update the changelog status and installation links
  to artifacts that really exist and describe their verified scope. Add a
  release date only when known. Recheck a fresh install of the published
  artifact, not just the local build.

### Maintainer command examples

**Examples only—not instructions to publish now.** After maintainer approval
for candidate preparation and after the workflows are available on `main`:

```bash
gh workflow run artifacts.yml --repo yitan1/Tensor0 --ref main
gh run list --repo yitan1/Tensor0 --workflow artifacts.yml --branch main
```

Record the selected successful candidate run ID and commit; verify the required
same-commit CI. Only after explicit release approval, an existing matching tag,
and confirmed PyPI publisher and required-reviewer environment configuration:

```bash
# Replace the example run ID with the verified candidate run.
gh workflow run publish.yml --repo yitan1/Tensor0 --ref main \
  -f candidate_run_id=123456789 -f release_tag=v0.1.0
```

Dispatch does not replace the required review of the `pypi` deployment. Neither
a tag nor a GitHub release automatically publishes. No configuration, tag,
hosted verification or publication is claimed by these examples.

No signing, reproducible-byte builds, universal wheel portability or release
automation guarantees are implied by this checklist.
