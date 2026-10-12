# Release and artifact acceptance

## Current release

**0.2.0 was published on 2026-10-12**:
[Core on PyPI](https://pypi.org/project/tensor0/0.2.0/) ·
[CUDA companion on PyPI](https://pypi.org/project/tensor0-cuda12/0.2.0/) ·
[GitHub Release v0.2.0](https://github.com/yitan1/Tensor0/releases/tag/v0.2.0).
All eleven published distributions have hashes matching the original accepted
artifacts: eight core wheels and one core sdist, plus one repaired Linux addon
wheel and one addon sdist. The tested CPU wheel matrix is CPython **3.11–3.14**
on Linux **x86_64**, manylinux_2_28 (**glibc >=2.28**), and macOS **15+ ARM64**.
The experimental CUDA pair passed bounded A100 (`sm_80`) acceptance before
publication. The immutable `v0.2.0` tag identifies source commit
`d679afe1b5930b57dd199f9107e7036796da672c`, not later maintenance changes on `main`.
Historical 0.1.0 artifacts and its tag remain unchanged; never move a release tag.

The following checklist is for **future releases**, not a historical audit.
Each release needs its own acceptance evidence and explicit maintainer approval.
Do not tag or publish merely to exercise packaging.

## Candidate preparation

- [ ] Select and record a clean source commit, intended distribution version,
  and CPU-only or explicitly experimental CUDA scope. For an intentional
  release, coordinate Python, Cargo and citation metadata rather than inferring
  a release number from `Unreleased`.
- [ ] Review public API changes, migration notes and known limitations in
  [CHANGELOG.md](https://github.com/yitan1/Tensor0/blob/main/CHANGELOG.md).
  Keep the new release entry marked as a candidate until publication is confirmed.
- [ ] Verify dependency pins and vendored XLA FFI headers agree with JAX/JAXLIB
  0.10.1, and check licenses/notices and package metadata.
- [ ] Run the [development verification](development.md#verification) and strict
  [documentation build](development.md#documentation-site). Record exact
  commands, toolchain/runtime versions, outcomes and skips. Do not claim a
  hosted Python matrix unless those jobs actually executed successfully.

## Build and inspect distributions

The [artifact workflow](https://github.com/yitan1/Tensor0/blob/main/.github/workflows/artifacts.yml)
is an artifact-only validation route, not PyPI publishing. The current wheel
matrix, released in 0.2.0, is CPython **3.11–3.14** on both Linux x86_64 CPU
(manylinux_2_28, glibc >=2.28) and macOS **15+ ARM64 CPU**
(`macosx_15_0_arm64`, not universal2). Hosted macOS 15 acceptance passed for 0.2.0.
Verify this matrix anew for each release. Wheels must be built from the extracted
sdist, inspected and tested in isolated installed environments. Consult the
actual workflow configuration, run logs and artifacts: a target matrix is not
an assertion that hosted jobs have passed. Smoke checks do not replace the full
CPU suite or validate CUDA, other architectures or untested Python wheel ABIs.

Candidate preparation is manual on `main`. The workflow assembles one sdist,
eight interpreter/platform-specific wheels and `SHA256SUMS` into
`tensor0-dist-<sha>`, where `<sha>` is the full candidate commit hash. It does
not create tags or publish.

The Linux job builds the single shared core sdist once. All eight wheel jobs
consume those exact source bytes; the four macOS jobs extract outside the checkout
and use a fresh native target with no cache, `/usr/bin/clang++`, a 15.0 deployment
target and one Cargo build slot. Native ARM64 Python/Rust, macOS 15, matching
FFI headers, Mach-O architecture, wheel metadata, installed origin, wheel and
native payload hashes, native CPU availability and pinned runtime are checked.
All eight ABIs run public smoke/examples against the original non-editable
wheel, with pinned native CPU FFI availability, installed origin, wheel hash and
native payload equality checked. On each platform, Python 3.11 and 3.14 run the
full installed-wheel Python suite (excluding only the standalone C++ contract
file). Python 3.12 and 3.13 run bounded existing copy, update, reduction, dot and
accumulation JIT/AD contracts instead; their wheel jobs remain mandatory.
macOS Rust core and standalone C++ contracts run once on 3.11. Linux Rust
workspace and standalone C++ contracts run once in required same-commit CI.
No CUDA extra is installed on macOS. Serial compilation can still exceed hosted
runner memory; configuration alone is not a successful run.

For code/test/build changes or manual CI dispatch, daily Linux CI runs full
Python tests on 3.11 and 3.14, retaining all build-check files on both interpreters:
these include Python-driven packaging and metadata checks, not just
platform-independent compiler checks. Rust/C++ contracts run
only on the baseline interpreter; the blocking Pyright job is unchanged.
macOS remains a manual/release gate, not an additional daily build matrix.
No numerical cases or tolerances are removed or relaxed. Documentation-only
pushes and PRs use a light CI classification/result gate and strict documentation
check without native builds or Pyright; their skipped CPU jobs do not count as
release evidence. For a documentation-only candidate SHA, manually dispatch
**CI** on `main` to obtain full same-commit CPU/typing verification. The publisher
accepts a successful full push or manual CI run from the same repository on
`main`, and still requires both original CPU endpoint jobs and blocking Pyright
to be individually successful. A newer light green run cannot hide another
eligible full run at the same SHA. See
[Daily CI triggers and required checks](development.md#daily-ci-triggers-and-required-checks).

The intermediate-ABI contracts select these existing pytest functions (all
parameters are retained):

- `tests/stride/jax/ad/test_copy.py::test_broadcast_map_transpose_accumulates_and_supports_nested_ad`
- `tests/stride/jax/ad/test_update.py::test_update_jvp_vjp_and_linear_transpose`
- `tests/stride/jax/ad/test_reduction.py::test_mixed_reduction_forward_coefficient_ad_batch_and_nested_transpose`
- `tests/stride/jax/ad/test_dot.py::test_vmap_symbolic_zero_and_native_lowering`
- `tests/stride/jax/ad/test_accumulation.py::test_compact_vmap_and_float32_derivatives_use_native`

These cover signed/broadcast copy and nested transpose, float32/complex64 update,
half/bfloat16/float32/float64/complex reduction, real/mixed-complex dot and native
float32 accumulation, including JIT, batching and forward/reverse AD. They are
compatibility probes, not substitutes for endpoint numerical coverage.
Changes to these gates apply only to future source commits; existing fixed-SHA
runs must be assessed using their own workflows and verification scripts.

The custom archive check only covers Tensor0-specific contents: native build
inputs, Python/type files, declared license files and excluded private/build
paths. It reads version and license patterns from `pyproject.toml` and checks the fixed
release ABI/platform tags against wheel metadata. Maturin generates metadata and wheel
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
  Rust/C++ toolchains, OS/architecture and build settings.
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
- [ ] If an experimental CUDA artifact is proposed, require
  [manual installed-wheel GPU acceptance](cuda.md#manual-installed-wheel-acceptance)
  of the exact candidate bytes and matching clean source commit, plus recorded
  CPU/no-GPU checks of that wheel. Retain build provenance, hashes, reports,
  driver/device/architecture and skips. Source-native contracts supplement but
  do not replace wheel tests. A CPU pass does not establish CUDA support; do not
  generalize beyond the documented [CUDA matrix](cuda.md).

The separate `cuda-build.yml` route retains path-filtered PR compile validation,
which is **not** a final release candidate. For a final candidate, manually run
it on `main` with the successful CPU `candidate_run_id` from `artifacts.yml` at
exactly the same commit. The workflow verifies the producer's repository,
workflow ID/path, manual event, main branch, completed success and source SHA,
then downloads `tensor0-dist-<sha>` and validates its exact nine-file distribution
set plus `SHA256SUMS` against the current source version. **No core rebuild**
occurs in this mode; `artifacts.yml` remains the sole final CPU producer.

The addon is built once from its independent sdist on the pinned manylinux_2_28
baseline with verified CUDA components. Its raw `py3-none-linux_x86_64` wheel
must undergo genuine `auditwheel repair --plat manylinux_2_28_x86_64 --only-plat`
without exclusions. Only the repaired addon is a candidate; no NVIDIA libraries
may be bundled. The fixed local forwarder eliminates CUDA `DT_NEEDED`, not by
binary patching or blind retagging. Configured repair is not actual compliance
evidence. The resulting `tensor0-cuda-build-<sha>` contains eight unchanged core
wheels, core sdist, addon wheel, addon sdist, `build-info.json` and `SHA256SUMS`.
The manifest covers all eleven distributions. Provenance binds the CPU producer
run ID/URL, commit, exact **Linux** CPython 3.11 pair hashes and both sdist hashes.

Require successful CPython 3.11–3.14 installed checks with each **Linux** CPU wheel and the
**same** addon: CPU smoke before adding the addon with `--no-deps`, actionable
missing-runtime failure, then runtime-only registration/statistics and unchanged
core native hash. No GPU kernels execute and no packages publish in this route.
Artifacts uploaded by the build job are not accepted until all ABI jobs succeed.
Retain those run results and exact-pair GPU evidence separately. Duplicate core
files in the CUDA artifact are acceptance conveniences, not a second publishing
input; publish the original verified CPU bytes only once.

The published 0.2.0 distributions use precise pairing pins; Cargo also uses
version `0.2.0`. Historical CPU 0.1.0 and its immutable artifacts remain untouched.
The extra selects the addon and GPU JAX automatically. For future candidates,
version selection alone is not publication approval, and earlier combined-wheel
CPU/A100 acceptance does not validate a new pair.
Require independent out-of-tree sdist builds, actual run evidence, then
[exact-pair acceptance](cuda.md#manual-installed-wheel-acceptance): install the
core for CPU regression first, add only addon/dependencies to the SAME environment,
and verify unchanged core bytes before GPU tests. Both installed direct URLs,
wheel hashes and clean source provenance are mandatory. No GPU runner or remote
driver modification is required by this implementation phase.

The `publish.yml` path stages **both distributions** from these exact
candidates, without rebuild or repair. `release_candidate=true` identifies a
manual candidate only: publication also requires completed successful runs and
all four ABI jobs, plus separate production CPU/GPU acceptance. Implementing
this path does not authorize a commit, push, dispatch or publication. For each
future release, fresh final-version artifacts and all acceptance evidence are
required. Earlier development-version artifacts do not validate a final release pair.

## CUDA distribution licensing

Tensor0's own code remains MIT. The dynamic CUDA prototype ships an optional
companion with 11 fixed, SDK-typed local CUDA forwarders to external
`libcudart.so.12`, not an embedded static runtime. It has no CUDA `DT_NEEDED`.
Neither cudart nor the NVIDIA driver may be bundled in these wheels. The external
runtime is supplied by `nvidia-cuda-runtime-cu12>=12.9.79,<13`; JAX's CUDA
dependencies are also provisioned separately. CPU-only binaries do not link CUDA.

As with other third-party components, before release a maintainer must review
rights and notices against the **actual candidate contents and source
provenance**, including applicable CUDA terms. Inspect the actual wheel/sdist payload and license metadata,
exclude runtime/driver and standalone toolchain files, and resolve any remaining
rights questions before approval. Do not infer NVIDIA sample-source use merely
from runtime linkage. License-file presence and numerical acceptance are not
legal clearance. Publishing requires separate maintainer approval for both projects.

## Publish only after approval

The publisher handles a **unified two-project release**, not a CPU-only mode.
Version 0.2.0 has been published for both Python projects; future releases must
complete candidate verification, exact-pair acceptance and explicit maintainer
approval before dispatch. The literal stable
`vX.Y.Z` tag must match Python, Cargo and addon source versions at the candidate
commit; development versions are deliberately rejected, not normalized.

Configure PyPI Trusted Publishing for **both `tensor0` and `tensor0-cuda12`**
with this exact GitHub identity:

| Publisher field | Value |
| --- | --- |
| Owner / repository | `yitan1` / `Tensor0` |
| Workflow filename | `publish.yml` |
| GitHub environment | `pypi` |

Each project's owner must configure its Trusted Publisher (a pending publisher
for a new project, or a publisher in an existing project's settings). The addon
project name/ownership and publisher require owner setup for a new project; a missing public page
does not reserve the name. The action can upload both projects using the same
workflow/environment identity only when **both** publishers are configured.
No API mutation or credential provisioning is performed by the verifier.

Both Trusted Publishers are configured for the published 0.2.0 pair. The GitHub
`pypi` environment restricts deployments to `main`; it currently has **no required
reviewers**, so there is no second GitHub environment approval after dispatch.
Explicit maintainer approval before dispatch remains release policy. A required
environment reviewer is an optional additional guard and must be configured
separately: referencing the environment in YAML does **not** enable it. Trusted
Publishing uses OIDC, not a stored PyPI token. Recheck both publisher identities
and environment protections for each release; the verifier does not prove their
external configuration.

### Manual evidence transport and review

Use the existing release for the intended immutable stable tag, preferably still
**draft**, to attach one manually prepared asset named `cuda-acceptance.tar.gz`.
A published release for the same tag is also allowed; changing its publication
status does not waive any gate. Do not supply an arbitrary external URL. Run the
production [installed-wheel acceptance](cuda.md#manual-installed-wheel-acceptance)
on the clean candidate commit with the exact build-info and **Linux** CPython 3.11 pair:
first CPU-only smoke before installing the addon, then GPU acceptance in the
same environment using that CPU report. Never substitute an experimental report.

Package the two output directories with these exact archive-relative files:

```text
cpu/report.json
cpu/preflight.log
cpu/cpu-smoke.log
gpu/report.json
gpu/preflight.log
gpu/suite-0.log
gpu/suite-0.xml
gpu/suite-1.log
gpu/suite-1.xml
gpu/suite-2.log
gpu/suite-2.xml
```

Do not add an outer archive directory, alternate reports, links or unrelated
files. Preserve original reports and logs unchanged. The CPU runner reports an
empty `suites` list and writes one `cpu-smoke.log`; it does not write CPU JUnit.
The production clean-source check is performed in preflight: the report records
`checkout.commit`/`checkout.root`, and consumes `checkout_status` rather than
emitting a `source.is_clean` field. The publisher follows this actual schema.

Review all logs/JUnit, source provenance, driver/device scope and skips before
recording the uploaded asset's numeric ID and the archive's lowercase SHA-256.
The verifier fetches the asset through authenticated GitHub API, checks membership
in the selected tag's release and verifies the pinned archive bytes. It reads
members without extracting or executing them. Reports must have production
`accepted=true`, correct CPU/CUDA modes, exact source/build-info, installed
versions/origins and direct URL wheel hashes. Both core binary hashes must match
the unique native `.so` bytes inside the verified **Linux** CPython 3.11 wheel, not merely
each other. The addon binary hash must likewise match `tensor0_cuda12/_stride_cuda.so`
in the verified addon wheel. Production preflight checks both installed payloads
against their wheel bytes, and records `cuda_binary_sha256` for the addon. CPU
and GPU reports must retain the same environment prefix, interpreter and native
core origin; installing the addon must not substitute a second virtual environment.
The GPU suites must be exactly the three production paths, each once, with zero
exit status, nonzero passes and no failures/errors. Only the two production
CPU-only skip reasons are allowed; XML and report statistics must agree.

These manually supplied reports are **not cryptographic execution attestation**.
An accepted flag and a pinned hash cannot prove that a human did not fabricate a
report. The archive hash binds reviewed bytes, not independent GPU execution.
The approving maintainer must examine source and logs; no automatic GPU CI, cloud
runner or remote driver modification is added.

### Dispatch and exact-artifact handoff

- [ ] Select successful manual own-repository `main` runs: CPU `artifacts.yml`
  and same-commit CUDA `cuda-build.yml`. Record both run IDs. Path-filtered PR
  CUDA artifacts are not release candidates.
- [ ] Confirm same-commit successful full push or manual CPU CI on own `main`
  for Python 3.11 and 3.14 and the blocking
  `Public-package Pyright (Python 3.11)` job. Advisory typing is not
  a substitute. The candidate commit must be an ancestor of `main`.
- [ ] Verify the immutable stable tag and all three source versions agree.
  Never move an existing tag to another candidate.
- [ ] Require all eight CPU producer wheel jobs and assembly to succeed. Each
  ABI must pass exact installed-wheel smoke/examples and native CPU contracts:
  full Python suites on 3.11/3.14, bounded major-operation JIT/AD tests on
  3.12/3.13. Rust/C++ contracts run once per platform as described above.
  Skipped or missing wheel jobs are rejected by publication verification.
- [ ] Check CUDA build and `verify-candidate` jobs and all four
  `Exact CPU candidate + same addon / Python 3.X (no GPU)` jobs succeed.
  Artifact upload and `release_candidate=true` alone are insufficient: upload
  occurs before the ABI matrix finishes. Skipped/missing/failed jobs are rejected.
- [ ] Review production acceptance and upload its archive to the selected release;
  record the asset ID and SHA-256. Confirm rights/metadata and destination.
- [ ] Dispatch `publish.yml` on `main` with `candidate_run_id`,
  `cuda_candidate_run_id`, `release_tag`, `acceptance_asset_id` and
  `acceptance_sha256` only after explicit approval and external publisher setup.

The verify job downloads the two identified artifacts, validates basename-only
unique manifests, rejects extra distributions, and compares all nine duplicate
core files byte-for-byte. It checks wheel/sdist names and metadata versions,
wheel tags, addon core pin and external runtime bound, rejects direct URL
metadata dependencies, and binds build-info to both producer runs and hashes.
It stages **exactly eleven files**: eight original CPU ABI/platform wheels, original CPU
sdist, the repaired `py3-none-manylinux_2_28_x86_64` addon wheel and addon sdist.
No raw Linux addon is accepted. No build or repair occurs in publishing.

The immutable `verified-pypi-dist-<sha>` artifact is the only input to the
publish job in the `pypi` environment, which has **no checkout or build**, only artifact
download and the Trusted Publishing action. Reports and verification hashes
are saved separately as `verified-release-evidence-<sha>` (14-day CI retention);
they are never sent to PyPI. Distribution staging has 7-day CI retention. Keep
the pinned release asset and durable acceptance records beyond CI expiry.

### Partial upload and retries

Uploading two projects/eleven files is **not atomic**. If some files upload and a
later upload fails, do not rebuild, overwrite files, move the tag or blindly
retry with `skip-existing`. Review the existing PyPI filenames and SHA-256 values
against `verification.json` and the immutable staging artifact for both projects.
If any existing file differs, stop and investigate; PyPI versions cannot be
replaced. Only missing files with the exact previously verified bytes may be
retried through a separately reviewed recovery procedure. This workflow does
not add a generic recovery mode or silently skip existing files.

### Maintainer command examples

**Examples only—not authorization to build or publish now.** After explicit
approval for a future stable candidate, record the successful CPU run and use it
as the input to manual CUDA candidate preparation. With the matching tag,
reviewed release evidence asset and both publishers configured:

```bash
# Replace every placeholder with reviewed future release values.
gh workflow run publish.yml --repo yitan1/Tensor0 --ref main \
  -f candidate_run_id=123456789 -f cuda_candidate_run_id=234567890 \
  -f release_tag=vX.Y.Z -f acceptance_asset_id=345678901 \
  -f acceptance_sha256=REVIEWED_LOWERCASE_SHA256
```

Dispatch requires explicit maintainer approval; any separately configured
`pypi` environment review is an additional guard, not the current configuration.
Neither a tag nor a GitHub release automatically publishes. Only after actual publication,
update release status/date and installation links and recheck fresh installs of
the published files. No signing, reproducible-byte builds, universal wheel
portability or independent execution-attestation guarantee is implied.
