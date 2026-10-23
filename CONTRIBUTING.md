# Contributing to Tensor0

Tensor0 is experimental; discuss substantial API or scope changes in a
[GitHub issue](https://github.com/yitan1/Tensor0/issues) before implementation.
Keep changes focused and include a regression test for bug fixes.

## Review and acceptance

Maintainers may push reviewed, locally tested changes directly to `main`; CI
runs after the push. Feature branches and pull requests are optional for larger
changes or collaboration. No permanent development branch is required. When
using pull requests, keep one per task, not per commit, and separate unrelated
refactoring and formatting. The maintainer may review locally and merge after
CI passes, without a separate GitHub Approve review. Recheck changes made after
local review. Agents must not merge PRs on their own.

Use the PR template to report maintenance cost, verification and changes to
acceptance standards. Human review must approve new dependencies, public APIs,
and changes to tests, numerical tolerances, skips, suppressions or CI rules.
Do not weaken a check just to obtain a green result: provide the requirement or
evidence for changing it. An implementing agent must not approve its own changes.

Reviewers should check scope and acceptance-standard changes first, then inspect
regression tests and the main implementation path. Passing CI does not establish
that an abstraction is necessary or that the code is maintainable.

Run the lightweight maintenance gate with
`.venv/bin/python .github/scripts/check_maintenance.py`.
Its entry-document budgets live in that script; detailed public documentation
belongs in `docs/`. It also checks static Python imports in `src/tensor0/`,
including stubs, for dependencies on tests, examples, benchmarks and private
local code. Dynamic imports and filesystem access still require review.

## Development and verification

The [development guide](docs/development.md) is the detailed source of truth for
local setup, native build profiles, compiler-job memory budgeting, CUDA opt-in,
test ownership and focused test commands. Start with its
[Local Setup](docs/development.md#local-setup) and
[Verification](docs/development.md#verification) sections rather than duplicating
an environment here. [Installation](docs/installation.md) lists build
prerequisites; [Support](docs/support.md) defines the supported numerical baseline.

Before requesting review:

- Run the affected test owners and the complete verification set documented in
  the development guide. Report failures and skipped device tests honestly.
- Keep public exports consistent with the [API inventory](docs/api.md), and
  update usage examples and `CHANGELOG.md` under `Unreleased` for user-facing
  changes. Describe migrations when changing the pre-1.0 API.
- Build documentation with the guide's
  [strict MkDocs command](docs/development.md#documentation-site).
- For packaging changes, follow the
  [installed-artifact acceptance checklist](docs/releasing.md); an editable
  development install is not distribution validation.

Use English for public documentation and code comments. Include exact test
commands and environment details in the pull request; do not claim unexecuted
platforms or Python versions as verified. There is no required automatic version
bump, tag or publication for a contribution.

For bug reports, include versions, device, dtype and a small reproducer. See the [reporting guidance](docs/support.md#reporting-problems).
