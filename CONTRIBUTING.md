# Contributing to Tensor0

Tensor0 is experimental; discuss substantial API or scope changes in a
[GitHub issue](https://github.com/yitan1/Tensor0/issues) before implementation.
Keep changes focused and include a regression test for bug fixes.

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

For bug reports, use the issue template with versions, device, dtype and a small
reproducer. See the [reporting guidance](docs/support.md#reporting-problems).
