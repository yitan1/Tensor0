# Development

Tensor0 is developed from a source checkout with Python, Rust, uv, maturin, and
pytest.

## Local Setup

```bash
uv sync --group dev
uv run maturin develop
```

## Verification

```bash
cargo test
uv run pytest tests -q
uv run python examples/basic_usage.py
uv run python examples/contractions.py
```

## Documentation Site

Build the documentation site locally with:

```bash
uv run --with mkdocs==1.6.1 mkdocs build --strict
```

The public documentation site is deployed to GitHub Pages by
`.github/workflows/docs.yml` on pushes to `main` and by manual workflow
dispatch.

The documentation site is public-facing. Internal planning notes under `local/`
are not part of the public navigation.
