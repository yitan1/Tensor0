# Examples

The public example script is `examples/basic_usage.py`.

Run it from the repository root:

```bash
uv run python examples/basic_usage.py
```

The example covers:

- layout and storage construction,
- dense roundtrip checks,
- blockwise composition,
- compact SVD reconstruction,
- U1, fermion parity, and SU2 transforms,
- JAX `jit` and `grad` smoke usage.

The example is covered by:

```bash
uv run pytest tests/test_examples.py -q
```
