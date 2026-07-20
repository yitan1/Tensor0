# Examples

The public example scripts are `examples/basic_usage.py` and
`examples/contractions.py`.

Run it from the repository root:

```bash
uv run python examples/basic_usage.py
uv run python examples/contractions.py
```

The basic usage example covers:

- ordinary dense tensors built with `Vect(dim)`,
- layout and storage construction,
- dense roundtrip checks,
- blockwise composition,
- identity and deterministic isometry construction,
- explicit tensor products and explicit-key random isometry construction,
- direct linear solve and numerical predicate checks,
- compact QR canonicalization and truncated Hermitian reconstruction,
- compact SVD reconstruction,
- U1, fermion parity, and SU2 transforms,
- JAX `jit` and `grad` smoke usage.

The contraction example covers:

- binary contraction and partial/full trace,
- named-label `contract` and integer-label `ncon`,
- explicit contraction order and HomSpace output partitions,
- fermionic twist,
- JAX `jit` and differentiation through TensorMap storage.

The example is covered by:

```bash
uv run pytest tests/test_examples.py -q
```
