# Tensor0

Tensor0 is an experimental symmetric tensor library with Rust-backed structural metadata and JAX-backed tensor storage; it is not yet a production-ready tensor-network framework, and the API may change while the core representation is being developed.

Documentation: <https://yitan1.github.io/Tensor0/>

## Features

- Built-in sector families including `U1Irrep`, `SU2Irrep`, cyclic `ZNIrrep`
  aliases, fermion parity, and selected product-sector aliases.
- `space(...)` and `hom(codomain, domain)` constructors for symmetric linear
  map spaces.
- `TensorMap` storage backed by JAX arrays.
- Block access, blockwise composition, compact SVD, and truncation helpers.
- `permute`, `braid`, `transpose`, and `repartition` transform helpers.
- Symmetry-aware index `flip` and `twist`, binary `tensorcontract`, single-tensor
  `tensortrace`, named-label `contract`, and integer-label `ncon` operations.
- `to_dense(...)` and `from_dense(...)` for small correctness checks.
- JAX `jit` and `grad` support through pytree registration.

## Installation

Tensor0 is currently installed from source. It requires Python 3.11 or newer and
a Rust toolchain.

The recommended setup is to install Tensor0 into a virtual environment:

```bash
git clone https://github.com/yitan1/Tensor0.git
cd Tensor0

python -m venv .venv
source .venv/bin/activate
python -m pip install .
```

If you manage your own Python environment, `python -m pip install .` can also
be run without creating a virtual environment.

## Local Development

From a checkout:

```bash
uv sync --group dev
uv run maturin develop
```

Run the Python and Rust tests:

```bash
uv run pytest tests -q
cargo test
```

Run the public example:

```bash
uv run python examples/basic_usage.py
uv run python examples/contractions.py
```

## Minimal Example

```python
import jax.numpy as jnp

from tensor0 import TensorMap, U1Irrep, get_degeneracystructure, hom, space

v = space(U1Irrep, {0: 2, 1: 3})
h = hom((v,), (v,))

total_dim = get_degeneracystructure(h).total_dim
tensor = TensorMap(h, jnp.arange(total_dim, dtype=jnp.float32))

for sector, block in tensor.blocks():
    print(sector, block.shape)
```

See `docs/usage.md`, `docs/contractions.md`, and the two scripts under
`examples/` for more examples.

## Acknowledgments

Tensor0's main design and architecture are based on [TensorKit.jl](https://github.com/Jutho/TensorKit.jl). Tensor0 focuses on bringing these ideas to Python with JAX-backed tensor storage and automatic differentiation support.
