# Tensor0 Usage Guide

This guide documents the current Tensor0 public API. It focuses on small,
deterministic examples that can be copied into tests or interactive sessions.

## Local Checkout Setup

From the repository root:

```bash
uv sync --group dev
uv run maturin develop
```

Run the current verification set with:

```bash
cargo test
uv run pytest tests -q
```

## Supported Sector Families

Tensor0 currently supports these public sector families and aliases:

- `U1Irrep`
- `Z2Irrep`, `Z3Irrep`, `Z4Irrep`
- `FermionParity`
- `SU2Irrep`
- `FermionNumber`
- `FermionParityU1Irrep`
- `U1SU2Irrep`
- `FermionParitySU2Irrep`
- `FermionParityU1SU2Irrep`

Arbitrary dynamic product sector families are not currently supported.

## Spaces, HomSpaces, and TensorMap

`space(...)` builds an elementary graded space. The dictionary maps sector labels
to degeneracy dimensions. `hom(codomain, domain)` builds the typed linear-map
space used by `TensorMap`.

```python
import jax.numpy as jnp

from tensor0 import (
    TensorMap,
    U1Irrep,
    get_degeneracystructure,
    hom,
    space,
)

v = space(U1Irrep, {0: 2, 1: 3})
h = hom((v,), (v,))

total_dim = get_degeneracystructure(h).total_dim
tensor = TensorMap(h, jnp.arange(total_dim, dtype=jnp.float32))

sector_0_block = tensor.block(0)
sector_1_block = tensor.block((1,))
ordered_blocks = tensor.blocks()
```

`TensorMap` stores a `HomSpace` and a 1D `VectorStorage`-compatible JAX array.
The storage length must match `get_degeneracystructure(h).total_dim`.
Use `scalar(tensor)` only for scalar TensorMaps with no visible indices.

## Composition

`TensorMap @ TensorMap` composes matching coupled blocks. Tensor0 follows the
`hom(codomain, domain)` convention, so the left domain must equal the right
codomain and the result has the left codomain and right domain.

```python
import jax.numpy as jnp

from tensor0 import TensorMap, U1Irrep, get_degeneracystructure, hom, space


def data_for(h):
    total_dim = get_degeneracystructure(h).total_dim
    return jnp.arange(1, total_dim + 1, dtype=jnp.float32) / 10.0


v = space(U1Irrep, {0: 2, 1: 3})
w = space(U1Irrep, {0: 5, 1: 7})
x = space(U1Irrep, {0: 11, 1: 13})

a_space = hom((v,), (w,))
b_space = hom((w,), (x,))
a = TensorMap(a_space, data_for(a_space))
b = TensorMap(b_space, data_for(b_space))

c = a @ b
assert c.space == hom((v,), (x,))
```

For partial contractions, traces, named tensor networks, and integer-label
networks, see the [contraction guide](contractions.md).

## SVD

`svd_vals(...)` returns singular values as a `SectorVector` over the infimum
bond space. `svd_compact(...)` returns `(u, s, vh)`, where `s` is a
`DiagonalTensorMap`. Use `s.to_tensor_map()` when reconstruction needs the
regular dense block composition path. `svd_full(...)` returns `(u, s, vh)` over
the fused codomain/domain spaces, with `s` as a regular `TensorMap`.

```python
import jax.numpy as jnp

from tensor0 import TensorMap, U1Irrep, get_degeneracystructure, hom, space, svd_compact


def data_for(h):
    total_dim = get_degeneracystructure(h).total_dim
    return jnp.arange(1, total_dim + 1, dtype=jnp.float32) / 10.0


left = space(U1Irrep, {0: 2, 1: 4})
right = space(U1Irrep, {0: 3, 1: 2})
h = hom((left,), (right,))
tensor = TensorMap(h, data_for(h))

u, s, vh = svd_compact(tensor)
reconstructed = u @ s.to_tensor_map() @ vh
```

Use `svd_trunc(...)` with `notrunc()`, `truncrank(...)`, `trunctol(...)`,
`truncspace(...)`, or `truncerror(...)` when a truncation strategy is needed.
`rank(...)` and `cond(...)` are SVD-derived helpers; `rank` counts sector ranks
with quantum-dimension weighting, and `cond` currently supports the 2-norm.

## Transforms

The public transform helpers are `permute(...)`, `braid(...)`, `transpose(...)`,
and `repartition(...)`.

```python
import jax.numpy as jnp

from tensor0 import TensorMap, U1Irrep, get_degeneracystructure, hom, permute, space


def data_for(h):
    total_dim = get_degeneracystructure(h).total_dim
    return jnp.arange(total_dim, dtype=jnp.float32)


v = space(U1Irrep, {0: 2, 1: 1})
w = space(U1Irrep, {0: 1, 1: 2})
x = space(U1Irrep, {1: 1})
h = hom((v, w), (x,))
tensor = TensorMap(h, data_for(h))

result = permute(tensor, ((1,), (0, 2)))
assert result.space == hom((w,), (v.dual(), x))
```

For small correctness checks, compare transform results through public dense
conversion.

## Dense Conversion

`to_dense(...)` and `from_dense(...)` are public APIs intended for
correctness-first small examples and tests.

```python
import jax.numpy as jnp

from tensor0 import TensorMap, U1Irrep, from_dense, get_degeneracystructure, hom, space, to_dense

v = space(U1Irrep, {0: 2, 1: 3})
h = hom((v,), (v,))
total_dim = get_degeneracystructure(h).total_dim
tensor = TensorMap(h, jnp.arange(total_dim, dtype=jnp.float32))

dense = to_dense(tensor)
rebuilt = from_dense(tensor.space, dense)
```

Dense conversion is not a production performance path in the current version.

## JAX jit and grad

`TensorMap` is a JAX pytree. Its storage array is the dynamic leaf, and its
`HomSpace` metadata is static auxiliary data.

```python
import jax
import jax.numpy as jnp

from tensor0 import TensorMap, U1Irrep, get_degeneracystructure, hom, space


def data_for(h):
    total_dim = get_degeneracystructure(h).total_dim
    return jnp.arange(1, total_dim + 1, dtype=jnp.float32) / 10.0


v = space(U1Irrep, {0: 2, 1: 3})
w = space(U1Irrep, {0: 5, 1: 7})
x = space(U1Irrep, {0: 11, 1: 13})

a_space = hom((v,), (w,))
b_space = hom((w,), (x,))
a = TensorMap(a_space, data_for(a_space))
b = TensorMap(b_space, data_for(b_space))


@jax.jit
def compose(left, right):
    return left @ right


jitted = compose(a, b)


def loss(data):
    candidate = TensorMap(a.space, data)
    composed = candidate @ b
    return jnp.sum(composed.storage.data ** 2)


value, gradient = jax.value_and_grad(loss)(a.storage.data)
```

Changing only storage values keeps the same pytree structure. Changing the
`HomSpace` metadata creates a different static JAX specialization.

## Current Boundaries

Tensor0 currently does not include:

- GenericFusion
- anyonic braiding
- arbitrary dynamic product sector families
- full NumPy einsum syntax or automatic contraction-order optimization
- mutable block views or in-place public APIs
- production dense conversion
- native JAX kernels
- performance guarantees or published benchmark reports

This guide only documents the public API behavior available in the current
checkout.
