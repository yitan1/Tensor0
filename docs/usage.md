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
space used by `TensorMap`; its two sides are public `ProductSpace` values.

```python
import jax.numpy as jnp

from tensor0 import (
    TensorMap,
    U1Irrep,
    dim,
    hom,
    reduced_dim,
    space,
    storage_dim,
)

v = space(U1Irrep, {0: 2, 1: 3})
h = hom((v,), (v,))

assert dim(v) == 5
assert reduced_dim(v) == 5
assert dim(h.codomain) == 5

total_dim = storage_dim(h)
tensor = TensorMap(h, jnp.arange(total_dim, dtype=jnp.float32))

sector_0_block = tensor.block(0)
sector_1_block = tensor.block((1,))
ordered_blocks = tensor.blocks()

subblocks = tensor.subblocks()
(row_tree, col_tree), reduced = next(iter(subblocks))
assert tensor[row_tree, col_tree].shape == reduced.shape

# U1 has UniqueFusion. Domain indices use visible (dual) sector labels.
charge_one = tensor[1, -1]
```

Physical `dim`, elementary `reduced_dim`, and packed `storage_dim` are distinct:
the first includes quantum dimensions, the second sums degeneracy dimensions,
and the third is the required 1D storage length for a `HomSpace`. `TensorMap`
stores that `HomSpace` and a `VectorStorage`-compatible JAX array. Advanced
layout records and accessors, including `get_degeneracystructure`, remain public
from `tensor0.structure` rather than the package root.

A rank-zero map has no factor from which to infer its sector family, so it must
be typed explicitly:

```python
scalar_space = hom((), (), sector_type=U1Irrep)
scalar_tensor = TensorMap(scalar_space, jnp.asarray([2.0]))
```

The public space algebra is `fuse`, `unit_space`, `zero_space`, `infimum`,
`supremum`, `direct_sum`, `is_isomorphic`, `is_monomorphic`, and
`is_epimorphic`. Binary algebra requires matching sector families and dual
flags. Empty `ProductSpace` values retain their sector family in equality,
hashing, caching, and JAX static metadata; use `.spaces` when a tuple of factors
is needed.

Use `scalar(tensor)` only for scalar TensorMaps with no visible indices.
`tensor.subblocks()` is a reusable lazy view in canonical fusion-tree order;
each iteration creates and reads one subblock at a time. The view supports
`len(view)`, integer and negative indexing, and slices; a slice returns an
ordinary tuple. Use either
`tensor[row_tree, col_tree]` / `tensor.subblock(row_tree, col_tree)` or a tuple
of visible sectors for `UniqueFusion` families, such as `tensor[1, -1]` /
`tensor.subblock((1, -1))`. A tuple-valued product sector on a rank-one tensor
needs an outer tuple, for example `tensor[((0, 0),)]`. Returned JAX arrays are
immutable values rather than write-through views.

## Composition

`TensorMap @ TensorMap` composes matching coupled blocks. Tensor0 follows the
`hom(codomain, domain)` convention, so the left domain must equal the right
codomain and the result has the left codomain and right domain.

```python
import jax.numpy as jnp

from tensor0 import TensorMap, U1Irrep, hom, space
from tensor0.structure import get_degeneracystructure


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
`DiagonalTensorMap` and composes directly with ordinary tensor maps.
`svd_full(...)` returns `(u, s, vh)` over the fused codomain/domain spaces, with
`s` as a regular `TensorMap`.

```python
import jax.numpy as jnp

from tensor0 import TensorMap, U1Irrep, hom, space, svd_compact
from tensor0.structure import get_degeneracystructure


def data_for(h):
    total_dim = get_degeneracystructure(h).total_dim
    return jnp.arange(1, total_dim + 1, dtype=jnp.float32) / 10.0


left = space(U1Irrep, {0: 2, 1: 4})
right = space(U1Irrep, {0: 3, 1: 2})
h = hom((left,), (right,))
tensor = TensorMap(h, data_for(h))

u, s, vh = svd_compact(tensor)
reconstructed = u @ s @ vh

assert s.domain.spaces == (s.index_space,)
assert s.codomain.spaces == (s.index_space,)
```

`DiagonalTensorMap` exposes the shared tensor metadata and block interface,
diagonal-preserving arithmetic, adjoint, norm, inverse, and cutoff
pseudoinverse. `to_tensor_map()` remains available for interoperability, but is
not required for ordinary composition or SVD reconstruction. Pseudoinverse uses
`max(atol, rtol * max(abs(values)))` and maps values at or below the cutoff to
zero; both tolerances are non-negative. When omitted, `rtol` is
`10 * max(shape) * eps` for the promoted real dtype.

Use `svd_trunc(...)` with `notrunc()`, `truncrank(...)`, `trunctol(...)`,
`truncspace(...)`, or `truncerror(...)` when a truncation strategy is needed.
`rank(...)` and `cond(...)` are SVD-derived helpers; `rank` counts sector ranks
with quantum-dimension weighting, and `cond` currently supports the 2-norm.

## Predicates and Comparison

`TensorMap.dtype` and `DiagonalTensorMap.dtype` report the storage dtype.
`isdiag(...)`, `equal(...)`, and `allclose(...)` return scalar JAX boolean
arrays, so eager code may call `bool(...)` while jitted code keeps the result as
an array. Exact comparison requires equal spaces, dtypes, and values.
Approximate comparison requires equal spaces, uses JAX numerical promotion, and
requires both tolerances as keyword arguments:

```python
from tensor0 import allclose, equal, isdiag

diagonal_predicate = isdiag(s)
same_values = allclose(reconstructed, tensor, rtol=1e-5, atol=1e-6)
exact_copy = equal(tensor, tensor.copy())
```

Mixed ordinary/diagonal comparison uses mathematical tensor values. Neither
comparison contract is attached to `TensorMap.__eq__`.

## Transforms

The public transform helpers are `permute(...)`, `braid(...)`, `transpose(...)`,
`repartition(...)`, `flip(...)`, `twist(...)`, `insertleftunit(...)`,
`insertrightunit(...)`, and `removeunit(...)`. The corresponding `TensorMap`
methods have the same behavior and error contracts.

```python
import jax.numpy as jnp

from tensor0 import TensorMap, U1Irrep, hom, space, storage_dim


def data_for(h):
    total_dim = storage_dim(h)
    return jnp.arange(total_dim, dtype=jnp.float32)


v = space(U1Irrep, {0: 2, 1: 1})
w = space(U1Irrep, {0: 1, 1: 2})
x = space(U1Irrep, {1: 1})
h = hom((v, w), (x,))
tensor = TensorMap(h, data_for(h))

result = tensor.permute(((1,), (0, 2)))
assert result.space == hom((w,), (v.dual(), x))
```

`flip(tensor, indices, inv=False)` changes the duality presentation of the
selected 0-based visible indices without moving them across the
codomain/domain partition. It preserves visible sector multiplicities and
applies the fusion-tree Z-isomorphism coefficient:

```python
from tensor0 import flip

flipped = flip(tensor, (0, 2))
restored = flip(flipped, (0, 2), inv=True)

assert restored.space == tensor.space
```

`flip` is distinct from elementary-space `dual()` and is not generally
involutory: two forward flips can introduce a phase. Use one forward and one
inverse flip to restore the original tensor. Flipping both sides of a matching
contraction leg preserves the contraction result.

### Unit-Space Insertion and Removal

The unit-space operations have these signatures:

```python
insertleftunit(tensor, position=None, *, dual=False)
insertrightunit(tensor, position=None, *, dual=False)
removeunit(tensor, index)
```

`position` is a 0-based boundary in `0..tensor.numind`: it is the number of
original visible indices before the new unit, and the new unit appears at
visible index `position`. The default is `tensor.numind`, the final boundary.
For `M = tensor.numout`, the two insertion operations differ only at the
codomain/domain boundary:

| Boundary | `insertleftunit` attaches to | `insertrightunit` attaches to |
|---|---|---|
| `0 <= position < M` | codomain at `position` | codomain at `position` |
| `position == M` | first domain factor | final codomain factor |
| `M < position <= tensor.numind` | domain at `position - M` | domain at `position - M` |

Consequently, on a rank-zero TensorMap, left insertion creates a domain factor
and right insertion creates a codomain factor. Removing the newly inserted
visible index is an exact structural inverse:

```python
from tensor0 import insertleftunit, insertrightunit, removeunit

boundary = tensor.numout
left = insertleftunit(tensor, boundary)
right = insertrightunit(tensor, boundary)

assert left.numout == tensor.numout
assert left.numin == tensor.numin + 1
assert right.numout == tensor.numout + 1
assert right.numin == tensor.numin
assert removeunit(left, boundary).space == tensor.space
assert removeunit(right, boundary).space == tensor.space
```

`dual` selects the canonical unit or its dual as the factor attached to the
destination codomain or domain `ProductSpace`. A visible domain leg is, by the
`HomSpace` convention, the dual of its stored domain factor. Its visible
duality presentation is therefore opposite to the value of `dual`; codomain
legs present the value directly.

For all currently supported sector families, unit insertion preserves the
canonical flattened layout. The returned TensorMap therefore shares the
source's `VectorStorage` without writing to its payload, preserves dtype, and
does not perform a dense conversion or data remap. This sharing is safe for the
normal immutable JAX Array payload:

```python
assert left.storage is tensor.storage
assert right.storage is tensor.storage
assert removeunit(left, boundary).storage is tensor.storage
```

`position`, `index`, and `dual` are structural metadata and must be static
under `jax.jit`. Capture them in a closure, as below, or mark the corresponding
arguments static with `static_argnums` or `static_argnames`:

```python
import jax

insert_at_boundary = jax.jit(
    lambda value: insertleftunit(value, boundary, dual=True)
)
compiled_left = insert_at_boundary(tensor)
```

`removeunit` accepts a canonical unit factor and its dual, but rejects a
one-dimensional factor carrying a non-unit sector, a higher-multiplicity unit,
or a factor with any additional nonzero sector. Insertion requires a
`TensorMap`, an integer boundary or `None`, and a boolean `dual`; removal
requires a `TensorMap` and an integer index. Booleans are not accepted as
indices. Invalid types raise `TypeError`, while an out-of-range boundary or
index and removal of a non-unit factor raise `ValueError`.

Tensor0 supports simple monoidal units for its current built-in sector
families. These operations do not expose TensorKit's mutable `copy` option or
its elementary-space `conj` option, and they are not a general singleton-axis,
reshape, or squeeze API.

For small correctness checks, compare transform results through public dense
conversion.

## Dense Conversion

`to_dense(...)` and `from_dense(...)` are public APIs intended for
correctness-first small examples and tests.

```python
import jax.numpy as jnp

from tensor0 import TensorMap, U1Irrep, from_dense, hom, space, storage_dim, to_dense

v = space(U1Irrep, {0: 2, 1: 3})
h = hom((v,), (v,))
total_dim = storage_dim(h)
tensor = TensorMap(h, jnp.arange(total_dim, dtype=jnp.float32))

dense = to_dense(tensor)
rebuilt = from_dense(tensor.space, dense)
```

Dense conversion is not a production performance path in the current version.

## JAX jit and grad

`TensorMap` and `DiagonalTensorMap` are JAX pytrees. Their storage arrays are the
dynamic leaves, while space metadata is static auxiliary data.

```python
import jax
import jax.numpy as jnp

from tensor0 import TensorMap, U1Irrep, hom, space, storage_dim


def data_for(h):
    total_dim = storage_dim(h)
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
