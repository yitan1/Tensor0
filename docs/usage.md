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

- `Trivial` for ordinary tensors without symmetry
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

## Ordinary Tensors with Trivial Symmetry

`Trivial` is the no-symmetry sector family. Its only sector label is the empty
tuple `()`. It is a `SectorType` constant, not a callable sector-value class.
These four spellings return the same ordinary `ElementarySpace`:

```python
from tensor0 import ComplexSpace, Trivial, Vect, space

ordinary = Vect(4)

assert ordinary == Vect[Trivial](4)
assert ordinary == space(Trivial, {(): 4})
assert ordinary == ComplexSpace(4)
```

`Vect(dim=0, *, dual=False)` is the canonical concise constructor.
`ComplexSpace(...)` remains a compatibility spelling and returns exactly the
same native space without a separate class. Dimensions are non-boolean,
non-negative integers. Dual, zero, and unit spaces therefore use the same APIs:

```python
from tensor0 import ComplexSpace, Trivial, Vect, unit_space, zero_space

dual = Vect(4, dual=True)
assert dual == Vect(4).dual()
assert Vect() == zero_space(Trivial)
assert Vect(1) == unit_space(Trivial)
assert ComplexSpace(4) == Vect(4)
```

A rank-zero `HomSpace` has no visible factor from which to infer its sector
family, so pass `sector_type=Trivial` explicitly:

```python
import jax.numpy as jnp

from tensor0 import Trivial, from_dense, hom, scalar

rank_zero = hom((), (), sector_type=Trivial)
value = from_dense(rank_zero, jnp.asarray(2.0))
assert float(scalar(value)) == 2.0
```

Every correctly shaped dense array is valid for a Trivial `HomSpace`.
`from_dense()` accepts either the full visible shape or its codomain-by-domain
matrix shape. The sole coupled block uses the sector key `()` and has the
matrix shape:

```python
import jax.numpy as jnp

from tensor0 import Vect, from_dense, hom, to_dense

target = hom((Vect(2), Vect(3)), (Vect(4),))
dense = jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4)
tensor = from_dense(target, dense)

assert tensor.block(()).shape == (6, 4)
assert bool(jnp.array_equal(tensor.block(()), dense.reshape(6, 4)))
assert bool(jnp.array_equal(to_dense(tensor), dense))
```

Trivial tensors use the same composition, contraction, trace, transform, and
factorization APIs as symmetric tensors. For example:

```python
import jax.numpy as jnp

from tensor0 import (
    Vect,
    from_dense,
    hom,
    scalar,
    svd_compact,
    tensorcontract,
    tensortrace,
    to_dense,
)

left_dense = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)
right_dense = jnp.arange(12, dtype=jnp.float32).reshape(3, 4)
left = from_dense(hom((Vect(2),), (Vect(3),)), left_dense)
right = from_dense(hom((Vect(3),), (Vect(4),)), right_dense)

contracted = tensorcontract(
    left,
    right,
    axes=((1,), (0,)),
    output=(((0, 0),), ((1, 1),)),
)
assert bool(jnp.allclose(to_dense(contracted), left_dense @ right_dense))

square_dense = jnp.arange(9, dtype=jnp.float32).reshape(3, 3)
square = from_dense(
    hom((Vect(3),), (Vect(3),)),
    square_dense,
)
traced = tensortrace(square, axes=((0,), (1,)), output=((), ()))
assert bool(jnp.allclose(scalar(traced), jnp.trace(square_dense)))

u, s, vh = svd_compact(left)
assert bool(jnp.allclose(to_dense(u @ s @ vh), left_dense, atol=1e-5))
```

For Trivial tensors, `to_dense()` is a direct immutable JAX reshape of packed
storage. `from_dense()` likewise uses a direct reshape when the input already
has the target storage dtype; required dtype promotion may materialize a new
array. Neither API promises physical buffer aliasing or a mutable/write-through
view. Nontrivial sector families continue to use symmetry-aware projection and
reconstruction, including `tol` validation in `from_dense()`; they do not use
Trivial's dense numerical fast paths.

Tensor0 does not currently support product-sector families such as
`Trivial @ U1Irrep`, `PlanarTrivial`, real or Cartesian space families,
mutable dense/block views, or generic dense execution for nontrivial
symmetries. Use `tensor.block(())` for the sole Trivial block; `tensor[()]` is
not a dense-array access spelling.

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

## Inverse, Pseudoinverse, and Direct Solves

`inverse(tensor)` and `tensor.inverse()` invert every square reduced block and
return the reversed `HomSpace`. `pseudoinverse(tensor, *, atol=0.0, rtol=None)`
and its method form also support rectangular maps and apply one global singular
value cutoff. Both functional forms accept ordinary and diagonal tensor maps.

Use direct solves when the goal is an equation rather than an inverse:

```python
from tensor0 import inverse, left_solve, right_solve

operator_inverse = inverse(operator)
x_left = left_solve(operator, rhs)   # operator @ x_left == rhs
x_right = right_solve(lhs, operator) # x_right @ operator == lhs
```

The operator in either solve must be a blockwise square isomorphism. Both
solves call JAX's linear solver directly on each reduced block; they do not
form an explicit inverse. Singularity and nonsingularity are numerical
preconditions, so traced execution follows JAX numerical behavior rather than
raising data-dependent Python exceptions.

Pseudoinverse rejects singular values at or below
`max(atol, rtol * max(singular_values))`. If `rtol` is omitted, it is ten times
the largest reduced block dimension times machine epsilon for the promoted
real dtype. This policy is shared by ordinary and diagonal tensor maps.

## Morphism Constructors

`identity(space, *, dtype=None)` constructs the identity on an elementary or
product space. `isomorphism(codomain, domain, *, dtype=None)` and
`unitary(codomain, domain, *, dtype=None)` require isomorphic spaces, while
`isometry(codomain, domain, *, dtype=None)` requires the domain to embed into
the codomain sector by sector.

```python
import jax.numpy as jnp

from tensor0 import (
    U1Irrep,
    allclose,
    identity,
    isometry,
    isomorphism,
    space,
    unitary,
)

small = space(U1Irrep, {0: 2, 1: 1})
large = space(U1Irrep, {0: 3, 1: 2})

unit = identity(small, dtype=jnp.float32)
isomorphic = isomorphism(small, small, dtype=jnp.float32)
basis_change = unitary(small, small, dtype=jnp.float32)
embedding = isometry(large, small, dtype=jnp.float32)

assert bool(allclose(isomorphic, unit, rtol=1e-5, atol=1e-6))
assert bool(allclose(basis_change, unit, rtol=1e-5, atol=1e-6))
assert bool(
    allclose(
        embedding.adjoint() @ embedding,
        unit,
        rtol=1e-5,
        atol=1e-6,
    )
)
```

Product-to-fused isomorphisms and unitaries use Tensor0's existing fusion-tree
gauge. Invalid non-isomorphic or non-monomorphic inputs fail before numerical
allocation. With `dtype=None`, allocation follows the configured JAX default;
explicit real and complex dtypes are preserved. `identity` is the only public
identity name—there is no `id` alias or constructor classmethod.

For the spaces currently supported by Tensor0, `isomorphism` and `unitary` use
the same deterministic gauge-fixed construction. The `unitary` name records
the stronger contract that reversing the map agrees with both its inverse and
its adjoint; it does not select a different numerical construction.

## Tensor Product

`tensor_product(left, right)` is the explicit disconnected tensor-network
product. Its codomain factors are ordered as left then right, followed by the
left-then-right domain factors in the domain partition:

```python
from tensor0 import U1Irrep, identity, space, tensor_product

left_space = space(U1Irrep, {0: 2})
right_space = space(U1Irrep, {0: 3})
left = identity(left_space)
right = identity(right_space)

product = tensor_product(left, right)
assert product.codomain.spaces == (left_space, right_space)
assert product.domain.spaces == (left_space, right_space)
```

Both inputs must use the same sector family. The implementation uses the
tested disconnected-contraction semantics, including fusion coefficients,
fermionic input order, scalar operands, JIT, and differentiation. Tensor0 does
not attach this operation to an overloaded tensor-product operator.

## Explicit-Key Random Construction

`random_normal(key, space, *, dtype=None)` samples the packed reduced storage
of a `HomSpace` directly. `random_isometry(key, codomain, domain, *,
dtype=None)` samples and blockwise orthogonalizes an isometric embedding while
preserving the exact requested product-space partitions. Tensor0 passes the supplied key to exactly one
`jax.random.normal` call without pre-splitting or mutating it, returning a
replacement key, or reading global RNG state. JAX may internally split the key
when implementing complex sampling:

```python
import jax
import jax.numpy as jnp

from tensor0 import U1Irrep, equal, hom, is_isometric, random_isometry, random_normal, space

factor = space(U1Irrep, {0: 2, 1: 1})
target = hom((factor,), (factor,))
key = jax.random.key(0)

sample = random_normal(key, target, dtype=jnp.complex64)
repeated = random_normal(key, target, dtype=jnp.complex64)
assert bool(equal(sample, repeated))

large = space(U1Irrep, {0: 3, 1: 2})
embedding = random_isometry(key, large, factor, dtype=jnp.float32)
assert embedding.codomain.spaces == (large,)
assert embedding.domain.spaces == (factor,)
assert bool(is_isometric(embedding, rtol=1e-5, atol=1e-6))
```

`dtype=None` follows `jax.random.normal`'s configured real default. An explicit
complex dtype follows JAX's complex-normal semantics, and an empty `HomSpace`
produces empty storage of the requested dtype. Capture the space and dtype in a
jitted wrapper, for example `jax.jit(lambda key: random_normal(key, target))`,
so the spaces remain static metadata. There are no `randn` or `randisometry`
aliases, classmethods, implicit keys, or global generators.

## Compact QR/LQ and Orthogonalization

`qr_compact(tensor)` returns `(q, r)` with `q @ r == tensor`, while
`lq_compact(tensor)` returns `(l, q)` with `l @ q == tensor`. Both use the
deterministic non-dual connecting space
`infimum(fuse(tensor.codomain), fuse(tensor.domain))`. For QR this space is
`q.domain == r.codomain`; for LQ it is `l.domain == q.codomain`.

```python
import jax.numpy as jnp

from tensor0 import U1Irrep, allclose, from_blocks, hom, lq_compact, qr_compact, space

factor = space(U1Irrep, {0: 2})
target = hom((factor,), (factor,))
tensor = from_blocks(
    target,
    {0: jnp.asarray([[2.0, 1.0], [1.0, 3.0]], dtype=jnp.float32)},
)

left_q, r = qr_compact(tensor)
l, right_q = lq_compact(tensor)

assert left_q.domain == r.codomain
assert l.domain == right_q.codomain
assert bool(allclose(left_q @ r, tensor, rtol=1e-5, atol=1e-6))
assert bool(allclose(l @ right_q, tensor, rtol=1e-5, atol=1e-6))
```

Each nonzero diagonal entry of `R` or `L` is normalized to a positive real
value by moving its sign or complex phase into the corresponding column or row
of `Q`. A zero diagonal entry uses the neutral phase `1`, preserving
reconstruction and static compact shapes.

`left_orth(tensor, *, alg=None, trunc=None)` and
`right_orth(tensor, *, alg=None, trunc=None)` are the tensor-network-oriented
interfaces. Without optional arguments they use `qr_compact` and `lq_compact`,
respectively. With `alg="svd"`, the compact SVD is grouped as `(u, s @ vh)` for
left orthogonalization or `(u @ s, vh)` for right orthogonalization.

Supplying a truncation strategy selects the SVD path automatically and reduces
the connecting space. An explicit `alg="qr"` or `alg="lq"` cannot be combined
with `trunc`. These high-level functions return only the two grouped factors;
call `svd_trunc` directly when the discarded-weight error is required. Polar
orthogonalization and backend algorithm objects are not supported.

Factor dtypes follow JAX linear-algebra promotion. In particular, integer or
boolean storage produces inexact QR/LQ factors rather than being cast back to
the input dtype; typed empty outputs use the same promotion rule.

Rank-deficient and zero blocks support eager and JIT value computation, but
their output shapes do not shrink with numerical rank. Differentiation is
promised only for appropriate full-rank blocks with nonzero normalized
diagonals; Tensor0 does not promise QR/LQ derivatives at rank deficiency.

## Hermitian Eigenvalues and Eigenvectors

`is_hermitian(tensor, *, atol=0.0, rtol=0.0)` returns a scalar JAX boolean
array. Its default check is exact; nonzero tolerances are always explicit.
`eigh_vals(tensor)` returns a real `SectorVector`, and `eigh_full(tensor)`
returns `(d, v)` with eigenvalues ordered within each sector and reconstruction
`tensor == v @ d @ v.adjoint()`. `eigh_trunc(tensor, *, trunc=None)` returns the
same `(d, v)` shape after selecting by absolute eigenvalue magnitude while
preserving eigenvalue signs and JAX's ascending within-sector order.

```python
from tensor0 import allclose, eigh_full, eigh_trunc, eigh_vals, is_hermitian, truncrank

assert bool(is_hermitian(tensor, rtol=1e-5, atol=1e-6))
values = eigh_vals(tensor)
d, v = eigh_full(tensor)
reconstructed = v @ d @ v.adjoint()
assert bool(allclose(reconstructed, tensor, rtol=1e-5, atol=1e-6))

truncated_d, truncated_v = eigh_trunc(tensor, trunc=truncrank(1))
projected = truncated_v @ truncated_d @ truncated_v.adjoint()
```

All three functions first require the static `TensorMap` space to be an
endomorphism with equal codomain and domain. Hermiticity itself is a numerical
precondition: `eigh_vals` and `eigh_full` do not convert a predicate to a Python
boolean, raise on data inside JIT, project, or symmetrize a non-Hermitian input.
Eager callers that need validation should call
`bool(is_hermitian(tensor, ...))` before decomposition; jitted callers can keep
the predicate as a JAX scalar.

`eigh_trunc` accepts the same rank, tolerance, error, space, and combined
truncation strategies as `svd_trunc`, but intentionally does not return a
discarded-weight error because signed Hermitian spectra need a separate error
contract. Nontrivial strategies select value-dependent output shapes and must
therefore run eagerly. The no-truncation path has fixed shapes and supports
whole-function JIT; results selected eagerly remain ordinary JAX-compatible
tensor maps for downstream jitted computation.

Eigenvalue and eigenvector dtypes also follow JAX promotion: integer or boolean
storage produces inexact eigenvectors and real inexact eigenvalues, including
for typed empty outputs.

At degenerate spectra, eigenvalues and reconstruction remain supported in
eager and JIT execution, but the eigenvector basis is only as deterministic as
the JAX backend. Eigenvector derivatives and a unique basis are not promised
at degeneracy; gradient guarantees are restricted to nondegenerate spectra.

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
the shared ordinary/diagonal reduced-block cutoff described above and maps
values at or below the cutoff to zero; both tolerances are non-negative.

Use `svd_trunc(...)` with `notrunc()`, `truncrank(...)`, `trunctol(...)`,
`truncspace(...)`, or `truncerror(...)` when a truncation strategy is needed.
`rank(...)` and `cond(...)` are SVD-derived helpers; `rank` counts sector ranks
with quantum-dimension weighting, and `cond` currently supports the 2-norm.

## Predicates and Comparison

`TensorMap.dtype` and `DiagonalTensorMap.dtype` report the storage dtype.
`is_diagonal(...)`, `is_isometric(...)`, `is_unitary(...)`,
`is_positive_definite(...)`, `equal(...)`, and `allclose(...)` return scalar
JAX boolean arrays, so eager code may call `bool(...)` while jitted code keeps
the result as an array. Exact comparison requires equal spaces, dtypes, and values.
Approximate comparison requires equal spaces, uses JAX numerical promotion, and
requires both tolerances as keyword arguments:

```python
from tensor0 import allclose, equal, is_diagonal, is_positive_definite, is_unitary

diagonal_predicate = is_diagonal(s)
same_values = allclose(reconstructed, tensor, rtol=1e-5, atol=1e-6)
exact_copy = equal(tensor, tensor.copy())
unitary_predicate = is_unitary(identity(s.index_space))
positive_predicate = is_positive_definite(identity(s.index_space))
```

Mixed ordinary/diagonal comparison uses mathematical tensor values. Neither
comparison contract is attached to `TensorMap.__eq__`.

`is_isometric(tensor, side="left")` checks `tensor.adjoint() @ tensor` against
the domain identity. With `side="right"`, it checks
`tensor @ tensor.adjoint()` against the codomain identity. The side is static
configuration in jitted code, so capture it in a wrapper such as
`jax.jit(lambda value: is_isometric(value, side="right"))`. A unitary checks
both identities and first requires statically isomorphic spaces. A statically
impossible relation returns scalar false without reading numerical storage.
Positive definiteness requires an equal-space endomorphism and checks both
Hermiticity and that every eigenvalue exceeds
`max(atol, rtol * max(abs(eigenvalues)))`. Empty isometries, unitaries, and
positive-definite endomorphisms use the explicit vacuous truth value `True`.

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

`to_dense(...)` and `from_dense(...)` are public APIs. Trivial tensors use the
direct immutable array path described above. Other sector families use these
APIs as correctness-first projection and reconstruction utilities for small
examples and tests.

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

Dense conversion does not enable dense numerical execution for nontrivial
sector families.

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
- generic dense numerical execution for nontrivial sector families
- native JAX kernels
- performance guarantees or published benchmark reports

This guide only documents the public API behavior available in the current
checkout.
