# Concepts

Tensor0 represents symmetric linear maps as typed metadata plus JAX-backed
storage. The metadata describes sector structure and block layout; the storage
contains the reduced block data.

## Sector Families

Sector families label symmetry charges. `Trivial` represents no symmetry and
has the single empty-tuple sector label `()`. Tensor0 also exposes built-in
families such as `U1Irrep`, `SU2Irrep`, `Z2Irrep`, `Z3Irrep`, `Z4Irrep`,
`FermionParity`, and selected product-sector aliases.

## Spaces and HomSpaces

`space(...)` builds an elementary graded space from sector labels to degeneracy
dimensions. `hom(codomain, domain)` builds the typed linear-map space used by
`TensorMap`. For ordinary no-symmetry spaces, `Vect(dim)` is the canonical
constructor and `ComplexSpace(dim)` is a compatibility spelling.

## TensorMap Storage

`TensorMap` stores a `HomSpace` and a one-dimensional JAX array. The array
length must match the degeneracy structure derived from the `HomSpace`.

For `Trivial`, packed storage is the ordinary dense array flattened in
codomain-then-domain row-major order. `to_dense()` exposes an immutable reshape
of that storage. Other sector families retain symmetry-aware reduced blocks
and reconstruct dense arrays from their fusion metadata.

## Blocks and Transforms

Blocks are accessed by coupled sector. Transform helpers such as `permute(...)`,
`braid(...)`, `transpose(...)`, and `repartition(...)` update the visible index
structure and move reduced block data consistently with the symmetry metadata.

`insertleftunit(...)` and `insertrightunit(...)` add a canonical monoidal-unit
factor at a visible-index boundary, while `removeunit(...)` removes a validated
unit factor. Unit factors have degeneracy one and a unique unit fusion channel,
so these rank-changing operations preserve the current canonical flat layout
and share the same `VectorStorage` without writing to its payload. With normal
JAX Array payloads this is safe because arrays are immutable. At the
codomain/domain boundary, left insertion attaches the unit to the domain and
right insertion attaches it to the codomain.

## Visible Indices and Contractions

Visible axes are ordered as codomain axes followed by domain axes. Contraction
operations therefore specify both output axis order and the output
codomain/domain partition. Contracted spaces must be dual-compatible.

`tensorcontract` and `tensortrace` express this structure with integer axis
references. The `contract` and `ncon` frontends attach labels to the same
visible axes and lower to those primitive operations.

## JAX Integration

`TensorMap` is a JAX pytree. Storage arrays are dynamic leaves, while `HomSpace`
metadata is static auxiliary data. Changing metadata creates a different JAX
specialization.
