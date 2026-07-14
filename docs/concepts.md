# Concepts

Tensor0 represents symmetric linear maps as typed metadata plus JAX-backed
storage. The metadata describes sector structure and block layout; the storage
contains the reduced block data.

## Sector Families

Sector families label symmetry charges. Tensor0 currently exposes built-in
families such as `U1Irrep`, `SU2Irrep`, `Z2Irrep`, `Z3Irrep`, `Z4Irrep`,
`FermionParity`, and selected product-sector aliases.

## Spaces and HomSpaces

`space(...)` builds an elementary graded space from sector labels to degeneracy
dimensions. `hom(codomain, domain)` builds the typed linear-map space used by
`TensorMap`.

## TensorMap Storage

`TensorMap` stores a `HomSpace` and a one-dimensional JAX array. The array
length must match the degeneracy structure derived from the `HomSpace`.

## Blocks and Transforms

Blocks are accessed by coupled sector. Transform helpers such as `permute(...)`,
`braid(...)`, `transpose(...)`, and `repartition(...)` update the visible index
structure and move reduced block data consistently with the symmetry metadata.

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
