# API Overview

This page groups the current public API exported by `tensor0.__all__`. It is a
hand-written overview, not generated API reference.

## Sector Families

- `U1Irrep`
- `SU2Irrep`
- `Z2Irrep`, `Z3Irrep`, `Z4Irrep`
- `FermionParity`
- `FermionNumber`
- `FermionParityU1Irrep`
- `U1SU2Irrep`
- `FermionParitySU2Irrep`
- `FermionParityU1SU2Irrep`

## Spaces and Metadata

- `space`
- `hom`
- `Vect`
- `ElementarySpace`
- `HomSpace`
- `SectorType`
- `FusionTree`
- `SectorDict`
- `SectorStructure`
- `DegeneracyStructure`
- `BlockStructure`
- `SubblockStructure`
- `get_sectorstructure`
- `get_degeneracystructure`

## Tensor Storage and Construction

- `TensorMap`
- `DiagonalTensorMap`
- `SectorVector`
- `VectorStorage`
- `zeros`
- `ones`
- `zero_like`
- `from_blocks`
- `diag`
- `diagm`
- `isdiag`
- `scalar`

## Tensor Operations

- `add`
- `scale`
- `dot`
- `inner`
- `norm`
- `normalize`
- `tr`
- `adjoint`
- `real`
- `imag`
- `complex`

## Transforms

- `permute`
- `braid`
- `transpose`
- `repartition`
- `flip`
- `twist`

## Contractions

- `idx`
- `tensorcontract`
- `tensortrace`
- `contract`
- `ncon`

See the [contraction guide](contractions.md) for axes, labels, output
partitions, order, and scalar-result conventions.

## Factorizations and Truncation

- `svd_vals`
- `svd_compact`
- `svd_full`
- `svd_trunc`
- `rank`
- `cond`
- `notrunc`
- `truncrank`
- `trunctol`
- `truncspace`
- `truncerror`

## Dense Conversion

- `to_dense`
- `from_dense`
