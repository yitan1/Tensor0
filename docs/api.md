# API Overview

This page is the authoritative inventory for the public names exported by
`tensor0.__all__` and `tensor0.structure.__all__`. Tests check both inventories
against these lists.

## Root API

<!-- tensor0-root-api:start -->

### Sector Families

- `U1Irrep`
- `SU2Irrep`
- `Z2Irrep`
- `Z3Irrep`
- `Z4Irrep`
- `FermionParity`
- `FermionNumber`
- `FermionParityU1Irrep`
- `U1SU2Irrep`
- `FermionParitySU2Irrep`
- `FermionParityU1SU2Irrep`

### Spaces and Public Values

- `space`
- `hom`
- `Vect`
- `ElementarySpace`
- `ProductSpace`
- `HomSpace`
- `SectorType`
- `FusionTree`
- `SectorDict`
- `dim`
- `reduced_dim`
- `storage_dim`
- `fuse`
- `unit_space`
- `zero_space`
- `infimum`
- `supremum`
- `direct_sum`
- `is_isomorphic`
- `is_monomorphic`
- `is_epimorphic`

### Tensor Storage and Construction

- `TensorMap`
- `DiagonalTensorMap`
- `SectorVector`
- `VectorStorage`
- `zeros`
- `ones`
- `zero_like`
- `from_blocks`
- `identity`
- `isomorphism`
- `unitary`
- `isometry`
- `random_normal`
- `diag`
- `diagm`
- `is_diagonal`
- `scalar`
- `to_dense`
- `from_dense`

### Tensor Operations and Comparison

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
- `equal`
- `allclose`
- `tensor_product`

### Transforms

- `permute`
- `braid`
- `transpose`
- `repartition`
- `flip`
- `twist`
- `insertleftunit`
- `insertrightunit`
- `removeunit`

### Contractions

- `idx`
- `tensorcontract`
- `tensortrace`
- `contract`
- `ncon`

### Factorizations and Truncation

- `qr_compact`
- `lq_compact`
- `left_orth`
- `right_orth`
- `eigh_vals`
- `eigh_full`
- `is_hermitian`
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

<!-- tensor0-root-api:end -->

See the [usage guide](usage.md) for construction, comparison, transforms, and
factorization examples. See the [contraction guide](contractions.md) for axes,
labels, output partitions, order, and scalar-result conventions.

## Structure API

These names are public from `tensor0.structure`. The layout records and accessors
are intentionally not re-exported from the package root.

<!-- tensor0-structure-api:start -->

- `BlockStructure`
- `DegeneracyStructure`
- `SectorDict`
- `SectorStructure`
- `SectorType`
- `SubblockStructure`
- `Vect`
- `dim`
- `direct_sum`
- `fuse`
- `get_blockstructure`
- `get_degeneracystructure`
- `get_sectorstructure`
- `hom`
- `infimum`
- `is_epimorphic`
- `is_isomorphic`
- `is_monomorphic`
- `reduced_dim`
- `space`
- `storage_dim`
- `supremum`
- `unit_space`
- `zero_space`

<!-- tensor0-structure-api:end -->

For code written against the earlier alpha surface, replace imports such as
`from tensor0 import get_degeneracystructure` with
`from tensor0.structure import get_degeneracystructure`. This is an immediate
pre-v1 cleanup; no deprecated root aliases are retained.
