//! Typed sector values and TensorKit-like sector rules.

use std::collections::BTreeSet;

mod fermion_parity;
mod ordering;
mod product;
mod spec;
mod spec_rules;
mod su2;
mod traits;
mod u1;
pub(crate) mod wigner_symbols;
mod zn;

pub use fermion_parity::FermionParity;
pub use product::{
    FermionNumber, FermionParitySU2Irrep, FermionParityU1Irrep, FermionParityU1SU2Irrep,
    ProductSector, U1SU2Irrep,
};
pub use spec::{GroupSpec, SectorSpec};
pub use su2::SU2Irrep;
pub use traits::{
    BraidingStyle, EncodedSectorValue, FusionStyle, Sector, SectorCardinality, SectorTuple, SortKey,
};
pub use u1::U1Irrep;
pub use zn::{Z2Irrep, Z3Irrep, Z4Irrep, ZNIrrep};

pub(crate) fn fusion_sectors<I: Sector>(sectors: &[I]) -> Vec<I> {
    match sectors.len() {
        0 => vec![I::unit()],
        1 => vec![sectors[0].clone()],
        _ => {
            let last = sectors
                .last()
                .expect("multi-factor fusion output has a rightmost sector");
            let mut outputs = BTreeSet::new();
            for prefix in fusion_sectors(&sectors[..sectors.len() - 1]) {
                for output in prefix.fusion_outputs(last) {
                    outputs.insert(output);
                }
            }
            outputs.into_iter().collect()
        }
    }
}
