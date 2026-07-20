use smallvec::smallvec;

use crate::error::{Result, Tensor0Error};

use super::{
    require_width, BraidingStyle, EncodedSectorValue, FusionStyle, Sector, SectorCardinality,
    SectorSpec, SortKey,
};

/// The unique sector value for spaces without a nontrivial symmetry.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Trivial;

impl Sector for Trivial {
    fn sector_spec() -> SectorSpec {
        SectorSpec::trivial()
    }

    fn encoded_width() -> usize {
        0
    }

    fn decode_value(value: &[i64]) -> Result<Self> {
        require_width(value, 0)?;
        Ok(Trivial)
    }

    fn encode_value(&self) -> EncodedSectorValue {
        smallvec![]
    }

    fn unit() -> Self {
        Trivial
    }

    fn dual(&self) -> Self {
        Trivial
    }

    fn quantum_dim(&self) -> usize {
        1
    }

    fn fusion_style() -> FusionStyle {
        FusionStyle::UniqueFusion
    }

    fn braiding_style() -> BraidingStyle {
        BraidingStyle::Bosonic
    }

    fn cardinality() -> Result<SectorCardinality> {
        Ok(SectorCardinality::Finite(1))
    }

    fn fusion_outputs(&self, _rhs: &Self) -> Vec<Self> {
        vec![Trivial]
    }

    fn n_symbol(_a: &Self, _b: &Self, _c: &Self) -> usize {
        1
    }

    fn r_symbol(_a: &Self, _b: &Self, _c: &Self) -> f64 {
        1.0
    }

    fn sort_key(&self) -> SortKey {
        smallvec![]
    }

    fn sort_index(&self) -> Result<u128> {
        Ok(0)
    }

    fn value_at(index: u128) -> Result<Self> {
        if index == 0 {
            Ok(Trivial)
        } else {
            Err(Tensor0Error::SectorIndexOverflow)
        }
    }
}
