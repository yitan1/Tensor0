use smallvec::smallvec;

use crate::error::{Result, Tensor0Error};

use super::{
    require_width, BraidingStyle, EncodedSectorValue, FusionStyle, Sector, SectorCardinality,
    SectorSpec, SortKey,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FermionParity {
    value: i64,
}

impl FermionParity {
    pub fn new(value: i64) -> Result<FermionParity> {
        Ok(FermionParity {
            value: value.rem_euclid(2),
        })
    }

    pub fn is_odd(&self) -> bool {
        self.value == 1
    }
}

impl Sector for FermionParity {
    fn sector_spec() -> SectorSpec {
        SectorSpec::fermion_parity()
    }

    fn encoded_width() -> usize {
        1
    }

    fn decode_value(value: &[i64]) -> Result<Self> {
        require_width(value, 1)?;
        FermionParity::new(value[0])
    }

    fn encode_value(&self) -> EncodedSectorValue {
        smallvec![self.value]
    }

    fn unit() -> Self {
        FermionParity { value: 0 }
    }

    fn dual(&self) -> Self {
        *self
    }

    fn quantum_dim(&self) -> usize {
        1
    }

    fn fusion_style() -> FusionStyle {
        FusionStyle::UniqueFusion
    }

    fn braiding_style() -> BraidingStyle {
        BraidingStyle::Fermionic
    }

    fn cardinality() -> Result<SectorCardinality> {
        Ok(SectorCardinality::Finite(2))
    }

    fn fusion_outputs(&self, rhs: &Self) -> Vec<Self> {
        vec![FermionParity {
            value: self.value ^ rhs.value,
        }]
    }

    fn n_symbol(a: &Self, b: &Self, c: &Self) -> usize {
        usize::from(a.fusion_outputs(b).iter().any(|out| out == c))
    }

    fn r_symbol(a: &Self, b: &Self, c: &Self) -> f64 {
        if Self::n_symbol(a, b, c) == 0 {
            return 0.0;
        }
        if a.value == 1 && b.value == 1 && c.value == 0 {
            -1.0
        } else {
            1.0
        }
    }

    fn sort_key(&self) -> SortKey {
        smallvec![self.value as u128]
    }

    fn sort_index(&self) -> Result<u128> {
        Ok(self.value as u128)
    }

    fn value_at(index: u128) -> Result<Self> {
        if index < 2 {
            FermionParity::new(index as i64)
        } else {
            Err(Tensor0Error::SectorIndexOverflow)
        }
    }
}
