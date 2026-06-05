use std::cmp::Ordering;

use smallvec::smallvec;

use crate::error::{Result, Tensor0Error};

use super::{
    BraidingStyle, EncodedSectorValue, FusionStyle, Sector, SectorCardinality, SectorSpec, SortKey,
};

/// U(1) irrep stored as TensorKit-compatible twice-charge.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct U1Irrep {
    charge2: i64,
}

impl U1Irrep {
    /// Construct from twice-charge. For example, `charge2(1)` means charge `1/2`.
    pub fn charge2(charge2: i64) -> Result<U1Irrep> {
        Ok(U1Irrep { charge2 })
    }
}

impl Sector for U1Irrep {
    fn sector_spec() -> SectorSpec {
        SectorSpec::u1()
    }

    fn encoded_width() -> usize {
        1
    }

    fn decode_value(value: &[i64]) -> Result<Self> {
        require_width(value, 1)?;
        U1Irrep::charge2(value[0])
    }

    fn encode_value(&self) -> EncodedSectorValue {
        smallvec![self.charge2]
    }

    fn unit() -> Self {
        U1Irrep { charge2: 0 }
    }

    fn dual(&self) -> Self {
        U1Irrep {
            charge2: self
                .charge2
                .checked_neg()
                .expect("U1Irrep dual twice-charge overflowed"),
        }
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
        Ok(SectorCardinality::Infinite)
    }

    fn fusion_outputs(&self, rhs: &Self) -> Vec<Self> {
        vec![U1Irrep {
            charge2: self
                .charge2
                .checked_add(rhs.charge2)
                .expect("U1Irrep fusion twice-charge overflowed"),
        }]
    }

    fn n_symbol(a: &Self, b: &Self, c: &Self) -> usize {
        usize::from(a.fusion_outputs(b).iter().any(|out| out == c))
    }

    fn r_symbol(a: &Self, b: &Self, c: &Self) -> f64 {
        if Self::n_symbol(a, b, c) == 0 {
            0.0
        } else {
            1.0
        }
    }

    fn sort_key(&self) -> SortKey {
        smallvec![u1_sort_index(self.charge2)]
    }

    fn sort_index(&self) -> Result<u128> {
        Ok(u1_sort_index(self.charge2))
    }
}

impl Ord for U1Irrep {
    fn cmp(&self, other: &Self) -> Ordering {
        self.sort_key().cmp(&other.sort_key())
    }
}

impl PartialOrd for U1Irrep {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn require_width(value: &[i64], expected: usize) -> Result<()> {
    if value.len() == expected {
        Ok(())
    } else {
        Err(Tensor0Error::BadSectorWidth {
            expected,
            actual: value.len(),
        })
    }
}

fn u1_sort_index(charge2: i64) -> u128 {
    if charge2 > 0 {
        (charge2 as u128).saturating_mul(2).saturating_sub(1)
    } else {
        (-(charge2 as i128) as u128).saturating_mul(2)
    }
}
