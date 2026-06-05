use smallvec::smallvec;

use crate::error::{Result, Tensor0Error};

use super::{
    BraidingStyle, EncodedSectorValue, FusionStyle, GroupSpec, Sector, SectorCardinality,
    SectorSpec, SortKey,
};

/// Irrep value for the cyclic group Z_N.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ZNIrrep<const N: usize> {
    value: i64,
}

pub type Z2Irrep = ZNIrrep<2>;
pub type Z3Irrep = ZNIrrep<3>;
pub type Z4Irrep = ZNIrrep<4>;

impl<const N: usize> ZNIrrep<N> {
    pub fn new(value: i64) -> Result<ZNIrrep<N>> {
        let n = modulus::<N>()?;
        Ok(ZNIrrep {
            value: value.rem_euclid(n),
        })
    }
}

impl<const N: usize> Sector for ZNIrrep<N> {
    fn sector_spec() -> SectorSpec {
        SectorSpec::Irrep {
            group: GroupSpec::ZN {
                n: raw_modulus::<N>(),
            },
        }
    }

    fn encoded_width() -> usize {
        1
    }

    fn decode_value(value: &[i64]) -> Result<Self> {
        require_width(value, 1)?;
        ZNIrrep::new(value[0])
    }

    fn encode_value(&self) -> EncodedSectorValue {
        smallvec![self.value]
    }

    fn unit() -> Self {
        ZNIrrep::new(0).expect("ZNIrrep unit requires N > 0")
    }

    fn dual(&self) -> Self {
        ZNIrrep::new(
            self.value
                .checked_neg()
                .expect("ZNIrrep dual value overflowed"),
        )
        .expect("ZNIrrep dual requires N > 0")
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
        Ok(SectorCardinality::Finite(modulus::<N>()? as u128))
    }

    fn fusion_outputs(&self, rhs: &Self) -> Vec<Self> {
        let n = modulus::<N>().expect("ZNIrrep fusion requires N > 0") as i128;
        let value = (self.value as i128 + rhs.value as i128).rem_euclid(n) as i64;
        vec![ZNIrrep::new(value).expect("ZNIrrep fusion output requires N > 0")]
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
        smallvec![self.value as u128]
    }

    fn sort_index(&self) -> Result<u128> {
        Ok(self.value as u128)
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

fn modulus<const N: usize>() -> Result<i64> {
    if N == 0 {
        return Err(Tensor0Error::BadZnModulus { n: 0 });
    }
    i64::try_from(N).map_err(|_| Tensor0Error::SectorIndexOverflow)
}

fn raw_modulus<const N: usize>() -> i64 {
    assert!(N > 0, "ZNIrrep sector spec requires N > 0");
    i64::try_from(N).expect("ZNIrrep sector spec requires N <= i64::MAX")
}
