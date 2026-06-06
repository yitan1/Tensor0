use std::cmp::Ordering;

use ndarray::Array4;
use smallvec::smallvec;

use crate::error::{Result, Tensor0Error};

use super::wigner_symbols::{
    su2_clebsch_gordan_spin2, su2_f_symbol_spin2, validate_triangle_spin2,
};
use super::{
    require_width, BraidingStyle, EncodedSectorValue, FusionStyle, Sector, SectorCardinality,
    SectorSpec, SortKey,
};

/// SU(2) irrep stored as twice-spin.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SU2Irrep {
    spin2: i64,
}

impl SU2Irrep {
    /// Construct from twice-spin. For example, `spin2(1)` means spin `1/2`.
    pub fn spin2(spin2: i64) -> Result<SU2Irrep> {
        if spin2 < 0 {
            return Err(Tensor0Error::Message(
                "SU2 spin must be non-negative".to_string(),
            ));
        }
        Ok(SU2Irrep { spin2 })
    }

    pub fn spin2_value(&self) -> i64 {
        self.spin2
    }
}

impl Sector for SU2Irrep {
    fn sector_spec() -> SectorSpec {
        SectorSpec::su2()
    }

    fn encoded_width() -> usize {
        1
    }

    fn decode_value(value: &[i64]) -> Result<Self> {
        require_width(value, 1)?;
        SU2Irrep::spin2(value[0])
    }

    fn encode_value(&self) -> EncodedSectorValue {
        smallvec![self.spin2]
    }

    fn unit() -> Self {
        SU2Irrep { spin2: 0 }
    }

    fn dual(&self) -> Self {
        *self
    }

    fn quantum_dim(&self) -> usize {
        self.spin2 as usize + 1
    }

    fn fusion_style() -> FusionStyle {
        FusionStyle::SimpleFusion
    }

    fn braiding_style() -> BraidingStyle {
        BraidingStyle::Bosonic
    }

    fn cardinality() -> Result<SectorCardinality> {
        Ok(SectorCardinality::Infinite)
    }

    fn fusion_outputs(&self, rhs: &Self) -> Vec<Self> {
        let min = (self.spin2 - rhs.spin2).abs();
        let max = self.spin2 + rhs.spin2;
        (min..=max)
            .step_by(2)
            .map(|spin2| SU2Irrep { spin2 })
            .collect()
    }

    fn n_symbol(a: &Self, b: &Self, c: &Self) -> usize {
        usize::from(a.fusion_outputs(b).iter().any(|out| out == c))
    }

    fn f_symbol(a: &Self, b: &Self, c: &Self, d: &Self, e: &Self, f: &Self) -> Result<f64> {
        su2_f_symbol(*a, *b, *c, *d, *e, *f)
    }

    fn r_symbol(a: &Self, b: &Self, c: &Self) -> f64 {
        if Self::n_symbol(a, b, c) == 0 {
            return 0.0;
        }
        let exponent = (a.spin2 + b.spin2 - c.spin2) / 2;
        if exponent % 2 == 0 {
            1.0
        } else {
            -1.0
        }
    }

    fn fusion_tensor(a: &Self, b: &Self, c: &Self) -> Result<Array4<f64>> {
        su2_fusion_tensor(*a, *b, *c)
    }

    fn sort_key(&self) -> SortKey {
        smallvec![self.spin2 as u128]
    }

    fn sort_index(&self) -> Result<u128> {
        Ok(self.spin2 as u128)
    }

    fn value_at(index: u128) -> Result<Self> {
        let spin2 = i64::try_from(index).map_err(|_| Tensor0Error::SectorIndexOverflow)?;
        SU2Irrep::spin2(spin2)
    }
}

impl Ord for SU2Irrep {
    fn cmp(&self, other: &Self) -> Ordering {
        self.sort_key().cmp(&other.sort_key())
    }
}

impl PartialOrd for SU2Irrep {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn su2_f_symbol(
    s1: SU2Irrep,
    s2: SU2Irrep,
    s3: SU2Irrep,
    s4: SU2Irrep,
    s5: SU2Irrep,
    s6: SU2Irrep,
) -> Result<f64> {
    if SU2Irrep::n_symbol(&s1, &s2, &s5) == 0
        || SU2Irrep::n_symbol(&s5, &s3, &s4) == 0
        || SU2Irrep::n_symbol(&s2, &s3, &s6) == 0
        || SU2Irrep::n_symbol(&s1, &s6, &s4) == 0
    {
        return Ok(0.0);
    }

    su2_f_symbol_spin2(
        s1.spin2_value(),
        s2.spin2_value(),
        s3.spin2_value(),
        s4.spin2_value(),
        s5.spin2_value(),
        s6.spin2_value(),
    )
}

fn su2_fusion_tensor(a: SU2Irrep, b: SU2Irrep, c: SU2Irrep) -> Result<Array4<f64>> {
    let ja2 = a.spin2_value();
    let jb2 = b.spin2_value();
    let jc2 = c.spin2_value();
    validate_triangle_spin2(ja2, jb2, jc2)?;

    let dim_a = su2_dim_spin2(ja2)?;
    let dim_b = su2_dim_spin2(jb2)?;
    let dim_c = su2_dim_spin2(jc2)?;

    let mut tensor = Array4::zeros((dim_a, dim_b, dim_c, 1));
    for kc in 0..dim_c {
        for kb in 0..dim_b {
            for ka in 0..dim_a {
                tensor[[ka, kb, kc, 0]] = su2_clebsch_gordan_spin2(
                    ja2,
                    ja2 - 2 * ka as i64,
                    jb2,
                    jb2 - 2 * kb as i64,
                    jc2,
                    jc2 - 2 * kc as i64,
                )?;
            }
        }
    }

    Ok(tensor)
}

fn su2_dim_spin2(spin2: i64) -> Result<usize> {
    let spin2 = usize::try_from(spin2)
        .map_err(|_| Tensor0Error::Message("SU2 spin must be non-negative".to_string()))?;
    Ok(spin2 + 1)
}
