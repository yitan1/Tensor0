use crate::error::{Result, Tensor0Error};
use crate::fingerprint::fingerprint;
use crate::sector::Sector;

use super::spec::{ElementarySpaceSpec, SectorDimSpec};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GradedSpace<I: Sector> {
    dims: Vec<(I, usize)>,
    is_dual: bool,
    fingerprint: u128,
}

impl<I: Sector> GradedSpace<I> {
    pub fn new(dims: Vec<(I, usize)>, is_dual: bool) -> Result<Self> {
        let mut dims = dims
            .into_iter()
            .filter(|(_, dim)| *dim != 0)
            .collect::<Vec<_>>();

        dims.sort_by(|left, right| left.0.cmp(&right.0));
        for window in dims.windows(2) {
            if window[0].0 == window[1].0 {
                return Err(Tensor0Error::Message(
                    "sector appears multiple times".to_string(),
                ));
            }
        }

        let fingerprint = fingerprint(&elementary_spec::<I>(&dims, is_dual, 0))?;
        Ok(GradedSpace {
            dims,
            is_dual,
            fingerprint,
        })
    }

    pub fn from_spec(spec: ElementarySpaceSpec) -> Result<Self> {
        let actual = spec.sector_spec.canonicalize()?;
        let expected = I::sector_spec().canonicalize()?;
        if actual != expected {
            return Err(Tensor0Error::SectorSpecMismatch {
                expected: format!("{expected:?}"),
                actual: format!("{actual:?}"),
            });
        }

        let dims = spec
            .sectors
            .into_iter()
            .map(|sector_dim| Ok((I::decode_value(&sector_dim.sector)?, sector_dim.dim)))
            .collect::<Result<Vec<_>>>()?;
        GradedSpace::new(dims, spec.is_dual)
    }

    pub fn to_spec(&self) -> ElementarySpaceSpec {
        elementary_spec(&self.dims, self.is_dual, self.fingerprint)
    }

    pub fn dual(&self) -> Result<Self> {
        let is_dual = !self.is_dual;
        let fingerprint = fingerprint(&elementary_spec::<I>(&self.dims, is_dual, 0))?;
        Ok(GradedSpace {
            dims: self.dims.clone(),
            is_dual,
            fingerprint,
        })
    }

    pub fn sectors(&self) -> Vec<(I, usize)> {
        self.dims
            .iter()
            .map(|(sector, dim)| {
                let sector = if self.is_dual {
                    sector.dual()
                } else {
                    sector.clone()
                };
                (sector, *dim)
            })
            .collect()
    }

    pub fn sector_dim(&self, sector: &I) -> usize {
        let key = if self.is_dual {
            sector.dual()
        } else {
            sector.clone()
        };
        self.dims
            .binary_search_by(|(stored_sector, _)| stored_sector.cmp(&key))
            .map(|index| self.dims[index].1)
            .unwrap_or(0)
    }

    pub fn dim(&self) -> usize {
        self.sectors()
            .into_iter()
            .map(|(sector, _)| {
                sector
                    .quantum_dim()
                    .checked_mul(self.sector_dim(&sector))
                    .expect("graded space dimension overflowed")
            })
            .try_fold(0usize, |total, dim| {
                total
                    .checked_add(dim)
                    .ok_or("graded space dimension overflowed")
            })
            .expect("graded space dimension overflowed")
    }

    pub fn is_dual(&self) -> bool {
        self.is_dual
    }

    pub fn fingerprint(&self) -> u128 {
        self.fingerprint
    }
}

pub fn infimum_space<I: Sector>(
    left: &GradedSpace<I>,
    right: &GradedSpace<I>,
) -> Result<GradedSpace<I>> {
    if left.is_dual() != right.is_dual() {
        return Err(Tensor0Error::Message(
            "infimum spaces must have the same dual flag".to_string(),
        ));
    }

    let is_dual = left.is_dual();
    let dims = left
        .sectors()
        .into_iter()
        .filter_map(|(sector, left_dim)| {
            let right_dim = right.sector_dim(&sector);
            let dim = left_dim.min(right_dim);
            if dim == 0 {
                return None;
            }
            let stored_sector = if is_dual { sector.dual() } else { sector };
            Some((stored_sector, dim))
        })
        .collect::<Vec<_>>();
    GradedSpace::new(dims, is_dual)
}

fn elementary_spec<I: Sector>(
    dims: &[(I, usize)],
    is_dual: bool,
    fingerprint: u128,
) -> ElementarySpaceSpec {
    ElementarySpaceSpec {
        sector_spec: I::sector_spec(),
        sectors: dims
            .iter()
            .map(|(sector, dim)| SectorDimSpec {
                sector: sector.encode_value().to_vec(),
                dim: *dim,
            })
            .collect(),
        is_dual,
        fingerprint,
    }
}
