use std::collections::BTreeMap;
use std::sync::Arc;

use crate::error::{Result, Tensor0Error};
use crate::sector::Sector;

use super::spec::{ElementarySpaceSpec, SectorDimSpec};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct GradedSpace<I: Sector> {
    sector_dims: Arc<[(I, usize)]>,
    is_dual: bool,
}

impl<I: Sector> GradedSpace<I> {
    pub fn zero(is_dual: bool) -> Self {
        GradedSpace {
            sector_dims: Vec::new().into(),
            is_dual,
        }
    }

    pub fn unit() -> Self {
        GradedSpace {
            sector_dims: vec![(I::unit(), 1)].into(),
            is_dual: false,
        }
    }

    /// Return whether this is the canonical unit space or its dual.
    pub fn is_unit(&self) -> bool {
        matches!(
            self.sector_dims.as_ref(),
            [(sector, 1)] if sector == &I::unit()
        )
    }

    pub fn new(sector_dims: Vec<(I, usize)>, is_dual: bool) -> Result<Self> {
        let mut sector_dims = sector_dims
            .into_iter()
            .filter(|(_, dim)| *dim != 0)
            .collect::<Vec<_>>();

        sector_dims.sort_by(|left, right| left.0.cmp(&right.0));
        for window in sector_dims.windows(2) {
            if window[0].0 == window[1].0 {
                return Err(Tensor0Error::Message(
                    "sector appears multiple times".to_string(),
                ));
            }
        }

        Ok(GradedSpace {
            sector_dims: sector_dims.into(),
            is_dual,
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

        let sector_dims = spec
            .sectors
            .into_iter()
            .map(|sector_dim| Ok((I::decode_value(&sector_dim.sector)?, sector_dim.dim)))
            .collect::<Result<Vec<_>>>()?;
        GradedSpace::new(sector_dims, spec.is_dual)
    }

    pub fn to_spec(&self) -> ElementarySpaceSpec {
        elementary_spec(self.sector_dims.as_ref(), self.is_dual)
    }

    pub fn dual(&self) -> Self {
        GradedSpace {
            sector_dims: Arc::clone(&self.sector_dims),
            is_dual: !self.is_dual,
        }
    }

    /// Toggle the duality presentation while preserving visible sectors.
    pub fn flip(&self) -> Self {
        let mut sector_dims = self
            .sector_dims
            .iter()
            .map(|(sector, dim)| (sector.dual(), *dim))
            .collect::<Vec<_>>();
        sector_dims.sort_by(|left, right| left.0.cmp(&right.0));
        GradedSpace {
            sector_dims: sector_dims.into(),
            is_dual: !self.is_dual,
        }
    }

    pub fn is_dual(&self) -> bool {
        self.is_dual
    }

    pub fn sectors(&self) -> Vec<(I, usize)> {
        self.sector_dims
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

    pub(crate) fn visible_sectors(&self) -> impl ExactSizeIterator<Item = I> + '_ {
        self.sector_dims.iter().map(|(sector, _)| {
            if self.is_dual {
                sector.dual()
            } else {
                sector.clone()
            }
        })
    }

    pub fn sector_dim(&self, sector: &I) -> usize {
        let key = if self.is_dual {
            sector.dual()
        } else {
            sector.clone()
        };
        self.sector_dims
            .binary_search_by(|(stored_sector, _)| stored_sector.cmp(&key))
            .map(|index| self.sector_dims[index].1)
            .unwrap_or(0)
    }

    pub fn has_sector(&self, sector: &I) -> bool {
        self.sector_dim(sector) != 0
    }

    pub fn reduced_dim(&self) -> usize {
        self.sector_dims
            .iter()
            .map(|(_, dim)| *dim)
            .try_fold(0usize, |total, dim| {
                total
                    .checked_add(dim)
                    .ok_or("graded space reduced dimension overflowed")
            })
            .expect("graded space reduced dimension overflowed")
    }

    pub fn dim(&self) -> usize {
        self.sector_dims
            .iter()
            .map(|(sector, dim)| {
                let quantum_dim = if self.is_dual {
                    sector.dual().quantum_dim()
                } else {
                    sector.quantum_dim()
                };
                quantum_dim
                    .checked_mul(*dim)
                    .expect("graded space dimension overflowed")
            })
            .try_fold(0usize, |total, dim| {
                total
                    .checked_add(dim)
                    .ok_or("graded space dimension overflowed")
            })
            .expect("graded space dimension overflowed")
    }

    pub fn direct_sum(&self, rhs: &Self) -> Result<Self> {
        if self.is_dual() != rhs.is_dual() {
            return Err(Tensor0Error::Message(
                "direct sum must have the same dual flag".to_string(),
            ));
        }

        let is_dual = self.is_dual();
        let mut visible_sector_dims = BTreeMap::<I, usize>::new();
        for (sector, dim) in self.sectors().into_iter().chain(rhs.sectors()) {
            let entry = visible_sector_dims.entry(sector).or_insert(0);
            *entry = entry.checked_add(dim).ok_or_else(|| {
                Tensor0Error::Message("direct sum dimension overflowed".to_string())
            })?;
        }
        let sector_dims = visible_sector_dims
            .into_iter()
            .map(|(sector, dim)| {
                let sector = if is_dual { sector.dual() } else { sector };
                (sector, dim)
            })
            .collect();
        GradedSpace::new(sector_dims, is_dual)
    }

    pub fn is_isomorphic(&self, rhs: &Self) -> bool {
        self.is_monomorphic(rhs) && self.is_epimorphic(rhs)
    }

    pub fn is_monomorphic(&self, rhs: &Self) -> bool {
        self.sectors()
            .into_iter()
            .all(|(sector, dim)| dim <= rhs.sector_dim(&sector))
    }

    pub fn is_epimorphic(&self, rhs: &Self) -> bool {
        rhs.is_monomorphic(self)
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
    let sector_dims = left
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
    GradedSpace::new(sector_dims, is_dual)
}

pub fn supremum_space<I: Sector>(
    left: &GradedSpace<I>,
    right: &GradedSpace<I>,
) -> Result<GradedSpace<I>> {
    if left.is_dual() != right.is_dual() {
        return Err(Tensor0Error::Message(
            "supremum spaces must have the same dual flag".to_string(),
        ));
    }

    let is_dual = left.is_dual();
    let mut visible_sector_dims = BTreeMap::<I, usize>::new();
    for (sector, dim) in left.sectors().into_iter().chain(right.sectors()) {
        let entry = visible_sector_dims.entry(sector).or_insert(0);
        *entry = (*entry).max(dim);
    }
    let sector_dims = visible_sector_dims
        .into_iter()
        .map(|(sector, dim)| {
            let sector = if is_dual { sector.dual() } else { sector };
            (sector, dim)
        })
        .collect();
    GradedSpace::new(sector_dims, is_dual)
}

fn elementary_spec<I: Sector>(sector_dims: &[(I, usize)], is_dual: bool) -> ElementarySpaceSpec {
    ElementarySpaceSpec {
        sector_spec: I::sector_spec(),
        sectors: sector_dims
            .iter()
            .map(|(sector, dim)| SectorDimSpec {
                sector: sector.encode_value().to_vec(),
                dim: *dim,
            })
            .collect(),
        is_dual,
    }
}
