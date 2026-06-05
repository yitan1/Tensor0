use std::collections::BTreeSet;

use crate::error::{Result, Tensor0Error};
use crate::fingerprint::fingerprint;
use crate::fusion_tree::{enumerate_fusion_trees, FusionTree};
use crate::sector::{fusion_sectors, FusionStyle, Sector};

use super::graded::GradedSpace;
use super::spec::ProductSpaceSpec;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ProductSectorTuple<I: Sector> {
    pub sectors: Vec<I>,
    pub is_dual: Vec<bool>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProductSpace<I: Sector> {
    factors: Vec<GradedSpace<I>>,
    fingerprint: u128,
}

impl<I: Sector> ProductSpace<I> {
    pub fn new(factors: Vec<GradedSpace<I>>) -> Result<Self> {
        let fingerprint = fingerprint(&product_spec::<I>(&factors, 0))?;
        Ok(ProductSpace {
            factors,
            fingerprint,
        })
    }

    pub fn from_spec(spec: ProductSpaceSpec) -> Result<Self> {
        let actual = spec.sector_spec.canonicalize()?;
        let expected = I::sector_spec().canonicalize()?;
        if actual != expected {
            return Err(Tensor0Error::SectorSpecMismatch {
                expected: format!("{expected:?}"),
                actual: format!("{actual:?}"),
            });
        }

        let factors = spec
            .factors
            .into_iter()
            .map(GradedSpace::<I>::from_spec)
            .collect::<Result<Vec<_>>>()?;
        ProductSpace::new(factors)
    }

    pub fn to_spec(&self) -> ProductSpaceSpec {
        product_spec(&self.factors, self.fingerprint)
    }

    pub fn factors(&self) -> &[GradedSpace<I>] {
        &self.factors
    }

    pub fn dims(&self) -> Vec<usize> {
        self.factors.iter().map(GradedSpace::dim).collect()
    }

    pub fn dim(&self) -> usize {
        self.dims()
            .into_iter()
            .try_fold(1usize, |total, dim| {
                total
                    .checked_mul(dim)
                    .ok_or("product space dimension overflowed")
            })
            .expect("product space dimension overflowed")
    }

    pub fn sector_dims(&self, sectors: &[I]) -> Option<Vec<usize>> {
        if sectors.len() != self.factors.len() {
            return None;
        }

        Some(
            self.factors
                .iter()
                .zip(sectors)
                .map(|(factor, sector)| factor.sector_dim(sector))
                .collect(),
        )
    }

    pub fn sector_dim(&self, sectors: &[I]) -> Option<usize> {
        self.sector_dims(sectors).map(|dims| {
            dims.into_iter()
                .try_fold(1usize, |total, dim| {
                    total
                        .checked_mul(dim)
                        .ok_or("product sector dimension overflowed")
                })
                .expect("product sector dimension overflowed")
        })
    }

    pub fn fingerprint(&self) -> u128 {
        self.fingerprint
    }

    pub(crate) fn block_sectors(&self) -> Result<Vec<I>> {
        match self.factors.len() {
            0 => Ok(vec![I::unit()]),
            1 => {
                let mut blocksectors = self.factors[0]
                    .sectors()
                    .into_iter()
                    .map(|(sector, _)| sector)
                    .collect::<Vec<_>>();
                blocksectors.sort();
                Ok(blocksectors)
            }
            _ => {
                let mut blocksectors = BTreeSet::new();
                for tuple in self.sector_tuples()? {
                    for coupled in fusion_sectors(&tuple.sectors) {
                        blocksectors.insert(coupled);
                    }
                }
                Ok(blocksectors.into_iter().collect())
            }
        }
    }

    pub(crate) fn sector_tuples(&self) -> Result<Vec<ProductSectorTuple<I>>> {
        match self.factors.len() {
            0 => Ok(vec![ProductSectorTuple {
                sectors: vec![],
                is_dual: vec![],
            }]),
            _ => {
                let mut tuples = vec![ProductSectorTuple {
                    sectors: vec![],
                    is_dual: vec![],
                }];

                for factor in self.factors.iter().rev() {
                    let sectors = factor.sectors();
                    let mut next = Vec::with_capacity(tuples.len().saturating_mul(sectors.len()));
                    for tuple in tuples {
                        for (sector, _) in &sectors {
                            let mut tuple_sectors = Vec::with_capacity(tuple.sectors.len() + 1);
                            tuple_sectors.push(sector.clone());
                            tuple_sectors.extend(tuple.sectors.iter().cloned());

                            let mut tuple_is_dual = Vec::with_capacity(tuple.is_dual.len() + 1);
                            tuple_is_dual.push(factor.is_dual());
                            tuple_is_dual.extend(tuple.is_dual.iter().copied());

                            next.push(ProductSectorTuple {
                                sectors: tuple_sectors,
                                is_dual: tuple_is_dual,
                            });
                        }
                    }
                    tuples = next;
                }

                Ok(tuples)
            }
        }
    }

    pub(crate) fn fusion_trees(&self, coupled: &I) -> Result<Vec<FusionTree<I>>> {
        let mut trees = Vec::new();
        for tuple in self.sector_tuples()? {
            trees.extend(enumerate_fusion_trees(
                &tuple.sectors,
                &tuple.is_dual,
                coupled,
            )?);
        }
        Ok(trees)
    }
}

pub fn fuse_product_space<I: Sector>(product: &ProductSpace<I>) -> Result<GradedSpace<I>> {
    if I::fusion_style() == FusionStyle::GenericFusion {
        return Err(Tensor0Error::Message(
            "fuse_product_space does not support GenericFusion sector families".to_string(),
        ));
    }

    let mut factors = product.factors().iter();
    let Some(first) = factors.next() else {
        return GradedSpace::new(vec![(I::unit(), 1)], false);
    };

    let mut fused = fuse_single_space(first)?;
    for factor in factors {
        fused = fuse_two_spaces(&fused, &fuse_single_space(factor)?)?;
    }
    Ok(fused)
}

fn product_spec<I: Sector>(factors: &[GradedSpace<I>], fingerprint: u128) -> ProductSpaceSpec {
    ProductSpaceSpec {
        sector_spec: I::sector_spec(),
        factors: factors.iter().map(GradedSpace::to_spec).collect(),
        fingerprint,
    }
}

fn fuse_single_space<I: Sector>(space: &GradedSpace<I>) -> Result<GradedSpace<I>> {
    GradedSpace::new(space.sectors(), false)
}

fn fuse_two_spaces<I: Sector>(
    left: &GradedSpace<I>,
    right: &GradedSpace<I>,
) -> Result<GradedSpace<I>> {
    let mut dims = std::collections::BTreeMap::<I, usize>::new();
    for (left_sector, left_dim) in left.sectors() {
        for (right_sector, right_dim) in right.sectors() {
            for coupled in left_sector.fusion_outputs(&right_sector) {
                let multiplicity = I::n_symbol(&left_sector, &right_sector, &coupled);
                if multiplicity == 0 {
                    continue;
                }
                let degeneracy = checked_mul(left_dim, right_dim, "fused space dimension")?;
                let degeneracy = checked_mul(degeneracy, multiplicity, "fused space dimension")?;

                let entry = dims.entry(coupled).or_insert(0);
                *entry = entry.checked_add(degeneracy).ok_or_else(|| {
                    Tensor0Error::Message("fused space dimension overflowed".to_string())
                })?;
            }
        }
    }
    GradedSpace::new(dims.into_iter().collect(), false)
}

fn checked_mul(left: usize, right: usize, context: &str) -> Result<usize> {
    left.checked_mul(right)
        .ok_or_else(|| Tensor0Error::Message(format!("{context} overflowed")))
}
