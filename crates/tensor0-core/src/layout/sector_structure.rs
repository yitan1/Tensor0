use std::collections::BTreeSet;

use crate::error::{Result, Tensor0Error};
use crate::fusion_tree::FusionTreePair;
use crate::sector::{FusionStyle, Sector};
use crate::space::{HomSpace, ProductSpace};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SectorStructure<I: Sector> {
    sector_key: SectorStructureKey<I>,
    blocksectors: Vec<I>,
    fusiontree_pairs: Vec<FusionTreePair<I>>,
}

impl<I: Sector> SectorStructure<I> {
    pub(super) fn matches_space(&self, space: &HomSpace<I>) -> bool {
        self.sector_key == sector_structure_key(space)
    }

    pub fn blocksectors(&self) -> &[I] {
        &self.blocksectors
    }

    pub fn fusiontree_pairs(&self) -> &[FusionTreePair<I>] {
        &self.fusiontree_pairs
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct SectorStructureKey<I: Sector> {
    codomain: Vec<SectorStructureFactorKey<I>>,
    domain: Vec<SectorStructureFactorKey<I>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct SectorStructureFactorKey<I: Sector> {
    is_dual: bool,
    sectors: Vec<I>,
}

fn sector_structure_key<I: Sector>(space: &HomSpace<I>) -> SectorStructureKey<I> {
    SectorStructureKey {
        codomain: product_sector_key(space.codomain()),
        domain: product_sector_key(space.domain()),
    }
}

pub fn build_sector_structure<I: Sector>(space: &HomSpace<I>) -> Result<SectorStructure<I>> {
    if I::fusion_style() == FusionStyle::GenericFusion {
        return Err(Tensor0Error::Message(
            "layout does not support GenericFusion sector families".to_string(),
        ));
    }

    let sector_key = sector_structure_key(space);
    let blocksectors = hom_block_sectors(space)?;
    let mut fusiontree_pairs = Vec::new();

    for blocksector in &blocksectors {
        let row_trees = space.codomain().fusion_trees(blocksector)?;
        let col_trees = space.domain().fusion_trees(blocksector)?;

        for row in &row_trees {
            for col in &col_trees {
                fusiontree_pairs.push(FusionTreePair {
                    row: row.clone(),
                    col: col.clone(),
                });
            }
        }
    }

    Ok(SectorStructure {
        sector_key,
        blocksectors,
        fusiontree_pairs,
    })
}

fn product_sector_key<I: Sector>(product: &ProductSpace<I>) -> Vec<SectorStructureFactorKey<I>> {
    product
        .factors()
        .iter()
        .map(|factor| SectorStructureFactorKey {
            is_dual: factor.is_dual(),
            sectors: factor
                .sectors()
                .into_iter()
                .map(|(sector, _)| sector)
                .collect(),
        })
        .collect()
}

fn hom_block_sectors<I: Sector>(space: &HomSpace<I>) -> Result<Vec<I>> {
    let codomain = space.codomain();
    let domain = space.domain();
    let codomain_len = codomain.factors().len();
    let domain_len = domain.factors().len();

    if codomain_len == 0 && domain_len == 0 {
        return Ok(vec![I::unit()]);
    }

    let codomain_blocksectors = codomain.block_sectors()?;
    let domain_blocksectors = domain.block_sectors()?;

    if codomain_len == 0 {
        return Ok(domain_blocksectors
            .into_iter()
            .filter(|sector| sector == &I::unit())
            .collect());
    }
    if domain_len == 0 {
        return Ok(codomain_blocksectors
            .into_iter()
            .filter(|sector| sector == &I::unit())
            .collect());
    }

    if domain_len <= codomain_len {
        let codomain_set = codomain_blocksectors.into_iter().collect::<BTreeSet<_>>();
        Ok(domain_blocksectors
            .into_iter()
            .filter(|sector| codomain_set.contains(sector))
            .collect())
    } else {
        let domain_set = domain_blocksectors.into_iter().collect::<BTreeSet<_>>();
        Ok(codomain_blocksectors
            .into_iter()
            .filter(|sector| domain_set.contains(sector))
            .collect())
    }
}
