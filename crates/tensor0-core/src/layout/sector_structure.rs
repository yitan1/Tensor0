use std::collections::BTreeSet;

use serde::Serialize;

use crate::error::{Result, Tensor0Error};
use crate::fingerprint::fingerprint;
use crate::fusion_tree::FusionTreePair;
use crate::sector::{FusionStyle, Sector, SectorSpec};
use crate::space::{HomSpace, ProductSpace};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SectorStructure<I: Sector> {
    sector_fingerprint: u128,
    blocksectors: Vec<I>,
    fusiontree_pairs: Vec<FusionTreePair<I>>,
}

impl<I: Sector> SectorStructure<I> {
    pub fn sector_fingerprint(&self) -> u128 {
        self.sector_fingerprint
    }

    pub fn blocksectors(&self) -> &[I] {
        &self.blocksectors
    }

    pub fn fusiontree_pairs(&self) -> &[FusionTreePair<I>] {
        &self.fusiontree_pairs
    }
}

#[derive(Serialize)]
struct SectorStructureFingerprintKey {
    sector_spec: SectorSpec,
    codomain: Vec<SectorStructureFactorKey>,
    domain: Vec<SectorStructureFactorKey>,
}

#[derive(Serialize)]
struct SectorStructureFactorKey {
    is_dual: bool,
    sectors: Vec<Vec<i64>>,
}

pub fn sector_structure_fingerprint<I: Sector>(space: &HomSpace<I>) -> Result<u128> {
    let key = SectorStructureFingerprintKey {
        sector_spec: I::sector_spec().canonicalize()?,
        codomain: product_sector_key(space.codomain()),
        domain: product_sector_key(space.domain()),
    };
    fingerprint(&key)
}

pub fn build_sector_structure<I: Sector>(space: &HomSpace<I>) -> Result<SectorStructure<I>> {
    if I::fusion_style() == FusionStyle::GenericFusion {
        return Err(Tensor0Error::Message(
            "layout does not support GenericFusion sector families".to_string(),
        ));
    }

    let sector_fingerprint = sector_structure_fingerprint(space)?;
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
        sector_fingerprint,
        blocksectors,
        fusiontree_pairs,
    })
}

fn product_sector_key<I: Sector>(product: &ProductSpace<I>) -> Vec<SectorStructureFactorKey> {
    product
        .factors()
        .iter()
        .map(|factor| SectorStructureFactorKey {
            is_dual: factor.is_dual(),
            sectors: factor
                .sectors()
                .into_iter()
                .map(|(sector, _)| sector.encode_value().into_iter().collect())
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
