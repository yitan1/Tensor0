use std::collections::BTreeSet;

use crate::error::{Result, Tensor0Error};
use crate::fusion_tree::{FusionTree, FusionTreeBlock, FusionTreePair};
use crate::sector::{FusionStyle, Sector};
use crate::space::{HomSpace, ProductSectorSupport};

use super::indices::Indices;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SectorStructure<I: Sector> {
    sector_key: SectorStructureKey<I>,
    blocksectors: Indices<I>,
    fusiontree_pairs: Indices<FusionTreePair<I>>,
}

impl<I: Sector> SectorStructure<I> {
    pub(crate) fn matches_space(&self, space: &HomSpace<I>) -> bool {
        self.sector_key == sector_structure_key(space)
    }

    /// Returns coupled block sectors in canonical layout order.
    pub fn blocksectors(&self) -> impl ExactSizeIterator<Item = &I> + '_ {
        self.blocksectors.iter()
    }

    /// Returns fusion tree pairs in canonical subblock layout order.
    pub fn fusiontree_pairs(&self) -> impl ExactSizeIterator<Item = &FusionTreePair<I>> + '_ {
        self.fusiontree_pairs.iter()
    }

    /// Returns the number of coupled block sectors.
    pub fn blocksector_count(&self) -> usize {
        self.blocksectors.len()
    }

    /// Returns the number of fusion tree pairs.
    pub fn fusiontree_pair_count(&self) -> usize {
        self.fusiontree_pairs.len()
    }

    /// Returns the block sector for a layout index.
    pub fn blocksector_at(&self, index: usize) -> Option<&I> {
        self.blocksectors.get_index(index)
    }

    /// Returns the fusion tree pair for a subblock layout index.
    pub fn fusiontree_pair_at(&self, index: usize) -> Option<&FusionTreePair<I>> {
        self.fusiontree_pairs.get_index(index)
    }

    /// Returns the canonical layout index for a coupled block sector.
    pub fn blocksector_index(&self, sector: &I) -> Option<usize> {
        self.blocksectors.index_of(sector)
    }

    /// Returns the canonical subblock layout index for a fusion tree pair.
    pub fn fusiontree_pair_index(&self, pair: &FusionTreePair<I>) -> Option<usize> {
        self.fusiontree_pairs.index_of(pair)
    }

    /// Returns the canonical subblock layout index for UniqueFusion uncoupled sectors.
    pub fn unique_fusiontree_pair_index(
        &self,
        space: &HomSpace<I>,
        row_uncoupled: &[I],
        col_uncoupled: &[I],
    ) -> Result<Option<usize>> {
        if I::fusion_style() != FusionStyle::UniqueFusion {
            return Err(Tensor0Error::Message(
                "sector indexing requires a UniqueFusion sector family".to_string(),
            ));
        }
        if !self.matches_space(space) {
            return Err(Tensor0Error::Message(
                "sectorstructure does not match HomSpace sector structure".to_string(),
            ));
        }
        if row_uncoupled.len() != space.numout() || col_uncoupled.len() != space.numin() {
            return Ok(None);
        }

        let block = FusionTreeBlock::new(
            row_uncoupled.to_vec(),
            space
                .codomain()
                .factors()
                .iter()
                .map(|factor| factor.is_dual())
                .collect(),
            col_uncoupled.to_vec(),
            space
                .domain()
                .factors()
                .iter()
                .map(|factor| factor.is_dual())
                .collect(),
        )?;

        match block.trees() {
            [] => Ok(None),
            [pair] => Ok(self.fusiontree_pair_index(pair)),
            _ => Err(Tensor0Error::Message(
                "UniqueFusion sectors produced multiple fusion tree pairs".to_string(),
            )),
        }
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
    let codomain = space.codomain().sector_support();
    let domain = space.domain().sector_support();
    sector_structure_key_from_supports(&codomain, &domain)
}

pub fn build_sector_structure<I: Sector>(space: &HomSpace<I>) -> Result<SectorStructure<I>> {
    if I::fusion_style() == FusionStyle::GenericFusion {
        return Err(Tensor0Error::Message(
            "layout does not support GenericFusion sector families".to_string(),
        ));
    }
    if space.numout() == 0 && space.numin() == 0 {
        let unit = I::unit();
        let tree = FusionTree::new(vec![], unit.clone(), vec![], vec![], vec![])?;
        return Ok(SectorStructure {
            sector_key: SectorStructureKey {
                codomain: vec![],
                domain: vec![],
            },
            blocksectors: Indices::new(vec![unit])?,
            fusiontree_pairs: Indices::new(vec![FusionTreePair {
                row: tree.clone(),
                col: tree,
            }])?,
        });
    }

    let codomain = space.codomain().sector_support();
    let domain = space.domain().sector_support();
    let sector_key = sector_structure_key_from_supports(&codomain, &domain);
    let blocksectors = hom_block_sectors_from_supports(&codomain, &domain);
    let mut fusiontree_pairs = Vec::new();
    for blocksector in &blocksectors {
        let row_trees = codomain.fusion_trees(blocksector)?;
        let col_trees = domain.fusion_trees(blocksector)?;
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
        blocksectors: Indices::new(blocksectors)?,
        fusiontree_pairs: Indices::new(fusiontree_pairs)?,
    })
}

fn sector_structure_key_from_supports<I: Sector>(
    codomain: &ProductSectorSupport<I>,
    domain: &ProductSectorSupport<I>,
) -> SectorStructureKey<I> {
    SectorStructureKey {
        codomain: product_sector_key(codomain),
        domain: product_sector_key(domain),
    }
}

fn product_sector_key<I: Sector>(
    product: &ProductSectorSupport<I>,
) -> Vec<SectorStructureFactorKey<I>> {
    product
        .sectors
        .iter()
        .zip(&product.is_dual)
        .map(|(sectors, is_dual)| SectorStructureFactorKey {
            is_dual: *is_dual,
            sectors: sectors.clone(),
        })
        .collect()
}

fn hom_block_sectors_from_supports<I: Sector>(
    codomain: &ProductSectorSupport<I>,
    domain: &ProductSectorSupport<I>,
) -> Vec<I> {
    let codomain_len = codomain.sectors.len();
    let domain_len = domain.sectors.len();

    if codomain_len == 1 && domain_len == 1 {
        let codomain_set = codomain.sectors[0].iter().collect::<BTreeSet<_>>();
        let mut blocksectors = domain.sectors[0]
            .iter()
            .filter(|sector| codomain_set.contains(sector))
            .cloned()
            .collect::<Vec<_>>();
        blocksectors.sort();
        return blocksectors;
    }

    let codomain_blocksectors = codomain.block_sectors();
    let domain_blocksectors = domain.block_sectors();

    if domain_len <= codomain_len {
        let codomain_set = codomain_blocksectors.into_iter().collect::<BTreeSet<_>>();
        domain_blocksectors
            .into_iter()
            .filter(|sector| codomain_set.contains(sector))
            .collect()
    } else {
        let domain_set = domain_blocksectors.into_iter().collect::<BTreeSet<_>>();
        codomain_blocksectors
            .into_iter()
            .filter(|sector| domain_set.contains(sector))
            .collect()
    }
}
