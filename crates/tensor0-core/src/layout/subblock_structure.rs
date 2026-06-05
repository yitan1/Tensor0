use std::collections::HashMap;

use crate::error::{Result, Tensor0Error};
use crate::fusion_tree::FusionTreePair;
use crate::sector::Sector;
use crate::space::HomSpace;

use super::degeneracy::{
    build_degeneracy_structure_from_sector_structure, DegeneracyStructure, SubblockStructure,
};
use super::sector_structure::{build_sector_structure, SectorStructure};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SubblockStructureMap<I: Sector> {
    pairs: Vec<FusionTreePair<I>>,
    structures: Vec<SubblockStructure>,
    index: HashMap<FusionTreePair<I>, usize>,
}

impl<I: Sector> SubblockStructureMap<I> {
    pub fn from_layout(
        sectorstructure: &SectorStructure<I>,
        degeneracystructure: &DegeneracyStructure,
    ) -> Result<Self> {
        Self::new(
            sectorstructure.fusiontree_pairs().to_vec(),
            degeneracystructure.subblockstructure.clone(),
        )
    }

    pub fn new(pairs: Vec<FusionTreePair<I>>, structures: Vec<SubblockStructure>) -> Result<Self> {
        if pairs.len() != structures.len() {
            return Err(Tensor0Error::Message(
                "subblock structure map requires matching pair and structure lengths".to_string(),
            ));
        }

        let mut index = HashMap::with_capacity(pairs.len());
        for (pair_index, pair) in pairs.iter().enumerate() {
            if index.insert(pair.clone(), pair_index).is_some() {
                return Err(Tensor0Error::Message(
                    "subblock structure map requires unique fusion tree pairs".to_string(),
                ));
            }
        }

        Ok(SubblockStructureMap {
            pairs,
            structures,
            index,
        })
    }

    pub fn pairs(&self) -> &[FusionTreePair<I>] {
        &self.pairs
    }

    pub fn structures(&self) -> &[SubblockStructure] {
        &self.structures
    }

    pub fn index_of(&self, pair: &FusionTreePair<I>) -> Option<usize> {
        self.index.get(pair).copied()
    }

    pub fn get(&self, pair: &FusionTreePair<I>) -> Option<&SubblockStructure> {
        self.index_of(pair)
            .and_then(|index| self.structures.get(index))
    }
}

pub fn subblockstructure<I: Sector>(space: &HomSpace<I>) -> Result<SubblockStructureMap<I>> {
    let sectorstructure = build_sector_structure(space)?;
    let degeneracystructure =
        build_degeneracy_structure_from_sector_structure(space, &sectorstructure)?;
    SubblockStructureMap::from_layout(&sectorstructure, &degeneracystructure)
}
