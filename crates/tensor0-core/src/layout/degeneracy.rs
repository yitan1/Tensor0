use crate::error::{Result, Tensor0Error};
use crate::fusion_tree::FusionTree;
use crate::sector::Sector;
use crate::space::{HomSpace, ProductSpace};

use super::sector_structure::{build_sector_structure, SectorStructure};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockStructure {
    pub row_dim: usize,
    pub col_dim: usize,
    pub start: usize,
    pub stop: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SubblockStructure {
    pub sizes: Vec<usize>,
    pub strides: Vec<usize>,
    pub offset: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DegeneracyStructure {
    pub total_dim: usize,
    pub blockstructure: Vec<BlockStructure>,
    pub subblockstructure: Vec<SubblockStructure>,
}

#[derive(Clone, Debug)]
struct DegeneracyTreeStructure {
    dim: usize,
    dims: Vec<usize>,
}

pub fn build_degeneracy_structure<I: Sector>(space: &HomSpace<I>) -> Result<DegeneracyStructure> {
    let sectorstructure = build_sector_structure(space)?;
    build_degeneracy_structure_unchecked(space, &sectorstructure)
}

pub fn build_degeneracy_structure_from_sector_structure<I: Sector>(
    space: &HomSpace<I>,
    sectorstructure: &SectorStructure<I>,
) -> Result<DegeneracyStructure> {
    if !sectorstructure.matches_space(space) {
        return Err(Tensor0Error::Message(
            "sectorstructure does not match HomSpace sector structure".to_string(),
        ));
    }

    build_degeneracy_structure_unchecked(space, sectorstructure)
}

pub(super) fn build_degeneracy_structure_unchecked<I: Sector>(
    space: &HomSpace<I>,
    sectorstructure: &SectorStructure<I>,
) -> Result<DegeneracyStructure> {
    let mut blockstructure = Vec::with_capacity(sectorstructure.blocksector_count());
    let mut subblockstructure = Vec::with_capacity(sectorstructure.fusiontree_pair_count());
    let mut start = 0usize;
    let mut tree_index = 0usize;

    for blocksector in sectorstructure.blocksectors() {
        let first_pair = sectorstructure
            .fusiontree_pair_at(tree_index)
            .expect("blocksector has at least one fusion tree pair");
        let first_row = &first_pair.row;

        let mut col_structure = Vec::new();
        let mut col_dim = 0usize;
        let mut probe_index = tree_index;
        while let Some(pair) = sectorstructure.fusiontree_pair_at(probe_index) {
            let row = &pair.row;
            let col = &pair.col;
            if row != first_row {
                break;
            }

            let dims = fusiontree_degeneracy_dims(space.domain(), col);
            let dim = degeneracy_dim(&dims)?;
            col_dim = checked_add(col_dim, dim, "basis dimension")?;
            col_structure.push(DegeneracyTreeStructure { dim, dims });
            probe_index += 1;
        }
        let col_count = col_structure.len();

        let mut row_structure = Vec::new();
        let mut row_dim = 0usize;
        let mut probe_index = tree_index;
        while let Some(pair) = sectorstructure.fusiontree_pair_at(probe_index) {
            let row = &pair.row;
            if row.coupled() != blocksector {
                break;
            }

            let dims = fusiontree_degeneracy_dims(space.codomain(), row);
            let dim = degeneracy_dim(&dims)?;
            row_dim = checked_add(row_dim, dim, "basis dimension")?;
            row_structure.push(DegeneracyTreeStructure { dim, dims });
            probe_index += col_count;
        }
        let row_count = row_structure.len();

        let mut row_offset = 0usize;
        for row_entry in &row_structure {
            let mut col_offset = 0usize;
            for col_entry in &col_structure {
                let row_start = checked_mul(row_offset, col_dim, "subblock offset")?;
                let offset = checked_add(start, row_start, "subblock offset")
                    .and_then(|offset| checked_add(offset, col_offset, "subblock offset"))?;
                let sizes = concat_dims(&row_entry.dims, &col_entry.dims);
                let strides = subblock_strides(&row_entry.dims, &col_entry.dims, col_dim)?;

                subblockstructure.push(SubblockStructure {
                    sizes,
                    strides,
                    offset,
                });

                col_offset = checked_add(col_offset, col_entry.dim, "column basis offset")?;
            }
            row_offset = checked_add(row_offset, row_entry.dim, "row basis offset")?;
        }

        tree_index += row_count * col_count;

        let block_len = checked_mul(row_dim, col_dim, "block dimension")?;
        let stop = checked_add(start, block_len, "layout offset")?;
        blockstructure.push(BlockStructure {
            row_dim,
            col_dim,
            start,
            stop,
        });
        start = stop;
    }

    Ok(DegeneracyStructure {
        total_dim: start,
        blockstructure,
        subblockstructure,
    })
}

fn fusiontree_degeneracy_dims<I: Sector>(
    product: &ProductSpace<I>,
    tree: &FusionTree<I>,
) -> Vec<usize> {
    product
        .factors()
        .iter()
        .zip(tree.uncoupled())
        .map(|(factor, sector)| factor.sector_dim(sector))
        .collect()
}

fn degeneracy_dim(dims: &[usize]) -> Result<usize> {
    dims.iter().try_fold(1usize, |total, dim| {
        checked_mul(total, *dim, "degeneracy dimension")
    })
}

fn concat_dims(row_dims: &[usize], col_dims: &[usize]) -> Vec<usize> {
    row_dims.iter().chain(col_dims.iter()).copied().collect()
}

fn subblock_strides(row_dims: &[usize], col_dims: &[usize], col_dim: usize) -> Result<Vec<usize>> {
    let mut strides = vec![0; row_dims.len() + col_dims.len()];

    let mut stride = 1usize;
    for (index, dim) in col_dims.iter().enumerate().rev() {
        strides[row_dims.len() + index] = stride;
        stride = checked_mul(stride, *dim, "column subblock stride")?;
    }

    let mut stride = col_dim;
    for (index, dim) in row_dims.iter().enumerate().rev() {
        strides[index] = stride;
        stride = checked_mul(stride, *dim, "row subblock stride")?;
    }

    Ok(strides)
}

fn checked_add(left: usize, right: usize, context: &str) -> Result<usize> {
    left.checked_add(right)
        .ok_or_else(|| Tensor0Error::Message(format!("{context} overflowed")))
}

fn checked_mul(left: usize, right: usize, context: &str) -> Result<usize> {
    left.checked_mul(right)
        .ok_or_else(|| Tensor0Error::Message(format!("{context} overflowed")))
}
