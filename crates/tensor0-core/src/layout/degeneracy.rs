use crate::error::{Result, Tensor0Error};
use crate::fusion_tree::FusionTree;
use crate::sector::Sector;
use crate::space::{HomSpace, ProductSpace};

use super::sector_structure::{
    build_sector_structure, sector_structure_fingerprint, SectorStructure,
};

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
    if sectorstructure.sector_fingerprint() != sector_structure_fingerprint(space)? {
        return Err(Tensor0Error::Message(
            "sectorstructure does not match HomSpace sector structure".to_string(),
        ));
    }

    build_degeneracy_structure_unchecked(space, sectorstructure)
}

fn build_degeneracy_structure_unchecked<I: Sector>(
    space: &HomSpace<I>,
    sectorstructure: &SectorStructure<I>,
) -> Result<DegeneracyStructure> {
    let mut blockstructure = Vec::with_capacity(sectorstructure.blocksectors().len());
    let mut subblockstructure = Vec::with_capacity(sectorstructure.fusiontree_pairs().len());
    let mut start = 0usize;
    let mut tree_index = 0usize;

    for blocksector in sectorstructure.blocksectors() {
        let Some(first_pair) = sectorstructure.fusiontree_pairs().get(tree_index) else {
            return Err(Tensor0Error::Message(
                "sectorstructure block has no fusion tree pairs".to_string(),
            ));
        };
        let first_row = &first_pair.row;
        let first_col = &first_pair.col;
        debug_assert!(&first_row.coupled == blocksector);
        debug_assert!(&first_col.coupled == blocksector);

        let mut col_structure = Vec::new();
        let mut col_dim = 0usize;
        let mut probe_index = tree_index;
        while let Some(pair) = sectorstructure.fusiontree_pairs().get(probe_index) {
            let row = &pair.row;
            let col = &pair.col;
            if row != first_row {
                break;
            }
            if &row.coupled != blocksector || &col.coupled != blocksector {
                return Err(Tensor0Error::Message(
                    "sectorstructure fusion tree pairs are inconsistent with block sectors"
                        .to_string(),
                ));
            }

            let dims = fusiontree_degeneracy_dims(space.domain(), col);
            let dim = degeneracy_dim(&dims)?;
            col_dim = checked_add(col_dim, dim, "basis dimension")?;
            col_structure.push(DegeneracyTreeStructure { dim, dims });
            probe_index = checked_add(probe_index, 1, "fusion tree pair index")?;
        }
        if col_structure.is_empty() {
            return Err(Tensor0Error::Message(
                "sectorstructure block has no column fusion trees".to_string(),
            ));
        }
        let col_count = col_structure.len();

        let mut row_structure = Vec::new();
        let mut row_dim = 0usize;
        let mut probe_index = tree_index;
        while let Some(pair) = sectorstructure.fusiontree_pairs().get(probe_index) {
            let row = &pair.row;
            let col = &pair.col;
            if &row.coupled != blocksector {
                break;
            }
            if col != first_col || &col.coupled != blocksector {
                return Err(Tensor0Error::Message(
                    "sectorstructure fusion tree pairs are inconsistent with block sectors"
                        .to_string(),
                ));
            }

            let dims = fusiontree_degeneracy_dims(space.codomain(), row);
            let dim = degeneracy_dim(&dims)?;
            row_dim = checked_add(row_dim, dim, "basis dimension")?;
            row_structure.push(DegeneracyTreeStructure { dim, dims });
            probe_index = checked_add(probe_index, col_count, "fusion tree pair index")?;
        }
        if row_structure.is_empty() {
            return Err(Tensor0Error::Message(
                "sectorstructure block has no row fusion trees".to_string(),
            ));
        }
        let row_count = row_structure.len();

        let mut row_offset = 0usize;
        for (row_index, row_entry) in row_structure.iter().enumerate() {
            let mut col_offset = 0usize;
            for (col_index, col_entry) in col_structure.iter().enumerate() {
                let pair_index = checked_add(
                    tree_index,
                    checked_add(
                        checked_mul(row_index, col_count, "fusion tree pair index")?,
                        col_index,
                        "fusion tree pair index",
                    )?,
                    "fusion tree pair index",
                )?;
                let Some(pair) = sectorstructure.fusiontree_pairs().get(pair_index) else {
                    return Err(Tensor0Error::Message(
                        "sectorstructure block has an incomplete row group".to_string(),
                    ));
                };
                let pair_row = &pair.row;
                let pair_col = &pair.col;
                if &pair_row.coupled != blocksector || &pair_col.coupled != blocksector {
                    return Err(Tensor0Error::Message(
                        "sectorstructure fusion tree pairs are inconsistent with block sectors"
                            .to_string(),
                    ));
                }

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

        let block_pair_count = checked_mul(row_count, col_count, "fusion tree pair count")?;
        tree_index = checked_add(tree_index, block_pair_count, "fusion tree pair index")?;

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

    if subblockstructure.len() != sectorstructure.fusiontree_pairs().len() {
        return Err(Tensor0Error::Message(
            "sectorstructure fusion tree pairs are inconsistent with block sectors".to_string(),
        ));
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
    let dims = product
        .sector_dims(&tree.uncoupled)
        .expect("sectorstructure fusion tree arity matches product space");
    debug_assert!(
        !dims.contains(&0),
        "sectorstructure fusion tree sectors are visible in product space"
    );
    dims
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
