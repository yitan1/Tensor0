//! TensorKit-like tree transformer construction.
//!
//! This module is the bridge from fusion-tree operation maps to indexed
//! transform payloads. Future SU2, product-sector SimpleFusion aliases, and
//! eventual GenericFusion support should build nontrivial basis matrices here.

use ndarray::Array2;

use crate::error::{Result, Tensor0Error};
use crate::fusion_tree::auxiliary::{is_cyclic_permutation, linearize_permutation};
use crate::fusion_tree::braiding_ops::{braid_block, braid_pair, flip_pair};
use crate::fusion_tree::duality_ops::{transpose_block, transpose_pair};
use crate::fusion_tree::{fusion_blocks, FusionTreeBlock, FusionTreePair};
use crate::layout::SectorStructure;
use crate::sector::{FusionStyle, Sector};
use crate::space::HomSpace;

#[derive(Clone, Debug, PartialEq)]
pub struct AbelianTransformData {
    pub coeff: f64,
    /// Canonical source fusion-tree-pair/subblock index.
    pub src: usize,
    /// Canonical destination fusion-tree-pair/subblock index.
    pub dst: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GenericTransformData {
    pub transform: Array2<f64>,
    /// Canonical source indices corresponding to basis-transform columns.
    pub src_indices: Vec<usize>,
    /// Canonical destination indices corresponding to basis-transform rows.
    pub dst_indices: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum TreeTransformer {
    Abelian(Vec<AbelianTransformData>),
    Generic(Vec<GenericTransformData>),
}

/// Lowers a visible-leg flip to `(destination index, coefficient)` entries in
/// canonical source-tree order.
pub fn flip_entries<I: Sector>(
    src: &HomSpace<I>,
    dst: &HomSpace<I>,
    src_structure: &SectorStructure<I>,
    dst_structure: &SectorStructure<I>,
    indices: &[usize],
    inv: bool,
) -> Result<Vec<(usize, f64)>> {
    validate_structures(src, dst, src_structure, dst_structure)?;
    let expected_dst = src.flip(indices)?;
    if &expected_dst != dst {
        return Err(Tensor0Error::Message(
            "incompatible spaces for flipping".to_string(),
        ));
    }
    let mut entries = Vec::with_capacity(src_structure.fusiontree_pair_count());
    for src_pair in src_structure.fusiontree_pairs() {
        let (dst_pair, coeff) = flip_pair(src_pair, indices, inv)?;
        let dst_index = dst_structure
            .fusiontree_pair_index(&dst_pair)
            .ok_or_else(|| {
                Tensor0Error::Message("flip destination fusion tree pair was not found".to_string())
            })?;
        entries.push((dst_index, coeff));
    }
    Ok(entries)
}

pub fn tree_permuter<I: Sector>(
    src: &HomSpace<I>,
    dst: &HomSpace<I>,
    src_structure: &SectorStructure<I>,
    dst_structure: &SectorStructure<I>,
    p_codomain: &[usize],
    p_domain: &[usize],
) -> Result<TreeTransformer> {
    let levels_codomain = (0..src.numout()).collect::<Vec<_>>();
    let levels_domain = (src.numout()..src.numind()).collect::<Vec<_>>();
    tree_braider(
        src,
        dst,
        src_structure,
        dst_structure,
        p_codomain,
        p_domain,
        &levels_codomain,
        &levels_domain,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn tree_braider<I: Sector>(
    src: &HomSpace<I>,
    dst: &HomSpace<I>,
    src_structure: &SectorStructure<I>,
    dst_structure: &SectorStructure<I>,
    p_codomain: &[usize],
    p_domain: &[usize],
    levels_codomain: &[usize],
    levels_domain: &[usize],
) -> Result<TreeTransformer> {
    validate_structures(src, dst, src_structure, dst_structure)?;
    if levels_codomain.len() != src.numout() || levels_domain.len() != src.numin() {
        return Err(Tensor0Error::Message(
            "braid levels must match source codomain and domain arity".to_string(),
        ));
    }

    let expected_dst = src.permute(p_codomain, p_domain)?;
    if &expected_dst != dst {
        return Err(Tensor0Error::Message(
            "incompatible spaces for permuting".to_string(),
        ));
    }
    let is_identity = p_codomain.iter().copied().eq(0..src.numout())
        && p_domain.iter().copied().eq(src.numout()..src.numind());
    if is_identity {
        return match I::fusion_style() {
            FusionStyle::UniqueFusion => Ok(TreeTransformer::Abelian(abelian_tree_transformer(
                src_structure,
                dst_structure,
                |src_pair| Ok((src_pair.clone(), 1.0)),
            )?)),
            FusionStyle::SimpleFusion => Ok(TreeTransformer::Generic(generic_tree_transformer(
                src,
                src_structure,
                dst_structure,
                |src_block| {
                    let mut identity =
                        Array2::zeros((src_block.trees().len(), src_block.trees().len()));
                    identity.diag_mut().fill(1.0);
                    Ok((src_block.clone(), identity))
                },
            )?)),
            FusionStyle::GenericFusion => Err(Tensor0Error::Message(
                "TreeTransformer does not support GenericFusion sector families".to_string(),
            )),
        };
    }

    match I::fusion_style() {
        FusionStyle::UniqueFusion => {
            let data = abelian_tree_transformer(src_structure, dst_structure, |src_pair| {
                braid_pair(
                    src_pair,
                    p_codomain,
                    p_domain,
                    levels_codomain,
                    levels_domain,
                )
            })?;
            Ok(TreeTransformer::Abelian(data))
        }
        FusionStyle::SimpleFusion => {
            let data = generic_tree_transformer(src, src_structure, dst_structure, |src_block| {
                braid_block(
                    src_block,
                    p_codomain,
                    p_domain,
                    levels_codomain,
                    levels_domain,
                )
            })?;
            Ok(TreeTransformer::Generic(data))
        }
        FusionStyle::GenericFusion => Err(Tensor0Error::Message(
            "TreeTransformer does not support GenericFusion sector families".to_string(),
        )),
    }
}

pub fn tree_transposer<I: Sector>(
    src: &HomSpace<I>,
    dst: &HomSpace<I>,
    src_structure: &SectorStructure<I>,
    dst_structure: &SectorStructure<I>,
    p_codomain: &[usize],
    p_domain: &[usize],
) -> Result<TreeTransformer> {
    validate_structures(src, dst, src_structure, dst_structure)?;
    let expected_dst = src.permute(p_codomain, p_domain)?;
    if &expected_dst != dst {
        return Err(Tensor0Error::Message(
            "incompatible spaces for transposing".to_string(),
        ));
    }
    let permutation = linearize_permutation(p_codomain, p_domain, src.numout(), src.numin())?;
    if !is_cyclic_permutation(&permutation) {
        return Err(Tensor0Error::Message(
            "fusion tree transpose requires a cyclic planar permutation".to_string(),
        ));
    }

    match I::fusion_style() {
        FusionStyle::UniqueFusion => {
            let data = abelian_tree_transformer(src_structure, dst_structure, |src_pair| {
                transpose_pair(src_pair, p_codomain, p_domain)
            })?;
            Ok(TreeTransformer::Abelian(data))
        }
        FusionStyle::SimpleFusion => {
            let data = generic_tree_transformer(src, src_structure, dst_structure, |src_block| {
                transpose_block(src_block, p_codomain, p_domain)
            })?;
            Ok(TreeTransformer::Generic(data))
        }
        FusionStyle::GenericFusion => Err(Tensor0Error::Message(
            "TreeTransformer does not support GenericFusion sector families".to_string(),
        )),
    }
}

fn abelian_tree_transformer<I, F>(
    src_structure: &SectorStructure<I>,
    dst_structure: &SectorStructure<I>,
    transform: F,
) -> Result<Vec<AbelianTransformData>>
where
    I: Sector,
    F: Fn(&FusionTreePair<I>) -> Result<(FusionTreePair<I>, f64)>,
{
    let mut data = Vec::with_capacity(src_structure.fusiontree_pair_count());
    for (src_index, src_pair) in src_structure.fusiontree_pairs().enumerate() {
        let (dst_pair, coeff) = transform(src_pair)?;
        let dst_index = dst_structure
            .fusiontree_pair_index(&dst_pair)
            .ok_or_else(|| {
                Tensor0Error::Message(
                    "transform destination fusion tree pair was not found".to_string(),
                )
            })?;
        data.push(AbelianTransformData {
            coeff,
            src: src_index,
            dst: dst_index,
        });
    }
    Ok(data)
}

fn generic_tree_transformer<I, F>(
    src: &HomSpace<I>,
    src_structure: &SectorStructure<I>,
    dst_structure: &SectorStructure<I>,
    transform: F,
) -> Result<Vec<GenericTransformData>>
where
    I: Sector,
    F: Fn(&FusionTreeBlock<I>) -> Result<(FusionTreeBlock<I>, Array2<f64>)>,
{
    let src_blocks = fusion_blocks(src)?;
    let mut data = Vec::with_capacity(src_blocks.len());

    for src_block in &src_blocks {
        let (dst_block, basis_transform) = transform(src_block)?;
        debug_assert_eq!(basis_transform.nrows(), dst_block.trees().len());
        debug_assert_eq!(basis_transform.ncols(), src_block.trees().len());

        let src_indices = block_indices(src_block, src_structure, "source")?;
        let dst_indices = block_indices(&dst_block, dst_structure, "destination")?;
        data.push(GenericTransformData {
            transform: basis_transform,
            src_indices,
            dst_indices,
        });
    }
    Ok(data)
}

fn block_indices<I: Sector>(
    block: &FusionTreeBlock<I>,
    structure: &SectorStructure<I>,
    label: &str,
) -> Result<Vec<usize>> {
    block
        .trees()
        .iter()
        .map(|pair| {
            structure.fusiontree_pair_index(pair).ok_or_else(|| {
                Tensor0Error::Message(format!("transform {label} fusion tree pair was not found"))
            })
        })
        .collect()
}

fn validate_structures<I: Sector>(
    src: &HomSpace<I>,
    dst: &HomSpace<I>,
    src_structure: &SectorStructure<I>,
    dst_structure: &SectorStructure<I>,
) -> Result<()> {
    if !src_structure.matches_space(src) {
        return Err(Tensor0Error::Message(
            "source sectorstructure does not match source HomSpace sector structure".to_string(),
        ));
    }
    if !dst_structure.matches_space(dst) {
        return Err(Tensor0Error::Message(
            "destination sectorstructure does not match destination HomSpace sector structure"
                .to_string(),
        ));
    }
    Ok(())
}
