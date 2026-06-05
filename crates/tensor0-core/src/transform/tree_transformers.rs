//! TensorKit-like tree transformer construction.
//!
//! This module is the bridge from fusion-tree operation maps to
//! `TreeTransformer` payloads. Future SU2, product-sector SimpleFusion aliases,
//! and eventual GenericFusion support should build nontrivial basis matrices
//! here.

use ndarray::Array2;

use crate::error::{Result, Tensor0Error};
use crate::fusion_tree::braiding_ops::{braid_block, braid_pair};
use crate::fusion_tree::duality_ops::{transpose_block, transpose_pair};
use crate::fusion_tree::{fusion_blocks, FusionTreeBlock, FusionTreePair};
use crate::layout::{subblockstructure, SubblockStructure, SubblockStructureMap};
use crate::sector::{FusionStyle, Sector};
use crate::space::HomSpace;

#[derive(Clone, Debug, PartialEq)]
pub struct AbelianTransformData {
    pub coeff: f64,
    pub dst: SubblockStructure,
    pub src: SubblockStructure,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GenericTransformData {
    pub basis_transform: Array2<f64>,
    pub dst: GenericTransformStructures,
    pub src: GenericTransformStructures,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GenericTransformStructures {
    pub sizes: Vec<usize>,
    pub strides_offsets: Vec<(Vec<usize>, usize)>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum TreeTransformer {
    Abelian(Vec<AbelianTransformData>),
    Generic(Vec<GenericTransformData>),
}

pub fn tree_permuter<I: Sector>(
    src: &HomSpace<I>,
    dst: &HomSpace<I>,
    p_codomain: &[usize],
    p_domain: &[usize],
) -> Result<TreeTransformer> {
    let levels_codomain = (0..src.numout()).collect::<Vec<_>>();
    let levels_domain = (src.numout()..src.numind()).collect::<Vec<_>>();
    tree_braider(
        src,
        dst,
        p_codomain,
        p_domain,
        &levels_codomain,
        &levels_domain,
    )
}

pub fn tree_braider<I: Sector>(
    src: &HomSpace<I>,
    dst: &HomSpace<I>,
    p_codomain: &[usize],
    p_domain: &[usize],
    levels_codomain: &[usize],
    levels_domain: &[usize],
) -> Result<TreeTransformer> {
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

    match I::fusion_style() {
        FusionStyle::UniqueFusion => {
            let data = abelian_tree_transformer(src, dst, |src_pair| {
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
            let data = generic_tree_transformer(src, dst, |src_block| {
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
    p_codomain: &[usize],
    p_domain: &[usize],
) -> Result<TreeTransformer> {
    let expected_dst = src.permute(p_codomain, p_domain)?;
    if &expected_dst != dst {
        return Err(Tensor0Error::Message(
            "incompatible spaces for transposing".to_string(),
        ));
    }

    match I::fusion_style() {
        FusionStyle::UniqueFusion => {
            let data = abelian_tree_transformer(src, dst, |src_pair| {
                transpose_pair(src_pair, p_codomain, p_domain)
            })?;
            Ok(TreeTransformer::Abelian(data))
        }
        FusionStyle::SimpleFusion => {
            let data = generic_tree_transformer(src, dst, |src_block| {
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
    src: &HomSpace<I>,
    dst: &HomSpace<I>,
    transform: F,
) -> Result<Vec<AbelianTransformData>>
where
    I: Sector,
    F: Fn(&FusionTreePair<I>) -> Result<(FusionTreePair<I>, f64)>,
{
    if I::fusion_style() != FusionStyle::UniqueFusion {
        return Err(Tensor0Error::Message(
            "AbelianTreeTransformer requires UniqueFusion sector families".to_string(),
        ));
    }

    let src_subblocks = subblockstructure(src)?;
    let dst_subblocks = subblockstructure(dst)?;
    let mut data: Vec<Option<AbelianTransformData>> =
        Vec::with_capacity(src_subblocks.pairs().len());
    data.resize_with(src_subblocks.pairs().len(), || None);

    for (index, (src_pair, src_subblock)) in src_subblocks
        .pairs()
        .iter()
        .zip(src_subblocks.structures().iter())
        .enumerate()
    {
        let (dst_pair, coeff) = transform(src_pair)?;
        let Some(dst_subblock) = dst_subblocks.get(&dst_pair) else {
            return Err(Tensor0Error::Message(
                "transform destination fusion tree pair was not found".to_string(),
            ));
        };

        data[index] = Some(AbelianTransformData {
            coeff,
            dst: dst_subblock.clone(),
            src: src_subblock.clone(),
        });
    }

    Ok(data
        .into_iter()
        .map(|entry| entry.expect("abelian tree transformer slot is initialized"))
        .collect())
}

fn generic_tree_transformer<I, F>(
    src: &HomSpace<I>,
    dst: &HomSpace<I>,
    transform: F,
) -> Result<Vec<GenericTransformData>>
where
    I: Sector,
    F: Fn(&FusionTreeBlock<I>) -> Result<(FusionTreeBlock<I>, Array2<f64>)>,
{
    let src_subblocks = subblockstructure(src)?;
    let dst_subblocks = subblockstructure(dst)?;
    let src_blocks = fusion_blocks(src)?;
    let mut data: Vec<Option<GenericTransformData>> = Vec::with_capacity(src_blocks.len());
    data.resize_with(src_blocks.len(), || None);

    for (index, src_block) in src_blocks.iter().enumerate() {
        let (dst_block, basis_transform) = transform(src_block)?;
        debug_assert_eq!(basis_transform.nrows(), dst_block.trees().len());
        debug_assert_eq!(basis_transform.ncols(), src_block.trees().len());

        data[index] = Some(GenericTransformData {
            basis_transform,
            dst: repack_transformer_structure(&dst_block, &dst_subblocks),
            src: repack_transformer_structure(src_block, &src_subblocks),
        });
    }

    Ok(data
        .into_iter()
        .map(|entry| entry.expect("generic tree transformer slot is initialized"))
        .collect())
}

fn repack_transformer_structure<I: Sector>(
    block: &FusionTreeBlock<I>,
    subblocks: &SubblockStructureMap<I>,
) -> GenericTransformStructures {
    let first_pair = block
        .trees()
        .first()
        .expect("generic tree transform block is nonempty");
    let first_structure = subblocks
        .get(first_pair)
        .expect("generic tree transform fusion tree pair exists in subblock structure");
    let sizes = first_structure.sizes.clone();
    let mut strides_offsets = Vec::with_capacity(block.trees().len());

    for pair in block.trees() {
        let structure = subblocks
            .get(pair)
            .expect("generic tree transform fusion tree pair exists in subblock structure");
        strides_offsets.push((structure.strides.clone(), structure.offset));
    }

    GenericTransformStructures {
        sizes,
        strides_offsets,
    }
}
