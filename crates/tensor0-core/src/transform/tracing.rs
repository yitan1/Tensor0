//! Fused permutation and rank-reducing trace metadata.
//!
//! Source and destination indices refer directly to their supplied
//! [`SectorStructure`] canonical pair orders. Permutation basis changes and
//! canonical trace coefficients are combined without materializing an
//! intermediate canonical tensor.

use std::collections::BTreeSet;

use ndarray::Array2;

use crate::error::{Result, Tensor0Error};
use crate::fusion_tree::basic_ops::split;
use crate::fusion_tree::{FusionTree, FusionTreePair};
use crate::layout::SectorStructure;
use crate::sector::{BraidingStyle, FusionStyle, Sector};
use crate::space::HomSpace;

use super::tree_transformers::{AbelianTransformData, GenericTransformData, TreeTransformer};

/// Combines cached source-to-canonical basis metadata with canonical trace metadata.
pub fn trace_transformer<I: Sector>(
    canonical_src: &HomSpace<I>,
    dst: &HomSpace<I>,
    canonical_structure: &SectorStructure<I>,
    dst_structure: &SectorStructure<I>,
    basis_transformer: &TreeTransformer,
) -> Result<TreeTransformer> {
    validate_trace_transformer_inputs(canonical_src, dst, canonical_structure, dst_structure)?;

    match (I::fusion_style(), basis_transformer) {
        (FusionStyle::UniqueFusion, TreeTransformer::Abelian(data)) => {
            Ok(TreeTransformer::Abelian(trace_abelian_transformer(
                data,
                canonical_structure,
                dst,
                dst_structure,
            )?))
        }
        (FusionStyle::SimpleFusion, TreeTransformer::Generic(data)) => {
            Ok(TreeTransformer::Generic(trace_generic_transformer(
                data,
                canonical_structure,
                dst,
                dst_structure,
            )?))
        }
        (FusionStyle::UniqueFusion | FusionStyle::SimpleFusion, _) => Err(Tensor0Error::Message(
            "basis transformer kind does not match the sector fusion style".to_string(),
        )),
        (FusionStyle::GenericFusion, _) => Err(Tensor0Error::Message(
            "trace transformer does not support GenericFusion sector families".to_string(),
        )),
    }
}

fn trace_abelian_transformer<I: Sector>(
    basis_data: &[AbelianTransformData],
    canonical_structure: &SectorStructure<I>,
    dst: &HomSpace<I>,
    dst_structure: &SectorStructure<I>,
) -> Result<Vec<AbelianTransformData>> {
    let mut data = Vec::with_capacity(basis_data.len());
    for entry in basis_data {
        let canonical_pair = canonical_structure
            .fusiontree_pair_at(entry.dst)
            .ok_or_else(|| {
                Tensor0Error::Message(
                    "basis transformer destination index exceeds canonical sectorstructure"
                        .to_string(),
                )
            })?;
        let Some((dst_index, coefficient)) =
            trace_canonical_pair(canonical_pair, dst, dst_structure)?
        else {
            continue;
        };
        data.push(AbelianTransformData {
            src: entry.src,
            dst: dst_index,
            coeff: entry.coeff * coefficient,
        });
    }
    Ok(data)
}

fn trace_generic_transformer<I: Sector>(
    basis_data: &[GenericTransformData],
    canonical_structure: &SectorStructure<I>,
    dst: &HomSpace<I>,
    dst_structure: &SectorStructure<I>,
) -> Result<Vec<GenericTransformData>> {
    let mut groups = Vec::with_capacity(basis_data.len());
    for basis_group in basis_data {
        if basis_group.transform.nrows() != basis_group.dst_indices.len()
            || basis_group.transform.ncols() != basis_group.src_indices.len()
        {
            return Err(Tensor0Error::Message(
                "basis transform dimensions do not match its canonical indices".to_string(),
            ));
        }

        let mut canonical_rows = Vec::new();
        let mut destination_indices = BTreeSet::new();
        for (row, &canonical_index) in basis_group.dst_indices.iter().enumerate() {
            let canonical_pair = canonical_structure
                .fusiontree_pair_at(canonical_index)
                .ok_or_else(|| {
                    Tensor0Error::Message(
                        "basis transformer destination index exceeds canonical sectorstructure"
                            .to_string(),
                    )
                })?;
            let Some((dst_index, coefficient)) =
                trace_canonical_pair(canonical_pair, dst, dst_structure)?
            else {
                continue;
            };
            canonical_rows.push((row, dst_index, coefficient));
            destination_indices.insert(dst_index);
        }

        let mut dst_indices = destination_indices.into_iter().collect::<Vec<_>>();
        let mut transform = Array2::zeros((dst_indices.len(), basis_group.src_indices.len()));
        for (canonical_row, dst_index, coefficient) in canonical_rows {
            let destination_row = dst_indices
                .binary_search(&dst_index)
                .expect("trace destination index was collected");
            for source_column in 0..basis_group.src_indices.len() {
                transform[[destination_row, source_column]] +=
                    coefficient * basis_group.transform[[canonical_row, source_column]];
            }
        }

        let nonzero_rows = (0..transform.nrows())
            .filter(|row| transform.row(*row).iter().any(|value| *value != 0.0))
            .collect::<Vec<_>>();
        if nonzero_rows.is_empty() {
            continue;
        }
        if nonzero_rows.len() != transform.nrows() {
            let mut filtered = Array2::zeros((nonzero_rows.len(), transform.ncols()));
            for (new_row, old_row) in nonzero_rows.iter().enumerate() {
                filtered.row_mut(new_row).assign(&transform.row(*old_row));
            }
            dst_indices = nonzero_rows.iter().map(|row| dst_indices[*row]).collect();
            transform = filtered;
        }

        groups.push(GenericTransformData {
            transform,
            src_indices: basis_group.src_indices.clone(),
            dst_indices,
        });
    }
    Ok(groups)
}

fn validate_trace_transformer_inputs<I: Sector>(
    canonical_src: &HomSpace<I>,
    dst: &HomSpace<I>,
    canonical_structure: &SectorStructure<I>,
    dst_structure: &SectorStructure<I>,
) -> Result<()> {
    if !matches!(
        I::braiding_style(),
        BraidingStyle::Bosonic | BraidingStyle::Fermionic
    ) {
        return Err(Tensor0Error::Message(
            "trace transformer requires symmetric braiding".to_string(),
        ));
    }
    if !canonical_structure.matches_space(canonical_src) {
        return Err(Tensor0Error::Message(
            "canonical sectorstructure does not match canonical HomSpace sector structure"
                .to_string(),
        ));
    }
    if !dst_structure.matches_space(dst) {
        return Err(Tensor0Error::Message(
            "destination sectorstructure does not match destination HomSpace sector structure"
                .to_string(),
        ));
    }

    validate_trace_spaces(canonical_src, dst)
}

fn validate_trace_spaces<I: Sector>(src: &HomSpace<I>, dst: &HomSpace<I>) -> Result<()> {
    if dst.numout() > src.numout() || dst.numin() > src.numin() {
        return Err(Tensor0Error::Message(
            "trace destination arity exceeds source arity".to_string(),
        ));
    }

    let src_open_out = &src.codomain().factors()[..dst.numout()];
    let src_open_in = &src.domain().factors()[..dst.numin()];
    if src_open_out != dst.codomain().factors() || src_open_in != dst.domain().factors() {
        return Err(Tensor0Error::Message(
            "trace destination must equal the source open-factor prefixes".to_string(),
        ));
    }

    let src_trace_out = &src.codomain().factors()[dst.numout()..];
    let src_trace_in = &src.domain().factors()[dst.numin()..];
    if src_trace_out != src_trace_in {
        return Err(Tensor0Error::Message(
            "trace codomain and domain tail spaces must match".to_string(),
        ));
    }
    Ok(())
}

fn trace_canonical_pair<I: Sector>(
    pair: &FusionTreePair<I>,
    dst: &HomSpace<I>,
    dst_structure: &SectorStructure<I>,
) -> Result<Option<(usize, f64)>> {
    let (dst_row, row_tail) = split(&pair.row, dst.numout())?;
    let (dst_col, col_tail) = split(&pair.col, dst.numin())?;
    if row_tail != col_tail {
        return Ok(None);
    }

    let dst_pair = FusionTreePair {
        row: dst_row,
        col: dst_col,
    };
    let dst_index = dst_structure
        .fusiontree_pair_index(&dst_pair)
        .ok_or_else(|| {
            Tensor0Error::Message("trace destination fusion tree pair was not found".to_string())
        })?;
    Ok(Some((dst_index, trace_coefficient(&row_tail))))
}

fn trace_coefficient<I: Sector>(tail: &FusionTree<I>) -> f64 {
    let first = &tail.uncoupled()[0];
    let mut coefficient = tail.coupled().quantum_dim() as f64 / first.quantum_dim() as f64;
    for (sector, is_dual) in tail.uncoupled().iter().zip(tail.is_dual()).skip(1) {
        if !is_dual {
            coefficient *= sector.twist();
        }
    }
    coefficient
}
