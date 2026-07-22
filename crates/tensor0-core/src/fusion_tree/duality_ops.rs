//! Duality and double-tree repartition operations.
//!
//! This module owns bend/fold, repartition, and transpose primitives at the
//! fusion-tree-pair level. These operations may use pivotal, Frobenius-Schur,
//! and quantum-dimension coefficients, but still return tree-basis linear maps
//! rather than tensor storage instructions.

use std::collections::{hash_map::Entry, HashMap};

use ndarray::Array2;

use crate::error::{Result, Tensor0Error};
use crate::fusion_tree::{FusionTree, FusionTreeBlock, FusionTreePair};
use crate::sector::{FusionStyle, Sector};

use super::auxiliary::linearize_permutation;
use super::basic_ops::{multi_fmove, multi_fmove_inv};

type FMoveTerms<I> = Vec<(FusionTree<I>, f64)>;
type FMoveCache<I> = HashMap<FusionTree<I>, FMoveTerms<I>>;
type FMoveInvCache<I> = HashMap<(I, FusionTree<I>), FMoveTerms<I>>;

pub(crate) fn repartition_pair<I: Sector>(
    src: &FusionTreePair<I>,
    target_numout: usize,
) -> Result<(FusionTreePair<I>, f64)> {
    let numind = src.row.uncoupled().len() + src.col.uncoupled().len();
    if target_numout > numind {
        return Err(Tensor0Error::Message(
            "cannot repartition beyond fusion tree pair arity".to_string(),
        ));
    }

    let mut dst = src.clone();
    let mut coeff = 1.0;

    while dst.row.uncoupled().len() < target_numout {
        let (next_dst, step) = bendleft_pair(&dst)?;
        coeff *= step;
        dst = next_dst;
    }

    while dst.row.uncoupled().len() > target_numout {
        let (next_dst, step) = bendright_pair(&dst)?;
        coeff *= step;
        dst = next_dst;
    }

    Ok((dst, coeff))
}

pub(crate) fn repartition_block<I: Sector>(
    src: &FusionTreeBlock<I>,
    target_numout: usize,
) -> Result<(FusionTreeBlock<I>, Array2<f64>)> {
    if target_numout > src.numind() {
        return Err(Tensor0Error::Message(
            "cannot repartition beyond fusion tree block arity".to_string(),
        ));
    }

    let mut dst = src.clone();
    let mut transform = Array2::eye(src.trees().len());

    while dst.numout() < target_numout {
        let (next_dst, step) = bendleft_block(&dst)?;
        transform = step.dot(&transform);
        dst = next_dst;
    }

    while dst.numout() > target_numout {
        let (next_dst, step) = bendright_block(&dst)?;
        transform = step.dot(&transform);
        dst = next_dst;
    }

    Ok((dst, transform))
}

/// Applies a transpose whose cyclic-planar permutation was validated by the caller.
pub(crate) fn transpose_pair<I: Sector>(
    src: &FusionTreePair<I>,
    p_codomain: &[usize],
    p_domain: &[usize],
) -> Result<(FusionTreePair<I>, f64)> {
    if I::fusion_style() != FusionStyle::UniqueFusion {
        return Err(Tensor0Error::Message(
            "transpose pair requires UniqueFusion".to_string(),
        ));
    }

    let permutation = linearize_permutation(
        p_codomain,
        p_domain,
        src.row.uncoupled().len(),
        src.col.uncoupled().len(),
    )?;

    let (mut dst, mut coeff) = repartition_pair(src, p_codomain.len())?;
    if permutation.is_empty() {
        return Ok((dst, coeff));
    }

    let mut first_pos = permutation
        .iter()
        .position(|index| *index == 0)
        .expect("valid cyclic permutation contains source index zero");
    if first_pos == 0 {
        return Ok((dst, coeff));
    }

    let half = permutation.len() / 2;
    while first_pos > 0 && first_pos < half {
        let (next_dst, step) = cycleanticlockwise_pair(&dst)?;
        coeff *= step;
        dst = next_dst;
        first_pos -= 1;
    }
    while first_pos >= half {
        let (next_dst, step) = cycleclockwise_pair(&dst)?;
        coeff *= step;
        dst = next_dst;
        first_pos = (first_pos + 1) % permutation.len();
        if first_pos == 0 {
            break;
        }
    }

    Ok((dst, coeff))
}

/// Applies a transpose whose cyclic-planar permutation was validated by the caller.
pub(crate) fn transpose_block<I: Sector>(
    src: &FusionTreeBlock<I>,
    p_codomain: &[usize],
    p_domain: &[usize],
) -> Result<(FusionTreeBlock<I>, Array2<f64>)> {
    let permutation = linearize_permutation(p_codomain, p_domain, src.numout(), src.numin())?;

    let (mut dst, mut transform) = repartition_block(src, p_codomain.len())?;
    if permutation.is_empty() {
        return Ok((dst, transform));
    }

    let mut first_pos = permutation
        .iter()
        .position(|index| *index == 0)
        .expect("valid cyclic permutation contains source index zero");
    if first_pos == 0 {
        return Ok((dst, transform));
    }

    let half = permutation.len() / 2;
    while first_pos > 0 && first_pos < half {
        let (next_dst, step) = cycleanticlockwise_block(&dst)?;
        transform = step.dot(&transform);
        dst = next_dst;
        first_pos -= 1;
    }
    while first_pos >= half {
        let (next_dst, step) = cycleclockwise_block(&dst)?;
        transform = step.dot(&transform);
        dst = next_dst;
        first_pos = (first_pos + 1) % permutation.len();
        if first_pos == 0 {
            break;
        }
    }

    Ok((dst, transform))
}

fn cycleclockwise_pair<I: Sector>(src: &FusionTreePair<I>) -> Result<(FusionTreePair<I>, f64)> {
    if !src.row.uncoupled().is_empty() {
        let (intermediate, fold_coeff) = foldright_pair(src)?;
        let (dst, bend_coeff) = bendleft_pair(&intermediate)?;
        Ok((dst, bend_coeff * fold_coeff))
    } else {
        let (intermediate, bend_coeff) = bendleft_pair(src)?;
        let (dst, fold_coeff) = foldright_pair(&intermediate)?;
        Ok((dst, fold_coeff * bend_coeff))
    }
}

fn cycleclockwise_block<I: Sector>(
    src: &FusionTreeBlock<I>,
) -> Result<(FusionTreeBlock<I>, Array2<f64>)> {
    if src.numout() > 0 {
        let (intermediate, fold_transform) = foldright_block(src)?;
        let (dst, bend_transform) = bendleft_block(&intermediate)?;
        Ok((dst, bend_transform.dot(&fold_transform)))
    } else {
        let (intermediate, bend_transform) = bendleft_block(src)?;
        let (dst, fold_transform) = foldright_block(&intermediate)?;
        Ok((dst, fold_transform.dot(&bend_transform)))
    }
}

fn cycleanticlockwise_pair<I: Sector>(src: &FusionTreePair<I>) -> Result<(FusionTreePair<I>, f64)> {
    if !src.col.uncoupled().is_empty() {
        let (intermediate, fold_coeff) = foldleft_pair(src)?;
        let (dst, bend_coeff) = bendright_pair(&intermediate)?;
        Ok((dst, bend_coeff * fold_coeff))
    } else {
        let (intermediate, bend_coeff) = bendright_pair(src)?;
        let (dst, fold_coeff) = foldleft_pair(&intermediate)?;
        Ok((dst, fold_coeff * bend_coeff))
    }
}

fn cycleanticlockwise_block<I: Sector>(
    src: &FusionTreeBlock<I>,
) -> Result<(FusionTreeBlock<I>, Array2<f64>)> {
    if src.numin() > 0 {
        let (intermediate, fold_transform) = foldleft_block(src)?;
        let (dst, bend_transform) = bendright_block(&intermediate)?;
        Ok((dst, bend_transform.dot(&fold_transform)))
    } else {
        let (intermediate, bend_transform) = bendright_block(src)?;
        let (dst, fold_transform) = foldleft_block(&intermediate)?;
        Ok((dst, fold_transform.dot(&bend_transform)))
    }
}

fn foldright_pair<I: Sector>(pair: &FusionTreePair<I>) -> Result<(FusionTreePair<I>, f64)> {
    if I::fusion_style() != FusionStyle::UniqueFusion {
        return Err(Tensor0Error::Message(
            "foldright pair requires UniqueFusion".to_string(),
        ));
    }

    let row = &pair.row;
    let col = &pair.col;
    debug_assert!(!row.uncoupled().is_empty());
    let a = row.uncoupled()[0].clone();
    let frobenius_schur = I::frobenius_schur_phase(&a)?;
    let is_dual_a = row.is_dual()[0];

    let mut row_terms = multi_fmove(row)?;
    if row_terms.len() != 1 {
        return Err(Tensor0Error::Message(
            "foldright multi_Fmove requires a unique fusion tree term".to_string(),
        ));
    }
    let (row_prime, row_coeff) = row_terms.remove(0);
    let b = row_prime.coupled().clone();
    let c = row.coupled().clone();
    let a_symbol = I::a_symbol(&a, &b, &c)?;
    let mut col_terms = multi_fmove_inv(&a.dual(), &b, col, !is_dual_a)?;
    if col_terms.len() != 1 {
        return Err(Tensor0Error::Message(
            "foldright inverse multi_Fmove requires a unique fusion tree term".to_string(),
        ));
    }
    let (col_prime, col_coeff) = col_terms.remove(0);

    let scale = ((c.quantum_dim() as f64) / (b.quantum_dim() as f64)).sqrt();
    // GenericFusion/complex symbols must restore TensorKit's conjugation here.
    let mut coeff = scale * row_coeff * a_symbol * col_coeff;
    if is_dual_a {
        coeff *= frobenius_schur;
    }

    Ok((
        FusionTreePair {
            row: row_prime,
            col: col_prime,
        },
        coeff,
    ))
}

fn foldright_block<I: Sector>(
    src: &FusionTreeBlock<I>,
) -> Result<(FusionTreeBlock<I>, Array2<f64>)> {
    if src.numout() == 0 {
        return Err(Tensor0Error::Message(
            "foldright source requires an outgoing leg".to_string(),
        ));
    }

    let row_uncoupled_dst = src.row_uncoupled()[1..].to_vec();
    let row_is_dual_dst = src.row_is_dual()[1..].to_vec();
    let mut col_uncoupled_dst = Vec::with_capacity(src.numin() + 1);
    col_uncoupled_dst.push(src.row_uncoupled()[0].dual());
    col_uncoupled_dst.extend_from_slice(src.col_uncoupled());
    let mut col_is_dual_dst = Vec::with_capacity(src.numin() + 1);
    col_is_dual_dst.push(!src.row_is_dual()[0]);
    col_is_dual_dst.extend_from_slice(src.col_is_dual());

    let dst = FusionTreeBlock::new(
        row_uncoupled_dst,
        row_is_dual_dst,
        col_uncoupled_dst,
        col_is_dual_dst,
    )?;
    let dst_index = dst.tree_index();
    let mut transform = Array2::zeros((dst.trees().len(), src.trees().len()));
    if I::fusion_style() == FusionStyle::UniqueFusion {
        for (source_index, pair) in src.trees().iter().enumerate() {
            let (target_pair, coeff) = foldright_pair(pair)?;
            let Some(target_index) = dst_index.get(&target_pair) else {
                return Err(Tensor0Error::Message(
                    "foldright destination fusion tree pair was not found".to_string(),
                ));
            };
            transform[[target_index, source_index]] = coeff;
        }
        return Ok((dst, transform));
    }

    let a = src.row_uncoupled()[0].clone();
    let frobenius_schur = I::frobenius_schur_phase(&a)?;
    let is_dual_a = src.row_is_dual()[0];
    let mut row_fmove_cache: FMoveCache<I> = HashMap::new();
    let mut col_fmove_inv_cache: FMoveInvCache<I> = HashMap::new();
    let mut a_symbol_cache: HashMap<(I, I), f64> = HashMap::new();

    for (source_index, pair) in src.trees().iter().enumerate() {
        let row = &pair.row;
        let col = &pair.col;
        let row_terms = match row_fmove_cache.entry(row.clone()) {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => entry.insert(multi_fmove(row)?),
        };

        for (row_prime, row_coeff) in row_terms.iter() {
            let b = row_prime.coupled().clone();
            let c = row.coupled().clone();
            let a_key = (b.clone(), c.clone());
            let a_symbol = match a_symbol_cache.entry(a_key) {
                Entry::Occupied(entry) => *entry.get(),
                Entry::Vacant(entry) => *entry.insert(I::a_symbol(&a, &b, &c)?),
            };

            let inv_key = (b.clone(), col.clone());
            let col_terms = match col_fmove_inv_cache.entry(inv_key) {
                Entry::Occupied(entry) => entry.into_mut(),
                Entry::Vacant(entry) => {
                    entry.insert(multi_fmove_inv(&a.dual(), &b, col, !is_dual_a)?)
                }
            };

            let scale = ((c.quantum_dim() as f64) / (b.quantum_dim() as f64)).sqrt();
            for (col_prime, col_coeff) in col_terms.iter() {
                // GenericFusion/complex symbols must restore TensorKit's conjugation here.
                let mut coeff = scale * row_coeff * a_symbol * col_coeff;
                if is_dual_a {
                    coeff *= frobenius_schur;
                }
                if coeff == 0.0 {
                    continue;
                }

                let target_pair = FusionTreePair {
                    row: row_prime.clone(),
                    col: col_prime.clone(),
                };
                let Some(target_index) = dst_index.get(&target_pair) else {
                    return Err(Tensor0Error::Message(
                        "foldright destination fusion tree pair was not found".to_string(),
                    ));
                };
                transform[[target_index, source_index]] += coeff;
            }
        }
    }

    Ok((dst, transform))
}

fn bendright_pair<I: Sector>(pair: &FusionTreePair<I>) -> Result<(FusionTreePair<I>, f64)> {
    let (target_pair, a, b, c) = bendright_target(pair)?;
    let row = &pair.row;
    let mut scale = ((c.quantum_dim() as f64) / (a.quantum_dim() as f64)).sqrt();
    if row.is_dual()[row.is_dual().len() - 1] {
        // GenericFusion/complex symbols must restore TensorKit's conjugation here.
        scale *= I::frobenius_schur_phase(&b.dual())?;
    }
    let coeff = scale * I::b_symbol(&a, &b, &c)?;
    Ok((target_pair, coeff))
}

fn bendright_target<I: Sector>(pair: &FusionTreePair<I>) -> Result<(FusionTreePair<I>, I, I, I)> {
    let row = &pair.row;
    let col = &pair.col;
    let row_arity = row.uncoupled().len();
    let col_arity = col.uncoupled().len();

    let a = match row_arity {
        0 => unreachable!("bendright pair has at least one outgoing leg"),
        1 => I::unit(),
        2 => row.uncoupled()[0].clone(),
        _ => row.innerlines()[row_arity - 3].clone(),
    };
    let b = row.uncoupled()[row_arity - 1].clone();
    let c = row.coupled().clone();

    let row_uncoupled = row.uncoupled()[..row_arity - 1].to_vec();
    let row_is_dual = row.is_dual()[..row_arity - 1].to_vec();
    let row_innerlines = if row_arity > 2 {
        row.innerlines()[..row.innerlines().len() - 1].to_vec()
    } else {
        vec![]
    };
    let row_vertices = if row_arity > 1 {
        row.vertices()[..row.vertices().len() - 1].to_vec()
    } else {
        vec![]
    };
    let row_prime = FusionTree::new(
        row_uncoupled,
        a.clone(),
        row_is_dual,
        row_innerlines,
        row_vertices,
    )?;

    let mut col_uncoupled = col.uncoupled().to_vec();
    col_uncoupled.push(b.dual());
    let mut col_is_dual = col.is_dual().to_vec();
    col_is_dual.push(!row.is_dual()[row_arity - 1]);
    let mut col_innerlines = if col_arity > 1 {
        col.innerlines().to_vec()
    } else {
        vec![]
    };
    if col_arity > 1 {
        col_innerlines.push(c.clone());
    }
    let mut col_vertices = if col_arity > 0 {
        col.vertices().to_vec()
    } else {
        vec![]
    };
    if col_arity > 0 {
        col_vertices.push(0);
    }
    let col_prime = FusionTree::new(
        col_uncoupled,
        a.clone(),
        col_is_dual,
        col_innerlines,
        col_vertices,
    )?;

    Ok((
        FusionTreePair {
            row: row_prime,
            col: col_prime,
        },
        a,
        b,
        c,
    ))
}

fn bendright_block<I: Sector>(
    src: &FusionTreeBlock<I>,
) -> Result<(FusionTreeBlock<I>, Array2<f64>)> {
    let row_arity = src.numout();
    debug_assert!(row_arity > 0);
    let b = src.row_uncoupled()[row_arity - 1].clone();
    let is_dual_b = src.row_is_dual()[row_arity - 1];
    let row_uncoupled_dst = src.row_uncoupled()[..row_arity - 1].to_vec();
    let row_is_dual_dst = src.row_is_dual()[..row_arity - 1].to_vec();
    let mut col_uncoupled_dst = src.col_uncoupled().to_vec();
    col_uncoupled_dst.push(b.dual());
    let mut col_is_dual_dst = src.col_is_dual().to_vec();
    col_is_dual_dst.push(!is_dual_b);

    let dst = FusionTreeBlock::new(
        row_uncoupled_dst,
        row_is_dual_dst,
        col_uncoupled_dst,
        col_is_dual_dst,
    )?;
    let dst_index = dst.tree_index();
    let mut transform = Array2::zeros((dst.trees().len(), src.trees().len()));

    for (source_index, pair) in src.trees().iter().enumerate() {
        let (target_pair, coeff) = bendright_pair(pair)?;
        if coeff == 0.0 {
            continue;
        }
        let Some(target_index) = dst_index.get(&target_pair) else {
            return Err(Tensor0Error::Message(
                "bendright destination fusion tree pair was not found".to_string(),
            ));
        };
        transform[[target_index, source_index]] = coeff;
    }

    Ok((dst, transform))
}

fn foldleft_pair<I: Sector>(pair: &FusionTreePair<I>) -> Result<(FusionTreePair<I>, f64)> {
    if I::fusion_style() != FusionStyle::UniqueFusion {
        return Err(Tensor0Error::Message(
            "foldleft pair requires UniqueFusion".to_string(),
        ));
    }

    let swapped_pair = FusionTreePair {
        row: pair.col.clone(),
        col: pair.row.clone(),
    };
    let (swapped_prime, coeff) = foldright_pair(&swapped_pair)?;
    Ok((
        FusionTreePair {
            row: swapped_prime.col,
            col: swapped_prime.row,
        },
        // GenericFusion/complex symbols must restore TensorKit's conjugation here.
        coeff,
    ))
}

fn foldleft_block<I: Sector>(
    src: &FusionTreeBlock<I>,
) -> Result<(FusionTreeBlock<I>, Array2<f64>)> {
    if src.numin() == 0 {
        return Err(Tensor0Error::Message(
            "foldleft source requires an incoming leg".to_string(),
        ));
    }

    let mut row_uncoupled_dst = Vec::with_capacity(src.numout() + 1);
    row_uncoupled_dst.push(src.col_uncoupled()[0].dual());
    row_uncoupled_dst.extend_from_slice(src.row_uncoupled());
    let mut row_is_dual_dst = Vec::with_capacity(src.numout() + 1);
    row_is_dual_dst.push(!src.col_is_dual()[0]);
    row_is_dual_dst.extend_from_slice(src.row_is_dual());
    let col_uncoupled_dst = src.col_uncoupled()[1..].to_vec();
    let col_is_dual_dst = src.col_is_dual()[1..].to_vec();

    let dst = FusionTreeBlock::new(
        row_uncoupled_dst,
        row_is_dual_dst,
        col_uncoupled_dst,
        col_is_dual_dst,
    )?;
    let dst_index = dst.tree_index();
    let mut transform = Array2::zeros((dst.trees().len(), src.trees().len()));
    if I::fusion_style() == FusionStyle::UniqueFusion {
        for (source_index, pair) in src.trees().iter().enumerate() {
            let (target_pair, coeff) = foldleft_pair(pair)?;
            let Some(target_index) = dst_index.get(&target_pair) else {
                return Err(Tensor0Error::Message(
                    "foldleft destination fusion tree pair was not found".to_string(),
                ));
            };
            transform[[target_index, source_index]] = coeff;
        }
        return Ok((dst, transform));
    }

    let a = src.col_uncoupled()[0].clone();
    let frobenius_schur = I::frobenius_schur_phase(&a)?;
    let is_dual_a = src.col_is_dual()[0];
    let mut col_fmove_cache: FMoveCache<I> = HashMap::new();
    let mut row_fmove_inv_cache: FMoveInvCache<I> = HashMap::new();
    let mut a_symbol_cache: HashMap<(I, I), f64> = HashMap::new();

    for (source_index, pair) in src.trees().iter().enumerate() {
        let row = &pair.row;
        let col = &pair.col;
        let col_terms = match col_fmove_cache.entry(col.clone()) {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => entry.insert(multi_fmove(col)?),
        };

        for (col_prime, col_coeff) in col_terms.iter() {
            let b = col_prime.coupled().clone();
            let c = col.coupled().clone();
            let a_key = (b.clone(), c.clone());
            let a_symbol = match a_symbol_cache.entry(a_key) {
                Entry::Occupied(entry) => *entry.get(),
                Entry::Vacant(entry) => *entry.insert(I::a_symbol(&a, &b, &c)?),
            };

            let inv_key = (b.clone(), row.clone());
            let row_terms = match row_fmove_inv_cache.entry(inv_key) {
                Entry::Occupied(entry) => entry.into_mut(),
                Entry::Vacant(entry) => {
                    entry.insert(multi_fmove_inv(&a.dual(), &b, row, !is_dual_a)?)
                }
            };

            let scale = ((c.quantum_dim() as f64) / (b.quantum_dim() as f64)).sqrt();
            for (row_prime, row_coeff) in row_terms.iter() {
                // GenericFusion/complex symbols must restore TensorKit's conjugation here.
                let mut coeff = scale * row_coeff * a_symbol * col_coeff;
                if is_dual_a {
                    coeff *= frobenius_schur;
                }
                if coeff == 0.0 {
                    continue;
                }

                let target_pair = FusionTreePair {
                    row: row_prime.clone(),
                    col: col_prime.clone(),
                };
                let Some(target_index) = dst_index.get(&target_pair) else {
                    return Err(Tensor0Error::Message(
                        "foldleft destination fusion tree pair was not found".to_string(),
                    ));
                };
                transform[[target_index, source_index]] += coeff;
            }
        }
    }

    Ok((dst, transform))
}

fn bendleft_pair<I: Sector>(pair: &FusionTreePair<I>) -> Result<(FusionTreePair<I>, f64)> {
    let swapped_pair = FusionTreePair {
        row: pair.col.clone(),
        col: pair.row.clone(),
    };
    let (swapped_prime, coeff) = bendright_pair(&swapped_pair)?;
    Ok((
        FusionTreePair {
            row: swapped_prime.col,
            col: swapped_prime.row,
        },
        // GenericFusion/complex symbols must restore TensorKit's conjugation here.
        coeff,
    ))
}

fn bendleft_block<I: Sector>(
    src: &FusionTreeBlock<I>,
) -> Result<(FusionTreeBlock<I>, Array2<f64>)> {
    let col_arity = src.numin();
    debug_assert!(col_arity > 0);
    let b = src.col_uncoupled()[col_arity - 1].clone();
    let is_dual_b = src.col_is_dual()[col_arity - 1];
    let col_uncoupled_dst = src.col_uncoupled()[..col_arity - 1].to_vec();
    let col_is_dual_dst = src.col_is_dual()[..col_arity - 1].to_vec();
    let mut row_uncoupled_dst = src.row_uncoupled().to_vec();
    row_uncoupled_dst.push(b.dual());
    let mut row_is_dual_dst = src.row_is_dual().to_vec();
    row_is_dual_dst.push(!is_dual_b);

    let dst = FusionTreeBlock::new(
        row_uncoupled_dst,
        row_is_dual_dst,
        col_uncoupled_dst,
        col_is_dual_dst,
    )?;
    let dst_index = dst.tree_index();
    let mut transform = Array2::zeros((dst.trees().len(), src.trees().len()));

    for (source_index, pair) in src.trees().iter().enumerate() {
        let (target_pair, coeff) = bendleft_pair(pair)?;
        if coeff == 0.0 {
            continue;
        }
        let Some(target_index) = dst_index.get(&target_pair) else {
            return Err(Tensor0Error::Message(
                "bendleft destination fusion tree pair was not found".to_string(),
            ));
        };
        transform[[target_index, source_index]] = coeff;
    }

    Ok((dst, transform))
}

#[cfg(test)]
mod tests {
    use ndarray::Array2;

    use crate::fusion_tree::FusionTreeBlock;
    use crate::sector::SU2Irrep;

    use super::{repartition_block, transpose_block};

    fn su2(spin2: i64) -> SU2Irrep {
        SU2Irrep::spin2(spin2).unwrap()
    }

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1.0e-12,
            "actual={actual}, expected={expected}",
        );
    }

    fn assert_identity(matrix: &Array2<f64>) {
        assert_eq!(matrix.nrows(), matrix.ncols());
        for row in 0..matrix.nrows() {
            for col in 0..matrix.ncols() {
                let expected = if row == col { 1.0 } else { 0.0 };
                assert_close(matrix[[row, col]], expected);
            }
        }
    }

    fn two_out_two_in_half_block() -> FusionTreeBlock<SU2Irrep> {
        let half = su2(1);
        FusionTreeBlock::new(
            vec![half, half],
            vec![false, false],
            vec![half, half],
            vec![false, false],
        )
        .unwrap()
    }

    fn inverse_visible_permutation(
        p_codomain: &[usize],
        p_domain: &[usize],
        source_numout: usize,
    ) -> (Vec<usize>, Vec<usize>) {
        let target_order = p_codomain
            .iter()
            .chain(p_domain.iter())
            .copied()
            .collect::<Vec<_>>();
        let mut inverse = vec![0; target_order.len()];
        for (target_index, source_index) in target_order.into_iter().enumerate() {
            inverse[source_index] = target_index;
        }

        (
            inverse[..source_numout].to_vec(),
            inverse[source_numout..].to_vec(),
        )
    }

    #[test]
    fn repartition_block_and_inverse_are_identity_on_su2_basis() {
        let src = two_out_two_in_half_block();

        let (dst, transform) = repartition_block(&src, 1).unwrap();
        let (roundtrip, inverse_transform) = repartition_block(&dst, src.numout()).unwrap();

        assert_eq!(dst.numout(), 1);
        assert_eq!(dst.numin(), 3);
        assert_eq!(roundtrip, src);
        assert_identity(&inverse_transform.dot(&transform));
    }

    #[test]
    fn transpose_block_and_inverse_are_identity_for_cyclic_shift() {
        let src = two_out_two_in_half_block();
        let p_codomain = [1, 3];
        let p_domain = [0, 2];

        let (dst, transform) = transpose_block(&src, &p_codomain, &p_domain).unwrap();
        let (inverse_codomain, inverse_domain) =
            inverse_visible_permutation(&p_codomain, &p_domain, src.numout());
        let (roundtrip, inverse_transform) =
            transpose_block(&dst, &inverse_codomain, &inverse_domain).unwrap();

        assert_eq!(roundtrip, src);
        assert_identity(&inverse_transform.dot(&transform));
    }
}
