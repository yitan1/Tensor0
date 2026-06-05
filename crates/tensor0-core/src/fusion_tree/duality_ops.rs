//! Duality and double-tree repartition operations.
//!
//! Future code in this module should own bend/fold, repartition, transpose, and
//! trace primitives at the fusion-tree-pair level. These operations may use
//! pivotal, Frobenius-Schur, and quantum-dimension coefficients, but should
//! still return tree-basis linear maps rather than tensor storage instructions.

use std::collections::{hash_map::Entry, HashMap};

use ndarray::Array2;

use crate::error::{Result, Tensor0Error};
use crate::fusion_tree::{FusionTree, FusionTreeBlock, FusionTreePair};
use crate::sector::{FusionStyle, Sector};

use super::basic_ops::{multi_fmove, multi_fmove_inv};
use super::permutation_ops::{is_cyclic_permutation, linearize_permutation};

type FMoveTerms<I> = Vec<(FusionTree<I>, f64)>;
type FMoveCache<I> = HashMap<FusionTree<I>, FMoveTerms<I>>;
type FMoveInvCache<I> = HashMap<(I, FusionTree<I>), FMoveTerms<I>>;

pub(crate) fn repartition_pair<I: Sector>(
    src: &FusionTreePair<I>,
    target_numout: usize,
) -> Result<(FusionTreePair<I>, f64)> {
    let numind = src.row.uncoupled.len() + src.col.uncoupled.len();
    if target_numout > numind {
        return Err(Tensor0Error::Message(
            "cannot repartition beyond fusion tree pair arity".to_string(),
        ));
    }

    let mut dst = src.clone();
    let mut coeff = 1.0;

    while dst.row.uncoupled.len() < target_numout {
        let (next_dst, step) = bendleft_pair(&dst)?;
        coeff *= step;
        dst = next_dst;
    }

    while dst.row.uncoupled.len() > target_numout {
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
        src.row.uncoupled.len(),
        src.col.uncoupled.len(),
    )?;
    if !is_cyclic_permutation(&permutation) {
        return Err(Tensor0Error::Message(
            "fusion tree transpose requires a cyclic planar permutation".to_string(),
        ));
    }

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

pub(crate) fn transpose_block<I: Sector>(
    src: &FusionTreeBlock<I>,
    p_codomain: &[usize],
    p_domain: &[usize],
) -> Result<(FusionTreeBlock<I>, Array2<f64>)> {
    let permutation = linearize_permutation(p_codomain, p_domain, src.numout(), src.numin())?;
    if !is_cyclic_permutation(&permutation) {
        return Err(Tensor0Error::Message(
            "fusion tree transpose requires a cyclic planar permutation".to_string(),
        ));
    }

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
    if !src.row.uncoupled.is_empty() {
        let (tmp, first) = foldright_pair(src)?;
        let (dst, second) = bendleft_pair(&tmp)?;
        Ok((dst, second * first))
    } else {
        let (tmp, first) = bendleft_pair(src)?;
        let (dst, second) = foldright_pair(&tmp)?;
        Ok((dst, second * first))
    }
}

fn cycleclockwise_block<I: Sector>(
    src: &FusionTreeBlock<I>,
) -> Result<(FusionTreeBlock<I>, Array2<f64>)> {
    if src.numout() > 0 {
        let (tmp, first) = foldright_block(src)?;
        let (dst, second) = bendleft_block(&tmp)?;
        Ok((dst, second.dot(&first)))
    } else {
        let (tmp, first) = bendleft_block(src)?;
        let (dst, second) = foldright_block(&tmp)?;
        Ok((dst, second.dot(&first)))
    }
}

fn cycleanticlockwise_pair<I: Sector>(src: &FusionTreePair<I>) -> Result<(FusionTreePair<I>, f64)> {
    if !src.col.uncoupled.is_empty() {
        let (tmp, first) = foldleft_pair(src)?;
        let (dst, second) = bendright_pair(&tmp)?;
        Ok((dst, second * first))
    } else {
        let (tmp, first) = bendright_pair(src)?;
        let (dst, second) = foldleft_pair(&tmp)?;
        Ok((dst, second * first))
    }
}

fn cycleanticlockwise_block<I: Sector>(
    src: &FusionTreeBlock<I>,
) -> Result<(FusionTreeBlock<I>, Array2<f64>)> {
    if src.numin() > 0 {
        let (tmp, first) = foldleft_block(src)?;
        let (dst, second) = bendright_block(&tmp)?;
        Ok((dst, second.dot(&first)))
    } else {
        let (tmp, first) = bendright_block(src)?;
        let (dst, second) = foldleft_block(&tmp)?;
        Ok((dst, second.dot(&first)))
    }
}

fn foldright_pair<I: Sector>(pair: &FusionTreePair<I>) -> Result<(FusionTreePair<I>, f64)> {
    if I::fusion_style() != FusionStyle::UniqueFusion {
        return Err(Tensor0Error::Message(
            "foldright pair requires UniqueFusion".to_string(),
        ));
    }

    let f1 = &pair.row;
    let f2 = &pair.col;
    debug_assert!(!f1.uncoupled.is_empty());
    let a = f1.uncoupled[0].clone();
    let frobenius_schur = I::frobenius_schur_phase(&a)?;
    let is_dual_a = f1.is_dual[0];

    let mut f1_terms = multi_fmove(f1)?;
    if f1_terms.len() != 1 {
        return Err(Tensor0Error::Message(
            "foldright multi_Fmove requires a unique fusion tree term".to_string(),
        ));
    }
    let (f1_prime, coeff1) = f1_terms.remove(0);
    let b = f1_prime.coupled.clone();
    let c = f1.coupled.clone();
    let a_symbol = I::a_symbol(&a, &b, &c)?;
    let mut f2_terms = multi_fmove_inv(&a.dual(), &b, f2, !is_dual_a)?;
    if f2_terms.len() != 1 {
        return Err(Tensor0Error::Message(
            "foldright inverse multi_Fmove requires a unique fusion tree term".to_string(),
        ));
    }
    let (f2_prime, coeff2) = f2_terms.remove(0);

    let coeff0 = ((c.quantum_dim() as f64) / (b.quantum_dim() as f64)).sqrt();
    // GenericFusion/complex symbols must restore TensorKit's conjugation here.
    let mut coeff = coeff0 * coeff1 * a_symbol * coeff2;
    if is_dual_a {
        coeff *= frobenius_schur;
    }

    Ok((
        FusionTreePair {
            row: f1_prime,
            col: f2_prime,
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

    let uncoupled1_dst = src.row_uncoupled()[1..].to_vec();
    let is_dual1_dst = src.row_is_dual()[1..].to_vec();
    let mut uncoupled2_dst = Vec::with_capacity(src.numin() + 1);
    uncoupled2_dst.push(src.row_uncoupled()[0].dual());
    uncoupled2_dst.extend_from_slice(src.col_uncoupled());
    let mut is_dual2_dst = Vec::with_capacity(src.numin() + 1);
    is_dual2_dst.push(!src.row_is_dual()[0]);
    is_dual2_dst.extend_from_slice(src.col_is_dual());

    let dst = FusionTreeBlock::new(uncoupled1_dst, is_dual1_dst, uncoupled2_dst, is_dual2_dst)?;
    let dst_index = dst.index_map();
    let mut transform = Array2::zeros((dst.trees().len(), src.trees().len()));
    if I::fusion_style() == FusionStyle::UniqueFusion {
        for (source_index, pair) in src.trees().iter().enumerate() {
            let (target_pair, coeff) = foldright_pair(pair)?;
            let Some(target_index) = dst_index.get(&target_pair).copied() else {
                return Err(Tensor0Error::Message(
                    "foldright destination fusion tree pair was not found".to_string(),
                ));
            };
            transform[[target_index, source_index]] = coeff;
        }
        return Ok((dst, transform));
    }

    let f1 = &src.trees()[0].row;
    let a = f1.uncoupled[0].clone();
    let frobenius_schur = I::frobenius_schur_phase(&a)?;
    let is_dual_a = f1.is_dual[0];
    let mut f1_cache: FMoveCache<I> = HashMap::new();
    let mut f2_cache: FMoveInvCache<I> = HashMap::new();
    let mut a_symbol_cache: HashMap<(I, I), f64> = HashMap::new();

    for (source_index, pair) in src.trees().iter().enumerate() {
        let f1 = &pair.row;
        let f2 = &pair.col;
        let f1_terms = match f1_cache.entry(f1.clone()) {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => entry.insert(multi_fmove(f1)?),
        };

        for (f1_prime, coeff1) in f1_terms.iter() {
            let b = f1_prime.coupled.clone();
            let c = f1.coupled.clone();
            let a_key = (b.clone(), c.clone());
            let a_symbol = match a_symbol_cache.entry(a_key) {
                Entry::Occupied(entry) => *entry.get(),
                Entry::Vacant(entry) => *entry.insert(I::a_symbol(&a, &b, &c)?),
            };

            let inv_key = (b.clone(), f2.clone());
            let f2_terms = match f2_cache.entry(inv_key) {
                Entry::Occupied(entry) => entry.into_mut(),
                Entry::Vacant(entry) => {
                    entry.insert(multi_fmove_inv(&a.dual(), &b, f2, !is_dual_a)?)
                }
            };

            let coeff0 = ((c.quantum_dim() as f64) / (b.quantum_dim() as f64)).sqrt();
            for (f2_prime, coeff2) in f2_terms.iter() {
                // GenericFusion/complex symbols must restore TensorKit's conjugation here.
                let mut coeff = coeff0 * coeff1 * a_symbol * coeff2;
                if is_dual_a {
                    coeff *= frobenius_schur;
                }
                if coeff == 0.0 {
                    continue;
                }

                let target_pair = FusionTreePair {
                    row: f1_prime.clone(),
                    col: f2_prime.clone(),
                };
                let Some(target_index) = dst_index.get(&target_pair).copied() else {
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
    let ((a, b, c), target_pair) = _bendright_treepair(pair)?;
    let f1 = &pair.row;
    let mut coeff0 = ((c.quantum_dim() as f64) / (a.quantum_dim() as f64)).sqrt();
    if f1.is_dual[f1.is_dual.len() - 1] {
        // GenericFusion/complex symbols must restore TensorKit's conjugation here.
        coeff0 *= I::frobenius_schur_phase(&b.dual())?;
    }
    let coeff = coeff0 * I::b_symbol(&a, &b, &c)?;
    Ok((target_pair, coeff))
}

fn _bendright_treepair<I: Sector>(
    pair: &FusionTreePair<I>,
) -> Result<((I, I, I), FusionTreePair<I>)> {
    let f1 = &pair.row;
    let f2 = &pair.col;
    let n1 = f1.uncoupled.len();
    let n2 = f2.uncoupled.len();

    let a = match n1 {
        0 => unreachable!("bendright pair has at least one outgoing leg"),
        1 => I::unit(),
        2 => f1.uncoupled[0].clone(),
        _ => f1.innerlines[n1 - 3].clone(),
    };
    let b = f1.uncoupled[n1 - 1].clone();
    let c = f1.coupled.clone();

    let uncoupled1 = f1.uncoupled[..n1 - 1].to_vec();
    let is_dual1 = f1.is_dual[..n1 - 1].to_vec();
    let innerlines1 = if n1 > 2 {
        f1.innerlines[..f1.innerlines.len() - 1].to_vec()
    } else {
        vec![]
    };
    let vertices1 = if n1 > 1 {
        f1.vertices[..f1.vertices.len() - 1].to_vec()
    } else {
        vec![]
    };
    let f1_prime = FusionTree::new(uncoupled1, a.clone(), is_dual1, innerlines1, vertices1)?;

    let mut uncoupled2 = f2.uncoupled.clone();
    uncoupled2.push(b.dual());
    let mut is_dual2 = f2.is_dual.clone();
    is_dual2.push(!f1.is_dual[n1 - 1]);
    let mut innerlines2 = if n2 > 1 {
        f2.innerlines.clone()
    } else {
        vec![]
    };
    if n2 > 1 {
        innerlines2.push(c.clone());
    }
    let mut vertices2 = if n2 > 0 { f2.vertices.clone() } else { vec![] };
    if n2 > 0 {
        vertices2.push(0);
    }
    let f2_prime = FusionTree::new(uncoupled2, a.clone(), is_dual2, innerlines2, vertices2)?;

    Ok((
        (a, b, c),
        FusionTreePair {
            row: f1_prime,
            col: f2_prime,
        },
    ))
}

fn bendright_block<I: Sector>(
    src: &FusionTreeBlock<I>,
) -> Result<(FusionTreeBlock<I>, Array2<f64>)> {
    let n1 = src.numout();
    debug_assert!(n1 > 0);
    let b = src.row_uncoupled()[n1 - 1].clone();
    let is_dual_b = src.row_is_dual()[n1 - 1];
    let uncoupled1_dst = src.row_uncoupled()[..n1 - 1].to_vec();
    let is_dual1_dst = src.row_is_dual()[..n1 - 1].to_vec();
    let mut uncoupled2_dst = src.col_uncoupled().to_vec();
    uncoupled2_dst.push(b.dual());
    let mut is_dual2_dst = src.col_is_dual().to_vec();
    is_dual2_dst.push(!is_dual_b);

    let dst = FusionTreeBlock::new(uncoupled1_dst, is_dual1_dst, uncoupled2_dst, is_dual2_dst)?;
    let dst_index = dst.index_map();
    let mut transform = Array2::zeros((dst.trees().len(), src.trees().len()));

    for (source_index, pair) in src.trees().iter().enumerate() {
        let (target_pair, coeff) = bendright_pair(pair)?;
        if coeff == 0.0 {
            continue;
        }
        let Some(target_index) = dst_index.get(&target_pair).copied() else {
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

    let mut uncoupled1_dst = Vec::with_capacity(src.numout() + 1);
    uncoupled1_dst.push(src.col_uncoupled()[0].dual());
    uncoupled1_dst.extend_from_slice(src.row_uncoupled());
    let mut is_dual1_dst = Vec::with_capacity(src.numout() + 1);
    is_dual1_dst.push(!src.col_is_dual()[0]);
    is_dual1_dst.extend_from_slice(src.row_is_dual());
    let uncoupled2_dst = src.col_uncoupled()[1..].to_vec();
    let is_dual2_dst = src.col_is_dual()[1..].to_vec();

    let dst = FusionTreeBlock::new(uncoupled1_dst, is_dual1_dst, uncoupled2_dst, is_dual2_dst)?;
    let dst_index = dst.index_map();
    let mut transform = Array2::zeros((dst.trees().len(), src.trees().len()));
    if I::fusion_style() == FusionStyle::UniqueFusion {
        for (source_index, pair) in src.trees().iter().enumerate() {
            let (target_pair, coeff) = foldleft_pair(pair)?;
            let Some(target_index) = dst_index.get(&target_pair).copied() else {
                return Err(Tensor0Error::Message(
                    "foldleft destination fusion tree pair was not found".to_string(),
                ));
            };
            transform[[target_index, source_index]] = coeff;
        }
        return Ok((dst, transform));
    }

    let f2 = &src.trees()[0].col;
    let a = f2.uncoupled[0].clone();
    let frobenius_schur = I::frobenius_schur_phase(&a)?;
    let is_dual_a = f2.is_dual[0];
    let mut f2_cache: FMoveCache<I> = HashMap::new();
    let mut f1_cache: FMoveInvCache<I> = HashMap::new();
    let mut a_symbol_cache: HashMap<(I, I), f64> = HashMap::new();

    for (source_index, pair) in src.trees().iter().enumerate() {
        let f1 = &pair.row;
        let f2 = &pair.col;
        let f2_terms = match f2_cache.entry(f2.clone()) {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => entry.insert(multi_fmove(f2)?),
        };

        for (f2_prime, coeff2) in f2_terms.iter() {
            let b = f2_prime.coupled.clone();
            let c = f2.coupled.clone();
            let a_key = (b.clone(), c.clone());
            let a_symbol = match a_symbol_cache.entry(a_key) {
                Entry::Occupied(entry) => *entry.get(),
                Entry::Vacant(entry) => *entry.insert(I::a_symbol(&a, &b, &c)?),
            };

            let inv_key = (b.clone(), f1.clone());
            let f1_terms = match f1_cache.entry(inv_key) {
                Entry::Occupied(entry) => entry.into_mut(),
                Entry::Vacant(entry) => {
                    entry.insert(multi_fmove_inv(&a.dual(), &b, f1, !is_dual_a)?)
                }
            };

            let coeff0 = ((c.quantum_dim() as f64) / (b.quantum_dim() as f64)).sqrt();
            for (f1_prime, coeff1) in f1_terms.iter() {
                // GenericFusion/complex symbols must restore TensorKit's conjugation here.
                let mut coeff = coeff0 * coeff1 * a_symbol * coeff2;
                if is_dual_a {
                    // GenericFusion/complex symbols must restore TensorKit's conjugation here.
                    coeff *= frobenius_schur;
                }
                if coeff == 0.0 {
                    continue;
                }

                let target_pair = FusionTreePair {
                    row: f1_prime.clone(),
                    col: f2_prime.clone(),
                };
                let Some(target_index) = dst_index.get(&target_pair).copied() else {
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
    let n2 = src.numin();
    debug_assert!(n2 > 0);
    let b = src.col_uncoupled()[n2 - 1].clone();
    let is_dual_b = src.col_is_dual()[n2 - 1];
    let uncoupled2_dst = src.col_uncoupled()[..n2 - 1].to_vec();
    let is_dual2_dst = src.col_is_dual()[..n2 - 1].to_vec();
    let mut uncoupled1_dst = src.row_uncoupled().to_vec();
    uncoupled1_dst.push(b.dual());
    let mut is_dual1_dst = src.row_is_dual().to_vec();
    is_dual1_dst.push(!is_dual_b);

    let dst = FusionTreeBlock::new(uncoupled1_dst, is_dual1_dst, uncoupled2_dst, is_dual2_dst)?;
    let dst_index = dst.index_map();
    let mut transform = Array2::zeros((dst.trees().len(), src.trees().len()));

    for (source_index, pair) in src.trees().iter().enumerate() {
        let (target_pair, coeff) = bendleft_pair(pair)?;
        if coeff == 0.0 {
            continue;
        }
        let Some(target_index) = dst_index.get(&target_pair).copied() else {
            return Err(Tensor0Error::Message(
                "bendleft destination fusion tree pair was not found".to_string(),
            ));
        };
        transform[[target_index, source_index]] = coeff;
    }

    Ok((dst, transform))
}
