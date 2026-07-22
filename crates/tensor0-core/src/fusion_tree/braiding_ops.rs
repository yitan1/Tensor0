//! Braiding, permutation, and index-flip operations on fusion-tree bases.
//!
//! Permutation is represented using TensorKit's repartition-braid-repartition
//! structure. Future anyonic support should add explicit braid-word operations
//! here instead of representing every braid as a plain permutation.

use std::collections::HashMap;

use ndarray::Array2;

use crate::error::{Result, Tensor0Error};
use crate::fusion_tree::{enumerate_fusion_trees, FusionTree, FusionTreeBlock, FusionTreePair};
use crate::sector::{BraidingStyle, FusionStyle, Sector};

use super::auxiliary::{linearize_permutation, permutation_to_swaps};
use super::duality_ops::{repartition_block, repartition_pair};

pub(crate) fn braid_pair<I: Sector>(
    src: &FusionTreePair<I>,
    p_codomain: &[usize],
    p_domain: &[usize],
    levels_codomain: &[usize],
    levels_domain: &[usize],
) -> Result<(FusionTreePair<I>, f64)> {
    let n1 = src.row.uncoupled().len();
    let n2 = src.col.uncoupled().len();
    debug_assert_eq!(levels_codomain.len(), n1);
    debug_assert_eq!(levels_domain.len(), n2);
    let permutation = linearize_permutation(p_codomain, p_domain, n1, n2)?;
    let levels = linearized_braid_levels(levels_codomain, levels_domain);

    let (full_pair, pre) = repartition_pair(src, n1 + n2)?;
    let FusionTreePair {
        row: full_row,
        col: full_col,
    } = full_pair;
    let (row_prime, braid) = braid_tree(&full_row, &permutation, &levels)?;
    let (dst, post) = repartition_pair(
        &FusionTreePair {
            row: row_prime,
            col: full_col,
        },
        p_codomain.len(),
    )?;

    Ok((dst, pre * braid * post))
}

pub(crate) fn braid_block<I: Sector>(
    src: &FusionTreeBlock<I>,
    p_codomain: &[usize],
    p_domain: &[usize],
    levels_codomain: &[usize],
    levels_domain: &[usize],
) -> Result<(FusionTreeBlock<I>, Array2<f64>)> {
    let n1 = src.numout();
    let n2 = src.numin();
    let numind = src.numind();
    debug_assert_eq!(levels_codomain.len(), n1);
    debug_assert_eq!(levels_domain.len(), n2);
    let permutation = linearize_permutation(p_codomain, p_domain, n1, n2)?;
    let mut levels = linearized_braid_levels(levels_codomain, levels_domain);

    let (mut dst, mut transform) = repartition_block(src, numind)?;
    for swap in permutation_to_swaps(&permutation) {
        let inv = levels[swap] > levels[swap + 1];
        let (next_dst, step) = artin_braid_block(&dst, swap, inv)?;
        transform = step.dot(&transform);
        dst = next_dst;
        levels.swap(swap, swap + 1);
    }

    if p_domain.is_empty() {
        Ok((dst, transform))
    } else {
        let (dst, step) = repartition_block(&dst, p_codomain.len())?;
        Ok((dst, step.dot(&transform)))
    }
}

/// Flips already-validated visible indices on one fusion-tree pair.
///
/// `indices` must be unique and in range for `src`.
pub(crate) fn flip_pair<I: Sector>(
    src: &FusionTreePair<I>,
    indices: &[usize],
    inv: bool,
) -> Result<(FusionTreePair<I>, f64)> {
    let mut dst = src.clone();
    let mut factor = 1.0;
    for &index in indices {
        let step = flip_visible_index(&mut dst, index, inv)?;
        factor *= step;
    }
    Ok((dst, factor))
}

fn flip_visible_index<I: Sector>(
    pair: &mut FusionTreePair<I>,
    index: usize,
    inv: bool,
) -> Result<f64> {
    let numout = pair.row.uncoupled().len();
    let factor = if index < numout {
        let sector = &pair.row.uncoupled()[index];
        let was_dual = pair.row.is_dual()[index];
        let factor = if was_dual != inv {
            I::frobenius_schur_phase(sector)? * sector.twist()
        } else {
            1.0
        };
        pair.row.is_dual_mut()[index] = !was_dual;
        factor
    } else {
        let col_index = index - numout;
        let sector = &pair.col.uncoupled()[col_index];
        let was_dual = pair.col.is_dual()[col_index];
        let factor = if was_dual == inv {
            sector.twist()
        } else {
            I::frobenius_schur_phase(sector)?
        };
        pair.col.is_dual_mut()[col_index] = !was_dual;
        factor
    };
    Ok(factor)
}

fn braid_tree<I: Sector>(
    tree: &FusionTree<I>,
    permutation: &[usize],
    levels: &[usize],
) -> Result<(FusionTree<I>, f64)> {
    debug_assert_eq!(levels.len(), tree.uncoupled().len());
    if I::fusion_style() != FusionStyle::UniqueFusion {
        return Err(Tensor0Error::Message(
            "braid tree requires UniqueFusion".to_string(),
        ));
    }

    if matches!(
        I::braiding_style(),
        BraidingStyle::Bosonic | BraidingStyle::Fermionic
    ) {
        let mut coeff = 1.0;
        for i in 0..permutation.len() {
            for j in 0..i {
                if permutation[j] > permutation[i] {
                    let a = &tree.uncoupled()[permutation[j]];
                    let b = &tree.uncoupled()[permutation[i]];
                    let output = a
                        .fusion_outputs(b)
                        .next()
                        .expect("UniqueFusion sectors have one fusion output");
                    coeff *= I::r_symbol(a, b, &output);
                }
            }
        }

        let uncoupled = permutation
            .iter()
            .map(|index| tree.uncoupled()[*index].clone())
            .collect::<Vec<_>>();
        let is_dual = permutation
            .iter()
            .map(|index| tree.is_dual()[*index])
            .collect::<Vec<_>>();
        let tree_prime = enumerate_fusion_trees(&uncoupled, &is_dual, tree.coupled())?
            .pop()
            .ok_or_else(|| {
                Tensor0Error::Message(
                    "UniqueFusion symmetric braid produced no fusion tree".to_string(),
                )
            })?;
        Ok((tree_prime, coeff))
    } else {
        let mut tree = tree.clone();
        let mut coeff = 1.0;
        let mut levels = levels.to_vec();

        for swap in permutation_to_swaps(permutation) {
            let inv = levels[swap] > levels[swap + 1];
            let (tree_prime, coeff_prime) = artin_braid_tree(&tree, swap, inv)?;
            coeff *= coeff_prime;
            tree = tree_prime;
            levels.swap(swap, swap + 1);
        }

        Ok((tree, coeff))
    }
}

fn artin_braid_tree<I: Sector>(
    tree: &FusionTree<I>,
    i: usize,
    inv: bool,
) -> Result<(FusionTree<I>, f64)> {
    let n = tree.uncoupled().len();
    if i + 1 >= n {
        return Err(Tensor0Error::Message(
            "artin_braid index out of range".to_string(),
        ));
    }
    if I::fusion_style() != FusionStyle::UniqueFusion {
        return Err(Tensor0Error::Message(
            "artin_braid tree requires UniqueFusion".to_string(),
        ));
    }

    let (uncoupled_prime, is_dual_prime) = swapped_tree_labels(tree.uncoupled(), tree.is_dual(), i);
    let mut terms =
        artin_braid_multiplicity_free_terms(tree, i, inv, &uncoupled_prime, &is_dual_prime)?;
    if terms.len() != 1 {
        return Err(Tensor0Error::Message(
            "artin_braid tree requires a unique target fusion tree".to_string(),
        ));
    }
    Ok(terms.remove(0))
}

fn artin_braid_block<I: Sector>(
    src: &FusionTreeBlock<I>,
    i: usize,
    inv: bool,
) -> Result<(FusionTreeBlock<I>, Array2<f64>)> {
    if src.numin() != 0 {
        return Err(Tensor0Error::Message(
            "artin_braid expects a full outgoing fusion tree block".to_string(),
        ));
    }
    if i + 1 >= src.numout() {
        return Err(Tensor0Error::Message(
            "artin_braid index out of range".to_string(),
        ));
    }

    let left = &src.row_uncoupled()[i];
    let right = &src.row_uncoupled()[i + 1];
    let is_unit_braid = left == &I::unit() || right == &I::unit();
    let (uncoupled_prime, is_dual_prime) =
        swapped_tree_labels(src.row_uncoupled(), src.row_is_dual(), i);
    let dst = FusionTreeBlock::new(
        uncoupled_prime.clone(),
        is_dual_prime.clone(),
        vec![],
        vec![],
    )?;
    let mut transform = Array2::zeros((dst.trees().len(), src.trees().len()));
    let dst_index = dst.index_map();

    if !is_unit_braid && I::fusion_style() == FusionStyle::GenericFusion {
        let message = if i == 0 {
            "artin_braid does not support GenericFusion R-matrix block braiding"
        } else {
            "artin_braid does not support GenericFusion F/R-matrix block braiding"
        };
        return Err(Tensor0Error::Message(message.to_string()));
    }

    fill_artin_braid_block(
        src,
        i,
        inv,
        &uncoupled_prime,
        &is_dual_prime,
        &dst_index,
        &mut transform,
    )?;

    Ok((dst, transform))
}

fn fill_artin_braid_block<I: Sector>(
    src: &FusionTreeBlock<I>,
    i: usize,
    inv: bool,
    uncoupled_prime: &[I],
    is_dual_prime: &[bool],
    dst_index: &HashMap<FusionTreePair<I>, usize>,
    transform: &mut Array2<f64>,
) -> Result<()> {
    for (source_index, FusionTreePair { row, col }) in src.trees().iter().enumerate() {
        for (row_prime, coeff) in
            artin_braid_multiplicity_free_terms(row, i, inv, uncoupled_prime, is_dual_prime)?
        {
            if coeff == 0.0 {
                continue;
            }

            let target_pair = FusionTreePair {
                row: row_prime,
                col: col.clone(),
            };
            let Some(target_index) = dst_index.get(&target_pair).copied() else {
                return Err(Tensor0Error::Message(
                    "artin_braid destination fusion tree pair was not found".to_string(),
                ));
            };
            transform[[target_index, source_index]] = coeff;
        }
    }

    Ok(())
}

fn artin_braid_multiplicity_free_terms<I: Sector>(
    tree: &FusionTree<I>,
    i: usize,
    inv: bool,
    uncoupled: &[I],
    is_dual: &[bool],
) -> Result<Vec<(FusionTree<I>, f64)>> {
    let left = &tree.uncoupled()[i];
    let right = &tree.uncoupled()[i + 1];
    if left == &I::unit() || right == &I::unit() {
        return Ok(vec![(
            artin_braid_through_unit(tree, uncoupled, is_dual, i)?,
            1.0,
        )]);
    }
    if I::braiding_style() == BraidingStyle::NoBraiding {
        return Err(Tensor0Error::Message(
            "artin_braid does not support sectors with NoBraiding".to_string(),
        ));
    }

    if i == 0 {
        let c = if tree.uncoupled().len() > 2 {
            tree.innerlines()[0].clone()
        } else {
            tree.coupled().clone()
        };
        let coeff = if inv {
            // GenericFusion/complex symbols must restore TensorKit's conjugation here.
            I::r_symbol(right, left, &c)
        } else {
            I::r_symbol(left, right, &c)
        };
        return Ok(vec![(
            FusionTree::new(
                uncoupled.to_vec(),
                tree.coupled().clone(),
                is_dual.to_vec(),
                tree.innerlines().to_vec(),
                tree.vertices().to_vec(),
            )?,
            coeff,
        )]);
    }

    let inner_extended = artin_inner_extended(tree);
    let a = &inner_extended[i - 1];
    let b = &tree.uncoupled()[i];
    let c = &inner_extended[i];
    let d = &tree.uncoupled()[i + 1];
    let e = &inner_extended[i + 1];
    let outputs2 = e.fusion_outputs(&b.dual()).collect::<Vec<_>>();

    let mut terms = Vec::new();
    for c_prime in a
        .fusion_outputs(d)
        .filter(|sector| outputs2.contains(sector))
    {
        let coeff = if inv {
            // GenericFusion/complex symbols must restore TensorKit's conjugation here.
            I::r_symbol(d, c, e)
                * I::f_symbol(d, a, b, e, &c_prime, c)?
                * I::r_symbol(d, a, &c_prime)
        } else {
            // GenericFusion/complex symbols must restore TensorKit's conjugation here.
            I::r_symbol(c, d, e)
                * I::f_symbol(d, a, b, e, &c_prime, c)?
                * I::r_symbol(a, d, &c_prime)
        };

        let mut innerlines = tree.innerlines().to_vec();
        innerlines[i - 1] = c_prime;
        terms.push((
            FusionTree::new(
                uncoupled.to_vec(),
                tree.coupled().clone(),
                is_dual.to_vec(),
                innerlines,
                tree.vertices().to_vec(),
            )?,
            coeff,
        ));
    }

    Ok(terms)
}

fn artin_braid_through_unit<I: Sector>(
    tree: &FusionTree<I>,
    uncoupled: &[I],
    is_dual: &[bool],
    i: usize,
) -> Result<FusionTree<I>> {
    let mut innerlines = tree.innerlines().to_vec();
    let mut vertices = tree.vertices().to_vec();

    if i > 0 {
        let inner_extended = artin_inner_extended(tree);

        let replacement = if tree.uncoupled()[i] == I::unit() {
            inner_extended[i + 1].clone()
        } else {
            inner_extended[i - 1].clone()
        };
        innerlines[i - 1] = replacement;
        vertices.swap(i - 1, i);
    }

    FusionTree::new(
        uncoupled.to_vec(),
        tree.coupled().clone(),
        is_dual.to_vec(),
        innerlines,
        vertices,
    )
}

fn artin_inner_extended<I: Sector>(tree: &FusionTree<I>) -> Vec<I> {
    let mut inner_extended = Vec::with_capacity(tree.uncoupled().len());
    inner_extended.push(tree.uncoupled()[0].clone());
    inner_extended.extend(tree.innerlines().iter().cloned());
    inner_extended.push(tree.coupled().clone());
    inner_extended
}

fn swapped_tree_labels<I: Sector>(
    uncoupled: &[I],
    is_dual: &[bool],
    i: usize,
) -> (Vec<I>, Vec<bool>) {
    let mut uncoupled_prime = uncoupled.to_vec();
    uncoupled_prime.swap(i, i + 1);
    let mut is_dual_prime = is_dual.to_vec();
    is_dual_prime.swap(i, i + 1);
    (uncoupled_prime, is_dual_prime)
}

fn linearized_braid_levels(levels1: &[usize], levels2: &[usize]) -> Vec<usize> {
    let mut levels = levels1.to_vec();
    levels.extend(levels2.iter().rev().copied());
    levels
}

#[cfg(test)]
mod tests {
    use ndarray::Array2;

    use crate::fusion_tree::FusionTreeBlock;
    use crate::sector::SU2Irrep;

    use super::braid_block;

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

    fn four_out_half_block() -> FusionTreeBlock<SU2Irrep> {
        let half = su2(1);
        FusionTreeBlock::new(
            vec![half, half, half, half],
            vec![false, false, false, false],
            vec![],
            vec![],
        )
        .unwrap()
    }

    #[test]
    fn braid_block_and_inverse_are_identity_on_su2_basis() {
        let src = four_out_half_block();
        let p_codomain = [1, 0, 2, 3];
        let levels = [0, 1, 2, 3];
        let inverse_levels = [1, 0, 2, 3];

        let (dst, transform) = braid_block(&src, &p_codomain, &[], &levels, &[]).unwrap();
        let (roundtrip, inverse_transform) =
            braid_block(&dst, &p_codomain, &[], &inverse_levels, &[]).unwrap();

        assert_eq!(roundtrip, src);
        assert_identity(&inverse_transform.dot(&transform));
    }
}
