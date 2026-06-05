//! Braiding and permutation operations on fusion-tree bases.
//!
//! Permutation is represented using TensorKit's repartition-braid-repartition
//! structure. Future anyonic support should add explicit braid-word operations
//! here instead of representing every braid as a plain permutation.

use std::collections::{BTreeSet, HashMap};

use ndarray::Array2;

use crate::error::{Result, Tensor0Error};
use crate::fusion_tree::{enumerate_fusion_trees, FusionTree, FusionTreeBlock, FusionTreePair};
use crate::sector::{BraidingStyle, FusionStyle, Sector};

use super::duality_ops::{repartition_block, repartition_pair};
use super::permutation_ops::{linearize_permutation, permutation_to_swaps};

pub(crate) fn braid_pair<I: Sector>(
    src: &FusionTreePair<I>,
    p1: &[usize],
    p2: &[usize],
    l1: &[usize],
    l2: &[usize],
) -> Result<(FusionTreePair<I>, f64)> {
    let n1 = src.row.uncoupled.len();
    let n2 = src.col.uncoupled.len();
    debug_assert_eq!(l1.len(), n1);
    debug_assert_eq!(l2.len(), n2);
    let p = linearize_permutation(p1, p2, n1, n2)?;
    let levels = linearized_braid_levels(l1, l2);

    let (full_pair, coeff1) = repartition_pair(src, n1 + n2)?;
    let FusionTreePair { row: f, col: f0 } = full_pair;
    let (f_prime, coeff2) = braid_tree(&f, &p, &levels)?;
    let (dst, coeff3) = repartition_pair(
        &FusionTreePair {
            row: f_prime,
            col: f0,
        },
        p1.len(),
    )?;

    Ok((dst, coeff1 * coeff2 * coeff3))
}

pub(crate) fn braid_block<I: Sector>(
    src: &FusionTreeBlock<I>,
    p1: &[usize],
    p2: &[usize],
    l1: &[usize],
    l2: &[usize],
) -> Result<(FusionTreeBlock<I>, Array2<f64>)> {
    let n1 = src.numout();
    let n2 = src.numin();
    let numind = src.numind();
    debug_assert_eq!(l1.len(), n1);
    debug_assert_eq!(l2.len(), n2);
    let p = linearize_permutation(p1, p2, n1, n2)?;
    let mut levels = linearized_braid_levels(l1, l2);

    let (mut dst, mut u) = repartition_block(src, numind)?;
    for s in permutation_to_swaps(&p) {
        let inv = levels[s] > levels[s + 1];
        let (next_dst, u_tmp) = artin_braid_block(&dst, s, inv)?;
        u = u_tmp.dot(&u);
        dst = next_dst;
        levels.swap(s, s + 1);
    }

    if p2.is_empty() {
        Ok((dst, u))
    } else {
        let (dst, u_tmp) = repartition_block(&dst, p1.len())?;
        Ok((dst, u_tmp.dot(&u)))
    }
}

fn braid_tree<I: Sector>(
    f: &FusionTree<I>,
    p: &[usize],
    levels: &[usize],
) -> Result<(FusionTree<I>, f64)> {
    debug_assert_eq!(levels.len(), f.uncoupled.len());
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
        for i in 0..p.len() {
            for j in 0..i {
                if p[j] > p[i] {
                    let a = &f.uncoupled[p[j]];
                    let b = &f.uncoupled[p[i]];
                    let outputs = a.fusion_outputs(b);
                    coeff *= I::r_symbol(a, b, &outputs[0]);
                }
            }
        }

        let uncoupled = p
            .iter()
            .map(|index| f.uncoupled[*index].clone())
            .collect::<Vec<_>>();
        let is_dual = p.iter().map(|index| f.is_dual[*index]).collect::<Vec<_>>();
        let mut trees = enumerate_fusion_trees(&uncoupled, &is_dual, &f.coupled)?;
        let f_prime = trees
            .pop()
            .expect("UniqueFusion symmetric braid produces a single fusion tree");
        Ok((f_prime, coeff))
    } else {
        let mut f = f.clone();
        let mut coeff = 1.0;
        let mut levels = levels.to_vec();

        for s in permutation_to_swaps(p) {
            let inv = levels[s] > levels[s + 1];
            let (f_prime, coeff_prime) = artin_braid_tree(&f, s, inv)?;
            coeff *= coeff_prime;
            f = f_prime;
            levels.swap(s, s + 1);
        }

        Ok((f, coeff))
    }
}

fn artin_braid_tree<I: Sector>(
    f: &FusionTree<I>,
    i: usize,
    inv: bool,
) -> Result<(FusionTree<I>, f64)> {
    let n = f.uncoupled.len();
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

    let mut uncoupled_prime = f.uncoupled.clone();
    uncoupled_prime.swap(i, i + 1);
    let mut is_dual_prime = f.is_dual.clone();
    is_dual_prime.swap(i, i + 1);
    let left = &f.uncoupled[i];
    let right = &f.uncoupled[i + 1];

    if left == &I::unit() || right == &I::unit() {
        return Ok((
            _artin_braid_through_unit(f, &uncoupled_prime, &is_dual_prime, i)?,
            1.0,
        ));
    }
    if I::braiding_style() == BraidingStyle::NoBraiding {
        return Err(Tensor0Error::Message(
            "artin_braid does not support sectors with NoBraiding".to_string(),
        ));
    }

    if i == 0 {
        let c = if n > 2 {
            f.innerlines[0].clone()
        } else {
            f.coupled.clone()
        };
        let r = if inv {
            // GenericFusion/complex symbols must restore TensorKit's conjugation here.
            I::r_symbol(right, left, &c)
        } else {
            I::r_symbol(left, right, &c)
        };
        return Ok((
            FusionTree::new(
                uncoupled_prime,
                f.coupled.clone(),
                is_dual_prime,
                f.innerlines.clone(),
                f.vertices.clone(),
            )?,
            r,
        ));
    }

    let inner_extended = artin_inner_extended(f);

    let a = &inner_extended[i - 1];
    let b = &f.uncoupled[i];
    let c = &inner_extended[i];
    let d = &f.uncoupled[i + 1];
    let e = &inner_extended[i + 1];

    let c_prime = a
        .fusion_outputs(d)
        .into_iter()
        .next()
        .expect("UniqueFusion artin braid has a valid intermediate channel");
    let coeff = if inv {
        // GenericFusion/complex symbols must restore TensorKit's conjugation here.
        I::r_symbol(d, c, e) * I::f_symbol(d, a, b, e, &c_prime, c)? * I::r_symbol(d, a, &c_prime)
    } else {
        // GenericFusion/complex symbols must restore TensorKit's conjugation here.
        I::r_symbol(c, d, e) * I::f_symbol(d, a, b, e, &c_prime, c)? * I::r_symbol(a, d, &c_prime)
    };

    let mut innerlines = f.innerlines.clone();
    innerlines[i - 1] = c_prime;
    Ok((
        FusionTree::new(
            uncoupled_prime,
            f.coupled.clone(),
            is_dual_prime,
            innerlines,
            f.vertices.clone(),
        )?,
        coeff,
    ))
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
    let mut uncoupled_prime = src.row_uncoupled().to_vec();
    uncoupled_prime.swap(i, i + 1);
    let mut is_dual_prime = src.row_is_dual().to_vec();
    is_dual_prime.swap(i, i + 1);
    let dst = FusionTreeBlock::new(
        uncoupled_prime.clone(),
        is_dual_prime.clone(),
        vec![],
        vec![],
    )?;
    let mut transform = Array2::zeros((dst.trees().len(), src.trees().len()));
    let dst_index = dst.index_map();

    if is_unit_braid {
        fill_artin_braid_unit_block(
            src,
            i,
            &uncoupled_prime,
            &is_dual_prime,
            &dst_index,
            &mut transform,
        )?;
        return Ok((dst, transform));
    }
    if I::braiding_style() == BraidingStyle::NoBraiding {
        return Err(Tensor0Error::Message(
            "artin_braid does not support sectors with NoBraiding".to_string(),
        ));
    }

    match I::fusion_style() {
        FusionStyle::UniqueFusion | FusionStyle::SimpleFusion => {
            fill_artin_braid_multiplicity_free_block(
                src,
                i,
                inv,
                &uncoupled_prime,
                &is_dual_prime,
                &dst_index,
                &mut transform,
            )?
        }
        FusionStyle::GenericFusion if i == 0 => {
            return Err(Tensor0Error::Message(
                "artin_braid does not support GenericFusion R-matrix block braiding".to_string(),
            ));
        }
        FusionStyle::GenericFusion => {
            return Err(Tensor0Error::Message(
                "artin_braid does not support GenericFusion F/R-matrix block braiding".to_string(),
            ));
        }
    }

    Ok((dst, transform))
}

fn fill_artin_braid_unit_block<I: Sector>(
    src: &FusionTreeBlock<I>,
    i: usize,
    uncoupled_prime: &[I],
    is_dual_prime: &[bool],
    dst_index: &HashMap<FusionTreePair<I>, usize>,
    transform: &mut Array2<f64>,
) -> Result<()> {
    for (source_index, FusionTreePair { row: f1, col: f2 }) in src.trees().iter().enumerate() {
        let f1_prime = _artin_braid_through_unit(f1, uncoupled_prime, is_dual_prime, i)?;
        let target_pair = FusionTreePair {
            row: f1_prime,
            col: f2.clone(),
        };
        let Some(target_index) = dst_index.get(&target_pair).copied() else {
            return Err(Tensor0Error::Message(
                "artin_braid destination fusion tree pair was not found".to_string(),
            ));
        };
        transform[[target_index, source_index]] = 1.0;
    }

    Ok(())
}

fn fill_artin_braid_multiplicity_free_block<I: Sector>(
    src: &FusionTreeBlock<I>,
    i: usize,
    inv: bool,
    uncoupled_prime: &[I],
    is_dual_prime: &[bool],
    dst_index: &HashMap<FusionTreePair<I>, usize>,
    transform: &mut Array2<f64>,
) -> Result<()> {
    let n = src.numout();
    let left = &src.row_uncoupled()[i];
    let right = &src.row_uncoupled()[i + 1];

    for (source_index, FusionTreePair { row: f1, col: f2 }) in src.trees().iter().enumerate() {
        if i == 0 {
            let c = if n > 2 {
                f1.innerlines[0].clone()
            } else {
                f1.coupled.clone()
            };
            let coeff = if inv {
                // GenericFusion/complex symbols must restore TensorKit's conjugation here.
                I::r_symbol(right, left, &c)
            } else {
                I::r_symbol(left, right, &c)
            };
            if coeff == 0.0 {
                continue;
            }

            let target_pair = FusionTreePair {
                row: FusionTree::new(
                    uncoupled_prime.to_vec(),
                    f1.coupled.clone(),
                    is_dual_prime.to_vec(),
                    f1.innerlines.clone(),
                    f1.vertices.clone(),
                )?,
                col: f2.clone(),
            };
            let Some(target_index) = dst_index.get(&target_pair).copied() else {
                return Err(Tensor0Error::Message(
                    "artin_braid destination fusion tree pair was not found".to_string(),
                ));
            };
            transform[[target_index, source_index]] = coeff;
            continue;
        }

        let inner_extended = artin_inner_extended(f1);
        let a = &inner_extended[i - 1];
        let b = &f1.uncoupled[i];
        let c = &inner_extended[i];
        let d = &f1.uncoupled[i + 1];
        let e = &inner_extended[i + 1];
        let outputs1 = a.fusion_outputs(d).into_iter().collect::<BTreeSet<_>>();
        let outputs2 = e
            .fusion_outputs(&b.dual())
            .into_iter()
            .collect::<BTreeSet<_>>();

        for c_prime in outputs1.intersection(&outputs2).cloned() {
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
            if coeff == 0.0 {
                continue;
            }

            let mut innerlines = f1.innerlines.clone();
            innerlines[i - 1] = c_prime;
            let target_pair = FusionTreePair {
                row: FusionTree::new(
                    uncoupled_prime.to_vec(),
                    f1.coupled.clone(),
                    is_dual_prime.to_vec(),
                    innerlines,
                    f1.vertices.clone(),
                )?,
                col: f2.clone(),
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

fn _artin_braid_through_unit<I: Sector>(
    f: &FusionTree<I>,
    uncoupled: &[I],
    is_dual: &[bool],
    i: usize,
) -> Result<FusionTree<I>> {
    let mut innerlines = f.innerlines.clone();
    let mut vertices = f.vertices.clone();

    if i > 0 {
        let inner_extended = artin_inner_extended(f);

        let replacement = if f.uncoupled[i] == I::unit() {
            inner_extended[i + 1].clone()
        } else {
            inner_extended[i - 1].clone()
        };
        innerlines[i - 1] = replacement;
        vertices.swap(i - 1, i);
    }

    FusionTree::new(
        uncoupled.to_vec(),
        f.coupled.clone(),
        is_dual.to_vec(),
        innerlines,
        vertices,
    )
}

fn artin_inner_extended<I: Sector>(f: &FusionTree<I>) -> Vec<I> {
    let mut inner_extended = Vec::with_capacity(f.uncoupled.len());
    inner_extended.push(f.uncoupled[0].clone());
    inner_extended.extend(f.innerlines.iter().cloned());
    inner_extended.push(f.coupled.clone());
    inner_extended
}

fn linearized_braid_levels(levels1: &[usize], levels2: &[usize]) -> Vec<usize> {
    let mut levels = levels1.to_vec();
    levels.extend(levels2.iter().rev().copied());
    levels
}
