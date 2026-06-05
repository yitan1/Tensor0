//! Basic fusion-tree reassociation operations.
//!
//! This module owns F-move based operations and explicit reassociation of tree
//! brackets. These operations return tree-basis terms and stay independent of
//! `HomSpace` layout packing.

use crate::error::{Result, Tensor0Error};
use crate::fusion_tree::FusionTree;
use crate::sector::Sector;

pub(crate) fn multi_fmove<I: Sector>(tree: &FusionTree<I>) -> Result<Vec<(FusionTree<I>, f64)>> {
    let arity = tree.uncoupled.len();
    match arity {
        0 => Err(Tensor0Error::Message(
            "multi_Fmove requires at least one uncoupled sector".to_string(),
        )),
        1 => Ok(vec![(
            FusionTree::new(vec![], I::unit(), vec![], vec![], vec![])?,
            1.0,
        )]),
        2 => Ok(vec![(
            FusionTree::new(
                vec![tree.uncoupled[1].clone()],
                tree.uncoupled[1].clone(),
                vec![tree.is_dual[1]],
                vec![],
                vec![],
            )?,
            1.0,
        )]),
        _ => {
            let a = tree.uncoupled[0].clone();
            let tail_uncoupled = tree.uncoupled[1..].to_vec();
            let tail_is_dual = tree.is_dual[1..].to_vec();
            let mut trees = vec![FusionTree::new(
                tail_uncoupled,
                I::unit(),
                tail_is_dual,
                vec![I::unit(); arity - 3],
                vec![0; arity - 2],
            )?];

            for k in 2..arity {
                let mut next_trees = Vec::new();
                let (_left, d, _vertex) = vertex_info(tree, k + 1);
                let c = &tree.uncoupled[k];
                for candidate in trees {
                    let (b, _e, _candidate_vertex) = vertex_info(&candidate, k);
                    for e_prime in b.fusion_outputs(c) {
                        if I::n_symbol(&a.dual(), &d, &e_prime) == 0 {
                            continue;
                        }

                        let mut next = candidate.clone();
                        if k == arity - 1 {
                            next.coupled = e_prime;
                        } else {
                            next.innerlines[k - 2] = e_prime;
                        }
                        next_trees.push(next);
                    }
                }
                trees = next_trees;
            }

            let mut terms = Vec::new();
            for candidate in trees {
                let coeff = multi_associator(tree, &candidate)?;
                if coeff != 0.0 {
                    terms.push((candidate, coeff));
                }
            }
            Ok(terms)
        }
    }
}

pub(crate) fn multi_fmove_inv<I: Sector>(
    a: &I,
    c: &I,
    tree: &FusionTree<I>,
    is_dual_a: bool,
) -> Result<Vec<(FusionTree<I>, f64)>> {
    if I::n_symbol(a, &tree.coupled, c) == 0 {
        return Err(Tensor0Error::Message(
            "cannot fuse sectors for inverse multi_Fmove".to_string(),
        ));
    }

    let arity = tree.uncoupled.len();
    match arity {
        0 => Ok(vec![(
            FusionTree::new(vec![a.clone()], c.clone(), vec![is_dual_a], vec![], vec![])?,
            1.0,
        )]),
        1 => Ok(vec![(
            FusionTree::new(
                vec![a.clone(), tree.uncoupled[0].clone()],
                c.clone(),
                vec![is_dual_a, tree.is_dual[0]],
                vec![],
                vec![0],
            )?,
            1.0,
        )]),
        _ => {
            let mut uncoupled = Vec::with_capacity(arity + 1);
            uncoupled.push(a.clone());
            uncoupled.extend_from_slice(&tree.uncoupled);
            let mut is_dual = Vec::with_capacity(arity + 1);
            is_dual.push(is_dual_a);
            is_dual.extend_from_slice(&tree.is_dual);
            let mut trees = vec![FusionTree::new(
                uncoupled,
                c.clone(),
                is_dual,
                vec![I::unit(); arity - 1],
                vec![0; arity],
            )?];

            for k in (2..=arity).rev() {
                let c_sector = &tree.uncoupled[k - 1];
                let (b, _e_prime, _tree_vertex) = vertex_info(tree, k);
                let mut next_trees = Vec::new();

                for candidate in trees {
                    let (_left, d, _candidate_vertex) = vertex_info(&candidate, k + 1);
                    for e in a.fusion_outputs(&b) {
                        if I::n_symbol(&e, c_sector, &d) == 0 {
                            continue;
                        }

                        let mut next = candidate.clone();
                        next.innerlines[k - 2] = e;
                        next_trees.push(next);
                    }
                }
                trees = next_trees;
            }

            let mut terms = Vec::new();
            for candidate in trees {
                // GenericFusion/complex symbols must restore TensorKit's conjugation here.
                let coeff = multi_associator(&candidate, tree)?;
                if coeff != 0.0 {
                    terms.push((candidate, coeff));
                }
            }
            Ok(terms)
        }
    }
}

fn multi_associator<I: Sector>(long: &FusionTree<I>, short: &FusionTree<I>) -> Result<f64> {
    let arity = long.uncoupled.len();
    if short.uncoupled.len() + 1 != arity {
        return Ok(0.0);
    }
    if long.uncoupled[1..] != short.uncoupled[..] || long.is_dual[1..] != short.is_dual[..] {
        return Ok(0.0);
    }

    let a = &long.uncoupled[0];
    let mut coeff = 1.0;
    for k in 2..arity {
        let c = &long.uncoupled[k];
        let (_left, d, _vertex) = vertex_info(long, k + 1);
        let (b, e_prime, _short_vertex) = vertex_info(short, k);
        let (_previous, e, _long_vertex) = vertex_info(long, k);
        coeff *= I::f_symbol(a, &b, c, &d, &e, &e_prime)?;
    }
    Ok(coeff)
}

fn vertex_info<I: Sector>(tree: &FusionTree<I>, k: usize) -> (I, I, usize) {
    let left = if k == 2 {
        tree.uncoupled[0].clone()
    } else {
        tree.innerlines[k - 3].clone()
    };
    let output = if k == tree.uncoupled.len() {
        tree.coupled.clone()
    } else {
        tree.innerlines[k - 2].clone()
    };
    let vertex = tree.vertices[k - 2];
    (left, output, vertex)
}
