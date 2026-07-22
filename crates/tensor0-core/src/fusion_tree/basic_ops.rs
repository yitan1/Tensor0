//! Basic fusion-tree reassociation operations.
//!
//! This module owns F-move based operations and explicit reassociation of tree
//! brackets. These operations return tree-basis terms and stay independent of
//! `HomSpace` layout packing.

use crate::error::{Result, Tensor0Error};
use crate::fusion_tree::FusionTree;
use crate::sector::Sector;

/// Splits a canonical fusion tree after its first `prefix_len` uncoupled legs.
pub fn split<I: Sector>(
    tree: &FusionTree<I>,
    prefix_len: usize,
) -> Result<(FusionTree<I>, FusionTree<I>)> {
    let arity = tree.uncoupled().len();
    if prefix_len > arity {
        return Err(Tensor0Error::Message(
            "fusion tree split exceeds tree arity".to_string(),
        ));
    }

    let boundary = if prefix_len == 0 {
        I::unit()
    } else if prefix_len == arity {
        tree.coupled().clone()
    } else if prefix_len == 1 {
        tree.uncoupled()[0].clone()
    } else {
        tree.innerlines()[prefix_len - 2].clone()
    };

    let prefix = FusionTree::new(
        tree.uncoupled()[..prefix_len].to_vec(),
        boundary.clone(),
        tree.is_dual()[..prefix_len].to_vec(),
        tree.innerlines()[..prefix_len.saturating_sub(2)].to_vec(),
        tree.vertices()[..prefix_len.saturating_sub(1)].to_vec(),
    )?;

    let mut suffix_uncoupled = Vec::with_capacity(arity - prefix_len + 1);
    suffix_uncoupled.push(boundary);
    suffix_uncoupled.extend_from_slice(&tree.uncoupled()[prefix_len..]);

    let mut suffix_is_dual = Vec::with_capacity(arity - prefix_len + 1);
    suffix_is_dual.push(false);
    suffix_is_dual.extend_from_slice(&tree.is_dual()[prefix_len..]);

    let mut extended_lines = Vec::with_capacity(arity.saturating_add(1));
    if let Some(first) = tree.uncoupled().first() {
        extended_lines.push(first.clone());
    }
    extended_lines.extend_from_slice(tree.innerlines());
    extended_lines.push(tree.coupled().clone());
    let suffix_innerline_count = (arity - prefix_len).saturating_sub(1);
    let suffix_innerlines =
        extended_lines[prefix_len..prefix_len + suffix_innerline_count].to_vec();

    let mut extended_vertices = Vec::with_capacity(arity.saturating_add(1));
    extended_vertices.push(0);
    extended_vertices.extend_from_slice(tree.vertices());
    let suffix_vertices = extended_vertices[prefix_len..arity].to_vec();

    let suffix = FusionTree::new(
        suffix_uncoupled,
        tree.coupled().clone(),
        suffix_is_dual,
        suffix_innerlines,
        suffix_vertices,
    )?;
    Ok((prefix, suffix))
}

/// Joins two canonical fusion trees along their shared, non-dual boundary sector.
pub fn join<I: Sector>(prefix: &FusionTree<I>, suffix: &FusionTree<I>) -> Result<FusionTree<I>> {
    let boundary = suffix.uncoupled().first().ok_or_else(|| {
        Tensor0Error::Message("fusion tree suffix requires a boundary sector".to_string())
    })?;
    if prefix.coupled() != boundary || suffix.is_dual()[0] {
        return Err(Tensor0Error::Message(
            "fusion trees cannot be joined across the requested boundary".to_string(),
        ));
    }

    let prefix_arity = prefix.uncoupled().len();
    let suffix_arity = suffix.uncoupled().len();

    let mut uncoupled = prefix.uncoupled().to_vec();
    uncoupled.extend_from_slice(&suffix.uncoupled()[1..]);
    let mut is_dual = prefix.is_dual().to_vec();
    is_dual.extend_from_slice(&suffix.is_dual()[1..]);

    let innerlines = if prefix_arity == 0 {
        if suffix_arity <= 2 {
            vec![]
        } else {
            suffix.innerlines()[1..].to_vec()
        }
    } else if prefix_arity == 1 {
        suffix.innerlines().to_vec()
    } else if suffix_arity == 1 {
        prefix.innerlines().to_vec()
    } else {
        let mut innerlines = prefix.innerlines().to_vec();
        innerlines.push(prefix.coupled().clone());
        innerlines.extend_from_slice(suffix.innerlines());
        innerlines
    };

    let vertices = if prefix_arity == 0 {
        if suffix_arity <= 1 {
            vec![]
        } else {
            suffix.vertices()[1..].to_vec()
        }
    } else if prefix_arity == 1 {
        suffix.vertices().to_vec()
    } else {
        let mut vertices = prefix.vertices().to_vec();
        vertices.extend_from_slice(suffix.vertices());
        vertices
    };

    FusionTree::new(
        uncoupled,
        suffix.coupled().clone(),
        is_dual,
        innerlines,
        vertices,
    )
}

pub(crate) fn multi_fmove<I: Sector>(tree: &FusionTree<I>) -> Result<Vec<(FusionTree<I>, f64)>> {
    let arity = tree.uncoupled().len();
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
                vec![tree.uncoupled()[1].clone()],
                tree.uncoupled()[1].clone(),
                vec![tree.is_dual()[1]],
                vec![],
                vec![],
            )?,
            1.0,
        )]),
        _ => {
            let a = tree.uncoupled()[0].clone();
            let dual_a = a.dual();
            let tail_uncoupled = tree.uncoupled()[1..].to_vec();
            let tail_is_dual = tree.is_dual()[1..].to_vec();
            let mut trees = vec![FusionTree::new(
                tail_uncoupled,
                I::unit(),
                tail_is_dual,
                vec![I::unit(); arity - 3],
                vec![0; arity - 2],
            )?];

            // Generate candidate trees by moving the first sector to the top vertex.
            for k in 2..arity {
                let mut next_trees = Vec::new();
                let (_, d) = vertex_channels(tree, k + 1);
                let c = &tree.uncoupled()[k];
                for candidate in trees {
                    let (b, _) = vertex_channels(&candidate, k);
                    for e_prime in b.fusion_outputs(c) {
                        if I::n_symbol(&dual_a, d, &e_prime) == 0 {
                            continue;
                        }

                        let mut next = candidate.clone();
                        if k == arity - 1 {
                            *next.coupled_mut() = e_prime;
                        } else {
                            next.innerlines_mut()[k - 2] = e_prime;
                        }
                        next_trees.push(next);
                    }
                }
                trees = next_trees;
            }

            // Evaluate the F-symbol product for each generated tree.
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
    if I::n_symbol(a, tree.coupled(), c) == 0 {
        return Err(Tensor0Error::Message(
            "cannot fuse sectors for inverse multi_Fmove".to_string(),
        ));
    }

    let arity = tree.uncoupled().len();
    match arity {
        0 => Ok(vec![(
            FusionTree::new(vec![a.clone()], c.clone(), vec![is_dual_a], vec![], vec![])?,
            1.0,
        )]),
        1 => Ok(vec![(
            FusionTree::new(
                vec![a.clone(), tree.uncoupled()[0].clone()],
                c.clone(),
                vec![is_dual_a, tree.is_dual()[0]],
                vec![],
                vec![0],
            )?,
            1.0,
        )]),
        _ => {
            let mut uncoupled = Vec::with_capacity(arity + 1);
            uncoupled.push(a.clone());
            uncoupled.extend_from_slice(tree.uncoupled());
            let mut is_dual = Vec::with_capacity(arity + 1);
            is_dual.push(is_dual_a);
            is_dual.extend_from_slice(tree.is_dual());
            let mut trees = vec![FusionTree::new(
                uncoupled,
                c.clone(),
                is_dual,
                vec![I::unit(); arity - 1],
                vec![0; arity],
            )?];

            // Generate candidate trees by fusing the new sector into the tree.
            for k in (2..=arity).rev() {
                let c_sector = &tree.uncoupled()[k - 1];
                let (b, _) = vertex_channels(tree, k);
                let outputs = a.fusion_outputs(b).collect::<Vec<_>>();
                let mut next_trees = Vec::new();

                for candidate in trees {
                    let (_, d) = vertex_channels(&candidate, k + 1);
                    for e in &outputs {
                        if I::n_symbol(e, c_sector, d) == 0 {
                            continue;
                        }

                        let mut next = candidate.clone();
                        next.innerlines_mut()[k - 2] = e.clone();
                        next_trees.push(next);
                    }
                }
                trees = next_trees;
            }

            // Evaluate the inverse F-symbol product for each generated tree.
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
    let arity = long.uncoupled().len();
    if short.uncoupled().len() + 1 != arity {
        return Ok(0.0);
    }
    if long.uncoupled()[1..] != short.uncoupled()[..] || long.is_dual()[1..] != short.is_dual()[..]
    {
        return Ok(0.0);
    }

    let a = &long.uncoupled()[0];
    let mut coeff = 1.0;
    for k in 2..arity {
        let c = &long.uncoupled()[k];
        let (_, d) = vertex_channels(long, k + 1);
        let (b, e_prime) = vertex_channels(short, k);
        let (_, e) = vertex_channels(long, k);
        coeff *= I::f_symbol(a, b, c, d, e, e_prime)?;
    }
    Ok(coeff)
}

fn vertex_channels<I: Sector>(tree: &FusionTree<I>, k: usize) -> (&I, &I) {
    let left = if k == 2 {
        &tree.uncoupled()[0]
    } else {
        &tree.innerlines()[k - 3]
    };
    let output = if k == tree.uncoupled().len() {
        tree.coupled()
    } else {
        &tree.innerlines()[k - 2]
    };
    (left, output)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::fusion_tree::FusionTree;
    use crate::sector::SU2Irrep;

    use super::{join, multi_fmove, multi_fmove_inv, split};

    fn su2(spin2: i64) -> SU2Irrep {
        SU2Irrep::spin2(spin2).unwrap()
    }

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1.0e-12,
            "actual={actual}, expected={expected}",
        );
    }

    fn four_half_to_unit_tree() -> FusionTree<SU2Irrep> {
        let half = su2(1);
        FusionTree::new(
            vec![half, half, half, half],
            su2(0),
            vec![false, false, false, false],
            vec![su2(0), su2(1)],
            vec![0, 0, 0],
        )
        .unwrap()
    }

    #[test]
    fn split_and_join_are_inverse_at_every_boundary() {
        let tree = four_half_to_unit_tree();

        for prefix_len in 0..=tree.uncoupled().len() {
            let (prefix, suffix) = split(&tree, prefix_len).unwrap();
            assert_eq!(join(&prefix, &suffix).unwrap(), tree);
        }

        assert!(split(&tree, tree.uncoupled().len() + 1).is_err());
    }

    #[test]
    fn join_rejects_invalid_boundaries() {
        let tree = four_half_to_unit_tree();
        let (prefix, _) = split(&tree, 2).unwrap();
        let empty = FusionTree::new(vec![], su2(0), vec![], vec![], vec![]).unwrap();
        assert!(join(&prefix, &empty).is_err());

        let (_, incompatible_suffix) = split(&tree, 3).unwrap();
        assert!(join(&prefix, &incompatible_suffix).is_err());
    }

    #[test]
    fn multi_fmove_and_inverse_reconstruct_source_tree() {
        let tree = four_half_to_unit_tree();
        let terms = multi_fmove(&tree).unwrap();
        assert!(!terms.is_empty());
        assert_close(
            terms.iter().map(|(_, coeff)| coeff * coeff).sum::<f64>(),
            1.0,
        );

        let mut residual = HashMap::from([(tree.clone(), -1.0)]);
        for (fmove_tree, fmove_coeff) in terms {
            let inverse_terms = multi_fmove_inv(
                &tree.uncoupled()[0],
                tree.coupled(),
                &fmove_tree,
                tree.is_dual()[0],
            )
            .unwrap();
            assert_close(
                inverse_terms
                    .iter()
                    .map(|(_, coeff)| coeff * coeff)
                    .sum::<f64>(),
                1.0,
            );

            for (candidate, inverse_coeff) in inverse_terms {
                *residual.entry(candidate).or_insert(0.0) += fmove_coeff * inverse_coeff;
            }
        }

        for coeff in residual.values() {
            assert_close(*coeff, 0.0);
        }
    }
}
