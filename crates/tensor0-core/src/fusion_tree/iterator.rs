use crate::error::{Result, Tensor0Error};
use crate::fusion_tree::FusionTree;
use crate::sector::Sector;

pub(crate) fn enumerate_fusion_trees<I: Sector>(
    uncoupled: &[I],
    is_dual: &[bool],
    coupled: &I,
) -> Result<Vec<FusionTree<I>>> {
    if uncoupled.len() != is_dual.len() {
        return Err(Tensor0Error::Message(
            "fusion tree arity mismatch".to_string(),
        ));
    }

    match uncoupled.len() {
        0 => {
            if coupled == &I::unit() {
                Ok(vec![FusionTree::new(
                    vec![],
                    coupled.clone(),
                    vec![],
                    vec![],
                    vec![],
                )?])
            } else {
                Ok(vec![])
            }
        }
        1 => {
            if &uncoupled[0] == coupled {
                Ok(vec![FusionTree::new(
                    vec![uncoupled[0].clone()],
                    coupled.clone(),
                    is_dual.to_vec(),
                    vec![],
                    vec![],
                )?])
            } else {
                Ok(vec![])
            }
        }
        2 => {
            let n = I::n_symbol(&uncoupled[0], &uncoupled[1], coupled);
            if n == 0 {
                return Ok(vec![]);
            }
            if n > 1 {
                return Err(Tensor0Error::Message(
                    "layout only supports multiplicity-free fusion".to_string(),
                ));
            }
            Ok(vec![FusionTree::new(
                uncoupled.to_vec(),
                coupled.clone(),
                is_dual.to_vec(),
                vec![],
                vec![0],
            )?])
        }
        _ => {
            let last = uncoupled
                .last()
                .expect("multi-factor fusion tree has a rightmost sector");
            let front_uncoupled = &uncoupled[..uncoupled.len() - 1];
            let front_is_dual = &is_dual[..is_dual.len() - 1];
            let mut trees = Vec::new();

            for innerline in coupled.fusion_outputs(&last.dual()) {
                let n = I::n_symbol(&innerline, last, coupled);
                if n == 0 {
                    continue;
                }
                if n > 1 {
                    return Err(Tensor0Error::Message(
                        "layout only supports multiplicity-free fusion".to_string(),
                    ));
                }

                for rest in enumerate_fusion_trees(front_uncoupled, front_is_dual, &innerline)? {
                    let mut innerlines = rest.innerlines;
                    innerlines.push(innerline.clone());
                    let mut vertices = rest.vertices;
                    vertices.push(0);

                    trees.push(FusionTree::new(
                        uncoupled.to_vec(),
                        coupled.clone(),
                        is_dual.to_vec(),
                        innerlines,
                        vertices,
                    )?);
                }
            }

            Ok(trees)
        }
    }
}
