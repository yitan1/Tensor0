use std::collections::{BTreeSet, HashMap};

use ndarray::{ArrayD, Axis, Dimension, IxDyn};

use crate::error::{Result, Tensor0Error};
use crate::sector::{fusion_sectors, FusionStyle, Sector};
use crate::space::HomSpace;

use super::iterator::enumerate_fusion_trees;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FusionTree<I: Sector> {
    pub uncoupled: Vec<I>,
    pub coupled: I,
    pub is_dual: Vec<bool>,
    pub innerlines: Vec<I>,
    pub vertices: Vec<usize>,
}

impl<I: Sector> FusionTree<I> {
    pub(crate) fn new(
        uncoupled: Vec<I>,
        coupled: I,
        is_dual: Vec<bool>,
        innerlines: Vec<I>,
        vertices: Vec<usize>,
    ) -> Result<Self> {
        if I::fusion_style() != FusionStyle::GenericFusion && vertices.iter().any(|&v| v != 0) {
            return Err(Tensor0Error::Message(
                "multiplicity-free fusion tree vertices must be canonical zero".to_string(),
            ));
        }

        Ok(FusionTree {
            uncoupled,
            coupled,
            is_dual,
            innerlines,
            vertices,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FusionTreePair<I: Sector> {
    pub row: FusionTree<I>,
    pub col: FusionTree<I>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct FusionTreeBlock<I: Sector> {
    trees: Vec<FusionTreePair<I>>,
}

impl<I: Sector> FusionTreeBlock<I> {
    pub(crate) fn new(
        row_uncoupled: Vec<I>,
        row_is_dual: Vec<bool>,
        col_uncoupled: Vec<I>,
        col_is_dual: Vec<bool>,
    ) -> Result<Self> {
        if row_uncoupled.len() != row_is_dual.len() || col_uncoupled.len() != col_is_dual.len() {
            return Err(Tensor0Error::Message(
                "fusion tree block arity mismatch".to_string(),
            ));
        }

        let row_outputs = fusion_sectors(&row_uncoupled)
            .into_iter()
            .collect::<BTreeSet<_>>();
        let col_outputs = fusion_sectors(&col_uncoupled)
            .into_iter()
            .collect::<BTreeSet<_>>();
        let mut trees = Vec::new();

        for coupled in row_outputs.intersection(&col_outputs) {
            let row_trees = enumerate_fusion_trees(&row_uncoupled, &row_is_dual, coupled)?;
            let col_trees = enumerate_fusion_trees(&col_uncoupled, &col_is_dual, coupled)?;
            for row in &row_trees {
                for col in &col_trees {
                    trees.push(FusionTreePair {
                        row: row.clone(),
                        col: col.clone(),
                    });
                }
            }
        }

        Ok(FusionTreeBlock { trees })
    }

    pub(crate) fn row_uncoupled(&self) -> &[I] {
        &self.trees[0].row.uncoupled
    }

    pub(crate) fn row_is_dual(&self) -> &[bool] {
        &self.trees[0].row.is_dual
    }

    pub(crate) fn col_uncoupled(&self) -> &[I] {
        &self.trees[0].col.uncoupled
    }

    pub(crate) fn col_is_dual(&self) -> &[bool] {
        &self.trees[0].col.is_dual
    }

    pub(crate) fn trees(&self) -> &[FusionTreePair<I>] {
        &self.trees
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.trees.is_empty()
    }

    pub(crate) fn numout(&self) -> usize {
        self.trees[0].row.uncoupled.len()
    }

    pub(crate) fn numin(&self) -> usize {
        self.trees[0].col.uncoupled.len()
    }

    pub(crate) fn numind(&self) -> usize {
        self.numout() + self.numin()
    }

    pub(crate) fn index_map(&self) -> HashMap<FusionTreePair<I>, usize> {
        self.trees
            .iter()
            .cloned()
            .enumerate()
            .map(|(index, pair)| (pair, index))
            .collect()
    }
}

pub(crate) fn fusion_blocks<I: Sector>(space: &HomSpace<I>) -> Result<Vec<FusionTreeBlock<I>>> {
    let codomain_tuples = space.codomain().sector_tuples()?;
    let domain_tuples = space.domain().sector_tuples()?;
    let mut blocks = Vec::new();

    for domain in &domain_tuples {
        for codomain in &codomain_tuples {
            let block = FusionTreeBlock::new(
                codomain.sectors.clone(),
                codomain.is_dual.clone(),
                domain.sectors.clone(),
                domain.is_dual.clone(),
            )?;
            if !block.is_empty() {
                blocks.push(block);
            }
        }
    }

    Ok(blocks)
}

pub fn fusiontree_tensor<I: Sector>(tree: &FusionTree<I>) -> Result<ArrayD<f64>> {
    validate_tree_shape(tree)?;

    match tree.uncoupled.len() {
        0 => fusiontree0_tensor(tree),
        1 => fusiontree1_tensor(tree),
        2 => fusiontree2_tensor(tree),
        _ => fusiontree_n_tensor(tree),
    }
}

pub fn fusiontree_pair_tensor<I: Sector>(
    row: &FusionTree<I>,
    col: &FusionTree<I>,
) -> Result<ArrayD<f64>> {
    if row.coupled != col.coupled {
        return Err(Tensor0Error::Message(
            "fusion tree pair requires matching coupled sectors".to_string(),
        ));
    }

    let row_tensor = fusiontree_tensor(row)?;
    let col_tensor = fusiontree_tensor(col)?;
    let coupled_dim = row.coupled.quantum_dim();
    let row_outer = row_tensor.len() / coupled_dim;
    let col_outer = col_tensor.len() / coupled_dim;

    let row_matrix = row_tensor
        .into_shape_with_order((row_outer, coupled_dim))
        .map_err(|err| Tensor0Error::Message(format!("row fusion tree reshape failed: {err}")))?;
    let col_matrix = col_tensor
        .into_shape_with_order((col_outer, coupled_dim))
        .map_err(|err| {
            Tensor0Error::Message(format!("column fusion tree reshape failed: {err}"))
        })?;
    // GenericFusion/complex symbols must restore TensorKit's conjugation here.
    let product = row_matrix
        .dot(&col_matrix.t())
        .as_standard_layout()
        .to_owned();
    let output_shape = row
        .uncoupled
        .iter()
        .chain(col.uncoupled.iter())
        .map(Sector::quantum_dim)
        .collect::<Vec<_>>();

    product
        .into_shape_with_order(IxDyn(&output_shape))
        .map_err(|err| Tensor0Error::Message(format!("fusion tree pair reshape failed: {err}")))
}

fn fusiontree0_tensor<I: Sector>(tree: &FusionTree<I>) -> Result<ArrayD<f64>> {
    if tree.coupled != I::unit() {
        return Err(Tensor0Error::Message(
            "empty fusion tree requires the unit coupled sector".to_string(),
        ));
    }
    Ok(ArrayD::from_elem(IxDyn(&[1]), 1.0))
}

fn fusiontree1_tensor<I: Sector>(tree: &FusionTree<I>) -> Result<ArrayD<f64>> {
    let sector = &tree.uncoupled[0];
    if sector != &tree.coupled {
        return Err(Tensor0Error::Message(
            "one-leg fusion tree requires uncoupled and coupled sectors to match".to_string(),
        ));
    }

    if tree.is_dual[0] {
        return dual_isomorphism(sector);
    }

    let dim = sector.quantum_dim();
    let mut tensor = ArrayD::zeros(IxDyn(&[dim, dim]));
    for index in 0..dim {
        tensor[IxDyn(&[index, index])] = 1.0;
    }
    Ok(tensor)
}

fn fusiontree2_tensor<I: Sector>(tree: &FusionTree<I>) -> Result<ArrayD<f64>> {
    let tensor = I::fusion_tensor(&tree.uncoupled[0], &tree.uncoupled[1], &tree.coupled)?;
    let vertex = if I::fusion_style() == FusionStyle::GenericFusion {
        let vertex = tree.vertices[0];
        if vertex >= tensor.shape()[3] {
            return Err(Tensor0Error::Message(
                "fusion tree vertex index out of range".to_string(),
            ));
        }
        vertex
    } else {
        0
    };
    let mut tensor = tensor.index_axis(Axis(3), vertex).to_owned().into_dyn();
    for axis in 0..2 {
        if tree.is_dual[axis] {
            tensor = apply_dual_isomorphism(tensor, axis, &tree.uncoupled[axis])?;
        }
    }
    Ok(tensor)
}

fn fusiontree_n_tensor<I: Sector>(tree: &FusionTree<I>) -> Result<ArrayD<f64>> {
    let first_innerline = tree.innerlines[0].clone();
    let first_tree = FusionTree {
        uncoupled: tree.uncoupled[..2].to_vec(),
        coupled: first_innerline.clone(),
        is_dual: tree.is_dual[..2].to_vec(),
        innerlines: vec![],
        vertices: vec![tree.vertices[0]],
    };

    let mut tail_uncoupled = Vec::with_capacity(tree.uncoupled.len() - 1);
    tail_uncoupled.push(first_innerline);
    tail_uncoupled.extend_from_slice(&tree.uncoupled[2..]);

    let mut tail_is_dual = Vec::with_capacity(tree.is_dual.len() - 1);
    tail_is_dual.push(false);
    tail_is_dual.extend_from_slice(&tree.is_dual[2..]);

    let tail_tree = FusionTree {
        uncoupled: tail_uncoupled,
        coupled: tree.coupled.clone(),
        is_dual: tail_is_dual,
        innerlines: tree.innerlines[1..].to_vec(),
        vertices: tree.vertices[1..].to_vec(),
    };

    let first = fusiontree_tensor(&first_tree)?;
    let tail = fusiontree_tensor(&tail_tree)?;
    tensordot_last_first(&first, &tail)
}

fn dual_isomorphism<I: Sector>(sector: &I) -> Result<ArrayD<f64>> {
    let tensor = I::fusion_tensor(&sector.dual(), sector, &I::unit())?;
    let dim = sector.quantum_dim();
    let scale = (dim as f64).sqrt();
    let mut z = ArrayD::zeros(IxDyn(&[dim, dim]));

    for left in 0..dim {
        for right in 0..dim {
            z[IxDyn(&[left, right])] = scale * tensor[[left, right, 0, 0]];
        }
    }

    Ok(z)
}

fn apply_dual_isomorphism<I: Sector>(
    tensor: ArrayD<f64>,
    axis: usize,
    sector: &I,
) -> Result<ArrayD<f64>> {
    let z = dual_isomorphism(sector)?;
    let shape = tensor.shape().to_vec();
    let axis_dim = shape[axis];

    let mut output = ArrayD::zeros(IxDyn(&shape));
    for (output_index, value) in output.indexed_iter_mut() {
        let output_index = output_index.slice();
        let mut input_index = output_index.to_vec();
        let transformed_index = output_index[axis];
        let mut total = 0.0;
        for source_index in 0..axis_dim {
            input_index[axis] = source_index;
            total += z[IxDyn(&[transformed_index, source_index])] * tensor[IxDyn(&input_index)];
        }
        *value = total;
    }

    Ok(output)
}

fn tensordot_last_first(left: &ArrayD<f64>, right: &ArrayD<f64>) -> Result<ArrayD<f64>> {
    let left_shape = left.shape();
    let right_shape = right.shape();
    let shared = left_shape[left.ndim() - 1];
    let left_outer = left.len() / shared;
    let right_outer = right.len() / shared;
    let left_matrix = left
        .clone()
        .into_shape_with_order((left_outer, shared))
        .map_err(|err| Tensor0Error::Message(format!("tensordot left reshape failed: {err}")))?;
    let right_matrix = right
        .clone()
        .into_shape_with_order((shared, right_outer))
        .map_err(|err| Tensor0Error::Message(format!("tensordot right reshape failed: {err}")))?;
    let product = left_matrix.dot(&right_matrix);

    let mut output_shape = left_shape[..left_shape.len() - 1].to_vec();
    output_shape.extend_from_slice(&right_shape[1..]);
    product
        .into_shape_with_order(IxDyn(&output_shape))
        .map_err(|err| Tensor0Error::Message(format!("tensordot output reshape failed: {err}")))
}

fn validate_tree_shape<I: Sector>(tree: &FusionTree<I>) -> Result<()> {
    let arity = tree.uncoupled.len();
    if tree.is_dual.len() != arity {
        return Err(Tensor0Error::Message(
            "fusion tree dual flag arity mismatch".to_string(),
        ));
    }

    let expected_innerlines = arity.saturating_sub(2);
    if tree.innerlines.len() != expected_innerlines {
        return Err(Tensor0Error::Message(
            "fusion tree innerline arity mismatch".to_string(),
        ));
    }

    let expected_vertices = if arity < 2 { 0 } else { arity - 1 };
    if tree.vertices.len() != expected_vertices {
        return Err(Tensor0Error::Message(
            "fusion tree vertex arity mismatch".to_string(),
        ));
    }

    Ok(())
}
