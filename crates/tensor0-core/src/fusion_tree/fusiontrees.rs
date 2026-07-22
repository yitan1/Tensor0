use std::collections::{BTreeSet, HashMap};
use std::fmt;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::Arc;

use ndarray::{ArrayD, Axis, IxDyn};

use crate::error::{Result, Tensor0Error};
use crate::sector::{fusion_sectors, FusionStyle, Sector};
use crate::space::HomSpace;

use super::auxiliary::tensordot;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct FusionTreeData<I: Sector> {
    uncoupled: Vec<I>,
    coupled: I,
    is_dual: Vec<bool>,
    innerlines: Vec<I>,
    vertices: Vec<usize>,
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct FusionTree<I: Sector> {
    data: Arc<FusionTreeData<I>>,
}

impl<I: Sector + fmt::Debug> fmt::Debug for FusionTree<I> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("FusionTree")
            .field("uncoupled", &self.data.uncoupled)
            .field("coupled", &self.data.coupled)
            .field("is_dual", &self.data.is_dual)
            .field("innerlines", &self.data.innerlines)
            .field("vertices", &self.data.vertices)
            .finish()
    }
}

impl<I: Sector> FusionTree<I> {
    pub fn new(
        uncoupled: Vec<I>,
        coupled: I,
        is_dual: Vec<bool>,
        innerlines: Vec<I>,
        vertices: Vec<usize>,
    ) -> Result<Self> {
        let tree = FusionTree {
            data: Arc::new(FusionTreeData {
                uncoupled,
                coupled,
                is_dual,
                innerlines,
                vertices,
            }),
        };

        validate_tree_shape(&tree)?;

        if I::fusion_style() != FusionStyle::GenericFusion
            && tree.vertices().iter().any(|&v| v != 0)
        {
            return Err(Tensor0Error::Message(
                "multiplicity-free fusion tree vertices must be canonical zero".to_string(),
            ));
        }

        Ok(tree)
    }

    pub fn uncoupled(&self) -> &[I] {
        &self.data.uncoupled
    }

    pub fn coupled(&self) -> &I {
        &self.data.coupled
    }

    pub fn is_dual(&self) -> &[bool] {
        &self.data.is_dual
    }

    pub fn innerlines(&self) -> &[I] {
        &self.data.innerlines
    }

    pub fn vertices(&self) -> &[usize] {
        &self.data.vertices
    }

    pub(crate) fn coupled_mut(&mut self) -> &mut I {
        &mut Arc::make_mut(&mut self.data).coupled
    }

    pub(crate) fn is_dual_mut(&mut self) -> &mut [bool] {
        &mut Arc::make_mut(&mut self.data).is_dual
    }

    pub(crate) fn innerlines_mut(&mut self) -> &mut [I] {
        &mut Arc::make_mut(&mut self.data).innerlines
    }

    fn into_data(self) -> FusionTreeData<I> {
        Arc::unwrap_or_clone(self.data)
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

enum FusionTreeBlockIndexData<I: Sector> {
    Unique,
    SimpleLinear,
    SimpleFingerprints(HashMap<u64, usize>),
    Generic(HashMap<FusionTreePair<I>, usize>),
}

const LINEAR_FUSION_TREE_BLOCK_MAX: usize = 16;

pub(crate) struct FusionTreeBlockIndex<'a, I: Sector> {
    block: &'a FusionTreeBlock<I>,
    data: FusionTreeBlockIndexData<I>,
}

impl<'a, I: Sector> FusionTreeBlockIndex<'a, I> {
    fn new(block: &'a FusionTreeBlock<I>) -> Self {
        let data = match I::fusion_style() {
            FusionStyle::UniqueFusion => {
                debug_assert!(block.trees.len() <= 1);
                FusionTreeBlockIndexData::Unique
            }
            FusionStyle::SimpleFusion => {
                if block.trees.len() <= LINEAR_FUSION_TREE_BLOCK_MAX {
                    FusionTreeBlockIndexData::SimpleLinear
                } else {
                    FusionTreeBlockIndexData::SimpleFingerprints(build_fingerprint_index(
                        block,
                        block_local_pair_fingerprint::<I>,
                    ))
                }
            }
            FusionStyle::GenericFusion => FusionTreeBlockIndexData::Generic(
                block
                    .trees
                    .iter()
                    .cloned()
                    .enumerate()
                    .map(|(index, pair)| (pair, index))
                    .collect(),
            ),
        };
        Self { block, data }
    }

    pub(crate) fn get(&self, pair: &FusionTreePair<I>) -> Option<usize> {
        #[cfg(debug_assertions)]
        {
            if matches!(
                &self.data,
                FusionTreeBlockIndexData::SimpleLinear
                    | FusionTreeBlockIndexData::SimpleFingerprints(_)
            ) {
                if let Some(first) = self.block.trees.first() {
                    debug_assert!(same_block_constants(first, pair));
                    debug_assert!(pair.row.coupled() == pair.col.coupled());
                }
            }
        }

        match &self.data {
            FusionTreeBlockIndexData::Unique => self
                .block
                .trees
                .first()
                .filter(|expected| *expected == pair)
                .map(|_| 0),
            FusionTreeBlockIndexData::SimpleLinear => self
                .block
                .trees
                .iter()
                .position(|candidate| block_local_pair_equal(candidate, pair)),
            FusionTreeBlockIndexData::SimpleFingerprints(index) => {
                lookup_fingerprint_index(self.block, index, pair, block_local_pair_fingerprint::<I>)
            }
            FusionTreeBlockIndexData::Generic(index) => index.get(pair).copied(),
        }
    }
}

fn block_local_pair_fingerprint<I: Sector>(pair: &FusionTreePair<I>) -> u64 {
    let mut hasher = DefaultHasher::new();
    pair.row.coupled().hash(&mut hasher);
    pair.row.innerlines().hash(&mut hasher);
    pair.col.innerlines().hash(&mut hasher);
    hasher.finish()
}

fn block_local_pair_equal<I: Sector>(left: &FusionTreePair<I>, right: &FusionTreePair<I>) -> bool {
    left.row.coupled() == right.row.coupled()
        && left.col.coupled() == right.col.coupled()
        && left.row.innerlines() == right.row.innerlines()
        && left.col.innerlines() == right.col.innerlines()
}

#[cfg(debug_assertions)]
fn same_block_constants<I: Sector>(left: &FusionTreePair<I>, right: &FusionTreePair<I>) -> bool {
    left.row.uncoupled() == right.row.uncoupled()
        && left.row.is_dual() == right.row.is_dual()
        && left.col.uncoupled() == right.col.uncoupled()
        && left.col.is_dual() == right.col.is_dual()
}

fn build_fingerprint_index<I, F>(block: &FusionTreeBlock<I>, fingerprint: F) -> HashMap<u64, usize>
where
    I: Sector,
    F: Fn(&FusionTreePair<I>) -> u64,
{
    let mut index = HashMap::with_capacity(block.trees.len());
    for (position, pair) in block.trees.iter().enumerate() {
        index.entry(fingerprint(pair)).or_insert(position);
    }
    index
}

fn lookup_fingerprint_index<I, F>(
    block: &FusionTreeBlock<I>,
    index: &HashMap<u64, usize>,
    pair: &FusionTreePair<I>,
    fingerprint: F,
) -> Option<usize>
where
    I: Sector,
    F: Fn(&FusionTreePair<I>) -> u64,
{
    let value = fingerprint(pair);
    let position = *index.get(&value)?;
    if block_local_pair_equal(&block.trees[position], pair) {
        return Some(position);
    }
    // A fingerprint only narrows the common path. An actual collision falls
    // back to an exact reduced-key scan without retaining collision storage.
    block
        .trees
        .iter()
        .position(|candidate| block_local_pair_equal(candidate, pair))
}

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
            let last = &uncoupled[uncoupled.len() - 1];
            let front_uncoupled = &uncoupled[..uncoupled.len() - 1];
            let front_is_dual = &is_dual[..is_dual.len() - 1];
            let uncoupled_vec = uncoupled.to_vec();
            let is_dual_vec = is_dual.to_vec();
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
                    let FusionTreeData {
                        mut innerlines,
                        mut vertices,
                        ..
                    } = rest.into_data();
                    innerlines.push(innerline.clone());
                    vertices.push(0);

                    trees.push(FusionTree::new(
                        uncoupled_vec.clone(),
                        coupled.clone(),
                        is_dual_vec.clone(),
                        innerlines,
                        vertices,
                    )?);
                }
            }

            Ok(trees)
        }
    }
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
        self.trees[0].row.uncoupled()
    }

    pub(crate) fn row_is_dual(&self) -> &[bool] {
        self.trees[0].row.is_dual()
    }

    pub(crate) fn col_uncoupled(&self) -> &[I] {
        self.trees[0].col.uncoupled()
    }

    pub(crate) fn col_is_dual(&self) -> &[bool] {
        self.trees[0].col.is_dual()
    }

    pub(crate) fn trees(&self) -> &[FusionTreePair<I>] {
        &self.trees
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.trees.is_empty()
    }

    pub(crate) fn numout(&self) -> usize {
        self.trees[0].row.uncoupled().len()
    }

    pub(crate) fn numin(&self) -> usize {
        self.trees[0].col.uncoupled().len()
    }

    pub(crate) fn numind(&self) -> usize {
        self.numout() + self.numin()
    }

    pub(crate) fn tree_index(&self) -> FusionTreeBlockIndex<'_, I> {
        FusionTreeBlockIndex::new(self)
    }
}

pub(crate) fn fusion_blocks<I: Sector>(space: &HomSpace<I>) -> Result<Vec<FusionTreeBlock<I>>> {
    let codomain = space.codomain().sector_support();
    let domain = space.domain().sector_support();
    let codomain_tuples = codomain.sector_tuples();
    let domain_tuples = domain.sector_tuples();
    let mut blocks = Vec::new();

    for domain_tuple in &domain_tuples {
        for codomain_tuple in &codomain_tuples {
            let block = FusionTreeBlock::new(
                codomain_tuple.clone(),
                codomain.is_dual.clone(),
                domain_tuple.clone(),
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
    match tree.uncoupled().len() {
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
    if row.coupled() != col.coupled() {
        return Err(Tensor0Error::Message(
            "fusion tree pair requires matching coupled sectors".to_string(),
        ));
    }

    let row_tensor = fusiontree_tensor(row)?;
    let col_tensor = fusiontree_tensor(col)?;
    let coupled_dim = row.coupled().quantum_dim();
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
        .uncoupled()
        .iter()
        .chain(col.uncoupled().iter())
        .map(Sector::quantum_dim)
        .collect::<Vec<_>>();

    product
        .into_shape_with_order(IxDyn(&output_shape))
        .map_err(|err| Tensor0Error::Message(format!("fusion tree pair reshape failed: {err}")))
}

fn fusiontree0_tensor<I: Sector>(tree: &FusionTree<I>) -> Result<ArrayD<f64>> {
    if tree.coupled() != &I::unit() {
        return Err(Tensor0Error::Message(
            "empty fusion tree requires the unit coupled sector".to_string(),
        ));
    }
    Ok(ArrayD::from_elem(IxDyn(&[1]), 1.0))
}

fn fusiontree1_tensor<I: Sector>(tree: &FusionTree<I>) -> Result<ArrayD<f64>> {
    let sector = &tree.uncoupled()[0];
    if sector != tree.coupled() {
        return Err(Tensor0Error::Message(
            "one-leg fusion tree requires uncoupled and coupled sectors to match".to_string(),
        ));
    }

    if tree.is_dual()[0] {
        return z_isomorphism_matrix(sector);
    }

    let dim = sector.quantum_dim();
    let mut tensor = ArrayD::zeros(IxDyn(&[dim, dim]));
    for index in 0..dim {
        tensor[IxDyn(&[index, index])] = 1.0;
    }
    Ok(tensor)
}

fn fusiontree2_tensor<I: Sector>(tree: &FusionTree<I>) -> Result<ArrayD<f64>> {
    let tensor = I::fusion_tensor(&tree.uncoupled()[0], &tree.uncoupled()[1], tree.coupled())?;
    let vertex = if I::fusion_style() == FusionStyle::GenericFusion {
        let vertex = tree.vertices()[0];
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
    if tree.is_dual()[0] {
        let z = z_isomorphism_matrix(&tree.uncoupled()[0])?;
        tensor = tensordot(&z, &tensor, &[1], &[0])?;
    }
    if tree.is_dual()[1] {
        let z = z_isomorphism_matrix(&tree.uncoupled()[1])?;
        tensor = tensordot(&z, &tensor, &[1], &[1])?;
        tensor = tensor
            .permuted_axes(IxDyn(&[1, 0, 2]))
            .as_standard_layout()
            .into_owned();
    }
    Ok(tensor)
}

fn fusiontree_n_tensor<I: Sector>(tree: &FusionTree<I>) -> Result<ArrayD<f64>> {
    let first_innerline = tree.innerlines()[0].clone();
    let first_tree = FusionTree::new(
        tree.uncoupled()[..2].to_vec(),
        first_innerline.clone(),
        tree.is_dual()[..2].to_vec(),
        vec![],
        vec![tree.vertices()[0]],
    )?;

    let mut tail_uncoupled = Vec::with_capacity(tree.uncoupled().len() - 1);
    tail_uncoupled.push(first_innerline);
    tail_uncoupled.extend_from_slice(&tree.uncoupled()[2..]);

    let mut tail_is_dual = Vec::with_capacity(tree.is_dual().len() - 1);
    tail_is_dual.push(false);
    tail_is_dual.extend_from_slice(&tree.is_dual()[2..]);

    let tail_tree = FusionTree::new(
        tail_uncoupled,
        tree.coupled().clone(),
        tail_is_dual,
        tree.innerlines()[1..].to_vec(),
        tree.vertices()[1..].to_vec(),
    )?;

    let first = fusiontree_tensor(&first_tree)?;
    let tail = fusiontree_tensor(&tail_tree)?;
    tensordot(&first, &tail, &[2], &[0])
}

fn z_isomorphism_matrix<I: Sector>(sector: &I) -> Result<ArrayD<f64>> {
    let tensor = I::fusion_tensor(&sector.dual(), sector, &I::unit())?;
    let scale = (sector.quantum_dim() as f64).sqrt();

    Ok(tensor
        .index_axis(Axis(2), 0)
        .index_axis(Axis(2), 0)
        .mapv(|value| scale * value)
        .into_dyn())
}

fn validate_tree_shape<I: Sector>(tree: &FusionTree<I>) -> Result<()> {
    let arity = tree.uncoupled().len();
    if tree.is_dual().len() != arity {
        return Err(Tensor0Error::Message(
            "fusion tree dual flag arity mismatch".to_string(),
        ));
    }

    let expected_innerlines = arity.saturating_sub(2);
    if tree.innerlines().len() != expected_innerlines {
        return Err(Tensor0Error::Message(
            "fusion tree innerline arity mismatch".to_string(),
        ));
    }

    let expected_vertices = if arity < 2 { 0 } else { arity - 1 };
    if tree.vertices().len() != expected_vertices {
        return Err(Tensor0Error::Message(
            "fusion tree vertex arity mismatch".to_string(),
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sector::{SU2Irrep, U1Irrep};

    fn u1(charge2: i64) -> U1Irrep {
        U1Irrep::charge2(charge2).unwrap()
    }

    fn su2(spin2: i64) -> SU2Irrep {
        SU2Irrep::spin2(spin2).unwrap()
    }

    #[test]
    fn enumerate_u1_two_leg_tree_records_unique_path() {
        let uncoupled = vec![u1(1), u1(-1)];

        let trees = enumerate_fusion_trees(&uncoupled, &[false, false], &u1(0)).unwrap();

        assert_eq!(
            trees,
            vec![FusionTree::new(uncoupled, u1(0), vec![false, false], vec![], vec![0],).unwrap()]
        );
    }

    #[test]
    fn enumerate_su2_four_half_to_unit_records_right_recursive_innerlines() {
        let half = su2(1);
        let uncoupled = vec![half, half, half, half];

        let trees =
            enumerate_fusion_trees(&uncoupled, &[false, false, false, false], &su2(0)).unwrap();

        let innerlines = trees
            .iter()
            .map(|tree| tree.innerlines().to_vec())
            .collect::<Vec<_>>();
        assert_eq!(innerlines, vec![vec![su2(0), su2(1)], vec![su2(2), su2(1)]]);
        assert!(trees.iter().all(|tree| tree.coupled() == &su2(0)));
        assert!(trees.iter().all(|tree| tree.vertices() == vec![0, 0, 0]));
    }

    #[test]
    fn fusion_tree_block_orders_su2_pairs_by_coupled_then_row_then_column() {
        let half = su2(1);
        let uncoupled = vec![half, half, half, half];

        let block =
            FusionTreeBlock::new(uncoupled.clone(), vec![false; 4], uncoupled, vec![false; 4])
                .unwrap();

        assert!(block.trees().len() >= 4);
        let first_pairs = &block.trees()[..4];
        assert!(first_pairs
            .iter()
            .all(|pair| pair.row.coupled() == &su2(0) && pair.col.coupled() == &su2(0)));

        let innerline_pairs = first_pairs
            .iter()
            .map(|pair| {
                (
                    pair.row.innerlines().to_vec(),
                    pair.col.innerlines().to_vec(),
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(
            innerline_pairs,
            vec![
                (vec![su2(0), su2(1)], vec![su2(0), su2(1)]),
                (vec![su2(0), su2(1)], vec![su2(2), su2(1)]),
                (vec![su2(2), su2(1)], vec![su2(0), su2(1)]),
                (vec![su2(2), su2(1)], vec![su2(2), su2(1)]),
            ]
        );
    }

    #[test]
    fn fusion_tree_block_index_resolves_forced_fingerprint_collisions() {
        let half = su2(1);
        let uncoupled = vec![half, half, half, half];
        let block =
            FusionTreeBlock::new(uncoupled.clone(), vec![false; 4], uncoupled, vec![false; 4])
                .unwrap();
        let index = build_fingerprint_index(&block, |_| 0);

        assert_eq!(index.len(), 1);
        for (expected, pair) in block.trees().iter().enumerate() {
            assert_eq!(
                lookup_fingerprint_index(&block, &index, pair, |_| 0),
                Some(expected)
            );
        }
    }

    #[test]
    fn fusion_tree_block_index_selects_and_resolves_linear_and_fingerprint_strategies() {
        let half = su2(1);
        for rank in [4, 5] {
            let uncoupled = vec![half; rank];
            let block = FusionTreeBlock::new(
                uncoupled.clone(),
                vec![false; rank],
                uncoupled,
                vec![false; rank],
            )
            .unwrap();
            let index = block.tree_index();
            let has_expected_strategy = match rank {
                4 => matches!(&index.data, FusionTreeBlockIndexData::SimpleLinear),
                5 => matches!(&index.data, FusionTreeBlockIndexData::SimpleFingerprints(_)),
                _ => unreachable!(),
            };
            assert!(has_expected_strategy);

            for (expected, pair) in block.trees().iter().enumerate() {
                assert_eq!(index.get(pair), Some(expected));
            }
        }
    }
}
