use std::collections::BTreeSet;

use crate::error::{Result, Tensor0Error};
use crate::fusion_tree::FusionTree;
use crate::sector::{FusionStyle, Sector};

use super::graded::GradedSpace;
use super::spec::ProductSpaceSpec;

pub(crate) struct ProductSectorSupport<I: Sector> {
    pub(crate) sectors: Vec<Vec<I>>,
    pub(crate) is_dual: Vec<bool>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ProductSpace<I: Sector> {
    factors: Vec<GradedSpace<I>>,
}

impl<I: Sector> ProductSpace<I> {
    pub fn new(factors: Vec<GradedSpace<I>>) -> Self {
        ProductSpace { factors }
    }

    pub fn one() -> Self {
        ProductSpace::new(vec![])
    }

    pub fn from_spec(spec: ProductSpaceSpec) -> Result<Self> {
        let actual = spec.sector_spec.canonicalize()?;
        let expected = I::sector_spec().canonicalize()?;
        if actual != expected {
            return Err(Tensor0Error::SectorSpecMismatch {
                expected: format!("{expected:?}"),
                actual: format!("{actual:?}"),
            });
        }

        let factors = spec
            .factors
            .into_iter()
            .map(GradedSpace::<I>::from_spec)
            .collect::<Result<Vec<_>>>()?;
        Ok(ProductSpace::new(factors))
    }

    pub fn to_spec(&self) -> ProductSpaceSpec {
        product_spec(&self.factors)
    }

    pub fn factors(&self) -> &[GradedSpace<I>] {
        &self.factors
    }

    pub fn len(&self) -> usize {
        self.factors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.factors.is_empty()
    }

    pub fn get(&self, index: usize) -> Option<&GradedSpace<I>> {
        self.factors.get(index)
    }

    pub fn iter(&self) -> std::slice::Iter<'_, GradedSpace<I>> {
        self.factors.iter()
    }

    /// Insert the canonical unit space at a 0-based factor boundary.
    pub fn insert_unit(&self, position: usize, dual: bool) -> Result<Self> {
        if position > self.len() {
            return Err(Tensor0Error::Message(format!(
                "unit insertion boundary {position} is out of range for rank {}",
                self.len(),
            )));
        }

        let mut factors = self.factors.clone();
        let unit = if dual {
            GradedSpace::<I>::unit().dual()
        } else {
            GradedSpace::<I>::unit()
        };
        factors.insert(position, unit);
        Ok(ProductSpace::new(factors))
    }

    /// Remove a canonical unit-space factor at a 0-based index.
    pub fn remove_unit(&self, index: usize) -> Result<Self> {
        let Some(factor) = self.get(index) else {
            return Err(Tensor0Error::Message(format!(
                "unit removal index {index} is out of range for rank {}",
                self.len(),
            )));
        };
        if !factor.is_unit() {
            return Err(Tensor0Error::Message(format!(
                "factor at index {index} is not a canonical unit space",
            )));
        }

        let mut factors = self.factors.clone();
        factors.remove(index);
        Ok(ProductSpace::new(factors))
    }

    pub fn dims(&self) -> Vec<usize> {
        self.factors.iter().map(GradedSpace::dim).collect()
    }

    pub fn dim(&self) -> usize {
        self.factors
            .iter()
            .map(GradedSpace::dim)
            .try_fold(1usize, |total, dim| {
                total
                    .checked_mul(dim)
                    .ok_or("product space dimension overflowed")
            })
            .expect("product space dimension overflowed")
    }

    pub fn sector_dims(&self, sectors: &[I]) -> Option<Vec<usize>> {
        if sectors.len() != self.factors.len() {
            return None;
        }

        Some(
            self.factors
                .iter()
                .zip(sectors)
                .map(|(factor, sector)| factor.sector_dim(sector))
                .collect(),
        )
    }

    pub fn sector_dim(&self, sectors: &[I]) -> Option<usize> {
        if sectors.len() != self.factors.len() {
            return None;
        }

        Some(
            self.factors
                .iter()
                .zip(sectors)
                .map(|(factor, sector)| factor.sector_dim(sector))
                .try_fold(1usize, |total, dim| {
                    total
                        .checked_mul(dim)
                        .ok_or("product sector dimension overflowed")
                })
                .expect("product sector dimension overflowed"),
        )
    }

    pub fn dual(&self) -> Self {
        let factors = self
            .factors
            .iter()
            .rev()
            .map(GradedSpace::dual)
            .collect::<Vec<_>>();
        ProductSpace::new(factors)
    }

    pub fn fuse(&self) -> Result<GradedSpace<I>> {
        fuse_product_space(self)
    }

    pub(crate) fn sector_support(&self) -> ProductSectorSupport<I> {
        ProductSectorSupport {
            sectors: self
                .factors
                .iter()
                .map(|factor| factor.visible_sectors().collect())
                .collect(),
            is_dual: self.factors.iter().map(GradedSpace::is_dual).collect(),
        }
    }
}

impl<I: Sector> ProductSectorSupport<I> {
    pub(crate) fn sector_tuples(&self) -> Vec<Vec<I>> {
        let mut tuples = vec![Vec::new()];
        for factor in self.sectors.iter().rev() {
            let mut next = Vec::with_capacity(tuples.len().saturating_mul(factor.len()));
            for tuple in tuples {
                for sector in factor {
                    let mut sectors = Vec::with_capacity(tuple.len() + 1);
                    sectors.push(sector.clone());
                    sectors.extend(tuple.iter().cloned());
                    next.push(sectors);
                }
            }
            tuples = next;
        }
        tuples
    }

    pub(crate) fn block_sectors(&self) -> Vec<I> {
        if self.sectors.iter().any(Vec::is_empty) {
            return Vec::new();
        }
        match self.sectors.len() {
            0 => vec![I::unit()],
            1 => {
                let mut blocksectors = self.sectors[0].clone();
                blocksectors.sort();
                blocksectors
            }
            _ => {
                let mut reachable = BTreeSet::from([I::unit()]);
                for factor in &self.sectors {
                    let mut next = BTreeSet::new();
                    for prefix in &reachable {
                        for sector in factor {
                            next.extend(prefix.fusion_outputs(sector));
                        }
                    }
                    reachable = next;
                }
                reachable.into_iter().collect()
            }
        }
    }

    pub(crate) fn fusion_trees(&self, coupled: &I) -> Result<Vec<FusionTree<I>>> {
        let mut trees = Vec::new();
        visit_target_fusion_trees(
            &self.sectors,
            &self.is_dual,
            coupled,
            coupled,
            &mut Vec::with_capacity(self.sectors.len()),
            &mut Vec::with_capacity(self.sectors.len().saturating_sub(2)),
            &mut |tree| trees.push(tree),
        )?;
        Ok(trees)
    }
}

fn visit_target_fusion_trees<I, F>(
    sectors: &[Vec<I>],
    is_dual: &[bool],
    target: &I,
    root_target: &I,
    uncoupled_rev: &mut Vec<I>,
    innerlines_rev: &mut Vec<I>,
    emit: &mut F,
) -> Result<()>
where
    I: Sector,
    F: FnMut(FusionTree<I>),
{
    match sectors.len() {
        0 => {
            if target == &I::unit() {
                emit(FusionTree::new(
                    vec![],
                    root_target.clone(),
                    vec![],
                    vec![],
                    vec![],
                )?);
            }
        }
        1 => {
            for sector in &sectors[0] {
                if sector != target {
                    continue;
                }
                uncoupled_rev.push(sector.clone());
                emit(fusion_tree_from_reverse_path(
                    uncoupled_rev,
                    root_target,
                    is_dual,
                    innerlines_rev,
                )?);
                uncoupled_rev.pop();
            }
        }
        arity => {
            let last_index = arity - 1;
            for sector in &sectors[last_index] {
                uncoupled_rev.push(sector.clone());
                for prefix_target in target.fusion_outputs(&sector.dual()) {
                    let multiplicity = I::n_symbol(&prefix_target, sector, target);
                    if multiplicity == 0 {
                        continue;
                    }
                    if multiplicity > 1 {
                        return Err(Tensor0Error::Message(
                            "layout only supports multiplicity-free fusion".to_string(),
                        ));
                    }
                    if arity > 2 {
                        innerlines_rev.push(prefix_target.clone());
                    }
                    visit_target_fusion_trees(
                        &sectors[..last_index],
                        is_dual,
                        &prefix_target,
                        root_target,
                        uncoupled_rev,
                        innerlines_rev,
                        emit,
                    )?;
                    if arity > 2 {
                        innerlines_rev.pop();
                    }
                }
                uncoupled_rev.pop();
            }
        }
    }
    Ok(())
}

fn fusion_tree_from_reverse_path<I: Sector>(
    uncoupled_rev: &[I],
    coupled: &I,
    is_dual: &[bool],
    innerlines_rev: &[I],
) -> Result<FusionTree<I>> {
    FusionTree::new(
        uncoupled_rev.iter().rev().cloned().collect(),
        coupled.clone(),
        is_dual.to_vec(),
        innerlines_rev.iter().rev().cloned().collect(),
        vec![0; uncoupled_rev.len().saturating_sub(1)],
    )
}

pub fn fuse_product_space<I: Sector>(product: &ProductSpace<I>) -> Result<GradedSpace<I>> {
    if I::fusion_style() == FusionStyle::GenericFusion {
        return Err(Tensor0Error::Message(
            "fuse_product_space does not support GenericFusion sector families".to_string(),
        ));
    }

    let mut factors = product.factors().iter();
    let Some(first) = factors.next() else {
        return GradedSpace::new(vec![(I::unit(), 1)], false);
    };

    let mut fused = fuse_single_space(first)?;
    for factor in factors {
        fused = fuse_two_spaces(&fused, &fuse_single_space(factor)?)?;
    }
    Ok(fused)
}

fn product_spec<I: Sector>(factors: &[GradedSpace<I>]) -> ProductSpaceSpec {
    ProductSpaceSpec {
        sector_spec: I::sector_spec(),
        factors: factors.iter().map(GradedSpace::to_spec).collect(),
    }
}

fn fuse_single_space<I: Sector>(space: &GradedSpace<I>) -> Result<GradedSpace<I>> {
    GradedSpace::new(space.sectors(), false)
}

fn fuse_two_spaces<I: Sector>(
    left: &GradedSpace<I>,
    right: &GradedSpace<I>,
) -> Result<GradedSpace<I>> {
    let mut dims = std::collections::BTreeMap::<I, usize>::new();
    for (left_sector, left_dim) in left.sectors() {
        for (right_sector, right_dim) in right.sectors() {
            for coupled in left_sector.fusion_outputs(&right_sector) {
                let multiplicity = I::n_symbol(&left_sector, &right_sector, &coupled);
                if multiplicity == 0 {
                    continue;
                }
                let degeneracy = checked_mul(left_dim, right_dim, "fused space dimension")?;
                let degeneracy = checked_mul(degeneracy, multiplicity, "fused space dimension")?;

                let entry = dims.entry(coupled).or_insert(0);
                *entry = entry.checked_add(degeneracy).ok_or_else(|| {
                    Tensor0Error::Message("fused space dimension overflowed".to_string())
                })?;
            }
        }
    }
    GradedSpace::new(dims.into_iter().collect(), false)
}

fn checked_mul(left: usize, right: usize, context: &str) -> Result<usize> {
    left.checked_mul(right)
        .ok_or_else(|| Tensor0Error::Message(format!("{context} overflowed")))
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeSet, HashMap};

    use crate::fusion_tree::{enumerate_fusion_trees, FusionTree};
    use crate::sector::{fusion_sectors, SU2Irrep, Sector, U1Irrep, U1SU2Irrep};

    use super::{GradedSpace, ProductSpace};

    fn u1(value: i64) -> U1Irrep {
        U1Irrep::charge2(value).unwrap()
    }

    fn su2(value: i64) -> SU2Irrep {
        SU2Irrep::spin2(value).unwrap()
    }

    fn graded_from_visible<I: Sector>(sectors: Vec<I>, is_dual: bool) -> GradedSpace<I> {
        GradedSpace::new(
            sectors
                .into_iter()
                .map(|sector| {
                    let sector = if is_dual { sector.dual() } else { sector };
                    (sector, 1)
                })
                .collect(),
            is_dual,
        )
        .unwrap()
    }

    fn tuple_reference<I: Sector>(product: &ProductSpace<I>, coupled: &I) -> Vec<FusionTree<I>> {
        let support = product.sector_support();
        let mut trees = Vec::new();
        for tuple in support.sector_tuples() {
            trees.extend(enumerate_fusion_trees(&tuple, &support.is_dual, coupled).unwrap());
        }
        trees
    }

    fn tuple_block_sector_reference<I: Sector>(product: &ProductSpace<I>) -> Vec<I> {
        let support = product.sector_support();
        let mut blocksectors = BTreeSet::new();
        for tuple in support.sector_tuples() {
            blocksectors.extend(fusion_sectors(&tuple));
        }
        blocksectors.into_iter().collect()
    }

    fn assert_reachable_block_sectors_match_tuple_reference<I: Sector + std::fmt::Debug>(
        product: &ProductSpace<I>,
    ) {
        assert_eq!(
            product.sector_support().block_sectors(),
            tuple_block_sector_reference(product),
        );
    }

    fn tree_counts<I: Sector>(trees: Vec<FusionTree<I>>) -> HashMap<FusionTree<I>, usize> {
        let mut counts = HashMap::new();
        for tree in trees {
            *counts.entry(tree).or_insert(0) += 1;
        }
        counts
    }

    fn assert_target_generation_matches_tuple_reference_as_multiset<I: Sector + std::fmt::Debug>(
        product: &ProductSpace<I>,
        extra_targets: &[I],
    ) {
        let support = product.sector_support();
        let mut targets = support.block_sectors();
        targets.extend_from_slice(extra_targets);
        targets.sort();
        targets.dedup();
        for target in targets {
            assert_eq!(
                tree_counts(support.fusion_trees(&target).unwrap()),
                tree_counts(tuple_reference(product, &target)),
            );
        }
    }

    #[test]
    fn target_generation_preserves_tree_multiset() {
        let u1_product = ProductSpace::new(vec![
            graded_from_visible(vec![u1(0), u1(1), u1(-1)], false),
            graded_from_visible(vec![u1(0), u1(2), u1(-2)], true),
            graded_from_visible(vec![u1(0), u1(1)], false),
            graded_from_visible(vec![u1(0), u1(-1)], true),
        ]);
        assert_target_generation_matches_tuple_reference_as_multiset(&u1_product, &[u1(20)]);

        let su2_product = ProductSpace::new(
            (0..4)
                .map(|index| graded_from_visible(vec![su2(0), su2(1), su2(2)], index % 2 == 1))
                .collect(),
        );
        assert_target_generation_matches_tuple_reference_as_multiset(&su2_product, &[su2(20)]);

        let product_values = vec![
            U1SU2Irrep::new((u1(0), su2(0))),
            U1SU2Irrep::new((u1(1), su2(1))),
        ];
        let product_sector_product = ProductSpace::new(vec![
            graded_from_visible(product_values.clone(), false),
            graded_from_visible(product_values.clone(), true),
            graded_from_visible(product_values, false),
        ]);
        assert_target_generation_matches_tuple_reference_as_multiset(
            &product_sector_product,
            &[U1SU2Irrep::new((u1(20), su2(20)))],
        );
    }

    #[test]
    fn reachable_block_sectors_preserve_tuple_reference() {
        assert_reachable_block_sectors_match_tuple_reference(&ProductSpace::<U1Irrep>::one());
        assert_reachable_block_sectors_match_tuple_reference(&ProductSpace::new(vec![
            graded_from_visible(vec![u1(0), u1(1), u1(-1)], true),
        ]));
        assert_reachable_block_sectors_match_tuple_reference(&ProductSpace::new(vec![
            graded_from_visible(vec![u1(0), u1(1), u1(-1)], false),
            graded_from_visible(vec![u1(0), u1(2), u1(-2)], true),
            graded_from_visible(vec![u1(0), u1(1)], false),
            graded_from_visible(vec![u1(0), u1(-1)], true),
        ]));
        assert_reachable_block_sectors_match_tuple_reference(&ProductSpace::new(
            (0..4)
                .map(|index| graded_from_visible(vec![su2(0), su2(1), su2(2)], index % 2 == 1))
                .collect(),
        ));

        let product_values = vec![
            U1SU2Irrep::new((u1(0), su2(0))),
            U1SU2Irrep::new((u1(1), su2(1))),
        ];
        assert_reachable_block_sectors_match_tuple_reference(&ProductSpace::new(vec![
            graded_from_visible(product_values.clone(), false),
            graded_from_visible(product_values.clone(), true),
            graded_from_visible(product_values, false),
        ]));
        assert_reachable_block_sectors_match_tuple_reference(&ProductSpace::new(vec![
            GradedSpace::zero(false),
            graded_from_visible(vec![u1(0), u1(1)], false),
        ]));
    }

    #[test]
    fn target_generation_uses_tensorkit_order() {
        let product = ProductSpace::new(vec![
            graded_from_visible(vec![su2(0), su2(3)], false),
            graded_from_visible(vec![su2(0), su2(1)], false),
            graded_from_visible(vec![su2(1)], false),
        ]);
        let coupled = su2(2);
        let order = |trees: Vec<FusionTree<SU2Irrep>>| {
            trees
                .into_iter()
                .map(|tree| {
                    (
                        tree.uncoupled()
                            .iter()
                            .map(SU2Irrep::spin2_value)
                            .collect::<Vec<_>>(),
                        tree.innerlines()
                            .iter()
                            .map(SU2Irrep::spin2_value)
                            .collect::<Vec<_>>(),
                    )
                })
                .collect::<Vec<_>>()
        };

        // TensorKit visits the last sector, then its target inner line, then the prefix tree.
        assert_eq!(
            order(product.sector_support().fusion_trees(&coupled).unwrap()),
            vec![(vec![0, 1, 1], vec![1]), (vec![3, 0, 1], vec![3])],
        );
        assert_eq!(
            order(tuple_reference(&product, &coupled)),
            vec![(vec![3, 0, 1], vec![3]), (vec![0, 1, 1], vec![1])],
        );
    }

    #[test]
    fn target_generation_preserves_rank_and_empty_factor_edges() {
        let rank_zero = ProductSpace::<U1Irrep>::one();
        assert_target_generation_matches_tuple_reference_as_multiset(&rank_zero, &[u1(1)]);

        let rank_one =
            ProductSpace::new(vec![graded_from_visible(vec![u1(0), u1(1), u1(-1)], true)]);
        assert_target_generation_matches_tuple_reference_as_multiset(&rank_one, &[u1(20)]);

        let empty_factor = ProductSpace::new(vec![
            GradedSpace::zero(false),
            graded_from_visible(vec![u1(0), u1(1)], false),
        ]);
        assert_target_generation_matches_tuple_reference_as_multiset(
            &empty_factor,
            &[u1(0), u1(1)],
        );
    }
}
