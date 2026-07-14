use tensor0_core::error::{Result, Tensor0Error};
use tensor0_core::fusion_tree::FusionTreePair;
use tensor0_core::layout::{
    build_degeneracy_structure, build_degeneracy_structure_from_sector_structure,
    build_sector_structure, DegeneracyStructure, SectorStructure, SubblockStructure,
};
use tensor0_core::sector::{
    BraidingStyle, EncodedSectorValue, FusionStyle, SU2Irrep, Sector, SectorCardinality,
    SectorSpec, SortKey, U1Irrep, U1SU2Irrep,
};
use tensor0_core::space::{GradedSpace, HomSpace, ProductSpace};

fn u1(charge2: i64) -> U1Irrep {
    U1Irrep::charge2(charge2).unwrap()
}

fn su2(spin2: i64) -> SU2Irrep {
    SU2Irrep::spin2(spin2).unwrap()
}

fn u1_su2(charge2: i64, spin2: i64) -> U1SU2Irrep {
    U1SU2Irrep::new((u1(charge2), su2(spin2)))
}

fn empty_product_of<I: Sector>() -> ProductSpace<I> {
    ProductSpace::new(vec![])
}

fn empty_product() -> ProductSpace<U1Irrep> {
    ProductSpace::new(vec![])
}

fn build_layout_parts<I: Sector>(space: &HomSpace<I>) -> (SectorStructure<I>, DegeneracyStructure) {
    let sectorstructure = build_sector_structure(space).unwrap();
    let degeneracystructure = build_degeneracy_structure(space).unwrap();
    (sectorstructure, degeneracystructure)
}

fn assert_space_layout_contract<I: Sector>(space: &HomSpace<I>) {
    let (sectorstructure, degeneracystructure) = build_layout_parts(space);
    assert_layout_contract(&sectorstructure, &degeneracystructure);
}

fn assert_layout_contract<I: Sector>(
    sectorstructure: &SectorStructure<I>,
    degeneracystructure: &DegeneracyStructure,
) {
    assert_eq!(
        degeneracystructure.blockstructure.len(),
        sectorstructure.blocksectors().len(),
    );
    assert_eq!(
        degeneracystructure.subblockstructure.len(),
        sectorstructure.fusiontree_pairs().len(),
    );

    let mut start = 0usize;
    for block in &degeneracystructure.blockstructure {
        assert_eq!(block.start, start);
        start += block.row_dim * block.col_dim;
        assert_eq!(block.stop, start);
    }
    assert_eq!(degeneracystructure.total_dim, start);

    for pair in sectorstructure.fusiontree_pairs() {
        assert!(pair.row.coupled() == pair.col.coupled());
        assert!(sectorstructure
            .blocksectors()
            .any(|sector| sector == pair.row.coupled()));
    }
}

fn blocksectors_vec<I: Sector>(sectorstructure: &SectorStructure<I>) -> Vec<I> {
    sectorstructure.blocksectors().cloned().collect()
}

fn assert_blocks_are_packed_with_dims(
    degeneracystructure: &DegeneracyStructure,
    expected_dims: &[(usize, usize)],
) {
    assert_eq!(
        degeneracystructure.blockstructure.len(),
        expected_dims.len()
    );

    let mut start = 0usize;
    for (block, &(row_dim, col_dim)) in degeneracystructure.blockstructure.iter().zip(expected_dims)
    {
        let stop = start + row_dim * col_dim;
        assert_eq!(
            (block.row_dim, block.col_dim, block.start, block.stop),
            (row_dim, col_dim, start, stop),
        );
        start = stop;
    }
    assert_eq!(degeneracystructure.total_dim, start);
}

fn assert_subblock(
    subblock: &SubblockStructure,
    sizes: &[usize],
    strides: &[usize],
    offset: usize,
) {
    assert_eq!(subblock.sizes.as_slice(), sizes);
    assert_eq!(subblock.strides.as_slice(), strides);
    assert_eq!(subblock.offset, offset);
}

#[test]
fn layout_structures_satisfy_core_contracts() {
    let scalar = HomSpace::new(empty_product(), empty_product());
    assert_space_layout_contract(&scalar);

    let single_factor = {
        let v = GradedSpace::new(vec![(u1(-1), 5), (u1(0), 2), (u1(1), 3)], false).unwrap();
        HomSpace::new(one_factor(v.clone()), one_factor(v))
    };
    assert_space_layout_contract(&single_factor);

    let two_factor = HomSpace::new(two_factor_u1_product(), two_factor_u1_product());
    assert_space_layout_contract(&two_factor);

    let su2_multitree = HomSpace::new(su2_four_half_product(), empty_product_of::<SU2Irrep>());
    assert_space_layout_contract(&su2_multitree);

    let product_alias = {
        let product = || {
            ProductSpace::new(vec![
                GradedSpace::new(vec![(u1_su2(1, 1), 2)], false).unwrap(),
                GradedSpace::new(vec![(u1_su2(-1, 1), 3)], false).unwrap(),
            ])
        };
        HomSpace::new(product(), product())
    };
    assert_space_layout_contract(&product_alias);
}

#[test]
fn sectorstructure_fusion_tree_pairs_align_with_degeneracy_subblocks_like_tensorkit() {
    let v = GradedSpace::new(vec![(u1(0), 2)], false).unwrap();
    let w = GradedSpace::new(vec![(u1(0), 3)], false).unwrap();
    let h = HomSpace::new(ProductSpace::new(vec![v, w]), empty_product());
    let (sectorstructure, degeneracystructure) = build_layout_parts(&h);

    for (index, pair) in sectorstructure.fusiontree_pairs().enumerate() {
        assert_eq!(sectorstructure.fusiontree_pair_index(pair), Some(index));
        assert_eq!(sectorstructure.fusiontree_pair_at(index), Some(pair),);
        let subblock = &degeneracystructure.subblockstructure[index];
        assert_subblock(subblock, &[2, 3], &[3, 1], 0);
    }
}

#[test]
fn sectorstructure_indexes_visible_sectors_and_fusion_tree_pairs() {
    let v = GradedSpace::new(vec![(u1(0), 2), (u1(1), 3)], false).unwrap();
    let h = HomSpace::new(one_factor(v.clone()), one_factor(v));
    let sectorstructure = build_sector_structure(&h).unwrap();

    assert_eq!(sectorstructure.blocksector_index(&u1(0)), Some(0));
    assert_eq!(sectorstructure.blocksector_index(&u1(1)), Some(1));
    assert_eq!(sectorstructure.blocksector_index(&u1(2)), None);

    for (index, pair) in sectorstructure.fusiontree_pairs().enumerate() {
        assert_eq!(sectorstructure.fusiontree_pair_index(pair), Some(index));
    }
}

fn one_factor(space: GradedSpace<U1Irrep>) -> ProductSpace<U1Irrep> {
    ProductSpace::new(vec![space])
}

fn two_factor_u1_product() -> ProductSpace<U1Irrep> {
    ProductSpace::new(vec![
        GradedSpace::new(vec![(u1(-1), 5), (u1(0), 2), (u1(1), 3)], false).unwrap(),
        GradedSpace::new(vec![(u1(-1), 13), (u1(0), 7), (u1(1), 11)], false).unwrap(),
    ])
}

fn two_factor_u1_product_with_dims(
    left_negative: usize,
    left_zero: usize,
    left_positive: usize,
    right_negative: usize,
    right_zero: usize,
    right_positive: usize,
) -> ProductSpace<U1Irrep> {
    ProductSpace::new(vec![
        GradedSpace::new(
            vec![
                (u1(-1), left_negative),
                (u1(0), left_zero),
                (u1(1), left_positive),
            ],
            false,
        )
        .unwrap(),
        GradedSpace::new(
            vec![
                (u1(-1), right_negative),
                (u1(0), right_zero),
                (u1(1), right_positive),
            ],
            false,
        )
        .unwrap(),
    ])
}

fn three_factor_u1_product() -> ProductSpace<U1Irrep> {
    ProductSpace::new(vec![
        GradedSpace::new(vec![(u1(1), 2)], false).unwrap(),
        GradedSpace::new(vec![(u1(-3), 5)], false).unwrap(),
        GradedSpace::new(vec![(u1(2), 7)], false).unwrap(),
    ])
}

fn su2_four_half_product() -> ProductSpace<SU2Irrep> {
    ProductSpace::new(vec![
        GradedSpace::new(vec![(su2(1), 1)], false).unwrap(),
        GradedSpace::new(vec![(su2(1), 1)], false).unwrap(),
        GradedSpace::new(vec![(su2(1), 1)], false).unwrap(),
        GradedSpace::new(vec![(su2(1), 1)], false).unwrap(),
    ])
}

fn assert_first_u1_pair_uncoupled_order<'a>(
    pairs: impl IntoIterator<Item = &'a FusionTreePair<U1Irrep>>,
) {
    let expected_uncoupled = [
        ([0, 0], [0, 0]),
        ([0, 0], [-1, 1]),
        ([0, 0], [1, -1]),
        ([-1, 1], [0, 0]),
        ([-1, 1], [-1, 1]),
        ([-1, 1], [1, -1]),
        ([1, -1], [0, 0]),
        ([1, -1], [-1, 1]),
        ([1, -1], [1, -1]),
    ];
    for (pair, (expected_row, expected_col)) in pairs
        .into_iter()
        .take(expected_uncoupled.len())
        .zip(expected_uncoupled)
    {
        let row = &pair.row;
        let col = &pair.col;
        assert_eq!(
            row.uncoupled(),
            expected_row.into_iter().map(u1).collect::<Vec<_>>(),
        );
        assert_eq!(
            col.uncoupled(),
            expected_col.into_iter().map(u1).collect::<Vec<_>>(),
        );
    }
}

#[test]
fn sectorstructure_reuse_tracks_visible_layout_contract() {
    let reference = HomSpace::new(
        one_factor(GradedSpace::new(vec![(u1(0), 2), (u1(1), 3)], false).unwrap()),
        one_factor(GradedSpace::new(vec![(u1(0), 5), (u1(1), 7)], false).unwrap()),
    );
    let same_visible_sectors = HomSpace::new(
        one_factor(GradedSpace::new(vec![(u1(0), 11), (u1(1), 13)], false).unwrap()),
        one_factor(GradedSpace::new(vec![(u1(0), 17), (u1(1), 19)], false).unwrap()),
    );
    let reused = assert_sectorstructure_reuse(&reference, &same_visible_sectors);
    assert_blocks_are_packed_with_dims(&reused, &[(11, 17), (13, 19)]);

    let visible_left = HomSpace::new(
        one_factor(GradedSpace::new(vec![(u1(0), 2)], false).unwrap()),
        one_factor(GradedSpace::new(vec![(u1(0), 2)], false).unwrap()),
    );
    let visible_right = HomSpace::new(
        one_factor(GradedSpace::new(vec![(u1(0), 2), (u1(1), 3)], false).unwrap()),
        one_factor(GradedSpace::new(vec![(u1(0), 2), (u1(1), 3)], false).unwrap()),
    );
    assert_sectorstructure_reuse_rejected(&visible_left, &visible_right);

    let plain = GradedSpace::new(vec![(u1(0), 2), (u1(1), 3)], false).unwrap();
    let dual = GradedSpace::new(vec![(u1(0), 2), (u1(1), 3)], false)
        .unwrap()
        .dual();
    let dual_left = HomSpace::new(one_factor(plain.clone()), one_factor(plain));
    let dual_right = HomSpace::new(one_factor(dual.clone()), one_factor(dual));
    assert_sectorstructure_reuse_rejected(&dual_left, &dual_right);

    let codomain = one_factor(GradedSpace::new(vec![(u1(0), 2)], false).unwrap());
    let domain = one_factor(GradedSpace::new(vec![(u1(1), 3)], false).unwrap());
    let boundary_left = HomSpace::new(codomain.clone(), domain.clone());
    let boundary_right = HomSpace::new(domain, codomain);
    assert_sectorstructure_reuse_rejected(&boundary_left, &boundary_right);

    let first = GradedSpace::new(vec![(u1(0), 2)], false).unwrap();
    let second = GradedSpace::new(vec![(u1(1), 3)], false).unwrap();
    let order_left = HomSpace::new(
        ProductSpace::new(vec![first.clone(), second.clone()]),
        empty_product(),
    );
    let order_right = HomSpace::new(ProductSpace::new(vec![second, first]), empty_product());
    assert_sectorstructure_reuse_rejected(&order_left, &order_right);
}

fn assert_sectorstructure_reuse<I: Sector>(
    cached: &HomSpace<I>,
    target: &HomSpace<I>,
) -> DegeneracyStructure {
    let sectorstructure = build_sector_structure(cached).unwrap();
    build_degeneracy_structure_from_sector_structure(target, &sectorstructure).unwrap()
}

fn assert_sectorstructure_reuse_rejected<I: Sector>(cached: &HomSpace<I>, target: &HomSpace<I>) {
    let sectorstructure = build_sector_structure(cached).unwrap();
    let err =
        build_degeneracy_structure_from_sector_structure(target, &sectorstructure).unwrap_err();

    assert_eq!(
        err.to_string(),
        "sectorstructure does not match HomSpace sector structure",
    );
}

#[test]
fn single_factor_layout_preserves_visible_order_dims_and_dual_flags() {
    let v = GradedSpace::new(vec![(u1(-1), 5), (u1(0), 2), (u1(1), 3)], false).unwrap();
    let space = HomSpace::new(one_factor(v.clone()), one_factor(v));
    let (sectorstructure, degeneracystructure) = build_layout_parts(&space);

    assert_eq!(
        blocksectors_vec(&sectorstructure),
        vec![u1(0), u1(1), u1(-1)]
    );
    assert_eq!(sectorstructure.fusiontree_pairs().len(), 3);
    assert_blocks_are_packed_with_dims(&degeneracystructure, &[(2, 2), (3, 3), (5, 5)]);

    let expected = [(u1(0), 2, 0), (u1(1), 3, 4), (u1(-1), 5, 13)];
    for ((subblock, pair), (sector, dim, start)) in degeneracystructure
        .subblockstructure
        .iter()
        .zip(sectorstructure.fusiontree_pairs())
        .zip(expected)
    {
        let row = &pair.row;
        let col = &pair.col;
        assert_subblock(subblock, &[dim, dim], &[dim, 1], start);
        assert_eq!(row.uncoupled(), &[sector]);
        assert_eq!(row.coupled(), &sector);
        assert_eq!(row.is_dual(), &[false]);
        assert_eq!(col, row);
    }

    let v = GradedSpace::new(vec![(u1(1), 3), (u1(-1), 5)], false)
        .unwrap()
        .dual();
    let space = HomSpace::new(one_factor(v.clone()), one_factor(v));
    let (sectorstructure, degeneracystructure) = build_layout_parts(&space);

    assert_eq!(blocksectors_vec(&sectorstructure), vec![u1(1), u1(-1)]);
    assert_blocks_are_packed_with_dims(&degeneracystructure, &[(5, 5), (3, 3)]);

    for (pair, sector) in sectorstructure.fusiontree_pairs().zip([u1(1), u1(-1)]) {
        let row = &pair.row;
        let col = &pair.col;
        assert_eq!(row.uncoupled(), &[sector]);
        assert_eq!(row.is_dual(), &[true]);
        assert_eq!(col, row);
    }
}

#[test]
fn two_factor_u1_layout_uses_tensorkit_tree_order_and_row_major_subblocks() {
    let codomain = two_factor_u1_product();
    let domain = two_factor_u1_product_with_dims(17, 19, 23, 29, 31, 37);
    let space = HomSpace::new(codomain, domain);
    let (sectorstructure, degeneracystructure) = build_layout_parts(&space);

    assert_eq!(
        blocksectors_vec(&sectorstructure),
        vec![u1(0), u1(1), u1(-1), u1(2), u1(-2)],
    );
    assert_blocks_are_packed_with_dims(
        &degeneracystructure,
        &[(108, 1885), (43, 1416), (61, 1078), (33, 851), (65, 493)],
    );

    assert_first_u1_pair_uncoupled_order(sectorstructure.fusiontree_pairs());

    let expected_offsets = [0, 589, 1218, 26390, 26979, 27608, 130065, 130654, 131283];
    for (subblock, expected_offset) in degeneracystructure.subblockstructure[..9]
        .iter()
        .zip(expected_offsets)
    {
        assert_eq!(subblock.offset, expected_offset);
    }
    assert_subblock(
        &degeneracystructure.subblockstructure[0],
        &[2, 7, 19, 31],
        &[13195, 1885, 31, 1],
        0,
    );
    assert_subblock(
        &degeneracystructure.subblockstructure[4],
        &[5, 11, 17, 37],
        &[20735, 1885, 37, 1],
        26979,
    );
}

#[test]
fn three_factor_one_sided_layouts_use_python_row_major_strides() {
    let cases = [
        (
            HomSpace::new(three_factor_u1_product(), empty_product()),
            (70, 1),
        ),
        (
            HomSpace::new(empty_product(), three_factor_u1_product()),
            (1, 70),
        ),
    ];

    for (space, block_dims) in cases {
        let (sectorstructure, degeneracystructure) = build_layout_parts(&space);

        assert_eq!(blocksectors_vec(&sectorstructure), vec![u1(0)]);
        assert_eq!(sectorstructure.fusiontree_pairs().len(), 1);
        assert_blocks_are_packed_with_dims(&degeneracystructure, &[block_dims]);
        assert_subblock(
            &degeneracystructure.subblockstructure[0],
            &[2, 5, 7],
            &[35, 7, 1],
            0,
        );
    }
}

#[test]
fn su2_layout_handles_non_abelian_blocks_and_multi_tree_basis() {
    let product = || {
        ProductSpace::new(vec![
            GradedSpace::new(vec![(su2(1), 2)], false).unwrap(),
            GradedSpace::new(vec![(su2(1), 3)], false).unwrap(),
        ])
    };
    let space = HomSpace::new(product(), product());
    let (sectorstructure, degeneracystructure) = build_layout_parts(&space);

    assert_eq!(blocksectors_vec(&sectorstructure), vec![su2(0), su2(2)]);
    assert_eq!(sectorstructure.fusiontree_pairs().len(), 2);
    assert_blocks_are_packed_with_dims(&degeneracystructure, &[(6, 6), (6, 6)]);

    let codomain = ProductSpace::new(vec![
        GradedSpace::new(vec![(su2(0), 2), (su2(1), 3)], false).unwrap(),
        GradedSpace::new(vec![(su2(0), 5), (su2(1), 7)], false).unwrap(),
    ]);
    let space = HomSpace::new(codomain, empty_product_of::<SU2Irrep>());
    let (sectorstructure, degeneracystructure) = build_layout_parts(&space);

    assert_eq!(blocksectors_vec(&sectorstructure), vec![su2(0)]);
    assert_eq!(sectorstructure.fusiontree_pairs().len(), 2);
    assert_blocks_are_packed_with_dims(&degeneracystructure, &[(31, 1)]);

    let expected = [
        (vec![su2(0), su2(0)], vec![2, 5], 0),
        (vec![su2(1), su2(1)], vec![3, 7], 10),
    ];
    for ((pair, subblock), (uncoupled, sizes, offset)) in sectorstructure
        .fusiontree_pairs()
        .zip(degeneracystructure.subblockstructure.iter())
        .zip(expected)
    {
        let row = &pair.row;
        let col = &pair.col;
        assert_eq!(row.uncoupled(), uncoupled);
        assert_eq!(row.coupled(), &su2(0));
        assert_eq!(subblock.sizes, sizes);
        assert_eq!(subblock.offset, offset);
        assert!(col.uncoupled().is_empty());
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct GenericLayoutSector(i64);

impl Sector for GenericLayoutSector {
    fn sector_spec() -> SectorSpec {
        SectorSpec::u1()
    }

    fn encoded_width() -> usize {
        1
    }

    fn decode_value(value: &[i64]) -> Result<Self> {
        if value.len() == 1 {
            Ok(GenericLayoutSector(value[0]))
        } else {
            Err(Tensor0Error::BadSectorWidth {
                expected: 1,
                actual: value.len(),
            })
        }
    }

    fn encode_value(&self) -> EncodedSectorValue {
        let mut encoded = EncodedSectorValue::new();
        encoded.push(self.0);
        encoded
    }

    fn unit() -> Self {
        GenericLayoutSector(0)
    }

    fn dual(&self) -> Self {
        *self
    }

    fn quantum_dim(&self) -> usize {
        1
    }

    fn fusion_style() -> FusionStyle {
        FusionStyle::GenericFusion
    }

    fn braiding_style() -> BraidingStyle {
        BraidingStyle::Anyonic
    }

    fn cardinality() -> Result<SectorCardinality> {
        Ok(SectorCardinality::Infinite)
    }

    fn fusion_outputs(&self, rhs: &Self) -> Vec<Self> {
        vec![GenericLayoutSector(self.0 + rhs.0)]
    }

    fn n_symbol(a: &Self, b: &Self, c: &Self) -> usize {
        if c.0 == a.0 + b.0 {
            2
        } else {
            0
        }
    }

    fn r_symbol(a: &Self, b: &Self, c: &Self) -> f64 {
        if Self::n_symbol(a, b, c) == 0 {
            0.0
        } else {
            1.0
        }
    }

    fn sort_key(&self) -> SortKey {
        let mut key = SortKey::new();
        key.push(self.0 as u128);
        key
    }

    fn sort_index(&self) -> Result<u128> {
        Ok(self.0 as u128)
    }

    fn value_at(index: u128) -> Result<Self> {
        let value = i64::try_from(index).map_err(|_| Tensor0Error::SectorIndexOverflow)?;
        Ok(GenericLayoutSector(value))
    }
}

#[test]
fn generic_fusion_layout_is_rejected_at_entry() {
    let space = HomSpace::new(
        empty_product_of::<GenericLayoutSector>(),
        empty_product_of::<GenericLayoutSector>(),
    );
    let err = build_sector_structure(&space).unwrap_err();

    assert_eq!(
        err.to_string(),
        "layout does not support GenericFusion sector families",
    );
}
