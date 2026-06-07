use tensor0_core::sector::{
    FermionNumber, FermionParity, FermionParitySU2Irrep, FermionParityU1SU2Irrep, SU2Irrep, Sector,
    U1Irrep, U1SU2Irrep,
};
use tensor0_core::space::{GradedSpace, HomSpace, ProductSpace};
use tensor0_core::transform::{
    tree_permuter, tree_transposer, AbelianTransformData, GenericTransformData, TreeTransformer,
};

fn u1(charge2: i64) -> U1Irrep {
    U1Irrep::charge2(charge2).unwrap()
}

fn su2(spin2: i64) -> SU2Irrep {
    SU2Irrep::spin2(spin2).unwrap()
}

fn parity(value: i64) -> FermionParity {
    FermionParity::new(value).unwrap()
}

fn fermion_number(charge2: i64, parity_value: i64) -> FermionNumber {
    FermionNumber::decode_value(&[charge2, parity_value]).unwrap()
}

fn u1_su2(charge2: i64, spin2: i64) -> U1SU2Irrep {
    U1SU2Irrep::new((u1(charge2), su2(spin2)))
}

fn parity_su2(parity_value: i64, spin2: i64) -> FermionParitySU2Irrep {
    FermionParitySU2Irrep::new((parity(parity_value), su2(spin2)))
}

fn parity_u1_su2(parity_value: i64, charge2: i64, spin2: i64) -> FermionParityU1SU2Irrep {
    FermionParityU1SU2Irrep::new((parity(parity_value), u1(charge2), su2(spin2)))
}

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1.0e-12,
        "actual={actual}, expected={expected}",
    );
}

fn one_factor(space: GradedSpace<U1Irrep>) -> ProductSpace<U1Irrep> {
    ProductSpace::new(vec![space])
}

fn empty_product() -> ProductSpace<U1Irrep> {
    ProductSpace::new(vec![])
}

fn expect_abelian(transformer: &TreeTransformer) -> &[AbelianTransformData] {
    let TreeTransformer::Abelian(data) = transformer else {
        panic!("expected Abelian transformer");
    };
    data
}

fn expect_generic(transformer: &TreeTransformer) -> &[GenericTransformData] {
    let TreeTransformer::Generic(data) = transformer else {
        panic!("expected Generic transformer");
    };
    data
}

fn assert_abelian_subblocks(
    data: &[AbelianTransformData],
    coeff: f64,
    src_sizes: &[usize],
    src_strides: &[usize],
    dst_sizes: &[usize],
    dst_strides: &[usize],
) {
    assert!(!data.is_empty());
    assert!(data.iter().all(|m| m.coeff == coeff));
    assert!(data.iter().all(|m| m.src.sizes == src_sizes));
    assert!(data.iter().all(|m| m.src.strides == src_strides));
    assert!(data.iter().all(|m| m.src.offset == 0));
    assert!(data.iter().all(|m| m.dst.sizes == dst_sizes));
    assert!(data.iter().all(|m| m.dst.strides == dst_strides));
    assert!(data.iter().all(|m| m.dst.offset == 0));
}

fn assert_single_generic_scalar_block(data: &[GenericTransformData], expected_coeff: f64) {
    assert_eq!(data.len(), 1);
    let block = &data[0];
    assert_eq!(block.basis_transform.shape(), &[1, 1]);
    assert_close(block.basis_transform[[0, 0]], expected_coeff);
    assert_eq!(block.src.sizes, vec![1, 1]);
    assert_eq!(block.dst.sizes, vec![1, 1]);
    assert_eq!(block.src.strides_offsets, vec![(vec![1, 1], 0)]);
    assert_eq!(block.dst.strides_offsets, vec![(vec![1, 1], 0)]);
}

fn assert_abelian_permute_coeff<I: Sector>(
    case: &str,
    src: HomSpace<I>,
    p_codomain: &[usize],
    p_domain: &[usize],
    expected_coeff: f64,
) {
    let dst = src.permute(p_codomain, p_domain).unwrap();
    let transformer = tree_permuter(&src, &dst, p_codomain, p_domain).unwrap();
    let data = expect_abelian(&transformer);

    assert!(!data.is_empty(), "{case}");
    assert!(
        data.iter().all(|entry| entry.coeff == expected_coeff),
        "{case}",
    );
}

fn assert_generic_permute_scalar_coeff<I: Sector>(
    src: HomSpace<I>,
    p_codomain: &[usize],
    p_domain: &[usize],
    expected_coeff: f64,
) {
    let dst = src.permute(p_codomain, p_domain).unwrap();
    let transformer = tree_permuter(&src, &dst, p_codomain, p_domain).unwrap();
    assert_single_generic_scalar_block(expect_generic(&transformer), expected_coeff);
}

#[test]
fn tree_transformer_rejects_incompatible_destination_for_permutation() {
    let v = GradedSpace::new(vec![(u1(0), 2)], false).unwrap();
    let w = GradedSpace::new(vec![(u1(1), 2)], false).unwrap();
    let src = HomSpace::new(one_factor(v), empty_product());
    let dst = HomSpace::new(one_factor(w), empty_product());

    let err = tree_permuter(&src, &dst, &[0], &[]).unwrap_err();

    assert!(err
        .to_string()
        .contains("incompatible spaces for permuting"));
}

#[test]
fn u1_unique_fusion_transformers_preserve_subblock_contracts() {
    {
        let v = GradedSpace::new(vec![(u1(0), 2), (u1(1), 3)], false).unwrap();
        let src = HomSpace::new(one_factor(v.clone()), one_factor(v));
        let dst = src.permute(&[0], &[1]).unwrap();
        let transformer = tree_permuter(&src, &dst, &[0], &[1]).unwrap();

        assert_eq!(dst, src);
        let data = expect_abelian(&transformer);
        assert!(!data.is_empty());
        assert!(data.iter().all(|m| m.coeff == 1.0));
        assert!(data.iter().all(|m| m.src == m.dst));
    }

    {
        let v = GradedSpace::new(vec![(u1(0), 2)], false).unwrap();
        let w = GradedSpace::new(vec![(u1(0), 3)], false).unwrap();
        let src = HomSpace::new(
            ProductSpace::new(vec![v.clone(), w.clone()]),
            empty_product(),
        );
        let dst = src.permute(&[1], &[0]).unwrap();
        let transformer = tree_permuter(&src, &dst, &[1], &[0]).unwrap();

        assert_eq!(dst.codomain().factors(), &[w]);
        assert_eq!(dst.domain().factors(), &[v.dual()]);
        assert_abelian_subblocks(
            expect_abelian(&transformer),
            1.0,
            &[2, 3],
            &[3, 1],
            &[3, 2],
            &[2, 1],
        );

        let transposer = tree_transposer(&src, &dst, &[1], &[0]).unwrap();
        assert_abelian_subblocks(
            expect_abelian(&transposer),
            1.0,
            &[2, 3],
            &[3, 1],
            &[3, 2],
            &[2, 1],
        );
    }

    {
        let v = GradedSpace::new(vec![(u1(1), 2)], false).unwrap();
        let w = GradedSpace::new(vec![(u1(2), 3)], false).unwrap();
        let x = GradedSpace::new(vec![(u1(3), 5)], false).unwrap();
        let src = HomSpace::new(
            ProductSpace::new(vec![v.clone(), w.clone()]),
            one_factor(x.clone()),
        );
        let dst = src.permute(&[2, 1], &[0]).unwrap();
        let transformer = tree_permuter(&src, &dst, &[2, 1], &[0]).unwrap();

        assert_eq!(dst.codomain().factors(), &[x.dual(), w.clone()]);
        assert_eq!(dst.domain().factors(), &[v.dual()]);
        assert_abelian_subblocks(
            expect_abelian(&transformer),
            1.0,
            &[2, 3, 5],
            &[15, 5, 1],
            &[5, 3, 2],
            &[6, 2, 1],
        );
    }
}

#[test]
fn fermion_parity_abelian_coefficients_follow_braiding_contract() {
    {
        let odd_left = GradedSpace::new(vec![(parity(1), 2)], false).unwrap();
        let odd_right = GradedSpace::new(vec![(parity(1), 3)], false).unwrap();
        let src = HomSpace::new(
            ProductSpace::new(vec![odd_left, odd_right]),
            ProductSpace::new(vec![]),
        );
        assert_abelian_permute_coeff("odd odd swap", src, &[1, 0], &[], -1.0);
    }

    {
        let even = GradedSpace::new(vec![(parity(0), 1)], false).unwrap();
        let odd_out = GradedSpace::new(vec![(parity(1), 1)], false).unwrap();
        let odd_in = GradedSpace::new(vec![(parity(1), 1)], false).unwrap();
        let src = HomSpace::new(
            ProductSpace::new(vec![even, odd_out]),
            ProductSpace::new(vec![odd_in]),
        );
        assert_abelian_permute_coeff("even odd swap", src, &[1, 0], &[2], 1.0);
    }

    {
        let left = GradedSpace::new(vec![(fermion_number(1, 1), 1)], false).unwrap();
        let right = GradedSpace::new(vec![(fermion_number(-1, 1), 1)], false).unwrap();
        let src = HomSpace::new(
            ProductSpace::new(vec![left, right]),
            ProductSpace::new(vec![]),
        );
        assert_abelian_permute_coeff("product fermion component", src, &[1, 0], &[], -1.0);
    }

    {
        let odd_out = GradedSpace::new(vec![(parity(1), 2)], false).unwrap();
        let odd_in = GradedSpace::new(vec![(parity(1), 3)], false).unwrap();
        let src = HomSpace::new(
            ProductSpace::new(vec![odd_out]),
            ProductSpace::new(vec![odd_in]),
        );
        assert_abelian_permute_coeff("cross boundary odd swap", src, &[1], &[0], -1.0);
    }

    {
        let first = GradedSpace::new(vec![(parity(1), 2)], false).unwrap();
        let second = GradedSpace::new(vec![(parity(1), 3)], false).unwrap();
        let third = GradedSpace::new(vec![(parity(1), 5)], false).unwrap();
        let fourth = GradedSpace::new(vec![(parity(1), 7)], false).unwrap();
        let src = HomSpace::new(
            ProductSpace::new(vec![first, second, third, fourth]),
            ProductSpace::new(vec![]),
        );
        assert_abelian_permute_coeff("two odd inversions cancel", src, &[1, 2, 0, 3], &[], 1.0);
    }
}

#[test]
fn su2_simplefusion_identity_keeps_unit_coefficients() {
    let half = GradedSpace::new(vec![(su2(1), 1)], false).unwrap();
    let src = HomSpace::new(
        ProductSpace::new(vec![half.clone(), half]),
        ProductSpace::<SU2Irrep>::new(vec![]),
    );

    let dst = src.permute(&[0, 1], &[]).unwrap();
    let transformer = tree_permuter(&src, &dst, &[0, 1], &[]).unwrap();
    let data = expect_generic(&transformer);

    assert!(!data.is_empty());
    for block in data {
        assert_eq!(
            block.basis_transform.nrows(),
            block.dst.strides_offsets.len()
        );
        assert_eq!(
            block.basis_transform.ncols(),
            block.src.strides_offsets.len()
        );
        assert_eq!(block.basis_transform.nrows(), block.basis_transform.ncols());
        for row in 0..block.basis_transform.nrows() {
            for col in 0..block.basis_transform.ncols() {
                let expected = if row == col { 1.0 } else { 0.0 };
                assert!((block.basis_transform[[row, col]] - expected).abs() < 1.0e-12);
            }
        }
    }
}

#[test]
fn su2_scalar_transform_coefficients_match_tensorkit_conventions() {
    {
        let half = GradedSpace::new(vec![(su2(1), 1)], false).unwrap();
        let src = HomSpace::new(
            ProductSpace::new(vec![half.clone()]),
            ProductSpace::new(vec![half]),
        );
        assert_generic_permute_scalar_coeff(src, &[1], &[0], 1.0);
    }

    {
        let half = GradedSpace::new(vec![(su2(1), 1)], false).unwrap();
        let src = HomSpace::new(
            ProductSpace::new(vec![half.clone(), half]),
            ProductSpace::<SU2Irrep>::new(vec![]),
        );
        assert_generic_permute_scalar_coeff(src, &[1], &[0], 0.5_f64.sqrt());
    }
}

#[test]
fn su2_transpose_matches_tensorkit_multi_fmove_coefficients() {
    let half = GradedSpace::new(vec![(su2(1), 1)], false).unwrap();
    let src = HomSpace::new(
        ProductSpace::new(vec![half.clone(), half.clone(), half.clone(), half]),
        ProductSpace::<SU2Irrep>::new(vec![]),
    );

    let dst = src.permute(&[1, 2, 3], &[0]).unwrap();
    let transformer = tree_transposer(&src, &dst, &[1, 2, 3], &[0]).unwrap();
    let data = expect_generic(&transformer);

    assert_eq!(data.len(), 1);
    let block = &data[0];
    assert_eq!(block.basis_transform.shape(), &[2, 2]);
    assert!((block.basis_transform[[0, 0]] + (1.0_f64 / 8.0).sqrt()).abs() < 1.0e-12);
    assert!((block.basis_transform[[0, 1]] - (3.0_f64 / 8.0).sqrt()).abs() < 1.0e-12);
    assert!((block.basis_transform[[1, 0]] - (3.0_f64 / 8.0).sqrt()).abs() < 1.0e-12);
    assert!((block.basis_transform[[1, 1]] - (1.0_f64 / 8.0).sqrt()).abs() < 1.0e-12);
}

#[test]
fn u1_su2_recoupling_keeps_product_block_structure() {
    let charged_half = GradedSpace::new(vec![(u1_su2(1, 1), 1)], false).unwrap();
    let opposite_half = GradedSpace::new(vec![(u1_su2(-1, 1), 1)], false).unwrap();
    let src = HomSpace::new(
        ProductSpace::new(vec![
            charged_half.clone(),
            opposite_half.clone(),
            charged_half,
            opposite_half,
        ]),
        ProductSpace::<U1SU2Irrep>::new(vec![]),
    );

    let dst = src.permute(&[1, 2, 3, 0], &[]).unwrap();
    let transformer = tree_permuter(&src, &dst, &[1, 2, 3, 0], &[]).unwrap();
    let data = expect_generic(&transformer);

    assert_eq!(data.len(), 1);
    let block = &data[0];
    assert_eq!(block.basis_transform.shape(), &[2, 2]);
    assert_eq!(block.src.strides_offsets.len(), 2);
    assert_eq!(block.dst.strides_offsets.len(), 2);
}

#[test]
fn product_sector_transformer_uses_componentwise_symbols() {
    {
        let left = GradedSpace::new(vec![(parity_su2(1, 1), 1)], false).unwrap();
        let right = GradedSpace::new(vec![(parity_su2(1, 1), 1)], false).unwrap();
        let src = HomSpace::new(
            ProductSpace::new(vec![left, right]),
            ProductSpace::<FermionParitySU2Irrep>::new(vec![]),
        );
        let expected = FermionParity::r_symbol(&parity(1), &parity(1), &parity(0))
            * SU2Irrep::r_symbol(&su2(1), &su2(1), &su2(0));

        assert_generic_permute_scalar_coeff(src, &[1, 0], &[], expected);
    }

    {
        let left_sector = parity_u1_su2(1, 1, 1);
        let right_sector = parity_u1_su2(1, -1, 1);
        let left = GradedSpace::new(vec![(left_sector.clone(), 1)], false).unwrap();
        let right = GradedSpace::new(vec![(right_sector.clone(), 1)], false).unwrap();
        let src = HomSpace::new(
            ProductSpace::new(vec![left, right]),
            ProductSpace::<FermionParityU1SU2Irrep>::new(vec![]),
        );
        let coupled = FermionParityU1SU2Irrep::unit();
        let braid = FermionParityU1SU2Irrep::r_symbol(&left_sector, &right_sector, &coupled);
        let bend = ((coupled.quantum_dim() as f64) / (right_sector.quantum_dim() as f64)).sqrt()
            * FermionParityU1SU2Irrep::b_symbol(&right_sector, &left_sector, &coupled).unwrap();

        assert_generic_permute_scalar_coeff(src, &[1], &[0], braid * bend);
    }
}
