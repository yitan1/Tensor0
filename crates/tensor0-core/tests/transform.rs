use tensor0_core::layout::build_sector_structure;
use tensor0_core::sector::{
    FermionNumber, FermionParity, FermionParitySU2Irrep, FermionParityU1SU2Irrep, SU2Irrep, Sector,
    U1Irrep, U1SU2Irrep,
};
use tensor0_core::space::{GradedSpace, HomSpace, ProductSpace};
use tensor0_core::transform::{
    flip_entries as build_flip_entries,
    reweighting::{twist_is_trivial, twist_subblock_factors},
    trace_transformer, tree_permuter as build_tree_permuter,
    tree_transposer as build_tree_transposer, AbelianTransformData, GenericTransformData,
    TreeTransformer,
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

fn empty_product<I: Sector>() -> ProductSpace<I> {
    ProductSpace::new(vec![])
}

fn parity_hom_space() -> HomSpace<FermionParity> {
    let factor = GradedSpace::new(vec![(parity(0), 1), (parity(1), 1)], false).unwrap();
    HomSpace::from_factor_spaces(vec![factor.clone()], vec![factor])
}

fn classified_twist_subblock_factors<I: Sector>(
    space: &HomSpace<I>,
    indices: &[usize],
    inv: bool,
) -> tensor0_core::error::Result<Option<Vec<f64>>> {
    if twist_is_trivial(space, indices)? {
        return Ok(None);
    }
    let structure = build_sector_structure(space)?;
    twist_subblock_factors(space, &structure, indices, inv)
}

fn trace_metadata<I: Sector>(
    src: &HomSpace<I>,
    dst: &HomSpace<I>,
) -> tensor0_core::error::Result<Vec<(usize, usize, f64)>> {
    let src_structure = build_sector_structure(src)?;
    let dst_structure = build_sector_structure(dst)?;
    trace_metadata_from_structures(src, dst, &src_structure, &dst_structure)
}

fn trace_metadata_from_structures<I: Sector>(
    src: &HomSpace<I>,
    dst: &HomSpace<I>,
    src_structure: &tensor0_core::layout::SectorStructure<I>,
    dst_structure: &tensor0_core::layout::SectorStructure<I>,
) -> tensor0_core::error::Result<Vec<(usize, usize, f64)>> {
    let p_codomain = (0..src.numout()).collect::<Vec<_>>();
    let p_domain = (src.numout()..src.numind()).collect::<Vec<_>>();
    let transformer = composed_trace_transformer(
        src,
        dst,
        src_structure,
        dst_structure,
        &p_codomain,
        &p_domain,
    )?;

    let mut entries = match transformer {
        TreeTransformer::Abelian(data) => data
            .into_iter()
            .map(|entry| (entry.src, entry.dst, entry.coeff))
            .collect(),
        TreeTransformer::Generic(groups) => {
            let mut entries = Vec::new();
            for group in groups {
                for (row, dst) in group.dst_indices.iter().enumerate() {
                    for (column, src) in group.src_indices.iter().enumerate() {
                        let coefficient = group.transform[[row, column]];
                        if coefficient != 0.0 {
                            entries.push((*src, *dst, coefficient));
                        }
                    }
                }
            }
            entries
        }
    };
    entries.sort_by_key(|(src, dst, _)| (*src, *dst));
    Ok(entries)
}

fn composed_trace_transformer<I: Sector>(
    src: &HomSpace<I>,
    dst: &HomSpace<I>,
    src_structure: &tensor0_core::layout::SectorStructure<I>,
    dst_structure: &tensor0_core::layout::SectorStructure<I>,
    p_codomain: &[usize],
    p_domain: &[usize],
) -> tensor0_core::error::Result<TreeTransformer> {
    let canonical = src.permute(p_codomain, p_domain)?;
    let canonical_structure = if &canonical == src {
        src_structure.clone()
    } else {
        build_sector_structure(&canonical)?
    };
    let basis_transformer = build_tree_permuter(
        src,
        &canonical,
        src_structure,
        &canonical_structure,
        p_codomain,
        p_domain,
    )?;
    trace_transformer(
        &canonical,
        dst,
        &canonical_structure,
        dst_structure,
        &basis_transformer,
    )
}

fn tree_permuter<I: Sector>(
    src: &HomSpace<I>,
    dst: &HomSpace<I>,
    p_codomain: &[usize],
    p_domain: &[usize],
) -> tensor0_core::error::Result<TreeTransformer> {
    let src_structure = build_sector_structure(src)?;
    let dst_structure = build_sector_structure(dst)?;
    build_tree_permuter(
        src,
        dst,
        &src_structure,
        &dst_structure,
        p_codomain,
        p_domain,
    )
}

fn tree_transposer<I: Sector>(
    src: &HomSpace<I>,
    dst: &HomSpace<I>,
    p_codomain: &[usize],
    p_domain: &[usize],
) -> tensor0_core::error::Result<TreeTransformer> {
    let src_structure = build_sector_structure(src)?;
    let dst_structure = build_sector_structure(dst)?;
    build_tree_transposer(
        src,
        dst,
        &src_structure,
        &dst_structure,
        p_codomain,
        p_domain,
    )
}

fn flip_entries<I: Sector>(
    src: &HomSpace<I>,
    indices: &[usize],
    inv: bool,
) -> tensor0_core::error::Result<Vec<(usize, f64)>> {
    let dst = src.flip(indices)?;
    let src_structure = build_sector_structure(src)?;
    let dst_structure = build_sector_structure(&dst)?;
    build_flip_entries(src, &dst, &src_structure, &dst_structure, indices, inv)
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

fn assert_single_abelian_mapping(data: &[AbelianTransformData], coeff: f64) {
    assert_eq!(data.len(), 1);
    assert_eq!(data[0].coeff, coeff);
    assert_eq!(data[0].src, 0);
    assert_eq!(data[0].dst, 0);
}

fn assert_single_flip_entry(entries: &[(usize, f64)], coeff: f64) {
    assert_eq!(entries, &[(0, coeff)]);
}

fn assert_single_generic_scalar_block(data: &[GenericTransformData], expected_coeff: f64) {
    assert_eq!(data.len(), 1);
    let block = &data[0];
    assert_eq!(block.transform.shape(), &[1, 1]);
    assert_close(block.transform[[0, 0]], expected_coeff);
    assert_eq!(block.src_indices, vec![0]);
    assert_eq!(block.dst_indices, vec![0]);
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
fn tree_transposer_rejects_noncyclic_permutation_for_empty_structure() {
    let empty = GradedSpace::<U1Irrep>::new(vec![], false).unwrap();
    let src = HomSpace::new(
        ProductSpace::new(vec![empty.clone(), empty.clone()]),
        ProductSpace::new(vec![empty.clone(), empty]),
    );
    let p_codomain = [1, 0];
    let p_domain = [3, 2];
    let dst = src.permute(&p_codomain, &p_domain).unwrap();
    let src_structure = build_sector_structure(&src).unwrap();
    let dst_structure = build_sector_structure(&dst).unwrap();

    let error = build_tree_transposer(
        &src,
        &dst,
        &src_structure,
        &dst_structure,
        &p_codomain,
        &p_domain,
    )
    .unwrap_err();

    assert_eq!(
        error.to_string(),
        "fusion tree transpose requires a cyclic planar permutation",
    );
}

#[test]
fn tree_transformer_rejects_incompatible_cached_sector_layouts() {
    let factor = GradedSpace::new(vec![(u1(0), 1)], false).unwrap();
    let src = HomSpace::new(one_factor(factor.clone()), one_factor(factor));
    let dst = src.permute(&[1], &[0]).unwrap();
    let src_structure = build_sector_structure(&src).unwrap();
    let dst_structure = build_sector_structure(&dst).unwrap();

    let other = GradedSpace::new(vec![(u1(1), 1)], false).unwrap();
    let wrong_src = HomSpace::new(one_factor(other.clone()), one_factor(other));
    let wrong_src_structure = build_sector_structure(&wrong_src).unwrap();
    let err = build_tree_permuter(&src, &dst, &wrong_src_structure, &dst_structure, &[1], &[0])
        .unwrap_err();
    assert_eq!(
        err.to_string(),
        "source sectorstructure does not match source HomSpace sector structure",
    );

    let wrong_dst = wrong_src.permute(&[1], &[0]).unwrap();
    let wrong_dst_structure = build_sector_structure(&wrong_dst).unwrap();
    let err = build_tree_permuter(&src, &dst, &src_structure, &wrong_dst_structure, &[1], &[0])
        .unwrap_err();
    assert_eq!(
        err.to_string(),
        "destination sectorstructure does not match destination HomSpace sector structure",
    );
}

#[test]
fn tree_transformer_reuses_sector_layout_across_degeneracy_dimensions() {
    let cached_half = GradedSpace::new(vec![(su2(1), 1)], false).unwrap();
    let target_half = GradedSpace::new(vec![(su2(1), 3)], false).unwrap();
    let source = |half: &GradedSpace<SU2Irrep>| {
        HomSpace::new(
            ProductSpace::new(vec![half.clone(), half.clone(), half.clone(), half.clone()]),
            empty_product::<SU2Irrep>(),
        )
    };
    let cached_src = source(&cached_half);
    let cached_dst = cached_src.permute(&[1, 2, 3], &[0]).unwrap();
    let target_src = source(&target_half);
    let target_dst = target_src.permute(&[1, 2, 3], &[0]).unwrap();
    let cached_src_structure = build_sector_structure(&cached_src).unwrap();
    let cached_dst_structure = build_sector_structure(&cached_dst).unwrap();
    let target_src_structure = build_sector_structure(&target_src).unwrap();
    let target_dst_structure = build_sector_structure(&target_dst).unwrap();

    let reused = build_tree_transposer(
        &target_src,
        &target_dst,
        &cached_src_structure,
        &cached_dst_structure,
        &[1, 2, 3],
        &[0],
    )
    .unwrap();
    let direct = build_tree_transposer(
        &target_src,
        &target_dst,
        &target_src_structure,
        &target_dst_structure,
        &[1, 2, 3],
        &[0],
    )
    .unwrap();

    assert_eq!(reused, direct);
    let data = expect_generic(&reused);
    assert_eq!(data[0].src_indices, vec![0, 1]);
    assert_eq!(data[0].dst_indices, vec![0, 1]);
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
        assert_single_abelian_mapping(expect_abelian(&transformer), 1.0);

        let transposer = tree_transposer(&src, &dst, &[1], &[0]).unwrap();
        assert_single_abelian_mapping(expect_abelian(&transposer), 1.0);
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
        assert_single_abelian_mapping(expect_abelian(&transformer), 1.0);
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
        assert_eq!(block.transform.nrows(), block.dst_indices.len());
        assert_eq!(block.transform.ncols(), block.src_indices.len());
        assert_eq!(block.transform.nrows(), block.transform.ncols());
        for row in 0..block.transform.nrows() {
            for col in 0..block.transform.ncols() {
                let expected = if row == col { 1.0 } else { 0.0 };
                assert!((block.transform[[row, col]] - expected).abs() < 1.0e-12);
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
    assert_eq!(block.transform.shape(), &[2, 2]);
    assert_eq!(block.src_indices, vec![0, 1]);
    assert_eq!(block.dst_indices, vec![0, 1]);
    assert!((block.transform[[0, 0]] + (1.0_f64 / 8.0).sqrt()).abs() < 1.0e-12);
    assert!((block.transform[[0, 1]] - (3.0_f64 / 8.0).sqrt()).abs() < 1.0e-12);
    assert!((block.transform[[1, 0]] - (3.0_f64 / 8.0).sqrt()).abs() < 1.0e-12);
    assert!((block.transform[[1, 1]] - (1.0_f64 / 8.0).sqrt()).abs() < 1.0e-12);
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
    assert_eq!(block.transform.shape(), &[2, 2]);
    assert_eq!(block.src_indices, vec![0, 1]);
    assert_eq!(block.dst_indices, vec![0, 1]);
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

#[test]
fn u1_trace_metadata_omits_nonmatching_tails_and_coalesces_destinations() {
    let factor = GradedSpace::new(vec![(u1(0), 1), (u1(1), 1)], false).unwrap();
    let product = || ProductSpace::new(vec![factor.clone(), factor.clone()]);
    let src = HomSpace::new(product(), product());
    let dst = HomSpace::new(one_factor(factor.clone()), one_factor(factor));

    assert_eq!(
        trace_metadata(&src, &dst).unwrap(),
        vec![(0, 0, 1.0), (1, 1, 1.0), (4, 0, 1.0), (5, 1, 1.0),],
    );
}

#[test]
fn u1_trace_metadata_splits_multiple_canonical_tail_pairs() {
    let open = GradedSpace::new(vec![(u1(2), 1)], false).unwrap();
    let first_trace = GradedSpace::new(vec![(u1(1), 1)], false).unwrap();
    let second_trace = GradedSpace::new(vec![(u1(-1), 1)], false).unwrap();
    let product = || {
        ProductSpace::new(vec![
            open.clone(),
            first_trace.clone(),
            second_trace.clone(),
        ])
    };
    let src = HomSpace::new(product(), product());
    let dst = HomSpace::new(one_factor(open.clone()), one_factor(open));

    assert_eq!(trace_metadata(&src, &dst).unwrap(), vec![(0, 0, 1.0)],);
}

#[test]
fn rank_zero_trace_metadata_handles_empty_source_and_destination_trees() {
    let src = HomSpace::new(empty_product::<U1Irrep>(), empty_product());
    let dst = HomSpace::new(empty_product::<U1Irrep>(), empty_product());

    assert_eq!(trace_metadata(&src, &dst).unwrap(), vec![(0, 0, 1.0)],);
}

#[test]
fn su2_trace_metadata_uses_tail_quantum_dimension_ratio() {
    let half = GradedSpace::new(vec![(su2(1), 1)], false).unwrap();
    let product = || ProductSpace::new(vec![half.clone(), half.clone()]);
    let src = HomSpace::new(product(), product());
    let dst = HomSpace::new(
        ProductSpace::new(vec![half.clone()]),
        ProductSpace::new(vec![half]),
    );

    assert_eq!(
        trace_metadata(&src, &dst).unwrap(),
        vec![(0, 0, 0.5), (1, 0, 1.5)],
    );
}

#[test]
fn su2_grouped_trace_fuses_nontrivial_permutation_matrix() {
    let half = GradedSpace::new(vec![(su2(1), 1)], false).unwrap();
    let src = HomSpace::new(
        ProductSpace::new(vec![half.clone(), half.clone(), half.clone(), half.dual()]),
        empty_product::<SU2Irrep>(),
    );
    let dst = HomSpace::new(
        ProductSpace::new(vec![half.clone(), half]),
        empty_product::<SU2Irrep>(),
    );
    let src_structure = build_sector_structure(&src).unwrap();
    let dst_structure = build_sector_structure(&dst).unwrap();

    let transformer =
        composed_trace_transformer(&src, &dst, &src_structure, &dst_structure, &[1, 2, 3], &[0])
            .unwrap();
    let groups = expect_generic(&transformer);

    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].transform.shape(), &[1, 2]);
    assert_eq!(groups[0].src_indices, vec![0, 1]);
    assert_eq!(groups[0].dst_indices, vec![0]);
    assert_close(groups[0].transform[[0, 0]], -(0.5_f64).sqrt());
    assert_close(groups[0].transform[[0, 1]], (1.5_f64).sqrt());
}

#[test]
fn su2_grouped_trace_coalesces_canonical_rows() {
    let half = GradedSpace::new(vec![(su2(1), 1)], false).unwrap();
    let product = || ProductSpace::new(vec![half.clone(), half.clone()]);
    let src = HomSpace::new(product(), product());
    let dst = HomSpace::new(
        ProductSpace::new(vec![half.clone()]),
        ProductSpace::new(vec![half]),
    );
    let src_structure = build_sector_structure(&src).unwrap();
    let dst_structure = build_sector_structure(&dst).unwrap();

    let transformer =
        composed_trace_transformer(&src, &dst, &src_structure, &dst_structure, &[0, 1], &[2, 3])
            .unwrap();
    let groups = expect_generic(&transformer);

    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].transform.shape(), &[1, 2]);
    assert_eq!(groups[0].src_indices, vec![0, 1]);
    assert_eq!(groups[0].dst_indices, vec![0]);
    assert_close(groups[0].transform[[0, 0]], 0.5);
    assert_close(groups[0].transform[[0, 1]], 1.5);
}

#[test]
fn fermion_trace_metadata_applies_twist_only_to_nondual_tail_sectors() {
    let odd = || GradedSpace::new(vec![(parity(1), 1)], false).unwrap();
    let odd_dual = || GradedSpace::new(vec![(parity(1), 1)], true).unwrap();
    let dst = HomSpace::new(
        empty_product::<FermionParity>(),
        empty_product::<FermionParity>(),
    );

    let nondual_src = HomSpace::from_factor_spaces(vec![odd()], vec![odd()]);
    assert_eq!(
        trace_metadata(&nondual_src, &dst).unwrap(),
        vec![(0, 0, -1.0)],
    );

    let dual_src = HomSpace::from_factor_spaces(vec![odd_dual()], vec![odd_dual()]);
    assert_eq!(trace_metadata(&dual_src, &dst).unwrap(), vec![(0, 0, 1.0)],);
}

#[test]
fn product_sector_full_trace_metadata_multiplies_dimension_and_twist() {
    let odd_half = GradedSpace::new(vec![(parity_su2(1, 1), 1)], false).unwrap();
    let src = HomSpace::from_factor_spaces(vec![odd_half.clone()], vec![odd_half]);
    let dst = HomSpace::new(
        empty_product::<FermionParitySU2Irrep>(),
        empty_product::<FermionParitySU2Irrep>(),
    );

    assert_eq!(trace_metadata(&src, &dst).unwrap(), vec![(0, 0, -2.0)],);
}

#[test]
fn trace_metadata_reuses_cached_structures_across_degeneracy_dimensions() {
    let cached = GradedSpace::new(vec![(u1(0), 1)], false).unwrap();
    let target = GradedSpace::new(vec![(u1(0), 3)], false).unwrap();
    let cached_product = || ProductSpace::new(vec![cached.clone(), cached.clone()]);
    let target_product = || ProductSpace::new(vec![target.clone(), target.clone()]);
    let cached_src = HomSpace::new(cached_product(), cached_product());
    let cached_dst = HomSpace::new(one_factor(cached.clone()), one_factor(cached));
    let target_src = HomSpace::new(target_product(), target_product());
    let target_dst = HomSpace::new(one_factor(target.clone()), one_factor(target));
    let src_structure = build_sector_structure(&cached_src).unwrap();
    let dst_structure = build_sector_structure(&cached_dst).unwrap();

    assert_eq!(
        trace_metadata_from_structures(&target_src, &target_dst, &src_structure, &dst_structure)
            .unwrap(),
        vec![(0, 0, 1.0)],
    );
}

#[test]
fn trace_metadata_rejects_incompatible_cached_sector_layouts() {
    let open = GradedSpace::new(vec![(u1(0), 1)], false).unwrap();
    let traced = GradedSpace::new(vec![(u1(1), 1)], false).unwrap();
    let product = || ProductSpace::new(vec![open.clone(), traced.clone()]);
    let src = HomSpace::new(product(), product());
    let dst = HomSpace::new(one_factor(open.clone()), one_factor(open.clone()));
    let src_structure = build_sector_structure(&src).unwrap();
    let dst_structure = build_sector_structure(&dst).unwrap();
    let basis_transformer =
        build_tree_permuter(&src, &src, &src_structure, &src_structure, &[0, 1], &[2, 3]).unwrap();

    let incompatible_canonical_spaces = [
        {
            let other = GradedSpace::new(vec![(u1(2), 1)], false).unwrap();
            let product = || ProductSpace::new(vec![open.clone(), other.clone()]);
            HomSpace::new(product(), product())
        },
        {
            let product = || ProductSpace::new(vec![traced.clone(), open.clone()]);
            HomSpace::new(product(), product())
        },
        {
            let traced_dual = traced.dual();
            let product = || ProductSpace::new(vec![open.clone(), traced_dual.clone()]);
            HomSpace::new(product(), product())
        },
    ];

    for incompatible_canonical in incompatible_canonical_spaces {
        let canonical_structure = build_sector_structure(&incompatible_canonical).unwrap();
        let err = trace_transformer(
            &src,
            &dst,
            &canonical_structure,
            &dst_structure,
            &basis_transformer,
        )
        .unwrap_err();
        assert_eq!(
            err.to_string(),
            "canonical sectorstructure does not match canonical HomSpace sector structure",
        );
    }

    let wrong_dst = HomSpace::new(one_factor(traced.clone()), one_factor(traced));
    let wrong_dst_structure = build_sector_structure(&wrong_dst).unwrap();
    let err = trace_transformer(
        &src,
        &dst,
        &src_structure,
        &wrong_dst_structure,
        &basis_transformer,
    )
    .unwrap_err();
    assert_eq!(
        err.to_string(),
        "destination sectorstructure does not match destination HomSpace sector structure",
    );
}

#[test]
fn twist_subblock_factors_follow_canonical_fusion_tree_pair_order() {
    let space = parity_hom_space();

    assert_eq!(
        classified_twist_subblock_factors(&space, &[0], false).unwrap(),
        Some(vec![1.0, -1.0])
    );
    assert_eq!(
        classified_twist_subblock_factors(&space, &[1], false).unwrap(),
        Some(vec![1.0, -1.0])
    );
    assert_eq!(
        classified_twist_subblock_factors(&space, &[0, 1], false).unwrap(),
        None
    );
    assert_eq!(
        classified_twist_subblock_factors(&space, &[0], true).unwrap(),
        Some(vec![1.0, -1.0])
    );
}

#[test]
fn twist_subblock_factors_distinguish_row_and_column_visible_indices() {
    let odd = || GradedSpace::new(vec![(parity(1), 1)], false).unwrap();
    let even = || GradedSpace::new(vec![(parity(0), 1)], false).unwrap();
    let space = HomSpace::from_factor_spaces(vec![odd(), odd()], vec![even(), even()]);

    assert_eq!(
        classified_twist_subblock_factors(&space, &[0], false).unwrap(),
        Some(vec![-1.0])
    );
    assert_eq!(
        classified_twist_subblock_factors(&space, &[2], false).unwrap(),
        None
    );
}

#[test]
fn twist_is_trivial_classifies_empty_and_bosonic_twists_as_identity() {
    let parity_space = parity_hom_space();
    assert!(twist_is_trivial(&parity_space, &[]).unwrap());

    let factor = GradedSpace::new(vec![(u1(0), 1), (u1(1), 1)], false).unwrap();
    let u1_space = HomSpace::from_factor_spaces(vec![factor.clone()], vec![factor]);
    assert!(twist_is_trivial(&u1_space, &[0]).unwrap());

    let even = GradedSpace::new(vec![(parity(0), 1)], false).unwrap();
    let even_space = HomSpace::from_factor_spaces(vec![even.clone()], vec![even]);
    assert!(twist_is_trivial(&even_space, &[0]).unwrap());
}

#[test]
fn twist_metadata_rejects_invalid_visible_indices() {
    let space = parity_hom_space();

    let out_of_range = classified_twist_subblock_factors(&space, &[2], false).unwrap_err();
    assert!(out_of_range.to_string().contains("out of range"));

    let duplicate = classified_twist_subblock_factors(&space, &[0, 0], false).unwrap_err();
    assert!(duplicate.to_string().contains("unique"));

    let odd = GradedSpace::new(vec![(parity(1), 1)], false).unwrap();
    let other_space = HomSpace::from_factor_spaces(vec![odd.clone()], vec![odd]);
    let other_structure = build_sector_structure(&other_space).unwrap();
    let mismatch = twist_subblock_factors(&space, &other_structure, &[0], false).unwrap_err();
    assert!(mismatch.to_string().contains("does not match"));
}

#[test]
fn u1_flip_entries_map_canonical_pairs_one_to_one() {
    let factor = GradedSpace::new(vec![(u1(-1), 2), (u1(0), 3), (u1(1), 5)], false).unwrap();
    let src = HomSpace::from_factor_spaces(vec![factor.clone()], vec![factor]);

    let entries = flip_entries(&src, &[0, 1], false).unwrap();

    assert!(!entries.is_empty());
    assert!(entries.iter().all(|(_, coeff)| *coeff == 1.0));
    assert!(entries
        .iter()
        .enumerate()
        .all(|(src, (dst, _))| src == *dst));
}

#[test]
fn u1_flip_entries_record_canonical_order_changes() {
    let factor = GradedSpace::new(vec![(u1(-1), 1), (u1(1), 1)], false).unwrap();
    let src = HomSpace::new(
        ProductSpace::new(vec![factor.clone(), factor]),
        empty_product::<U1Irrep>(),
    );

    let entries = flip_entries(&src, &[1], false).unwrap();

    assert_eq!(entries, vec![(1, 1.0), (0, 1.0)]);
}

#[test]
fn su2_flip_entries_encode_inverse_coefficients() {
    let half = GradedSpace::new(vec![(su2(1), 1)], false).unwrap();
    let src = HomSpace::from_factor_spaces(vec![half.clone()], vec![half]);

    let forward = flip_entries(&src, &[0], false).unwrap();
    assert_single_flip_entry(&forward, 1.0);

    let flipped = src.flip(&[0]).unwrap();
    let inverse = flip_entries(&flipped, &[0], true).unwrap();
    assert_single_flip_entry(&inverse, 1.0);

    let second_forward = flip_entries(&flipped, &[0], false).unwrap();
    assert_single_flip_entry(&second_forward, -1.0);
}

#[test]
fn su2_flip_entries_align_multitree_canonical_indices() {
    let half = GradedSpace::new(vec![(su2(1), 1)], false).unwrap();
    let src = HomSpace::new(
        ProductSpace::new(vec![half.clone(), half.clone(), half.clone(), half]),
        empty_product::<SU2Irrep>(),
    );

    let entries = flip_entries(&src, &[0, 2], false).unwrap();

    assert_eq!(entries, vec![(0, 1.0), (1, 1.0)]);
}

#[test]
fn product_sector_flip_composes_frobenius_schur_and_twist_components() {
    let odd = GradedSpace::new(vec![(fermion_number(1, 1), 1)], false).unwrap();
    let src = HomSpace::from_factor_spaces(vec![odd.clone()], vec![odd]);

    let row = flip_entries(&src, &[0], false).unwrap();
    let column = flip_entries(&src, &[1], false).unwrap();
    assert_single_flip_entry(&row, 1.0);
    assert_single_flip_entry(&column, -1.0);
}

#[test]
fn flip_entries_reject_destination_and_structure_mismatches() {
    let factor = GradedSpace::new(vec![(u1(0), 1)], false).unwrap();
    let src = HomSpace::from_factor_spaces(vec![factor.clone()], vec![factor.clone()]);
    let dst = src.flip(&[0]).unwrap();
    let src_structure = build_sector_structure(&src).unwrap();
    let dst_structure = build_sector_structure(&dst).unwrap();

    let err =
        build_flip_entries(&src, &src, &src_structure, &src_structure, &[0], false).unwrap_err();
    assert!(err.to_string().contains("incompatible spaces"));

    let other = GradedSpace::new(vec![(u1(1), 1)], false).unwrap();
    let wrong_src = HomSpace::from_factor_spaces(vec![other.clone()], vec![other]);
    let wrong_structure = build_sector_structure(&wrong_src).unwrap();
    let err =
        build_flip_entries(&src, &dst, &wrong_structure, &dst_structure, &[0], false).unwrap_err();
    assert!(err.to_string().contains("source sectorstructure"));
}

#[test]
fn flip_entries_reuse_sector_layout_across_degeneracy_dimensions() {
    let cached = GradedSpace::new(vec![(su2(1), 1)], false).unwrap();
    let target = GradedSpace::new(vec![(su2(1), 3)], false).unwrap();
    let make_hom = |factor: &GradedSpace<SU2Irrep>| {
        HomSpace::from_factor_spaces(vec![factor.clone()], vec![factor.clone()])
    };
    let cached_src = make_hom(&cached);
    let cached_dst = cached_src.flip(&[0]).unwrap();
    let target_src = make_hom(&target);
    let target_dst = target_src.flip(&[0]).unwrap();
    let cached_src_structure = build_sector_structure(&cached_src).unwrap();
    let cached_dst_structure = build_sector_structure(&cached_dst).unwrap();
    let target_src_structure = build_sector_structure(&target_src).unwrap();
    let target_dst_structure = build_sector_structure(&target_dst).unwrap();

    let reused = build_flip_entries(
        &target_src,
        &target_dst,
        &cached_src_structure,
        &cached_dst_structure,
        &[0],
        false,
    )
    .unwrap();
    let direct = build_flip_entries(
        &target_src,
        &target_dst,
        &target_src_structure,
        &target_dst_structure,
        &[0],
        false,
    )
    .unwrap();
    assert_eq!(reused, direct);
}
