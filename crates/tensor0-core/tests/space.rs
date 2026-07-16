use std::collections::BTreeMap;
use std::fmt::Debug;

use tensor0_core::error::{Result, Tensor0Error};
use tensor0_core::sector::{
    EncodedSectorValue, FermionParity, ProductSector, SU2Irrep, Sector, SectorSpec, U1Irrep,
    Z4Irrep,
};
use tensor0_core::space::{
    fuse_product_space, infimum_space, supremum_space, ElementarySpaceSpec, GradedSpace, HomSpace,
    ProductSpace, ProductSpaceSpec, SectorDimSpec,
};

type FermionNumber = ProductSector<(U1Irrep, FermionParity)>;

#[derive(Clone)]
struct SpaceCase<I: Sector> {
    primary: GradedSpace<I>,
    secondary: GradedSpace<I>,
    dual: GradedSpace<I>,
    product: ProductSpace<I>,
    empty_product: ProductSpace<I>,
}

#[derive(Clone)]
struct HomCase<I: Sector> {
    codomain: Vec<GradedSpace<I>>,
    domain: Vec<GradedSpace<I>>,
    hom: HomSpace<I>,
    p_codomain: Vec<usize>,
    p_domain: Vec<usize>,
}

fn u1(charge2: i64) -> U1Irrep {
    U1Irrep::charge2(charge2).unwrap()
}

fn su2(spin2: i64) -> SU2Irrep {
    SU2Irrep::spin2(spin2).unwrap()
}

fn fp(value: i64) -> FermionParity {
    FermionParity::new(value).unwrap()
}

fn z4(value: i64) -> Z4Irrep {
    Z4Irrep::new(value).unwrap()
}

fn fermion_number(charge2: i64, parity: i64) -> FermionNumber {
    FermionNumber::decode_value(&[charge2, parity]).unwrap()
}

fn gs<I: Sector>(sector_dims: Vec<(I, usize)>) -> GradedSpace<I> {
    GradedSpace::new(sector_dims, false).unwrap()
}

fn ps<I: Sector>(factors: Vec<GradedSpace<I>>) -> ProductSpace<I> {
    ProductSpace::new(factors)
}

fn ev(xs: &[i64]) -> EncodedSectorValue {
    xs.iter().copied().collect()
}

fn sd(xs: &[i64], dim: usize) -> SectorDimSpec {
    SectorDimSpec {
        sector: xs.to_vec(),
        dim,
    }
}

fn space_case<I: Sector>(primary: GradedSpace<I>, secondary: GradedSpace<I>) -> SpaceCase<I> {
    SpaceCase {
        dual: primary.dual(),
        product: ps(vec![primary.clone(), secondary.clone()]),
        empty_product: ps::<I>(vec![]),
        primary,
        secondary,
    }
}

fn u1_space_case() -> SpaceCase<U1Irrep> {
    space_case(
        gs(vec![(u1(1), 3), (u1(0), 2), (u1(-1), 5)]),
        gs(vec![(u1(0), 7), (u1(-1), 11)]),
    )
}

fn su2_space_case() -> SpaceCase<SU2Irrep> {
    space_case(
        gs(vec![(su2(1), 3), (su2(0), 2)]),
        gs(vec![(su2(1), 5), (su2(2), 7)]),
    )
}

fn fermion_parity_space_case() -> SpaceCase<FermionParity> {
    space_case(
        gs(vec![(fp(1), 3), (fp(0), 2)]),
        gs(vec![(fp(0), 7), (fp(1), 11)]),
    )
}

fn z4_space_case() -> SpaceCase<Z4Irrep> {
    space_case(
        gs(vec![(z4(3), 3), (z4(0), 2), (z4(1), 5)]),
        gs(vec![(z4(2), 7), (z4(1), 11)]),
    )
}

fn fermion_number_space_case() -> SpaceCase<FermionNumber> {
    space_case(
        gs(vec![
            (fermion_number(1, 1), 2),
            (fermion_number(0, 0), 4),
            (fermion_number(-1, 0), 5),
        ]),
        gs(vec![(fermion_number(0, 1), 3)]),
    )
}

fn hom_case<I: Sector>(
    codomain: Vec<GradedSpace<I>>,
    domain: Vec<GradedSpace<I>>,
    p_codomain: Vec<usize>,
    p_domain: Vec<usize>,
) -> HomCase<I> {
    let hom = HomSpace::<I>::new(ps(codomain.clone()), ps(domain.clone()));

    HomCase {
        codomain,
        domain,
        hom,
        p_codomain,
        p_domain,
    }
}

fn u1_tensorkit_shaped_hom_case() -> HomCase<U1Irrep> {
    let v1 = gs(vec![(u1(0), 2), (u1(1), 1)]);
    let v2 = gs(vec![(u1(-1), 3)]);
    let v3 = gs(vec![(u1(2), 5)]);
    let v4 = gs(vec![(u1(1), 7)]);
    let v5 = gs(vec![(u1(-2), 11)]);

    hom_case(
        vec![v1.clone(), v2.clone()],
        vec![v5.dual(), v4.dual(), v3.dual()],
        vec![1, 4, 3],
        vec![0, 2],
    )
}

fn expected_space_dim<I: Sector>(space: &GradedSpace<I>) -> usize {
    space
        .sectors()
        .into_iter()
        .map(|(sector, dim)| sector.quantum_dim() * dim)
        .sum()
}

fn expected_fused_two_factor_sectors<I: Sector>(
    left: &GradedSpace<I>,
    right: &GradedSpace<I>,
) -> Vec<(I, usize)> {
    let mut sector_dims = BTreeMap::<I, usize>::new();
    for (left_sector, left_dim) in left.sectors() {
        for (right_sector, right_dim) in right.sectors() {
            for coupled in left_sector.fusion_outputs(&right_sector) {
                let multiplicity = I::n_symbol(&left_sector, &right_sector, &coupled);
                if multiplicity == 0 {
                    continue;
                }

                let degeneracy = left_dim * right_dim * multiplicity;
                *sector_dims.entry(coupled).or_insert(0) += degeneracy;
            }
        }
    }
    sector_dims.into_iter().collect()
}

fn product_sector_support<I: Sector>(product: &ProductSpace<I>) -> Vec<(Vec<I>, Vec<usize>)> {
    let mut support = vec![(Vec::<I>::new(), Vec::<usize>::new())];

    for factor in product.factors() {
        let factor_sectors = factor.sectors();
        if factor_sectors.is_empty() {
            return Vec::new();
        }

        let mut next = Vec::new();
        for (sectors, dims) in support {
            for (sector, dim) in &factor_sectors {
                let mut sectors = sectors.clone();
                sectors.push(sector.clone());

                let mut dims = dims.clone();
                dims.push(*dim);

                next.push((sectors, dims));
            }
        }
        support = next;
    }

    support
}

fn expected_visible_legs<I: Sector>(
    codomain: &[GradedSpace<I>],
    domain: &[GradedSpace<I>],
) -> Vec<GradedSpace<I>> {
    let mut visible = codomain.to_vec();
    visible.extend(domain.iter().map(|space| space.dual()));
    visible
}

fn assert_space_sectors<I>(space: &GradedSpace<I>, expected: Vec<(I, usize)>)
where
    I: Sector + Debug,
{
    assert_eq!(space.sectors(), expected);
}

fn assert_err_contains<T>(result: Result<T>, expected: &str) {
    let Err(error) = result else {
        panic!("expected error containing {expected}");
    };
    let message = error.to_string();
    assert!(
        message.contains(expected),
        "expected error containing {expected}, got {message}",
    );
}

fn assert_graded_space_spec_roundtrip<I>(space: &GradedSpace<I>)
where
    I: Sector + Debug,
{
    let from_spec = GradedSpace::<I>::from_spec(space.to_spec()).unwrap();

    assert_eq!(from_spec, (*space).clone());
}

fn assert_product_space_spec_roundtrip<I>(space: &ProductSpace<I>)
where
    I: Sector + Debug,
{
    let from_spec = ProductSpace::<I>::from_spec(space.to_spec()).unwrap();

    assert_eq!(from_spec, (*space).clone());
}

fn assert_product_sector_support_contract<I>(product: &ProductSpace<I>)
where
    I: Sector + Debug,
{
    for (sectors, dims) in product_sector_support(product) {
        let expected_dim = dims.iter().product::<usize>();

        assert_eq!(product.sector_dims(&sectors), Some(dims));
        assert_eq!(product.sector_dim(&sectors), Some(expected_dim));
    }
}

fn assert_hom_space_spec_roundtrip<I>(space: &HomSpace<I>)
where
    I: Sector + Debug,
{
    let from_spec = HomSpace::<I>::from_spec(space.to_spec()).unwrap();

    assert_eq!(from_spec, (*space).clone());
}

fn assert_graded_space_contract<I>(case: &SpaceCase<I>)
where
    I: Sector + Debug,
{
    assert!(!case.primary.is_dual());
    assert!(case.dual.is_dual());

    assert_eq!(case.primary.dim(), expected_space_dim(&case.primary));
    assert_eq!(case.dual.dim(), expected_space_dim(&case.dual));

    for (sector, dim) in case.primary.sectors() {
        assert_eq!(case.primary.sector_dim(&sector), dim);
    }
    for (sector, dim) in case.dual.sectors() {
        assert_eq!(case.dual.sector_dim(&sector), dim);
    }

    let expected_dual = case
        .primary
        .sectors()
        .into_iter()
        .map(|(sector, dim)| (sector.dual(), dim))
        .collect::<Vec<_>>();
    assert_space_sectors(&case.dual, expected_dual);
    assert_eq!(case.dual.to_spec().sectors, case.primary.to_spec().sectors);

    assert_graded_space_spec_roundtrip(&case.primary);
    assert_graded_space_spec_roundtrip(&case.dual);
}

fn assert_product_space_contract<I>(case: &SpaceCase<I>)
where
    I: Sector + Debug,
{
    let expected_factors = vec![case.primary.clone(), case.secondary.clone()];
    let product_dims = vec![case.primary.dim(), case.secondary.dim()];
    let first_sector = case.primary.sectors().into_iter().next().unwrap().0;

    assert_eq!(case.product.factors(), expected_factors.as_slice());
    assert_eq!(case.product.dims(), product_dims);
    assert_eq!(
        case.product.dim(),
        case.primary.dim() * case.secondary.dim()
    );
    assert_product_sector_support_contract(&case.product);
    assert_eq!(case.product.sector_dims(&[first_sector]), None);

    assert!(case.empty_product.factors().is_empty());
    assert_eq!(case.empty_product.dims(), Vec::<usize>::new());
    assert_eq!(case.empty_product.dim(), 1);
    assert_eq!(
        case.empty_product.sector_dims(&Vec::<I>::new()),
        Some(vec![])
    );
    assert_eq!(case.empty_product.sector_dim(&Vec::<I>::new()), Some(1));

    assert_product_space_spec_roundtrip(&case.product);
    assert_product_space_spec_roundtrip(&case.empty_product);
}

fn assert_fuse_two_factor_contract<I>(left: &GradedSpace<I>, right: &GradedSpace<I>)
where
    I: Sector + Debug,
{
    let product = ps(vec![left.clone(), right.clone()]);
    let fused = fuse_product_space(&product).unwrap();

    assert_space_sectors(&fused, expected_fused_two_factor_sectors(left, right));
    assert_eq!(fused.dim(), product.dim());
    assert!(!fused.is_dual());
}

fn assert_hom_space_contract<I>(case: &HomCase<I>)
where
    I: Sector + Debug,
{
    let expected_visible = expected_visible_legs(&case.codomain, &case.domain);

    assert_eq!(case.hom.codomain().factors(), case.codomain.as_slice());
    assert_eq!(case.hom.domain().factors(), case.domain.as_slice());
    assert_eq!(case.hom.numout(), case.codomain.len());
    assert_eq!(case.hom.numin(), case.domain.len());
    assert_eq!(case.hom.numind(), expected_visible.len());
    for (index, expected) in expected_visible.iter().enumerate() {
        assert_eq!(case.hom.visible_leg(index).unwrap(), expected.clone());
    }
    assert_eq!(case.hom.visible_legs(), expected_visible);
    assert_err_contains(case.hom.visible_leg(case.hom.numind()), "visible index");

    let permuted = case.hom.permute(&case.p_codomain, &case.p_domain).unwrap();
    let permuted_codomain = case
        .p_codomain
        .iter()
        .map(|index| expected_visible[*index].clone())
        .collect::<Vec<_>>();
    let permuted_domain = case
        .p_domain
        .iter()
        .map(|index| expected_visible[*index].dual())
        .collect::<Vec<_>>();
    let permuted_visible = case
        .p_codomain
        .iter()
        .chain(&case.p_domain)
        .map(|index| expected_visible[*index].clone())
        .collect::<Vec<_>>();
    assert_eq!(permuted.codomain().factors(), permuted_codomain.as_slice());
    assert_eq!(permuted.domain().factors(), permuted_domain.as_slice());
    assert_eq!(permuted.visible_legs(), permuted_visible);

    let duplicate_codomain = vec![0];
    let duplicate_domain = (0..case.hom.numind() - 1).collect::<Vec<_>>();
    assert_err_contains(
        case.hom.permute(&duplicate_codomain, &duplicate_domain),
        "transform permutation must contain each visible index exactly once",
    );

    let out_of_range_codomain = vec![0];
    let mut out_of_range_domain = (1..case.hom.numind()).collect::<Vec<_>>();
    out_of_range_domain[0] = case.hom.numind();
    assert_err_contains(
        case.hom
            .permute(&out_of_range_codomain, &out_of_range_domain),
        "transform permutation must contain each visible index exactly once",
    );

    assert_hom_space_spec_roundtrip(&case.hom);
}

#[test]
fn graded_spaces_satisfy_core_contracts() {
    assert_graded_space_contract(&u1_space_case());
    assert_graded_space_contract(&su2_space_case());
    assert_graded_space_contract(&fermion_parity_space_case());
    assert_graded_space_contract(&z4_space_case());
    assert_graded_space_contract(&fermion_number_space_case());
}

#[test]
fn product_spaces_satisfy_core_contracts() {
    assert_product_space_contract(&u1_space_case());
    assert_product_space_contract(&su2_space_case());
    assert_product_space_contract(&fermion_parity_space_case());
    assert_product_space_contract(&z4_space_case());
    assert_product_space_contract(&fermion_number_space_case());
}

#[test]
fn fused_product_spaces_satisfy_core_contracts() {
    let u1 = u1_space_case();
    assert_fuse_two_factor_contract(&u1.primary, &u1.secondary);
    assert_fuse_two_factor_contract(&u1.primary, &u1.dual);

    let su2 = su2_space_case();
    assert_fuse_two_factor_contract(&su2.primary, &su2.secondary);
    assert_fuse_two_factor_contract(&su2.primary, &su2.dual);

    let fermion_parity = fermion_parity_space_case();
    assert_fuse_two_factor_contract(&fermion_parity.primary, &fermion_parity.secondary);
    assert_fuse_two_factor_contract(&fermion_parity.primary, &fermion_parity.dual);

    let z4 = z4_space_case();
    assert_fuse_two_factor_contract(&z4.primary, &z4.secondary);
    assert_fuse_two_factor_contract(&z4.primary, &z4.dual);

    let empty = ps::<U1Irrep>(vec![]);
    let fused_empty = fuse_product_space(&empty).unwrap();
    assert_space_sectors(&fused_empty, vec![(U1Irrep::unit(), 1)]);
    assert_eq!(fused_empty.dim(), empty.dim());
}

#[test]
fn hom_spaces_satisfy_core_contracts() {
    assert_hom_space_contract(&u1_tensorkit_shaped_hom_case());
}

#[test]
fn graded_space_convenience_apis_match_core_semantics() {
    let empty = GradedSpace::<U1Irrep>::zero(false);
    assert!(empty.sectors().is_empty());
    assert!(!empty.is_dual());
    assert_eq!(empty.dim(), 0);
    assert_eq!(empty.reduced_dim(), 0);
    assert!(!empty.has_sector(&u1(0)));

    let dual_empty = GradedSpace::<U1Irrep>::zero(true);
    assert!(dual_empty.sectors().is_empty());
    assert!(dual_empty.is_dual());

    let unit = GradedSpace::<SU2Irrep>::unit();
    assert_space_sectors(&unit, vec![(su2(0), 1)]);
    assert!(unit.has_sector(&su2(0)));
    assert_eq!(unit.reduced_dim(), 1);
    assert_eq!(unit.dim(), 1);

    let v = gs(vec![(su2(1), 3), (su2(2), 5)]);
    assert!(v.has_sector(&su2(1)));
    assert!(!v.has_sector(&su2(0)));
    assert_eq!(v.reduced_dim(), 8);
    assert_eq!(v.dim(), 21);
}

#[test]
fn graded_space_unit_predicate_requires_exact_canonical_unit_metadata() {
    let unit = GradedSpace::<U1Irrep>::unit();
    assert!(unit.is_unit());
    assert!(unit.dual().is_unit());
    assert!(gs(vec![(u1(0), 1), (u1(1), 0)]).is_unit());

    assert!(!GradedSpace::<U1Irrep>::zero(false).is_unit());
    assert!(!gs(vec![(u1(1), 1)]).is_unit());
    assert!(!gs(vec![(u1(0), 2)]).is_unit());
    assert!(!gs(vec![(u1(0), 1), (u1(1), 1)]).is_unit());
}

#[test]
fn product_space_convenience_apis_match_factor_storage() {
    let empty = ProductSpace::<U1Irrep>::one();
    assert!(empty.is_empty());
    assert_eq!(empty.len(), 0);
    assert_eq!(empty.get(0), None);
    assert_eq!(empty.iter().count(), 0);

    let left = gs(vec![(u1(0), 2)]);
    let right = gs(vec![(u1(1), 3)]);
    let product = ProductSpace::new(vec![left.clone(), right.clone()]);

    assert!(!product.is_empty());
    assert_eq!(product.len(), 2);
    assert_eq!(product.get(0), Some(&left));
    assert_eq!(product.get(1), Some(&right));
    assert_eq!(product.get(2), None);
    assert_eq!(
        product.iter().cloned().collect::<Vec<_>>(),
        vec![left, right],
    );
}

#[test]
fn product_space_unit_insertion_and_removal_use_zero_based_positions() {
    let left = gs(vec![(u1(1), 2)]);
    let right = gs(vec![(u1(-1), 3)]);
    let product = ProductSpace::new(vec![left.clone(), right.clone()]);

    for position in 0..=product.len() {
        for dual in [false, true] {
            let inserted = product.insert_unit(position, dual).unwrap();
            assert_eq!(inserted.len(), product.len() + 1);
            assert_eq!(inserted.get(position).unwrap().is_dual(), dual);
            assert!(inserted.get(position).unwrap().is_unit());
            assert_eq!(inserted.remove_unit(position).unwrap(), product);
        }
    }

    assert_err_contains(product.insert_unit(3, false), "insertion boundary");
    assert_err_contains(product.remove_unit(2), "removal index");
    assert_err_contains(product.remove_unit(0), "canonical unit space");
}

#[test]
fn hom_space_unit_insertion_obeys_left_right_partition_boundaries() {
    let out0 = gs(vec![(u1(1), 2)]);
    let out1 = gs(vec![(u1(-1), 3)]);
    let input = gs(vec![(u1(0), 5)]);
    let hom = HomSpace::from_factor_spaces(vec![out0, out1], vec![input]);

    for position in 0..=hom.numind() {
        let left = hom.insert_left_unit(position, false).unwrap();
        let right = hom.insert_right_unit(position, false).unwrap();

        assert_eq!(left.numind(), hom.numind() + 1);
        assert_eq!(right.numind(), hom.numind() + 1);
        assert_eq!(
            left.numout(),
            hom.numout() + usize::from(position < hom.numout())
        );
        assert_eq!(
            right.numout(),
            hom.numout() + usize::from(position <= hom.numout()),
        );
        assert!(left.visible_leg(position).unwrap().is_unit());
        assert!(right.visible_leg(position).unwrap().is_unit());
        assert_eq!(left.remove_unit(position).unwrap(), hom);
        assert_eq!(right.remove_unit(position).unwrap(), hom);
    }

    assert_err_contains(hom.insert_left_unit(4, false), "insertion boundary");
    assert_err_contains(hom.insert_right_unit(4, false), "insertion boundary");
    assert_err_contains(hom.remove_unit(3), "removal index");
    assert_err_contains(hom.remove_unit(0), "canonical unit space");
}

#[test]
fn rank_zero_unit_insertion_distinguishes_left_and_right_partition() {
    let scalar = HomSpace::<U1Irrep>::new(ProductSpace::one(), ProductSpace::one());

    let left = scalar.insert_left_unit(0, false).unwrap();
    assert_eq!((left.numout(), left.numin()), (0, 1));
    assert_eq!(left.remove_unit(0).unwrap(), scalar);

    let right = scalar.insert_right_unit(0, false).unwrap();
    assert_eq!((right.numout(), right.numin()), (1, 0));
    assert_eq!(right.remove_unit(0).unwrap(), scalar);
}

#[test]
fn hom_space_unit_dual_flag_describes_the_attached_side_factor() {
    let out = gs(vec![(u1(1), 2)]);
    let input = gs(vec![(u1(-1), 3)]);
    let hom = HomSpace::from_factor_spaces(vec![out], vec![input]);

    for dual in [false, true] {
        let codomain = hom.insert_right_unit(hom.numout(), dual).unwrap();
        assert_eq!(
            codomain.codomain().get(hom.numout()).unwrap().is_dual(),
            dual
        );
        assert_eq!(codomain.visible_leg(hom.numout()).unwrap().is_dual(), dual);

        let domain = hom.insert_left_unit(hom.numout(), dual).unwrap();
        assert_eq!(domain.domain().get(0).unwrap().is_dual(), dual);
        assert_eq!(domain.visible_leg(hom.numout()).unwrap().is_dual(), !dual);
    }
}

#[test]
fn graded_space_direct_sum_and_supremum_use_visible_sectors() {
    let left = gs(vec![(u1(0), 2), (u1(1), 3)]);
    let right = gs(vec![(u1(1), 5), (u1(-1), 7)]);

    let sum = left.direct_sum(&right).unwrap();
    assert_space_sectors(&sum, vec![(u1(0), 2), (u1(1), 8), (u1(-1), 7)]);

    let supremum = supremum_space(&left, &right).unwrap();
    assert_space_sectors(&supremum, vec![(u1(0), 2), (u1(1), 5), (u1(-1), 7)]);

    let dual_sum = left.dual().direct_sum(&right.dual()).unwrap();
    assert!(dual_sum.is_dual());
    assert_space_sectors(&dual_sum, vec![(u1(0), 2), (u1(-1), 8), (u1(1), 7)]);

    assert_err_contains(left.direct_sum(&right.dual()), "same dual flag");
    assert_err_contains(supremum_space(&left, &right.dual()), "same dual flag");
}

#[test]
fn graded_space_partial_order_uses_visible_sector_dimensions() {
    let small = gs(vec![(u1(0), 1), (u1(1), 2)]);
    let large = gs(vec![(u1(0), 3), (u1(1), 2), (u1(-1), 4)]);

    assert!(small.is_isomorphic(&small));
    assert!(!small.is_isomorphic(&large));
    assert!(small.is_monomorphic(&large));
    assert!(!large.is_monomorphic(&small));
    assert!(large.is_epimorphic(&small));
    assert!(!small.is_epimorphic(&large));

    let dual = gs(vec![(u1(1), 2), (u1(-1), 3)]).dual();
    let same_visible = gs(vec![(u1(-1), 2), (u1(1), 3)]);
    assert!(dual.is_isomorphic(&same_visible));
}

#[test]
fn product_space_dual_and_fuse_methods_match_core_semantics() {
    let left = gs(vec![(u1(1), 2)]);
    let right = gs(vec![(u1(-1), 3)]).dual();
    let product = ProductSpace::new(vec![left.clone(), right.clone()]);

    let dual = product.dual();
    let expected_dual_factors = vec![right.dual(), left.dual()];
    assert_eq!(dual.factors(), expected_dual_factors.as_slice());

    assert_eq!(
        product.fuse().unwrap(),
        fuse_product_space(&product).unwrap()
    );
}

#[test]
fn hom_space_factor_constructor_and_dual_match_arrow_semantics() {
    let v = gs(vec![(u1(0), 2)]);
    let w = gs(vec![(u1(1), 3)]);
    let x = gs(vec![(u1(-1), 5)]);

    let hom = HomSpace::from_factor_spaces(vec![v.clone(), w.clone()], vec![x.clone()]);
    assert_eq!(hom.codomain().factors(), &[v.clone(), w.clone()]);
    assert_eq!(hom.domain().factors(), std::slice::from_ref(&x));

    let dual = hom.dual();
    assert_eq!(dual.codomain().factors(), &[x]);
    assert_eq!(dual.domain().factors(), &[v, w]);
    assert_eq!(dual.dual(), hom);
}

#[test]
fn graded_space_skips_zero_dimension_sectors() {
    let empty = gs::<U1Irrep>(vec![(u1(0), 0)]);
    assert!(empty.sectors().is_empty());

    let v = gs(vec![(u1(1), 0), (u1(1), 5), (u1(-1), 0)]);
    assert_space_sectors(&v, vec![(u1(1), 5)]);
    assert_eq!(v.to_spec().sectors, vec![sd(&[1], 5)]);
}

#[test]
fn graded_space_rejects_duplicate_sectors() {
    assert_err_contains(
        GradedSpace::<U1Irrep>::new(vec![(u1(-1), 4), (u1(0), 5), (u1(-1), 1)], false),
        "sector appears multiple times",
    );
}

#[test]
fn graded_space_canonicalizes_sector_order_in_constructor_and_spec() {
    let v = gs(vec![(u1(2), 3), (u1(-1), 4), (u1(0), 5), (u1(1), 6)]);

    assert_space_sectors(&v, vec![(u1(0), 5), (u1(1), 6), (u1(-1), 4), (u1(2), 3)]);

    let spec = v.to_spec();
    assert_eq!(spec.sector_spec, SectorSpec::u1());
    assert_eq!(
        spec.sectors,
        vec![sd(&[0], 5), sd(&[1], 6), sd(&[-1], 4), sd(&[2], 3)]
    );

    let from_spec = GradedSpace::<U1Irrep>::from_spec(ElementarySpaceSpec {
        sector_spec: SectorSpec::u1(),
        sectors: vec![sd(&[1], 6), sd(&[0], 5), sd(&[-1], 4), sd(&[2], 3)],
        is_dual: false,
    })
    .unwrap();
    assert_eq!(from_spec, v);
}

#[test]
fn graded_space_from_spec_rejects_sector_family_mismatch() {
    let bad = ElementarySpaceSpec {
        sector_spec: SectorSpec::fermion_parity(),
        sectors: vec![sd(&[1], 1)],
        is_dual: false,
    };
    assert!(matches!(
        GradedSpace::<U1Irrep>::from_spec(bad).unwrap_err(),
        Tensor0Error::SectorSpecMismatch { .. },
    ));
}

#[test]
fn product_sector_graded_space_uses_product_sector_ordering() {
    let v = gs(vec![
        (fermion_number(1, 1), 2),
        (fermion_number(0, 1), 3),
        (fermion_number(0, 0), 4),
        (fermion_number(-1, 0), 5),
    ]);

    assert_eq!(
        v.sectors()
            .iter()
            .map(|(q, d)| (q.encode_value(), *d))
            .collect::<Vec<_>>(),
        vec![
            (ev(&[0, 0]), 4),
            (ev(&[0, 1]), 3),
            (ev(&[1, 1]), 2),
            (ev(&[-1, 0]), 5)
        ],
    );
}

#[test]
fn product_space_returns_zero_for_missing_sector_with_correct_arity() {
    let product = ps(vec![gs(vec![(u1(1), 3)]), gs(vec![(u1(-1), 5)])]);

    assert_eq!(product.sector_dims(&[u1(2), u1(-1)]), Some(vec![0, 5]));
    assert_eq!(product.sector_dim(&[u1(2), u1(-1)]), Some(0));
    assert_eq!(product.sector_dim(&[u1(1), u1(0)]), Some(0));
}

#[test]
fn product_space_with_zero_factor_has_zero_dimension_and_empty_fusion() {
    let zero = GradedSpace::<U1Irrep>::zero(false);
    let nonzero = gs(vec![(u1(1), 3)]);
    let product = ps(vec![zero, nonzero]);

    assert_eq!(product.dim(), 0);
    assert_eq!(product.dims(), vec![0, 3]);
    assert_eq!(product.sector_dim(&[u1(0), u1(1)]), Some(0));

    let fused = fuse_product_space(&product).unwrap();
    assert!(fused.sectors().is_empty());
    assert_eq!(fused.dim(), 0);
}

#[test]
fn product_space_spec_carries_sector_family_and_rejects_mismatches() {
    let product = ps::<U1Irrep>(vec![]);
    let spec = product.to_spec();

    assert_eq!(spec.sector_spec, SectorSpec::u1());

    let bad = ProductSpaceSpec {
        sector_spec: SectorSpec::su2(),
        factors: spec.factors,
    };
    assert!(matches!(
        ProductSpace::<U1Irrep>::from_spec(bad).unwrap_err(),
        Tensor0Error::SectorSpecMismatch { .. },
    ));

    let product = ps(vec![gs(vec![(u1(0), 2)])]);
    let mut bad = product.to_spec();
    bad.factors[0].sector_spec = SectorSpec::su2();
    assert!(matches!(
        ProductSpace::<U1Irrep>::from_spec(bad).unwrap_err(),
        Tensor0Error::SectorSpecMismatch { .. },
    ));
}

#[test]
fn fuse_product_space_counts_multiple_su2_fusion_paths_from_left_fold() {
    let half = gs(vec![(su2(1), 1)]);
    let product = ps(vec![half.clone(), half.clone(), half]);

    let fused = fuse_product_space(&product).unwrap();

    assert_space_sectors(&fused, vec![(su2(1), 2), (su2(3), 1)]);
}

#[test]
fn fuse_product_space_single_dual_factor_uses_visible_dual_labels() {
    let space = gs(vec![(u1(1), 5), (u1(-1), 2)]).dual();
    let product = ps(vec![space]);

    let fused = fuse_product_space(&product).unwrap();

    assert_space_sectors(&fused, vec![(u1(1), 2), (u1(-1), 5)]);
    assert!(!fused.is_dual());
}

#[test]
fn infimum_space_intersects_sector_dimensions() {
    let left = gs(vec![(u1(0), 10), (u1(1), 3)]);
    let right = gs(vec![(u1(0), 4), (u1(-1), 9)]);

    let common = infimum_space(&left, &right).unwrap();

    assert_space_sectors(&common, vec![(u1(0), 4)]);
    assert!(!common.is_dual());
}

#[test]
fn infimum_space_preserves_matching_dual_flag() {
    let left = gs(vec![(u1(0), 10), (u1(1), 3)]).dual();
    let right = gs(vec![(u1(0), 4), (u1(-1), 9)]).dual();

    let common = infimum_space(&left, &right).unwrap();

    assert_space_sectors(&common, vec![(u1(0), 4)]);
    assert!(common.is_dual());
}

#[test]
fn infimum_space_rejects_mismatched_dual_flags() {
    let left = gs(vec![(u1(0), 10)]);
    let right = left.dual();

    assert_err_contains(infimum_space(&left, &right), "same dual flag");
}

#[test]
fn graded_space_flip_preserves_visible_sectors_and_differs_from_dual() {
    let original = gs(vec![(u1(-2), 3), (u1(1), 5)]);
    let flipped = original.flip();

    assert_eq!(flipped.sectors(), original.sectors());
    assert!(flipped.is_dual());
    assert_ne!(flipped, original.dual());
    assert_eq!(flipped.flip(), original);
}

#[test]
fn graded_space_flip_is_exact_for_representative_sector_families() {
    fn assert_roundtrip<I: Sector + Debug>(space: GradedSpace<I>) {
        let visible = space.sectors().into_iter().collect::<BTreeMap<_, _>>();
        let flipped = space.flip();
        assert_eq!(
            flipped.sectors().into_iter().collect::<BTreeMap<_, _>>(),
            visible,
        );
        assert_eq!(flipped.is_dual(), !space.is_dual());
        assert_eq!(flipped.flip(), space);
    }

    assert_roundtrip(gs(vec![(su2(1), 2), (su2(2), 3)]));
    assert_roundtrip(gs(vec![(fp(0), 2), (fp(1), 3)]));
    assert_roundtrip(gs(vec![(z4(1), 2), (z4(3), 3)]));
    assert_roundtrip(gs(vec![
        (fermion_number(1, 1), 2),
        (fermion_number(-2, 0), 3),
    ]));
}

#[test]
fn hom_space_flip_keeps_partition_and_validates_all_indices() {
    let out = gs(vec![(u1(1), 2)]);
    let input = gs(vec![(u1(-1), 3)]).dual();
    let hom = HomSpace::from_factor_spaces(vec![out.clone()], vec![input.clone()]);

    let flipped = hom.flip(&[0, 1]).unwrap();
    assert_eq!(flipped.codomain().factors(), &[out.flip()]);
    assert_eq!(flipped.domain().factors(), &[input.flip()]);
    assert_eq!(
        flipped.visible_leg(0).unwrap(),
        hom.visible_leg(0).unwrap().flip()
    );
    assert_eq!(
        flipped.visible_leg(1).unwrap(),
        hom.visible_leg(1).unwrap().flip()
    );
    assert_eq!(flipped.flip(&[0, 1]).unwrap(), hom);

    assert_err_contains(hom.flip(&[0, 0]), "unique");
    assert_err_contains(hom.flip(&[2]), "out of range");
}
