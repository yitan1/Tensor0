use tensor0_core::error::Tensor0Error;
use tensor0_core::sector::{
    BraidingStyle, EncodedSectorValue, FermionNumber, FermionParity, FermionParitySU2Irrep,
    FermionParityU1Irrep, FermionParityU1SU2Irrep, FusionStyle, SU2Irrep, Sector,
    SectorCardinality, SectorSpec, Trivial, U1Irrep, U1SU2Irrep, Z4Irrep, ZNIrrep,
};

fn ev(xs: &[i64]) -> EncodedSectorValue {
    xs.iter().copied().collect()
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

fn fermion_number(charge2: i64, parity: i64) -> FermionNumber {
    FermionNumber::new((u1(charge2), fp(parity)))
}

fn u1_su2(charge2: i64, spin2: i64) -> U1SU2Irrep {
    U1SU2Irrep::decode_value(&[charge2, spin2]).unwrap()
}

fn fp_su2(parity: i64, spin2: i64) -> FermionParitySU2Irrep {
    FermionParitySU2Irrep::decode_value(&[parity, spin2]).unwrap()
}

fn fp_u1_su2(parity: i64, charge2: i64, spin2: i64) -> FermionParityU1SU2Irrep {
    FermionParityU1SU2Irrep::decode_value(&[parity, charge2, spin2]).unwrap()
}

fn z4(value: i64) -> Z4Irrep {
    Z4Irrep::new(value).unwrap()
}

struct SectorCase<I: Sector> {
    samples: Vec<I>,
    fusion_cases: Vec<(I, I)>,
    first_values: Vec<I>,
}

fn u1_case() -> SectorCase<U1Irrep> {
    SectorCase {
        samples: vec![u1(-3), u1(-1), u1(0), u1(1), u1(2)],
        fusion_cases: vec![(u1(1), u1(-3))],
        first_values: vec![u1(0), u1(1), u1(-1), u1(2), u1(-2)],
    }
}

fn trivial_case() -> SectorCase<Trivial> {
    SectorCase {
        samples: vec![Trivial],
        fusion_cases: vec![(Trivial, Trivial)],
        first_values: vec![Trivial],
    }
}

fn su2_case() -> SectorCase<SU2Irrep> {
    SectorCase {
        samples: vec![su2(0), su2(1), su2(2), su2(3), su2(4)],
        fusion_cases: vec![(su2(1), su2(1)), (su2(2), su2(1))],
        first_values: vec![su2(0), su2(1), su2(2), su2(3), su2(4)],
    }
}

fn fermion_parity_case() -> SectorCase<FermionParity> {
    SectorCase {
        samples: vec![fp(0), fp(1)],
        fusion_cases: vec![(fp(1), fp(1))],
        first_values: vec![fp(0), fp(1)],
    }
}

fn z4_case() -> SectorCase<Z4Irrep> {
    SectorCase {
        samples: vec![z4(0), z4(1), z4(2), z4(3)],
        fusion_cases: vec![(z4(3), z4(2))],
        first_values: vec![z4(0), z4(1), z4(2), z4(3)],
    }
}

fn fermion_number_case() -> SectorCase<FermionNumber> {
    SectorCase {
        samples: vec![
            fermion_number(0, 0),
            fermion_number(0, 1),
            fermion_number(1, 1),
            fermion_number(2, 0),
            fermion_number(-1, 0),
        ],
        fusion_cases: vec![(fermion_number(1, 1), fermion_number(1, 1))],
        first_values: vec![
            fermion_number(0, 0),
            fermion_number(0, 1),
            fermion_number(1, 0),
            fermion_number(1, 1),
            fermion_number(-1, 0),
            fermion_number(-1, 1),
        ],
    }
}

fn u1_su2_case() -> SectorCase<U1SU2Irrep> {
    SectorCase {
        samples: vec![
            u1_su2(0, 0),
            u1_su2(1, 1),
            u1_su2(-1, 1),
            u1_su2(2, 0),
            u1_su2(2, 2),
        ],
        fusion_cases: vec![(u1_su2(1, 1), u1_su2(1, 1))],
        first_values: vec![
            u1_su2(0, 0),
            u1_su2(0, 1),
            u1_su2(1, 0),
            u1_su2(0, 2),
            u1_su2(1, 1),
            u1_su2(-1, 0),
        ],
    }
}

fn fp_su2_case() -> SectorCase<FermionParitySU2Irrep> {
    SectorCase {
        samples: vec![fp_su2(0, 0), fp_su2(1, 1), fp_su2(0, 2)],
        fusion_cases: vec![(fp_su2(1, 1), fp_su2(1, 1))],
        first_values: vec![
            fp_su2(0, 0),
            fp_su2(0, 1),
            fp_su2(1, 0),
            fp_su2(0, 2),
            fp_su2(1, 1),
        ],
    }
}

fn fp_u1_su2_case() -> SectorCase<FermionParityU1SU2Irrep> {
    SectorCase {
        samples: vec![
            fp_u1_su2(0, 0, 0),
            fp_u1_su2(1, 1, 1),
            fp_u1_su2(0, -1, 1),
            fp_u1_su2(1, 2, 2),
        ],
        fusion_cases: vec![(fp_u1_su2(1, 1, 1), fp_u1_su2(1, -1, 1))],
        first_values: vec![
            fp_u1_su2(0, 0, 0),
            fp_u1_su2(0, 0, 1),
            fp_u1_su2(0, 1, 0),
            fp_u1_su2(1, 0, 0),
            fp_u1_su2(0, 0, 2),
            fp_u1_su2(0, 1, 1),
        ],
    }
}

fn check(name: &str, ok: bool, label: &str) {
    assert!(ok, "{name}: {label}");
}

fn assert_metadata<I: Sector>(
    expected_spec: SectorSpec,
    expected_width: usize,
    expected_fusion: FusionStyle,
    expected_braiding: BraidingStyle,
    expected_cardinality: SectorCardinality,
) {
    let name = std::any::type_name::<I>();
    assert_eq!(I::sector_spec(), expected_spec, "{name} sector spec");
    assert_eq!(I::encoded_width(), expected_width, "{name} encoded width");
    assert_eq!(I::fusion_style(), expected_fusion, "{name} fusion style");
    assert_eq!(
        I::braiding_style(),
        expected_braiding,
        "{name} braiding style",
    );
    assert_eq!(
        I::cardinality().unwrap(),
        expected_cardinality,
        "{name} cardinality",
    );
}

#[test]
fn primitive_sectors_satisfy_core_contracts() {
    assert_core_sector_contracts(trivial_case());
    assert_core_sector_contracts(u1_case());
    assert_core_sector_contracts(su2_case());
    assert_core_sector_contracts(fermion_parity_case());
    assert_core_sector_contracts(z4_case());
}

#[test]
fn product_sectors_satisfy_core_contracts() {
    assert_core_sector_contracts(fermion_number_case());
    assert_core_sector_contracts(u1_su2_case());
    assert_core_sector_contracts(fp_su2_case());
    assert_core_sector_contracts(fp_u1_su2_case());
}

#[test]
fn primitive_sector_types_expose_expected_metadata() {
    assert_metadata::<Trivial>(
        SectorSpec::trivial(),
        0,
        FusionStyle::UniqueFusion,
        BraidingStyle::Bosonic,
        SectorCardinality::Finite(1),
    );
    assert_metadata::<U1Irrep>(
        SectorSpec::u1(),
        1,
        FusionStyle::UniqueFusion,
        BraidingStyle::Bosonic,
        SectorCardinality::Infinite,
    );
    assert_metadata::<SU2Irrep>(
        SectorSpec::su2(),
        1,
        FusionStyle::SimpleFusion,
        BraidingStyle::Bosonic,
        SectorCardinality::Infinite,
    );
    assert_metadata::<FermionParity>(
        SectorSpec::fermion_parity(),
        1,
        FusionStyle::UniqueFusion,
        BraidingStyle::Fermionic,
        SectorCardinality::Finite(2),
    );
    assert_metadata::<Z4Irrep>(
        SectorSpec::zn(4).unwrap(),
        1,
        FusionStyle::UniqueFusion,
        BraidingStyle::Bosonic,
        SectorCardinality::Finite(4),
    );
}

#[test]
fn su2_irrep_matches_representative_fusion_rules() {
    let half = su2(1);
    let one = su2(2);

    assert_eq!(
        half.fusion_outputs(&half).collect::<Vec<_>>(),
        vec![su2(0), su2(2)]
    );
    assert_eq!(
        one.fusion_outputs(&half).collect::<Vec<_>>(),
        vec![su2(1), su2(3)]
    );
}

#[test]
fn su2_irrep_rejects_negative_spin_and_bad_width() {
    assert!(matches!(
        SU2Irrep::spin2(-1).unwrap_err(),
        Tensor0Error::Message(message) if message.contains("SU2 spin must be non-negative")
    ));
    assert!(matches!(
        SU2Irrep::decode_value(&[-1]).unwrap_err(),
        Tensor0Error::Message(message) if message.contains("SU2 spin must be non-negative")
    ));
    assert!(matches!(
        SU2Irrep::decode_value(&[1, 2]).unwrap_err(),
        Tensor0Error::BadSectorWidth {
            expected: 1,
            actual: 2
        },
    ));
}

#[test]
fn primitive_sector_values_canonicalize_family_specific_inputs() {
    let odd = fp(1);
    let even = fp(0);
    assert_eq!(FermionParity::decode_value(&[-1]).unwrap(), odd);
    assert_eq!(odd.encode_value(), ev(&[1]));
    assert!(odd.is_odd());
    assert!(!even.is_odd());

    let a = z4(-1);
    let b = z4(3);
    assert_eq!(a, b);
    assert_eq!(a.encode_value(), ev(&[3]));
}

#[test]
fn fermion_number_exposes_fermion_parity() {
    let a = fermion_number(1, 1);

    assert!(a.fermion_parity());
}

#[test]
fn product_aliases_have_expected_metadata() {
    assert_metadata::<U1SU2Irrep>(
        SectorSpec::product(vec![SectorSpec::u1(), SectorSpec::su2()]).unwrap(),
        2,
        FusionStyle::SimpleFusion,
        BraidingStyle::Bosonic,
        SectorCardinality::Infinite,
    );
    assert_metadata::<FermionParitySU2Irrep>(
        SectorSpec::product(vec![SectorSpec::fermion_parity(), SectorSpec::su2()]).unwrap(),
        2,
        FusionStyle::SimpleFusion,
        BraidingStyle::Fermionic,
        SectorCardinality::Infinite,
    );
    assert_metadata::<FermionParityU1Irrep>(
        SectorSpec::product(vec![SectorSpec::fermion_parity(), SectorSpec::u1()]).unwrap(),
        2,
        FusionStyle::UniqueFusion,
        BraidingStyle::Fermionic,
        SectorCardinality::Infinite,
    );
    assert_metadata::<FermionParityU1SU2Irrep>(
        SectorSpec::product(vec![
            SectorSpec::fermion_parity(),
            SectorSpec::u1(),
            SectorSpec::su2(),
        ])
        .unwrap(),
        3,
        FusionStyle::SimpleFusion,
        BraidingStyle::Fermionic,
        SectorCardinality::Infinite,
    );
}

#[test]
fn invalid_decode_and_invalid_const_generic_return_clear_errors() {
    assert!(matches!(
        Trivial::decode_value(&[0]).unwrap_err(),
        Tensor0Error::BadSectorWidth {
            expected: 0,
            actual: 1
        },
    ));
    assert!(matches!(
        U1Irrep::decode_value(&[1, 2]).unwrap_err(),
        Tensor0Error::BadSectorWidth {
            expected: 1,
            actual: 2
        },
    ));
    assert!(matches!(
        FermionParity::decode_value(&[]).unwrap_err(),
        Tensor0Error::BadSectorWidth {
            expected: 1,
            actual: 0
        },
    ));
    assert!(matches!(
        FermionNumber::decode_value(&[1]).unwrap_err(),
        Tensor0Error::BadSectorWidth {
            expected: 2,
            actual: 1
        },
    ));
    assert!(matches!(
        ZNIrrep::<0>::cardinality().unwrap_err(),
        Tensor0Error::BadZnModulus { n: 0 },
    ));
}

fn assert_encode_roundtrip<I: Sector>(name: &str, sectors: &[I]) {
    for sector in sectors {
        let encoded = sector.encode_value();
        let decoded = I::decode_value(&encoded).unwrap();
        check(name, decoded == sector.clone(), "encode/decode");
    }
}

fn assert_dual_involution<I: Sector>(name: &str, sectors: &[I]) {
    check(name, I::unit().dual() == I::unit(), "unit dual");
    for sector in sectors {
        check(name, sector.dual().dual() == sector.clone(), "dual");
    }
}

fn assert_unit_fusion_identity<I: Sector>(name: &str, sectors: &[I]) {
    let unit = I::unit();
    for sector in sectors {
        check(
            name,
            unit.fusion_outputs(sector).collect::<Vec<_>>() == vec![sector.clone()],
            "left unit",
        );
        check(
            name,
            sector.fusion_outputs(&unit).collect::<Vec<_>>() == vec![sector.clone()],
            "right unit",
        );
    }
}

fn assert_fusion_contract<I: Sector>(name: &str, a: &I, b: &I, candidates: &[I]) {
    let outputs = a.fusion_outputs(b).collect::<Vec<_>>();
    let dimension_sum = outputs
        .iter()
        .map(|c| c.quantum_dim() * I::n_symbol(a, b, c))
        .sum::<usize>();
    check(
        name,
        a.quantum_dim() * b.quantum_dim() == dimension_sum,
        "fusion dim",
    );

    for output in &outputs {
        check(name, I::n_symbol(a, b, output) == 1, "output n");
    }
    for candidate in candidates {
        let expected_n = usize::from(outputs.contains(candidate));
        check(
            name,
            I::n_symbol(a, b, candidate) == expected_n,
            "candidate n",
        );
        if expected_n == 0 {
            check(name, I::r_symbol(a, b, candidate) == 0.0, "non-fusing R");
        }
    }
}

fn assert_symmetric_braiding_contract<I: Sector>(name: &str, a: &I, b: &I) {
    check(
        name,
        matches!(
            I::braiding_style(),
            BraidingStyle::Bosonic | BraidingStyle::Fermionic
        ),
        "symmetric braiding style",
    );
    for c in a.fusion_outputs(b) {
        let double_braid = I::r_symbol(a, b, &c) * I::r_symbol(b, a, &c);
        check(name, (double_braid - 1.0).abs() < 1.0e-12, "double braid");
    }
}

fn assert_value_at_contract<I: Sector>(name: &str, samples: &[I], first_values: &[I]) {
    for sector in samples {
        let index = sector.sort_index().unwrap();
        check(
            name,
            I::value_at(index).unwrap() == sector.clone(),
            "value_at(sort_index)",
        );
    }

    for (index, expected) in first_values.iter().enumerate() {
        let value = I::value_at(index as u128).unwrap();
        check(name, value == expected.clone(), "value_at");
        check(
            name,
            value.sort_index().unwrap() == index as u128,
            "sort_index",
        );
    }

    let iter_values = I::values()
        .unwrap()
        .take(first_values.len())
        .collect::<tensor0_core::error::Result<Vec<_>>>()
        .unwrap();
    check(
        name,
        iter_values.iter().eq(first_values.iter()),
        "values prefix",
    );

    let mut sorted_values = first_values.iter().rev().cloned().collect::<Vec<_>>();
    sorted_values.sort();
    assert!(sorted_values.iter().eq(first_values.iter()), "{name}: Ord");

    if let SectorCardinality::Finite(size) = I::cardinality().unwrap() {
        let values = I::values()
            .unwrap()
            .collect::<tensor0_core::error::Result<Vec<_>>>()
            .unwrap();
        check(name, values.len() as u128 == size, "finite cardinality");
        check(name, I::value_at(size).is_err(), "finite out of range");
    }
}

fn assert_core_sector_contracts<I: Sector>(case: SectorCase<I>) {
    let name = std::any::type_name::<I>();
    assert_encode_roundtrip(name, &case.samples);
    assert_dual_involution(name, &case.samples);
    assert_unit_fusion_identity(name, &case.samples);
    assert_value_at_contract(name, &case.samples, &case.first_values);
    for (a, b) in &case.fusion_cases {
        assert_fusion_contract(name, a, b, &case.samples);
        assert_symmetric_braiding_contract(name, a, b);
    }
}
