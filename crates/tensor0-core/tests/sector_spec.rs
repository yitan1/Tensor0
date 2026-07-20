use tensor0_core::error::Tensor0Error;
use tensor0_core::sector::{EncodedSectorValue, Sector, SectorSpec, Trivial, ZNIrrep};

fn ev(xs: &[i64]) -> EncodedSectorValue {
    xs.iter().copied().collect()
}

fn z4_spec() -> SectorSpec {
    SectorSpec::zn(4).unwrap()
}

fn u1_su2_spec() -> SectorSpec {
    SectorSpec::product(vec![SectorSpec::u1(), SectorSpec::su2()]).unwrap()
}

fn su2_u1_spec() -> SectorSpec {
    SectorSpec::product(vec![SectorSpec::su2(), SectorSpec::u1()]).unwrap()
}

fn parity_su2_spec() -> SectorSpec {
    SectorSpec::product(vec![SectorSpec::fermion_parity(), SectorSpec::su2()]).unwrap()
}

fn parity_u1_su2_spec() -> SectorSpec {
    SectorSpec::product(vec![
        SectorSpec::fermion_parity(),
        SectorSpec::u1(),
        SectorSpec::su2(),
    ])
    .unwrap()
}

fn assert_quantum_dim(spec: SectorSpec, value: &[i64], expected: usize) {
    assert_eq!(spec.quantum_dim(value).unwrap(), expected);
}

fn assert_canonical_value(spec: SectorSpec, value: &[i64], expected: &[i64]) {
    assert_eq!(spec.canonicalize_value(value).unwrap(), ev(expected));
}

fn assert_quantum_dim_bad_width(spec: SectorSpec, value: &[i64], expected: usize, actual: usize) {
    assert!(matches!(
        spec.quantum_dim(value).unwrap_err(),
        Tensor0Error::BadSectorWidth { expected: e, actual: a } if e == expected && a == actual
    ));
}

#[test]
fn sector_spec_decodes_encoded_values_for_quantum_dim_and_canonicalization() {
    let u1_su2 = u1_su2_spec();

    assert_quantum_dim(SectorSpec::trivial(), &[], 1);
    assert_canonical_value(SectorSpec::trivial(), &[], &[]);
    assert_quantum_dim(SectorSpec::u1(), &[7], 1);
    assert_quantum_dim(SectorSpec::fermion_parity(), &[1], 1);
    assert_quantum_dim(z4_spec(), &[3], 1);
    assert_quantum_dim(SectorSpec::su2(), &[0], 1);
    assert_quantum_dim(SectorSpec::su2(), &[1], 2);
    assert_quantum_dim(SectorSpec::su2(), &[2], 3);
    assert_quantum_dim(u1_su2.clone(), &[4, 2], 3);
    assert_quantum_dim(su2_u1_spec(), &[2, 4], 3);
    assert_quantum_dim(parity_su2_spec(), &[1, 1], 2);
    assert_quantum_dim(parity_u1_su2_spec(), &[1, 4, 2], 3);

    assert_canonical_value(SectorSpec::fermion_parity(), &[-1], &[1]);
    assert_canonical_value(z4_spec(), &[-1], &[3]);
    assert_canonical_value(u1_su2, &[4, 2], &[4, 2]);
}

#[test]
fn sector_spec_rejects_invalid_encoded_values() {
    assert!(SectorSpec::su2()
        .quantum_dim(&[-1])
        .unwrap_err()
        .to_string()
        .contains("SU2 spin"));
    assert_quantum_dim_bad_width(SectorSpec::su2(), &[1, 2], 1, 2);
}

#[test]
fn sector_spec_rejects_bad_encoded_value_widths() {
    let product = u1_su2_spec();

    assert_quantum_dim_bad_width(SectorSpec::trivial(), &[0], 0, 1);
    assert_quantum_dim_bad_width(z4_spec(), &[], 1, 0);
    assert_quantum_dim_bad_width(z4_spec(), &[1, 2], 1, 2);
    assert_quantum_dim_bad_width(product.clone(), &[4], 1, 0);
    assert_quantum_dim_bad_width(product.clone(), &[4, 2, 9], 2, 3);
    assert!(matches!(
        product.canonicalize_value(&[4]).unwrap_err(),
        Tensor0Error::BadSectorWidth {
            expected: 1,
            actual: 0
        },
    ));
}

#[test]
fn sector_spec_canonicalizes_products_and_rejects_bad_specs() {
    let nested = SectorSpec::product(vec![
        SectorSpec::u1(),
        SectorSpec::product(vec![SectorSpec::fermion_parity(), z4_spec()]).unwrap(),
    ])
    .unwrap();
    assert_eq!(
        nested,
        SectorSpec::product(vec![
            SectorSpec::u1(),
            SectorSpec::fermion_parity(),
            z4_spec(),
        ])
        .unwrap(),
    );

    let raw_nested = SectorSpec::Product {
        components: vec![
            SectorSpec::u1(),
            SectorSpec::Product {
                components: vec![SectorSpec::fermion_parity(), z4_spec()],
            },
        ],
    };
    assert_eq!(raw_nested.canonicalize().unwrap(), nested);

    assert!(matches!(
        SectorSpec::zn(0).unwrap_err(),
        Tensor0Error::BadZnModulus { n: 0 },
    ));
    assert!(matches!(
        SectorSpec::product(vec![SectorSpec::u1()]).unwrap_err(),
        Tensor0Error::BadProductSectorArity { actual: 1 },
    ));
    assert!(matches!(
        (SectorSpec::Product {
            components: vec![SectorSpec::u1()],
        })
        .canonicalize()
        .unwrap_err(),
        Tensor0Error::BadProductSectorArity { actual: 1 },
    ));
}

#[test]
fn sector_spec_serde_uses_python_jax_metadata_shape() {
    let product = SectorSpec::product(vec![
        SectorSpec::u1(),
        SectorSpec::fermion_parity(),
        z4_spec(),
    ])
    .unwrap();

    assert_eq!(
        serde_json::to_value(&product).unwrap(),
        serde_json::json!({
            "kind": "product",
            "components": [
                {
                    "kind": "irrep",
                    "group": { "kind": "u1" }
                },
                {
                    "kind": "fermion_parity"
                },
                {
                    "kind": "irrep",
                    "group": { "kind": "zn", "n": 4 }
                }
            ]
        }),
    );
    assert_eq!(
        serde_json::to_value(SectorSpec::trivial()).unwrap(),
        serde_json::json!({ "kind": "trivial" }),
    );
    assert_eq!(
        serde_json::to_value(SectorSpec::su2()).unwrap(),
        serde_json::json!({
            "kind": "irrep",
            "group": { "kind": "su2" }
        }),
    );
}

#[test]
fn trivial_spec_roundtrips_and_differs_from_zn_one_metadata() {
    let trivial = SectorSpec::trivial();
    let zn_one = SectorSpec::zn(1).unwrap();
    let encoded = serde_json::to_string(&trivial).unwrap();
    let decoded: SectorSpec = serde_json::from_str(&encoded).unwrap();

    assert_eq!(decoded, trivial);
    assert_ne!(trivial, zn_one);
    assert_eq!(Trivial::encoded_width(), 0);
    assert_eq!(ZNIrrep::<1>::encoded_width(), 1);
    assert_eq!(trivial.canonicalize_value(&[]).unwrap(), ev(&[]));
    assert_eq!(zn_one.canonicalize_value(&[0]).unwrap(), ev(&[0]));
}

#[cfg(target_pointer_width = "64")]
#[test]
#[should_panic(expected = "ZNIrrep sector spec requires N <= i64::MAX")]
fn oversized_zn_const_generic_does_not_emit_fake_metadata() {
    let _ = ZNIrrep::<{ usize::MAX }>::sector_spec();
}

#[test]
#[should_panic(expected = "ZNIrrep sector spec requires N > 0")]
fn zero_zn_const_generic_does_not_emit_invalid_metadata() {
    let _ = ZNIrrep::<0>::sector_spec();
}
