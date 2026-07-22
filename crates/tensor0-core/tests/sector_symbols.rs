use tensor0_core::sector::{
    FermionNumber, FermionParity, SU2Irrep, Sector, Trivial, U1Irrep, U1SU2Irrep, Z2Irrep, Z4Irrep,
};

fn su2(spin2: i64) -> SU2Irrep {
    SU2Irrep::spin2(spin2).unwrap()
}

fn su2_half_channels() -> (SU2Irrep, SU2Irrep, SU2Irrep) {
    (su2(1), su2(0), su2(2))
}

fn u1(charge2: i64) -> U1Irrep {
    U1Irrep::charge2(charge2).unwrap()
}

fn z2(value: i64) -> Z2Irrep {
    Z2Irrep::new(value).unwrap()
}

fn z4(value: i64) -> Z4Irrep {
    Z4Irrep::new(value).unwrap()
}

fn fp(value: i64) -> FermionParity {
    FermionParity::new(value).unwrap()
}

fn fermion_number(charge2: i64, parity: i64) -> FermionNumber {
    FermionNumber::new((u1(charge2), fp(parity)))
}

fn u1_su2(charge2: i64, spin2: i64) -> U1SU2Irrep {
    U1SU2Irrep::new((u1(charge2), su2(spin2)))
}

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1.0e-12,
        "actual={actual}, expected={expected}",
    );
}

fn assert_vec_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert_close(*actual, *expected);
    }
}

fn assert_tensor_shape(tensor: &ndarray::Array4<f64>, expected: &[usize]) {
    assert_eq!(tensor.shape(), expected);
}

fn assert_tensor_data(tensor: &ndarray::Array4<f64>, expected: &[f64]) {
    assert_vec_close(tensor.as_slice().unwrap(), expected);
}

fn assert_n<I: Sector>(a: I, b: I, c: I, expected: usize) {
    assert_eq!(I::n_symbol(&a, &b, &c), expected);
}

fn assert_f<I: Sector>(a: I, b: I, c: I, d: I, e: I, f: I, expected: f64) {
    assert_close(I::f_symbol(&a, &b, &c, &d, &e, &f).unwrap(), expected);
}

fn assert_r<I: Sector>(a: I, b: I, c: I, expected: f64) {
    assert_close(I::r_symbol(&a, &b, &c), expected);
}

fn assert_frobenius_schur_phase<I: Sector>(a: I, expected: f64) {
    assert_close(I::frobenius_schur_phase(&a).unwrap(), expected);
}

fn assert_fusion_tensor<I: Sector>(a: I, b: I, c: I, expected_shape: &[usize], expected: &[f64]) {
    let tensor = I::fusion_tensor(&a, &b, &c).unwrap();
    assert_tensor_shape(&tensor, expected_shape);
    assert_tensor_data(&tensor, expected);
}

fn su2_f_symbol(
    a: SU2Irrep,
    b: SU2Irrep,
    c: SU2Irrep,
    d: SU2Irrep,
    e: SU2Irrep,
    f: SU2Irrep,
) -> tensor0_core::error::Result<f64> {
    SU2Irrep::f_symbol(&a, &b, &c, &d, &e, &f)
}

fn su2_fusion_tensor(
    a: SU2Irrep,
    b: SU2Irrep,
    c: SU2Irrep,
) -> tensor0_core::error::Result<ndarray::Array4<f64>> {
    SU2Irrep::fusion_tensor(&a, &b, &c)
}

#[test]
fn abelian_and_fermion_symbols_match_tensorkit_conventions() {
    assert_n(Trivial, Trivial, Trivial, 1);
    assert_f(Trivial, Trivial, Trivial, Trivial, Trivial, Trivial, 1.0);
    assert_close(
        Trivial::a_symbol(&Trivial, &Trivial, &Trivial).unwrap(),
        1.0,
    );
    assert_close(
        Trivial::b_symbol(&Trivial, &Trivial, &Trivial).unwrap(),
        1.0,
    );
    assert_frobenius_schur_phase(Trivial, 1.0);
    assert_r(Trivial, Trivial, Trivial, 1.0);
    assert_close(Trivial.twist(), 1.0);
    assert_fusion_tensor(Trivial, Trivial, Trivial, &[1, 1, 1, 1], &[1.0]);

    let q1 = u1(1);
    let q2 = u1(2);
    let q3 = u1(3);
    let q4 = u1(4);
    let q6 = u1(6);
    let q7 = u1(7);

    assert_n(q1, q2, q3, 1);
    assert_n(q1, q2, q4, 0);
    assert_f(q1, q2, q4, q7, q3, q6, 1.0);
    assert_f(q1, q2, q4, q7, q4, q6, 0.0);
    assert_fusion_tensor(q1, q2, q3, &[1, 1, 1, 1], &[1.0]);
    assert_fusion_tensor(q1, q2, q4, &[1, 1, 1, 1], &[0.0]);

    let even = fp(0);
    let odd = fp(1);
    assert_f(odd, odd, odd, odd, even, even, 1.0);
    assert_r(odd, odd, even, -1.0);
    assert_fusion_tensor(odd, odd, even, &[1, 1, 1, 1], &[1.0]);

    assert_fusion_tensor(z2(1), z2(1), z2(0), &[1, 1, 1, 1], &[1.0]);
    assert_fusion_tensor(z2(1), z2(1), z2(1), &[1, 1, 1, 1], &[0.0]);

    let z4_left = z4(3);
    let z4_right = z4(2);
    let z4_coupled = z4(1);
    let z4_wrong = z4(0);
    assert_n(z4_left, z4_right, z4_coupled, 1);
    assert_n(z4_left, z4_right, z4_wrong, 0);
    assert_f(z4(3), z4(2), z4(3), z4(0), z4(1), z4(1), 1.0);
    assert_f(z4(3), z4(2), z4(3), z4(0), z4(0), z4(1), 0.0);
    assert_fusion_tensor(z4_left, z4_right, z4_coupled, &[1, 1, 1, 1], &[1.0]);
    assert_fusion_tensor(z4_left, z4_right, z4_wrong, &[1, 1, 1, 1], &[0.0]);
}

#[test]
fn product_sector_symbols_compose_componentwise() {
    let left = fermion_number(1, 1);
    let right = fermion_number(2, 1);
    let coupled = fermion_number(3, 0);

    assert_r(left.clone(), right.clone(), coupled.clone(), -1.0);
    assert_fusion_tensor(left, right, coupled, &[1, 1, 1, 1], &[1.0]);

    let half_charge_half_spin = u1_su2(1, 1);
    let negative_half_charge_half_spin = u1_su2(-1, 1);
    let scalar = u1_su2(0, 0);
    let wrong_charge = u1_su2(2, 0);
    let product_tensor = U1SU2Irrep::fusion_tensor(
        &half_charge_half_spin,
        &negative_half_charge_half_spin,
        &scalar,
    )
    .unwrap();
    let su2_tensor = su2_fusion_tensor(su2(1), su2(1), su2(0)).unwrap();

    assert_tensor_shape(&product_tensor, su2_tensor.shape());
    assert_tensor_data(&product_tensor, su2_tensor.as_slice().unwrap());
    assert_fusion_tensor(
        half_charge_half_spin,
        negative_half_charge_half_spin,
        wrong_charge,
        &[2, 2, 1, 1],
        &[0.0, 0.0, 0.0, 0.0],
    );
}

#[test]
fn representative_sector_symbols_satisfy_multiplicity_free_contracts() {
    assert_multiplicity_free_symbol_contracts(u1(1), u1(2), u1(-3), u1(1));
    assert_multiplicity_free_symbol_contracts(fp(1), fp(1), fp(1), fp(1));
    assert_multiplicity_free_symbol_contracts(z4(3), z4(2), z4(1), z4(3));
    assert_multiplicity_free_symbol_contracts(su2(1), su2(1), su2(1), su2(1));
    assert_multiplicity_free_symbol_contracts(
        fermion_number(1, 1),
        fermion_number(2, 1),
        fermion_number(-3, 0),
        fermion_number(1, 1),
    );
    assert_multiplicity_free_symbol_contracts(
        u1_su2(1, 1),
        u1_su2(-1, 1),
        u1_su2(1, 1),
        u1_su2(1, 1),
    );
}

#[test]
fn su2_half_spin_symbols_match_tensorkit_conventions() {
    let (h, singlet, triplet) = su2_half_channels();
    let sqrt3_over_2 = 3.0_f64.sqrt() / 2.0;

    assert_close(su2_f_symbol(h, h, h, h, singlet, singlet).unwrap(), -0.5);
    assert_close(
        su2_f_symbol(h, h, h, h, singlet, triplet).unwrap(),
        sqrt3_over_2,
    );
    assert_close(
        su2_f_symbol(h, h, h, h, triplet, singlet).unwrap(),
        sqrt3_over_2,
    );
    assert_close(su2_f_symbol(h, h, h, h, triplet, triplet).unwrap(), 0.5);
    assert_r(h, h, singlet, -1.0);
    assert_r(h, h, triplet, 1.0);
}

#[test]
fn frobenius_schur_phase_matches_tensorkit_conventions() {
    assert_frobenius_schur_phase(u1(1), 1.0);
    assert_frobenius_schur_phase(su2(1), -1.0);
    assert_frobenius_schur_phase(su2(2), 1.0);
}

#[test]
fn sector_twists_match_tensorkit_symmetric_conventions() {
    assert_close(u1(3).twist(), 1.0);
    assert_close(su2(1).twist(), 1.0);
    assert_close(fp(0).twist(), 1.0);
    assert_close(fp(1).twist(), -1.0);
    assert_close(fermion_number(0, 1).twist(), -1.0);
    assert_close(u1_su2(0, 1).twist(), 1.0);
}

#[test]
fn su2_f_symbol_matches_fusion_tensor_contraction() {
    let (h, singlet, triplet) = su2_half_channels();
    let three_half = su2(3);
    let cases = [
        (h, h, h, h, singlet, singlet),
        (h, h, h, h, singlet, triplet),
        (h, h, h, h, triplet, singlet),
        (h, h, h, h, triplet, triplet),
        (h, h, h, three_half, triplet, triplet),
    ];

    for (a, b, c, d, e, f) in cases {
        assert_close(
            su2_f_symbol(a, b, c, d, e, f).unwrap(),
            su2_f_symbol_from_fusion_tensor_contraction(a, b, c, d, e, f),
        );
    }
}

#[test]
fn su2_fusion_tensor_uses_row_major_tensorkit_magnetic_order() {
    let (h, singlet, _) = su2_half_channels();
    let tensor = su2_fusion_tensor(h, h, singlet).unwrap();
    let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();

    assert_tensor_shape(&tensor, &[2, 2, 1, 1]);
    assert_tensor_data(&tensor, &[0.0, inv_sqrt2, -inv_sqrt2, 0.0]);
}

#[test]
fn su2_fusion_tensors_are_orthonormal_for_representative_channels() {
    assert_fusion_tensors_orthonormal(su2(1), su2(1));
    assert_fusion_tensors_orthonormal(su2(2), su2(2));
}

#[test]
fn su2_fusion_tensor_rejects_invalid_channel() {
    let singlet = su2(0);
    let triplet = su2(2);

    let triangle_error = su2_fusion_tensor(singlet, singlet, triplet)
        .unwrap_err()
        .to_string();
    assert!(triangle_error.contains("triangle"));
}

fn assert_fusion_tensor_has_expected_shape<I: Sector>(a: &I, b: &I, c: &I) {
    let tensor = I::fusion_tensor(a, b, c).unwrap();
    assert_tensor_shape(
        &tensor,
        &[a.quantum_dim(), b.quantum_dim(), c.quantum_dim(), 1],
    );
}

fn assert_symmetric_braiding<I: Sector>(a: I, b: I) {
    for c in a.fusion_outputs(&b) {
        let rr = I::r_symbol(&a, &b, &c) * I::r_symbol(&b, &a, &c);
        assert_close(rr, 1.0);
    }
}

fn common_outputs<I: Sector>(
    left: impl IntoIterator<Item = I>,
    right: impl IntoIterator<Item = I>,
) -> Vec<I> {
    let right = right.into_iter().collect::<Vec<_>>();
    let mut outputs = left
        .into_iter()
        .filter(|candidate| right.contains(candidate))
        .collect::<Vec<_>>();
    outputs.sort();
    outputs.dedup();
    outputs
}

fn assert_a_b_symbols_follow_f_symbol_definitions<I: Sector>(a: &I, b: &I, c: &I) {
    let unit = I::unit();
    let a_dual = a.dual();
    let b_dual = b.dual();
    let scale =
        ((a.quantum_dim() as f64) * (b.quantum_dim() as f64) / (c.quantum_dim() as f64)).sqrt();

    let expected_a = scale
        * I::frobenius_schur_phase(a).unwrap()
        * I::f_symbol(&a_dual, a, b, b, &unit, c).unwrap();
    let expected_b = scale * I::f_symbol(a, b, &b_dual, a, c, &unit).unwrap();

    assert_close(I::a_symbol(a, b, c).unwrap(), expected_a);
    assert_close(I::b_symbol(a, b, c).unwrap(), expected_b);
}

fn assert_triangle_equation<I: Sector>(a: I, b: I) {
    let unit = I::unit();
    for c in a.fusion_outputs(&b) {
        assert_close(I::f_symbol(&unit, &a, &b, &c, &a, &c).unwrap(), 1.0);
        assert_close(I::f_symbol(&a, &unit, &b, &c, &a, &b).unwrap(), 1.0);
        assert_close(I::f_symbol(&a, &b, &unit, &c, &c, &b).unwrap(), 1.0);
    }
}

fn assert_f_move_unitary<I: Sector>(a: I, b: I, c: I) {
    let mut targets = Vec::new();
    for e in a.fusion_outputs(&b) {
        targets.extend(e.fusion_outputs(&c));
    }
    targets.sort();
    targets.dedup();

    for d in targets {
        let es = a
            .fusion_outputs(&b)
            .filter(|e| I::n_symbol(e, &c, &d) > 0)
            .collect::<Vec<_>>();
        let fs = b
            .fusion_outputs(&c)
            .filter(|f| I::n_symbol(&a, f, &d) > 0)
            .collect::<Vec<_>>();

        assert_eq!(es.len(), fs.len());
        for (i, e_left) in es.iter().enumerate() {
            for (j, e_right) in es.iter().enumerate() {
                let overlap = fs.iter().fold(0.0, |sum, f| {
                    sum + I::f_symbol(&a, &b, &c, &d, e_left, f).unwrap()
                        * I::f_symbol(&a, &b, &c, &d, e_right, f).unwrap()
                });
                assert_close(overlap, if i == j { 1.0 } else { 0.0 });
            }
        }
        for (i, f_left) in fs.iter().enumerate() {
            for (j, f_right) in fs.iter().enumerate() {
                let overlap = es.iter().fold(0.0, |sum, e| {
                    sum + I::f_symbol(&a, &b, &c, &d, e, f_left).unwrap()
                        * I::f_symbol(&a, &b, &c, &d, e, f_right).unwrap()
                });
                assert_close(overlap, if i == j { 1.0 } else { 0.0 });
            }
        }
    }
}

fn assert_pentagon_equation<I: Sector>(a: I, b: I, c: I, d: I) {
    for f in a.fusion_outputs(&b) {
        for h in c.fusion_outputs(&d) {
            for g in f.fusion_outputs(&c) {
                for i in b.fusion_outputs(&h) {
                    for e in common_outputs(g.fusion_outputs(&d), a.fusion_outputs(&i)) {
                        let p1 = I::f_symbol(&f, &c, &d, &e, &g, &h).unwrap()
                            * I::f_symbol(&a, &b, &h, &e, &f, &i).unwrap();
                        let p2 = b.fusion_outputs(&c).fold(0.0, |sum, j| {
                            sum + I::f_symbol(&a, &b, &c, &g, &f, &j).unwrap()
                                * I::f_symbol(&a, &j, &d, &e, &g, &i).unwrap()
                                * I::f_symbol(&b, &c, &d, &i, &j, &h).unwrap()
                        });
                        assert_close(p1, p2);
                    }
                }
            }
        }
    }
}

fn assert_hexagon_equation<I: Sector>(a: I, b: I, c: I) {
    for e in c.fusion_outputs(&a) {
        let rcae = I::r_symbol(&c, &a, &e);
        let race = I::r_symbol(&a, &c, &e);
        for f in c.fusion_outputs(&b) {
            let rcbf = I::r_symbol(&c, &b, &f);
            let rbcf = I::r_symbol(&b, &c, &f);
            for d in common_outputs(e.fusion_outputs(&b), a.fusion_outputs(&f)) {
                let facbdef = I::f_symbol(&a, &c, &b, &d, &e, &f).unwrap();
                let rfr1 = rcae * facbdef * rcbf;
                let rfr2 = race * facbdef * rbcf;

                let (frf1, frf2) = a.fusion_outputs(&b).fold((0.0, 0.0), |(frf1, frf2), g| {
                    let fcabdeg = I::f_symbol(&c, &a, &b, &d, &e, &g).unwrap();
                    let fabcdgf = I::f_symbol(&a, &b, &c, &d, &g, &f).unwrap();
                    (
                        frf1 + fcabdeg * I::r_symbol(&c, &g, &d) * fabcdgf,
                        frf2 + fcabdeg * I::r_symbol(&g, &c, &d) * fabcdgf,
                    )
                });
                assert_close(rfr1, frf1);
                assert_close(rfr2, frf2);
            }
        }
    }
}

fn assert_multiplicity_free_symbol_contracts<I: Sector>(a: I, b: I, c: I, d: I) {
    for out in a.fusion_outputs(&b) {
        assert_eq!(I::n_symbol(&a, &b, &out), 1);
        assert_fusion_tensor_has_expected_shape(&a, &b, &out);
        assert_a_b_symbols_follow_f_symbol_definitions(&a, &b, &out);
    }

    assert_symmetric_braiding(a.clone(), b.clone());
    assert_symmetric_braiding(b.clone(), c.clone());
    assert_triangle_equation(a.clone(), b.clone());
    assert_pentagon_equation(a.clone(), b.clone(), c.clone(), d);
    assert_hexagon_equation(a.clone(), b.clone(), c.clone());
    assert_f_move_unitary(a, b, c);
}

fn assert_fusion_tensors_orthonormal<I: Sector>(a: I, b: I) {
    let channels = a.fusion_outputs(&b).collect::<Vec<_>>();
    let tensors = channels
        .iter()
        .map(|c| (c.clone(), I::fusion_tensor(&a, &b, c).unwrap()))
        .collect::<Vec<_>>();

    for (left_channel, left_tensor) in &tensors {
        for (right_channel, right_tensor) in &tensors {
            for left_n in 0..left_tensor.shape()[3] {
                for right_n in 0..right_tensor.shape()[3] {
                    for left_m in 0..left_tensor.shape()[2] {
                        for right_m in 0..right_tensor.shape()[2] {
                            let mut overlap = 0.0;
                            for ma in 0..left_tensor.shape()[0] {
                                for mb in 0..left_tensor.shape()[1] {
                                    overlap += left_tensor[[ma, mb, left_m, left_n]]
                                        * right_tensor[[ma, mb, right_m, right_n]];
                                }
                            }
                            let expected = if left_channel == right_channel
                                && left_n == right_n
                                && left_m == right_m
                            {
                                1.0
                            } else {
                                0.0
                            };
                            assert_close(overlap, expected);
                        }
                    }
                }
            }
        }
    }
}

fn tensor_value(tensor: &ndarray::Array4<f64>, ka: usize, kb: usize, kc: usize) -> f64 {
    tensor[[ka, kb, kc, 0]]
}

fn su2_f_symbol_from_fusion_tensor_contraction(
    a: SU2Irrep,
    b: SU2Irrep,
    c: SU2Irrep,
    d: SU2Irrep,
    e: SU2Irrep,
    f: SU2Irrep,
) -> f64 {
    let tensor_ab_e = su2_fusion_tensor(a, b, e).unwrap();
    let tensor_e_c_d = su2_fusion_tensor(e, c, d).unwrap();
    let tensor_b_c_f = su2_fusion_tensor(b, c, f).unwrap();
    let tensor_a_f_d = su2_fusion_tensor(a, f, d).unwrap();

    let mut total = 0.0;
    for ma in 0..tensor_ab_e.shape()[0] {
        for mb in 0..tensor_ab_e.shape()[1] {
            for me in 0..tensor_ab_e.shape()[2] {
                for mc in 0..tensor_e_c_d.shape()[1] {
                    for mf in 0..tensor_b_c_f.shape()[2] {
                        total += tensor_value(&tensor_a_f_d, ma, mf, 0)
                            * tensor_value(&tensor_b_c_f, mb, mc, mf)
                            * tensor_value(&tensor_ab_e, ma, mb, me)
                            * tensor_value(&tensor_e_c_d, me, mc, 0);
                    }
                }
            }
        }
    }
    total
}
