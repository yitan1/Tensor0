use crate::error::{Result, Tensor0Error};

const ZERO_TOLERANCE: f64 = 1.0e-15;
// Phase 4 only needs deterministic small-spin debug/recoupling helpers, not an
// arbitrary-spin numerical engine. This cap keeps f64 factorial products from
// silently entering unreliable large-value regimes.
const MAX_SMALL_SPIN_FACTORIAL_ARGUMENT: i64 = 32;

/// Compute the bare SU2 Wigner 6j symbol from twice-spin labels.
///
/// This is not TensorKit's Racah W coefficient; `su2_f_symbol_spin2` applies
/// the extra Racah phase used by `WignerSymbols.racahW`.
pub fn wigner_6j_spin2(a: i64, b: i64, c: i64, d: i64, e: i64, f: i64) -> Result<f64> {
    validate_triangle(a, b, c)?;
    validate_triangle(a, e, f)?;
    validate_triangle(d, b, f)?;
    validate_triangle(d, e, c)?;

    let delta = triangle_delta(a, b, c)?
        * triangle_delta(a, e, f)?
        * triangle_delta(d, b, f)?
        * triangle_delta(d, e, c)?;

    let x1 = half_sum3(a, b, c, "6j lower summation bound")?;
    let x2 = half_sum3(a, e, f, "6j lower summation bound")?;
    let x3 = half_sum3(d, b, f, "6j lower summation bound")?;
    let x4 = half_sum3(d, e, c, "6j lower summation bound")?;
    let y1 = half_sum4(a, b, d, e, "6j upper summation bound")?;
    let y2 = half_sum4(b, c, e, f, "6j upper summation bound")?;
    let y3 = half_sum4(a, c, d, f, "6j upper summation bound")?;

    let z_min = max4(x1, x2, x3, x4);
    let z_max = y1.min(y2).min(y3);
    if z_min > z_max {
        return Err(Tensor0Error::Message(
            "SU2 Wigner 6j labels are not admissible".to_string(),
        ));
    }

    let mut sum = 0.0;
    for z in z_min..=z_max {
        let numerator = phase(z) * factorial(checked_add(z, 1, "6j factorial argument")?)?;
        let denominator = factorial(checked_sub(z, x1, "6j factorial argument")?)?
            * factorial(checked_sub(z, x2, "6j factorial argument")?)?
            * factorial(checked_sub(z, x3, "6j factorial argument")?)?
            * factorial(checked_sub(z, x4, "6j factorial argument")?)?
            * factorial(checked_sub(y1, z, "6j factorial argument")?)?
            * factorial(checked_sub(y2, z, "6j factorial argument")?)?
            * factorial(checked_sub(y3, z, "6j factorial argument")?)?;
        sum += numerator / denominator;
    }

    Ok(clean_zero(delta * sum))
}

/// Compute the SU2 F-symbol with TensorKitSectors' Racah W convention from
/// twice-spin labels.
pub fn su2_f_symbol_spin2(s1: i64, s2: i64, s3: i64, s4: i64, s5: i64, s6: i64) -> Result<f64> {
    let dim_factor = (su2_dim_spin2(s5)? as f64 * su2_dim_spin2(s6)? as f64).sqrt();
    let racah_w = su2_racah_w_spin2(s1, s2, s4, s3, s5, s6)?;
    Ok(clean_zero(dim_factor * racah_w))
}

fn su2_racah_w_spin2(j1: i64, j2: i64, j: i64, j3: i64, j12: i64, j23: i64) -> Result<f64> {
    // Matches WignerSymbols.racahW(j1, j2, j, j3, j12, j23), including
    // the Racah phase relative to the bare Wigner 6j symbol.
    let exponent = half_sum4(j1, j2, j3, j, "Racah W phase")?;
    let sign = phase(exponent);
    Ok(sign * wigner_6j_spin2(j1, j2, j12, j3, j, j23)?)
}

/// Compute `<ja, ma; jb, mb | jc, mc>` from twice-spin and twice-magnetic
/// labels.
pub fn su2_clebsch_gordan_spin2(
    ja2: i64,
    ma2: i64,
    jb2: i64,
    mb2: i64,
    jc2: i64,
    mc2: i64,
) -> Result<f64> {
    validate_magnetic_label(ja2, ma2, "ma2")?;
    validate_magnetic_label(jb2, mb2, "mb2")?;
    validate_magnetic_label(jc2, mc2, "mc2")?;
    validate_triangle(ja2, jb2, jc2)?;

    if checked_add(ma2, mb2, "magnetic label sum")? != mc2 {
        return Ok(0.0);
    }

    let triangle_factor = factorial(half_expr(&[jc2, ja2, -jb2], "CG prefactor")?)?
        * factorial(half_expr(&[jc2, -ja2, jb2], "CG prefactor")?)?
        * factorial(half_expr(&[ja2, jb2, -jc2], "CG prefactor")?)?
        / factorial(checked_add(
            half_sum3(ja2, jb2, jc2, "CG prefactor")?,
            1,
            "CG prefactor",
        )?)?;
    let coupled_magnetic_factor = factorial(half_expr(&[jc2, mc2], "CG magnetic prefactor")?)?
        * factorial(half_expr(&[jc2, -mc2], "CG magnetic prefactor")?)?;
    let uncoupled_magnetic_factor = factorial(half_expr(&[ja2, -ma2], "CG magnetic prefactor")?)?
        * factorial(half_expr(&[ja2, ma2], "CG magnetic prefactor")?)?
        * factorial(half_expr(&[jb2, -mb2], "CG magnetic prefactor")?)?
        * factorial(half_expr(&[jb2, mb2], "CG magnetic prefactor")?)?;
    let prefactor = (su2_dim_spin2(jc2)? as f64
        * triangle_factor
        * coupled_magnetic_factor
        * uncoupled_magnetic_factor)
        .sqrt();

    let t1 = half_expr(&[ja2, jb2, -jc2], "CG summation bound")?;
    let t2 = half_expr(&[ja2, -ma2], "CG summation bound")?;
    let t3 = half_expr(&[jb2, mb2], "CG summation bound")?;
    let u1 = half_expr(&[jc2, -jb2, ma2], "CG summation bound")?;
    let u2 = half_expr(&[jc2, -ja2, -mb2], "CG summation bound")?;

    let k_min = 0.max(-u1).max(-u2);
    let k_max = t1.min(t2).min(t3);
    if k_min > k_max {
        return Ok(0.0);
    }

    let mut sum = 0.0;
    for k in k_min..=k_max {
        let denominator = factorial(k)?
            * factorial(checked_sub(t1, k, "CG factorial argument")?)?
            * factorial(checked_sub(t2, k, "CG factorial argument")?)?
            * factorial(checked_sub(t3, k, "CG factorial argument")?)?
            * factorial(checked_add(u1, k, "CG factorial argument")?)?
            * factorial(checked_add(u2, k, "CG factorial argument")?)?;
        sum += phase(k) / denominator;
    }

    Ok(clean_zero(prefactor * sum))
}

fn su2_dim_spin2(spin2: i64) -> Result<usize> {
    let spin2 = usize::try_from(spin2)
        .map_err(|_| Tensor0Error::Message("SU2 spin must be non-negative".to_string()))?;
    spin2
        .checked_add(1)
        .ok_or_else(|| Tensor0Error::Message("SU2 quantum dimension overflowed".to_string()))
}

fn validate_magnetic_label(spin2: i64, magnetic2: i64, name: &str) -> Result<()> {
    if magnetic2 < -spin2 || magnetic2 > spin2 {
        return Err(Tensor0Error::Message(format!(
            "SU2 magnetic label {name}={magnetic2} is outside [-{spin2}, {spin2}]",
        )));
    }
    if magnetic2.rem_euclid(2) != spin2.rem_euclid(2) {
        return Err(Tensor0Error::Message(format!(
            "SU2 magnetic label {name}={magnetic2} has the wrong parity for spin2={spin2}",
        )));
    }
    Ok(())
}

fn validate_triangle(a: i64, b: i64, c: i64) -> Result<()> {
    if a < 0 || b < 0 || c < 0 {
        return Err(Tensor0Error::Message(
            "SU2 spin must be non-negative".to_string(),
        ));
    }
    let ab = checked_add(a, b, "SU2 triangle condition")?;
    let ac = checked_add(a, c, "SU2 triangle condition")?;
    let bc = checked_add(b, c, "SU2 triangle condition")?;
    if ab < c || ac < b || bc < a {
        return Err(Tensor0Error::Message(
            "SU2 labels do not satisfy the triangle condition".to_string(),
        ));
    }
    if checked_add(ab, c, "SU2 triangle parity")?.rem_euclid(2) != 0 {
        return Err(Tensor0Error::Message(
            "SU2 labels do not satisfy the triangle parity condition".to_string(),
        ));
    }
    Ok(())
}

fn triangle_delta(a: i64, b: i64, c: i64) -> Result<f64> {
    let numerator = factorial(half_expr(&[a, b, -c], "6j triangle delta")?)?
        * factorial(half_expr(&[a, -b, c], "6j triangle delta")?)?
        * factorial(half_expr(&[-a, b, c], "6j triangle delta")?)?;
    let denominator = factorial(checked_add(
        half_sum3(a, b, c, "6j triangle delta")?,
        1,
        "6j triangle delta",
    )?)?;
    Ok((numerator / denominator).sqrt())
}

fn factorial(n: i64) -> Result<f64> {
    if n < 0 {
        return Err(Tensor0Error::Message(format!(
            "SU2 symbol factorial argument must be non-negative, got {n}",
        )));
    }
    if n > MAX_SMALL_SPIN_FACTORIAL_ARGUMENT {
        return Err(Tensor0Error::Message(format!(
            "SU2 symbol helpers only support small spins: factorial argument {n} exceeds {MAX_SMALL_SPIN_FACTORIAL_ARGUMENT}",
        )));
    }
    let n = usize::try_from(n).map_err(|_| {
        Tensor0Error::Message("SU2 symbol factorial argument overflowed".to_string())
    })?;
    Ok((1..=n).fold(1.0, |total, value| total * value as f64))
}

fn half_sum3(a: i64, b: i64, c: i64, context: &str) -> Result<i64> {
    half_expr(&[a, b, c], context)
}

fn half_sum4(a: i64, b: i64, c: i64, d: i64, context: &str) -> Result<i64> {
    half_expr(&[a, b, c, d], context)
}

fn half_expr(terms: &[i64], context: &str) -> Result<i64> {
    let total = terms
        .iter()
        .try_fold(0i64, |total, term| checked_add(total, *term, context))?;
    if total.rem_euclid(2) != 0 {
        return Err(Tensor0Error::Message(format!(
            "SU2 symbol half-integer expression is not integral in {context}",
        )));
    }
    Ok(total / 2)
}

fn checked_add(left: i64, right: i64, context: &str) -> Result<i64> {
    left.checked_add(right).ok_or_else(|| {
        Tensor0Error::Message(format!(
            "SU2 symbol integer arithmetic overflowed in {context}",
        ))
    })
}

fn checked_sub(left: i64, right: i64, context: &str) -> Result<i64> {
    left.checked_sub(right).ok_or_else(|| {
        Tensor0Error::Message(format!(
            "SU2 symbol integer arithmetic overflowed in {context}",
        ))
    })
}

fn phase(exponent: i64) -> f64 {
    if exponent.rem_euclid(2) == 0 {
        1.0
    } else {
        -1.0
    }
}

fn max4(a: i64, b: i64, c: i64, d: i64) -> i64 {
    a.max(b).max(c).max(d)
}

fn clean_zero(value: f64) -> f64 {
    if value.abs() < ZERO_TOLERANCE {
        0.0
    } else {
        value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1.0e-12,
            "actual={actual}, expected={expected}",
        );
    }

    #[test]
    fn numeric_wigner_symbol_helpers_use_spin2_labels() {
        let sqrt3_over_2 = 3.0_f64.sqrt() / 2.0;
        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();

        assert_close(wigner_6j_spin2(1, 1, 0, 1, 1, 0).unwrap(), -0.5);
        assert_close(su2_f_symbol_spin2(1, 1, 1, 1, 0, 2).unwrap(), sqrt3_over_2);
        assert_close(
            su2_clebsch_gordan_spin2(1, 1, 1, -1, 0, 0).unwrap(),
            inv_sqrt2,
        );
    }

    #[test]
    fn clebsch_gordan_rejects_invalid_magnetic_labels_and_preserves_zero_selection_rule() {
        let parity_error = su2_clebsch_gordan_spin2(1, 0, 1, 0, 0, 0)
            .unwrap_err()
            .to_string();
        assert!(parity_error.contains("magnetic label"));
        assert!(parity_error.contains("wrong parity"));

        let range_error = su2_clebsch_gordan_spin2(1, 3, 1, -1, 0, 0)
            .unwrap_err()
            .to_string();
        assert!(range_error.contains("magnetic label"));
        assert!(range_error.contains("outside"));

        assert_close(su2_clebsch_gordan_spin2(1, 1, 1, 1, 0, 0).unwrap(), 0.0);
    }

    #[test]
    fn wigner_helpers_clean_float_cancellation_zero() {
        assert_eq!(wigner_6j_spin2(2, 12, 12, 22, 18, 18).unwrap(), 0.0);
    }

    #[test]
    fn wigner_helpers_use_narrow_zero_tolerance() {
        assert_eq!(clean_zero(5.0e-16), 0.0);
        assert_eq!(clean_zero(5.0e-15), 5.0e-15);
    }

    #[test]
    fn wigner_helpers_reject_outside_small_spin_range() {
        let error = wigner_6j_spin2(64, 64, 64, 64, 64, 64)
            .unwrap_err()
            .to_string();
        assert!(error.contains("small spins"));
    }
}
