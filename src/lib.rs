use pyo3::prelude::*;

// Gas constant in J/(mol·K)
const R: f64 = 8.314;

// ==================== Root finding ====================

/// Robust bisection for the free-monomer concentration.
///
/// Solves `inv(c_monomer) = conc` for `c_monomer` and returns the aggregated fraction
/// `1 - c_monomer / conc`. `inv` must be monotonically increasing with `inv(x) >= x`, so the
/// root always lies in `[0, min(conc, x_max)]`, where `x_max` is the singularity of `inv`
/// (e.g. `1/K`).
///
/// Designed to stay well-behaved for extreme fitting parameters:
/// - The bracket is bounded by `conc`, so a tiny `K` (huge `1/K`) cannot leave the
///   bisection far from converged after `num_itr` halvings.
/// - Non-finite `inv` values (`+inf` at the singularity, `NaN` from `inf * 0`) are treated
///   as "above the root", which is where they occur.
/// - Iteration stops early once the interval can no longer shrink in floating point.
/// - The result is always within `[0, 1]` (`NaN` only if the inputs are `NaN`).
///
/// Callers validate `conc` and the parameters first; the
/// guards below are only a defensive fallback.
fn solve_aggregation<F: Fn(f64) -> f64>(conc: f64, x_max: f64, num_itr: usize, inv: F) -> f64 {
    if !(conc > 0.0) || x_max.is_nan() {
        return f64::NAN;
    }
    let mut x_low = 0.0;
    let mut x_high = conc.min(x_max).max(0.0);
    if x_high == 0.0 {
        // x_max == 0 (K -> infinity): no free monomer can remain.
        return 1.0;
    }

    for _ in 0..num_itr {
        let x_mid = 0.5 * (x_low + x_high);
        if x_mid <= x_low || x_mid >= x_high {
            break; // interval exhausted at floating-point resolution
        }
        let f_mid = inv(x_mid);
        if f_mid.is_finite() && f_mid <= conc {
            x_low = x_mid;
        } else {
            x_high = x_mid;
        }
    }

    let x_mid = 0.5 * (x_low + x_high);
    (1.0 - x_mid / conc).clamp(0.0, 1.0)
}

/// Equilibrium constant from van 't Hoff parameters, `exp(-dH / RT + dS / R)`.
/// Overflow to `inf` / underflow to `0` is fine: the solvers handle both limits.
fn equilibrium_constant(delta_h: f64, delta_s: f64, t: f64) -> f64 {
    (-delta_h / (R * t) + delta_s / R).exp()
}

// ==================== Isodesmic Model ====================

/// Calculate the total concentration from monomer concentration (inverse model).
fn inv_isodesmic_model(c_monomer: f64, k: f64) -> f64 {
    let ck = k * c_monomer;
    if !(ck < 1.0) {
        return if ck.is_nan() { f64::NAN } else { f64::INFINITY };
    }
    let denominator = 1.0 - ck;
    c_monomer / (denominator * denominator)
}

/// Calculate the fraction of aggregated species (direct formula).
#[pyfunction]
fn isodesmic_model_direct(x: f64, k: f64) -> PyResult<f64> {
    if !(x.is_finite() && x > 0.0) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "concentration must be a positive finite number, got {x}"
        )));
    }
    Ok(isodesmic_direct_impl(x, k))
}

/// Closed-form isodesmic aggregation for a validated concentration `x > 0`.
fn isodesmic_direct_impl(x: f64, k: f64) -> f64 {
    if !(k >= 0.0) {
        return f64::NAN;
    }
    let b = k * x;
    if b == 0.0 {
        return 0.0;
    }
    let s = (4.0 * b + 1.0).sqrt();
    let den = 2.0 * b + 1.0 + s;
    if !den.is_finite() {
        return 1.0;
    }
    ((2.0 * b + 4.0 * b / (s + 1.0)) / den).clamp(0.0, 1.0)
}

/// Calculate the aggregation from total concentration (bisection method).
#[pyfunction]
fn isodesmic_model(conc: f64, k: f64, num_itr: usize) -> PyResult<f64> {
    if !(conc.is_finite() && conc > 0.0) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "concentration must be a positive finite number, got {conc}"
        )));
    }
    if !(k >= 0.0) {
        return Ok(f64::NAN);
    }
    Ok(solve_aggregation(conc, 1.0 / k, num_itr, |x| {
        inv_isodesmic_model(x, k)
    }))
}

/// Calculate isodesmic aggregation (direct formula, temperature-dependent).
#[pyfunction]
fn temp_isodesmic_model_direct(
    temp: Vec<f64>,
    delta_h: f64,
    delta_s: f64,
    c_tot: f64,
    scaler: f64,
) -> PyResult<Vec<f64>> {
    if !(c_tot.is_finite() && c_tot > 0.0) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "concentration must be a positive finite number, got {c_tot}"
        )));
    }
    let mut result = Vec::with_capacity(temp.len());
    for &t in &temp {
        if !(t.is_finite() && t > 0.0) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "temperature must be a positive finite number (K), got {t}"
            )));
        }
        let k = equilibrium_constant(delta_h, delta_s, t);
        result.push(isodesmic_direct_impl(c_tot, k) * scaler);
    }
    Ok(result)
}

/// Calculate isodesmic aggregation (bisection method, temperature-dependent).
#[pyfunction]
fn temp_isodesmic_model(
    temp: Vec<f64>,
    delta_h: f64,
    delta_s: f64,
    c_tot: f64,
    scaler: f64,
) -> PyResult<Vec<f64>> {
    if !(c_tot.is_finite() && c_tot > 0.0) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "concentration must be a positive finite number, got {c_tot}"
        )));
    }
    let mut result = Vec::with_capacity(temp.len());
    for &t in &temp {
        if !(t.is_finite() && t > 0.0) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "temperature must be a positive finite number (K), got {t}"
            )));
        }
        let k = equilibrium_constant(delta_h, delta_s, t);
        let agg = isodesmic_model(c_tot, k, 100)?;
        result.push(agg * scaler);
    }
    Ok(result)
}

// ==================== Cooperative Model ====================

/// Calculate the total concentration from monomer concentration (inverse model).
fn inv_cooperative_model(c_monomer: f64, k: f64, sigma: f64) -> f64 {
    if k == 0.0 {
        return c_monomer;
    }
    let ck = k * c_monomer;
    if !(ck < 1.0) {
        return if ck.is_nan() { f64::NAN } else { f64::INFINITY };
    }
    let denominator = 1.0 - ck;
    c_monomer + sigma * ck * c_monomer * (2.0 - ck) / (denominator * denominator)
}

/// Calculate the aggregation from total concentration (bisection method).
#[pyfunction]
fn cooperative_model(conc: f64, k: f64, sigma: f64, num_itr: usize) -> PyResult<f64> {
    if !(conc.is_finite() && conc > 0.0) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "concentration must be a positive finite number, got {conc}"
        )));
    }
    if !(k >= 0.0) || !(sigma >= 0.0) {
        return Ok(f64::NAN);
    }
    Ok(solve_aggregation(conc, 1.0 / k, num_itr, |x| {
        inv_cooperative_model(x, k, sigma)
    }))
}

/// Calculate cooperative aggregation (bisection method, temperature-dependent).
#[pyfunction]
fn temp_cooperative_model(
    temp: Vec<f64>,
    delta_h: f64,
    delta_s: f64,
    delta_h_nuc: f64,
    c_tot: f64,
    scaler: f64,
) -> PyResult<Vec<f64>> {
    if !(c_tot.is_finite() && c_tot > 0.0) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "concentration must be a positive finite number, got {c_tot}"
        )));
    }
    let mut result = Vec::with_capacity(temp.len());
    for &t in &temp {
        if !(t.is_finite() && t > 0.0) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "temperature must be a positive finite number (K), got {t}"
            )));
        }
        let k = equilibrium_constant(delta_h, delta_s, t);
        let sigma = (-delta_h_nuc / (R * t)).exp();
        let agg = cooperative_model(c_tot, k, sigma, 100)?;
        result.push(agg * scaler);
    }
    Ok(result)
}

// ==================== Mixed Model ====================

/// Calculate the total concentration in mixed model (inverse).
fn inv_coop_iso_model(c_monomer: f64, k_iso: f64, k_coop: f64, sigma: f64) -> f64 {
    let iso = inv_isodesmic_model(c_monomer, k_iso);
    let coop = inv_cooperative_model(c_monomer, k_coop, sigma);
    iso + coop - c_monomer
}

/// Calculate the aggregation from total concentration (bisection method, mixed model).
#[pyfunction]
fn coop_iso_model(conc: f64, k_iso: f64, k_coop: f64, sigma: f64, num_itr: usize) -> PyResult<f64> {
    if !(conc.is_finite() && conc > 0.0) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "concentration must be a positive finite number, got {conc}"
        )));
    }
    if !(k_iso >= 0.0) || !(k_coop >= 0.0) || !(sigma >= 0.0) {
        return Ok(f64::NAN);
    }
    let x_max = (1.0 / k_iso).min(1.0 / k_coop);
    Ok(solve_aggregation(conc, x_max, num_itr, |x| {
        inv_coop_iso_model(x, k_iso, k_coop, sigma)
    }))
}

/// Calculate mixed model aggregation (bisection method, temperature-dependent).
#[pyfunction]
fn temp_coop_iso_model(
    temp: Vec<f64>,
    delta_h_iso: f64,
    delta_s_iso: f64,
    delta_h_coop: f64,
    delta_s_coop: f64,
    delta_h_nuc_coop: f64,
    c_tot: f64,
    scaler: f64,
) -> PyResult<Vec<f64>> {
    if !(c_tot.is_finite() && c_tot > 0.0) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "concentration must be a positive finite number, got {c_tot}"
        )));
    }
    let mut result = Vec::with_capacity(temp.len());
    for &t in &temp {
        if !(t.is_finite() && t > 0.0) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "temperature must be a positive finite number (K), got {t}"
            )));
        }
        let k_iso = equilibrium_constant(delta_h_iso, delta_s_iso, t);
        let k_coop = equilibrium_constant(delta_h_coop, delta_s_coop, t);
        let sigma = (-delta_h_nuc_coop / (R * t)).exp();
        let agg = coop_iso_model(c_tot, k_iso, k_coop, sigma, 100)?;
        result.push(agg * scaler);
    }
    Ok(result)
}

// ==================== Cooperative Model (nucleus size N) ====================

/// Calculate the total concentration from monomer concentration (inverse model).
///
/// Nucleation–elongation model with an arbitrary nucleus size `N = nuc_size` (N >= 2):
/// a species of size `s` carries the cooperativity penalty `sigma^(min(s, N) - 1)`.
/// With `x = k * c_monomer`, the concentration of an `s`-mer is
/// `(sigma^(min(s, N) - 1) / k) * x^s`, so the total (monomer-unit) concentration is
///
///   c_tot = c_monomer + sum_{s>=2} s * (sigma^(min(s, N) - 1) / k) * x^s
///
/// This is evaluated as the closed-form elongation term (which assumes `sigma^(N-1)`
/// for every `s >= 2`) plus a finite correction over the nucleus interior `s = 2 ..= N-1`:
///
///   c_tot = c_monomer
///         + (sigma^(N-1) / k) * x^2 * (2 - x) / (1 - x)^2                    // elongation
///         + (1 / k) * sum_{s=2}^{N-1} s * (sigma^(s-1) - sigma^(N-1)) * x^s  // nucleus correction
///
/// The multiplicity factor `s` (from summing `s * [M_s]`) applies to the correction as
/// well as the elongation term. For `N = 2` the correction sum is empty and this reduces
/// exactly to the basic cooperative model (`inv_cooperative_model`).
///
fn inv_cooperative_model_n(c_monomer: f64, k: f64, sigma: f64, nuc_size: u32) -> f64 {
    if k == 0.0 {
        return c_monomer;
    }
    let ck = k * c_monomer;
    if !(ck < 1.0) {
        // c_tot diverges to +infinity as ck -> 1^-. Returning +infinity (rather than an
        // error or 0.0) keeps the bisection robust and accurate at full aggregation:
        // it correctly brackets the root just below the singularity.
        return if ck.is_nan() { f64::NAN } else { f64::INFINITY };
    }

    let denominator = 1.0 - ck;
    let sigma_pow_max = sigma.powi(nuc_size as i32 - 1); // sigma^(N-1)

    // Closed-form elongation term: uses sigma^(N-1) for every s >= 2.
    let elongation = sigma_pow_max * ck * c_monomer * (2.0 - ck) / (denominator * denominator);

    // Correction over the nucleus interior s = 2 ..= N-1, restoring the multiplicity
    // factor s and the correct penalty sigma^(s-1). The s = N term is zero, so the loop
    // stops at N-1; for N = 2 the loop body never runs and correction stays 0.
    //
    // The naive factor `sigma^(s-1) - sigma^(N-1)` cancels catastrophically as sigma -> 1
    // (both powers -> 1), losing up to ~7 digits near sigma = 1 - 1e-8. Rewrite it as
    //   sigma^(s-1) - sigma^(N-1) = sigma^(s-1) * (1 - sigma^(N-s))
    // and evaluate `1 - sigma^m` cancellation-free via -expm1(m * ln(sigma)), using
    // ln(sigma) = ln_1p(sigma - 1) so the log itself stays accurate near sigma = 1.
    let ln_sigma = (sigma - 1.0).ln_1p(); // = ln(sigma), accurate for sigma ~ 1
    let mut correction = 0.0;
    let mut x_pow_s_minus_1 = ck; // x^(s-1), starting at s = 2 (x^s / k = x^(s-1) * c_monomer)
    let mut sigma_pow = sigma; // sigma^(s-1), starting at s = 2 -> sigma^1
    for s in 2..nuc_size {
        let m = (nuc_size - s) as f64; // N - s >= 1
        let one_minus_sigma_pow = -(m * ln_sigma).exp_m1(); // 1 - sigma^(N-s)
        correction += (s as f64) * sigma_pow * one_minus_sigma_pow * x_pow_s_minus_1;
        x_pow_s_minus_1 *= ck;
        sigma_pow *= sigma;
    }
    correction *= c_monomer;

    c_monomer + elongation + correction
}

/// Calculate the aggregation from total concentration (bisection method).
#[pyfunction]
fn cooperative_model_n(
    conc: f64,
    k: f64,
    sigma: f64,
    nuc_size: u32,
    num_itr: usize,
) -> PyResult<f64> {
    if nuc_size < 2 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Nucleation size must be at least 2.".to_string(),
        ));
    }
    if !(conc.is_finite() && conc > 0.0) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "concentration must be a positive finite number, got {conc}"
        )));
    }
    if !(k >= 0.0) || !(sigma >= 0.0) {
        return Ok(f64::NAN);
    }
    Ok(solve_aggregation(conc, 1.0 / k, num_itr, |x| {
        inv_cooperative_model_n(x, k, sigma, nuc_size)
    }))
}

/// Calculate cooperative aggregation (bisection method, temperature-dependent).
#[pyfunction]
fn temp_cooperative_model_n(
    temp: Vec<f64>,
    delta_h: f64,
    delta_s: f64,
    delta_h_nuc: f64,
    c_tot: f64,
    scaler: f64,
    nuc_size: u32,
) -> PyResult<Vec<f64>> {
    if !(c_tot.is_finite() && c_tot > 0.0) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "concentration must be a positive finite number, got {c_tot}"
        )));
    }
    let mut result = Vec::with_capacity(temp.len());
    for &t in &temp {
        if !(t.is_finite() && t > 0.0) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "temperature must be a positive finite number (K), got {t}"
            )));
        }
        let k = equilibrium_constant(delta_h, delta_s, t);
        let sigma = (-delta_h_nuc / (R * t)).exp();
        let agg = cooperative_model_n(c_tot, k, sigma, nuc_size, 100)?;
        result.push(agg * scaler);
    }
    Ok(result)
}

/// A Python module implemented in Rust.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(isodesmic_model_direct, m)?)?;
    m.add_function(wrap_pyfunction!(isodesmic_model, m)?)?;
    m.add_function(wrap_pyfunction!(temp_isodesmic_model_direct, m)?)?;
    m.add_function(wrap_pyfunction!(temp_isodesmic_model, m)?)?;
    m.add_function(wrap_pyfunction!(cooperative_model, m)?)?;
    m.add_function(wrap_pyfunction!(temp_cooperative_model, m)?)?;
    m.add_function(wrap_pyfunction!(coop_iso_model, m)?)?;
    m.add_function(wrap_pyfunction!(temp_coop_iso_model, m)?)?;
    m.add_function(wrap_pyfunction!(cooperative_model_n, m)?)?;
    m.add_function(wrap_pyfunction!(temp_cooperative_model_n, m)?)?;
    Ok(())
}
