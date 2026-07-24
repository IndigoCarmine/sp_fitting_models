use pyo3::prelude::*;

// Gas constant in J/(mol·K)
const R: f64 = 8.314;

// ==================== Isodesmic Model ====================

/// Calculate the total concentration from monomer concentration (inverse model).
fn inv_isodesmic_model(c_monomer: f64, k: f64) -> Result<f64, String> {
    if k * c_monomer > 1.0 {
        return Err("K * c_monomer must be less than 1 for the isodesmic model.".to_string());
    }
    let denominator = 1.0 - k * c_monomer;
    return Ok(c_monomer / (denominator * denominator));
}

/// Calculate the fraction of aggregated species (direct formula).
#[pyfunction]
fn isodesmic_model_direct(x: f64, k: f64) -> f64 {
    let b = k * x;
    let z = (2.0 * b + 1.0 - (4.0 * b + 1.0).sqrt()) / (2.0 * b);
    1.0 - z / b
}

/// Calculate the aggregation from total concentration (bisection method).
#[pyfunction]
fn isodesmic_model(conc: f64, k: f64, num_itr: usize) -> PyResult<f64> {
    let mut x_low = 0.0;
    let mut x_high = 1.0 / k;

    for _ in 0..num_itr {
        let x_mid = (x_low + x_high) / 2.0;
        let f_mid = match inv_isodesmic_model(x_mid, k) {
            Ok(result) => result - conc,
            Err(e) => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(e)),
        };
        if f_mid <= 0.0 {
            x_low = x_mid;
        } else {
            x_high = x_mid;
        }
    }

    let x_mid = (x_low + x_high) / 2.0;
    match inv_isodesmic_model(x_mid, k) {
        Ok(_) => Ok(1.0 - x_mid / conc),
        Err(e) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(e)),
    }
}

/// Calculate isodesmic aggregation (direct formula, temperature-dependent).
#[pyfunction]
fn temp_isodesmic_model_direct(
    temp: Vec<f64>,
    delta_h: f64,
    delta_s: f64,
    c_tot: f64,
    scaler: f64,
) -> Vec<f64> {
    let mut result = Vec::with_capacity(temp.len());
    for &t in &temp {
        let k = (-delta_h / (R * t) + delta_s / R).exp();
        result.push(isodesmic_model_direct(c_tot, k) * scaler);
    }
    result
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
    let mut result = Vec::with_capacity(temp.len());
    for &t in &temp {
        let k = (-delta_h / (R * t) + delta_s / R).exp();
        let agg = isodesmic_model(c_tot, k, 100)?;
        result.push(agg * scaler);
    }
    Ok(result)
}

// ==================== Cooperative Model ====================

/// Calculate the total concentration from monomer concentration (inverse model).
fn inv_cooperative_model(c_monomer: f64, k: f64, sigma: f64) -> Result<f64, String> {
    if k == 0.0 {
        return Ok(c_monomer);
    }
    let ck = k * c_monomer;
    if ck >= 1.0 {
        return Err("K * c_monomer must be less than 1 for the cooperative model.".to_string());
    }
    let denominator = 1.0 - ck;
    Ok(c_monomer + sigma / k * (ck * ck * (2.0 - ck)) / (denominator * denominator))
}

/// Calculate the aggregation from total concentration (bisection method).
#[pyfunction]
fn cooperative_model(conc: f64, k: f64, sigma: f64, num_itr: usize) -> PyResult<f64> {
    let mut x_low = 0.0;
    let mut x_high = 1.0 / k;

    for _ in 0..num_itr {
        let x_mid = (x_low + x_high) / 2.0;
        let f_mid = match inv_cooperative_model(x_mid, k, sigma) {
            Ok(result) => result - conc,
            Err(e) => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(e)),
        };

        if f_mid <= 0.0 {
            x_low = x_mid;
        } else {
            x_high = x_mid;
        }
    }

    let x_mid = (x_low + x_high) / 2.0;
    Ok(1.0 - x_mid / conc)
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
    let mut result = Vec::with_capacity(temp.len());
    for &t in &temp {
        let k = (-delta_h / (R * t) + delta_s / R).exp();
        let sigma = (-delta_h_nuc / (R * t)).exp();
        let agg = cooperative_model(c_tot, k, sigma, 100)?;
        result.push(agg * scaler);
    }
    Ok(result)
}

// ==================== Mixed Model ====================

/// Calculate the total concentration in mixed model (inverse).
fn inv_coop_iso_model(c_monomer: f64, k_iso: f64, k_coop: f64, sigma: f64) -> Result<f64, String> {
    let iso = inv_isodesmic_model(c_monomer, k_iso)?;
    let coop = inv_cooperative_model(c_monomer, k_coop, sigma)?;
    Ok(iso + coop - c_monomer)
}

/// Calculate the aggregation from total concentration (bisection method, mixed model).
#[pyfunction]
fn coop_iso_model(conc: f64, k_iso: f64, k_coop: f64, sigma: f64, num_itr: usize) -> PyResult<f64> {
    let mut x_low = 0.0;
    let mut x_high = (1.0 / k_iso).min(1.0 / k_coop);

    for _ in 0..num_itr {
        let x_mid = (x_low + x_high) / 2.0;
        let f_mid = match inv_coop_iso_model(x_mid, k_iso, k_coop, sigma) {
            Ok(result) => result - conc,
            Err(e) => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(e)),
        };

        if f_mid <= 0.0 {
            x_low = x_mid;
        } else {
            x_high = x_mid;
        }
    }

    let x_mid = (x_low + x_high) / 2.0;
    Ok(1.0 - x_mid / conc)
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
    let mut result = Vec::with_capacity(temp.len());
    for &t in &temp {
        let k_iso = (-delta_h_iso / (R * t) + delta_s_iso / R).exp();
        let k_coop = (-delta_h_coop / (R * t) + delta_s_coop / R).exp();
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
fn inv_cooperative_model_n(
    c_monomer: f64,
    k: f64,
    sigma: f64,
    nuc_size: u32,
) -> Result<f64, String> {
    if k == 0.0 {
        return Ok(c_monomer);
    }
    let ck = k * c_monomer;
    if ck >= 1.0 {
        // c_tot diverges to +infinity as ck -> 1^-. Returning +infinity (rather than an
        // error or 0.0) keeps the bisection in `cooperative_model_n` robust and accurate at
        // full aggregation: f_mid = +inf - conc > 0 correctly brackets the root just below
        // the singularity, so x_high is pulled down toward the true free-monomer value.
        return Ok(f64::INFINITY);
    }

    let denominator = 1.0 - ck;
    let sigma_pow_max = sigma.powi(nuc_size as i32 - 1); // sigma^(N-1)

    // Closed-form elongation term: uses sigma^(N-1) for every s >= 2.
    let elongation = sigma_pow_max / k * (ck * ck * (2.0 - ck)) / (denominator * denominator);

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
    let mut ck_pow = ck * ck; // x^s, starting at s = 2
    let mut sigma_pow = sigma; // sigma^(s-1), starting at s = 2 -> sigma^1
    for s in 2..nuc_size {
        let m = (nuc_size - s) as f64; // N - s >= 1
        let one_minus_sigma_pow = -(m * ln_sigma).exp_m1(); // 1 - sigma^(N-s)
        correction += (s as f64) * sigma_pow * one_minus_sigma_pow * ck_pow;
        ck_pow *= ck;
        sigma_pow *= sigma;
    }
    correction /= k;

    Ok(c_monomer + elongation + correction)
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
    let mut x_low = 0.0;
    let mut x_high = 1.0 / k;
    if nuc_size < 2 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Nucleation size must be at least 2.".to_string(),
        ));
    }
    for _ in 0..num_itr {
        let x_mid = (x_low + x_high) / 2.0;
        let f_mid = match inv_cooperative_model_n(x_mid, k, sigma, nuc_size) {
            Ok(result) => result - conc,
            Err(e) => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(e)),
        };

        if f_mid <= 0.0 {
            x_low = x_mid;
        } else {
            x_high = x_mid;
        }
    }

    let x_mid = (x_low + x_high) / 2.0;
    Ok(1.0 - x_mid / conc)
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
    let mut result = Vec::with_capacity(temp.len());
    for &t in &temp {
        let k = (-delta_h / (R * t) + delta_s / R).exp();
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
