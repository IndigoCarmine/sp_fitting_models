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

// ==================== Cooperative Model (n=3) ====================

/// Calculate the total concentration from monomer concentration (inverse model).
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
        return Ok(0.0);
    }

    let denominator = 1.0 - ck;

    // additional term for n >= 3,  sum (sigma^(n-1) - sigma) / k * cK^n
    let mut additional_term = 0.0;
    let mut ck_pow = ck * ck;
    let mut sigma_pow = sigma;
    let sigma_pow_max = sigma.powi(nuc_size as i32 - 1);

    for _ in 1..=nuc_size - 1 {
        ck_pow *= ck;
        sigma_pow *= sigma;
        additional_term += (sigma_pow - sigma_pow_max) / k * ck_pow;
    }

    Ok(c_monomer
        + sigma_pow_max / k * (ck * ck * (2.0 - ck)) / (denominator * denominator)
        + additional_term)
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
