use pyo3::prelude::*;

// Gas constant in J/(mol·K)
const R: f64 = 8.314;

// ==================== Isodesmic Model ====================

/// Calculate the total concentration from monomer concentration (inverse model).
fn inv_isodesmic_model(c_monomer: &[f64], k: f64) -> Result<Vec<f64>, String> {
    let mut result = Vec::with_capacity(c_monomer.len());
    for &c in c_monomer {
        if k * c > 1.0 {
            return Err("K * c_monomer must be less than 1 for the isodesmic model.".to_string());
        }
        let denominator = 1.0 - k * c;
        result.push(c / (denominator * denominator));
    }
    Ok(result)
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
        let f_mid = match inv_isodesmic_model(&[x_mid], k) {
            Ok(result) => result[0] - conc,
            Err(e) => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(e)),
        };
        let f_low = match inv_isodesmic_model(&[x_low], k) {
            Ok(result) => result[0] - conc,
            Err(e) => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(e)),
        };

        if f_mid * f_low <= 0.0 {
            x_high = x_mid;
        } else {
            x_low = x_mid;
        }
    }

    let x_mid = (x_low + x_high) / 2.0;
    match inv_isodesmic_model(&[x_mid], k) {
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
fn inv_cooperative_model(c_monomer: &[f64], k: f64, sigma: f64) -> Result<Vec<f64>, String> {
    let mut result = Vec::with_capacity(c_monomer.len());
    for &c in c_monomer {
        if k == 0.0 {
            result.push(c);
            continue;
        }
        let ck = k * c;
        if ck >= 1.0 {
            return Err("K * c_monomer must be less than 1 for the cooperative model.".to_string());
        }
        let denominator = 1.0 - ck;
        result.push(c + sigma / k * (ck * ck * (2.0 - ck)) / (denominator * denominator));
    }
    Ok(result)
}

/// Calculate the aggregation from total concentration (bisection method).
#[pyfunction]
fn cooperative_model(conc: f64, k: f64, sigma: f64, num_itr: usize) -> PyResult<f64> {
    let mut x_low = 0.0;
    let mut x_high = 1.0 / k;

    for _ in 0..num_itr {
        let x_mid = (x_low + x_high) / 2.0;
        let f_mid = match inv_cooperative_model(&[x_mid], k, sigma) {
            Ok(result) => result[0] - conc,
            Err(e) => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(e)),
        };
        let f_low = match inv_cooperative_model(&[x_low], k, sigma) {
            Ok(result) => result[0] - conc,
            Err(e) => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(e)),
        };

        if f_mid * f_low <= 0.0 {
            x_high = x_mid;
        } else {
            x_low = x_mid;
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
fn inv_coop_iso_model(
    c_monomer: &[f64],
    k_iso: f64,
    k_coop: f64,
    sigma: f64,
) -> Result<Vec<f64>, String> {
    let iso = inv_isodesmic_model(c_monomer, k_iso)?;
    let coop = inv_cooperative_model(c_monomer, k_coop, sigma)?;
    
    Ok(iso.iter().zip(coop.iter()).zip(c_monomer.iter())
        .map(|((&i, &c), &m)| i + c - m)
        .collect())
}

/// Calculate the aggregation from total concentration (bisection method, mixed model).
#[pyfunction]
fn coop_iso_model(
    conc: f64,
    k_iso: f64,
    k_coop: f64,
    sigma: f64,
    num_itr: usize,
) -> PyResult<f64> {
    let mut x_low = 0.0;
    let mut x_high = (1.0 / k_iso).min(1.0 / k_coop);

    for _ in 0..num_itr {
        let x_mid = (x_low + x_high) / 2.0;
        let f_mid = match inv_coop_iso_model(&[x_mid], k_iso, k_coop, sigma) {
            Ok(result) => result[0] - conc,
            Err(e) => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(e)),
        };
        let f_low = match inv_coop_iso_model(&[x_low], k_iso, k_coop, sigma) {
            Ok(result) => result[0] - conc,
            Err(e) => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(e)),
        };

        if f_mid * f_low <= 0.0 {
            x_high = x_mid;
        } else {
            x_low = x_mid;
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
    Ok(())
}
