"""
Test comparisons between Rust and Python (original) implementations.

This test module verifies that the Rust backend produces equivalent results
to the original Python implementation using Numba.
"""

import numpy as np
import pytest
from sp_fitting_models.models import (
    isodesmic_model_direct,
    isodesmic_model,
    temp_isodesmic_model_direct,
    temp_isodesmic_model,
    cooperative_model,
    temp_cooperative_model,
    coop_iso_model,
    temp_coop_iso_model,
)
from sp_fitting_models.models.models_old import (
    isodesmic as py_isodesmic,
    cooperative as py_cooperative,
    mixed as py_mixed,
)


class TestIsodesmicModel:
    """Tests for isodesmic model implementations."""

    def test_isodesmic_model_direct_scalar(self):
        """Test scalar input for direct formula."""
        X = 1.0
        K = 0.5
        rust_result = isodesmic_model_direct(X, K)
        py_result = py_isodesmic.isodesmic_model_direct(X, K)
        np.testing.assert_allclose(rust_result, py_result, rtol=1e-10)

    def test_isodesmic_model_direct_array(self):
        """Test array input for direct formula."""
        X = np.array([0.5, 1.0, 1.5, 2.0])
        K = 0.3
        rust_result = isodesmic_model_direct(X, K)
        py_result = py_isodesmic.isodesmic_model_direct(X, K)
        np.testing.assert_allclose(rust_result, py_result, rtol=1e-10)

    def test_isodesmic_model_bisection_scalar(self):
        """Test scalar input for bisection method."""
        Conc = 2.0
        K = 0.3
        rust_result = isodesmic_model(Conc, K)
        py_result = py_isodesmic.isodesmic_model(np.array([Conc]), K)[0]
        np.testing.assert_allclose(rust_result, py_result, rtol=1e-8)

    def test_isodesmic_model_bisection_array(self):
        """Test array input for bisection method."""
        Conc = np.array([1.0, 2.0, 3.0])
        K = 0.25
        rust_result = isodesmic_model(Conc, K)
        py_result = py_isodesmic.isodesmic_model(Conc, K)
        np.testing.assert_allclose(rust_result, py_result, rtol=1e-8)

    def test_temp_isodesmic_model_direct(self):
        """Test temperature-dependent direct formula."""
        Temp = np.array([280.0, 298.0, 310.0, 320.0])
        deltaH = -50000.0  # J/mol
        deltaS = -100.0  # J/(mol·K)
        c_tot = 1.0
        scaler = 1.0

        rust_result = temp_isodesmic_model_direct(Temp, deltaH, deltaS, c_tot, scaler)
        py_result = py_isodesmic.temp_isodesmic_model_direct(Temp, deltaH, deltaS, c_tot, scaler)
        np.testing.assert_allclose(rust_result, py_result, rtol=1e-10)

    def test_temp_isodesmic_model(self):
        """Test temperature-dependent bisection method."""
        Temp = np.array([280.0, 298.0, 310.0, 320.0])
        deltaH = -50000.0  # J/mol
        deltaS = -100.0  # J/(mol·K)
        c_tot = 1.0
        scaler = 1.0

        rust_result = temp_isodesmic_model(Temp, deltaH, deltaS, c_tot, scaler)
        py_result = py_isodesmic.temp_isodesmic_model(Temp, deltaH, deltaS, c_tot, scaler)
        np.testing.assert_allclose(rust_result, py_result, rtol=1e-8)


class TestCooperativeModel:
    """Tests for cooperative model implementations."""

    def test_cooperative_model_bisection_scalar(self):
        """Test scalar input for bisection method."""
        Conc = 2.0
        K = 0.3
        sigma = 0.1
        rust_result = cooperative_model(Conc, K, sigma)
        py_result = py_cooperative.cooperative_model(np.array([Conc]), K, sigma)[0]
        np.testing.assert_allclose(rust_result, py_result, rtol=1e-8)

    def test_cooperative_model_bisection_array(self):
        """Test array input for bisection method."""
        Conc = np.array([1.0, 2.0, 3.0])
        K = 0.25
        sigma = 0.05
        rust_result = cooperative_model(Conc, K, sigma)
        py_result = py_cooperative.cooperative_model(Conc, K, sigma)
        np.testing.assert_allclose(rust_result, py_result, rtol=1e-8)

    def test_temp_cooperative_model(self):
        """Test temperature-dependent bisection method."""
        Temp = np.array([280.0, 298.0, 310.0, 320.0])
        deltaH = -60000.0  # J/mol
        deltaS = -150.0  # J/(mol·K)
        deltaHnuc = -30000.0  # J/mol
        c_tot = 1.0
        scaler = 1.0

        rust_result = temp_cooperative_model(Temp, deltaH, deltaS, deltaHnuc, c_tot, scaler)
        py_result = py_cooperative.temp_cooperative_model(Temp, deltaH, deltaS, deltaHnuc, c_tot, scaler)
        np.testing.assert_allclose(rust_result, py_result, rtol=1e-8)


class TestMixedModel:
    """Tests for mixed cooperative-isodesmic model implementations."""

    def test_coop_iso_model_scalar(self):
        """Test scalar input for mixed model."""
        Conc = 2.0
        K_iso = 0.3
        K_coop = 0.25
        sigma = 0.1
        rust_result = coop_iso_model(Conc, K_iso, K_coop, sigma)
        py_result = py_mixed.coop_iso_model(np.array([Conc]), K_iso, K_coop, sigma)[0]
        np.testing.assert_allclose(rust_result, py_result, rtol=1e-8)

    def test_coop_iso_model_array(self):
        """Test array input for mixed model."""
        Conc = np.array([1.0, 2.0, 3.0])
        K_iso = 0.2
        K_coop = 0.15
        sigma = 0.05
        rust_result = coop_iso_model(Conc, K_iso, K_coop, sigma)
        py_result = py_mixed.coop_iso_model(Conc, K_iso, K_coop, sigma)
        np.testing.assert_allclose(rust_result, py_result, rtol=1e-8)

    def test_temp_coop_iso_model(self):
        """Test temperature-dependent mixed model."""
        Temp = np.array([280.0, 298.0, 310.0, 320.0])
        deltaH_iso = -50000.0  # J/mol
        deltaS_iso = -100.0  # J/(mol·K)
        deltaH_coop = -60000.0  # J/mol
        deltaS_coop = -150.0  # J/(mol·K)
        deltaHnuc_coop = -30000.0  # J/mol
        c_tot = 1.0
        scaler = 1.0

        rust_result = temp_coop_iso_model(
            Temp, deltaH_iso, deltaS_iso, deltaH_coop, deltaS_coop, deltaHnuc_coop, c_tot, scaler
        )
        py_result = py_mixed.temp_coop_iso_model(
            Temp, deltaH_iso, deltaS_iso, deltaH_coop, deltaS_coop, deltaHnuc_coop, c_tot, scaler
        )
        np.testing.assert_allclose(rust_result, py_result, rtol=1e-8)


class TestEdgeCases:
    """Tests for edge cases and special conditions."""

    def test_isodesmic_low_concentration(self):
        """Test with very low concentration."""
        Conc = 0.01
        K = 0.1
        rust_result = isodesmic_model(Conc, K)
        py_result = py_isodesmic.isodesmic_model(np.array([Conc]), K)[0]
        np.testing.assert_allclose(rust_result, py_result, rtol=1e-6)

    def test_isodesmic_high_aggregation(self):
        """Test with parameters that give high aggregation."""
        Conc = 5.0
        K = 0.15
        rust_result = isodesmic_model(Conc, K)
        py_result = py_isodesmic.isodesmic_model(np.array([Conc]), K)[0]
        np.testing.assert_allclose(rust_result, py_result, rtol=1e-8)

    def test_cooperative_with_small_sigma(self):
        """Test cooperative model with small cooperativity (nearly isodesmic)."""
        Conc = 2.0
        K = 0.2
        sigma = 0.001  # Very small cooperativity
        rust_result = cooperative_model(Conc, K, sigma)
        py_result = py_cooperative.cooperative_model(np.array([Conc]), K, sigma)[0]
        np.testing.assert_allclose(rust_result, py_result, rtol=1e-7)

    def test_temperature_range(self):
        """Test with realistic temperature range."""
        Temp = np.linspace(273.15, 323.15, 20)  # 0°C to 50°C
        deltaH = -45000.0
        deltaS = -80.0
        c_tot = 0.5
        scaler = 1.0

        rust_result = temp_isodesmic_model(Temp, deltaH, deltaS, c_tot, scaler)
        py_result = py_isodesmic.temp_isodesmic_model(Temp, deltaH, deltaS, c_tot, scaler)
        np.testing.assert_allclose(rust_result, py_result, rtol=1e-8, atol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
