"""
Tests for mixed cooperative-isodesmic polymerization models.
"""

import numpy as np
import matplotlib.pyplot as plt
import pytest

from sp_fitting_models.models import (
    coop_iso_model,
    temp_coop_iso_model,
    inv_coop_iso_model,
)
from sp_fitting_models.models import temp_isodesmic_model


def test_mixed_inverse_consistency():
    """
    Test that the mixed model and its inverse are consistent.
    """
    c_tot = np.linspace(1, 1000, 1000) * 1e-6  # Total concentrations from 1 uM to 1000 uM
    K_iso = 1e6
    K_coop = 1e5
    sigma = 0.01

    # Calculate aggregation
    agg = coop_iso_model(c_tot, K_iso, K_coop, sigma)
    monomer_conc = (1 - agg) * c_tot

    # Reverse calculation
    c_tot_calculated = inv_coop_iso_model(monomer_conc, K_iso, K_coop, sigma)

    # Check consistency
    assert np.allclose(c_tot, c_tot_calculated, rtol=1e-6), "Mixed model inverse is inconsistent"
    print("✓ Mixed model inverse consistency test passed")


def test_temp_mixed_model():
    """
    Test temperature-dependent mixed model.
    """
    temps = np.linspace(280, 400, 200)

    # Parameters
    co_deltaH = -96000
    co_deltaS = -180
    co_deltaHnuc = 100000
    co_scaler = 1.0

    iso_deltaH = -96000
    iso_deltaS = -195

    concentrations = [1e-6, 2e-6, 5e-6, 10e-6]

    fig, ax = plt.subplots(figsize=(10, 6))

    for c in concentrations:
        # Mixed model
        agg = temp_coop_iso_model(
            Temp=temps,
            deltaH_iso=iso_deltaH,
            deltaS_iso=iso_deltaS,
            deltaH_coop=co_deltaH,
            deltaS_coop=co_deltaS,
            deltaHnuc_coop=co_deltaHnuc,
            c_tot=c,
            scaler=co_scaler,
        )
        ax.plot(temps - 273.15, agg, label=f"Mixed {c*1e6:.1f} µM", linewidth=2)

        # Isodesmic only (for comparison)
        agg_iso = temp_isodesmic_model(
            Temp=temps,
            deltaH=iso_deltaH,
            deltaS=iso_deltaS,
            c_tot=c,
            scaler=co_scaler,
        )
        ax.plot(temps - 273.15, agg_iso, "--", alpha=0.5, linewidth=1)

    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Aggregation")
    ax.set_title("Temperature-Dependent Mixed Model\n(Solid: Mixed, Dashed: Isodesmic only)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    print("✓ Temperature-dependent mixed model test completed")
    return fig


@pytest.mark.xfail(reason="Known issue: low deltaHnuc keeps non-zero high-temperature aggregation in mixed model")
def test_mixed_high_temp_should_approach_zero_for_low_deltaHnuc():
    """
    Investigative test for the reported anomaly:
    For low deltaHnuc (0-6000 J/mol), high-temperature aggregation is expected to
    approach zero, but current implementation keeps substantial residual aggregation.
    """
    temps = np.linspace(280, 400, 200)

    params = dict(
        deltaH_iso=-96000,
        deltaS_iso=-195,
        deltaH_coop=-96000,
        deltaS_coop=-180,
        c_tot=1e-6,
        scaler=1.0,
    )

    for deltaHnuc in [0, 2000, 4000, 6000]:
        agg = temp_coop_iso_model(Temp=temps, deltaHnuc_coop=deltaHnuc, **params)
        agg_high_temp = float(agg[-1])
        assert agg_high_temp < 0.05, (
            f"Expected near-zero aggregation at high temperature, "
            f"but got {agg_high_temp:.3f} for deltaHnuc={deltaHnuc} J/mol"
        )


if __name__ == "__main__":
    test_mixed_inverse_consistency()
    fig = test_temp_mixed_model()
    plt.show()
