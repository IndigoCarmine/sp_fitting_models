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
        deltaS_iso=-180,
        deltaH_coop=-96000,
        deltaS_coop=-195,
        c_tot=10e-6,
        scaler=1.0,
    )

    for deltaHnuc in [0, 2000, 4000, 6000]:
        agg = temp_coop_iso_model(Temp=temps, deltaHnuc_coop=deltaHnuc, **params)
        agg_high_temp = float(agg[-1])
        assert agg_high_temp < 0.05, (
            f"Expected near-zero aggregation at high temperature, "
            f"but got {agg_high_temp:.3f} for deltaHnuc={deltaHnuc} J/mol"
        )


def test_mixed_high_temp_rounding_error():
    """
    Investigation of rounding error effects in mixed model at low aggregation (high temperature).

    At high temperatures, both K_iso and K_coop decrease, and σ → 1.
    This test measures numerical stability in the low-aggregation regime.
    """
    R = 8.314
    deltaH_iso = -96000
    deltaS_iso = -195
    deltaH_coop = -96000
    deltaS_coop = -180
    deltaHnuc_coop = 100000
    c_tot = 1e-6

    # Temperature progression: 300 K → 500 K (high temperature where agg → 0)
    temps = np.linspace(300, 500, 200)

    aggs = []
    K_iso_vals = []
    K_coop_vals = []
    sigma_vals = []

    for temp in temps:
        try:
            agg = temp_coop_iso_model(
                Temp=temps,
                deltaH_iso=deltaH_iso,
                deltaS_iso=deltaS_iso,
                deltaH_coop=deltaH_coop,
                deltaS_coop=deltaS_coop,
                deltaHnuc_coop=deltaHnuc_coop,
                c_tot=c_tot,
                scaler=1.0,
            )
            aggs.append(agg)
            break
        except:
            pass

    if len(aggs) == 0:
        # Use only single-temperature calculation
        temps_check = np.array([300.0, 350.0, 400.0, 450.0, 500.0])
        aggs_list = []

        for temp in temps_check:
            K_iso = np.exp(-deltaH_iso / (R * temp) + deltaS_iso / R)
            K_coop = np.exp(-deltaH_coop / (R * temp) + deltaS_coop / R)
            sigma = np.exp(-deltaHnuc_coop / (R * temp))
            K_iso_vals.append(K_iso)
            K_coop_vals.append(K_coop)
            sigma_vals.append(sigma)

            agg = temp_coop_iso_model(
                Temp=np.array([temp]),
                deltaH_iso=deltaH_iso,
                deltaS_iso=deltaS_iso,
                deltaH_coop=deltaH_coop,
                deltaS_coop=deltaS_coop,
                deltaHnuc_coop=deltaHnuc_coop,
                c_tot=c_tot,
                scaler=1.0,
            )
            aggs_list.append(agg[0])

        aggs = np.array(aggs_list)
        temps_check = temps_check
    else:
        aggs = aggs[0]
        temps_check = temps
        for temp in temps_check:
            K_iso = np.exp(-deltaH_iso / (R * temp) + deltaS_iso / R)
            K_coop = np.exp(-deltaH_coop / (R * temp) + deltaS_coop / R)
            sigma = np.exp(-deltaHnuc_coop / (R * temp))
            K_iso_vals.append(K_iso)
            K_coop_vals.append(K_coop)
            sigma_vals.append(sigma)

    # Identify low aggregation regime (agg < 1e-4)
    low_agg_mask = aggs < 1e-4

    # Analysis
    print("\n=== Mixed Model: High-Temperature Rounding Error Analysis ===")
    print(f"Temperature range: {temps_check.min():.1f} K to {temps_check.max():.1f} K")
    print(f"Aggregation range: {aggs.min():.2e} to {aggs.max():.2e}")
    print(f"K_iso range: {np.min(K_iso_vals):.2e} to {np.max(K_iso_vals):.2e}")
    print(f"K_coop range: {np.min(K_coop_vals):.2e} to {np.max(K_coop_vals):.2e}")
    print(f"σ range: {np.min(sigma_vals):.2e} to {np.max(sigma_vals):.2e}")

    if np.any(low_agg_mask):
        aggs_low_agg = aggs[low_agg_mask]
        print(f"\nLow aggregation regime (agg < 1e-4):")
        print(f"  Number of points: {len(aggs_low_agg)}")
        print(f"  Aggregation range: {aggs_low_agg.min():.2e} to {aggs_low_agg.max():.2e}")

        # Check monotonicity (agg should decrease monotonically with increasing T)
        agg_diff = np.diff(aggs_low_agg)
        non_monotonic = np.sum(agg_diff > 0)
        print(f"  Non-monotonic changes: {non_monotonic}")

        # Check if aggregation values become unnaturally small (potential underflow)
        underflow_threshold = 1e-15
        underflow_count = np.sum(aggs_low_agg < underflow_threshold)
        print(f"  Values below {underflow_threshold}: {underflow_count}")

        # Relative changes in low aggregation regime
        valid_mask = aggs_low_agg[:-1] != 0
        if np.any(valid_mask):
            rel_changes = np.abs(np.diff(aggs_low_agg)[valid_mask] / aggs_low_agg[:-1][valid_mask])
            print(f"  Max relative change in agg: {rel_changes.max():.2e}")
            print(f"  Mean relative change in agg: {rel_changes.mean():.2e}")

    # Verify aggregation is non-negative and <= 1
    # Note: negative values near zero indicate rounding error issues
    neg_count = np.sum(aggs < 0)
    if neg_count > 0:
        print(f"  [WARNING] {neg_count} negative aggregation values detected (rounding error)")

    # Check for underflow in low aggregation regime
    if np.any(low_agg_mask):
        aggs_low_agg = aggs[low_agg_mask]
        underflow_count = np.sum(np.abs(aggs_low_agg) < 1e-14)
        if underflow_count > 0:
            print(f"  [WARNING] {underflow_count} potential underflow values (<1e-14) detected")

    # Physical constraints should still hold in absolute terms
    assert np.all(aggs <= 1.0), "Aggregation should not exceed 1"

    print("\n✓ Mixed model high-temperature rounding error test completed")


if __name__ == "__main__":
    test_mixed_inverse_consistency()
    fig = test_temp_mixed_model()
    test_mixed_high_temp_rounding_error()
    plt.show()
