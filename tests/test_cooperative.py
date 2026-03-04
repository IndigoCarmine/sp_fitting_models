"""
Tests for cooperative polymerization models.
"""

import numpy as np
import matplotlib.pyplot as plt
import pytest

from sp_fitting_models.models import (
    temp_cooperative_model,
    inv_cooperative_model,
)


def test_cooperative_inverse_consistency():
    """
    Test that the cooperative model and its inverse are consistent.
    """
    c_tot = np.linspace(1, 1000, 1000) * 1e-6  # Total concentrations from 1 uM to 1000 uM
    deltaH = -96000
    deltaS = -180
    deltaHnuc = 10000
    Temp = 300
    R = 8.314

    # Calculate K and sigma
    K = np.exp(-deltaH / (R * Temp) + deltaS / R)
    sigma = np.exp(-deltaHnuc / (R * Temp))

    # Calculate aggregation
    agg = [temp_cooperative_model(np.array([Temp]), deltaH, deltaS, deltaHnuc, c, scaler=1.0) for c in c_tot]
    agg = np.array(agg).flatten()
    monomer_conc = (1 - agg) * c_tot
    print(f"Aggregation range: {agg.min():.2e} to {agg.max():.2e}")
    print(f"Monomer concentration range: {monomer_conc.min():.2e} M to {monomer_conc.max():.2e} M")

    # Reverse calculation
    c_tot_calculated = inv_cooperative_model(monomer_conc, K, sigma)

    # Check consistency
    max_diff = np.max(np.abs(c_tot - c_tot_calculated) / c_tot)
    print(f"Maximum relative difference: {max_diff:.2e}")
    assert np.allclose(c_tot, c_tot_calculated, rtol=5e-2), "Cooperative model inverse is inconsistent"
    print("✓ Cooperative inverse consistency test passed")


def test_temp_cooperative_model():
    """
    Test temperature-dependent cooperative model.
    """
    temps = np.linspace(280, 400, 100)
    deltaH = -96000
    deltaS = -180
    deltaHnuc = 100000
    concentrations = np.array([10, 50, 100]) * 1e-6  # 10, 50, 100 µM

    fig, ax = plt.subplots(figsize=(8, 6))

    for c in concentrations:
        agg = temp_cooperative_model(
            Temp=temps,
            deltaH=deltaH,
            deltaS=deltaS,
            deltaHnuc=deltaHnuc,
            c_tot=c,
            scaler=1.0,
        )
        ax.plot(temps - 273.15, agg, label=f"Conc={c*1e6:.1f} µM")

    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Aggregation")
    ax.set_title("Temperature-Dependent Cooperative Model")
    ax.legend()
    ax.grid(True, alpha=0.3)

    print("✓ Temperature-dependent cooperative model test completed")
    return fig


def test_cooperative_high_temp_rounding_error():
    """
    Investigation of rounding error effects in cooperative model at low aggregation (high temperature).

    At high temperatures, the aggregation (agg) approaches zero due to σ → 1.
    This test measures numerical stability in the low-aggregation regime.
    """
    R = 8.314
    deltaH = -96000
    deltaS = -180
    deltaHnuc = 100000
    c_tot = 1e-6

    # Temperature progression: 300 K → 500 K (high temperature where agg → 0)
    temps = np.linspace(300, 500, 200)

    aggs = []
    K_vals = []
    sigma_vals = []

    for temp in temps:
        agg = temp_cooperative_model(np.array([temp]), deltaH, deltaS, deltaHnuc, c_tot, scaler=1.0)
        aggs.append(agg[0])

        K = np.exp(-deltaH / (R * temp) + deltaS / R)
        sigma = np.exp(-deltaHnuc / (R * temp))
        K_vals.append(K)
        sigma_vals.append(sigma)

    aggs = np.array(aggs)
    K_vals = np.array(K_vals)
    sigma_vals = np.array(sigma_vals)

    # Identify low aggregation regime (agg < 1e-4)
    low_agg_mask = aggs < 1e-4
    temps_low_agg = temps[low_agg_mask]
    aggs_low_agg = aggs[low_agg_mask]
    sigma_low_agg = sigma_vals[low_agg_mask]

    # Analysis
    print("\n=== Cooperative Model: High-Temperature Rounding Error Analysis ===")
    print(f"Temperature range: {temps.min():.1f} K to {temps.max():.1f} K")
    print(f"Aggregation range: {aggs.min():.2e} to {aggs.max():.2e}")
    print(f"σ range: {sigma_vals.min():.2e} to {sigma_vals.max():.2e}")

    if len(aggs_low_agg) > 0:
        print(f"\nLow aggregation regime (agg < 1e-4):")
        print(f"  Temperature range: {temps_low_agg.min():.1f} K to {temps_low_agg.max():.1f} K")
        print(f"  Aggregation range: {aggs_low_agg.min():.2e} to {aggs_low_agg.max():.2e}")
        print(f"  σ range: {sigma_low_agg.min():.2e} to {sigma_low_agg.max():.2e}")

        # Check monotonicity (agg should decrease monotonically with increasing T)
        agg_diff = np.diff(aggs_low_agg)
        non_monotonic = np.sum(agg_diff > 0)
        print(f"  Non-monotonic changes: {non_monotonic}")

        # Check if aggregation values become unnaturally small (potential underflow)
        underflow_threshold = 1e-15
        underflow_count = np.sum(aggs_low_agg < underflow_threshold)
        print(f"  Values below {underflow_threshold}: {underflow_count}")

        # Relative changes in low aggregation regime
        rel_changes = np.abs(np.diff(aggs_low_agg) / aggs_low_agg[:-1])
        print(f"  Max relative change in agg: {rel_changes.max():.2e}")
        print(f"  Mean relative change in agg: {rel_changes.mean():.2e}")

    # Verify aggregation is non-negative and <= 1
    # Note: negative values near zero indicate rounding error issues
    neg_count = np.sum(aggs < 0)
    if neg_count > 0:
        print(f"  [WARNING] {neg_count} negative aggregation values detected (rounding error)")

    # Check for underflow in low aggregation regime
    if len(aggs_low_agg) > 0:
        underflow_count = np.sum(np.abs(aggs_low_agg) < 1e-14)
        if underflow_count > 0:
            print(f"  [WARNING] {underflow_count} potential underflow values (<1e-14) detected")

        # Check non-monotonicity
        if non_monotonic > len(aggs_low_agg) * 0.1:
            print(f"  [WARNING] High proportion of non-monotonic changes ({non_monotonic/len(aggs_low_agg)*100:.1f}%)")

    # Physical constraints should still hold in absolute terms
    assert np.all(aggs <= 1.0), "Aggregation should not exceed 1"

    print("\n✓ Cooperative model high-temperature rounding error test completed")


if __name__ == "__main__":
    test_cooperative_inverse_consistency()
    fig = test_temp_cooperative_model()
    test_cooperative_high_temp_rounding_error()
    plt.show()
