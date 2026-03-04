"""
Tests for isodesmic polymerization models.
"""

import numpy as np
import matplotlib.pyplot as plt
import pytest

from sp_fitting_models.models import (
    isodesmic_model,
    temp_isodesmic_model,
    inv_isodesmic_model,
)


def test_isodesmic_inverse_consistency():
    """
    Test that the isodesmic model and its inverse are consistent.
    """
    c_tot = np.linspace(1, 1000, 1000) * 1e-6  # Total concentrations from 1 uM to 1000 uM
    K = 1e6

    # Calculate aggregation
    agg = isodesmic_model(c_tot, K)
    print(f"Aggregation range: {agg.min():.2e} to {agg.max():.2e}")
    monomer_conc = (1 - agg) * c_tot
    print(f"Monomer concentration range: {monomer_conc.min():.2e} M to {monomer_conc.max():.2e} M")

    # Reverse calculation
    c_tot_calculated = inv_isodesmic_model(monomer_conc, K)

    # Check consistency
    assert np.allclose(c_tot, c_tot_calculated, rtol=1e-6), "Isodesmic model inverse is inconsistent"
    print("✓ Isodesmic inverse consistency test passed")


def test_temp_isodesmic_model():
    """
    Test temperature-dependent isodesmic model.
    """
    temps = np.linspace(280, 400, 100)
    deltaH = -96000
    deltaS = -195
    concentrations = [1e-6, 5e-6, 10e-6]

    fig, ax = plt.subplots(figsize=(8, 6))

    for c in concentrations:
        agg = temp_isodesmic_model(
            Temp=temps,
            deltaH=deltaH,
            deltaS=deltaS,
            c_tot=c,
            scaler=1.0,
        )
        ax.plot(temps - 273.15, agg, label=f"Conc={c*1e6:.1f} µM")

    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Aggregation")
    ax.set_title("Temperature-Dependent Isodesmic Model")
    ax.legend()
    ax.grid(True, alpha=0.3)

    print("✓ Temperature-dependent isodesmic model test completed")
    return fig


def test_isodesmic_high_temp_rounding_error():
    """
    Investigation of rounding error effects in isodesmic model at low aggregation (high temperature).

    At high temperatures, K decreases, leading to lower aggregation.
    This test measures numerical stability in the low-aggregation regime.
    """
    R = 8.314
    deltaH = -96000
    deltaS = -263
    c_tot = 1e-6

    # Temperature progression: 300 K → 500 K (high temperature where agg → 0)
    temps = np.linspace(300, 500, 200)

    aggs = []
    K_vals = []

    for temp in temps:
        agg = temp_isodesmic_model(
            Temp=np.array([temp]),
            deltaH=deltaH,
            deltaS=deltaS,
            c_tot=c_tot,
            scaler=1.0,
        )
        aggs.append(agg[0])

        K = np.exp(-deltaH / (R * temp) + deltaS / R)
        K_vals.append(K)

    aggs = np.array(aggs)
    K_vals = np.array(K_vals)

    # Identify low aggregation regime (agg < 1e-4)
    low_agg_mask = aggs < 1e-4
    temps_low_agg = temps[low_agg_mask]
    aggs_low_agg = aggs[low_agg_mask]
    K_low_agg = K_vals[low_agg_mask]

    # Analysis
    print("\n=== Isodesmic Model: High-Temperature Rounding Error Analysis ===")
    print(f"Temperature range: {temps.min():.1f} K to {temps.max():.1f} K")
    print(f"Aggregation range: {aggs.min():.2e} to {aggs.max():.2e}")
    print(f"K range: {K_vals.min():.2e} to {K_vals.max():.2e}")

    if len(aggs_low_agg) > 0:
        print(f"\nLow aggregation regime (agg < 1e-4):")
        print(f"  Temperature range: {temps_low_agg.min():.1f} K to {temps_low_agg.max():.1f} K")
        print(f"  Aggregation range: {aggs_low_agg.min():.2e} to {aggs_low_agg.max():.2e}")
        print(f"  K range: {K_low_agg.min():.2e} to {K_low_agg.max():.2e}")

        # Check monotonicity (agg should decrease monotonically with increasing T as K decreases)
        agg_diff = np.diff(aggs_low_agg)
        non_monotonic = np.sum(agg_diff > 0)
        print(f"  Non-monotonic changes: {non_monotonic}")

        # Check if aggregation values become unnaturally small (potential underflow)
        underflow_threshold = 1e-15
        underflow_count = np.sum(aggs_low_agg < underflow_threshold)
        print(f"  Values below {underflow_threshold}: {underflow_count}")

        # Relative changes in low aggregation regime
        rel_changes = np.abs(np.diff(aggs_low_agg) / aggs_low_agg[:-1])
        if len(rel_changes) > 0 and np.sum(aggs_low_agg[:-1] != 0) > 0:
            valid_rel_changes = rel_changes[aggs_low_agg[:-1] != 0]
            if len(valid_rel_changes) > 0:
                print(f"  Max relative change in agg: {valid_rel_changes.max():.2e}")
                print(f"  Mean relative change in agg: {valid_rel_changes.mean():.2e}")

        # Check for underflow in low aggregation regime

        underflow_count = np.sum(np.abs(aggs_low_agg) < 1e-14)
        if underflow_count > 0:
            print(f"  [WARNING] {underflow_count} potential underflow values (<1e-14) detected")

        # Check non-monotonicity
        if non_monotonic > len(aggs_low_agg) * 0.1:
            print(f"  [WARNING] High proportion of non-monotonic changes ({non_monotonic/len(aggs_low_agg)*100:.1f}%)")

    # Verify aggregation is non-negative and <= 1
    # Note: non-positive values near zero indicate rounding error issues
    neg_count = np.sum(aggs < 0)
    if neg_count > 0:
        print(f"  [WARNING] {neg_count} negative aggregation values detected (rounding error)")

    # Physical constraints should still hold in absolute terms
    assert np.all(aggs <= 1.0), "Aggregation should not exceed 1"

    print("\n✓ Isodesmic model high-temperature rounding error test completed")


if __name__ == "__main__":
    test_isodesmic_inverse_consistency()
    fig = test_temp_isodesmic_model()
    test_isodesmic_high_temp_rounding_error()
    plt.show()
