"""
Tests for cooperative polymerization models.
"""

import numpy as np
import matplotlib.pyplot as plt

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


if __name__ == "__main__":
    test_cooperative_inverse_consistency()
    fig = test_temp_cooperative_model()
    plt.show()
