"""
Tests for isodesmic polymerization models.
"""

import numpy as np
import matplotlib.pyplot as plt

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


if __name__ == "__main__":
    test_isodesmic_inverse_consistency()
    fig = test_temp_isodesmic_model()
    plt.show()
