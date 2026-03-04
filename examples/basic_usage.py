"""
Basic examples of using sp_fitting_models.

This script demonstrates the basic usage of different polymerization models
and how to visualize their behavior.
"""

import numpy as np
import matplotlib.pyplot as plt

from sp_fitting_models.models import (
    temp_isodesmic_model,
    temp_cooperative_model,
    temp_coop_iso_model,
)


def example_isodesmic():
    """Example: Temperature-dependent isodesmic model."""
    print("Example 1: Isodesmic Model")

    temps = np.linspace(280, 400, 200)
    concentrations = [1e-6, 5e-6, 10e-6]

    deltaH = -96000  # J/mol
    deltaS = -195  # J/(mol·K)

    fig, ax = plt.subplots(figsize=(10, 6))

    for c in concentrations:
        agg = temp_isodesmic_model(
            Temp=temps,
            deltaH=deltaH,
            deltaS=deltaS,
            c_tot=c,
            scaler=1.0,
        )
        ax.plot(temps - 273.15, agg, label=f"{c*1e6:.1f} µM", linewidth=2)

    ax.set_xlabel("Temperature (°C)", fontsize=12)
    ax.set_ylabel("Aggregation", fontsize=12)
    ax.set_title("Isodesmic Model", fontsize=14)
    ax.legend(title="Concentration", fontsize=10)
    ax.grid(True, alpha=0.3)

    return fig


def example_cooperative():
    """Example: Temperature-dependent cooperative model."""
    print("Example 2: Cooperative Model")

    temps = np.linspace(280, 400, 200)
    concentrations = [1e-6, 5e-6, 10e-6]

    deltaH = -96000  # J/mol
    deltaS = -180  # J/(mol·K)
    deltaHnuc = 100000  # J/mol (nucleation penalty)

    fig, ax = plt.subplots(figsize=(10, 6))

    for c in concentrations:
        agg = temp_cooperative_model(
            Temp=temps,
            deltaH=deltaH,
            deltaS=deltaS,
            deltaHnuc=deltaHnuc,
            c_tot=c,
            scaler=1.0,
        )
        ax.plot(temps - 273.15, agg, label=f"{c*1e6:.1f} µM", linewidth=2)

    ax.set_xlabel("Temperature (°C)", fontsize=12)
    ax.set_ylabel("Aggregation", fontsize=12)
    ax.set_title("Cooperative Model (Sigmoidal)", fontsize=14)
    ax.legend(title="Concentration", fontsize=10)
    ax.grid(True, alpha=0.3)

    return fig


def example_mixed():
    """Example: Temperature-dependent mixed model."""
    print("Example 3: Mixed Cooperative-Isodesmic Model")

    temps = np.linspace(280, 400, 200)
    concentrations = [1e-6, 5e-6, 10e-6]

    # Cooperative pathway
    co_deltaH = -96000
    co_deltaS = -180
    co_deltaHnuc = 100000

    # Isodesmic pathway
    iso_deltaH = -96000
    iso_deltaS = -195

    fig, ax = plt.subplots(figsize=(10, 6))

    for c in concentrations:
        agg = temp_coop_iso_model(
            Temp=temps,
            deltaH_iso=iso_deltaH,
            deltaS_iso=iso_deltaS,
            deltaH_coop=co_deltaH,
            deltaS_coop=co_deltaS,
            deltaHnuc_coop=co_deltaHnuc,
            c_tot=c,
            scaler=1.0,
        )
        ax.plot(temps - 273.15, agg, label=f"{c*1e6:.1f} µM", linewidth=2)

    ax.set_xlabel("Temperature (°C)", fontsize=12)
    ax.set_ylabel("Aggregation", fontsize=12)
    ax.set_title("Mixed Model (Two Competing Pathways)", fontsize=14)
    ax.legend(title="Concentration", fontsize=10)
    ax.grid(True, alpha=0.3)

    return fig


def compare_models():
    """Compare all three models side by side."""
    print("Example 4: Comparing All Models")

    temps = np.linspace(280, 400, 200)
    c = 5e-6  # 5 µM concentration

    # Parameters
    deltaH = -96000
    deltaS_iso = -195
    deltaS_coop = -180
    deltaHnuc = 100000

    # Calculate aggregations
    agg_iso = temp_isodesmic_model(temps, deltaH, deltaS_iso, c, 1.0)
    agg_coop = temp_cooperative_model(temps, deltaH, deltaS_coop, deltaHnuc, c, 1.0)
    agg_mixed = temp_coop_iso_model(temps, deltaH, deltaS_iso, deltaH, deltaS_coop, deltaHnuc, c, 1.0)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(temps - 273.15, agg_iso, label="Isodesmic", linewidth=2)
    ax.plot(temps - 273.15, agg_coop, label="Cooperative", linewidth=2)
    ax.plot(temps - 273.15, agg_mixed, label="Mixed", linewidth=2, linestyle="--")

    ax.set_xlabel("Temperature (°C)", fontsize=12)
    ax.set_ylabel("Aggregation", fontsize=12)
    ax.set_title(f"Model Comparison (Conc = {c*1e6:.1f} µM)", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    return fig


if __name__ == "__main__":
    # Run all examples
    fig1 = example_isodesmic()
    fig2 = example_cooperative()
    # fig3 = example_mixed()
    # fig4 = compare_models()

    plt.show()
