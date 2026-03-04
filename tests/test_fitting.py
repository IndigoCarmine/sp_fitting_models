"""
Tests for fitting models to experimental data.
"""

import numpy as np
import matplotlib.pyplot as plt
import lmfit as lm

from sp_fitting_models.data import TempVsAggData
from sp_fitting_models.models import temp_cooperative_model, temp_coop_iso_model
from sp_fitting_models.fitting import objective_temp_cooperative, objective_temp_coop_iso


def generate_synthetic_cooperative_data() -> list[TempVsAggData]:
    """
    Generate synthetic data for testing the cooperative model fitting.
    """
    temps = np.linspace(280, 400, 100)
    concentrations = [50e-6, 100e-6]  # 50 and 100 µM

    # True parameters
    deltaH_true = -96000
    deltaS_true = -180
    deltaHnuc_true = 100000
    scaler_true = 1.0

    data_list = []
    for c in concentrations:
        agg = temp_cooperative_model(
            Temp=temps,
            deltaH=deltaH_true,
            deltaS=deltaS_true,
            deltaHnuc=deltaHnuc_true,
            c_tot=c,
            scaler=scaler_true,
        )
        # Add noise
        agg = np.asarray(agg)  # Ensure agg is ndarray
        agg += np.random.normal(0, 0.005, size=agg.shape)
        data_list.append(TempVsAggData(temp=temps, agg=agg, concentration=c))

    return data_list


def test_cooperative_fitting():
    """
    Test fitting the cooperative model to synthetic data.
    """
    print("\n=== Testing Cooperative Model Fitting ===")

    # Generate synthetic data
    data_list = generate_synthetic_cooperative_data()

    # Initial parameter guesses
    params = lm.Parameters()
    params.add("deltaH", value=-100000, min=-200000, max=0)
    params.add("deltaS", value=-180, min=-400, max=0)
    params.add("deltaHnuc", value=50000, min=0, max=200000)
    params.add("scaler", value=1.0, min=0.5, max=1.5)

    # Fit
    minner = lm.Minimizer(objective_temp_cooperative, params, fcn_args=(data_list,))
    result = minner.minimize()

    print(lm.fit_report(result))

    # Plot results
    fig, ax = plt.subplots(figsize=(10, 6))
    for d in data_list:
        ax.scatter(d.temp - 273.15, d.agg, label=f"Data {d.concentration*1e6:.0f} µM", alpha=0.6)

        fit_curve = temp_cooperative_model(
            Temp=d.temp,
            deltaH=result.params["deltaH"].value,  # type: ignore
            deltaS=result.params["deltaS"].value,  # type: ignore
            deltaHnuc=result.params["deltaHnuc"].value,  # type: ignore
            c_tot=d.concentration,
            scaler=result.params["scaler"].value,  # type: ignore
        )
        ax.plot(d.temp - 273.15, fit_curve, label=f"Fit {d.concentration*1e6:.0f} µM", linewidth=2)

    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Aggregation")
    ax.set_title("Cooperative Model Fitting")
    ax.legend()
    ax.grid(True, alpha=0.3)

    print("✓ Cooperative model fitting test completed")
    return fig, result


def generate_synthetic_mixed_data() -> list[TempVsAggData]:
    """
    Generate synthetic data for testing the mixed model fitting.
    """
    temps = np.linspace(280, 400, 100)
    concentrations = [1e-6, 5e-6, 10e-6]

    # True parameters
    co_deltaH = -96000
    co_deltaS = -180
    co_deltaHnuc = 100000
    iso_deltaH = -96000
    iso_deltaS = -195
    scaler = 1.0

    data_list = []
    for c in concentrations:
        agg = temp_coop_iso_model(
            Temp=temps,
            deltaH_iso=iso_deltaH,
            deltaS_iso=iso_deltaS,
            deltaH_coop=co_deltaH,
            deltaS_coop=co_deltaS,
            deltaHnuc_coop=co_deltaHnuc,
            c_tot=c,
            scaler=scaler,
        )
        # Add noise
        agg = np.asarray(agg)  # Ensure agg is ndarray
        agg += np.random.normal(0, 0.01, size=agg.shape)
        data_list.append(TempVsAggData(temp=temps, agg=agg, concentration=c))

    return data_list


def test_mixed_fitting():
    """
    Test fitting the mixed cooperative-isodesmic model to synthetic data.
    """
    print("\n=== Testing Mixed Model Fitting ===")

    # Generate synthetic data
    data_list = generate_synthetic_mixed_data()

    # Initial parameter guesses
    params = lm.Parameters()
    params.add("deltaH_iso", value=-100000, min=-200000, max=0)
    params.add("deltaS_iso", value=-200, min=-400, max=0)
    params.add("deltaH_coop", value=-100000, min=-200000, max=0)
    params.add("deltaS_coop", value=-180, min=-400, max=0)
    params.add("deltaHnuc_coop", value=50000, min=0, max=200000)
    params.add("scaler", value=1.0, min=0.5, max=1.5)

    # Fit
    minner = lm.Minimizer(objective_temp_coop_iso, params, fcn_args=(data_list,))
    result = minner.minimize()

    print(lm.fit_report(result))

    # Plot results
    fig, ax = plt.subplots(figsize=(10, 6))
    for d in data_list:
        ax.scatter(d.temp - 273.15, d.agg, label=f"Data {d.concentration*1e6:.0f} µM", alpha=0.6)

        fit_curve = temp_coop_iso_model(
            Temp=d.temp,
            deltaH_iso=result.params["deltaH_iso"].value,  # type: ignore
            deltaS_iso=result.params["deltaS_iso"].value,  # type: ignore
            deltaH_coop=result.params["deltaH_coop"].value,  # type: ignore
            deltaS_coop=result.params["deltaS_coop"].value,  # type: ignore
            deltaHnuc_coop=result.params["deltaHnuc_coop"].value,  # type: ignore
            c_tot=d.concentration,
            scaler=result.params["scaler"].value,  # type: ignore
        )
        ax.plot(d.temp - 273.15, fit_curve, label=f"Fit {d.concentration*1e6:.0f} µM", linewidth=2)

    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Aggregation")
    ax.set_title("Mixed Model Fitting")
    ax.legend()
    ax.grid(True, alpha=0.3)

    print("✓ Mixed model fitting test completed")
    return fig, result


if __name__ == "__main__":
    # Test cooperative fitting
    fig1, result1 = test_cooperative_fitting()

    # Test mixed fitting
    fig2, result2 = test_mixed_fitting()

    plt.show()
