import numpy as np
import matplotlib.pyplot as plt
import lmfit as lm
from typing import Any, cast
from sp_fitting_models.fitting.objective import temp_cooperative_model
import sp_fitting_models.models as models
from sp_fitting_models.data import TempVsAggData


# Generate synthetic data for testing
def generate_synthetic_data() -> list[TempVsAggData]:
    temps = np.linspace(280, 400, 100)  # Temperatures from 280K to 320K (deg: 7C to 47C)
    concentrations = [50, 100]  # in uM
    concentrations = [x * 1e-6 for x in concentrations]  # Convert to Molar
    deltaH_true = -96000
    deltaS_true = -180
    deltaHnuc_true = 100000
    scaler_true = 1.0
    data_list = []
    for c in concentrations:
        agg = models.temp_cooperative_model(
            Temp=temps,
            deltaH=deltaH_true,
            deltaS=deltaS_true,
            deltaHnuc=deltaHnuc_true,
            c_tot=c,
            scaler=scaler_true,
        )
        # Add some noise
        agg += np.random.normal(0, 0.005, size=agg.shape)
        data_list.append(TempVsAggData(temp=temps, agg=agg, concentration=c))
    return data_list


def test_temp_cooperative_model_fit():
    # Generate synthetic data
    data_list = generate_synthetic_data()

    # Initial parameter guesses
    params = lm.Parameters()
    params.add("deltaH", value=-100000)
    params.add("deltaS", value=-180)
    params.add("deltaHnuc", value=50)
    params.add("scaler", value=1.0)

    # Fit
    minner = lm.Minimizer(temp_cooperative_model, params, fcn_args=(data_list,))
    result = cast(Any, minner.minimize())

    print(lm.fit_report(result))

    # Plot results
    fig, ax = plt.subplots()
    for d in data_list:
        ax.scatter(d.temp, d.agg, label=f"Conc={d.concentration} (data)")
        fit_curve = models.temp_cooperative_model(
            Temp=d.temp,
            deltaH=result.params["deltaH"].value,
            deltaS=result.params["deltaS"].value,
            deltaHnuc=result.params["deltaHnuc"].value,
            c_tot=d.concentration,
            scaler=result.params["scaler"].value,
        )
        ax.plot(d.temp, fit_curve, label=f"Conc={d.concentration} (fit)")
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Aggregation")
    ax.legend()
    plt.show()


def test_cooperative_model_inv_function():
    # Test that the cooperative model and its inverse function are consistent
    c_tot = np.linspace(1, 1000, 1000) * 1e-6  # Total concentrations from 1 uM to 1000 uM
    deltaH = -96000
    deltaS = -180
    deltaHnuc = 10000
    scaler = 1.0

    agg = np.array(
        [
            models.temp_cooperative_model(
                Temp=np.array([300]), deltaH=deltaH, deltaS=deltaS, deltaHnuc=deltaHnuc, c_tot=c, scaler=scaler
            )[0]
            for c in c_tot
        ]
    )
    monomerconc = (1 - agg) * c_tot

    R = 8.314
    Temp = 300
    K = np.exp(-deltaH / (R * Temp) + deltaS / R)
    sigma = np.exp(-deltaHnuc / (R * Temp))
    c_tot_calculated = models.inv_cooperative_model(monomerconc, K, sigma)

    # plot
    fig, ax = plt.subplots()
    ax.scatter(c_tot, c_tot_calculated, label="Calculated vs True", color="blue")
    ax.plot(c_tot, c_tot, label="Ideal", color="red", linestyle="--")
    ax.set_xlabel("True Total Concentration (M)")
    ax.set_ylabel("Calculated Total Concentration (M)")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    # test_cooperative_model_inv_function()
    # test_temp_cooperative_model_fit()

    # draw synthetic data
    data_list = generate_synthetic_data()
    fig, ax = plt.subplots()
    for d in data_list:
        ax.scatter(d.temp, d.agg, label=f"Conc={d.concentration} (data)")
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Aggregation")
    ax.legend()
    plt.show()
