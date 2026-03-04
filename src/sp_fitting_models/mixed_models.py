import numpy as np
import matplotlib.pyplot as plt

import sp_fitting_models.models as models
import sp_fitting_models.data as data
import numba as nb


@nb.jit(nopython=True)
def inv_coop_iso_model(c_monomer: np.ndarray, K_iso: float, K_coop: float, sigma: float) -> np.ndarray:
    """
    Calculate the tolal concentration of the species in a mixed cooperative and isodesmic model from the monomer concentration, the equilibrium constants and the cooperativity parameter.

    The two pathway share the same monomer. The total concentration is the sum of the contributions from the two pathways minus the monomer concentration to avoid double counting.

    Parameters
    ----------
    c_monomer : np.ndarray
        The concentration of the monomer.
    K_iso : float
        The equilibrium constant for the isodesmic model.
    K_coop : float
        The equilibrium constant for the cooperative model.
    sigma : float
        The cooperativity parameter for the cooperative model.

    Returns
    -------
    np.ndarray
        The total concentration of the species.
    """
    return (
        models.inv_isodesmic_model(c_monomer, K_iso)
        + models.inv_cooperative_model(c_monomer, K_coop, sigma)
        - c_monomer
    )


@nb.jit(nopython=True)
def coop_iso_model(Conc: np.ndarray, K_iso: float, K_coop: float, sigma: float, num_itr=100) -> np.ndarray:
    """
    Calculate the aggregation from the total concentration of the species in a mixed cooperative and isodesmic model from the equilibrium constants and the cooperativity parameter.
    Parameters
    ----------
    Conc : np.ndarray
        The total concentration of the species.
    K_iso : float
        The equilibrium constant for the isodesmic model.
    K_coop : float
        The equilibrium constant for the cooperative model.
    sigma : float
        The cooperativity parameter for the cooperative model.

    Returns
    -------
    np.ndarray
        The aggregation of the species.
    """

    # inv_coop_iso_model is monotonically increasing.
    # but this function is short range and we can use the total concentration as an upper bound for the monomer concentration.
    x_low = np.zeros_like(Conc)
    x_high = min(1 / K_iso, 1 / K_coop) * np.ones_like(Conc)
    # bisection method to find the monomer concentration that corresponds to the total concentration.
    x_mid = np.zeros_like(Conc)
    for _ in range(num_itr):
        x_mid = (x_low + x_high) / 2
        f_mid = inv_coop_iso_model(x_mid, K_iso, K_coop, sigma) - Conc
        f_low = inv_coop_iso_model(x_low, K_iso, K_coop, sigma) - Conc

        # If f_mid and f_low have the same sign, then the root is in the upper half of the interval.
        # Otherwise, the root is in the lower half of the interval.
        mask = f_mid * f_low < 0
        x_high[mask] = x_mid[mask]
        x_low[~mask] = x_mid[~mask]

    if np.any(np.abs(inv_coop_iso_model(x_mid, K_iso, K_coop, sigma) - Conc) > 1e-8):
        raise ValueError("Bisection method did not converge to the correct solution.")

    return 1 - x_mid / Conc


@nb.jit(nopython=True)
def temp_coop_iso_model(
    Temp: np.ndarray,
    deltaH_iso: float,
    deltaS_iso: float,
    deltaH_coop: float,
    deltaS_coop: float,
    deltaHnuc_coop: float,
    c_tot: float,
    scaler: float,
) -> np.ndarray:
    R = 8.314
    K_iso = np.exp(-deltaH_iso / (R * Temp) + deltaS_iso / R)
    K_coop = np.exp(-deltaH_coop / (R * Temp) + deltaS_coop / R)
    sigma = np.exp(-deltaHnuc_coop / (R * Temp))

    aggs = np.zeros_like(Temp)
    for i in range(len(Temp)):
        agg = coop_iso_model(np.array([c_tot]), K_iso[i], K_coop[i], sigma[i])
        aggs[i] = agg[0] * scaler
    return aggs


def test_mixed():
    c_tot = np.linspace(1, 1000, 1000) * 1e-6  # Total concentrations from 1 uM to 1000 uM
    K_iso = 1e6
    K_coop = 1e5
    sigma = 0.01

    agg = coop_iso_model(c_tot, K_iso, K_coop, sigma)
    c_tot_calculated = inv_coop_iso_model((1 - agg) * c_tot, K_iso, K_coop, sigma)

    # plot
    fig, ax = plt.subplots()
    ax.scatter(c_tot, c_tot_calculated, label="Calculated vs True", color="blue")
    ax.plot(c_tot, c_tot, label="Ideal", color="red", linestyle="--")
    ax.set_xlabel("True Total Concentration (M)")
    ax.set_ylabel("Calculated Total Concentration (M)")
    ax.legend()
    plt.show()


def test_temp_mixed():
    co_deltaH = -96000
    co_deltaS = -180
    co_deltaHnuc = 100000
    co_scaler = 1.0

    iso_deltaH = -96000
    iso_deltaS = -195

    concentrations = [1e-6, 2e-6, 5e-6, 1e-5]
    temps = np.linspace(280, 400, 200)
    for c in concentrations:
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
        plt.plot(temps, agg, label=f"Conc={c} M")
    plt.xlabel("Temperature (K)")
    plt.ylabel("Aggregation")


def generate_synthetic_data() -> list[data.TempVsAggData]:
    # Generate synthetic data for testing the fitting of the cooperative model to data at different temperatures.
    co_deltaH = -96000
    co_deltaS = -180
    co_deltaHnuc = 100000
    co_scaler = 1.0

    iso_deltaH = -96000
    iso_deltaS = -195

    concentrations = [1e-6, 2e-6, 5e-6, 1e-5]
    temps = np.linspace(280, 400, 200)
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
            scaler=co_scaler,
        )
        data_list.append(data.TempVsAggData(temp=temps, agg=agg, concentration=c))
    return data_list


if __name__ == "__main__":
    # test_mixed()

    # draw synthetic data
    data_list = generate_synthetic_data()

    fig, ax = plt.subplots()
    for d in data_list:
        ax.plot(d.temp, d.agg, label=f"Conc={d.concentration} (data)")
    ax.set_xlabel("Temperature (K)")

    ax.set_ylabel("Aggregation")
    ax.legend()
    plt.show()
