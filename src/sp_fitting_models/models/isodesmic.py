"""
Isodesmic polymerization models.

The isodesmic model assumes that all equilibrium constants are equal for the addition
of a monomer to a growing polymer chain.
"""

import numpy as np
import numpy.typing as npt
import numba as nb

from .utils import R


@nb.jit(nopython=True)
def inv_isodesmic_model(c_monomer: np.ndarray, K: float) -> np.ndarray:
    """
    Calculate the total concentration from monomer concentration (inverse model).

    Parameters
    ----------
    c_monomer : np.ndarray
        The concentration of the monomer.
    K : float
        The equilibrium constant for the isodesmic model.

    Returns
    -------
    np.ndarray
        The total concentration of the species.

    Raises
    ------
    ValueError
        If K * c_monomer >= 1 (convergence condition violated).
    """
    if np.any(K * c_monomer > 1):
        raise ValueError("K * c_monomer must be less than 1 for the isodesmic model.")

    return c_monomer / (1 - K * c_monomer) ** 2


@nb.jit(nopython=True)
def isodesmic_model_direct(
    X: float | np.number | npt.NDArray[np.number], K: float | np.number | int | npt.NDArray[np.number]
):
    """
    Calculate the fraction of aggregated species in an isodesmic polymerization model (direct formula).

    Parameters
    ----------
    X : float | np.number | npt.NDArray[np.number]
        Total concentration.
    K : float | np.number | int | npt.NDArray[np.number]
        Equilibrium constant.

    Returns
    -------
    float | np.number | npt.NDArray[np.number]
        Fraction of aggregated species (1 - monomer fraction).
    """
    B = K * X
    # z = K * [M], where [M] is monomer concentration
    z = (2 * B + 1 - (4 * B + 1) ** 0.5) / (2 * B)
    # Aggregation fraction = 1 - [M]/X = 1 - z/B
    return 1 - z / B


@nb.jit(nopython=True)
def isodesmic_model(Conc: np.ndarray, K: float, num_itr: int = 100) -> np.ndarray:
    """
    Calculate the aggregation from total concentration in an isodesmic model (bisection method).

    Uses bisection method to find the monomer concentration that corresponds
    to the given total concentration.

    Parameters
    ----------
    Conc : np.ndarray
        The total concentration of the species.
    K : float
        The equilibrium constant for the isodesmic pathway.
    num_itr : int, optional
        Number of bisection iterations (default: 100).

    Returns
    -------
    np.ndarray
        The fraction of aggregated species.

    Raises
    ------
    ValueError
        If the bisection method does not converge.
    """
    # inv_isodesmic_model is monotonically increasing
    # Use bisection to find monomer concentration
    x_low = np.zeros_like(Conc)
    x_high = 1 / K * np.ones_like(Conc)

    x_mid = np.zeros_like(Conc)
    for _ in range(num_itr):
        x_mid = (x_low + x_high) / 2
        f_mid = inv_isodesmic_model(x_mid, K) - Conc
        f_low = inv_isodesmic_model(x_low, K) - Conc

        # If f_mid and f_low have the same sign, root is in upper half
        # Otherwise, root is in lower half
        mask = f_mid * f_low < 0
        x_high[mask] = x_mid[mask]
        x_low[~mask] = x_mid[~mask]

    if np.any(np.abs(inv_isodesmic_model(x_mid, K) - Conc) > 1e-8):
        raise ValueError("Bisection method did not converge to the correct solution.")

    return 1 - x_mid / Conc


@nb.jit(nopython=True)
def temp_isodesmic_model_direct(
    Temp: npt.NDArray[np.number], deltaH: float, deltaS: float, c_tot: float, scaler: float = 1
) -> np.ndarray:
    """
    Calculate the isodesmic aggregation based on temperature-dependent parameters (direct formula).

    Parameters
    ----------
    Temp : np.ndarray
        Temperature in Kelvin.
    deltaH : float
        Enthalpy change (J/mol).
    deltaS : float
        Entropy change (J/(mol·K)).
    c_tot : float
        Total concentration (M).
    scaler : float, optional
        Scaling factor for the output (default: 1).

    Returns
    -------
    np.ndarray
        Isodesmic aggregation values.
    """
    K = np.exp(-deltaH / (R * Temp) + deltaS / R)
    return isodesmic_model_direct(c_tot, K) * scaler


@nb.jit(nopython=True)
def temp_isodesmic_model(
    Temp: npt.NDArray[np.number], deltaH: float, deltaS: float, c_tot: float, scaler: float = 1
) -> np.ndarray:
    """
    Calculate the isodesmic aggregation based on temperature-dependent parameters (bisection method).

    Parameters
    ----------
    Temp : np.ndarray
        Temperature in Kelvin.
    deltaH : float
        Enthalpy change (J/mol).
    deltaS : float
        Entropy change (J/(mol·K)).
    c_tot : float
        Total concentration (M).
    scaler : float, optional
        Scaling factor for the output (default: 1).

    Returns
    -------
    np.ndarray
        Isodesmic aggregation values.
    """
    K = np.exp(-deltaH / (R * Temp) + deltaS / R)
    aggs = np.zeros_like(Temp)
    for i in range(len(Temp)):
        agg = isodesmic_model(np.array([c_tot]), K[i])
        aggs[i] = agg[0] * scaler
    return aggs
