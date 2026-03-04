"""
Cooperative polymerization models.

The cooperative model includes a nucleation step with a different equilibrium constant
than the elongation steps, leading to sigmoidal aggregation curves.
"""

import numpy as np
import numpy.typing as npt
import numba as nb

from .utils import R, solve_cubic_vectorized


@nb.jit(nopython=True)
def inv_cooperative_model(c_monomer: np.ndarray, K: float, sigma: float) -> np.ndarray:
    """
    Calculate the total concentration from monomer concentration (inverse model).

    Parameters
    ----------
    c_monomer : np.ndarray
        The concentration of the monomer.
    K : float
        The equilibrium constant for the cooperative model.
    sigma : float
        The cooperativity parameter (nucleation penalty).

    Returns
    -------
    np.ndarray
        The total concentration of the species.

    Raises
    ------
    ValueError
        If K * c_monomer >= 1 (convergence condition violated).
    """
    if K == 0:
        return c_monomer

    cK = K * c_monomer
    if np.any(cK >= 1):
        raise ValueError("K * c_monomer must be less than 1 for the cooperative model.")
    return c_monomer + sigma / K * (cK**2 * (2 - cK)) / (1 - cK) ** 2


# @nb.jit(nopython=True)
def cooperative_model(
    Conc: float | npt.NDArray[np.number],
    K: float | np.number | npt.NDArray[np.number],
    sigma: float | np.number | npt.NDArray[np.number],
    scaler: float | np.number = 1,
) -> float | npt.NDArray[np.number]:
    """
    Calculate the rate of supramolecular polymer formation based on a cooperative binding model.

    Parameters
    ----------
    Conc : float | npt.NDArray[np.number]
        Total concentration of the substrate.
    K : float | np.number | npt.NDArray[np.number]
        Equilibrium constant.
    sigma : float | np.number | npt.NDArray[np.number]
        Cooperativity factor (nucleation penalty, 0 < sigma < 1).
    scaler : float | np.number, optional
        Scaling factor for the output (default: 1).

    Returns
    -------
    float | npt.NDArray[np.number]
        Fraction of supramolecular polymer formation.
    """
    scaled_conc = K * Conc
    a = 1 - sigma
    b = -(2 * a + scaled_conc)
    c = 1 + 2 * scaled_conc
    d = -scaled_conc
    # Solve: a*x^3 + b*x^2 + c*x + d = 0
    # max concentration of monomer (scaled) is [0,1]
    # because of convergence condition
    x_low = np.zeros_like(scaled_conc)
    x_high = np.ones_like(scaled_conc)

    try:
        mono_conc = solve_cubic_vectorized(a, b, c, d, x_low, x_high)
    except (ValueError, RuntimeError):
        # If root finding fails (e.g., high temp, low aggregation), assume no aggregation
        return np.zeros_like(scaled_conc) * scaler
    mono_conc = mono_conc / K
    # Aggregation = 1 - monomer/total
    ans = 1 - np.divide(mono_conc, Conc, out=np.ones_like(mono_conc), where=Conc != 0)

    ans = [x if not np.isnan(x) else 0 for x in ans]
    ans = np.array(ans)

    return ans * scaler


# @nb.jit(nopython=True)
def temp_cooperative_model(
    Temp: npt.NDArray[np.number],
    deltaH: float,
    deltaS: float,
    deltaHnuc: float,
    c_tot: float | npt.NDArray[np.number],
    scaler: float = 1,
) -> float | npt.NDArray[np.number]:
    """
    Calculate the cooperative aggregation based on temperature-dependent parameters.

    Parameters
    ----------
    Temp : np.ndarray
        Temperature in Kelvin.
    deltaH : float
        Enthalpy change for elongation (J/mol).
    deltaS : float
        Entropy change for elongation (J/(mol·K)).
    deltaHnuc : float
        Nucleation enthalpy penalty (J/mol).
    c_tot : float
        Total concentration (M).
    scaler : float, optional
        Scaling factor for the output (default: 1).

    Returns
    -------
    np.ndarray
        Cooperative aggregation values.
    """
    K = np.exp(-deltaH / (R * Temp) + deltaS / R)
    sigma = np.exp(-deltaHnuc / (R * Temp))

    return cooperative_model(c_tot, K, sigma, scaler=scaler)
