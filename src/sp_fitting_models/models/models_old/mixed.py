"""
Mixed cooperative-isodesmic polymerization models.

These models describe systems where both cooperative and isodesmic pathways
coexist, sharing the same monomer pool.
"""

import numpy as np
import numba as nb

from .isodesmic import inv_isodesmic_model
from .cooperative import inv_cooperative_model
from .utils import R


@nb.jit(nopython=True)
def inv_coop_iso_model(c_monomer: np.ndarray, K_iso: float, K_coop: float, sigma: float) -> np.ndarray:
    """
    Calculate the total concentration in a mixed cooperative-isodesmic model (inverse).

    The two pathways share the same monomer. The total concentration is the sum of
    the contributions from both pathways minus the monomer concentration to avoid
    double counting.

    Parameters
    ----------
    c_monomer : np.ndarray
        The concentration of the monomer.
    K_iso : float
        The equilibrium constant for the isodesmic pathway.
    K_coop : float
        The equilibrium constant for the cooperative pathway.
    sigma : float
        The cooperativity parameter for the cooperative pathway.

    Returns
    -------
    np.ndarray
        The total concentration of the species.
    """
    return inv_isodesmic_model(c_monomer, K_iso) + inv_cooperative_model(c_monomer, K_coop, sigma) - c_monomer


@nb.jit(nopython=True)
def coop_iso_model(Conc: np.ndarray, K_iso: float, K_coop: float, sigma: float, num_itr: int = 100) -> np.ndarray:
    """
    Calculate the aggregation from total concentration in a mixed model.

    Uses bisection method to find the monomer concentration that corresponds
    to the given total concentration.

    Parameters
    ----------
    Conc : np.ndarray
        The total concentration of the species.
    K_iso : float
        The equilibrium constant for the isodesmic pathway.
    K_coop : float
        The equilibrium constant for the cooperative pathway.
    sigma : float
        The cooperativity parameter for the cooperative pathway.
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
    # inv_coop_iso_model is monotonically increasing
    # Use bisection to find monomer concentration
    x_low = np.zeros_like(Conc)
    x_high = min(1 / K_iso, 1 / K_coop) * np.ones_like(Conc)

    x_mid = np.zeros_like(Conc)
    for _ in range(num_itr):
        x_mid = (x_low + x_high) / 2
        f_mid = inv_coop_iso_model(x_mid, K_iso, K_coop, sigma) - Conc
        f_low = inv_coop_iso_model(x_low, K_iso, K_coop, sigma) - Conc

        # If f_mid and f_low have the same sign, root is in upper half
        # Otherwise, root is in lower half
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
    """
    Calculate the mixed model aggregation based on temperature-dependent parameters.

    Parameters
    ----------
    Temp : np.ndarray
        Temperature in Kelvin.
    deltaH_iso : float
        Enthalpy change for isodesmic pathway (J/mol).
    deltaS_iso : float
        Entropy change for isodesmic pathway (J/(mol·K)).
    deltaH_coop : float
        Enthalpy change for cooperative pathway (J/mol).
    deltaS_coop : float
        Entropy change for cooperative pathway (J/(mol·K)).
    deltaHnuc_coop : float
        Nucleation enthalpy penalty for cooperative pathway (J/mol).
    c_tot : float
        Total concentration (M).
    scaler : float
        Scaling factor for the output.

    Returns
    -------
    np.ndarray
        Mixed model aggregation values.
    """
    K_iso = np.exp(-deltaH_iso / (R * Temp) + deltaS_iso / R)
    K_coop = np.exp(-deltaH_coop / (R * Temp) + deltaS_coop / R)
    sigma = np.exp(-deltaHnuc_coop / (R * Temp))

    aggs = np.zeros_like(Temp)
    for i in range(len(Temp)):
        agg = coop_iso_model(np.array([c_tot]), K_iso[i], K_coop[i], sigma[i])
        aggs[i] = agg[0] * scaler
    return aggs
