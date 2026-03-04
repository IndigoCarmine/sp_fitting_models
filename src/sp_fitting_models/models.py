from typing import Callable

import numpy as np
import numpy.typing as npt

import numba as nb

R = 8.314


@nb.jit(nopython=True)
def inv_isodesmic_model(c_monomer: np.ndarray, K: float) -> np.ndarray:
    """
    Calculate the tolal concentration of the species in an isodesmic model from the monomer concentration and the equilibrium constant.
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
    """
    if np.any(K * c_monomer > 1):
        raise ValueError("K * c_monomer must be less than 1 for the isodesmic model.")
    return c_monomer / (1 - K * c_monomer) ** 2


@nb.jit(nopython=True)
def inv_cooperative_model(c_monomer: np.ndarray, K: float, sigma: float) -> np.ndarray:
    """
    Calculate the tolal concentration of the species in a cooperative model from the monomer concentration, the equilibrium constant and the cooperativity parameter.
    Parameters
    ----------
    c_monomer : np.ndarray
        The concentration of the monomer.
    K : float
        The equilibrium constant for the cooperative model.
    sigma : float
        The cooperativity parameter for the cooperative model.

    Returns
    -------
    np.ndarray
        The total concentration of the species.
    """
    if K == 0:
        return c_monomer

    cK = K * c_monomer
    if np.any(cK >= 1):
        raise ValueError("K * c_monomer must be less than 1 for the cooperative model.")
    return c_monomer + sigma / K * (cK * (2 - cK)) / (1 - cK) ** 2


_model_function = Callable[[np.ndarray, float, float], np.ndarray]


@nb.jit(nopython=True)
def solve_cubic_vectorized(a, b, c, d, x_low, x_high, max_iter=50) -> npt.NDArray[np.float64]:
    """
    Solve a x^3 + b x^2 + c x + d = 0 on the interval [x_low, x_high]
    using fully vectorized bisection.
    Args:
        a, b, c, d: Coefficients of the cubic equation.
        x_low, x_high: Bounds for the root search.
        max_iter: Maximum number of iterations for bisection.
    Returns:
        npt.NDArray[np.float64]: Approximated roots within the specified bounds.

    Raises:
        ValueError: If the root is not bracketed for some elements.
        RuntimeError: If the bisection does not converge within the maximum number of iterations.
    """

    # Shape unification
    a = np.asarray(a)
    b = np.asarray(b)
    c = np.asarray(c)
    d = np.asarray(d)

    # Initial bounds
    xl: npt.NDArray[np.float64] = np.asarray(x_low, dtype=np.float64)
    xr: npt.NDArray[np.float64] = np.asarray(x_high, dtype=np.float64)

    # Evaluate cubic
    def f(x):
        return a * x**3 + b * x**2 + c * x + d

    fl = f(xl)
    fr = f(xr)
    mask_valid = fl * fr <= 0
    if not np.all(mask_valid):
        raise ValueError("Root not bracketed for some elements.")

    # Bisection method loop
    for _ in range(max_iter):
        xm = 0.5 * (xl + xr)
        fm = f(xm)

        left_mask = fl * fm <= 0
        xr[left_mask] = xm[left_mask]
        xl[~left_mask] = xm[~left_mask]
        fl = f(xl)

    if not np.all(np.abs(xr - xl) < 1e-12):
        raise RuntimeError("Bisection did not converge within the maximum number of iterations.")
    return 0.5 * (xl + xr)


@nb.jit(nopython=True)
def cooperative_model(
    Conc: float | npt.NDArray[np.number],
    K: float | np.number,
    sigma: float | np.number,
    scaler: float | np.number = 1,
) -> float | npt.NDArray[np.number]:
    """
    Calculates the rate of supramolecular polymer formation based on a cooperative binding model.
    Args:
        Conc (float | npt.NDArray[np.number]): Total concentration of the substrate.
        K (float | np.number): Equilibrium constant.
        sigma (float | np.number): Cooperativity factor.
        scaler (float | np.number): Scaling factor for the output.
    Returns:
        float | npt.NDArray[np.number]: Rate of supramolecular polymer formation.
    """
    scaled_conc = K * Conc
    a = 1 - sigma
    b = -(2 * a + scaled_conc)
    c = 1 + 2 * scaled_conc
    d = -scaled_conc
    # x^3 + bx^2 + cx + d = 0
    # max concentration of monomer (scaled) is [0,1]
    # because of convergence condition
    x_low = np.zeros_like(scaled_conc)
    x_high = np.ones_like(scaled_conc)

    mono_conc = solve_cubic_vectorized(a, b, c, d, x_low, x_high)
    mono_conc = mono_conc / K
    # ans = 1- mono_conc/Conc
    ans = 1 - np.divide(mono_conc, Conc, out=np.ones_like(mono_conc), where=Conc != 0)

    ans = [x if not np.isnan(x) else 0 for x in ans]
    ans = np.array(ans)

    return ans * scaler


@nb.jit(nopython=True)
def isodesmic_model(X: np.number | npt.NDArray[np.number], K: np.number | float | int):
    """
    Calculates the fraction of monomer in an isodesmic polymerization model.
    Args:
        X (np.number | npt.NDArray[np.number]): Concentration of monomer (e.g., UV/Vis absorbance).
        K (np.number | float | int): Equilibrium constant.
    Returns:
        np.number | npt.NDArray[np.number]: Fraction of monomer.
    """
    B = K * X
    return (2 * B + 1 - (4 * B + 1) ** 0.5) / (2 * B)


@nb.jit(nopython=True)
def temp_cooperative_model(
    Temp: np.ndarray, deltaH: float, deltaS: float, deltaHnuc: float, c_tot: float, scaler: float = 1
) -> np.ndarray:
    """
    Calculates the cooperative binding based on temperature-dependent parameters.
    Args:
        Temp: Temperature.
        deltaH: Enthalpy change.
        deltaS: Entropy change.
        deltaHnuc: Nucleation enthalpy change.
        c_tot: Total concentration.
        scaler: Scaling factor.
    Returns:
        np.ndarray: Cooperative binding values.
    """

    K = np.exp(-deltaH / (R * Temp) + deltaS / R)
    sigma = np.exp(-deltaHnuc / (R * Temp))

    return cooperative_model(c_tot, K, sigma, scaler=scaler)


@nb.jit(nopython=True)
def temp_isodesmic_model(Temp, deltaH, deltaS, c_tot, scaler=1) -> np.ndarray:
    """
    Calculates the isodesmic binding based on temperature-dependent parameters.
    Args:
        Temp: Temperature.
        deltaH: Enthalpy change.
        deltaS: Entropy change.
        c_tot: Total concentration.
        scaler: Scaling factor.
    Returns:
        np.ndarray: Isodesmic binding values.
    """

    K = np.exp(-deltaH / (R * Temp) + deltaS / R)

    return isodesmic_model(c_tot, K) * scaler
