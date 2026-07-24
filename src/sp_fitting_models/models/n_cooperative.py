import numpy as np
import numpy.typing as npt
from sp_fitting_models._core import (
    cooperative_model_n as _cooperative_model_n,
    temp_cooperative_model_n as _temp_cooperative_model_n,
)


def inv_cooperative_model_n(
    c_monomer: npt.NDArray[np.number], K: float, sigma: float, nuc_size: int
) -> npt.NDArray[np.number]:
    """
    Calculate total concentration from monomer concentration (inverse model).

    Nucleation-elongation model with an arbitrary nucleus size ``N = nuc_size`` (``N >= 2``):
    a species of size ``s`` carries the cooperativity penalty ``sigma**(min(s, N) - 1)``.
    With ``cK = K * c_monomer`` the total (monomer-unit) concentration is::

        c_tot = c_monomer
              + sigma**(N-1) / K * cK**2 * (2 - cK) / (1 - cK)**2                   # elongation
              + 1 / K * sum_{s=2}^{N-1} s * (sigma**(s-1) - sigma**(N-1)) * cK**s   # nucleus correction

    The multiplicity factor ``s`` (from summing ``s * [M_s]``) applies to the correction
    as well as the elongation term. For ``N = 2`` the correction sum is empty and this
    reduces exactly to the basic cooperative model (:func:`inv_cooperative_model`). This
    matches the Rust implementation used by the forward solver ``cooperative_model_n``.
    """
    if nuc_size < 2:
        raise ValueError("nuc_size must be at least 2 for the cooperative_n model.")
    c_monomer = np.asarray(c_monomer, dtype=float)
    if K == 0:
        return c_monomer
    cK = K * c_monomer
    if np.any(cK >= 1):
        raise ValueError("K * c_monomer must be less than 1 for the cooperative model.")

    sigma_pow_max = sigma ** (nuc_size - 1)  # sigma^(N-1)
    elongation = sigma_pow_max / K * (cK**2 * (2 - cK)) / (1 - cK) ** 2

    # Correction over the nucleus interior s = 2 .. N-1. The s = N term is zero, so the
    # loop stops at N-1; for N = 2 it does not run and the correction stays zero.
    #
    # The naive factor ``sigma**(s-1) - sigma_pow_max`` cancels catastrophically as
    # sigma -> 1 (both powers -> 1), losing up to ~7 digits near sigma = 1 - 1e-8.
    # Rewrite it as ``sigma**(s-1) * (1 - sigma**(N-s))`` and evaluate ``1 - sigma**m``
    # cancellation-free via ``-expm1(m * log(sigma))``, taking ``log(sigma) =
    # log1p(sigma - 1)`` so the log stays accurate near sigma = 1.
    ln_sigma = np.log1p(sigma - 1.0)  # = log(sigma), accurate for sigma ~ 1
    correction = np.zeros_like(c_monomer)
    for s in range(2, nuc_size):
        one_minus_sigma_pow = -np.expm1((nuc_size - s) * ln_sigma)  # 1 - sigma**(N-s)
        correction += s * sigma ** (s - 1) * one_minus_sigma_pow * cK**s
    correction /= K

    return c_monomer + elongation + correction


def cooperative_model_n(
    Conc: float | npt.NDArray[np.number],
    K: float | np.number,
    sigma: float | np.number,
    nuc_size: int,
    num_itr: int = 100,
) -> float | npt.NDArray[np.number]:
    """
    Calculate the aggregation from total concentration in a cooperative model (bisection method).

    Parameters
    ----------
    Conc : float | npt.NDArray[np.number]
        The total concentration of the species.
    K : float | np.number
        The equilibrium constant for the cooperative pathway.
    sigma : float | np.number
        The cooperativity parameter for the cooperative pathway.
    nuc_size : int
        The nucleation size for the cooperative model (must be at least 2).
    num_itr : int, optional
        Number of bisection iterations (default: 100).

    Returns
    -------
    float | npt.NDArray[np.number]
        The fraction of aggregated species.
    """
    Conc = np.asarray(Conc)

    if Conc.ndim == 0:
        # Scalar case
        return _cooperative_model_n(float(Conc), float(K), float(sigma), nuc_size, num_itr)
    else:
        # Array case
        return np.array([_cooperative_model_n(float(c), float(K), float(sigma), nuc_size, num_itr) for c in Conc.flat])


def temp_cooperative_model_n(
    Temp: npt.NDArray[np.number],
    deltaH: float,
    deltaS: float,
    deltaHnuc: float,
    c_tot: float,
    scaler: float = 1.0,
    nuc_size: int = 3,
) -> npt.NDArray[np.number]:
    """
    Calculate the cooperative aggregation based on temperature-dependent parameters (bisection method).

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
    nuc_size : int, optional
        The nucleation size for the cooperative model (must be at least 2).

    Returns
    -------
    np.ndarray
        Cooperative aggregation values.
    """
    Temp = np.asarray(Temp, dtype=float)
    try:
        result = _temp_cooperative_model_n(
            Temp.tolist(), float(deltaH), float(deltaS), float(deltaHnuc), float(c_tot), float(scaler), nuc_size
        )
        return np.array(result)
    except ValueError:
        raise ValueError(
            "cannot use temp_cooperative_model_n with the given parameters. Please check the parameters and try again."
        )


if __name__ == "__main__":
    c_monomers = np.linspace(1e-10, 1e-3, 100)  # Monomer concentrations from 1 nM to 1 mM
    deltaH = -96000  # J/mol
    deltaS = -180  # J/(mol·K)
    deltaHnuc = 10000  # J/mol (nucleation penalty)

    c_totals = inv_cooperative_model_n(
        c_monomers,
        K=np.exp(-deltaH / (8.314 * 300) + deltaS / 8.314),
        sigma=np.exp(-deltaHnuc / (8.314 * 300)),
        nuc_size=3,
    )

    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    plt.plot(c_monomers * 1e6, c_totals * 1e6, marker="o")
    plt.xlabel("Monomer Concentration (µM)")
    plt.ylabel("Total Concentration (µM)")
    plt.title("Inverse Cooperative Model")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.show()
