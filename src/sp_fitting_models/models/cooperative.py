import numpy as np
import numpy.typing as npt
from sp_fitting_models._core import (
    cooperative_model as _cooperative_model,
    temp_cooperative_model as _temp_cooperative_model,
)


def inv_cooperative_model(c_monomer: npt.NDArray[np.number], K: float, sigma: float) -> npt.NDArray[np.number]:
    """
    Calculate total concentration from monomer concentration (inverse model).
    """
    c_monomer = np.asarray(c_monomer, dtype=float)
    if K == 0:
        return c_monomer
    cK = K * c_monomer
    if np.any(cK >= 1):
        raise ValueError("K * c_monomer must be less than 1 for the cooperative model.")
    return c_monomer + sigma / K * (cK**2 * (2 - cK)) / (1 - cK) ** 2


def cooperative_model(
    Conc: float | npt.NDArray[np.number],
    K: float | np.number,
    sigma: float | np.number,
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
        return _cooperative_model(float(Conc), float(K), float(sigma), num_itr)
    else:
        # Array case
        return np.array([_cooperative_model(float(c), float(K), float(sigma), num_itr) for c in Conc.flat])


def temp_cooperative_model(
    Temp: npt.NDArray[np.number],
    deltaH: float,
    deltaS: float,
    deltaHnuc: float,
    c_tot: float,
    scaler: float = 1.0,
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

    Returns
    -------
    np.ndarray
        Cooperative aggregation values.
    """
    Temp = np.asarray(Temp, dtype=float)
    try:
        result = _temp_cooperative_model(
            Temp.tolist(), float(deltaH), float(deltaS), float(deltaHnuc), float(c_tot), float(scaler)
        )
        return np.array(result)
    except ValueError:
        from .models_old.cooperative import temp_cooperative_model as _temp_cooperative_model_old

        try:
            return _temp_cooperative_model_old(Temp, deltaH, deltaS, deltaHnuc, c_tot, scaler)
        except ValueError:
            return np.zeros_like(Temp, dtype=float) * float(scaler)
