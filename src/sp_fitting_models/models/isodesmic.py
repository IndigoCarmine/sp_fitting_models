"""
Isodesmic polymerization models using Rust backend.

The isodesmic model assumes that all equilibrium constants are equal for the addition
of a monomer to a growing polymer chain.
"""

import numpy as np
import numpy.typing as npt
from typing import overload
from sp_fitting_models._core import (
    isodesmic_model_direct as _isodesmic_model_direct,
    isodesmic_model as _isodesmic_model,
    temp_isodesmic_model_direct as _temp_isodesmic_model_direct,
    temp_isodesmic_model as _temp_isodesmic_model,
)


def inv_isodesmic_model(c_monomer: npt.NDArray[np.number], K: float) -> npt.NDArray[np.number]:
    """
    Calculate total concentration from monomer concentration (inverse model).
    """
    c_monomer = np.asarray(c_monomer, dtype=float)
    if np.any(K * c_monomer > 1):
        raise ValueError("K * c_monomer must be less than 1 for the isodesmic model.")
    return c_monomer / (1 - K * c_monomer) ** 2


def isodesmic_model_direct(
    X: float | np.number | npt.NDArray[np.number],
    K: float | np.number | int | npt.NDArray[np.number],
) -> float | npt.NDArray[np.number]:
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
    X = np.asarray(X)
    K = np.asarray(K)

    if X.ndim == 0 and K.ndim == 0:
        # Both scalar
        return _isodesmic_model_direct(float(X), float(K))
    else:
        # Broadcast to handle arrays
        X_broadcast, K_broadcast = np.broadcast_arrays(X, K)
        result = np.array(
            [_isodesmic_model_direct(float(x), float(k)) for x, k in zip(X_broadcast.flat, K_broadcast.flat)]
        )
        return result.reshape(X_broadcast.shape) if X_broadcast.ndim > 0 else result[0]


@overload
def isodesmic_model(
    Conc: float,
    K: float | np.number,
    num_itr: int = 100,
) -> float: ...


@overload
def isodesmic_model(
    Conc: npt.NDArray[np.number],
    K: float | np.number,
    num_itr: int = 100,
) -> npt.NDArray[np.float64]: ...


def isodesmic_model(
    Conc: float | npt.NDArray[np.number],
    K: float | np.number,
    num_itr: int = 100,
) -> float | npt.NDArray[np.float64]:
    """
    Calculate the aggregation from total concentration in an isodesmic model (bisection method).

    Parameters
    ----------
    Conc : float | npt.NDArray[np.number]
        The total concentration of the species.
    K : float | np.number
        The equilibrium constant for the isodesmic pathway.
    num_itr : int, optional
        Number of bisection iterations (default: 100).

    Returns
    -------
    float | npt.NDArray[np.number]
        The fraction of aggregated species.
    """
    Conc = np.asarray(Conc)

    if Conc.ndim == 0:
        return _isodesmic_model_direct(float(Conc), float(K))

    return np.array([_isodesmic_model_direct(float(c), float(K)) for c in Conc.flat], dtype=np.float64).reshape(
        Conc.shape
    )


def temp_isodesmic_model_direct(
    Temp: npt.NDArray[np.number],
    deltaH: float,
    deltaS: float,
    c_tot: float,
    scaler: float = 1.0,
) -> npt.NDArray[np.number]:
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
    Temp = np.asarray(Temp, dtype=float)
    result = _temp_isodesmic_model_direct(Temp.tolist(), float(deltaH), float(deltaS), float(c_tot), float(scaler))
    return np.array(result)


def temp_isodesmic_model(
    Temp: npt.NDArray[np.number],
    deltaH: float,
    deltaS: float,
    c_tot: float,
    scaler: float = 1.0,
) -> npt.NDArray[np.number]:
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
    Temp = np.asarray(Temp, dtype=float)
    result = _temp_isodesmic_model(Temp.tolist(), float(deltaH), float(deltaS), float(c_tot), float(scaler))
    return np.array(result)
