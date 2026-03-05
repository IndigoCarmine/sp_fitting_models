import numpy as np
import numpy.typing as npt
from sp_fitting_models._core import (
    coop_iso_model as _coop_iso_model,
    temp_coop_iso_model as _temp_coop_iso_model,
)
from sp_fitting_models.models.isodesmic import inv_isodesmic_model
from sp_fitting_models.models.cooperative import inv_cooperative_model


def inv_coop_iso_model(
    c_monomer: npt.NDArray[np.number],
    K_iso: float,
    K_coop: float,
    sigma: float,
) -> npt.NDArray[np.number]:
    """
    Calculate total concentration from monomer concentration (inverse mixed model).
    
    The two pathways share the same monomer. The total concentration is the sum of
    the contributions from both pathways minus the monomer concentration to avoid
    double counting.
    """
    c_monomer = np.asarray(c_monomer, dtype=float)
    return inv_isodesmic_model(c_monomer, K_iso) + inv_cooperative_model(c_monomer, K_coop, sigma) - c_monomer


def coop_iso_model(
    Conc: float | npt.NDArray[np.number],
    K_iso: float | np.number,
    K_coop: float | np.number,
    sigma: float | np.number,
    num_itr: int = 100,
) -> float | npt.NDArray[np.number]:
    """
    Calculate the aggregation from total concentration in a mixed cooperative-isodesmic model
    (bisection method).

    Parameters
    ----------
    Conc : float | npt.NDArray[np.number]
        The total concentration of the species.
    K_iso : float | np.number
        The equilibrium constant for the isodesmic pathway.
    K_coop : float | np.number
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
        return _coop_iso_model(float(Conc), float(K_iso), float(K_coop), float(sigma), num_itr)
    else:
        # Array case
        return np.array(
            [_coop_iso_model(float(c), float(K_iso), float(K_coop), float(sigma), num_itr) for c in Conc.flat]
        )


def temp_coop_iso_model(
    Temp: npt.NDArray[np.number],
    deltaH_iso: float,
    deltaS_iso: float,
    deltaH_coop: float,
    deltaS_coop: float,
    deltaHnuc_coop: float,
    c_tot: float,
    scaler: float = 1.0,
) -> npt.NDArray[np.number]:
    """
    Calculate the mixed cooperative-isodesmic aggregation based on temperature-dependent
    parameters (bisection method).

    Parameters
    ----------
    Temp : np.ndarray
        Temperature in Kelvin.
    deltaH_iso : float
        Enthalpy change for elongation in isodesmic pathway (J/mol).
    deltaS_iso : float
        Entropy change for elongation in isodesmic pathway (J/(mol·K)).
    deltaH_coop : float
        Enthalpy change for elongation in cooperative pathway (J/mol).
    deltaS_coop : float
        Entropy change for elongation in cooperative pathway (J/(mol·K)).
    deltaHnuc_coop : float
        Nucleation enthalpy penalty for cooperative pathway (J/mol).
    c_tot : float
        Total concentration (M).
    scaler : float, optional
        Scaling factor for the output (default: 1).

    Returns
    -------
    np.ndarray
        Mixed cooperative-isodesmic aggregation values.
    """
    Temp = np.asarray(Temp, dtype=float)
    try:
        result = _temp_coop_iso_model(
            Temp.tolist(),
            float(deltaH_iso),
            float(deltaS_iso),
            float(deltaH_coop),
            float(deltaS_coop),
            float(deltaHnuc_coop),
            float(c_tot),
            float(scaler),
        )
        return np.array(result)
    except ValueError:
        from .models_old.mixed import temp_coop_iso_model as _temp_coop_iso_model_old

        try:
            return _temp_coop_iso_model_old(Temp, deltaH_iso, deltaS_iso, deltaH_coop, deltaS_coop, deltaHnuc_coop, c_tot, scaler)
        except ValueError:
            return np.zeros_like(Temp, dtype=float) * float(scaler)
