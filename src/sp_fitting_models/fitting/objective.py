"""
Objective functions for fitting models to experimental data using lmfit.
"""

import lmfit as lm
import numpy as np

from ..data import TempVsAggData
from ..models import temp_cooperative_model as model_temp_cooperative, temp_coop_iso_model


def objective_temp_cooperative(params: lm.Parameters, data: list[TempVsAggData]) -> np.ndarray:
    """
    Objective function for fitting the cooperative model to temperature-dependent data.

    Parameters
    ----------
    params : lm.Parameters
        The parameters to fit. Must contain 'deltaH', 'deltaS', 'deltaHnuc', and 'scaler'.
    data : list[TempVsAggData]
        The experimental data to fit. Each entry contains temperature, aggregation,
        and concentration information.

    Returns
    -------
    np.ndarray
        The residuals between the model and the data.
    """
    nX = sum(len(d.temp) for d in data)
    residual = np.zeros(nX)
    index = 0

    for d in data:
        model_pred = model_temp_cooperative(
            Temp=d.temp,
            deltaH=params["deltaH"].value,
            deltaS=params["deltaS"].value,
            deltaHnuc=params["deltaHnuc"].value,
            c_tot=d.concentration,
            scaler=params["scaler"].value,
        )
        res = d.agg - model_pred
        residual[index : index + len(res)] = res
        index += len(res)

    return residual


def temp_cooperative_model(params: lm.Parameters, data: list[TempVsAggData]) -> np.ndarray:
    """
    Backward-compatible alias for cooperative objective used in tests/examples.
    """
    return objective_temp_cooperative(params, data)


def objective_temp_coop_iso(params: lm.Parameters, data: list[TempVsAggData]) -> np.ndarray:
    """
    Objective function for fitting the mixed cooperative-isodesmic model to temperature data.

    Parameters
    ----------
    params : lm.Parameters
        The parameters to fit. Must contain:
        - 'deltaH_iso', 'deltaS_iso': isodesmic pathway parameters
        - 'deltaH_coop', 'deltaS_coop', 'deltaHnuc_coop': cooperative pathway parameters
        - 'scaler': scaling factor
    data : list[TempVsAggData]
        The experimental data to fit.

    Returns
    -------
    np.ndarray
        The residuals between the model and the data.
    """
    nX = sum(len(d.temp) for d in data)
    residual = np.zeros(nX)
    index = 0

    for d in data:
        model_pred = temp_coop_iso_model(
            Temp=d.temp,
            deltaH_iso=params["deltaH_iso"].value,
            deltaS_iso=params["deltaS_iso"].value,
            deltaH_coop=params["deltaH_coop"].value,
            deltaS_coop=params["deltaS_coop"].value,
            deltaHnuc_coop=params["deltaHnuc_coop"].value,
            c_tot=d.concentration,
            scaler=params["scaler"].value,
        )
        res = d.agg - model_pred
        residual[index : index + len(res)] = res
        index += len(res)

    return residual
