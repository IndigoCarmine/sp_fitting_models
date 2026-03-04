import lmfit as lm
import numpy as np

import sp_fitting_models.models as models
import sp_fitting_models.data as data


def temp_cooperative_model(params: lm.Parameters, data: list[data.TempVsAggData]) -> np.ndarray:
    """
    objective function for fitting the cooperative model to data at different temperatures.
    Parameters
    ----------
    params : lm.Parameters
        The parameters to fit. Must contain 'deltaH', 'deltaS' and 'sigma'.
    data : list[data.XYData]
        The data to fit. Must be a list of data.XYData objects, one for each temperature.

    Returns
    -------
    np.ndarray
        The residuals between the model and the data.
    """

    nX = [len(d.temp) for d in data]
    nX = np.sum(nX)

    residual = np.zeros(nX)
    index = 0

    for i, d in enumerate(data):
        res = d.agg - models.temp_cooperative_model(
            Temp=d.temp,
            deltaH=params["deltaH"].value,
            deltaS=params["deltaS"].value,
            deltaHnuc=params["deltaHnuc"].value,
            c_tot=d.concentration,
            scaler=params["scaler"].value,
        )
        residual[index : index + len(res)] = res
        index += len(res)
    return residual
