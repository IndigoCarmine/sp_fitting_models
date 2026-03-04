from dataclasses import dataclass
import numpy as np
import warnings


@dataclass
class TempVsAggData:
    temp: np.ndarray
    agg: np.ndarray

    concentration: float

    def __post_init__(self):
        if len(self.temp) != len(self.agg):
            raise ValueError("Temp and Agg must have the same length.")

        if np.any(self.agg is None) or np.any(self.temp is None):
            raise ValueError("Temp and Agg cannot contain None values.")

        if np.any(self.agg < 0) or np.any(self.temp < 0):
            warnings.warn(
                "Some temperature or aggregation values are negative. This may indicate an issue with the data."
            )

        # warnings
        if np.any(self.temp < 273.15):
            warnings.warn("some temperatures are below 0C. this temp is kelvin, so this is likely a mistake.")

        if np.any(self.agg > 1):
            warnings.warn(
                "some aggregation values are above 1. This may indicate an issue with the data or that the data is not normalized."
            )

        if self.concentration > 1e-3:
            warnings.warn(
                "concentration is above 1 mM. This value is Molar, so this may indicate an issue with the data or that the concentration is not in Molar units."
            )
