"""
Supramolecular Polymer Fitting Models

This module provides various models for fitting supramolecular polymerization data,
including isodesmic, cooperative, and mixed models.
"""

from .isodesmic import isodesmic_model, temp_isodesmic_model, inv_isodesmic_model
from .cooperative import (
    cooperative_model,
    temp_cooperative_model,
    inv_cooperative_model,
)
from .mixed import (
    coop_iso_model,
    temp_coop_iso_model,
    inv_coop_iso_model,
)
from .utils import solve_cubic_vectorized, R

__all__ = [
    # Isodesmic models
    "isodesmic_model",
    "temp_isodesmic_model",
    "inv_isodesmic_model",
    # Cooperative models
    "cooperative_model",
    "temp_cooperative_model",
    "inv_cooperative_model",
    # Mixed models
    "coop_iso_model",
    "temp_coop_iso_model",
    "inv_coop_iso_model",
    # Utilities
    "solve_cubic_vectorized",
    "R",
]
