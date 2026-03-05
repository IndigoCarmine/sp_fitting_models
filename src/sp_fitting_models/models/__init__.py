"""
Supramolecular polymerization models using Rust backend via PyO3.

This module provides cooperative, isodesmic, and mixed polymerization models.
All heavy computation is delegated to the Rust backend for performance.
"""

from .isodesmic import (
    inv_isodesmic_model,
    isodesmic_model_direct,
    isodesmic_model,
    temp_isodesmic_model_direct,
    temp_isodesmic_model,
)
from .cooperative import (
    inv_cooperative_model,
    cooperative_model,
    temp_cooperative_model,
)
from .mixed import (
    inv_coop_iso_model,
    coop_iso_model,
    temp_coop_iso_model,
)

__all__ = [
    "inv_isodesmic_model",
    "isodesmic_model_direct",
    "isodesmic_model",
    "temp_isodesmic_model_direct",
    "temp_isodesmic_model",
    "inv_cooperative_model",
    "cooperative_model",
    "temp_cooperative_model",
    "inv_coop_iso_model",
    "coop_iso_model",
    "temp_coop_iso_model",
]
