"""
Supramolecular Polymer Fitting Models

A Python library for fitting supramolecular polymerization data with various models
including isodesmic, cooperative, and mixed cooperative-isodesmic models.
"""

from .data import TempVsAggData
from . import models
from . import fitting

__version__ = "0.1.0"

__all__ = [
    "TempVsAggData",
    "models",
    "fitting",
]


def main() -> None:
    print("Hello from sp-fitting-models!")
    print(f"Version: {__version__}")
