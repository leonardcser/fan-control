"""
Thermal model fitting module (ML-based).
"""

from .cli import fit_mode
from .train import ThermalModel

__all__ = ["fit_mode", "ThermalModel"]
