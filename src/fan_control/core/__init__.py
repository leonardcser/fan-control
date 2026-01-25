"""Core utilities and common modules for fan control."""

from .hardware import HardwareController, HardwareError
from .plotting import generate_all_plots

__all__ = [
    "HardwareController",
    "HardwareError",
    "generate_all_plots",
]
