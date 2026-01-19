"""
Thermal model fitting module.

Fits physics-based thermal model parameters from collected data using scipy.optimize.
Generates validation metrics and plots to assess model accuracy.
"""

from .cli import fit_mode
from .equations import ModelConfig, predict_temp
from .fitter import fit_thermal_model
from .validator import compute_validation_metrics
from .plotting import generate_validation_plots

__all__ = [
    "fit_mode",
    "ModelConfig",
    "predict_temp",
    "fit_thermal_model",
    "compute_validation_metrics",
    "generate_validation_plots",
]
