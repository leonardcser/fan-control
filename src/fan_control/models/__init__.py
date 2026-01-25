"""
Dynamic Thermal Model Registry.

Provides a registry pattern for different model implementations:
- physics: Thermal RC network model
- pinn: Physics-Informed Neural Network (PyTorch)
- gbr: Gradient Boosting with monotonic constraints
- gp: Gaussian Process with uncertainty (GPyTorch)

Usage:
    from fan_control.models import get_model, register_model

    # Get a model by type
    model = get_model("gbr", config)

    # Register a custom model
    @register_model("custom")
    class CustomModel(DynamicThermalModel):
        ...
"""

from typing import Dict, Type, Any, Optional

from .base import DynamicThermalModel

# Model registry
_MODEL_REGISTRY: Dict[str, Type[DynamicThermalModel]] = {}


def register_model(name: str):
    """
    Decorator to register a model class in the registry.

    Args:
        name: Unique identifier for the model type

    Example:
        @register_model("gbr")
        class GradientBoostingModel(DynamicThermalModel):
            ...
    """
    def decorator(cls: Type[DynamicThermalModel]) -> Type[DynamicThermalModel]:
        if name in _MODEL_REGISTRY:
            raise ValueError(f"Model '{name}' is already registered")
        _MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def get_model(model_type: str, config: Dict[str, Any]) -> DynamicThermalModel:
    """
    Factory function to create a model instance by type.

    Args:
        model_type: Model type identifier (physics, pinn, gbr, gp)
        config: Model configuration dictionary

    Returns:
        Instantiated model

    Raises:
        ValueError: If model_type is not registered
    """
    if model_type not in _MODEL_REGISTRY:
        available = ", ".join(sorted(_MODEL_REGISTRY.keys()))
        raise ValueError(
            f"Unknown model type '{model_type}'. Available: {available}"
        )

    model_cls = _MODEL_REGISTRY[model_type]
    return model_cls(config)


def list_models() -> list[str]:
    """Return list of registered model types."""
    return sorted(_MODEL_REGISTRY.keys())


def load_model(
    model_type: str,
    path: str,
    config: Optional[Dict[str, Any]] = None,
) -> DynamicThermalModel:
    """
    Load a saved model from disk.

    Args:
        model_type: Model type identifier
        path: Path to saved model directory
        config: Optional config override

    Returns:
        Loaded model instance
    """
    from pathlib import Path as PathLib

    if model_type not in _MODEL_REGISTRY:
        available = ", ".join(sorted(_MODEL_REGISTRY.keys()))
        raise ValueError(
            f"Unknown model type '{model_type}'. Available: {available}"
        )

    model_cls = _MODEL_REGISTRY[model_type]
    return model_cls.load(PathLib(path), config)


# Import model implementations to trigger registration
# These imports must come after the registry is defined
from . import gbr  # noqa: E402, F401
from . import physics  # noqa: E402, F401
from . import pinn  # noqa: E402, F401
from . import gp  # noqa: E402, F401

# Public API
__all__ = [
    "DynamicThermalModel",
    "register_model",
    "get_model",
    "list_models",
    "load_model",
]
