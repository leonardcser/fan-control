"""
Abstract base class for dynamic thermal models.
All models predict T_{k+1} = f(T_k, PWM_k, P_k, T_amb).
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import numpy as np
import pandas as pd


class DynamicThermalModel(ABC):
    """
    Abstract base class for dynamic thermal models.

    All models must implement:
    - train(): Train on historical data
    - predict_next(): Single-step prediction T_{k+1} given current state
    - predict_horizon(): Multi-step rollout over a control sequence
    - save()/load(): Model persistence
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize model with configuration.

        Args:
            config: Model configuration dict with hyperparameters
        """
        self.config = config

        # Feature configuration
        features = config["features"]
        self.state_features = features["state"]
        self.control_features = features["control"]
        self.disturbance_features = features["disturbance"]

        # Combined feature list for input
        self.input_features = (
            self.state_features + self.control_features + self.disturbance_features
        )

        # Targets (next-step temperatures)
        self.targets = features["targets"]

    @abstractmethod
    def train(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Train the model on historical data.

        The dataframe should contain columns for:
        - Current temperatures (T_cpu, T_gpu)
        - Fan PWM values (pwm2, pwm4, pwm5)
        - Power draw (P_cpu, P_gpu)
        - Ambient temperature (T_amb)
        - Next-step temperatures (T_cpu_next, T_gpu_next) or derived from consecutive rows

        Args:
            df: Training dataframe with time-series data

        Returns:
            Dict of training metrics (loss, rmse, etc.)
        """
        pass

    @abstractmethod
    def predict_next(
        self,
        T_k: np.ndarray,
        PWM: np.ndarray,
        P: np.ndarray,
        T_amb: float,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict next-step temperatures given current state.

        Args:
            T_k: Current temperatures [T_cpu, T_gpu] shape (2,) or (n, 2)
            PWM: Fan PWM values [pwm2, pwm4, pwm5] shape (3,) or (n, 3)
            P: Power draw [P_cpu, P_gpu] shape (2,) or (n, 2)
            T_amb: Ambient temperature (scalar or array)

        Returns:
            Tuple of:
            - T_next: Predicted next temperatures [T_cpu, T_gpu]
            - std: Optional uncertainty estimate (None if model doesn't support)
        """
        pass

    @abstractmethod
    def predict_horizon(
        self,
        T_0: np.ndarray,
        PWM_seq: np.ndarray,
        P_seq: np.ndarray,
        T_amb: float,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict temperature trajectory over a control horizon.

        Args:
            T_0: Initial temperatures [T_cpu, T_gpu] shape (2,)
            PWM_seq: PWM sequence shape (horizon, 3) - constant PWM if 1D
            P_seq: Power sequence shape (horizon, 2) - constant power if 1D
            T_amb: Ambient temperature (assumed constant over horizon)

        Returns:
            Tuple of:
            - T_trajectory: Predicted temperatures shape (horizon+1, 2) including T_0
            - std_trajectory: Optional uncertainty shape (horizon+1, 2) or None
        """
        pass

    @abstractmethod
    def save(self, path: Path) -> None:
        """
        Save the trained model to disk.

        Args:
            path: Directory path to save model artifacts
        """
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: Path, config: Optional[Dict[str, Any]] = None) -> "DynamicThermalModel":
        """
        Load a trained model from disk.

        Args:
            path: Directory path containing model artifacts
            config: Optional config to override saved config

        Returns:
            Loaded model instance
        """
        pass
