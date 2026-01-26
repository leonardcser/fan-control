"""
Gradient Boosting Regressor thermal model with monotonic constraints.

Uses sklearn's HistGradientBoostingRegressor for fast, accurate predictions
with physics-informed monotonic constraints:
- Increasing power → increasing temperature (+1)
- Increasing fan speed → decreasing temperature (-1)
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from . import register_model
from .base import DynamicThermalModel

logger = logging.getLogger(__name__)


@register_model("gbr")
class GradientBoostingModel(DynamicThermalModel):
    """
    Gradient Boosting thermal model with monotonic constraints.

    Predicts T_{k+1} from (T_k, PWM_k, P_k, T_amb) using separate models
    for CPU and GPU temperatures.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # GBR-specific hyperparameters
        gbr_config = config["gbr"]
        self.n_estimators = gbr_config["n_estimators"]
        self.max_depth = gbr_config["max_depth"]
        self.learning_rate = gbr_config["learning_rate"]
        self.l2_regularization = gbr_config["l2_regularization"]
        self.random_state = gbr_config["random_state"]

        # Models (one per target)
        self.cpu_model: Optional[HistGradientBoostingRegressor] = None
        self.gpu_model: Optional[HistGradientBoostingRegressor] = None

        # Feature order for numpy predictions
        self.feature_names_in_: List[str] = []

    def _get_monotonic_constraints(self, feature_names: List[str]) -> List[int]:
        """
        Generate constraint vector for HistGradientBoostingRegressor.

        Returns:
            List of constraints: 1 (increasing), -1 (decreasing), 0 (none)
        """
        constraints = []
        for name in feature_names:
            if name.startswith("pwm"):
                # Increasing fan speed → lower temperature
                constraints.append(-1)
            elif name.startswith("P_"):
                # Increasing power → higher temperature
                constraints.append(1)
            elif name == "T_amb":
                # Higher ambient → higher temperature
                constraints.append(1)
            elif name.startswith("T_"):
                # Current temp → next temp (positive correlation)
                constraints.append(1)
            else:
                constraints.append(0)
        return constraints

    def _prepare_training_data(self, df: pd.DataFrame):
        """
        Prepare training data with next-step targets.

        If T_cpu_next/T_gpu_next columns exist, use them directly.
        Otherwise, derive from consecutive rows (assumes time-ordered data).

        Returns:
            Tuple of (X features, y_cpu targets, y_gpu targets)
        """
        # Check if next-step columns exist
        has_next = "T_cpu_next" in df.columns and "T_gpu_next" in df.columns

        if has_next:
            # Use provided next-step values
            X = df[self.input_features].copy()
            y_cpu = df["T_cpu_next"]
            y_gpu = df["T_gpu_next"]
        else:
            # Derive from consecutive rows
            # Shift targets back by 1 (row i predicts row i+1 temps)
            df_copy = df.copy()
            df_copy["T_cpu_next"] = df_copy["T_cpu"].shift(-1)
            df_copy["T_gpu_next"] = df_copy["T_gpu"].shift(-1)

            # Drop last row (no next value)
            df_copy = df_copy.iloc[:-1]

            X = df_copy[self.input_features].copy()
            y_cpu = df_copy["T_cpu_next"]
            y_gpu = df_copy["T_gpu_next"]

        return X, y_cpu, y_gpu

    def train(
        self, df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """
        Train GBR models for CPU and GPU temperature prediction.

        Args:
            df: Training dataframe with temperature/power/PWM columns
            val_df: Unused, for API compatibility

        Returns:
            Dict of training metrics
        """
        del val_df  # Unused
        # Determine available features
        available = [f for f in self.input_features if f in df.columns]
        self.feature_names_in_ = available

        constraints = self._get_monotonic_constraints(available)
        logger.info(f"Training features: {available}")
        logger.info(f"Monotonic constraints: {constraints}")

        # Common model parameters
        model_params = {
            "max_iter": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "l2_regularization": self.l2_regularization,
            "random_state": self.random_state,
            "early_stopping": True,
            "validation_fraction": 0.1,
            "n_iter_no_change": 10,
            "monotonic_cst": constraints,
        }

        # Prepare training data (data is already filtered by preprocessing)
        X, y_cpu, y_gpu = self._prepare_training_data(df)

        metrics = {}

        # Train CPU model
        if "T_cpu" in df.columns and len(X) > 0:
            logger.info(f"Training CPU model on {len(X)} samples...")
            self.cpu_model = HistGradientBoostingRegressor(**model_params)
            self.cpu_model.fit(X, y_cpu)

            # Training metrics
            y_pred = self.cpu_model.predict(X)
            metrics["cpu_rmse"] = float(np.sqrt(mean_squared_error(y_cpu, y_pred)))
            metrics["cpu_mae"] = float(mean_absolute_error(y_cpu, y_pred))
            metrics["cpu_r2"] = float(r2_score(y_cpu, y_pred))
            logger.info(
                f"CPU model: RMSE={metrics['cpu_rmse']:.2f}°C, "
                f"R2={metrics['cpu_r2']:.4f}"
            )

        # Train GPU model
        if "T_gpu" in df.columns and len(X) > 0:
            logger.info(f"Training GPU model on {len(X)} samples...")
            self.gpu_model = HistGradientBoostingRegressor(**model_params)
            self.gpu_model.fit(X, y_gpu)

            y_pred = self.gpu_model.predict(X)
            metrics["gpu_rmse"] = float(np.sqrt(mean_squared_error(y_gpu, y_pred)))
            metrics["gpu_mae"] = float(mean_absolute_error(y_gpu, y_pred))
            metrics["gpu_r2"] = float(r2_score(y_gpu, y_pred))
            logger.info(
                f"GPU model: RMSE={metrics['gpu_rmse']:.2f}°C, "
                f"R2={metrics['gpu_r2']:.4f}"
            )

        return metrics

    def predict_next(
        self,
        T_k: np.ndarray,
        PWM: np.ndarray,
        P: np.ndarray,
        T_amb: float,
        extra_features: Optional[Dict[str, float]] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict next-step temperatures.

        Args:
            T_k: Current temps [T_cpu, T_gpu]
            PWM: Fan values [pwm2, pwm4, pwm5]
            P: Power [P_cpu, P_gpu]
            T_amb: Ambient temperature
            extra_features: Optional dict of additional features (e.g., cpu_busy_pct)

        Returns:
            (T_next, None) - GBR doesn't provide uncertainty
        """
        # Build feature vector
        features = {}
        features["T_cpu"] = T_k[0]
        features["T_gpu"] = T_k[1]
        features["pwm2"] = PWM[0]
        features["pwm4"] = PWM[1]
        features["pwm5"] = PWM[2]
        features["P_cpu"] = P[0]
        features["P_gpu"] = P[1]
        features["T_amb"] = T_amb

        # Add extra features if provided (e.g., cpu_busy_pct for throttling)
        if extra_features:
            features.update(extra_features)

        # Extract in correct order
        X = np.array([[features.get(f, 0.0) for f in self.feature_names_in_]])

        # Predict
        T_cpu_next = self.cpu_model.predict(X)[0] if self.cpu_model else T_k[0]
        T_gpu_next = self.gpu_model.predict(X)[0] if self.gpu_model else T_k[1]

        return np.array([T_cpu_next, T_gpu_next]), None

    def predict_horizon(
        self,
        T_0: np.ndarray,
        PWM_seq: np.ndarray,
        P_seq: np.ndarray,
        T_amb: float,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict temperature trajectory over horizon.

        Args:
            T_0: Initial temps [T_cpu, T_gpu]
            PWM_seq: PWM sequence (horizon, 3) or (3,) for constant
            P_seq: Power sequence (horizon, 2) or (2,) for constant
            T_amb: Ambient temperature

        Returns:
            (trajectory, None) - shape (horizon+1, 2)
        """
        # Handle constant inputs
        PWM_seq = np.atleast_2d(PWM_seq)
        P_seq = np.atleast_2d(P_seq)

        horizon = max(len(PWM_seq), len(P_seq))

        # Expand constant inputs to full horizon
        if len(PWM_seq) == 1:
            PWM_seq = np.tile(PWM_seq, (horizon, 1))
        if len(P_seq) == 1:
            P_seq = np.tile(P_seq, (horizon, 1))

        # Initialize trajectory
        trajectory = np.zeros((horizon + 1, 2))
        trajectory[0] = T_0

        # Roll forward
        T_k = T_0.copy()
        for k in range(horizon):
            T_k, _ = self.predict_next(T_k, PWM_seq[k], P_seq[k], T_amb)
            trajectory[k + 1] = T_k

        return trajectory, None

    def save(self, path: Path) -> None:
        """Save model to directory."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save sklearn models
        if self.cpu_model:
            joblib.dump(self.cpu_model, path / "cpu_model.pkl")
        if self.gpu_model:
            joblib.dump(self.gpu_model, path / "gpu_model.pkl")

        # Save metadata
        metadata = {
            "model_type": "gbr",
            "feature_names": self.feature_names_in_,
            "config": self.config,
        }
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: Path, config: Optional[Dict[str, Any]] = None) -> "GradientBoostingModel":
        """Load model from directory."""
        path = Path(path)

        # Load metadata
        with open(path / "metadata.json") as f:
            metadata = json.load(f)

        # Use saved config or override
        model_config = config if config else metadata.get("config", {})
        model = cls(model_config)
        model.feature_names_in_ = metadata.get("feature_names", [])

        # Load sklearn models
        cpu_path = path / "cpu_model.pkl"
        gpu_path = path / "gpu_model.pkl"

        if cpu_path.exists():
            model.cpu_model = joblib.load(cpu_path)
        if gpu_path.exists():
            model.gpu_model = joblib.load(gpu_path)

        logger.info(f"Model loaded from {path}")
        return model
