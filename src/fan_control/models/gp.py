"""
Gaussian Process thermal model with uncertainty quantification.

Uses GPyTorch for scalable GP regression, providing:
- Mean predictions for temperature dynamics
- Uncertainty estimates (standard deviation)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

import torch
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls import ExactMarginalLogLikelihood


from . import register_model
from .base import DynamicThermalModel

logger = logging.getLogger(__name__)


class ThermalGP(ExactGP):
    """
    Exact Gaussian Process for thermal dynamics.

    Uses RBF kernel with automatic relevance determination (ARD).
    """

    def __init__(self, train_x, train_y, likelihood, input_dim: int):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=input_dim))

    def forward(self, x) -> MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


@register_model("gp")
class GaussianProcessModel(DynamicThermalModel):
    """
    Gaussian Process thermal model with uncertainty.

    Provides both mean predictions and confidence intervals,
    useful for robust MPC and safety-aware control.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # GP-specific config
        gp_config = config["gp"]
        self.training_iter = gp_config["training_iter"]
        self.learning_rate = gp_config["learning_rate"]
        self.max_train_samples = gp_config["max_train_samples"]

        # Device - GPyTorch requires CPU on macOS (MPS lacks linalg_qr support)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Input: [T_cpu, T_gpu, pwm2, pwm4, pwm5, P_cpu, P_gpu, T_amb]
        self.input_dim = 8

        # Separate GP for CPU and GPU
        self.cpu_model: Optional[ThermalGP] = None
        self.gpu_model: Optional[ThermalGP] = None
        self.cpu_likelihood: Optional[GaussianLikelihood] = None
        self.gpu_likelihood: Optional[GaussianLikelihood] = None

        # Store training data for GP (needed for predictions)
        self.train_x: Optional[torch.Tensor] = None
        self.train_y_cpu: Optional[torch.Tensor] = None
        self.train_y_gpu: Optional[torch.Tensor] = None

        # Normalization
        self.input_mean: Optional[np.ndarray] = None
        self.input_std: Optional[np.ndarray] = None
        self.cpu_mean: float = 0.0
        self.cpu_std: float = 1.0
        self.gpu_mean: float = 0.0
        self.gpu_std: float = 1.0

    def _normalize_input(self, X: np.ndarray) -> np.ndarray:
        if self.input_mean is None:
            return X
        return (X - self.input_mean) / (self.input_std + 1e-8)

    def _train_gp(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        name: str,
    ) -> Tuple[ThermalGP, GaussianLikelihood]:
        """Train a single GP model."""
        likelihood = GaussianLikelihood().to(self.device)
        model = ThermalGP(train_x, train_y, likelihood, self.input_dim).to(self.device)

        model.train()
        likelihood.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        mll = ExactMarginalLogLikelihood(likelihood, model)

        for i in range(self.training_iter):
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()

            if (i + 1) % 25 == 0:
                logger.debug(
                    f"{name} GP iter {i + 1}/{self.training_iter}, loss={loss.item():.4f}"
                )

        model.eval()
        likelihood.eval()

        return model, likelihood

    def train(self, df: pd.DataFrame) -> Dict[str, float]:
        """Train GP models for CPU and GPU."""
        # Prepare data
        if "T_cpu_next" not in df.columns:
            df = df.copy()
            df["T_cpu_next"] = df["T_cpu"].shift(-1)
            df["T_gpu_next"] = df["T_gpu"].shift(-1)
            df = df.iloc[:-1]

        # Subsample if too large (GP is O(n^3))
        if len(df) > self.max_train_samples:
            df = df.sample(n=self.max_train_samples, random_state=42)
            logger.info(
                f"Subsampled to {self.max_train_samples} points for GP training"
            )

        # Extract features
        X = df[
            ["T_cpu", "T_gpu", "pwm2", "pwm4", "pwm5", "P_cpu", "P_gpu", "T_amb"]
        ].values
        y_cpu = df["T_cpu_next"].values
        y_gpu = df["T_gpu_next"].values

        # Normalize inputs
        self.input_mean = X.mean(axis=0)
        self.input_std = X.std(axis=0)
        X_norm = self._normalize_input(X)

        # Normalize outputs
        self.cpu_mean = y_cpu.mean()
        self.cpu_std = y_cpu.std()
        self.gpu_mean = y_gpu.mean()
        self.gpu_std = y_gpu.std()

        y_cpu_norm = (y_cpu - self.cpu_mean) / (self.cpu_std + 1e-8)
        y_gpu_norm = (y_gpu - self.gpu_mean) / (self.gpu_std + 1e-8)

        # Convert to tensors
        self.train_x = torch.FloatTensor(X_norm).to(self.device)
        self.train_y_cpu = torch.FloatTensor(y_cpu_norm).to(self.device)
        self.train_y_gpu = torch.FloatTensor(y_gpu_norm).to(self.device)

        # Train CPU GP
        logger.info(f"Training CPU GP on {len(X)} samples...")
        self.cpu_model, self.cpu_likelihood = self._train_gp(
            self.train_x, self.train_y_cpu, "CPU"
        )

        # Train GPU GP
        logger.info(f"Training GPU GP on {len(X)} samples...")
        self.gpu_model, self.gpu_likelihood = self._train_gp(
            self.train_x, self.train_y_gpu, "GPU"
        )

        # Evaluate
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            cpu_pred = self.cpu_likelihood(self.cpu_model(self.train_x))
            gpu_pred = self.gpu_likelihood(self.gpu_model(self.train_x))

            cpu_mean = cpu_pred.mean.cpu().numpy() * self.cpu_std + self.cpu_mean
            gpu_mean = gpu_pred.mean.cpu().numpy() * self.gpu_std + self.gpu_mean

        cpu_rmse = float(np.sqrt(np.mean((cpu_mean - y_cpu) ** 2)))
        gpu_rmse = float(np.sqrt(np.mean((gpu_mean - y_gpu) ** 2)))

        metrics = {
            "cpu_rmse": cpu_rmse,
            "gpu_rmse": gpu_rmse,
            "cpu_mae": float(np.mean(np.abs(cpu_mean - y_cpu))),
            "gpu_mae": float(np.mean(np.abs(gpu_mean - y_gpu))),
            "n_train_samples": len(X),
        }

        logger.info(f"GP trained: CPU RMSE={cpu_rmse:.2f}°C, GPU RMSE={gpu_rmse:.2f}°C")
        return metrics

    def predict_next(
        self,
        T_k: np.ndarray,
        PWM: np.ndarray,
        P: np.ndarray,
        T_amb: float,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict next temperatures with uncertainty.

        Returns:
            (T_next, std) - mean prediction and standard deviation
        """
        if self.cpu_model is None or self.gpu_model is None:
            raise RuntimeError("Model not trained")

        # Build input
        X = np.array([[T_k[0], T_k[1], PWM[0], PWM[1], PWM[2], P[0], P[1], T_amb]])
        X_norm = self._normalize_input(X)
        X_tensor = torch.FloatTensor(X_norm).to(self.device)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            cpu_pred = self.cpu_likelihood(self.cpu_model(X_tensor))
            gpu_pred = self.gpu_likelihood(self.gpu_model(X_tensor))

            # Mean predictions
            cpu_mean = cpu_pred.mean.cpu().numpy()[0] * self.cpu_std + self.cpu_mean
            gpu_mean = gpu_pred.mean.cpu().numpy()[0] * self.gpu_std + self.gpu_mean

            # Standard deviation
            cpu_std = np.sqrt(cpu_pred.variance.cpu().numpy()[0]) * self.cpu_std
            gpu_std = np.sqrt(gpu_pred.variance.cpu().numpy()[0]) * self.gpu_std

        T_next = np.array([cpu_mean, gpu_mean])
        std = np.array([cpu_std, gpu_std])

        return T_next, std

    def predict_horizon(
        self,
        T_0: np.ndarray,
        PWM_seq: np.ndarray,
        P_seq: np.ndarray,
        T_amb: float,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict trajectory with uncertainty propagation.

        Note: Uncertainty grows over the horizon due to compounding.
        """
        PWM_seq = np.atleast_2d(PWM_seq)
        P_seq = np.atleast_2d(P_seq)

        horizon = max(len(PWM_seq), len(P_seq))

        if len(PWM_seq) == 1:
            PWM_seq = np.tile(PWM_seq, (horizon, 1))
        if len(P_seq) == 1:
            P_seq = np.tile(P_seq, (horizon, 1))

        trajectory = np.zeros((horizon + 1, 2))
        std_trajectory = np.zeros((horizon + 1, 2))
        trajectory[0] = T_0
        std_trajectory[0] = 0.0  # No uncertainty at t=0

        T_k = T_0.copy()
        cumulative_var = np.zeros(2)

        for k in range(horizon):
            T_next, std = self.predict_next(T_k, PWM_seq[k], P_seq[k], T_amb)
            trajectory[k + 1] = T_next

            # Propagate uncertainty (approximate - assumes independence)
            if std is not None:
                cumulative_var += std**2
                std_trajectory[k + 1] = np.sqrt(cumulative_var)

            T_k = T_next

        return trajectory, std_trajectory

    def save(self, path: Path) -> None:
        """Save GP model to directory."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save model states
        if self.cpu_model:
            torch.save(
                {
                    "model_state": self.cpu_model.state_dict(),
                    "likelihood_state": self.cpu_likelihood.state_dict(),
                },
                path / "cpu_gp.pt",
            )

        if self.gpu_model:
            torch.save(
                {
                    "model_state": self.gpu_model.state_dict(),
                    "likelihood_state": self.gpu_likelihood.state_dict(),
                },
                path / "gpu_gp.pt",
            )

        # Save training data (needed for GP predictions)
        if self.train_x is not None:
            torch.save(
                {
                    "train_x": self.train_x.cpu(),
                    "train_y_cpu": self.train_y_cpu.cpu(),
                    "train_y_gpu": self.train_y_gpu.cpu(),
                },
                path / "train_data.pt",
            )

        # Save metadata
        metadata = {
            "model_type": "gp",
            "input_dim": self.input_dim,
            "input_mean": self.input_mean.tolist()
            if self.input_mean is not None
            else None,
            "input_std": self.input_std.tolist()
            if self.input_std is not None
            else None,
            "cpu_mean": self.cpu_mean,
            "cpu_std": self.cpu_std,
            "gpu_mean": self.gpu_mean,
            "gpu_std": self.gpu_std,
            "config": self.config,
        }

        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"GP model saved to {path}")

    @classmethod
    def load(
        cls, path: Path, config: Optional[Dict[str, Any]] = None
    ) -> "GaussianProcessModel":
        """Load GP model from directory."""
        path = Path(path)

        with open(path / "metadata.json") as f:
            metadata = json.load(f)

        model_config = config if config else metadata.get("config", {})
        model = cls(model_config)

        # Restore normalization
        if metadata["input_mean"]:
            model.input_mean = np.array(metadata["input_mean"])
            model.input_std = np.array(metadata["input_std"])
        model.cpu_mean = metadata["cpu_mean"]
        model.cpu_std = metadata["cpu_std"]
        model.gpu_mean = metadata["gpu_mean"]
        model.gpu_std = metadata["gpu_std"]

        # Load training data
        train_data = torch.load(path / "train_data.pt", weights_only=True)
        model.train_x = train_data["train_x"].to(model.device)
        model.train_y_cpu = train_data["train_y_cpu"].to(model.device)
        model.train_y_gpu = train_data["train_y_gpu"].to(model.device)

        # Restore CPU GP
        model.cpu_likelihood = GaussianLikelihood().to(model.device)
        model.cpu_model = ThermalGP(
            model.train_x, model.train_y_cpu, model.cpu_likelihood, model.input_dim
        ).to(model.device)

        cpu_state = torch.load(path / "cpu_gp.pt", weights_only=True)
        model.cpu_model.load_state_dict(cpu_state["model_state"])
        model.cpu_likelihood.load_state_dict(cpu_state["likelihood_state"])
        model.cpu_model.eval()
        model.cpu_likelihood.eval()

        # Restore GPU GP
        model.gpu_likelihood = GaussianLikelihood().to(model.device)
        model.gpu_model = ThermalGP(
            model.train_x, model.train_y_gpu, model.gpu_likelihood, model.input_dim
        ).to(model.device)

        gpu_state = torch.load(path / "gpu_gp.pt", weights_only=True)
        model.gpu_model.load_state_dict(gpu_state["model_state"])
        model.gpu_likelihood.load_state_dict(gpu_state["likelihood_state"])
        model.gpu_model.eval()
        model.gpu_likelihood.eval()

        logger.info(f"GP model loaded from {path}")
        return model
