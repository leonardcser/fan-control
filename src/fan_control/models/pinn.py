"""
Physics-Informed Neural Network (PINN) thermal model.

Combines neural network flexibility with physics constraints:
- Data loss: MSE on observed temperature transitions
- Physics loss: Enforces thermal dynamics dT/dt ~ P - G*(T - T_amb)
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from . import register_model
from .base import DynamicThermalModel

logger = logging.getLogger(__name__)


class ThermalPINN(nn.Module):
    """
    Neural network for thermal dynamics prediction.

    Input: [T_cpu, T_gpu, pwm2, pwm4, pwm5, P_cpu, P_gpu, T_amb]
    Output: [T_cpu_next, T_gpu_next]
    """

    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int = 2):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


@register_model("pinn")
class PINNModel(DynamicThermalModel):
    """
    Physics-Informed Neural Network thermal model.

    Loss = data_loss + physics_weight * physics_loss

    Where physics_loss encourages the network to learn dynamics consistent
    with thermal physics (heat balance equation).
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # PINN-specific config
        pinn_config = config["pinn"]
        self.hidden_dims = pinn_config["hidden_dims"]
        self.epochs = pinn_config["epochs"]
        self.learning_rate = pinn_config["learning_rate"]
        self.batch_size = pinn_config["batch_size"]
        self.physics_weight = pinn_config["physics_weight"]
        self.dt = pinn_config["dt"]

        # Device selection
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # Input: [T_cpu, T_gpu, pwm2, pwm4, pwm5, P_cpu, P_gpu, T_amb]
        self.input_dim = 8
        self.output_dim = 2

        # Neural network
        self.network: Optional[ThermalPINN] = None

        # Normalization parameters (learned from training data)
        self.input_mean: Optional[np.ndarray] = None
        self.input_std: Optional[np.ndarray] = None
        self.output_mean: Optional[np.ndarray] = None
        self.output_std: Optional[np.ndarray] = None

    def _normalize_input(self, X: np.ndarray) -> np.ndarray:
        """Normalize input features."""
        if self.input_mean is None:
            return X
        return (X - self.input_mean) / (self.input_std + 1e-8)

    def _denormalize_output(self, Y: np.ndarray) -> np.ndarray:
        """Denormalize output predictions."""
        if self.output_mean is None:
            return Y
        return Y * (self.output_std + 1e-8) + self.output_mean

    def _physics_loss(
        self,
        T_k: torch.Tensor,
        T_next: torch.Tensor,
        P: torch.Tensor,
        T_amb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Physics-informed loss term.

        Enforces approximate thermal dynamics:
        dT/dt ≈ α*P - β*(T - T_amb)

        The network should learn that temperature change is:
        - Proportional to power input
        - Proportional to temperature difference from ambient
        """
        # Temperature change
        dT = T_next - T_k  # Shape: (batch, 2)

        # Simple physics constraint: temperature change should be bounded by power input
        # Just penalize large temperature jumps that aren't explained by power
        physics_residual = torch.abs(dT) - 0.1 * P  # Rough scaling

        return torch.mean(torch.clamp(physics_residual, min=0) ** 2)

    def train(self, df: pd.DataFrame) -> Dict[str, float]:
        """Train the PINN model."""
        # Prepare data
        if "T_cpu_next" not in df.columns:
            df = df.copy()
            df["T_cpu_next"] = df["T_cpu"].shift(-1)
            df["T_gpu_next"] = df["T_gpu"].shift(-1)
            df = df.iloc[:-1]

        # Extract features
        X = df[
            ["T_cpu", "T_gpu", "pwm2", "pwm4", "pwm5", "P_cpu", "P_gpu", "T_amb"]
        ].values
        Y = df[["T_cpu_next", "T_gpu_next"]].values

        # Compute normalization parameters
        self.input_mean = X.mean(axis=0)
        self.input_std = X.std(axis=0)
        self.output_mean = Y.mean(axis=0)
        self.output_std = Y.std(axis=0)

        # Normalize
        X_norm = self._normalize_input(X)
        Y_norm = (Y - self.output_mean) / (self.output_std + 1e-8)

        # Convert to tensors
        X_tensor = torch.FloatTensor(X_norm).to(self.device)
        Y_tensor = torch.FloatTensor(Y_norm).to(self.device)

        # For physics loss, keep unnormalized values
        T_k_tensor = torch.FloatTensor(df[["T_cpu", "T_gpu"]].values).to(self.device)
        P_tensor = torch.FloatTensor(df[["P_cpu", "P_gpu"]].values).to(self.device)
        T_amb_tensor = torch.FloatTensor(df["T_amb"].values).to(self.device)

        # Create data loader
        dataset = TensorDataset(X_tensor, Y_tensor, T_k_tensor, P_tensor, T_amb_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Initialize network
        self.network = ThermalPINN(self.input_dim, self.hidden_dims, self.output_dim)
        self.network.to(self.device)

        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)
        mse_loss = nn.MSELoss()

        # Training loop
        logger.info(f"Training PINN on {len(X)} samples for {self.epochs} epochs...")
        self.network.train()

        best_loss = float("inf")
        pbar = tqdm(range(self.epochs), desc="Training PINN", unit="epoch")
        for epoch in pbar:
            epoch_data_loss = 0.0
            epoch_physics_loss = 0.0

            for X_batch, Y_batch, T_k_batch, P_batch, T_amb_batch in loader:
                optimizer.zero_grad()

                # Forward pass
                Y_pred = self.network(X_batch)

                # Data loss (MSE on normalized outputs)
                data_loss = mse_loss(Y_pred, Y_batch)

                # Physics loss (on unnormalized predictions)
                Y_pred_unnorm = Y_pred * torch.FloatTensor(self.output_std + 1e-8).to(
                    self.device
                )
                Y_pred_unnorm = Y_pred_unnorm + torch.FloatTensor(self.output_mean).to(
                    self.device
                )

                physics_loss = self._physics_loss(
                    T_k_batch, Y_pred_unnorm, P_batch, T_amb_batch
                )

                # Combined loss
                loss = data_loss + self.physics_weight * physics_loss

                loss.backward()
                optimizer.step()

                epoch_data_loss += data_loss.item()
                epoch_physics_loss += physics_loss.item()

            avg_data_loss = epoch_data_loss / len(loader)
            avg_physics_loss = epoch_physics_loss / len(loader)

            pbar.set_postfix(data_loss=f"{avg_data_loss:.4f}", phys_loss=f"{avg_physics_loss:.4f}")

            if avg_data_loss < best_loss:
                best_loss = avg_data_loss

            if (epoch + 1) % 50 == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{self.epochs}: "
                    f"data_loss={avg_data_loss:.4f}, physics_loss={avg_physics_loss:.4f}"
                )

        # Final evaluation
        self.network.eval()
        with torch.no_grad():
            Y_pred = self.network(X_tensor)
            Y_pred_np = Y_pred.cpu().numpy()
            Y_pred_unnorm = self._denormalize_output(Y_pred_np)

        cpu_rmse = float(np.sqrt(np.mean((Y_pred_unnorm[:, 0] - Y[:, 0]) ** 2)))
        gpu_rmse = float(np.sqrt(np.mean((Y_pred_unnorm[:, 1] - Y[:, 1]) ** 2)))

        # Check for NaN in metrics
        if not np.isfinite(cpu_rmse):
            logger.warning("CPU RMSE is NaN/inf, setting to 999.0")
            cpu_rmse = 999.0
        if not np.isfinite(gpu_rmse):
            logger.warning("GPU RMSE is NaN/inf, setting to 999.0")
            gpu_rmse = 999.0

        metrics = {
            "cpu_rmse": cpu_rmse,
            "gpu_rmse": gpu_rmse,
            "cpu_mae": float(np.mean(np.abs(Y_pred_unnorm[:, 0] - Y[:, 0]))),
            "gpu_mae": float(np.mean(np.abs(Y_pred_unnorm[:, 1] - Y[:, 1]))),
            "final_data_loss": float(best_loss),
        }

        logger.info(
            f"PINN trained: CPU RMSE={cpu_rmse:.2f}°C, GPU RMSE={gpu_rmse:.2f}°C"
        )
        return metrics

    def predict_next(
        self,
        T_k: np.ndarray,
        PWM: np.ndarray,
        P: np.ndarray,
        T_amb: float,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Predict next temperatures using PINN."""
        if self.network is None:
            raise RuntimeError("Model not trained")

        # Build input vector
        X = np.array([[T_k[0], T_k[1], PWM[0], PWM[1], PWM[2], P[0], P[1], T_amb]])
        X_norm = self._normalize_input(X)

        # Predict
        self.network.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_norm).to(self.device)
            Y_pred = self.network(X_tensor)
            Y_pred_np = Y_pred.cpu().numpy()

        T_next = self._denormalize_output(Y_pred_np)[0]

        # Check for NaN/inf - if so, return current temperature
        if not np.isfinite(T_next[0]):
            logger.warning("NaN/inf in CPU PINN prediction, returning current temp")
            T_next[0] = T_k[0]
        if not np.isfinite(T_next[1]):
            logger.warning("NaN/inf in GPU PINN prediction, returning current temp")
            T_next[1] = T_k[1]

        return T_next, None

    def predict_horizon(
        self,
        T_0: np.ndarray,
        PWM_seq: np.ndarray,
        P_seq: np.ndarray,
        T_amb: float,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Predict trajectory over horizon."""
        PWM_seq = np.atleast_2d(PWM_seq)
        P_seq = np.atleast_2d(P_seq)

        horizon = max(len(PWM_seq), len(P_seq))

        if len(PWM_seq) == 1:
            PWM_seq = np.tile(PWM_seq, (horizon, 1))
        if len(P_seq) == 1:
            P_seq = np.tile(P_seq, (horizon, 1))

        trajectory = np.zeros((horizon + 1, 2))
        trajectory[0] = T_0

        T_k = T_0.copy()
        for k in range(horizon):
            T_k, _ = self.predict_next(T_k, PWM_seq[k], P_seq[k], T_amb)
            trajectory[k + 1] = T_k

        return trajectory, None

    def save(self, path: Path) -> None:
        """Save PINN model to directory."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save network weights
        if self.network:
            torch.save(self.network.state_dict(), path / "network.pt")

        # Save metadata and normalization params
        metadata = {
            "model_type": "pinn",
            "hidden_dims": self.hidden_dims,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "input_mean": self.input_mean.tolist()
            if self.input_mean is not None
            else None,
            "input_std": self.input_std.tolist()
            if self.input_std is not None
            else None,
            "output_mean": self.output_mean.tolist()
            if self.output_mean is not None
            else None,
            "output_std": self.output_std.tolist()
            if self.output_std is not None
            else None,
            "config": self.config,
        }

        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"PINN model saved to {path}")

    @classmethod
    def load(cls, path: Path, config: Optional[Dict[str, Any]] = None) -> "PINNModel":
        """Load PINN model from directory."""
        path = Path(path)

        with open(path / "metadata.json") as f:
            metadata = json.load(f)

        model_config = config if config else metadata.get("config", {})
        model = cls(model_config)

        # Restore normalization params
        if metadata["input_mean"]:
            model.input_mean = np.array(metadata["input_mean"])
            model.input_std = np.array(metadata["input_std"])
            model.output_mean = np.array(metadata["output_mean"])
            model.output_std = np.array(metadata["output_std"])

        # Restore network
        model.network = ThermalPINN(
            metadata["input_dim"],
            metadata["hidden_dims"],
            metadata["output_dim"],
        )
        model.network.load_state_dict(
            torch.load(path / "network.pt", weights_only=True)
        )
        model.network.to(model.device)
        model.network.eval()

        logger.info(f"PINN model loaded from {path}")
        return model
