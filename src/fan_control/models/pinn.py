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

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

from . import register_model
from .base import DynamicThermalModel

logger = logging.getLogger(__name__)


class ThermalNetwork(nn.Module):
    """
    Neural network for thermal dynamics prediction with residual connections.

    Input: [T_cpu, T_gpu, pwm2, pwm4, pwm5, P_cpu, P_gpu, T_amb]
    Output: [T_cpu_next, T_gpu_next]
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Input projection
        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.input_bn = nn.BatchNorm1d(hidden_dims[0])

        # Hidden layers with residual connections and batch norm
        self.hidden_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for i in range(len(hidden_dims) - 1):
            self.hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dims[i + 1]))
            self.dropouts.append(nn.Dropout(dropout))

        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

        # Residual connection from input temperatures to output (skip connection)
        # This helps the network learn temperature *changes* rather than absolute values
        self.temp_skip = nn.Linear(2, output_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract input temperatures for skip connection
        T_in = x[:, :2]  # [T_cpu, T_gpu]

        # Input layer
        h = torch.nn.functional.gelu(self.input_bn(self.input_layer(x)))

        # Hidden layers with residual connections
        for layer, bn, dropout in zip(
            self.hidden_layers, self.batch_norms, self.dropouts
        ):
            h_new = layer(h)
            h_new = bn(h_new)
            h_new = torch.nn.functional.gelu(h_new)
            h_new = dropout(h_new)

            # Residual connection (if dimensions match)
            if h.shape[1] == h_new.shape[1]:
                h = h + h_new
            else:
                h = h_new

        # Output layer
        T_out = self.output_layer(h)

        # Add skip connection from input temperatures
        # This makes it easier for the network to learn small temperature changes
        T_out = T_out + self.temp_skip(T_in)

        return T_out


class ThermalPINN(L.LightningModule):
    """
    PyTorch Lightning module for Physics-Informed Neural Network.

    Combines data loss (MSE) with physics loss (thermal dynamics constraints).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        learning_rate: float,
        physics_weight: float,
        output_mean: np.ndarray,
        output_std: np.ndarray,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["output_mean", "output_std"])

        self.network = ThermalNetwork(input_dim, hidden_dims, output_dim, dropout)
        self.learning_rate = learning_rate
        self.physics_weight = physics_weight

        # Store normalization params as buffers (moved to device automatically)
        self.register_buffer("output_mean", torch.FloatTensor(output_mean))
        self.register_buffer("output_std", torch.FloatTensor(output_std))

        self.mse_loss = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def _physics_loss(
        self,
        T_k: torch.Tensor,
        T_next: torch.Tensor,
        PWM: torch.Tensor,
        P: torch.Tensor,
        T_amb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Physics-informed loss term enforcing thermal dynamics.

        Enforces heat balance equation:
        C * dT/dt = P - G(PWM) * (T - T_amb)
        """
        # Temperature change (per timestep dt=1.0s)
        dT = T_next - T_k  # Shape: (batch, 2) for [CPU, GPU]

        # Temperature difference from ambient
        T_diff = T_k - T_amb.unsqueeze(1)  # Shape: (batch, 2)

        # Thermal conductance (baseline + PWM contributions)
        G_cpu = 1.0 + 0.05 * PWM[:, 0] + 0.02 * PWM[:, 1] + 0.02 * PWM[:, 2]
        G_gpu = 0.8 + 0.01 * PWM[:, 0] + 0.03 * PWM[:, 1] + 0.04 * PWM[:, 2]
        G = torch.stack([G_cpu, G_gpu], dim=1)  # Shape: (batch, 2)

        # Thermal capacitance (rough estimates)
        C = torch.tensor([100.0, 80.0], device=T_k.device).unsqueeze(0)  # Shape: (1, 2)

        # Expected temperature change from thermal dynamics
        # dT/dt = (P - G * (T - T_amb)) / C
        dT_expected = (P - G * T_diff) / C

        # Physics residual: difference between predicted and physics-expected change
        physics_residual = dT - dT_expected

        # MSE of physics residual
        physics_loss = torch.mean(physics_residual**2)

        # Additional constraints: temperature shouldn't change faster than physically possible
        max_dT = 5.0  # Maximum reasonable temperature change per second
        constraint_loss = torch.mean(torch.clamp(torch.abs(dT) - max_dT, min=0) ** 2)

        return physics_loss + 0.1 * constraint_loss

    def training_step(self, batch, _batch_idx):
        X_batch, Y_batch, T_k_batch, PWM_batch, P_batch, T_amb_batch = batch

        # Forward pass
        Y_pred = self(X_batch)

        # Data loss (MSE on normalized outputs)
        data_loss = self.mse_loss(Y_pred, Y_batch)

        # Physics loss (on unnormalized predictions)
        Y_pred_unnorm = Y_pred * self.output_std + self.output_mean
        physics_loss = self._physics_loss(
            T_k_batch, Y_pred_unnorm, PWM_batch, P_batch, T_amb_batch
        )

        # Combined loss
        loss = data_loss + self.physics_weight * physics_loss

        # Logging
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/data_loss", data_loss, on_step=False, on_epoch=True)
        self.log("train/physics_loss", physics_loss, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, _batch_idx):
        X_batch, Y_batch, T_k_batch, PWM_batch, P_batch, T_amb_batch = batch

        # Forward pass
        Y_pred = self(X_batch)

        # Data loss (MSE on normalized outputs)
        data_loss = self.mse_loss(Y_pred, Y_batch)

        # Physics loss (on unnormalized predictions)
        Y_pred_unnorm = Y_pred * self.output_std + self.output_mean
        physics_loss = self._physics_loss(
            T_k_batch, Y_pred_unnorm, PWM_batch, P_batch, T_amb_batch
        )

        # Combined loss
        loss = data_loss + self.physics_weight * physics_loss

        # Logging
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/data_loss", data_loss, on_step=False, on_epoch=True)
        self.log("val/physics_loss", physics_loss, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):  # type: ignore[override]
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=1e-4
        )
        max_epochs = (
            self.trainer.max_epochs if self.trainer.max_epochs is not None else 100
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs, eta_min=self.learning_rate * 0.01
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }


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
        self.dropout = pinn_config.get("dropout", 0.1)
        self.early_stopping_patience = pinn_config.get("early_stopping_patience", 5)

        # Log directory from config (defaults to "logs")
        self.log_dir = config.get("log_dir", "logs")

        # Input: [T_cpu, T_gpu, pwm2, pwm4, pwm5, P_cpu, P_gpu, T_amb]
        self.input_dim = 8
        self.output_dim = 2

        # Lightning module
        self.lightning_model: Optional[ThermalPINN] = None

        # Normalization parameters (learned from training data)
        self.input_mean: Optional[np.ndarray] = None
        self.input_std: Optional[np.ndarray] = None
        self.output_mean: Optional[np.ndarray] = None
        self.output_std: Optional[np.ndarray] = None

    def _normalize_input(self, X: np.ndarray) -> np.ndarray:
        """Normalize input features."""
        if self.input_mean is None or self.input_std is None:
            return X
        return (X - self.input_mean) / (self.input_std + 1e-8)

    def _denormalize_output(self, Y: np.ndarray) -> np.ndarray:
        """Denormalize output predictions."""
        if self.output_mean is None or self.output_std is None:
            return Y
        return Y * (self.output_std + 1e-8) + self.output_mean

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare dataframe by adding next temperature columns."""
        if "T_cpu_next" not in df.columns:
            df = df.copy()
            df["T_cpu_next"] = df["T_cpu"].shift(-1)
            df["T_gpu_next"] = df["T_gpu"].shift(-1)
            df = df.iloc[:-1]
        return df

    def _create_dataloader(self, df: pd.DataFrame, shuffle: bool = True) -> DataLoader:
        """Create a dataloader from a dataframe."""
        import os

        # Extract features
        X = np.asarray(
            df[
                ["T_cpu", "T_gpu", "pwm2", "pwm4", "pwm5", "P_cpu", "P_gpu", "T_amb"]
            ].values,
            dtype=np.float32,
        )
        Y = np.asarray(df[["T_cpu_next", "T_gpu_next"]].values, dtype=np.float32)

        # Normalize using stored parameters
        assert self.input_mean is not None and self.output_mean is not None
        assert self.output_std is not None
        X_norm = self._normalize_input(X)
        Y_norm = (Y - self.output_mean) / (self.output_std + 1e-8)

        # Convert to tensors
        X_tensor = torch.FloatTensor(X_norm)
        Y_tensor = torch.FloatTensor(Y_norm)

        # For physics loss, keep unnormalized values
        T_k_data = np.asarray(df[["T_cpu", "T_gpu"]].values, dtype=np.float32)
        PWM_data = np.asarray(df[["pwm2", "pwm4", "pwm5"]].values, dtype=np.float32)
        P_data = np.asarray(df[["P_cpu", "P_gpu"]].values, dtype=np.float32)
        T_amb_data = np.asarray(df["T_amb"].values, dtype=np.float32)

        T_k_tensor = torch.FloatTensor(T_k_data)
        PWM_tensor = torch.FloatTensor(PWM_data)
        P_tensor = torch.FloatTensor(P_data)
        T_amb_tensor = torch.FloatTensor(T_amb_data)

        num_workers = min(4, os.cpu_count() or 1)
        use_cuda = torch.cuda.is_available()

        dataset = TensorDataset(
            X_tensor, Y_tensor, T_k_tensor, PWM_tensor, P_tensor, T_amb_tensor
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            persistent_workers=True if num_workers > 0 else False,
            pin_memory=use_cuda,
        )

    def train(
        self, df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """Train the PINN model using PyTorch Lightning."""
        # Prepare data
        df = self._prepare_dataframe(df)

        # Extract features for normalization computation
        X = np.asarray(
            df[
                ["T_cpu", "T_gpu", "pwm2", "pwm4", "pwm5", "P_cpu", "P_gpu", "T_amb"]
            ].values,
            dtype=np.float32,
        )
        Y = np.asarray(df[["T_cpu_next", "T_gpu_next"]].values, dtype=np.float32)

        # Compute normalization parameters from training data
        self.input_mean = X.mean(axis=0)
        self.input_std = X.std(axis=0)
        self.output_mean = Y.mean(axis=0)
        self.output_std = Y.std(axis=0)

        # Create train dataloader
        train_loader = self._create_dataloader(df, shuffle=True)

        # Create validation dataloader if provided
        val_loader = None
        if val_df is not None:
            val_df = self._prepare_dataframe(val_df)
            val_loader = self._create_dataloader(val_df, shuffle=False)
            logger.info(f"Using {len(val_df)} validation samples")

        # Ensure normalization parameters are computed
        assert self.output_mean is not None and self.output_std is not None, (
            "Normalization parameters must be computed before creating model"
        )

        # Initialize Lightning model
        self.lightning_model = ThermalPINN(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            output_dim=self.output_dim,
            learning_rate=self.learning_rate,
            physics_weight=self.physics_weight,
            output_mean=self.output_mean,
            output_std=self.output_std,
            dropout=self.dropout,
        )

        # Configure TensorBoard logger
        tb_logger = TensorBoardLogger(
            save_dir=self.log_dir,
            name="pinn",
            version=None,  # Auto-increment version
            log_graph=False,
        )

        # Configure callbacks
        callbacks = []
        if val_loader is not None:
            # Early stopping
            early_stop_callback = EarlyStopping(
                monitor="val/loss",
                patience=self.early_stopping_patience,
                mode="min",
                verbose=True,
            )
            callbacks.append(early_stop_callback)

        # Configure trainer with optimizations
        trainer = L.Trainer(
            max_epochs=self.epochs,
            accelerator="auto",
            devices=1,
            logger=tb_logger,
            callbacks=callbacks if callbacks else None,
            enable_checkpointing=False,
            enable_progress_bar=True,
            gradient_clip_val=1.0,
            log_every_n_steps=10,
            precision="16-mixed",  # Mixed precision for faster training
        )

        # Train
        logger.info(f"Training PINN on {len(df)} samples for {self.epochs} epochs...")
        logger.info(f"TensorBoard logs: {tb_logger.log_dir}")
        trainer.fit(self.lightning_model, train_loader, val_loader)

        # Final evaluation on training data
        self.lightning_model.eval()
        X_norm = self._normalize_input(X)
        X_tensor = torch.FloatTensor(X_norm)
        with torch.no_grad():
            Y_pred = self.lightning_model(X_tensor.to(self.lightning_model.device))
            Y_pred_np = Y_pred.cpu().numpy()
            Y_pred_unnorm = self._denormalize_output(Y_pred_np)

        cpu_rmse = float(np.sqrt(np.mean((Y_pred_unnorm[:, 0] - Y[:, 0]) ** 2)))
        gpu_rmse = float(np.sqrt(np.mean((Y_pred_unnorm[:, 1] - Y[:, 1]) ** 2)))

        metrics = {
            "cpu_rmse": cpu_rmse,
            "gpu_rmse": gpu_rmse,
            "cpu_mae": float(np.mean(np.abs(Y_pred_unnorm[:, 0] - Y[:, 0]))),
            "gpu_mae": float(np.mean(np.abs(Y_pred_unnorm[:, 1] - Y[:, 1]))),
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
        if self.lightning_model is None:
            raise RuntimeError("Model not trained")

        # Build input vector
        X = np.array([[T_k[0], T_k[1], PWM[0], PWM[1], PWM[2], P[0], P[1], T_amb]])
        X_norm = self._normalize_input(X)

        # Predict
        self.lightning_model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_norm).to(self.lightning_model.device)
            Y_pred = self.lightning_model(X_tensor)
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

        # Save Lightning checkpoint
        if self.lightning_model:
            torch.save(self.lightning_model.state_dict(), path / "model.ckpt")

        # Save metadata and normalization params
        metadata = {
            "model_type": "pinn",
            "hidden_dims": self.hidden_dims,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "learning_rate": self.learning_rate,
            "physics_weight": self.physics_weight,
            "dropout": self.dropout,
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

        assert model.output_mean is not None and model.output_std is not None

        # Load Lightning checkpoint
        model.lightning_model = ThermalPINN(
            input_dim=metadata["input_dim"],
            hidden_dims=metadata["hidden_dims"],
            output_dim=metadata["output_dim"],
            learning_rate=metadata.get("learning_rate", 0.001),
            physics_weight=metadata.get("physics_weight", 0.5),
            output_mean=model.output_mean,
            output_std=model.output_std,
            dropout=metadata.get("dropout", 0.1),
        )
        model.lightning_model.load_state_dict(torch.load(path / "model.ckpt"))
        model.lightning_model.eval()

        logger.info(f"PINN model loaded from {path}")
        return model
