"""
Physics-based Thermal RC Network model.

Models thermal dynamics using resistance-capacitance (RC) circuit analogy:
- Thermal mass (C) stores heat energy
- Thermal resistance (R) governs heat flow
- T_{k+1} = T_k + dt/C * (Q_in - Q_out)

Where:
- Q_in = Power dissipation (P_cpu, P_gpu)
- Q_out = Heat removed by fans and ambient convection
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from . import register_model
from .base import DynamicThermalModel

logger = logging.getLogger(__name__)


@register_model("physics")
class PhysicsModel(DynamicThermalModel):
    """
    Thermal RC network model with learnable parameters.

    The model uses a first-order thermal dynamics equation:
    dT/dt = (1/C) * [P - (T - T_amb) * (R_base + R_fan * f(PWM))]

    Discretized (Euler):
    T_{k+1} = T_k + dt * (P/C - (T_k - T_amb) * G_eff)

    Where G_eff (effective thermal conductance) depends on fan speed:
    G_eff = G_base + sum(G_fan_i * PWM_i / 100)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Physics-specific config
        physics_config = config["physics"]
        self.dt = physics_config["dt"]

        # Learnable parameters - initial values (will be optimized during training)
        # CPU thermal parameters
        self.C_cpu = physics_config["C_cpu"]
        self.G_cpu_base = physics_config["G_cpu_base"]
        self.G_cpu_pwm2 = physics_config["G_cpu_pwm2"]
        self.G_cpu_pwm4 = physics_config["G_cpu_pwm4"]
        self.G_cpu_pwm5 = physics_config["G_cpu_pwm5"]

        # GPU thermal parameters
        self.C_gpu = physics_config["C_gpu"]
        self.G_gpu_base = physics_config["G_gpu_base"]
        self.G_gpu_pwm2 = physics_config["G_gpu_pwm2"]
        self.G_gpu_pwm4 = physics_config["G_gpu_pwm4"]
        self.G_gpu_pwm5 = physics_config["G_gpu_pwm5"]

        # Parameter bounds for optimization
        self.param_bounds = {
            "C": (10.0, 500.0),
            "G_base": (0.1, 10.0),
            "G_fan": (0.001, 0.5),
        }

    def _pack_params(self) -> np.ndarray:
        """Pack all parameters into a vector for optimization."""
        return np.array([
            self.C_cpu, self.G_cpu_base, self.G_cpu_pwm2, self.G_cpu_pwm4, self.G_cpu_pwm5,
            self.C_gpu, self.G_gpu_base, self.G_gpu_pwm2, self.G_gpu_pwm4, self.G_gpu_pwm5,
        ])

    def _unpack_params(self, params: np.ndarray) -> None:
        """Unpack parameter vector into model attributes."""
        self.C_cpu = params[0]
        self.G_cpu_base = params[1]
        self.G_cpu_pwm2 = params[2]
        self.G_cpu_pwm4 = params[3]
        self.G_cpu_pwm5 = params[4]
        self.C_gpu = params[5]
        self.G_gpu_base = params[6]
        self.G_gpu_pwm2 = params[7]
        self.G_gpu_pwm4 = params[8]
        self.G_gpu_pwm5 = params[9]

    def _get_bounds(self) -> list:
        """Get optimization bounds for all parameters."""
        return [
            self.param_bounds["C"],  # C_cpu
            self.param_bounds["G_base"],  # G_cpu_base
            self.param_bounds["G_fan"],  # G_cpu_pwm2
            self.param_bounds["G_fan"],  # G_cpu_pwm4
            self.param_bounds["G_fan"],  # G_cpu_pwm5
            self.param_bounds["C"],  # C_gpu
            self.param_bounds["G_base"],  # G_gpu_base
            self.param_bounds["G_fan"],  # G_gpu_pwm2
            self.param_bounds["G_fan"],  # G_gpu_pwm4
            self.param_bounds["G_fan"],  # G_gpu_pwm5
        ]

    def _compute_G_eff(
        self, pwm2: float, pwm4: float, pwm5: float, is_cpu: bool = True
    ) -> float:
        """
        Compute effective thermal conductance based on fan speeds.

        Args:
            pwm2, pwm4, pwm5: Fan speeds (0-100)
            is_cpu: Whether to use CPU or GPU parameters

        Returns:
            Effective thermal conductance (W/°C)
        """
        if is_cpu:
            return (
                self.G_cpu_base
                + self.G_cpu_pwm2 * pwm2 / 100
                + self.G_cpu_pwm4 * pwm4 / 100
                + self.G_cpu_pwm5 * pwm5 / 100
            )
        else:
            return (
                self.G_gpu_base
                + self.G_gpu_pwm2 * pwm2 / 100
                + self.G_gpu_pwm4 * pwm4 / 100
                + self.G_gpu_pwm5 * pwm5 / 100
            )

    def _step(
        self,
        T: float,
        P: float,
        G_eff: float,
        C: float,
        T_amb: float,
    ) -> float:
        """
        Single Euler step for temperature dynamics.

        dT/dt = P/C - G_eff * (T - T_amb) / C
        T_{k+1} = T_k + dt * dT/dt
        """
        dT_dt = (P - G_eff * (T - T_amb)) / C
        return T + self.dt * dT_dt

    def train(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Train by fitting thermal parameters to minimize prediction error.

        Uses scipy.optimize to find parameters that best fit the observed
        temperature dynamics.
        """
        # Prepare data - need consecutive time steps
        if "T_cpu_next" not in df.columns:
            df = df.copy()
            df["T_cpu_next"] = df["T_cpu"].shift(-1)
            df["T_gpu_next"] = df["T_gpu"].shift(-1)
            df = df.iloc[:-1]

        # Extract arrays for fast computation
        T_cpu = df["T_cpu"].values
        T_gpu = df["T_gpu"].values
        T_cpu_next = df["T_cpu_next"].values
        T_gpu_next = df["T_gpu_next"].values
        P_cpu = df["P_cpu"].values
        P_gpu = df["P_gpu"].values
        T_amb = df["T_amb"].values
        pwm2 = df["pwm2"].values
        pwm4 = df["pwm4"].values
        pwm5 = df["pwm5"].values

        def loss_fn(params: np.ndarray) -> float:
            """MSE loss over all samples."""
            self._unpack_params(params)

            total_loss = 0.0
            n = len(T_cpu)

            for i in range(n):
                # CPU prediction
                G_cpu = self._compute_G_eff(pwm2[i], pwm4[i], pwm5[i], is_cpu=True)
                T_cpu_pred = self._step(T_cpu[i], P_cpu[i], G_cpu, self.C_cpu, T_amb[i])
                total_loss += (T_cpu_pred - T_cpu_next[i]) ** 2

                # GPU prediction
                G_gpu = self._compute_G_eff(pwm2[i], pwm4[i], pwm5[i], is_cpu=False)
                T_gpu_pred = self._step(T_gpu[i], P_gpu[i], G_gpu, self.C_gpu, T_amb[i])
                total_loss += (T_gpu_pred - T_gpu_next[i]) ** 2

            return total_loss / (2 * n)

        # Optimize parameters
        logger.info("Fitting physics model parameters...")
        initial_params = self._pack_params()

        result = minimize(
            loss_fn,
            initial_params,
            method="L-BFGS-B",
            bounds=self._get_bounds(),
            options={"maxiter": 500, "disp": False},
        )

        self._unpack_params(result.x)

        # Compute final metrics
        cpu_errors = []
        gpu_errors = []

        for i in range(len(T_cpu)):
            G_cpu = self._compute_G_eff(pwm2[i], pwm4[i], pwm5[i], is_cpu=True)
            T_cpu_pred = self._step(T_cpu[i], P_cpu[i], G_cpu, self.C_cpu, T_amb[i])
            cpu_errors.append(T_cpu_pred - T_cpu_next[i])

            G_gpu = self._compute_G_eff(pwm2[i], pwm4[i], pwm5[i], is_cpu=False)
            T_gpu_pred = self._step(T_gpu[i], P_gpu[i], G_gpu, self.C_gpu, T_amb[i])
            gpu_errors.append(T_gpu_pred - T_gpu_next[i])

        cpu_errors = np.array(cpu_errors)
        gpu_errors = np.array(gpu_errors)

        metrics = {
            "cpu_rmse": float(np.sqrt(np.mean(cpu_errors**2))),
            "cpu_mae": float(np.mean(np.abs(cpu_errors))),
            "gpu_rmse": float(np.sqrt(np.mean(gpu_errors**2))),
            "gpu_mae": float(np.mean(np.abs(gpu_errors))),
            "total_loss": float(result.fun),
        }

        logger.info(f"Physics model fitted: CPU RMSE={metrics['cpu_rmse']:.2f}°C, "
                    f"GPU RMSE={metrics['gpu_rmse']:.2f}°C")
        logger.info(f"Fitted parameters: C_cpu={self.C_cpu:.1f}, G_cpu_base={self.G_cpu_base:.3f}")
        logger.info(f"                   C_gpu={self.C_gpu:.1f}, G_gpu_base={self.G_gpu_base:.3f}")

        return metrics

    def predict_next(
        self,
        T_k: np.ndarray,
        PWM: np.ndarray,
        P: np.ndarray,
        T_amb: float,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Predict next temperatures using RC model."""
        # Compute effective conductances
        G_cpu = self._compute_G_eff(PWM[0], PWM[1], PWM[2], is_cpu=True)
        G_gpu = self._compute_G_eff(PWM[0], PWM[1], PWM[2], is_cpu=False)

        # Step forward
        T_cpu_next = self._step(T_k[0], P[0], G_cpu, self.C_cpu, T_amb)
        T_gpu_next = self._step(T_k[1], P[1], G_gpu, self.C_gpu, T_amb)

        return np.array([T_cpu_next, T_gpu_next]), None

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
        """Save model parameters to directory."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        params = {
            "model_type": "physics",
            "dt": self.dt,
            "cpu": {
                "C": self.C_cpu,
                "G_base": self.G_cpu_base,
                "G_pwm2": self.G_cpu_pwm2,
                "G_pwm4": self.G_cpu_pwm4,
                "G_pwm5": self.G_cpu_pwm5,
            },
            "gpu": {
                "C": self.C_gpu,
                "G_base": self.G_gpu_base,
                "G_pwm2": self.G_gpu_pwm2,
                "G_pwm4": self.G_gpu_pwm4,
                "G_pwm5": self.G_gpu_pwm5,
            },
            "config": self.config,
        }

        with open(path / "params.json", "w") as f:
            json.dump(params, f, indent=2)

        logger.info(f"Physics model saved to {path}")

    @classmethod
    def load(cls, path: Path, config: Optional[Dict[str, Any]] = None) -> "PhysicsModel":
        """Load model from directory."""
        path = Path(path)

        with open(path / "params.json") as f:
            params = json.load(f)

        model_config = config if config else params.get("config", {})
        model = cls(model_config)

        model.dt = params["dt"]
        model.C_cpu = params["cpu"]["C"]
        model.G_cpu_base = params["cpu"]["G_base"]
        model.G_cpu_pwm2 = params["cpu"]["G_pwm2"]
        model.G_cpu_pwm4 = params["cpu"]["G_pwm4"]
        model.G_cpu_pwm5 = params["cpu"]["G_pwm5"]

        model.C_gpu = params["gpu"]["C"]
        model.G_gpu_base = params["gpu"]["G_base"]
        model.G_gpu_pwm2 = params["gpu"]["G_pwm2"]
        model.G_gpu_pwm4 = params["gpu"]["G_pwm4"]
        model.G_gpu_pwm5 = params["gpu"]["G_pwm5"]

        logger.info(f"Physics model loaded from {path}")
        return model
