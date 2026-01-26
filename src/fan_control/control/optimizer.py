"""
Fan speed optimizer using dynamic thermal models.

Finds minimal fan speeds to satisfy temperature constraints using the
DynamicThermalModel interface for predictions.
"""

import logging
import numpy as np
from pathlib import Path
from scipy.optimize import minimize
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

from fan_control.models import load_model
from fan_control.models.base import DynamicThermalModel


class Optimizer:
    """
    Finds minimal fan speeds to satisfy temperature constraints.

    Uses COBYLA optimization with temperature constraints and
    caches predictions for performance.
    """

    def __init__(
        self,
        model: DynamicThermalModel,
        config: Dict,
        params: Optional[Dict] = None,
    ):
        """
        Initialize optimizer with a dynamic thermal model.

        Args:
            model: Trained DynamicThermalModel instance
            config: Hardware config with device definitions
            params: Optional params with optimizer settings
        """
        self.model = model
        self.devices = config["devices"]

        # Optimizer config from params.yaml or fallback
        if params and "optimizer" in params:
            self.optimizer_config = params["optimizer"]
        else:
            self.optimizer_config = {
                "method": "COBYLA",
                "options": {"maxiter": 200, "rhobeg": 15.0, "tol": 0.5},
            }

        self.pwm_names = list(self.devices.keys())
        self.bounds = self._get_bounds()
        self.n_pwm = len(self.pwm_names)

    @classmethod
    def from_path(
        cls,
        model_path: Path,
        model_type: str,
        config: Dict,
        params: Optional[Dict] = None,
    ) -> "Optimizer":
        """
        Create optimizer by loading model from path.

        Args:
            model_path: Path to saved model directory
            model_type: Model type (physics, pinn, gbr, gp)
            config: Hardware config
            params: Optional params with model and optimizer settings
        """
        model_config = params.get("model", {}) if params else {}
        model = load_model(model_type, str(model_path), model_config)
        return cls(model, config, params)

    def _get_bounds(self) -> List[Tuple[float, float]]:
        """Get (min, max) bounds for each PWM channel."""
        bounds = []
        for name in self.pwm_names:
            dev_cfg = self.devices[name]
            min_val = dev_cfg["min_pwm"]
            bounds.append((float(min_val), 100.0))
        return bounds

    def _snap_pwm(self, value: float, device_name: str) -> int:
        """Snap PWM value to avoid dead zone between 0 and stall_pwm.

        Values in (0, stall_pwm) are snapped to 0 (off) since the fan
        would stall anyway and waste power without moving air.
        """
        dev_cfg = self.devices[device_name]
        stall_pwm = dev_cfg.get("stall_pwm", 0)
        min_pwm = dev_cfg["min_pwm"]

        # Clamp to bounds first
        value = max(min_pwm, min(100.0, value))

        # Snap dead zone to 0 (if min_pwm allows 0)
        if min_pwm == 0 and 0 < value < stall_pwm:
            return 0

        return int(round(value))

    def optimize(
        self,
        current_state: Dict[str, float],
        targets: Dict[str, float],
        weights: Optional[Dict[str, float]] = None,
        initial_guess: Optional[Dict[str, float]] = None,
    ) -> Dict[str, int]:
        """
        Find optimal fan speeds to meet temperature targets.

        Args:
            current_state: Current readings {P_cpu, P_gpu, T_amb, T_cpu, T_gpu}
            targets: Target temperatures {T_cpu, T_gpu}
            weights: Optional PWM weights for objective
            initial_guess: Optional starting PWM values

        Returns:
            Optimal PWM values {pwm2, pwm4, pwm5}
        """
        # Initial guess
        if initial_guess:
            x0 = np.array([float(initial_guess[name]) for name in self.pwm_names])
        else:
            x0 = np.array([b[0] for b in self.bounds])

        # Weights for objective
        if weights is None:
            w = np.ones(self.n_pwm)
        else:
            w = np.array([weights.get(name, 1.0) for name in self.pwm_names])

        target_cpu = targets["T_cpu"]
        target_gpu = targets["T_gpu"]

        # Extract current state
        T_k = np.array(
            [
                current_state.get("T_cpu", 50.0),
                current_state.get("T_gpu", 40.0),
            ]
        )
        P = np.array(
            [
                current_state.get("P_cpu", 0.0),
                current_state.get("P_gpu", 0.0),
            ]
        )
        T_amb = current_state.get("T_amb", 25.0)

        # Cache for predictions
        cache = {}

        def get_temps(x: np.ndarray) -> Tuple[float, float]:
            """Get cached temperature predictions."""
            key = tuple(np.round(x, 4))
            if key not in cache:
                PWM = np.array([x[i] for i in range(self.n_pwm)])
                T_next, _ = self.model.predict_next(T_k, PWM, P, T_amb)
                cache[key] = (float(T_next[0]), float(T_next[1]))
            return cache[key]

        def objective(x: np.ndarray) -> float:
            """Minimize weighted sum of fan speeds."""
            return np.dot(x, w) * 0.01

        def constraint_cpu(x: np.ndarray) -> float:
            t_cpu, _ = get_temps(x)
            return target_cpu - t_cpu

        def constraint_gpu(x: np.ndarray) -> float:
            _, t_gpu = get_temps(x)
            return target_gpu - t_gpu

        # Build constraints including bounds for COBYLA
        constraints = [
            {"type": "ineq", "fun": constraint_cpu},
            {"type": "ineq", "fun": constraint_gpu},
        ]
        for i, (lo, hi) in enumerate(self.bounds):
            constraints.append({"type": "ineq", "fun": lambda x, i=i, lo=lo: x[i] - lo})
            constraints.append({"type": "ineq", "fun": lambda x, i=i, hi=hi: hi - x[i]})

        # Optimize
        method = self.optimizer_config["method"]
        options = self.optimizer_config["options"]
        result = minimize(
            objective,
            x0,
            method=method,
            constraints=constraints,
            options=options,
        )

        # Convert to integer PWM values with dead zone snapping
        optimal_pwms = {}
        for i, name in enumerate(self.pwm_names):
            optimal_pwms[name] = self._snap_pwm(result.x[i], name)

        return optimal_pwms

    def optimize_horizon(
        self,
        current_state: Dict[str, float],
        targets: Dict[str, float],
        prediction_horizon: int = 10,
        control_horizon: int = 5,
        cost_weights: Optional[Dict[str, float]] = None,
        initial_guess: Optional[Dict[str, float]] = None,
    ) -> Dict[str, int]:
        """
        Optimize over a prediction horizon (MPC-style).

        Finds PWM values that minimize fan speed and smoothness cost while keeping
        temperatures below targets over the full horizon.

        Args:
            current_state: Current state
            targets: Target temperatures
            prediction_horizon: Number of steps to predict (N)
            control_horizon: Number of steps to optimize controls (M <= N)
            cost_weights: Weights for 'effort' and 'smoothness'
            initial_guess: Optional starting PWM values (single step or sequence)

        Returns:
            Optimal PWM values for current step (receding horizon)
        """
        # Ensure M <= N
        control_horizon = min(control_horizon, prediction_horizon)

        # Weights from config (required)
        if cost_weights is None:
            raise ValueError("cost_weights is required for MPC optimization")
        w_effort = cost_weights["effort"]
        w_smooth = cost_weights["smoothness"]
        w_temp = cost_weights["temperature"]

        target_cpu = targets["T_cpu"]
        target_gpu = targets["T_gpu"]

        # Soft limit margins - proportional to available headroom
        # GPU has tighter limit (65°C) so use smaller margin
        margin_cpu = 15.0  # Start penalizing 15°C before CPU limit (at 70°C for 85°C target)
        margin_gpu = 10.0  # Start penalizing 10°C before GPU limit (at 55°C for 65°C target)
        soft_limit_cpu = target_cpu - margin_cpu
        soft_limit_gpu = target_gpu - margin_gpu

        # Optimization variables: [u_0, u_1, ..., u_{M-1}] flattened
        # Size: n_pwm * control_horizon

        # Initial guess construction
        current_pwm_vec = None
        if initial_guess:
            # If provided single step, repeat it
            u0_single = np.array(
                [float(initial_guess[name]) for name in self.pwm_names]
            )
            x0 = np.tile(u0_single, control_horizon)
            current_pwm_vec = u0_single
        else:
            # Default to minimum bound
            u0_single = np.array([b[0] for b in self.bounds])
            x0 = np.tile(u0_single, control_horizon)
            current_pwm_vec = u0_single

        T_k = np.array(
            [
                current_state.get("T_cpu", 50.0),
                current_state.get("T_gpu", 40.0),
            ]
        )
        P = np.array(
            [
                current_state.get("P_cpu", 0.0),
                current_state.get("P_gpu", 0.0),
            ]
        )
        T_amb = current_state.get("T_amb", 25.0)

        logger.debug(f"MPC input: P=[{P[0]:.1f}, {P[1]:.1f}]W, T_amb={T_amb:.1f}°C")

        cache = {}

        def get_trajectory(x: np.ndarray) -> np.ndarray:
            """
            Get cached temperature trajectory.
            x is flattened control sequence.
            Returns array of shape (prediction_horizon + 1, 2)
            """
            key = tuple(np.round(x, 4))
            if key not in cache:
                # Reshape x to (control_horizon, n_pwm)
                U_opt = x.reshape((control_horizon, self.n_pwm))

                # Construct full control sequence of length prediction_horizon
                # Repeat the last control action for steps beyond control_horizon
                if prediction_horizon > control_horizon:
                    last_u = U_opt[-1:]
                    extra_steps = prediction_horizon - control_horizon
                    U_full = np.vstack([U_opt, np.tile(last_u, (extra_steps, 1))])
                else:
                    U_full = U_opt

                # Simulate trajectory
                traj = np.zeros((prediction_horizon + 1, 2))
                traj[0] = T_k
                T_curr = T_k.copy()

                for k in range(prediction_horizon):
                    PWM = U_full[k]
                    T_curr, _ = self.model.predict_next(T_curr, PWM, P, T_amb)
                    traj[k + 1] = T_curr

                cache[key] = traj
            return cache[key]

        def objective(x: np.ndarray) -> float:
            """
            Cost function:
            J = w_effort * ||u||^2 + w_smooth * ||Δu||^2 + w_temp * temp_cost

            The temperature cost penalizes temperatures approaching limits,
            creating a smooth barrier that keeps the system away from constraints.
            """
            U = x.reshape((control_horizon, self.n_pwm))

            # Effort cost (sum of squared PWMs, normalized by 100^2)
            effort = np.sum(U**2) / 10000.0

            # Smoothness cost (sum of squared differences, normalized)
            diffs = np.diff(U, axis=0)
            smoothness = np.sum(diffs**2) / 10000.0

            # Penalize difference from current state (continuity)
            if current_pwm_vec is not None:
                jump = U[0] - current_pwm_vec
                smoothness += np.sum(jump**2) / 10000.0

            # Temperature tracking cost - soft barrier approaching limits
            traj = get_trajectory(x)

            # Quadratic penalty for temperatures above soft limits
            # This creates a smooth cost that increases as T approaches the hard limit
            temp_cost = 0.0
            for k in range(prediction_horizon + 1):
                T_cpu_k = traj[k, 0]
                T_gpu_k = traj[k, 1]

                # CPU: penalize when above soft_limit_cpu
                if T_cpu_k > soft_limit_cpu:
                    # Normalized by margin squared so cost ~ 1.0 at hard limit
                    temp_cost += ((T_cpu_k - soft_limit_cpu) / margin_cpu) ** 2

                # GPU: penalize when above soft_limit_gpu
                if T_gpu_k > soft_limit_gpu:
                    temp_cost += ((T_gpu_k - soft_limit_gpu) / margin_gpu) ** 2

            # Average over horizon
            temp_cost /= (prediction_horizon + 1)

            return w_effort * effort + w_smooth * smoothness + w_temp * temp_cost

        def constraint_cpu_max(x: np.ndarray) -> float:
            traj = get_trajectory(x)
            return target_cpu - np.max(traj[:, 0])

        def constraint_gpu_max(x: np.ndarray) -> float:
            traj = get_trajectory(x)
            return target_gpu - np.max(traj[:, 1])

        constraints = [
            {"type": "ineq", "fun": constraint_cpu_max},
            {"type": "ineq", "fun": constraint_gpu_max},
        ]

        # Rate limit: max PWM change per step (prevents sudden jumps)
        max_rate = 25.0  # Max 25% change per second

        # Rate constraint from current PWM to first control action
        # Two constraints: u[0] - curr <= max_rate AND curr - u[0] <= max_rate
        if current_pwm_vec is not None:
            for i in range(self.n_pwm):
                curr = current_pwm_vec[i]
                # u[0,i] - curr <= max_rate
                constraints.append(
                    {
                        "type": "ineq",
                        "fun": lambda x, i=i, c=curr: max_rate - (x[i] - c),
                    }
                )
                # curr - u[0,i] <= max_rate
                constraints.append(
                    {
                        "type": "ineq",
                        "fun": lambda x, i=i, c=curr: max_rate - (c - x[i]),
                    }
                )

        # Rate constraints between consecutive control actions
        for k in range(control_horizon - 1):
            for i in range(self.n_pwm):
                idx_curr = k * self.n_pwm + i
                idx_next = (k + 1) * self.n_pwm + i
                # u[k+1] - u[k] <= max_rate
                constraints.append(
                    {
                        "type": "ineq",
                        "fun": lambda x, ic=idx_curr, inext=idx_next: max_rate
                        - (x[inext] - x[ic]),
                    }
                )
                # u[k] - u[k+1] <= max_rate
                constraints.append(
                    {
                        "type": "ineq",
                        "fun": lambda x, ic=idx_curr, inext=idx_next: max_rate
                        - (x[ic] - x[inext]),
                    }
                )

        # Bounds constraints for all variables
        for k in range(control_horizon):
            for i in range(self.n_pwm):
                idx = k * self.n_pwm + i
                lo, hi = self.bounds[i]
                constraints.append(
                    {"type": "ineq", "fun": lambda x, idx=idx, lo=lo: x[idx] - lo}
                )
                constraints.append(
                    {"type": "ineq", "fun": lambda x, idx=idx, hi=hi: hi - x[idx]}
                )

        method = self.optimizer_config["method"]
        options = self.optimizer_config["options"]

        result = minimize(
            objective,
            x0,
            method=method,
            constraints=constraints,
            options=options,
        )

        # Extract first control action (Receding Horizon)
        x_opt = result.x
        u_0 = x_opt[: self.n_pwm]

        # Debug logging - show what the optimizer is thinking
        traj = get_trajectory(x_opt)
        T_cpu_pred = traj[:, 0]
        T_gpu_pred = traj[:, 1]
        U_opt = x_opt.reshape((control_horizon, self.n_pwm))

        # Compute cost breakdown
        effort = np.sum(U_opt**2) / 10000.0
        diffs = np.diff(U_opt, axis=0)
        smoothness = np.sum(diffs**2) / 10000.0
        if current_pwm_vec is not None:
            jump = U_opt[0] - current_pwm_vec
            smoothness += np.sum(jump**2) / 10000.0

        temp_cost = 0.0
        for k in range(prediction_horizon + 1):
            if T_cpu_pred[k] > soft_limit_cpu:
                temp_cost += ((T_cpu_pred[k] - soft_limit_cpu) / margin_cpu) ** 2
            if T_gpu_pred[k] > soft_limit_gpu:
                temp_cost += ((T_gpu_pred[k] - soft_limit_gpu) / margin_gpu) ** 2
        temp_cost /= (prediction_horizon + 1)

        cpu_margin = target_cpu - np.max(T_cpu_pred)
        gpu_margin = target_gpu - np.max(T_gpu_pred)

        logger.debug(
            f"MPC: T_now=[{T_k[0]:.1f}, {T_k[1]:.1f}] | "
            f"T_pred_max=[{np.max(T_cpu_pred):.1f}, {np.max(T_gpu_pred):.1f}] | "
            f"Margin=[{cpu_margin:.1f}, {gpu_margin:.1f}] | "
            f"Cost: eff={w_effort*effort:.3f} smo={w_smooth*smoothness:.3f} tmp={w_temp*temp_cost:.3f} | "
            f"u0={U_opt[0].astype(int)}"
        )

        # Convert to integer PWM values with dead zone snapping
        optimal_pwms = {}
        for i, name in enumerate(self.pwm_names):
            optimal_pwms[name] = self._snap_pwm(u_0[i], name)

        return optimal_pwms
