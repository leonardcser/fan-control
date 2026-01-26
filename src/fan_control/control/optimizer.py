"""
Fan speed optimizer using dynamic thermal models.

Finds minimal fan speeds to satisfy temperature constraints using the
DynamicThermalModel interface for predictions.
"""

import numpy as np
from pathlib import Path
from scipy.optimize import minimize
from typing import Dict, List, Optional, Tuple

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

        # Convert to integer PWM values
        optimal_pwms = {}
        for i, name in enumerate(self.pwm_names):
            min_val, max_val = self.bounds[i]
            clipped = np.clip(result.x[i], min_val, max_val)
            optimal_pwms[name] = int(round(clipped))

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

        # Default weights
        if cost_weights is None:
            w_effort = 1.0
            w_smooth = 0.1
        else:
            w_effort = cost_weights.get("effort", 1.0)
            w_smooth = cost_weights.get("smoothness", 0.1)

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

        target_cpu = targets["T_cpu"]
        target_gpu = targets["T_gpu"]

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
            J = sum(w_effort * ||u_k||^2 + w_smooth * ||u_k - u_{k-1}||^2)
            """
            U = x.reshape((control_horizon, self.n_pwm))

            # Effort cost (sum of squared PWMs)
            effort = np.sum(U**2)

            # Smoothness cost (sum of squared differences)
            # For k=0, assume previous control was similar or just penalize change from current
            # Here we penalize change between optimized steps.
            diffs = np.diff(U, axis=0)
            smoothness = np.sum(diffs**2)

            # Penalize difference from current state (continuity)
            if current_pwm_vec is not None:
                jump = U[0] - current_pwm_vec
                smoothness += np.sum(jump**2)

            return (w_effort * effort + w_smooth * smoothness) * 1e-4

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

        optimal_pwms = {}
        for i, name in enumerate(self.pwm_names):
            min_val, max_val = self.bounds[i]
            clipped = np.clip(u_0[i], min_val, max_val)
            optimal_pwms[name] = int(round(clipped))

        return optimal_pwms
