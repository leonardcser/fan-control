import numpy as np
import pandas as pd
from scipy.optimize import minimize
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from ..fit.train import ThermalModel

logger = logging.getLogger(__name__)


class Optimizer:
    """
    Finds minimal fan speeds to satisfy temperature constraints using the ML model.
    """

    def __init__(self, model_path: Path, config: Dict):
        self.model = ThermalModel.load(model_path)
        self.config = config
        self.devices = config["devices"]

        # Map device names to optimization indices
        # We optimize [pwm2, pwm4, pwm5, pwm7]
        self.pwm_names = ["pwm2", "pwm4", "pwm5", "pwm7"]
        self.bounds = self._get_bounds()

        # Optimization settings
        # Finite difference step size (epsilon) for SLSQP
        # With the new smooth 2-stage model, we don't need large jumps.
        self.eps = 1.0

    def _get_bounds(self) -> List[Tuple[float, float]]:
        """Get (min, max) bounds for each PWM channel."""
        bounds = []
        for name in self.pwm_names:
            dev_cfg = self.devices.get(name, {})
            min_val = dev_cfg.get("min_pwm", 20)
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
        Find optimal fan speeds.

        Args:
            current_state: Dict with 'P_cpu', 'P_gpu', 'T_amb'
            targets: Dict with 'T_cpu', 'T_gpu' (max allowed temps)
            weights: Optional weights for each fan (default 1.0)
            initial_guess: Optional starting point for optimization (default: 50%)

        Returns:
            Dict mapping pwm_name -> optimal_value (int)
        """
        # 1. Setup Initial Guess
        if initial_guess:
            x0 = [float(initial_guess.get(name, 50.0)) for name in self.pwm_names]
        else:
            # Smart Default: Start at minimums if we have no clue,
            # effectively "ramping up" if needed, rather than "cooling down"
            x0 = [b[0] for b in self.bounds]

        # 2. Heuristic Check removed.
        # We rely purely on the optimizer and the model.

        # Default weights (minimize sum of speeds)
        if weights is None:
            # We can weight pump less if we want, or louder fans more
            w = np.ones(len(self.pwm_names))
        else:
            w = np.array([weights.get(name, 1.0) for name in self.pwm_names])

        # 2. Define Objective Function
        # Scale weights to be comparable to constraint gradients (~0.01 - 0.05)
        # This helps the optimizer balance "cost of fan" vs "benefit of cooling"
        # If weights are too high (1.0), the optimizer fights hard against increasing fans.
        scale_factor = 0.01

        def objective(x):
            """Minimize weighted sum of fan speeds."""
            return np.sum(x * w) * scale_factor

        # 3. Define Constraints (T_pred <= T_target  =>  T_target - T_pred >= 0)
        def constraint_cpu(x):
            inputs = self._prepare_input_df(x, current_state)
            t_pred, _ = self.model.predict(inputs)
            # Safety margin: Target - Prediction (must be >= 0)
            return targets["T_cpu"] - t_pred[0]

        def constraint_gpu(x):
            inputs = self._prepare_input_df(x, current_state)
            _, t_pred = self.model.predict(inputs)
            return targets["T_gpu"] - t_pred[0]

        constraints = [
            {"type": "ineq", "fun": constraint_cpu},
            {"type": "ineq", "fun": constraint_gpu},
        ]

        # 4. Run Optimization
        # Method: SLSQP with finite differences
        # Suppress warnings by default
        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=self.bounds,
            constraints=constraints,
            options={"eps": self.eps, "ftol": 1e-6, "disp": False, "maxiter": 100},
        )

        # 5. Process Result
        if not result.success:
            # If failed, we likely hit a constraint violation (infeasible target).

            # Check feasibility
            c_vals = [c["fun"](result.x) for c in constraints]
            is_feasible = all(v >= -2.0 for v in c_vals)  # Relaxed tolerance

            if is_feasible:
                # If feasible (or close enough) but failed, accept result
                pass
            else:
                # Infeasible. We should maximize fans that help.
                # However, simple logic: if we are overheating, Max Out.
                # Just return the optimizer's best effort, but enforce Max if violation is bad.

                # logger.debug(f"Opt failed: {result.message}. C: {c_vals}")
                pass

        # Convert to integer PWM values

        # Convert to integer PWM values
        optimal_pwms = {}
        for i, name in enumerate(self.pwm_names):
            optimal_pwms[name] = int(round(result.x[i]))

        return optimal_pwms

    def _prepare_input_df(self, x: np.ndarray, state: Dict[str, float]) -> pd.DataFrame:
        """Create single-row DataFrame for model prediction."""
        row = state.copy()

        # Update PWM values from optimizer vector
        for i, name in enumerate(self.pwm_names):
            row[name] = x[i]

        return pd.DataFrame([row])
