import numpy as np
from scipy.optimize import minimize
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from ..fit.train import ThermalModel


class Optimizer:
    """
    Finds minimal fan speeds to satisfy temperature constraints using the ML model.
    Uses COBYLA with cached numpy predictions for performance.
    """

    def __init__(self, model_path: Path, config: Dict):
        self.model = ThermalModel.load(model_path)
        self.devices = config["devices"]
        self.optimizer_config = config["optimizer"]

        self.pwm_names = list(self.devices.keys())
        self.bounds = self._get_bounds()

        # Pre-compute feature order matching model's expected input
        # Model expects: [P_cpu, P_gpu, T_amb, pwm1, pwm2, ...]
        self.feature_order = self.model.feature_names_in_
        self.n_pwm = len(self.pwm_names)

        # Map pwm names to indices in feature vector
        self.pwm_indices = {name: self.feature_order.index(name) for name in self.pwm_names}

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
        """Find optimal fan speeds."""
        # Initial guess
        if initial_guess:
            x0 = np.array([float(initial_guess[name]) for name in self.pwm_names])
        else:
            x0 = np.array([b[0] for b in self.bounds])

        # Weights for objective
        if weights is None:
            w = np.ones(self.n_pwm)
        else:
            w = np.array([weights[name] for name in self.pwm_names])

        target_cpu = targets["T_cpu"]
        target_gpu = targets["T_gpu"]

        # Pre-build base feature vector with state values
        base_features = np.zeros(len(self.feature_order))
        for i, name in enumerate(self.feature_order):
            if name in current_state:
                base_features[i] = current_state[name]

        # Cache for predictions (avoid redundant model calls)
        cache = {}

        def get_temps(x: np.ndarray) -> Tuple[float, float]:
            """Get cached temperature predictions."""
            key = tuple(np.round(x, 4))
            if key not in cache:
                features = base_features.copy()
                for i, name in enumerate(self.pwm_names):
                    features[self.pwm_indices[name]] = x[i]
                t_cpu, t_gpu = self.model.predict_numpy(features)
                cache[key] = (t_cpu[0], t_gpu[0])
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

        # COBYLA - gradient-free, fewer model calls per iteration
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
