#!/usr/bin/env python3
"""
Compare different modeling approaches for thermal prediction.

Tests:
1. Current baseline: GBM predicting T_cpu directly
2. Physics-constrained: T = T_amb + P × R(fans) where R is learned
3. Differential: Train on ΔT between fan settings at same power
4. Two-stage: Predict T_baseline from power, then ΔT from fans

Focus on CPU only. Goal is to learn meaningful fan effects, not just power relationship.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.optimize import curve_fit, minimize
import warnings

warnings.filterwarnings("ignore")

# Configuration
DATA_PATH = Path("data/fan_control_20260119_181120/fan_control_20260119_181120.csv")
OUTPUT_DIR = Path("eda/model_comparison_results")
OUTPUT_DIR.mkdir(exist_ok=True)

FEATURES = ["P_cpu", "T_amb", "pwm2", "pwm4", "pwm5", "pwm7"]
FAN_FEATURES = ["pwm2", "pwm4", "pwm5", "pwm7"]


def load_and_preprocess(path: Path) -> pd.DataFrame:
    """Load data and preprocess."""
    df = pd.read_csv(path)

    # Filter valid data
    df = df[df["P_cpu"] > 10.0].copy()

    # Normalize PWM to 0-100
    for col in FAN_FEATURES:
        df[col] = df[col] / 2.55

    # Filter thermal saturation
    df = df[df["T_cpu"] < 90.0]

    print(f"Loaded {len(df)} samples")
    print(f"P_cpu range: {df['P_cpu'].min():.0f} - {df['P_cpu'].max():.0f} W")
    print(f"T_cpu range: {df['T_cpu'].min():.0f} - {df['T_cpu'].max():.0f} °C")

    return df


def evaluate_model(y_true, y_pred, name: str) -> dict:
    """Compute evaluation metrics."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    return {"name": name, "rmse": rmse, "r2": r2, "mae": mae}


def test_fan_sensitivity(predict_fn, base_state: dict, fan_name: str) -> float:
    """Test how much temperature changes when varying a single fan."""
    low = base_state.copy()
    low[fan_name] = 20 if fan_name != "pwm7" else 40  # Use realistic minimums

    high = base_state.copy()
    high[fan_name] = 100

    t_low = predict_fn(low)
    t_high = predict_fn(high)

    return t_low - t_high  # Positive = fan helps cooling


# =============================================================================
# Model 1: Current Baseline (GBM with monotonic constraints)
# =============================================================================
class BaselineGBM:
    """Current approach: GBM predicting T_cpu directly."""

    def __init__(self):
        self.model = None
        self.features = FEATURES

    def fit(self, df: pd.DataFrame):
        X = df[self.features]
        y = df["T_cpu"]

        # Monotonic constraints: power/T_amb increase temp, fans decrease
        constraints = []
        for f in self.features:
            if f.startswith("pwm"):
                constraints.append(-1)
            elif f in ["P_cpu", "T_amb"]:
                constraints.append(1)
            else:
                constraints.append(0)

        self.model = HistGradientBoostingRegressor(
            monotonic_cst=constraints,
            max_iter=200,
            learning_rate=0.1,
            max_depth=4,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
        )
        self.model.fit(X, y)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict(df[self.features])

    def predict_single(self, state: dict) -> float:
        df = pd.DataFrame([state])
        return self.predict(df)[0]


# =============================================================================
# Model 2: Physics-Constrained (Thermal Resistance)
# =============================================================================
class PhysicsConstrainedModel:
    """
    Physics model: T_cpu = T_amb + P_cpu × R_total(fans)

    R_total is the thermal resistance, which decreases with fan speed.
    We model R as: R_base / (1 + sum(a_i × pwm_i))

    This forces fans to have a physical effect through the resistance term.
    """

    def __init__(self):
        self.params = None

    def _resistance(self, pwm2, pwm4, pwm5, pwm7, params):
        """Compute thermal resistance from fan speeds."""
        R_base, a2, a4, a5, a7, R_min = params

        # Resistance decreases with fan speed
        # Use form: R = R_min + R_base / (1 + weighted_fan_effect)
        fan_effect = a2 * pwm2 + a4 * pwm4 + a5 * pwm5 + a7 * pwm7
        R = R_min + R_base / (1 + fan_effect / 100)
        return R

    def _predict_temp(self, X, params):
        """Predict temperature using physics model."""
        P_cpu = X[:, 0]
        T_amb = X[:, 1]
        pwm2 = X[:, 2]
        pwm4 = X[:, 3]
        pwm5 = X[:, 4]
        pwm7 = X[:, 5]

        R = self._resistance(pwm2, pwm4, pwm5, pwm7, params)
        T_pred = T_amb + P_cpu * R
        return T_pred

    def fit(self, df: pd.DataFrame):
        X = df[FEATURES].values
        y = df["T_cpu"].values

        def loss(params):
            # Ensure positive parameters
            if any(p < 0 for p in params):
                return 1e10
            pred = self._predict_temp(X, params)
            return np.mean((y - pred) ** 2)

        # Initial guess: R_base, a2, a4, a5, a7, R_min
        # R ~ 0.5 °C/W is typical for air cooling
        x0 = [0.3, 1.0, 0.5, 0.5, 0.5, 0.2]

        # Bounds: all positive
        bounds = [(0.01, 2.0), (0.0, 5.0), (0.0, 5.0), (0.0, 5.0), (0.0, 5.0), (0.01, 1.0)]

        result = minimize(loss, x0, method="L-BFGS-B", bounds=bounds)
        self.params = result.x

        print(f"  Physics model params:")
        print(f"    R_base={self.params[0]:.4f}, R_min={self.params[5]:.4f}")
        print(f"    a_pwm2={self.params[1]:.4f}, a_pwm4={self.params[2]:.4f}")
        print(f"    a_pwm5={self.params[3]:.4f}, a_pwm7={self.params[4]:.4f}")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = df[FEATURES].values
        return self._predict_temp(X, self.params)

    def predict_single(self, state: dict) -> float:
        df = pd.DataFrame([state])
        return self.predict(df)[0]


# =============================================================================
# Model 3: Physics Model with Interaction Terms
# =============================================================================
class PhysicsWithInteractions:
    """
    Extended physics model with fan interactions.

    Based on the airflow diagram:
    - pwm2 (radiator) effectiveness depends on case air temp (affected by pwm4, pwm5)
    - pwm7 (pump) affects how well heat transfers to radiator
    - pwm4/pwm5 provide fresh air that pwm2 then uses

    Model: T = T_amb + P × R_total
    Where R_total = R_waterblock + R_radiator(pwm2, pwm7) + R_case(pwm4, pwm5)
    With interaction: pwm2 effectiveness boosted by pwm7 (pump-fan coupling)
    """

    def __init__(self):
        self.params = None

    def _predict_temp(self, X, params):
        P_cpu = X[:, 0]
        T_amb = X[:, 1]
        pwm2 = X[:, 2]
        pwm4 = X[:, 3]
        pwm5 = X[:, 4]
        pwm7 = X[:, 5]

        (
            R_base,  # Base thermal resistance
            k_rad,  # Radiator fan coefficient
            k_pump,  # Pump coefficient
            k_pump_rad,  # Pump-radiator interaction
            k_case,  # Case airflow coefficient (pwm4 + pwm5)
            R_min,  # Minimum achievable resistance
        ) = params

        # Radiator effectiveness (pwm2) boosted by pump (pwm7)
        rad_effect = k_rad * pwm2 * (1 + k_pump_rad * pwm7 / 100)

        # Pump direct effect on waterblock-to-radiator transfer
        pump_effect = k_pump * pwm7

        # Case airflow (pwm4 feeds GPU area and radiator, pwm5 general airflow)
        case_effect = k_case * (pwm4 + pwm5)

        # Total thermal conductance (inverse of resistance)
        conductance = rad_effect + pump_effect + case_effect

        R_total = R_min + R_base / (1 + conductance / 100)
        T_pred = T_amb + P_cpu * R_total
        return T_pred

    def fit(self, df: pd.DataFrame):
        X = df[FEATURES].values
        y = df["T_cpu"].values

        def loss(params):
            if any(p < 0 for p in params):
                return 1e10
            pred = self._predict_temp(X, params)
            return np.mean((y - pred) ** 2)

        # Initial guess
        x0 = [0.3, 1.0, 0.3, 0.5, 0.3, 0.2]
        bounds = [
            (0.01, 2.0),
            (0.0, 5.0),
            (0.0, 5.0),
            (0.0, 5.0),
            (0.0, 5.0),
            (0.01, 1.0),
        ]

        result = minimize(loss, x0, method="L-BFGS-B", bounds=bounds)
        self.params = result.x

        print(f"  Physics+Interactions params:")
        print(f"    R_base={self.params[0]:.4f}, R_min={self.params[5]:.4f}")
        print(f"    k_rad={self.params[1]:.4f}, k_pump={self.params[2]:.4f}")
        print(f"    k_pump_rad={self.params[3]:.4f}, k_case={self.params[4]:.4f}")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = df[FEATURES].values
        return self._predict_temp(X, self.params)

    def predict_single(self, state: dict) -> float:
        df = pd.DataFrame([state])
        return self.predict(df)[0]


# =============================================================================
# Model 4: Two-Stage Model
# =============================================================================
class TwoStageModel:
    """
    Two-stage approach:
    1. Learn T_baseline = f(P_cpu, T_amb) at reference fan settings
    2. Learn ΔT = g(fans) - how much fans reduce temperature

    T_cpu = T_baseline - ΔT(fans)

    This forces the model to explicitly learn fan effects.
    """

    def __init__(self):
        self.baseline_model = None
        self.delta_model = None
        self.ref_fans = {"pwm2": 50, "pwm4": 50, "pwm5": 50, "pwm7": 70}

    def fit(self, df: pd.DataFrame):
        # Stage 1: Fit baseline temperature (power relationship)
        # Use all data but predict what temp would be at reference fan settings
        # We approximate this by fitting T vs P at mid-range fan settings

        # First, fit a simple power-to-temp relationship
        self.baseline_model = HistGradientBoostingRegressor(
            monotonic_cst=[1, 1],  # Both P and T_amb increase temp
            max_iter=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42,
        )

        X_baseline = df[["P_cpu", "T_amb"]].values
        y_baseline = df["T_cpu"].values
        self.baseline_model.fit(X_baseline, y_baseline)

        # Stage 2: Fit delta from fans
        # ΔT = T_baseline_pred - T_actual (positive when fans cool below baseline)
        T_baseline_pred = self.baseline_model.predict(X_baseline)
        delta_T = T_baseline_pred - df["T_cpu"].values

        # Delta should increase with fan speed (more cooling)
        self.delta_model = HistGradientBoostingRegressor(
            monotonic_cst=[1, 1, 1, 1],  # All fans should increase cooling (positive delta)
            max_iter=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42,
        )

        X_fans = df[FAN_FEATURES].values
        self.delta_model.fit(X_fans, delta_T)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        T_baseline = self.baseline_model.predict(df[["P_cpu", "T_amb"]].values)
        delta_T = self.delta_model.predict(df[FAN_FEATURES].values)
        return T_baseline - delta_T

    def predict_single(self, state: dict) -> float:
        df = pd.DataFrame([state])
        return self.predict(df)[0]


# =============================================================================
# Model 5: Resistance Learning with GBM
# =============================================================================
class ResistanceGBM:
    """
    Learn thermal resistance R directly, then compute T = T_amb + P × R.

    R = (T_cpu - T_amb) / P_cpu

    Train GBM to predict R from fan settings with monotonic constraints
    (more fan = lower R).
    """

    def __init__(self):
        self.model = None

    def fit(self, df: pd.DataFrame):
        # Compute thermal resistance
        df = df.copy()
        df["R_thermal"] = (df["T_cpu"] - df["T_amb"]) / df["P_cpu"]

        # Filter outliers (very low power gives noisy R)
        df = df[df["P_cpu"] > 30]

        X = df[FAN_FEATURES]
        y = df["R_thermal"]

        # All fans should decrease resistance
        constraints = [-1, -1, -1, -1]

        self.model = HistGradientBoostingRegressor(
            monotonic_cst=constraints,
            max_iter=200,
            learning_rate=0.1,
            max_depth=4,
            random_state=42,
        )
        self.model.fit(X, y)

        print(f"  R_thermal range in training: {y.min():.4f} - {y.max():.4f} °C/W")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        R_pred = self.model.predict(df[FAN_FEATURES])
        T_pred = df["T_amb"].values + df["P_cpu"].values * R_pred
        return T_pred

    def predict_single(self, state: dict) -> float:
        df = pd.DataFrame([state])
        return self.predict(df)[0]


# =============================================================================
# Evaluation
# =============================================================================
def plot_predictions(models: dict, df_test: pd.DataFrame, output_dir: Path):
    """Plot predicted vs actual for all models."""
    n_models = len(models)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    y_true = df_test["T_cpu"].values

    for idx, (name, model) in enumerate(models.items()):
        if idx >= len(axes):
            break

        y_pred = model.predict(df_test)
        ax = axes[idx]

        ax.scatter(y_true, y_pred, alpha=0.5, s=20)
        ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--", lw=2)
        ax.set_xlabel("Actual T_cpu (°C)")
        ax.set_ylabel("Predicted T_cpu (°C)")

        metrics = evaluate_model(y_true, y_pred, name)
        ax.set_title(f"{name}\nRMSE={metrics['rmse']:.2f}°C, R²={metrics['r2']:.3f}")
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(len(models), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / "pred_vs_actual.png", dpi=150)
    plt.close()


def plot_fan_sensitivity(models: dict, output_dir: Path):
    """Plot temperature response to each fan for all models."""
    # Base state for testing
    base_state = {"P_cpu": 100, "T_amb": 25, "pwm2": 50, "pwm4": 30, "pwm5": 30, "pwm7": 60}

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    fan_ranges = {
        "pwm2": (20, 100),
        "pwm4": (0, 80),
        "pwm5": (0, 100),
        "pwm7": (40, 100),
    }

    for idx, (fan, (fan_min, fan_max)) in enumerate(fan_ranges.items()):
        ax = axes[idx]
        fan_values = np.linspace(fan_min, fan_max, 50)

        for name, model in models.items():
            temps = []
            for fv in fan_values:
                state = base_state.copy()
                state[fan] = fv
                temps.append(model.predict_single(state))

            ax.plot(fan_values, temps, label=name, linewidth=2)

        ax.set_xlabel(f"{fan} (%)")
        ax.set_ylabel("Predicted T_cpu (°C)")
        ax.set_title(f"Temperature vs {fan}\n(P_cpu=100W, other fans at baseline)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "fan_sensitivity.png", dpi=150)
    plt.close()


def plot_power_response(models: dict, output_dir: Path):
    """Plot temperature vs power for different fan configurations."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    power_values = np.linspace(30, 130, 50)

    # Config 1: All fans at minimum
    ax = axes[0]
    min_fans = {"T_amb": 25, "pwm2": 20, "pwm4": 15, "pwm5": 15, "pwm7": 40}
    for name, model in models.items():
        temps = []
        for P in power_values:
            state = min_fans.copy()
            state["P_cpu"] = P
            temps.append(model.predict_single(state))
        ax.plot(power_values, temps, label=name, linewidth=2)

    ax.set_xlabel("P_cpu (W)")
    ax.set_ylabel("Predicted T_cpu (°C)")
    ax.set_title("Temperature vs Power\n(All fans at MINIMUM)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Config 2: All fans at maximum
    ax = axes[1]
    max_fans = {"T_amb": 25, "pwm2": 100, "pwm4": 80, "pwm5": 100, "pwm7": 100}
    for name, model in models.items():
        temps = []
        for P in power_values:
            state = max_fans.copy()
            state["P_cpu"] = P
            temps.append(model.predict_single(state))
        ax.plot(power_values, temps, label=name, linewidth=2)

    ax.set_xlabel("P_cpu (W)")
    ax.set_ylabel("Predicted T_cpu (°C)")
    ax.set_title("Temperature vs Power\n(All fans at MAXIMUM)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "power_response.png", dpi=150)
    plt.close()


def print_fan_sensitivity_table(models: dict):
    """Print table of fan sensitivities."""
    base_state = {"P_cpu": 100, "T_amb": 25, "pwm2": 50, "pwm4": 30, "pwm5": 30, "pwm7": 60}

    print("\n" + "=" * 80)
    print("FAN SENSITIVITY (ΔT when fan goes from min to max, positive = cooling)")
    print("=" * 80)

    headers = ["Model", "pwm2", "pwm4", "pwm5", "pwm7", "Total"]
    print(f"{headers[0]:<25} {headers[1]:>8} {headers[2]:>8} {headers[3]:>8} {headers[4]:>8} {headers[5]:>8}")
    print("-" * 80)

    for name, model in models.items():
        sensitivities = []
        for fan in FAN_FEATURES:
            delta = test_fan_sensitivity(model.predict_single, base_state, fan)
            sensitivities.append(delta)

        total = sum(sensitivities)
        print(
            f"{name:<25} {sensitivities[0]:>8.2f} {sensitivities[1]:>8.2f} "
            f"{sensitivities[2]:>8.2f} {sensitivities[3]:>8.2f} {total:>8.2f}"
        )


def main():
    print("=" * 80)
    print("THERMAL MODEL COMPARISON")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    df = load_and_preprocess(DATA_PATH)

    # Split data
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    print(f"Train: {len(df_train)}, Test: {len(df_test)}")

    # Train models
    models = {}

    print("\n1. Training Baseline GBM...")
    baseline = BaselineGBM()
    baseline.fit(df_train)
    models["1. Baseline GBM"] = baseline

    print("\n2. Training Physics-Constrained Model...")
    physics = PhysicsConstrainedModel()
    physics.fit(df_train)
    models["2. Physics Simple"] = physics

    print("\n3. Training Physics + Interactions Model...")
    physics_int = PhysicsWithInteractions()
    physics_int.fit(df_train)
    models["3. Physics+Interact"] = physics_int

    print("\n4. Training Two-Stage Model...")
    two_stage = TwoStageModel()
    two_stage.fit(df_train)
    models["4. Two-Stage"] = two_stage

    print("\n5. Training Resistance GBM...")
    resistance = ResistanceGBM()
    resistance.fit(df_train)
    models["5. Resistance GBM"] = resistance

    # Evaluate
    print("\n" + "=" * 80)
    print("EVALUATION METRICS (Test Set)")
    print("=" * 80)

    y_true = df_test["T_cpu"].values
    results = []

    for name, model in models.items():
        y_pred = model.predict(df_test)
        metrics = evaluate_model(y_true, y_pred, name)
        results.append(metrics)
        print(f"{name:<25} RMSE={metrics['rmse']:>6.2f}°C  R²={metrics['r2']:.4f}  MAE={metrics['mae']:>6.2f}°C")

    # Fan sensitivity analysis
    print_fan_sensitivity_table(models)

    # Generate plots
    print("\nGenerating plots...")
    plot_predictions(models, df_test, OUTPUT_DIR)
    plot_fan_sensitivity(models, OUTPUT_DIR)
    plot_power_response(models, OUTPUT_DIR)

    print(f"\nPlots saved to {OUTPUT_DIR}/")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
Key observations to check:
1. Do physics models show larger fan sensitivity than baseline GBM?
2. Is pwm2 (radiator) the most impactful fan? (It should be for CPU cooling)
3. Does pwm7 (pump) show meaningful effect in physics models?
4. Is the total fan effect realistic (5-15°C for full sweep at 100W)?

If physics models show much larger fan effects than baseline GBM,
it suggests the GBM was learning primarily the power relationship
and treating fan variation as noise.
""")


if __name__ == "__main__":
    main()
