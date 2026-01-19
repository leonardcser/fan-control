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
import yaml
import argparse
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.optimize import minimize
import warnings

warnings.filterwarnings("ignore")

# Configuration
OUTPUT_DIR = Path("eda/model_comparison_results")
OUTPUT_DIR.mkdir(exist_ok=True)
CONFIG_PATH = Path("config.yaml")

# Load config to get device information
with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)

# Will be set after loading data
FEATURES = None
FAN_FEATURES = None
DEVICE_CONFIG = None


def load_and_preprocess(path: Path) -> tuple[pd.DataFrame, list[str]]:
    """Load data and preprocess. Returns (dataframe, available_targets)."""
    global FEATURES, FAN_FEATURES, DEVICE_CONFIG

    df = pd.read_csv(path)

    # Get fan PWMs from config that are enabled and exist in data
    enabled_devices = {k: v for k, v in CONFIG["devices"].items()}
    FAN_FEATURES = [k for k in enabled_devices.keys() if k in df.columns]
    FEATURES = ["P_cpu", "T_amb"] + FAN_FEATURES
    DEVICE_CONFIG = enabled_devices

    print(f"Available features: {FEATURES}")
    print(f"Available fans: {FAN_FEATURES}")

    # Detect available targets
    targets = []
    if "T_cpu" in df.columns:
        targets.append("T_cpu")
    if "T_gpu" in df.columns:
        targets.append("T_gpu")

    print(f"Available targets: {targets}")

    # Filter valid data
    df = df[df["P_cpu"] > 10.0].copy()

    # Normalize PWM to 0-100
    for col in FAN_FEATURES:
        if col in df.columns:
            df[col] = df[col] / 2.55

    # Filter thermal saturation for available targets
    for target in targets:
        df = df[df[target] < 90.0]

    print(f"Loaded {len(df)} samples")
    print(f"P_cpu range: {df['P_cpu'].min():.0f} - {df['P_cpu'].max():.0f} W")
    for target in targets:
        print(f"{target} range: {df[target].min():.0f} - {df[target].max():.0f} °C")

    return df, targets


def evaluate_model(y_true, y_pred, name: str, target: str) -> dict:
    """Compute evaluation metrics."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    return {"name": name, "rmse": rmse, "r2": r2, "mae": mae, "target": target}


def test_fan_sensitivity(predict_fn, base_state: dict, fan_name: str) -> float:
    """Test how much temperature changes when varying a single fan."""
    # Get minimum from config
    fan_min = DEVICE_CONFIG[fan_name]["min_pwm"] if fan_name in DEVICE_CONFIG else 0

    low = base_state.copy()
    low[fan_name] = fan_min

    high = base_state.copy()
    high[fan_name] = 100

    t_low = predict_fn(low)
    t_high = predict_fn(high)

    return t_low - t_high  # Positive = fan helps cooling


# =============================================================================
# Model 1: Current Baseline (GBM with monotonic constraints)
# =============================================================================
class BaselineGBM:
    """Current approach: GBM predicting temperature directly."""

    def __init__(self, target: str = "T_cpu"):
        self.model = None
        self.features = FEATURES
        self.target = target

    def fit(self, df: pd.DataFrame):
        X = df[self.features]
        y = df[self.target]

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
    Physics model: T = T_amb + P_cpu × R_total(fans)

    R_total is the thermal resistance, which decreases with fan speed.
    We model R as: R_base / (1 + sum(a_i × pwm_i))

    This forces fans to have a physical effect through the resistance term.
    """

    def __init__(self, target: str = "T_cpu"):
        self.params = None
        self.n_fans = None
        self.target = target

    def _resistance(self, fan_speeds, params):
        """Compute thermal resistance from fan speeds."""
        R_base = params[0]
        fan_coeffs = params[1:-1]
        R_min = params[-1]

        # Resistance decreases with fan speed
        # Use form: R = R_min + R_base / (1 + weighted_fan_effect)
        fan_effect = sum(a * pwm for a, pwm in zip(fan_coeffs, fan_speeds))
        R = R_min + R_base / (1 + fan_effect / 100)
        return R

    def _predict_temp(self, X, params):
        """Predict temperature using physics model."""
        P_cpu = X[:, 0]
        T_amb = X[:, 1]
        fan_speeds = [X[:, i] for i in range(2, X.shape[1])]

        R = self._resistance(fan_speeds, params)
        T_pred = T_amb + P_cpu * R
        return T_pred

    def fit(self, df: pd.DataFrame):
        X = df[FEATURES].values
        y = df[self.target].values
        self.n_fans = len(FAN_FEATURES)

        def loss(params):
            # Ensure positive parameters
            if any(p < 0 for p in params):
                return 1e10
            pred = self._predict_temp(X, params)
            return np.mean((y - pred) ** 2)

        # Initial guess: R_base, a_fan1, a_fan2, ..., R_min
        # R ~ 0.5 °C/W is typical for air cooling
        x0 = [0.3] + [1.0] * self.n_fans + [0.2]

        # Bounds: all positive
        bounds = [(0.01, 2.0)] + [(0.0, 5.0)] * self.n_fans + [(0.01, 1.0)]

        result = minimize(loss, x0, method="L-BFGS-B", bounds=bounds)
        self.params = result.x

        print("  Physics model params:")
        print(f"    R_base={self.params[0]:.4f}, R_min={self.params[-1]:.4f}")
        for i, fan in enumerate(FAN_FEATURES):
            print(f"    a_{fan}={self.params[i+1]:.4f}")

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
    - pwm2 (radiator) primary cooling effect
    - pwm4/pwm5 (case fans) provide fresh air that helps radiator

    Model: T = T_amb + P × R_total
    Where R_total considers both direct fan effects and interactions
    """

    def __init__(self, target: str = "T_cpu"):
        self.params = None
        self.n_fans = None
        self.target = target

    def _predict_temp(self, X, params):
        P_cpu = X[:, 0]
        T_amb = X[:, 1]
        fan_speeds = [X[:, i] for i in range(2, X.shape[1])]

        R_base = params[0]
        fan_coeffs = params[1:1+self.n_fans]
        R_min = params[-1]

        # Direct fan effects
        fan_effect = sum(k * pwm for k, pwm in zip(fan_coeffs, fan_speeds))

        # Total thermal conductance (inverse of resistance)
        conductance = fan_effect

        R_total = R_min + R_base / (1 + conductance / 100)
        T_pred = T_amb + P_cpu * R_total
        return T_pred

    def fit(self, df: pd.DataFrame):
        X = df[FEATURES].values
        y = df[self.target].values
        self.n_fans = len(FAN_FEATURES)

        def loss(params):
            if any(p < 0 for p in params):
                return 1e10
            pred = self._predict_temp(X, params)
            return np.mean((y - pred) ** 2)

        # Initial guess: R_base, k_fan1, k_fan2, ..., R_min
        x0 = [0.3] + [1.0] * self.n_fans + [0.2]
        bounds = [(0.01, 2.0)] + [(0.0, 5.0)] * self.n_fans + [(0.01, 1.0)]

        result = minimize(loss, x0, method="L-BFGS-B", bounds=bounds)
        self.params = result.x

        print("  Physics+Interactions params:")
        print(f"    R_base={self.params[0]:.4f}, R_min={self.params[-1]:.4f}")
        for i, fan in enumerate(FAN_FEATURES):
            print(f"    k_{fan}={self.params[i+1]:.4f}")

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

    T = T_baseline - ΔT(fans)

    This forces the model to explicitly learn fan effects.
    """

    def __init__(self, target: str = "T_cpu"):
        self.baseline_model = None
        self.delta_model = None
        self.target = target
        self.ref_fans = {"pwm2": 50, "pwm4": 50, "pwm5": 50}

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
        y_baseline = df[self.target].values
        self.baseline_model.fit(X_baseline, y_baseline)

        # Stage 2: Fit delta from fans
        # ΔT = T_baseline_pred - T_actual (positive when fans cool below baseline)
        T_baseline_pred = self.baseline_model.predict(X_baseline)
        delta_T = T_baseline_pred - df[self.target].values

        # Delta should increase with fan speed (more cooling)
        # Create monotonic constraints for however many fans we have
        fan_constraints = [1] * len(FAN_FEATURES)

        self.delta_model = HistGradientBoostingRegressor(
            monotonic_cst=fan_constraints,
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

    R = (T - T_amb) / P_cpu

    Train GBM to predict R from fan settings with monotonic constraints
    (more fan = lower R).
    """

    def __init__(self, target: str = "T_cpu"):
        self.model = None
        self.target = target

    def fit(self, df: pd.DataFrame):
        # Compute thermal resistance
        df = df.copy()
        df["R_thermal"] = (df[self.target] - df["T_amb"]) / df["P_cpu"]

        # Filter outliers (very low power gives noisy R)
        df = df[df["P_cpu"] > 30]

        X = df[FAN_FEATURES]
        y = df["R_thermal"]

        # All fans should decrease resistance
        constraints = [-1] * len(FAN_FEATURES)

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
def plot_predictions(models: dict, df_test: pd.DataFrame, output_dir: Path, target: str):
    """Plot predicted vs actual for all models."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    y_true = df_test[target].values

    for idx, (name, model) in enumerate(models.items()):
        if idx >= len(axes):
            break

        y_pred = model.predict(df_test)
        ax = axes[idx]

        ax.scatter(y_true, y_pred, alpha=0.5, s=20)
        ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--", lw=2)
        ax.set_xlabel(f"Actual {target} (°C)")
        ax.set_ylabel(f"Predicted {target} (°C)")

        metrics = evaluate_model(y_true, y_pred, name, target)
        ax.set_title(f"{name}\nRMSE={metrics['rmse']:.2f}°C, R²={metrics['r2']:.3f}")
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(len(models), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / f"pred_vs_actual_{target}.png", dpi=150)
    plt.close()


def plot_fan_sensitivity(models: dict, output_dir: Path, target: str):
    """Plot temperature response to each fan for all models."""
    # Base state for testing - use middle of range for each fan
    base_state = {"P_cpu": 100, "T_amb": 25}
    for fan in FAN_FEATURES:
        levels = CONFIG["data_collection"].get(f"{fan}_levels", [0, 100])
        base_state[fan] = (min(levels) + max(levels)) / 2

    # Get fan ranges from config data_collection levels
    fan_ranges = {}
    for fan in FAN_FEATURES:
        levels = CONFIG["data_collection"].get(f"{fan}_levels", [0, 100])
        fan_ranges[fan] = (min(levels), max(levels))

    n_plots = len(fan_ranges)
    n_cols = min(2, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5 * n_rows))
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

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
        ax.set_ylabel(f"Predicted {target} (°C)")
        ax.set_title(f"Temperature vs {fan}\n(P_cpu=100W, other fans at baseline)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f"fan_sensitivity_{target}.png", dpi=150)
    plt.close()


def plot_power_response(models: dict, output_dir: Path, target: str):
    """Plot temperature vs power for different fan configurations."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    power_values = np.linspace(30, 130, 50)

    # Config 1: All fans at minimum - use min_pwm from config
    ax = axes[0]
    min_fans = {"T_amb": 25}
    for fan in FAN_FEATURES:
        min_fans[fan] = DEVICE_CONFIG[fan]["min_pwm"]

    for name, model in models.items():
        temps = []
        for P in power_values:
            state = min_fans.copy()
            state["P_cpu"] = P
            temps.append(model.predict_single(state))
        ax.plot(power_values, temps, label=name, linewidth=2)

    ax.set_xlabel("P_cpu (W)")
    ax.set_ylabel(f"Predicted {target} (°C)")
    ax.set_title("Temperature vs Power\n(All fans at MINIMUM)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Config 2: All fans at maximum - use max from data_collection levels
    ax = axes[1]
    max_fans = {"T_amb": 25}
    for fan in FAN_FEATURES:
        levels = CONFIG["data_collection"].get(f"{fan}_levels", [100])
        max_fans[fan] = max(levels)

    for name, model in models.items():
        temps = []
        for P in power_values:
            state = max_fans.copy()
            state["P_cpu"] = P
            temps.append(model.predict_single(state))
        ax.plot(power_values, temps, label=name, linewidth=2)

    ax.set_xlabel("P_cpu (W)")
    ax.set_ylabel(f"Predicted {target} (°C)")
    ax.set_title("Temperature vs Power\n(All fans at MAXIMUM)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f"power_response_{target}.png", dpi=150)
    plt.close()


def print_fan_sensitivity_table(models: dict, target: str):
    """Print table of fan sensitivities."""
    base_state = {"P_cpu": 100, "T_amb": 25}
    # Set fans to mid-range values
    for fan in FAN_FEATURES:
        levels = CONFIG["data_collection"].get(f"{fan}_levels", [0, 100])
        base_state[fan] = (min(levels) + max(levels)) / 2

    print("\n" + "=" * 80)
    print(f"FAN SENSITIVITY FOR {target} (ΔT when fan goes from min to max, positive = cooling)")
    print("=" * 80)

    headers = ["Model"] + FAN_FEATURES + ["Total"]
    header_str = f"{headers[0]:<25}"
    for h in headers[1:]:
        header_str += f" {h:>8}"
    print(header_str)
    print("-" * 80)

    for name, model in models.items():
        sensitivities = []
        for fan in FAN_FEATURES:
            delta = test_fan_sensitivity(model.predict_single, base_state, fan)
            sensitivities.append(delta)

        total = sum(sensitivities)
        row_str = f"{name:<25}"
        for s in sensitivities:
            row_str += f" {s:>8.2f}"
        row_str += f" {total:>8.2f}"
        print(row_str)


def run_analysis_for_target(df: pd.DataFrame, target: str, output_dir: Path):
    """Run complete analysis for a single target (T_cpu or T_gpu)."""
    print("\n" + "=" * 80)
    print(f"ANALYSIS FOR {target}")
    print("=" * 80)

    # Split data
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    print(f"Train: {len(df_train)}, Test: {len(df_test)}")

    # Train models
    models = {}

    print(f"\n1. Training Baseline GBM for {target}...")
    baseline = BaselineGBM(target=target)
    baseline.fit(df_train)
    models["1. Baseline GBM"] = baseline

    print(f"\n2. Training Physics-Constrained Model for {target}...")
    physics = PhysicsConstrainedModel(target=target)
    physics.fit(df_train)
    models["2. Physics Simple"] = physics

    print(f"\n3. Training Physics + Interactions Model for {target}...")
    physics_int = PhysicsWithInteractions(target=target)
    physics_int.fit(df_train)
    models["3. Physics+Interact"] = physics_int

    print(f"\n4. Training Two-Stage Model for {target}...")
    two_stage = TwoStageModel(target=target)
    two_stage.fit(df_train)
    models["4. Two-Stage"] = two_stage

    print(f"\n5. Training Resistance GBM for {target}...")
    resistance = ResistanceGBM(target=target)
    resistance.fit(df_train)
    models["5. Resistance GBM"] = resistance

    # Evaluate
    print("\n" + "=" * 80)
    print(f"EVALUATION METRICS FOR {target} (Test Set)")
    print("=" * 80)

    y_true = df_test[target].values
    results = []

    for name, model in models.items():
        y_pred = model.predict(df_test)
        metrics = evaluate_model(y_true, y_pred, name, target)
        results.append(metrics)
        print(f"{name:<25} RMSE={metrics['rmse']:>6.2f}°C  R²={metrics['r2']:.4f}  MAE={metrics['mae']:>6.2f}°C")

    # Fan sensitivity analysis
    print_fan_sensitivity_table(models, target)

    # Generate plots
    print(f"\nGenerating plots for {target}...")
    plot_predictions(models, df_test, output_dir, target)
    plot_fan_sensitivity(models, output_dir, target)
    plot_power_response(models, output_dir, target)

    return results


def main():
    parser = argparse.ArgumentParser(description="Compare thermal models")
    parser.add_argument("--run", required=True, help="Run directory path (e.g., data/fan_control_20260119_201646)")
    args = parser.parse_args()

    # Construct data path
    run_dir = Path(args.run)
    run_name = run_dir.name
    data_path = run_dir / f"{run_name}.csv"

    print("=" * 80)
    print("THERMAL MODEL COMPARISON")
    print("=" * 80)

    # Load data
    print(f"\nLoading data from {data_path}...")
    df, targets = load_and_preprocess(data_path)

    # Run analysis for each available target
    all_results = []
    for target in targets:
        results = run_analysis_for_target(df, target, OUTPUT_DIR)
        all_results.extend(results)

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
