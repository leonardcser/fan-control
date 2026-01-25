"""
Evaluation stage for DVC pipeline.
Evaluates model with unified metrics: single-step, rollout, physics checks.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from fan_control.models import load_model

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def evaluate_single_step(model, df: pd.DataFrame) -> Dict[str, float]:
    """
    Evaluate single-step prediction accuracy.

    Returns RMSE, MAE, R² for CPU and GPU predictions.
    """
    # Prepare targets
    if "T_cpu_next" not in df.columns:
        df = df.copy()
        df["T_cpu_next"] = df["T_cpu"].shift(-1)
        df["T_gpu_next"] = df["T_gpu"].shift(-1)
        df = df.iloc[:-1]

    y_cpu_true = df["T_cpu_next"].values
    y_gpu_true = df["T_gpu_next"].values

    # Predict
    y_cpu_pred = []
    y_gpu_pred = []

    for _, row in df.iterrows():
        T_k = np.array([row["T_cpu"], row["T_gpu"]])
        PWM = np.array([row["pwm2"], row["pwm4"], row["pwm5"]])
        P = np.array([row["P_cpu"], row["P_gpu"]])
        T_amb = row["T_amb"]

        T_next, _ = model.predict_next(T_k, PWM, P, T_amb)
        y_cpu_pred.append(T_next[0])
        y_gpu_pred.append(T_next[1])

    y_cpu_pred = np.array(y_cpu_pred)
    y_gpu_pred = np.array(y_gpu_pred)

    return {
        "cpu_rmse": float(np.sqrt(mean_squared_error(y_cpu_true, y_cpu_pred))),
        "cpu_mae": float(mean_absolute_error(y_cpu_true, y_cpu_pred)),
        "cpu_r2": float(r2_score(y_cpu_true, y_cpu_pred)),
        "gpu_rmse": float(np.sqrt(mean_squared_error(y_gpu_true, y_gpu_pred))),
        "gpu_mae": float(mean_absolute_error(y_gpu_true, y_gpu_pred)),
        "gpu_r2": float(r2_score(y_gpu_true, y_gpu_pred)),
    }


def evaluate_rollout(
    model, df: pd.DataFrame, horizons: List[int]
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate multi-step rollout accuracy at different horizons.

    Starts from random initial states and rolls forward.
    """
    results = {}

    for horizon in horizons:
        cpu_errors = []
        gpu_errors = []

        # Sample starting points (need at least horizon rows after)
        max_start = len(df) - horizon - 1
        if max_start <= 0:
            logger.warning(f"Not enough data for horizon {horizon}")
            continue

        # Sample up to 50 starting points
        n_samples = min(50, max_start)
        starts = np.random.choice(max_start, size=n_samples, replace=False)

        for start_idx in starts:
            # Get initial state
            row = df.iloc[start_idx]
            T_0 = np.array([row["T_cpu"], row["T_gpu"]])

            # Get sequences for this rollout
            segment = df.iloc[start_idx : start_idx + horizon + 1]
            PWM_seq = segment[["pwm2", "pwm4", "pwm5"]].values[:-1]
            P_seq = segment[["P_cpu", "P_gpu"]].values[:-1]
            T_amb = segment["T_amb"].values[0]

            # Predict trajectory
            trajectory, _ = model.predict_horizon(T_0, PWM_seq, P_seq, T_amb)

            # Get actual final temperatures
            final_row = df.iloc[start_idx + horizon]
            T_actual = np.array([final_row["T_cpu"], final_row["T_gpu"]])

            # Error at horizon end
            cpu_errors.append(trajectory[-1, 0] - T_actual[0])
            gpu_errors.append(trajectory[-1, 1] - T_actual[1])

        cpu_errors = np.array(cpu_errors)
        gpu_errors = np.array(gpu_errors)

        results[f"horizon_{horizon}"] = {
            "cpu_rmse": float(np.sqrt(np.mean(cpu_errors**2))),
            "cpu_mae": float(np.mean(np.abs(cpu_errors))),
            "gpu_rmse": float(np.sqrt(np.mean(gpu_errors**2))),
            "gpu_mae": float(np.mean(np.abs(gpu_errors))),
        }

    return results


def evaluate_physics_consistency(model, T_amb: float = 25.0) -> Dict[str, Any]:
    """
    Check physics consistency: monotonicity w.r.t. PWM and power.

    Higher PWM should lead to lower (or equal) temperature.
    Higher power should lead to higher (or equal) temperature.
    """
    # Test conditions
    T_k = np.array([60.0, 50.0])  # Starting temps
    P_base = np.array([100.0, 150.0])  # Base power

    # PWM monotonicity test
    pwm_levels = [20, 40, 60, 80, 100]
    cpu_temps_by_pwm = []
    gpu_temps_by_pwm = []

    for pwm in pwm_levels:
        PWM = np.array([pwm, pwm, pwm])
        T_next, _ = model.predict_next(T_k, PWM, P_base, T_amb)
        cpu_temps_by_pwm.append(T_next[0])
        gpu_temps_by_pwm.append(T_next[1])

    # Check monotonicity (temps should decrease or stay same as PWM increases)
    cpu_pwm_monotonic = all(
        cpu_temps_by_pwm[i] >= cpu_temps_by_pwm[i + 1]
        for i in range(len(cpu_temps_by_pwm) - 1)
    )
    gpu_pwm_monotonic = all(
        gpu_temps_by_pwm[i] >= gpu_temps_by_pwm[i + 1]
        for i in range(len(gpu_temps_by_pwm) - 1)
    )

    # Power monotonicity test
    PWM_base = np.array([50.0, 50.0, 50.0])
    power_levels = [50, 100, 150, 200]
    cpu_temps_by_power = []
    gpu_temps_by_power = []

    for p in power_levels:
        P = np.array([p, p])
        T_next, _ = model.predict_next(T_k, PWM_base, P, T_amb)
        cpu_temps_by_power.append(T_next[0])
        gpu_temps_by_power.append(T_next[1])

    # Check monotonicity (temps should increase or stay same as power increases)
    cpu_power_monotonic = all(
        cpu_temps_by_power[i] <= cpu_temps_by_power[i + 1]
        for i in range(len(cpu_temps_by_power) - 1)
    )
    gpu_power_monotonic = all(
        gpu_temps_by_power[i] <= gpu_temps_by_power[i + 1]
        for i in range(len(gpu_temps_by_power) - 1)
    )

    return {
        "pwm_monotonic": {
            "cpu": cpu_pwm_monotonic,
            "gpu": gpu_pwm_monotonic,
        },
        "power_monotonic": {
            "cpu": cpu_power_monotonic,
            "gpu": gpu_power_monotonic,
        },
        "details": {
            "cpu_temps_by_pwm": [float(t) for t in cpu_temps_by_pwm],
            "gpu_temps_by_pwm": [float(t) for t in gpu_temps_by_pwm],
            "cpu_temps_by_power": [float(t) for t in cpu_temps_by_power],
            "gpu_temps_by_power": [float(t) for t in gpu_temps_by_power],
        },
    }


def evaluate_uncertainty(model, df: pd.DataFrame) -> Dict[str, float]:
    """
    Evaluate uncertainty calibration (for GP model).

    Checks if predicted uncertainty correlates with actual error.
    """
    if "T_cpu_next" not in df.columns:
        df = df.copy()
        df["T_cpu_next"] = df["T_cpu"].shift(-1)
        df["T_gpu_next"] = df["T_gpu"].shift(-1)
        df = df.iloc[:-1]

    errors = []
    uncertainties = []

    for _, row in df.iterrows():
        T_k = np.array([row["T_cpu"], row["T_gpu"]])
        PWM = np.array([row["pwm2"], row["pwm4"], row["pwm5"]])
        P = np.array([row["P_cpu"], row["P_gpu"]])
        T_amb = row["T_amb"]

        T_next, std = model.predict_next(T_k, PWM, P, T_amb)

        if std is not None:
            error = np.abs(T_next[0] - row["T_cpu_next"])
            errors.append(error)
            uncertainties.append(std[0])

    if not errors:
        return {"has_uncertainty": False}

    errors = np.array(errors)
    uncertainties = np.array(uncertainties)

    # Calibration: what fraction of errors are within predicted std?
    within_1_std = np.mean(errors <= uncertainties)
    within_2_std = np.mean(errors <= 2 * uncertainties)

    # Correlation between uncertainty and error
    correlation = np.corrcoef(errors, uncertainties)[0, 1]

    return {
        "has_uncertainty": True,
        "within_1_std": float(within_1_std),
        "within_2_std": float(within_2_std),
        "error_uncertainty_corr": float(correlation) if not np.isnan(correlation) else 0.0,
        "mean_uncertainty": float(np.mean(uncertainties)),
    }


if __name__ == "__main__":
    # Load parameters
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    model_config = params["model"]
    model_type = model_config["type"]
    eval_config = params.get("evaluate", {})
    horizons = eval_config.get("horizons", [10, 20, 50])

    data_dir = Path("data/processed")
    models_dir = Path(f"out/models/{model_type}")
    metrics_dir = Path("out/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Load validation data
    val_path = data_dir / "val.csv"
    if not val_path.exists():
        logger.error(f"Validation data not found at {val_path}")
        raise FileNotFoundError(f"Validation data not found at {val_path}")

    val_df = pd.read_csv(val_path)
    logger.info(f"Loaded {len(val_df)} validation samples from {val_path}")

    # Load model
    if not models_dir.exists():
        logger.error(f"Model not found at {models_dir}")
        raise FileNotFoundError(f"Model not found at {models_dir}")

    model = load_model(model_type, str(models_dir), model_config)
    logger.info(f"Loaded {model_type} model from {models_dir}")

    # Evaluate
    metrics = {
        "model_type": model_type,
        "val": {},
    }

    # 1. Single-step metrics
    logger.info("Evaluating single-step predictions...")
    single_step = evaluate_single_step(model, val_df)
    metrics["val"]["single_step"] = single_step
    print(f"\n=== Single-Step Metrics ===")
    print(f"CPU: RMSE={single_step['cpu_rmse']:.2f}°C, R²={single_step['cpu_r2']:.4f}")
    print(f"GPU: RMSE={single_step['gpu_rmse']:.2f}°C, R²={single_step['gpu_r2']:.4f}")

    # 2. Rollout metrics
    logger.info(f"Evaluating rollout at horizons {horizons}...")
    rollout = evaluate_rollout(model, val_df, horizons)
    metrics["val"]["rollout"] = rollout
    print(f"\n=== Rollout Metrics ===")
    for h, h_metrics in rollout.items():
        print(f"{h}: CPU RMSE={h_metrics['cpu_rmse']:.2f}°C, GPU RMSE={h_metrics['gpu_rmse']:.2f}°C")

    # 3. Physics consistency
    logger.info("Checking physics consistency...")
    physics = evaluate_physics_consistency(model)
    metrics["val"]["physics"] = physics
    print(f"\n=== Physics Consistency ===")
    print(f"PWM monotonic: CPU={physics['pwm_monotonic']['cpu']}, GPU={physics['pwm_monotonic']['gpu']}")
    print(f"Power monotonic: CPU={physics['power_monotonic']['cpu']}, GPU={physics['power_monotonic']['gpu']}")

    # 4. Uncertainty calibration (if available)
    logger.info("Evaluating uncertainty...")
    uncertainty = evaluate_uncertainty(model, val_df)
    metrics["val"]["uncertainty"] = uncertainty
    if uncertainty["has_uncertainty"]:
        print(f"\n=== Uncertainty Calibration ===")
        print(f"Within 1σ: {uncertainty['within_1_std']:.1%}")
        print(f"Within 2σ: {uncertainty['within_2_std']:.1%}")
        print(f"Error-uncertainty correlation: {uncertainty['error_uncertainty_corr']:.3f}")

    # Save metrics
    metrics_path = metrics_dir / "val_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Saved validation metrics to {metrics_path}")
