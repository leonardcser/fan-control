"""
Evaluation stage for DVC pipeline.
Evaluates model with unified metrics: single-step, rollout, physics checks.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from fan_control.models import load_model

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def evaluate_single_step(
    model, df: pd.DataFrame
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """
    Evaluate single-step prediction accuracy.

    Returns:
        Tuple of (metrics dict, predictions dict)
        - metrics: RMSE, MAE, R² for CPU and GPU predictions.
        - predictions: actual and predicted values for plotting.
    """
    # Prepare targets - shift within episodes to get T_next
    if "T_cpu_next" not in df.columns:
        df = df.copy()
        df = df.sort_values(["episode_id", "sample_index"])
        df["T_cpu_next"] = df.groupby("episode_id")["T_cpu"].shift(-1)
        df["T_gpu_next"] = df.groupby("episode_id")["T_gpu"].shift(-1)
        df = df.dropna(subset=["T_cpu_next", "T_gpu_next"])

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

    metrics = {
        "cpu_rmse": float(np.sqrt(mean_squared_error(y_cpu_true, y_cpu_pred))),
        "cpu_mae": float(mean_absolute_error(y_cpu_true, y_cpu_pred)),
        "cpu_r2": float(r2_score(y_cpu_true, y_cpu_pred)),
        "gpu_rmse": float(np.sqrt(mean_squared_error(y_gpu_true, y_gpu_pred))),
        "gpu_mae": float(mean_absolute_error(y_gpu_true, y_gpu_pred)),
        "gpu_r2": float(r2_score(y_gpu_true, y_gpu_pred)),
    }

    predictions = {
        "cpu_true": y_cpu_true,
        "cpu_pred": y_cpu_pred,
        "gpu_true": y_gpu_true,
        "gpu_pred": y_gpu_pred,
    }

    return metrics, predictions


def evaluate_rollout(
    model, df: pd.DataFrame, horizons: List[int]
) -> Tuple[Dict[str, Dict[str, float]], Dict[int, Dict[str, np.ndarray]]]:
    """
    Evaluate multi-step rollout accuracy at different horizons.

    Starts from random initial states and rolls forward.

    Returns:
        Tuple of (metrics dict, errors dict)
        - metrics: RMSE and MAE at each horizon.
        - errors: raw error arrays for each horizon for plotting.
    """
    results = {}
    all_errors = {}

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

        all_errors[horizon] = {
            "cpu_errors": cpu_errors,
            "gpu_errors": gpu_errors,
        }

    return results, all_errors


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


def evaluate_uncertainty(
    model, df: pd.DataFrame
) -> Tuple[Dict[str, float], Optional[Dict[str, np.ndarray]]]:
    """
    Evaluate uncertainty calibration (for GP model).

    Checks if predicted uncertainty correlates with actual error.

    Returns:
        Tuple of (metrics dict, uncertainty data dict or None)
        - metrics: calibration metrics.
        - data: errors and uncertainties arrays for plotting (None if no uncertainty).
    """
    if "T_cpu_next" not in df.columns:
        df = df.copy()
        df = df.sort_values(["episode_id", "sample_index"])
        df["T_cpu_next"] = df.groupby("episode_id")["T_cpu"].shift(-1)
        df["T_gpu_next"] = df.groupby("episode_id")["T_gpu"].shift(-1)
        df = df.dropna(subset=["T_cpu_next", "T_gpu_next"])

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
        return {"has_uncertainty": False}, None

    errors = np.array(errors)
    uncertainties = np.array(uncertainties)

    # Calibration: what fraction of errors are within predicted std?
    within_1_std = np.mean(errors <= uncertainties)
    within_2_std = np.mean(errors <= 2 * uncertainties)

    # Correlation between uncertainty and error
    correlation = np.corrcoef(errors, uncertainties)[0, 1]

    metrics = {
        "has_uncertainty": True,
        "within_1_std": float(within_1_std),
        "within_2_std": float(within_2_std),
        "error_uncertainty_corr": float(correlation) if not np.isnan(correlation) else 0.0,
        "mean_uncertainty": float(np.mean(uncertainties)),
    }

    data = {
        "errors": errors,
        "uncertainties": uncertainties,
    }

    return metrics, data


def plot_predicted_vs_actual(
    predictions: Dict[str, np.ndarray],
    metrics: Dict[str, float],
    images_dir: Path,
) -> None:
    """Plot predicted vs actual temperature scatter plots."""
    _, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, (component, label) in enumerate([("cpu", "CPU"), ("gpu", "GPU")]):
        ax = axes[idx]
        y_true = predictions[f"{component}_true"]
        y_pred = predictions[f"{component}_pred"]

        # Scatter plot with seaborn
        sns.scatterplot(x=y_true, y=y_pred, alpha=0.3, s=10, color="steelblue", ax=ax)

        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect")

        # Metrics annotation
        rmse = metrics[f"{component}_rmse"]
        r2 = metrics[f"{component}_r2"]
        mae = metrics[f"{component}_mae"]
        ax.text(
            0.05,
            0.95,
            f"RMSE: {rmse:.3f}°C\nMAE: {mae:.3f}°C\nR²: {r2:.4f}",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        ax.set_xlabel("Actual Temperature (°C)")
        ax.set_ylabel("Predicted Temperature (°C)")
        ax.set_title(f"{label} Temperature: Predicted vs Actual")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    plt.savefig(images_dir / "predicted_vs_actual.png", dpi=150)
    plt.close()
    logger.info(f"Saved predicted vs actual plot to {images_dir / 'predicted_vs_actual.png'}")


def plot_residual_distribution(
    predictions: Dict[str, np.ndarray],
    images_dir: Path,
) -> None:
    """Plot residual (error) distributions."""
    _, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (component, label) in enumerate([("cpu", "CPU"), ("gpu", "GPU")]):
        ax = axes[idx]
        residuals = predictions[f"{component}_pred"] - predictions[f"{component}_true"]

        # Histogram with KDE using seaborn
        sns.histplot(residuals, bins=50, stat="density", alpha=0.7, color="steelblue", kde=True, ax=ax)

        # Add normal fit annotation
        mu, std = residuals.mean(), residuals.std()
        ax.text(
            0.95,
            0.95,
            f"μ={mu:.3f}, σ={std:.3f}",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_xlabel("Prediction Error (°C)")
        ax.set_ylabel("Density")
        ax.set_title(f"{label} Residual Distribution")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(images_dir / "residual_distribution.png", dpi=150)
    plt.close()
    logger.info(f"Saved residual distribution plot to {images_dir / 'residual_distribution.png'}")


def plot_rollout_error_vs_horizon(
    rollout_metrics: Dict[str, Dict[str, float]],
    horizons: List[int],
    images_dir: Path,
) -> None:
    """Plot RMSE vs prediction horizon."""
    _, ax = plt.subplots(figsize=(10, 6))

    cpu_rmse = []
    gpu_rmse = []
    valid_horizons = []

    for h in horizons:
        key = f"horizon_{h}"
        if key in rollout_metrics:
            cpu_rmse.append(rollout_metrics[key]["cpu_rmse"])
            gpu_rmse.append(rollout_metrics[key]["gpu_rmse"])
            valid_horizons.append(h)

    # Prepare data for seaborn
    plot_df = pd.DataFrame({
        "Horizon": valid_horizons * 2,
        "RMSE": cpu_rmse + gpu_rmse,
        "Component": ["CPU"] * len(valid_horizons) + ["GPU"] * len(valid_horizons),
    })

    sns.lineplot(data=plot_df, x="Horizon", y="RMSE", hue="Component", marker="o", markersize=8, linewidth=2, ax=ax)

    ax.set_xlabel("Prediction Horizon (steps)")
    ax.set_ylabel("RMSE (°C)")
    ax.set_title("Rollout Error vs Prediction Horizon")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(valid_horizons)

    plt.tight_layout()
    plt.savefig(images_dir / "rollout_error_vs_horizon.png", dpi=150)
    plt.close()
    logger.info(f"Saved rollout error plot to {images_dir / 'rollout_error_vs_horizon.png'}")


def plot_rollout_error_distribution(
    rollout_errors: Dict[int, Dict[str, np.ndarray]],
    images_dir: Path,
) -> None:
    """Plot box plots of rollout errors at different horizons."""
    if not rollout_errors:
        return

    horizons = sorted(rollout_errors.keys())

    _, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (component, label) in enumerate([("cpu", "CPU"), ("gpu", "GPU")]):
        ax = axes[idx]

        # Prepare data for seaborn boxplot
        box_data = []
        for h in horizons:
            for err in rollout_errors[h][f"{component}_errors"]:
                box_data.append({"Horizon": str(h), "Error": err})

        box_df = pd.DataFrame(box_data)

        sns.boxplot(data=box_df, x="Horizon", y="Error", color="lightsteelblue", ax=ax)

        ax.axhline(0, color="red", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_xlabel("Prediction Horizon (steps)")
        ax.set_ylabel("Prediction Error (°C)")
        ax.set_title(f"{label} Rollout Error Distribution")
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(images_dir / "rollout_error_distribution.png", dpi=150)
    plt.close()
    logger.info(f"Saved rollout error distribution plot to {images_dir / 'rollout_error_distribution.png'}")


def plot_physics_monotonicity(
    physics_details: Dict[str, List[float]],
    images_dir: Path,
) -> None:
    """Plot physics consistency: temperature vs PWM and power."""
    pwm_levels = [20, 40, 60, 80, 100]
    power_levels = [50, 100, 150, 200]

    _, axes = plt.subplots(2, 2, figsize=(12, 10))

    # CPU vs PWM
    ax = axes[0, 0]
    pwm_df = pd.DataFrame({"PWM": pwm_levels, "Temperature": physics_details["cpu_temps_by_pwm"]})
    sns.lineplot(data=pwm_df, x="PWM", y="Temperature", marker="o", markersize=8, linewidth=2, color="tab:blue", ax=ax)
    ax.set_xlabel("PWM (%)")
    ax.set_ylabel("Predicted Temperature (°C)")
    ax.set_title("CPU Temperature vs Fan PWM")
    ax.grid(True, alpha=0.3)

    # GPU vs PWM
    ax = axes[0, 1]
    pwm_df = pd.DataFrame({"PWM": pwm_levels, "Temperature": physics_details["gpu_temps_by_pwm"]})
    sns.lineplot(data=pwm_df, x="PWM", y="Temperature", marker="o", markersize=8, linewidth=2, color="tab:orange", ax=ax)
    ax.set_xlabel("PWM (%)")
    ax.set_ylabel("Predicted Temperature (°C)")
    ax.set_title("GPU Temperature vs Fan PWM")
    ax.grid(True, alpha=0.3)

    # CPU vs Power
    ax = axes[1, 0]
    power_df = pd.DataFrame({"Power": power_levels, "Temperature": physics_details["cpu_temps_by_power"]})
    sns.lineplot(data=power_df, x="Power", y="Temperature", marker="s", markersize=8, linewidth=2, color="tab:blue", ax=ax)
    ax.set_xlabel("Power (W)")
    ax.set_ylabel("Predicted Temperature (°C)")
    ax.set_title("CPU Temperature vs Power")
    ax.grid(True, alpha=0.3)

    # GPU vs Power
    ax = axes[1, 1]
    power_df = pd.DataFrame({"Power": power_levels, "Temperature": physics_details["gpu_temps_by_power"]})
    sns.lineplot(data=power_df, x="Power", y="Temperature", marker="s", markersize=8, linewidth=2, color="tab:orange", ax=ax)
    ax.set_xlabel("Power (W)")
    ax.set_ylabel("Predicted Temperature (°C)")
    ax.set_title("GPU Temperature vs Power")
    ax.grid(True, alpha=0.3)

    plt.suptitle("Physics Consistency: Monotonicity Checks", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(images_dir / "physics_monotonicity.png", dpi=150)
    plt.close()
    logger.info(f"Saved physics monotonicity plot to {images_dir / 'physics_monotonicity.png'}")


def plot_uncertainty_calibration(
    uncertainty_data: Dict[str, np.ndarray],
    uncertainty_metrics: Dict[str, float],
    images_dir: Path,
) -> None:
    """Plot uncertainty calibration: error vs predicted uncertainty."""
    errors = uncertainty_data["errors"]
    uncertainties = uncertainty_data["uncertainties"]

    _, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter plot: error vs uncertainty
    ax = axes[0]
    sns.scatterplot(x=uncertainties, y=errors, alpha=0.3, s=10, color="steelblue", ax=ax)
    ax.plot([0, uncertainties.max()], [0, uncertainties.max()], "r--", linewidth=2, label="Perfect calibration")
    ax.plot([0, uncertainties.max()], [0, 2 * uncertainties.max()], "g--", linewidth=1, alpha=0.7, label="2σ bound")
    ax.set_xlabel("Predicted Uncertainty (σ)")
    ax.set_ylabel("Actual Error (°C)")
    ax.set_title("Uncertainty Calibration")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Calibration summary using seaborn barplot
    ax = axes[1]
    calib_df = pd.DataFrame({
        "Category": ["Within 1σ", "Within 1σ", "Within 2σ", "Within 2σ"],
        "Type": ["Expected (Normal)", "Actual", "Expected (Normal)", "Actual"],
        "Value": [0.683, uncertainty_metrics["within_1_std"], 0.954, uncertainty_metrics["within_2_std"]],
    })

    sns.barplot(data=calib_df, x="Category", y="Value", hue="Type", palette=["lightgray", "steelblue"], ax=ax)

    ax.set_ylabel("Fraction of Samples")
    ax.set_title("Uncertainty Calibration Summary")
    ax.legend(title=None)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis="y")

    # Add correlation annotation
    corr = uncertainty_metrics["error_uncertainty_corr"]
    ax.text(
        0.95,
        0.05,
        f"Error-Uncertainty Corr: {corr:.3f}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(images_dir / "uncertainty_calibration.png", dpi=150)
    plt.close()
    logger.info(f"Saved uncertainty calibration plot to {images_dir / 'uncertainty_calibration.png'}")


def save_dvc_plot_data(
    predictions: Dict[str, np.ndarray],
    rollout_metrics: Dict[str, Dict[str, float]],
    rollout_errors: Dict[int, Dict[str, np.ndarray]],
    physics_results: Dict[str, Any],
    uncertainty_data: Optional[Dict[str, np.ndarray]],
    horizons: List[int],
    output_dir: Path,
) -> None:
    """Save JSON data for DVC plots to enable experiment comparison."""
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1. Predicted vs actual (sampled for reasonable file size)
    n_samples = len(predictions["cpu_true"])
    sample_indices = np.random.choice(n_samples, size=min(1000, n_samples), replace=False)
    sample_indices.sort()

    pred_vs_actual_data = []
    for i in sample_indices:
        pred_vs_actual_data.append({
            "cpu_actual": float(predictions["cpu_true"][i]),
            "cpu_predicted": float(predictions["cpu_pred"][i]),
            "gpu_actual": float(predictions["gpu_true"][i]),
            "gpu_predicted": float(predictions["gpu_pred"][i]),
        })

    with open(plots_dir / "predicted_vs_actual.json", "w") as f:
        json.dump(pred_vs_actual_data, f, indent=2)

    # 2. Residuals distribution (binned histogram data)
    for component in ["cpu", "gpu"]:
        residuals = predictions[f"{component}_pred"] - predictions[f"{component}_true"]
        hist, bin_edges = np.histogram(residuals, bins=50, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        residual_data = [
            {"error": float(bc), "density": float(h)}
            for bc, h in zip(bin_centers, hist)
        ]

        with open(plots_dir / f"residuals_{component}.json", "w") as f:
            json.dump(residual_data, f, indent=2)

    # 3. Rollout error vs horizon
    rollout_data = []
    for h in horizons:
        key = f"horizon_{h}"
        if key in rollout_metrics:
            rollout_data.append({
                "horizon": h,
                "cpu_rmse": rollout_metrics[key]["cpu_rmse"],
                "cpu_mae": rollout_metrics[key]["cpu_mae"],
                "gpu_rmse": rollout_metrics[key]["gpu_rmse"],
                "gpu_mae": rollout_metrics[key]["gpu_mae"],
            })

    with open(plots_dir / "rollout_by_horizon.json", "w") as f:
        json.dump(rollout_data, f, indent=2)

    # 4. Rollout error distribution per horizon
    for h, errors in rollout_errors.items():
        error_data = [
            {"cpu_error": float(cpu), "gpu_error": float(gpu)}
            for cpu, gpu in zip(errors["cpu_errors"], errors["gpu_errors"])
        ]
        with open(plots_dir / f"rollout_errors_h{h}.json", "w") as f:
            json.dump(error_data, f, indent=2)

    # 5. Physics monotonicity data
    pwm_levels = [20, 40, 60, 80, 100]
    power_levels = [50, 100, 150, 200]
    details = physics_results["details"]

    physics_pwm_data = [
        {
            "pwm": pwm,
            "cpu_temp": details["cpu_temps_by_pwm"][i],
            "gpu_temp": details["gpu_temps_by_pwm"][i],
        }
        for i, pwm in enumerate(pwm_levels)
    ]

    with open(plots_dir / "physics_pwm.json", "w") as f:
        json.dump(physics_pwm_data, f, indent=2)

    physics_power_data = [
        {
            "power": power,
            "cpu_temp": details["cpu_temps_by_power"][i],
            "gpu_temp": details["gpu_temps_by_power"][i],
        }
        for i, power in enumerate(power_levels)
    ]

    with open(plots_dir / "physics_power.json", "w") as f:
        json.dump(physics_power_data, f, indent=2)

    # 6. Uncertainty calibration (if available)
    if uncertainty_data is not None:
        # Sample for reasonable file size
        n = len(uncertainty_data["errors"])
        sample_idx = np.random.choice(n, size=min(1000, n), replace=False)
        sample_idx.sort()

        uncertainty_plot_data = [
            {
                "error": float(uncertainty_data["errors"][i]),
                "uncertainty": float(uncertainty_data["uncertainties"][i]),
            }
            for i in sample_idx
        ]

        with open(plots_dir / "uncertainty.json", "w") as f:
            json.dump(uncertainty_plot_data, f, indent=2)

    logger.info(f"Saved DVC plot data to {plots_dir}")


def generate_evaluation_plots(
    predictions: Dict[str, np.ndarray],
    single_step_metrics: Dict[str, float],
    rollout_metrics: Dict[str, Dict[str, float]],
    rollout_errors: Dict[int, Dict[str, np.ndarray]],
    physics_results: Dict[str, Any],
    uncertainty_metrics: Dict[str, float],
    uncertainty_data: Optional[Dict[str, np.ndarray]],
    horizons: List[int],
    output_dir: Path,
) -> None:
    """Generate all evaluation plots."""
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    images_dir = plots_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating evaluation plots...")

    # Save JSON data for DVC plots
    save_dvc_plot_data(
        predictions=predictions,
        rollout_metrics=rollout_metrics,
        rollout_errors=rollout_errors,
        physics_results=physics_results,
        uncertainty_data=uncertainty_data,
        horizons=horizons,
        output_dir=output_dir,
    )

    # 1. Predicted vs actual
    plot_predicted_vs_actual(predictions, single_step_metrics, images_dir)

    # 2. Residual distribution
    plot_residual_distribution(predictions, images_dir)

    # 3. Rollout error vs horizon
    plot_rollout_error_vs_horizon(rollout_metrics, horizons, images_dir)

    # 4. Rollout error distribution (box plots)
    plot_rollout_error_distribution(rollout_errors, images_dir)

    # 5. Physics monotonicity
    plot_physics_monotonicity(physics_results["details"], images_dir)

    # 6. Uncertainty calibration (if available)
    if uncertainty_data is not None:
        plot_uncertainty_calibration(uncertainty_data, uncertainty_metrics, images_dir)

    logger.info(f"All evaluation plots saved to {images_dir}")


if __name__ == "__main__":
    # Load parameters
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    model_config = params["model"]
    model_type = model_config["type"]
    eval_config = params.get("evaluate", {})
    horizons = eval_config.get("horizons", [10, 20, 50])

    data_dir = Path("data/processed")
    models_dir = Path("out/train/models") / model_type
    eval_output_dir = Path("out/evaluate")
    eval_output_dir.mkdir(parents=True, exist_ok=True)

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
    single_step, predictions = evaluate_single_step(model, val_df)
    metrics["val"]["single_step"] = single_step
    print("\n=== Single-Step Metrics ===")
    print(f"CPU: RMSE={single_step['cpu_rmse']:.2f}°C, R²={single_step['cpu_r2']:.4f}")
    print(f"GPU: RMSE={single_step['gpu_rmse']:.2f}°C, R²={single_step['gpu_r2']:.4f}")

    # 2. Rollout metrics
    logger.info(f"Evaluating rollout at horizons {horizons}...")
    rollout, rollout_errors = evaluate_rollout(model, val_df, horizons)
    metrics["val"]["rollout"] = rollout
    print("\n=== Rollout Metrics ===")
    for h, h_metrics in rollout.items():
        print(f"{h}: CPU RMSE={h_metrics['cpu_rmse']:.2f}°C, GPU RMSE={h_metrics['gpu_rmse']:.2f}°C")

    # 3. Physics consistency
    logger.info("Checking physics consistency...")
    physics = evaluate_physics_consistency(model)
    metrics["val"]["physics"] = physics
    print("\n=== Physics Consistency ===")
    print(f"PWM monotonic: CPU={physics['pwm_monotonic']['cpu']}, GPU={physics['pwm_monotonic']['gpu']}")
    print(f"Power monotonic: CPU={physics['power_monotonic']['cpu']}, GPU={physics['power_monotonic']['gpu']}")

    # 4. Uncertainty calibration (if available)
    logger.info("Evaluating uncertainty...")
    uncertainty, uncertainty_data = evaluate_uncertainty(model, val_df)
    metrics["val"]["uncertainty"] = uncertainty
    if uncertainty["has_uncertainty"]:
        print("\n=== Uncertainty Calibration ===")
        print(f"Within 1σ: {uncertainty['within_1_std']:.1%}")
        print(f"Within 2σ: {uncertainty['within_2_std']:.1%}")
        print(f"Error-uncertainty correlation: {uncertainty['error_uncertainty_corr']:.3f}")

    # 5. Generate plots
    generate_evaluation_plots(
        predictions=predictions,
        single_step_metrics=single_step,
        rollout_metrics=rollout,
        rollout_errors=rollout_errors,
        physics_results=physics,
        uncertainty_metrics=uncertainty,
        uncertainty_data=uncertainty_data,
        horizons=horizons,
        output_dir=eval_output_dir,
    )

    # Save metrics
    metrics_path = eval_output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Saved validation metrics to {metrics_path}")
