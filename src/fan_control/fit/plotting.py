"""
Validation plotting for thermal model fitting.

Generates plots to visualize model accuracy:
- Predicted vs actual temperatures
- Residual distributions and patterns
- Parameter sensitivity (from covariance)
"""

import warnings
from pathlib import Path
from typing import Dict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Use non-interactive backend for batch plotting
matplotlib.use("Agg")

# Set style
sns.set_theme(style="darkgrid")


def plot_predicted_vs_actual(
    cpu_info: Dict,
    gpu_info: Dict,
    cpu_metrics: Dict,
    gpu_metrics: Dict,
    output_path: Path,
) -> None:
    """
    Plot predicted vs actual temperatures for CPU and GPU.

    Creates 1x2 grid with scatter plots and perfect prediction line.

    Args:
        cpu_info: CPU fit info with T_measured, T_predicted
        gpu_info: GPU fit info with T_measured, T_predicted
        cpu_metrics: CPU validation metrics (for R² annotation)
        gpu_metrics: GPU validation metrics (for R² annotation)
        output_path: Path to save plot
    """
    _, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, (component, info, metrics) in enumerate(
        [
            ("CPU", cpu_info, cpu_metrics),
            ("GPU", gpu_info, gpu_metrics),
        ]
    ):
        ax = axes[idx]

        T_measured = info["T_measured"]
        T_predicted = info["T_predicted"]

        # Scatter plot
        ax.scatter(T_measured, T_predicted, alpha=0.6, s=30, label="Data points")

        # Perfect prediction line
        min_temp = min(T_measured.min(), T_predicted.min())
        max_temp = max(T_measured.max(), T_predicted.max())
        ax.plot(
            [min_temp, max_temp],
            [min_temp, max_temp],
            "r--",
            lw=2,
            label="Perfect prediction",
        )

        # Annotations
        ax.set_xlabel(f"Measured {component} Temperature (°C)", fontsize=12)
        ax.set_ylabel(f"Predicted {component} Temperature (°C)", fontsize=12)
        ax.set_title(
            f"{component} Model: Predicted vs Actual", fontsize=14, fontweight="bold"
        )
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

        # Add metrics text box
        textstr = (
            f"R² = {metrics['r2']:.4f}\n"
            f"RMSE = {metrics['rmse']:.2f} °C\n"
            f"MAE = {metrics['mae']:.2f} °C\n"
            f"n = {metrics['n_points']}"
        )
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        ax.text(
            0.95,
            0.05,
            textstr,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox=props,
        )

        # Equal aspect for square plot
        ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_residuals(
    cpu_info: Dict,
    gpu_info: Dict,
    cpu_metrics: Dict,
    gpu_metrics: Dict,
    output_path: Path,
) -> None:
    """
    Plot residual distributions and patterns.

    Creates 2x2 grid:
    - Top row: CPU residual histogram and residuals vs predicted
    - Bottom row: GPU residual histogram and residuals vs predicted

    Args:
        cpu_info: CPU fit info with residuals, T_predicted
        gpu_info: GPU fit info with residuals, T_predicted
        cpu_metrics: CPU validation metrics
        gpu_metrics: GPU validation metrics
        output_path: Path to save plot
    """
    _, axes = plt.subplots(2, 2, figsize=(14, 10))

    for row, (component, info, metrics) in enumerate(
        [
            ("CPU", cpu_info, cpu_metrics),
            ("GPU", gpu_info, gpu_metrics),
        ]
    ):
        residuals = info["residuals"]
        T_predicted = info["T_predicted"]

        # Histogram
        ax_hist = axes[row, 0]
        ax_hist.hist(residuals, bins=30, alpha=0.7, edgecolor="black")
        ax_hist.axvline(0, color="r", linestyle="--", lw=2, label="Zero residual")
        ax_hist.set_xlabel("Residual (°C)", fontsize=11)
        ax_hist.set_ylabel("Frequency", fontsize=11)
        ax_hist.set_title(
            f"{component} Residual Distribution", fontsize=12, fontweight="bold"
        )
        ax_hist.legend()
        ax_hist.grid(True, alpha=0.3)

        # Add statistics
        textstr = (
            f"Mean = {metrics['mean_residual']:.3f} °C\n"
            f"Std = {metrics['std_residual']:.2f} °C\n"
            f"Max = {metrics['max_error']:.2f} °C"
        )
        props = dict(boxstyle="round", facecolor="lightblue", alpha=0.5)
        ax_hist.text(
            0.95,
            0.95,
            textstr,
            transform=ax_hist.transAxes,
            fontsize=9,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=props,
        )

        # Residuals vs predicted
        ax_scatter = axes[row, 1]
        ax_scatter.scatter(T_predicted, residuals, alpha=0.6, s=30)
        ax_scatter.axhline(0, color="r", linestyle="--", lw=2, label="Zero residual")
        ax_scatter.set_xlabel(f"Predicted {component} Temperature (°C)", fontsize=11)
        ax_scatter.set_ylabel("Residual (°C)", fontsize=11)
        ax_scatter.set_title(
            f"{component} Residuals vs Predicted", fontsize=12, fontweight="bold"
        )
        ax_scatter.legend()
        ax_scatter.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_parameter_sensitivity(
    cpu_params: Dict[str, float],
    gpu_params: Dict[str, float],
    cpu_cov: np.ndarray,
    gpu_cov: np.ndarray,
    output_path: Path,
) -> None:
    """
    Plot parameter values with uncertainty bars from covariance.

    Creates 2x1 grid (CPU and GPU) with bar charts showing parameter values
    and error bars from the diagonal of the covariance matrix.

    Args:
        cpu_params: Fitted CPU parameters
        gpu_params: Fitted GPU parameters
        cpu_cov: CPU covariance matrix
        gpu_cov: GPU covariance matrix
        output_path: Path to save plot
    """
    _, axes = plt.subplots(2, 1, figsize=(12, 10))

    for idx, (component, params, cov) in enumerate(
        [
            ("CPU", cpu_params, cpu_cov),
            ("GPU", gpu_params, gpu_cov),
        ]
    ):
        ax = axes[idx]

        # Extract parameter names and values
        param_names = list(params.keys())
        param_values = list(params.values())

        # Extract uncertainties from covariance diagonal
        param_errors = np.sqrt(np.diag(cov))

        # Create bar chart
        x_pos = np.arange(len(param_names))
        bars = ax.bar(
            x_pos,
            param_values,
            yerr=param_errors,
            capsize=5,
            alpha=0.7,
            edgecolor="black",
        )

        # Customize
        ax.set_xlabel("Parameter", fontsize=12)
        ax.set_ylabel("Parameter Value", fontsize=12)
        ax.set_title(
            f"{component} Model Parameters with Uncertainties",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xticks(x_pos)
        ax.set_xticklabels(param_names, rotation=45, ha="right")
        ax.grid(True, axis="y", alpha=0.3)

        # Color bars by magnitude
        for bar, value in zip(bars, param_values):
            if value < 0.001:
                bar.set_color("lightcoral")
            elif value < 0.01:
                bar.set_color("lightblue")
            else:
                bar.set_color("lightgreen")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def generate_validation_plots(
    cpu_params: Dict[str, float],
    gpu_params: Dict[str, float],
    cpu_cov: np.ndarray,
    gpu_cov: np.ndarray,
    cpu_info: Dict,
    gpu_info: Dict,
    cpu_metrics: Dict,
    gpu_metrics: Dict,
    output_dir: Path,
) -> None:
    """
    Generate all validation plots and save to output directory.

    Creates:
    - predicted_vs_actual.png
    - residuals.png
    - parameter_sensitivity.png

    Args:
        cpu_params: Fitted CPU parameters
        gpu_params: Fitted GPU parameters
        cpu_cov: CPU covariance matrix
        gpu_cov: GPU covariance matrix
        cpu_info: CPU fit info
        gpu_info: GPU fit info
        cpu_metrics: CPU validation metrics
        gpu_metrics: GPU validation metrics
        output_dir: Directory to save plots
    """
    print("\nGenerating validation plots...")

    # Predicted vs actual
    plot_predicted_vs_actual(
        cpu_info,
        gpu_info,
        cpu_metrics,
        gpu_metrics,
        output_dir / "predicted_vs_actual.png",
    )

    # Residuals
    plot_residuals(
        cpu_info, gpu_info, cpu_metrics, gpu_metrics, output_dir / "residuals.png"
    )

    # Parameter sensitivity
    plot_parameter_sensitivity(
        cpu_params,
        gpu_params,
        cpu_cov,
        gpu_cov,
        output_dir / "parameter_sensitivity.png",
    )

    print("Validation plots generated successfully!")
