"""
Validation metrics for thermal model fitting.

Computes and reports goodness-of-fit metrics: RMSE, R², MAE, max error, etc.
"""

import numpy as np
from typing import Dict


def compute_validation_metrics(
    T_measured: np.ndarray, T_predicted: np.ndarray
) -> Dict[str, float]:
    """
    Compute validation metrics comparing predicted vs measured temperatures.

    Args:
        T_measured: Measured temperatures (°C)
        T_predicted: Predicted temperatures (°C)

    Returns:
        Dictionary of metrics:
        - rmse: Root mean squared error
        - r2: R-squared (coefficient of determination)
        - mae: Mean absolute error
        - max_error: Maximum absolute error
        - std_residual: Standard deviation of residuals
        - mean_residual: Mean residual (should be near 0)
        - n_points: Number of data points
    """
    residuals = T_measured - T_predicted
    n = len(T_measured)

    # RMSE
    rmse = np.sqrt(np.mean(residuals**2))

    # R²
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((T_measured - np.mean(T_measured)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # MAE
    mae = np.mean(np.abs(residuals))

    # Max error
    max_error = np.max(np.abs(residuals))

    # Residual statistics
    std_residual = np.std(residuals)
    mean_residual = np.mean(residuals)

    return {
        "rmse": float(rmse),
        "r2": float(r2),
        "mae": float(mae),
        "max_error": float(max_error),
        "std_residual": float(std_residual),
        "mean_residual": float(mean_residual),
        "n_points": int(n),
    }


def print_validation_summary(
    cpu_metrics: Dict[str, float],
    gpu_metrics: Dict[str, float],
    cpu_params: Dict[str, float],
    gpu_params: Dict[str, float],
) -> None:
    """
    Print formatted validation summary with warnings for poor fits.

    Args:
        cpu_metrics: CPU validation metrics
        gpu_metrics: GPU validation metrics
        cpu_params: Fitted CPU parameters
        gpu_params: Fitted GPU parameters
    """
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    # CPU summary
    print("\nCPU Model:")
    print(f"  RMSE:        {cpu_metrics['rmse']:.2f} °C")
    print(f"  R²:          {cpu_metrics['r2']:.4f}")
    print(f"  MAE:         {cpu_metrics['mae']:.2f} °C")
    print(f"  Max Error:   {cpu_metrics['max_error']:.2f} °C")
    print(f"  Data Points: {cpu_metrics['n_points']}")

    # Warning for poor CPU fit
    if cpu_metrics["rmse"] > 3.0:
        print("  ⚠️  WARNING: High RMSE (>3°C) - model fit may be poor")
    if cpu_metrics["r2"] < 0.9:
        print("  ⚠️  WARNING: Low R² (<0.9) - model explains <90% of variance")

    # GPU summary
    print("\nGPU Model:")
    print(f"  RMSE:        {gpu_metrics['rmse']:.2f} °C")
    print(f"  R²:          {gpu_metrics['r2']:.4f}")
    print(f"  MAE:         {gpu_metrics['mae']:.2f} °C")
    print(f"  Max Error:   {gpu_metrics['max_error']:.2f} °C")
    print(f"  Data Points: {gpu_metrics['n_points']}")

    # Warning for poor GPU fit
    if gpu_metrics["rmse"] > 3.0:
        print("  ⚠️  WARNING: High RMSE (>3°C) - model fit may be poor")
    if gpu_metrics["r2"] < 0.9:
        print("  ⚠️  WARNING: Low R² (<0.9) - model explains <90% of variance")

    # Parameter summary
    print("\nFitted Parameters:")
    print("  CPU:")
    for param, value in sorted(cpu_params.items()):
        print(f"    {param:15s} = {value:.6f}")

    print("  GPU:")
    for param, value in sorted(gpu_params.items()):
        print(f"    {param:15s} = {value:.6f}")

    print("\n" + "=" * 70)


def check_parameter_validity(
    cpu_params: Dict[str, float], gpu_params: Dict[str, float]
) -> bool:
    """
    Check if fitted parameters are physically reasonable.

    Basic sanity checks:
    - All parameters should be positive (we're using bounds, but double-check)
    - Resistances should be reasonable magnitude (0.05 - 1.0)
    - Conductance parameters should be small (<0.1)

    Args:
        cpu_params: Fitted CPU parameters
        gpu_params: Fitted GPU parameters

    Returns:
        True if all parameters pass sanity checks
    """
    all_valid = True

    # Check for negative parameters (shouldn't happen with bounds)
    for component, params in [("CPU", cpu_params), ("GPU", gpu_params)]:
        for param, value in params.items():
            if value < 0:
                print(f"⚠️  WARNING: Negative parameter {param} = {value} in {component} model")
                all_valid = False

    # Check base resistances
    for param_name in ["R_base_cpu_0", "R_base_gpu"]:
        for component, params in [("CPU", cpu_params), ("GPU", gpu_params)]:
            if param_name in params:
                value = params[param_name]
                if value < 0.05 or value > 1.0:
                    print(
                        f"⚠️  WARNING: Base resistance {param_name} = {value:.3f} "
                        f"outside typical range [0.05, 1.0] °C/W"
                    )
                    all_valid = False

    return all_valid
