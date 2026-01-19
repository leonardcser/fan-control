"""
Parameter fitting using scipy.optimize.

Loads thermal data, filters for quality, and fits model parameters using
scipy.optimize.curve_fit.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, Tuple, cast
from scipy.optimize import curve_fit

from .equations import (
    ModelConfig,
    build_cpu_predictor,
    build_gpu_predictor,
)


def load_and_filter_data(
    csv_path: Path, component: str, min_power: float = 5.0
) -> pd.DataFrame:
    """
    Load data from CSV and filter for quality measurements.

    Filters applied:
    - Only equilibrated measurements (equilibrated == True)
    - Component power > min_power (default 5W to avoid sensor noise)
    - Non-null ambient temperature
    - Convert PWM from 0-255 scale to 0-1 normalized scale

    Args:
        csv_path: Path to data.csv file
        component: 'cpu' or 'gpu'
        min_power: Minimum power threshold in watts (default 5W)

    Returns:
        Filtered DataFrame with normalized PWM values
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Data file not found: {csv_path}")

    df: pd.DataFrame = pd.read_csv(csv_path)

    # Filter for equilibrated measurements
    initial_count = len(df)
    df = cast(pd.DataFrame, df[df["equilibrated"]].copy())
    equilibrated_count = len(df)

    # Filter by component power
    power_col = f"P_{component}"
    if power_col not in df.columns:
        raise ValueError(f"Missing power column: {power_col}")

    df = cast(pd.DataFrame, df[df[power_col] > min_power].copy())
    power_filtered_count = len(df)

    # Filter out missing ambient temperature
    df = cast(pd.DataFrame, df[pd.notna(df["T_amb"])].copy())
    final_count = len(df)

    # Normalize PWM values from 0-255 to 0-1
    pwm_cols = [col for col in df.columns if col.startswith("pwm")]
    for col in pwm_cols:
        df[col] = df[col] / 255.0

    print(f"Data filtering for {component.upper()}:")
    print(f"  Initial points: {initial_count}")
    print(
        f"  After equilibrated filter: {equilibrated_count} ({initial_count - equilibrated_count} dropped)"
    )
    print(
        f"  After power > {min_power}W filter: {power_filtered_count} ({equilibrated_count - power_filtered_count} dropped)"
    )
    print(
        f"  After ambient temp filter: {final_count} ({power_filtered_count - final_count} dropped)"
    )

    if final_count < 20:
        raise ValueError(
            f"Insufficient data for {component} fitting: {final_count} points "
            f"(need at least 20). Collect more data or adjust filters."
        )

    return df


def fit_component_model(
    component: str, model_config: ModelConfig, df: pd.DataFrame
) -> Tuple[Dict[str, float], np.ndarray, Dict[str, Any]]:
    """
    Fit thermal model parameters for a single component (CPU or GPU).

    Args:
        component: 'cpu' or 'gpu'
        model_config: Model configuration with structure and parameter info
        df: Filtered DataFrame with normalized PWM values (0-1 scale)

    Returns:
        Tuple of (fitted_params_dict, covariance_matrix, fit_info)
        - fitted_params_dict: {param_name: fitted_value}
        - covariance_matrix: Parameter covariance from curve_fit
        - fit_info: Additional fit information (residuals, etc.)
    """
    # Get component configuration
    if component == "cpu":
        comp_config = model_config.cpu
        predictor_builder = build_cpu_predictor
    elif component == "gpu":
        comp_config = model_config.gpu
        predictor_builder = build_gpu_predictor
    else:
        raise ValueError(f"Unknown component: {component}")

    # Get parameter names for this component
    param_names = comp_config.get_param_names()
    n_params = len(param_names)

    print(f"\nFitting {component.upper()} model with {n_params} parameters:")
    print(f"  Parameters: {param_names}")

    # Build initial guess and bounds arrays
    p0 = [model_config.initial_guesses[p] for p in param_names]
    bounds_lower = [model_config.bounds[p][0] for p in param_names]
    bounds_upper = [model_config.bounds[p][1] for p in param_names]
    bounds = (bounds_lower, bounds_upper)

    # Extract data arrays
    power = df[f"P_{component}"].values
    T_measured = df[f"T_{component}"].values
    T_amb = df["T_amb"].values

    # Get PWM columns
    pwm_cols = [col for col in df.columns if col.startswith("pwm")]

    # Build predictor function
    predictor = predictor_builder(model_config)

    # Wrapper function for curve_fit
    # curve_fit expects: f(xdata, *params) -> ydata
    # We need to convert our predictor format to this
    def curve_fit_wrapper(_xdata_dummy, *params_array):
        """
        Wrapper to convert curve_fit format to our predictor format.

        xdata_dummy is ignored - we use the data from the closure.
        params_array is the parameter values being optimized.
        """
        # Convert params array to dictionary
        params_dict = dict(zip(param_names, params_array))

        # Predict temperature for each row
        predictions = []
        for i in range(len(df)):
            # Build PWM dictionary for this row
            pwm_dict = {col: df[col].iloc[i] for col in pwm_cols}

            # Predict temperature
            T_pred = predictor(pwm_dict, power[i], T_amb[i], params_dict)
            predictions.append(T_pred)

        return np.array(predictions)

    # Perform curve fitting
    print("  Running scipy.optimize.curve_fit...")
    try:
        # Use dummy xdata (indices) since we handle data in closure
        xdata = np.arange(len(df))

        popt, pcov = curve_fit(
            curve_fit_wrapper,
            xdata,
            T_measured,
            p0=p0,
            bounds=bounds,
            maxfev=10000,  # Increase max iterations for complex models
        )

        # Convert optimized params to dictionary
        fitted_params = dict(zip(param_names, popt))

        # Calculate residuals and fit statistics
        T_predicted = curve_fit_wrapper(xdata, *popt)
        residuals = T_measured - T_predicted

        fit_info = {
            "residuals": residuals,
            "T_measured": T_measured,
            "T_predicted": T_predicted,
            "n_points": len(df),
        }

        print("  Fitting successful!")
        print("  Fitted parameters:")
        for param_name, value in fitted_params.items():
            print(f"    {param_name}: {value:.6f}")

        return fitted_params, pcov, fit_info

    except RuntimeError as e:
        raise RuntimeError(
            f"Fitting failed for {component} model. "
            "Try collecting more diverse data or adjusting parameter bounds. "
            f"Error: {e}"
        )


def fit_thermal_model(
    csv_path: Path, config: Dict
) -> Tuple[
    Dict[str, Dict[str, float]], Dict[str, np.ndarray], Dict[str, Dict[str, Any]]
]:
    """
    Main entry point for fitting thermal model parameters.

    Fits both CPU and GPU models sequentially.

    Args:
        csv_path: Path to data.csv file
        config: Full config dictionary (must contain 'model' section)

    Returns:
        Tuple of (all_fitted_params, all_covariances, all_fit_info)
        - all_fitted_params: {'cpu': {param: value}, 'gpu': {param: value}}
        - all_covariances: {'cpu': cov_matrix, 'gpu': cov_matrix}
        - all_fit_info: {'cpu': fit_info, 'gpu': fit_info}
    """
    # Parse model configuration
    model_config = ModelConfig.from_config_dict(config)

    print("=" * 70)
    print("THERMAL MODEL FITTING")
    print("=" * 70)

    # Fit CPU model
    print("\n" + "-" * 70)
    print("FITTING CPU MODEL")
    print("-" * 70)
    df_cpu = load_and_filter_data(csv_path, "cpu")
    cpu_params, cpu_cov, cpu_info = fit_component_model("cpu", model_config, df_cpu)

    # Fit GPU model
    print("\n" + "-" * 70)
    print("FITTING GPU MODEL")
    print("-" * 70)
    df_gpu = load_and_filter_data(csv_path, "gpu")
    gpu_params, gpu_cov, gpu_info = fit_component_model("gpu", model_config, df_gpu)

    # Combine results
    all_fitted_params = {"cpu": cpu_params, "gpu": gpu_params}
    all_covariances = {"cpu": cpu_cov, "gpu": gpu_cov}
    all_fit_info = {"cpu": cpu_info, "gpu": gpu_info}

    print("\n" + "=" * 70)
    print("FITTING COMPLETE")
    print("=" * 70)

    return all_fitted_params, all_covariances, all_fit_info
