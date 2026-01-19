"""
CLI interface for thermal model fitting.

Fits thermal model parameters from collected data and generates validation reports.
"""

import sys
from datetime import datetime
from pathlib import Path

import yaml

from .fitter import fit_thermal_model
from .validator import (
    compute_validation_metrics,
    print_validation_summary,
    check_parameter_validity,
)
from .plotting import generate_validation_plots


def fit_mode(args) -> None:
    """
    Model fitting mode - fit thermal model parameters from collected data.

    Args:
        args: Parsed command-line arguments with config and run attributes
    """
    print("\n" + "=" * 70)
    print("THERMAL MODEL PARAMETER FITTING")
    print("=" * 70 + "\n")

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"✗ Config file not found: {config_path}")
        sys.exit(1)

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f"✗ Error parsing config file: {e}")
        sys.exit(1)

    # Validate model section exists
    if "model" not in config:
        print("✗ Config missing 'model' section")
        print("\nPlease add model configuration to config.yaml with:")
        print("  - structure: Define which devices affect CPU/GPU")
        print("  - initial_guesses: Starting parameter values")
        print("  - bounds: Parameter constraints [lower, upper]")
        print("\nSee PHYSICS_MODEL.md for details on the model structure.")
        sys.exit(1)

    # Validate run directory
    run_dir = Path(args.run)
    if not run_dir.exists():
        print(f"✗ Run directory not found: {run_dir}")
        sys.exit(1)

    # Check for data.csv
    csv_path = run_dir / "data.csv"
    if not csv_path.exists():
        print(f"✗ Data file not found: {csv_path}")
        print(f"  Expected data.csv in run directory: {run_dir}")
        sys.exit(1)

    print(f"Config: {config_path}")
    print(f"Run directory: {run_dir}")
    print(f"Data file: {csv_path}")

    # Create output directory for fitted model
    model_dir = run_dir / "model"
    model_dir.mkdir(exist_ok=True)
    print(f"Output directory: {model_dir}\n")

    # Fit thermal model
    try:
        all_fitted_params, all_covariances, all_fit_info = fit_thermal_model(
            csv_path, config
        )
    except ValueError as e:
        print(f"\n✗ Fitting failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error during fitting: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # Extract results
    cpu_params = all_fitted_params["cpu"]
    gpu_params = all_fitted_params["gpu"]
    cpu_cov = all_covariances["cpu"]
    gpu_cov = all_covariances["gpu"]
    cpu_info = all_fit_info["cpu"]
    gpu_info = all_fit_info["gpu"]

    # Compute validation metrics
    cpu_metrics = compute_validation_metrics(
        cpu_info["T_measured"], cpu_info["T_predicted"]
    )
    gpu_metrics = compute_validation_metrics(
        gpu_info["T_measured"], gpu_info["T_predicted"]
    )

    # Print validation summary
    print_validation_summary(cpu_metrics, gpu_metrics, cpu_params, gpu_params)

    # Check parameter validity
    check_parameter_validity(cpu_params, gpu_params)

    # Generate validation plots
    generate_validation_plots(
        cpu_params,
        gpu_params,
        cpu_cov,
        gpu_cov,
        cpu_info,
        gpu_info,
        cpu_metrics,
        gpu_metrics,
        model_dir,
    )

    # Save fitted parameters to YAML
    # Convert numpy types to native Python for readable YAML
    def convert_numpy(obj):
        """Recursively convert numpy types to Python native types."""
        if isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        elif hasattr(obj, 'item'):  # numpy scalar
            return obj.item()
        else:
            return obj

    output_yaml_path = model_dir / "fitted_parameters.yaml"
    output_data = {
        "fitted_date": datetime.now().isoformat(),
        "config_source": str(config_path),
        "data_source": str(csv_path),
        "fitted_parameters": convert_numpy({**cpu_params, **gpu_params}),
        "validation_metrics": {
            "cpu": convert_numpy(cpu_metrics),
            "gpu": convert_numpy(gpu_metrics),
        },
        "model_structure": config["model"]["structure"],
    }

    with open(output_yaml_path, "w") as f:
        yaml.dump(output_data, f, default_flow_style=False, sort_keys=False)

    print(f"\n✓ Fitted parameters saved to: {output_yaml_path}")
    print(f"✓ Validation plots saved to: {model_dir}/")
    print("\nFitting complete!")
