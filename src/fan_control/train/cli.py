"""
ML pipeline orchestration: plot, fit, simulate.
"""

import sys
from pathlib import Path
import yaml
import logging

from ..fit.train import train_model
from ..plot.plotting import generate_all_plots
from ..simulate.simulator import run_simulation
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def train_mode(args) -> None:
    """
    Reproduce ML pipeline - orchestrates plot, fit, and simulate stages in sequence.

    Args:
        args: Parsed command-line arguments with config and run paths
    """
    print("\n" + "=" * 70)
    print("ML PIPELINE REPRODUCTION")
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

    # Validate ml_model section exists
    if "ml_model" not in config:
        print("✗ Config missing 'ml_model' section")
        sys.exit(1)

    # Validate run directory
    run_dir = Path(args.run)
    if not run_dir.exists():
        print(f"✗ Run directory not found: {run_dir}")
        sys.exit(1)

    print(f"Config: {config_path}")
    print(f"Run directory: {run_dir}\n")

    # Run all stages in sequence
    _run_plot_stage(run_dir)
    _run_fit_stage(config, run_dir)
    _run_simulate_stage(config, run_dir)

    print("\n" + "=" * 70)
    print("✓ Pipeline complete!")
    print("=" * 70)


def _run_fit_stage(config, run_dir):
    """Run model training stage."""
    print("-" * 70)
    print("STAGE 1: MODEL TRAINING (FIT)")
    print("-" * 70)

    output_dir = run_dir / "fit"

    print(f"Output directory: {output_dir}\n")

    try:
        train_model(config, run_dir, output_dir)
        print("✓ Training complete!")
        print(f"✓ Model saved to: {output_dir}/thermal_model.pkl\n")
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


def _run_plot_stage(run_dir):
    """Run visualization stage."""
    print("-" * 70)
    print("STAGE 2: VISUALIZATION (PLOT)")
    print("-" * 70)

    # Check for CSV files
    csv_files = sorted(run_dir.glob("*.csv"))
    if not csv_files:
        print(f"✗ No CSV files found in run directory: {run_dir}")
        sys.exit(1)

    print(f"CSV files found: {len(csv_files)}")
    for csv_file in csv_files:
        print(f"  - {csv_file.name}")

    # Load and merge data
    try:
        df_list = []
        for f in csv_files:
            df_list.append(pd.read_csv(f))
        df = pd.concat(df_list, ignore_index=True)
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        sys.exit(1)

    # Create plots directory
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    print(f"Output directory: {plots_dir}\n")

    # Generate plots
    try:
        generate_all_plots(df, plots_dir)
        print("✓ Visualization plots generated successfully!")
        print(f"✓ Location: {plots_dir}/\n")
    except Exception as e:
        print(f"✗ Error generating plots: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


def _run_simulate_stage(config, run_dir):
    """Run simulation stage."""
    print("-" * 70)
    print("STAGE 3: CONTROLLER SIMULATION")
    print("-" * 70)

    # Locate Model
    model_path = run_dir / "fit" / "thermal_model.pkl"
    if not model_path.exists():
        print(f"✗ Model not found at: {model_path}")
        print("  Please run with --fit flag first.")
        sys.exit(1)

    # Output Directory
    output_dir = run_dir / "simulate"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model: {model_path}")
    print(f"Output: {output_dir}\n")

    try:
        run_simulation(model_path, config, output_dir)
        print("✓ Simulation complete!\n")
    except Exception as e:
        print(f"\n✗ Simulation failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
