"""
CLI interface for simulation mode.
"""

import sys
import yaml
from pathlib import Path

from .simulator import run_simulation


def simulate_mode(args) -> None:
    """
    Run simulation using trained model on synthetic thermal scenarios.
    """
    print("\n" + "=" * 70)
    print("CONTROLLER SIMULATION")
    print("=" * 70 + "\n")

    # Load Config
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

    # Check Run Directory
    run_dir = Path(args.run)
    if not run_dir.exists():
        print(f"✗ Run directory not found: {run_dir}")
        sys.exit(1)

    # Locate Model
    model_path = run_dir / "fit" / "thermal_model.pkl"
    if not model_path.exists():
        print(f"✗ Model not found at: {model_path}")
        print("  Please run 'fan-control fit' first.")
        sys.exit(1)

    # Output Directory
    output_dir = run_dir / "simulate"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Config: {config_path}")
    print(f"Model: {model_path}")
    print(f"Output: {output_dir}\n")

    try:
        run_simulation(model_path, config, output_dir)
        print("\n✓ Simulation complete!")
    except Exception as e:
        print(f"\n✗ Simulation failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
