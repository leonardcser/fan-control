"""
CLI interface for thermal model training (ML).
"""

import sys
from pathlib import Path
import yaml
import logging

from .train import train_model

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def fit_mode(args) -> None:
    """
    Model training mode - train ML model from collected data.

    Args:
        args: Parsed command-line arguments with config and run attributes
    """
    print("\n" + "=" * 70)
    print("THERMAL MODEL TRAINING (ML)")
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

    # Create output directory
    output_dir = run_dir / "fit"

    print(f"Config: {config_path}")
    print(f"Data directory: {run_dir}")
    print(f"Output directory: {output_dir}\n")

    try:
        train_model(config, run_dir, output_dir)
        print("\n✓ Training complete!")
        print(f"✓ Model saved to: {output_dir}/thermal_model.pkl")
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
