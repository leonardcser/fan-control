"""
CLI interface for the fan controller.
"""

import sys
import logging
from pathlib import Path
from .controller import FanController

# Configure logging to stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def run_mode(args) -> None:
    """
    Run the fan controller loop.
    """
    print("\n" + "=" * 70)
    print("FAN CONTROL - OPTIMIZER LOOP")
    print("=" * 70 + "\n")

    config_path = Path(args.config)
    run_dir = Path(args.run)

    # Locate model file
    model_path = run_dir / "fit" / "thermal_model.pkl"

    if not model_path.exists():
        print(f"✗ Model not found at: {model_path}")
        print("  Please run 'fan-control fit' first.")
        sys.exit(1)

    print(f"Config: {config_path}")
    print(f"Model: {model_path}")
    print("\nStarting controller... (Press Ctrl+C to stop)")

    try:
        controller = FanController(config_path, model_path)
        controller.start()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        sys.exit(1)
