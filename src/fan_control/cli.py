"""
CLI interface for the fan controller.
Provides commands: run, cool
"""

import argparse
import sys
import logging
from pathlib import Path

from .control.controller import FanController
from .control.cool import cool_mode

# Configure logging to stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def run_command(args) -> None:
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
        print("  Please run 'fan-control run' with a valid model path first.")
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


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Fan controller for thermal optimization",
        prog="fan-control",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # run command
    run_parser = subparsers.add_parser("run", help="Run the fan controller optimizer")
    run_parser.add_argument(
        "--config",
        required=True,
        help="Path to config.yaml",
    )
    run_parser.add_argument(
        "--run",
        required=True,
        help="Path to run directory containing fit/thermal_model.pkl",
    )
    run_parser.set_defaults(func=run_command)

    # cool command
    cool_parser = subparsers.add_parser("cool", help="Set all fans to 100%")
    cool_parser.add_argument(
        "--config",
        required=True,
        help="Path to config.yaml",
    )
    cool_parser.set_defaults(func=cool_mode)

    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
