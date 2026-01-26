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

def setup_logging(debug: bool = False) -> None:
    """Configure logging to stdout."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
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

    # Load config to find model type
    if not config_path.exists():
        print(f"✗ Config file not found: {config_path}")
        sys.exit(1)

    import yaml

    with open(config_path.parent / "params.yaml") as f:
        params = yaml.safe_load(f)

    model_type = params["model"]["type"]

    # Locate model file based on DVC output path: out/train/models/${model.type}/
    model_path = Path("out/train/models") / model_type

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
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # run command
    run_parser = subparsers.add_parser("run", help="Run the fan controller optimizer")
    run_parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml)",
    )
    run_parser.set_defaults(func=run_command)

    # cool command
    cool_parser = subparsers.add_parser("cool", help="Set all fans to 100%%")
    cool_parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml)",
    )
    cool_parser.set_defaults(func=cool_mode)

    args = parser.parse_args()

    # Setup logging based on --debug flag
    setup_logging(debug=args.debug)

    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
