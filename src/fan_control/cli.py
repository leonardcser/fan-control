"""Command-line interface for thermal data collection."""

import argparse

from dotenv import load_dotenv

from .collect.cli import collect_mode
from .control.cli import run_mode
from .control.cool import cool_mode
from .train.cli import train_mode


def main() -> None:
    """Main entry point."""
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Thermal Data Collector for Physics-Based Fan Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Collect thermal data using default config
  sudo fan-control collect --config config.yaml

  # Reproduce ML pipeline (plot, fit, simulate)
  fan-control repro --config config.yaml --run data/fan_control_20260119_040722

  # Run the optimized fan controller
  sudo fan-control run --config config.yaml --run data/fan_control_20260119_040722

  # Cool down the system
  sudo fan-control cool --config config.yaml
        """,
    )

    subparsers = parser.add_subparsers(
        dest="command", help="Command to run", required=True
    )

    # Collect subcommand
    collect_parser = subparsers.add_parser(
        "collect",
        help="Collect thermal data for model fitting",
    )
    collect_parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to configuration YAML file (default: config.yaml)",
    )
    collect_parser.add_argument(
        "--output",
        help="Override output directory from config",
    )


    # Run subcommand
    run_parser = subparsers.add_parser(
        "run",
        help="Run the optimized fan controller loop",
    )
    run_parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to configuration YAML file (default: config.yaml)",
    )
    run_parser.add_argument(
        "--run",
        required=True,
        help="Path to run directory containing the trained model",
    )

    # Cool subcommand
    cool_parser = subparsers.add_parser(
        "cool",
        help="Set all fans to max speed to cool down the system",
    )
    cool_parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to configuration YAML file (default: config.yaml)",
    )

    # Repro subcommand (unified ML pipeline: plot, fit, simulate)
    repro_parser = subparsers.add_parser(
        "repro",
        help="Reproduce ML pipeline (generate plots, fit model, simulate controller)",
    )
    repro_parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to configuration YAML file (default: config.yaml)",
    )
    repro_parser.add_argument(
        "--run",
        required=True,
        help="Path to run directory containing collected CSV data",
    )

    args = parser.parse_args()

    if args.command == "collect":
        collect_mode(args)
    elif args.command == "repro":
        train_mode(args)
    elif args.command == "run":
        run_mode(args)
    elif args.command == "cool":
        cool_mode(args)


if __name__ == "__main__":
    main()
