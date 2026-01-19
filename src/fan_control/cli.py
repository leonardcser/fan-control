"""Command-line interface for thermal data collection."""

import argparse

from dotenv import load_dotenv

from .collect.cli import collect_mode
from .fit.cli import fit_mode
from .plot.cli import plot_mode
from .control.cli import run_mode
from .control.cool import cool_mode
from .validate.cli import validate_mode


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

  # Fit thermal model from collected data
  fan-control fit --config config.yaml --run data/fan_control_20260119_040722

  # Validate/Simulate controller behavior
  fan-control validate --config config.yaml --run data/fan_control_20260119_040722

  # Run the optimized fan controller
  sudo fan-control run --config config.yaml --run data/fan_control_20260119_040722
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

    # Fit subcommand
    fit_parser = subparsers.add_parser(
        "fit",
        help="Fit thermal model parameters from collected data",
    )
    fit_parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to configuration YAML file (default: config.yaml)",
    )
    fit_parser.add_argument(
        "--run",
        required=True,
        help="Path to run directory containing collected CSV data",
    )

    # Plot subcommand
    plot_parser = subparsers.add_parser(
        "plot",
        help="Generate visualization plots from collected data",
    )
    plot_parser.add_argument(
        "--run",
        required=True,
        help="Path to run directory containing collected CSV data",
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

    # Validate subcommand
    validate_parser = subparsers.add_parser(
        "validate",
        help="Simulate controller behavior on synthetic data",
    )
    validate_parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to configuration YAML file (default: config.yaml)",
    )
    validate_parser.add_argument(
        "--run",
        required=True,
        help="Path to run directory containing the trained model",
    )

    args = parser.parse_args()

    if args.command == "fit":
        fit_mode(args)
    elif args.command == "collect":
        collect_mode(args)
    elif args.command == "plot":
        plot_mode(args)
    elif args.command == "run":
        run_mode(args)
    elif args.command == "cool":
        cool_mode(args)
    elif args.command == "validate":
        validate_mode(args)


if __name__ == "__main__":
    main()
