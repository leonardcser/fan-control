"""Command-line interface for thermal data collection."""

import argparse

from dotenv import load_dotenv

from .collect.cli import collect_mode
from .fit.cli import fit_mode


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

  # Collect with custom output directory
  sudo fan-control collect --config config.yaml --output ./my_data

  # Fit thermal model from collected data
  fan-control fit --config config.yaml --run data/fan_control_20260119_040722

The collected data will be used to fit the thermal model parameters
described in PHYSICS_MODEL.md.
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
        help="Path to run directory containing data.csv",
    )

    args = parser.parse_args()

    if args.command == "fit":
        fit_mode(args)
    elif args.command == "collect":
        collect_mode(args)


if __name__ == "__main__":
    main()
