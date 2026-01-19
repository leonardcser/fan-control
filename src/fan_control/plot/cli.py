"""CLI interface for thermal data visualization."""

import sys
from pathlib import Path

from .plotting import generate_all_plots
from ..fit.fitter import load_and_merge_csvs


def plot_mode(args) -> None:
    """
    Visualization mode - generate plots from collected data.

    Args:
        args: Parsed command-line arguments with run attribute
    """
    print("\n" + "=" * 70)
    print("THERMAL DATA VISUALIZATION")
    print("=" * 70 + "\n")

    # Validate run directory
    run_dir = Path(args.run)
    if not run_dir.exists():
        print(f"✗ Run directory not found: {run_dir}")
        sys.exit(1)

    # Check for CSV files
    csv_files = sorted(run_dir.glob("*.csv"))
    if not csv_files:
        print(f"✗ No CSV files found in run directory: {run_dir}")
        sys.exit(1)

    print(f"Run directory: {run_dir}")
    print(f"CSV files found: {len(csv_files)}")
    for csv_file in csv_files:
        print(f"  - {csv_file.name}")

    # Load and merge data
    try:
        df = load_and_merge_csvs(run_dir)
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
    except Exception as e:
        print(f"✗ Error generating plots: {e}")
        sys.exit(1)

    print("\n✓ Visualization plots generated successfully!")
    print(f"✓ Location: {plots_dir}/")
