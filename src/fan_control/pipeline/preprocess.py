"""
Preprocessing stage for DVC pipeline.
Loads raw CSVs, filters, normalizes, and splits into train/val sets.
"""

import pandas as pd
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
import yaml
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def generate_eda_plots(df: pd.DataFrame, output_dir: Path) -> None:
    """Generate EDA plots from filtered data."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get PWM columns
    pwm_cols = [col for col in df.columns if col.startswith("pwm")]
    key_cols = pwm_cols + ["P_cpu", "P_gpu", "T_cpu", "T_gpu", "T_amb"]

    # 1. Distributions - histograms of key variables
    _, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    for i, col in enumerate(key_cols[:8]):
        if i < len(axes):
            axes[i].hist(df[col], bins=30, edgecolor="black", alpha=0.7)
            axes[i].set_xlabel(col)
            axes[i].set_ylabel("Count")
            axes[i].set_title(f"Distribution of {col}")
    plt.tight_layout()
    plt.savefig(output_dir / "distributions.png", dpi=150)
    plt.close()
    logger.info("Saved distributions.png")

    # 2. Correlation heatmap
    _, ax = plt.subplots(figsize=(10, 8))
    corr_matrix = df[key_cols].corr()
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png", dpi=150)
    plt.close()
    logger.info("Saved correlation_heatmap.png")

    # 3. Power vs Temperature scatter plots (colored by PWM)
    _, axes = plt.subplots(1, 2, figsize=(14, 5))

    # CPU: T_cpu vs P_cpu, colored by pwm2
    sc1 = axes[0].scatter(
        df["P_cpu"], df["T_cpu"], c=df[pwm_cols[0]], cmap="viridis", alpha=0.6
    )
    axes[0].set_xlabel("P_cpu (W)")
    axes[0].set_ylabel("T_cpu (°C)")
    axes[0].set_title(f"CPU Temperature vs Power (colored by {pwm_cols[0]})")
    plt.colorbar(sc1, ax=axes[0], label=f"{pwm_cols[0]} (%)")

    # GPU: T_gpu vs P_gpu, colored by pwm5 (if exists) or last pwm
    gpu_pwm = pwm_cols[-1] if len(pwm_cols) > 1 else pwm_cols[0]
    sc2 = axes[1].scatter(
        df["P_gpu"], df["T_gpu"], c=df[gpu_pwm], cmap="viridis", alpha=0.6
    )
    axes[1].set_xlabel("P_gpu (W)")
    axes[1].set_ylabel("T_gpu (°C)")
    axes[1].set_title(f"GPU Temperature vs Power (colored by {gpu_pwm})")
    plt.colorbar(sc2, ax=axes[1], label=f"{gpu_pwm} (%)")

    plt.tight_layout()
    plt.savefig(output_dir / "power_vs_temp.png", dpi=150)
    plt.close()
    logger.info("Saved power_vs_temp.png")

    # 4. PWM sweep curves - Temperature vs PWM at different power levels
    _, axes = plt.subplots(1, 2, figsize=(14, 5))

    # CPU power bins
    df["P_cpu_bin"] = pd.cut(
        df["P_cpu"], bins=5, labels=["Very Low", "Low", "Medium", "High", "Very High"]
    )
    for label, group in df.groupby("P_cpu_bin", observed=True):
        if len(group) > 0:
            sorted_group = group.sort_values(by=pwm_cols[0])
            axes[0].scatter(
                sorted_group[pwm_cols[0]],
                sorted_group["T_cpu"],
                label=f"P_cpu: {label}",
                alpha=0.6,
            )
    axes[0].set_xlabel(f"{pwm_cols[0]} (%)")
    axes[0].set_ylabel("T_cpu (°C)")
    axes[0].set_title("CPU Temp vs Fan Speed by Power Level")
    axes[0].legend()

    # GPU power bins
    df["P_gpu_bin"] = pd.cut(
        df["P_gpu"], bins=5, labels=["Very Low", "Low", "Medium", "High", "Very High"]
    )
    for label, group in df.groupby("P_gpu_bin", observed=True):
        if len(group) > 0:
            sorted_group = group.sort_values(by=gpu_pwm)
            axes[1].scatter(
                sorted_group[gpu_pwm],
                sorted_group["T_gpu"],
                label=f"P_gpu: {label}",
                alpha=0.6,
            )
    axes[1].set_xlabel(f"{gpu_pwm} (%)")
    axes[1].set_ylabel("T_gpu (°C)")
    axes[1].set_title("GPU Temp vs Fan Speed by Power Level")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_dir / "pwm_sweep_curves.png", dpi=150)
    plt.close()
    logger.info("Saved pwm_sweep_curves.png")

    # Drop temp columns
    df.drop(columns=["P_cpu_bin", "P_gpu_bin"], inplace=True)

    # 5. Box plots by workload (if description column exists)
    if "description" in df.columns:
        _, axes = plt.subplots(1, 2, figsize=(14, 6))

        df_sorted = df.copy()
        # Sort by description for better visualization
        order = list(
            df_sorted.groupby("description")["P_cpu"].mean().sort_values().index
        )

        sns.boxplot(data=df_sorted, x="description", y="T_cpu", order=order, ax=axes[0])
        axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha="right")
        axes[0].set_xlabel("Workload")
        axes[0].set_ylabel("T_cpu (°C)")
        axes[0].set_title("CPU Temperature by Workload")

        order_gpu = list(
            df_sorted.groupby("description")["P_gpu"].mean().sort_values().index
        )
        sns.boxplot(
            data=df_sorted, x="description", y="T_gpu", order=order_gpu, ax=axes[1]
        )
        axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha="right")
        axes[1].set_xlabel("Workload")
        axes[1].set_ylabel("T_gpu (°C)")
        axes[1].set_title("GPU Temperature by Workload")

        plt.tight_layout()
        plt.savefig(output_dir / "boxplots_by_workload.png", dpi=150)
        plt.close()
        logger.info("Saved boxplots_by_workload.png")

    # 6. Pairplot of key variables
    pairplot_cols = ["P_cpu", "P_gpu", "T_cpu", "T_gpu"] + pwm_cols[:2]
    pairplot_df = df[pairplot_cols].copy()
    g = sns.pairplot(pairplot_df, diag_kind="hist", plot_kws={"alpha": 0.5})
    g.figure.suptitle("Pairplot of Key Variables", y=1.02)
    plt.savefig(output_dir / "pairplot.png", dpi=150)
    plt.close()
    logger.info("Saved pairplot.png")

    logger.info(f"All EDA plots saved to {output_dir}")


if __name__ == "__main__":
    # Load parameters from params.yaml
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    preprocess_params = params["preprocess"]
    min_power_cfg = preprocess_params["min_power"]
    max_temp_cfg = preprocess_params["max_temp"]
    test_size = preprocess_params["test_size"]

    # Load config for ML parameters if needed
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    data_dir = Path("data/raw")
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load all CSV files from data/raw
    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        logger.error(f"No CSV files found in {data_dir}")
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    df_list = []
    for f in csv_files:
        try:
            df_list.append(pd.read_csv(f))
            logger.info(f"Loaded {len(df_list[-1])} rows from {f.name}")
        except Exception as e:
            logger.error(f"Failed to load {f}: {e}")

    full_df = pd.concat(df_list, ignore_index=True)
    logger.info(f"Total samples after loading: {len(full_df)}")

    # 2. Filter by minimum power
    min_cpu_power = min_power_cfg["cpu"]
    min_gpu_power = min_power_cfg["gpu"]

    df = full_df[
        (full_df["P_cpu"] > min_cpu_power) | (full_df["P_gpu"] > min_gpu_power)
    ].copy()
    logger.info(
        f"Filtered to {len(df)} samples with P_cpu > {min_cpu_power}W or P_gpu > {min_gpu_power}W"
    )

    # 3. Filter out thermally-saturated data (thermal throttling)
    max_cpu_temp = max_temp_cfg["cpu"]
    max_gpu_temp = max_temp_cfg["gpu"]

    initial_count = len(df)
    df = df[(df["T_cpu"] < max_cpu_temp) & (df["T_gpu"] < max_gpu_temp)]
    filtered_count = initial_count - len(df)
    if filtered_count > 0:
        logger.info(
            f"Filtered {filtered_count} thermal-throttling samples "
            f"(T_cpu >= {max_cpu_temp} or T_gpu >= {max_gpu_temp})"
        )

    # 4. Normalize PWM to 0-100 range (from 0-255)
    for col in df.columns:
        if col.startswith("pwm"):
            df[col] = df[col] / 2.55

    # 5. Generate EDA plots from filtered data
    plots_dir = Path("data/processed/plots")
    generate_eda_plots(df.copy(), plots_dir)

    # 6. Split into train/val
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=42)

    # 7. Save processed CSVs
    train_path = output_dir / "train.csv"
    val_path = output_dir / "val.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    logger.info(f"Saved {len(train_df)} training samples to {train_path}")
    logger.info(f"Saved {len(val_df)} validation samples to {val_path}")
