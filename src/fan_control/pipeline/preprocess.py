"""
Preprocessing stage for DVC pipeline.
Loads raw CSVs, filters, normalizes, and splits into train/val sets.

Handles time-series data with episode structure for MPC model training.
Includes data balancing strategies to address imbalanced datasets.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import yaml
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


def add_lag_features(df: pd.DataFrame, lags: list[int]) -> pd.DataFrame:
    """Add lagged temperature features within each episode.

    Args:
        df: DataFrame with episode_id, T_cpu, T_gpu columns
        lags: List of lag values (e.g., [1, 2, 5] adds T_cpu_lag1, T_cpu_lag2, etc.)

    Returns:
        DataFrame with added lag features
    """
    if "episode_id" not in df.columns:
        logger.warning("No episode_id column - cannot add lag features")
        return df

    df = df.sort_values(["episode_id", "sample_index"]).copy()

    for lag in lags:
        df[f"T_cpu_lag{lag}"] = df.groupby("episode_id")["T_cpu"].shift(lag)
        df[f"T_gpu_lag{lag}"] = df.groupby("episode_id")["T_gpu"].shift(lag)

    # Drop rows where lag features are NaN (start of each episode)
    before = len(df)
    df = df.dropna(subset=[f"T_cpu_lag{max(lags)}", f"T_gpu_lag{max(lags)}"])
    logger.info(f"Added lag features (lags={lags}), dropped {before - len(df)} rows with NaN lags")

    return df


def downsample_steady_state(
    df: pd.DataFrame, transient_threshold: float, steady_ratio: float
) -> pd.DataFrame:
    """Downsample steady-state data while keeping all transient data.

    Args:
        df: DataFrame with episode_id, T_cpu columns
        transient_threshold: |dT/dt| above this is considered transient
        steady_ratio: Fraction of steady-state samples to keep

    Returns:
        DataFrame with balanced data
    """
    if "episode_id" not in df.columns:
        logger.warning("No episode_id - using global diff for transient detection")
        df["dT_cpu"] = df["T_cpu"].diff()
    else:
        df = df.sort_values(["episode_id", "sample_index"]).copy()
        df["dT_cpu"] = df.groupby("episode_id")["T_cpu"].diff()

    # Split into transient and steady-state
    is_transient = df["dT_cpu"].abs() > transient_threshold
    transient_df = df[is_transient].copy()
    steady_df = df[~is_transient].copy()

    logger.info(
        f"Before downsampling: {len(transient_df)} transient, {len(steady_df)} steady-state"
    )

    # Randomly sample steady-state data
    n_keep = int(len(steady_df) * steady_ratio)
    if n_keep < len(steady_df):
        steady_df = steady_df.sample(n=n_keep, random_state=42)

    # Combine and drop temp column
    result = pd.concat([transient_df, steady_df], ignore_index=True)
    result = result.drop(columns=["dT_cpu"])

    logger.info(
        f"After downsampling: {len(transient_df)} transient + {len(steady_df)} steady = {len(result)} total"
    )

    return result


def remove_near_duplicates(
    df: pd.DataFrame, temp_precision: float, pwm_precision: int
) -> pd.DataFrame:
    """Remove near-duplicate rows based on rounded values.

    Args:
        df: DataFrame with T_cpu, T_gpu, pwm columns
        temp_precision: Round temperature to this precision
        pwm_precision: Round PWM to this precision

    Returns:
        DataFrame with duplicates removed
    """
    df = df.copy()

    # Create rounded columns for deduplication
    df["_T_cpu_r"] = (df["T_cpu"] / temp_precision).round() * temp_precision
    df["_T_gpu_r"] = (df["T_gpu"] / temp_precision).round() * temp_precision

    pwm_cols = [c for c in df.columns if c.startswith("pwm") and not c.startswith("_")]
    for col in pwm_cols:
        df[f"_{col}_r"] = (df[col] / pwm_precision).round() * pwm_precision

    # Also round power
    df["_P_cpu_r"] = df["P_cpu"].round()
    df["_P_gpu_r"] = df["P_gpu"].round()

    # Dedup key columns
    dedup_cols = ["_T_cpu_r", "_T_gpu_r", "_P_cpu_r", "_P_gpu_r"] + [
        f"_{c}_r" for c in pwm_cols
    ]

    before = len(df)
    df = df.drop_duplicates(subset=dedup_cols, keep="first")

    # Drop temp columns
    df = df.drop(columns=[c for c in df.columns if c.startswith("_")])

    logger.info(f"Removed {before - len(df)} near-duplicate rows ({100*(before-len(df))/before:.1f}%)")

    return df


def stratified_sample(
    df: pd.DataFrame,
    temp_bins: list[float],
    pwm_bins: list[float],
    max_per_bin: int,
) -> pd.DataFrame:
    """Stratified sampling to balance data across temp/pwm regions.

    Args:
        df: DataFrame with T_cpu, pwm columns
        temp_bins: Temperature bin edges
        pwm_bins: PWM total bin edges
        max_per_bin: Maximum samples per (temp, pwm) bin

    Returns:
        Balanced DataFrame
    """
    df = df.copy()

    # Calculate PWM total if not present
    pwm_cols = [c for c in df.columns if c.startswith("pwm")]
    df["_pwm_total"] = df[pwm_cols].sum(axis=1)

    # Create bins
    df["_T_bin"] = pd.cut(df["T_cpu"], temp_bins, labels=False)
    df["_P_bin"] = pd.cut(df["_pwm_total"], pwm_bins, labels=False)

    # Sample up to max_per_bin from each (T_bin, P_bin) combination
    sampled = []
    for (t_bin, p_bin), group in df.groupby(["_T_bin", "_P_bin"]):
        if len(group) > max_per_bin:
            sampled.append(group.sample(n=max_per_bin, random_state=42))
        else:
            sampled.append(group)

    result = pd.concat(sampled, ignore_index=True)
    result = result.drop(columns=["_pwm_total", "_T_bin", "_P_bin"])

    logger.info(f"Stratified sampling: {len(df)} -> {len(result)} samples")

    return result
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def generate_timeseries_plots(df: pd.DataFrame, output_dir: Path) -> None:
    """Generate time-series specific EDA plots."""
    if "episode_id" not in df.columns:
        logger.info("No episode_id column - skipping time-series plots")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    episodes = df["episode_id"].unique()

    # 1. Episode timeline plot (one episode per load type for representative coverage)
    if "load_description" in df.columns:
        # Sample one episode per load_description to cover different regimes
        sample_episodes = (
            df.groupby("load_description")["episode_id"]
            .first()
            .values
        )
    else:
        sample_episodes = episodes[:3]

    num_episodes = len(sample_episodes)
    fig, axes = plt.subplots(num_episodes, 1, figsize=(16, 4 * num_episodes))
    if num_episodes == 1:
        axes = [axes]

    for i, ep_id in enumerate(sample_episodes):
        ep_data = df[df["episode_id"] == ep_id].sort_values("sample_index")
        ax = axes[i]
        ax2 = ax.twinx()

        # Plot temperatures
        ax.plot(ep_data["sample_index"], ep_data["T_cpu"], "r-", label="T_cpu", lw=2)
        ax.plot(ep_data["sample_index"], ep_data["T_gpu"], "b-", label="T_gpu", lw=2)
        ax.set_ylabel("Temperature (°C)", color="red")
        ax.tick_params(axis="y", labelcolor="red")

        # Plot PWM values
        pwm_cols = [c for c in df.columns if c.startswith("pwm") and df[c].notna().any()]
        colors = ["green", "orange", "purple"]
        for j, pwm in enumerate(pwm_cols[:3]):
            ax2.step(
                ep_data["sample_index"],
                ep_data[pwm],
                where="post",
                label=pwm,
                color=colors[j % len(colors)],
                alpha=0.7,
                linestyle="--",
            )
        ax2.set_ylabel("PWM (0-100)", color="green")
        ax2.tick_params(axis="y", labelcolor="green")
        ax2.set_ylim(0, 105)

        # Title with load description
        load_desc = ep_data["load_description"].iloc[0] if "load_description" in ep_data.columns else "Unknown"
        ax.set_title(f"Episode: {ep_id[:40]}... | Load: {load_desc}", fontsize=10)
        ax.set_xlabel("Time (s)")
        ax.legend(loc="upper left")
        ax2.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "episode_timelines.png", dpi=150)
    plt.close()
    logger.info("Saved episode_timelines.png")

    # 2. PWM step response analysis
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Find PWM step changes and measure temperature response
    pwm_cols = [c for c in df.columns if c.startswith("pwm") and df[c].notna().any()]
    if len(pwm_cols) > 0:
        # Aggregate step response data across all episodes
        step_data = []
        for ep_id in episodes:
            ep_df = df[df["episode_id"] == ep_id].sort_values("sample_index").reset_index(drop=True)
            for pwm in pwm_cols:
                pwm_diff = ep_df[pwm].diff()
                step_indices = pwm_diff[pwm_diff.abs() > 5].index.tolist()
                for idx in step_indices:
                    if idx + 10 < len(ep_df) and idx > 0:
                        delta_pwm = ep_df.loc[idx, pwm] - ep_df.loc[idx - 1, pwm]
                        delta_T_cpu = ep_df.loc[idx + 10, "T_cpu"] - ep_df.loc[idx, "T_cpu"]
                        delta_T_gpu = ep_df.loc[idx + 10, "T_gpu"] - ep_df.loc[idx, "T_gpu"]
                        step_data.append({
                            "pwm": pwm,
                            "delta_pwm": delta_pwm,
                            "delta_T_cpu": delta_T_cpu,
                            "delta_T_gpu": delta_T_gpu,
                        })

        if step_data:
            step_df = pd.DataFrame(step_data)

            # CPU response to PWM changes
            for i, pwm in enumerate(pwm_cols[:2]):
                pwm_data = step_df[step_df["pwm"] == pwm]
                if len(pwm_data) > 0:
                    axes[0, i].scatter(pwm_data["delta_pwm"], pwm_data["delta_T_cpu"], alpha=0.6)
                    axes[0, i].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
                    axes[0, i].axvline(x=0, color="gray", linestyle="--", alpha=0.5)
                    axes[0, i].set_xlabel(f"Δ{pwm}")
                    axes[0, i].set_ylabel("ΔT_cpu after 10s")
                    axes[0, i].set_title(f"CPU Response to {pwm} Steps")
                    axes[0, i].grid(True, alpha=0.3)

            # GPU response to PWM changes
            for i, pwm in enumerate(pwm_cols[:2]):
                pwm_data = step_df[step_df["pwm"] == pwm]
                if len(pwm_data) > 0:
                    axes[1, i].scatter(pwm_data["delta_pwm"], pwm_data["delta_T_gpu"], alpha=0.6)
                    axes[1, i].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
                    axes[1, i].axvline(x=0, color="gray", linestyle="--", alpha=0.5)
                    axes[1, i].set_xlabel(f"Δ{pwm}")
                    axes[1, i].set_ylabel("ΔT_gpu after 10s")
                    axes[1, i].set_title(f"GPU Response to {pwm} Steps")
                    axes[1, i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "pwm_step_response.png", dpi=150)
    plt.close()
    logger.info("Saved pwm_step_response.png")

    # 3. Episode statistics summary
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    episode_stats = df.groupby("episode_id").agg({
        "T_cpu": ["mean", "std", "min", "max"],
        "T_gpu": ["mean", "std", "min", "max"],
        "sample_index": "max",
    }).reset_index()
    episode_stats.columns = ["_".join(col).strip("_") for col in episode_stats.columns]

    # Temperature range per episode
    axes[0, 0].bar(range(len(episode_stats)), episode_stats["T_cpu_max"] - episode_stats["T_cpu_min"], alpha=0.7)
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("T_cpu Range (°C)")
    axes[0, 0].set_title("CPU Temperature Range per Episode")

    axes[0, 1].bar(range(len(episode_stats)), episode_stats["T_gpu_max"] - episode_stats["T_gpu_min"], alpha=0.7, color="orange")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("T_gpu Range (°C)")
    axes[0, 1].set_title("GPU Temperature Range per Episode")

    # Sample count per episode
    axes[1, 0].bar(range(len(episode_stats)), episode_stats["sample_index_max"] + 1, alpha=0.7, color="green")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Samples")
    axes[1, 0].set_title("Samples per Episode")

    # Temperature std per episode (dynamics indicator)
    axes[1, 1].bar(range(len(episode_stats)), episode_stats["T_cpu_std"], alpha=0.7, label="T_cpu", color="red")
    axes[1, 1].bar(range(len(episode_stats)), episode_stats["T_gpu_std"], alpha=0.5, label="T_gpu", color="blue")
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].set_ylabel("Std Dev (°C)")
    axes[1, 1].set_title("Temperature Variability per Episode")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(output_dir / "episode_statistics.png", dpi=150)
    plt.close()
    logger.info("Saved episode_statistics.png")


def generate_eda_plots(df: pd.DataFrame, output_dir: Path) -> None:
    """Generate EDA plots from filtered data."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get PWM columns
    pwm_cols = [col for col in df.columns if col.startswith("pwm") and df[col].notna().any()]
    key_cols = pwm_cols + ["P_cpu", "P_gpu", "T_cpu", "T_gpu", "T_amb"]
    key_cols = [c for c in key_cols if c in df.columns]

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
    corr_matrix = df[key_cols].corr(numeric_only=True)
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
    # Support both old "description" and new "load_description" column names
    desc_col = "load_description" if "load_description" in df.columns else "description"
    if desc_col in df.columns:
        _, axes = plt.subplots(1, 2, figsize=(14, 6))

        df_sorted = df.copy()
        # Sort by description for better visualization
        order = list(
            df_sorted.groupby(desc_col)["P_cpu"].mean().sort_values(ascending=True).index
        )

        sns.boxplot(data=df_sorted, x=desc_col, y="T_cpu", order=order, ax=axes[0])
        axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha="right")
        axes[0].set_xlabel("Workload")
        axes[0].set_ylabel("T_cpu (°C)")
        axes[0].set_title("CPU Temperature by Workload")

        order_gpu = list(
            df_sorted.groupby(desc_col)["P_gpu"].mean().sort_values(ascending=True).index
        )
        sns.boxplot(
            data=df_sorted, x=desc_col, y="T_gpu", order=order_gpu, ax=axes[1]
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
    balance_cfg = preprocess_params.get("balance", {})
    lag_cfg = preprocess_params.get("lag_features", {})
    strategy = preprocess_params.get("strategy", "baseline")

    # Configure balancing based on strategy
    # Strategies: baseline, dedup, transient, stratified, combined
    logger.info(f"Using preprocessing strategy: {strategy}")
    if strategy == "baseline":
        # No balancing - use all data
        balance_cfg["remove_duplicates"]["enabled"] = False
        balance_cfg["downsample_steady_state"]["enabled"] = False
        balance_cfg["stratified_sampling"]["enabled"] = False
    elif strategy == "dedup":
        # Only remove near-duplicates
        balance_cfg["remove_duplicates"]["enabled"] = True
        balance_cfg["downsample_steady_state"]["enabled"] = False
        balance_cfg["stratified_sampling"]["enabled"] = False
    elif strategy == "transient":
        # Keep more transient data for dynamics modeling
        balance_cfg["remove_duplicates"]["enabled"] = True
        balance_cfg["downsample_steady_state"]["enabled"] = True
        balance_cfg["stratified_sampling"]["enabled"] = False
    elif strategy == "stratified":
        # Balance across temperature/PWM bins
        balance_cfg["remove_duplicates"]["enabled"] = True
        balance_cfg["downsample_steady_state"]["enabled"] = False
        balance_cfg["stratified_sampling"]["enabled"] = True
    elif strategy == "combined":
        # All strategies combined
        balance_cfg["remove_duplicates"]["enabled"] = True
        balance_cfg["downsample_steady_state"]["enabled"] = True
        balance_cfg["stratified_sampling"]["enabled"] = True

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

    # 3b. Filter out low-temperature data where fans have minimal effect
    min_temp_cfg = preprocess_params.get("min_temp", {})
    if min_temp_cfg:
        min_cpu_temp = min_temp_cfg.get("cpu", 0)
        min_gpu_temp = min_temp_cfg.get("gpu", 0)
        initial_count = len(df)
        # Keep data where either CPU or GPU is above min temp (fans can help)
        df = df[(df["T_cpu"] >= min_cpu_temp) | (df["T_gpu"] >= min_gpu_temp)]
        filtered_count = initial_count - len(df)
        if filtered_count > 0:
            logger.info(
                f"Filtered {filtered_count} low-temp samples where fans have minimal effect "
                f"(T_cpu < {min_cpu_temp} and T_gpu < {min_gpu_temp})"
            )

    # 4. Drop rows with missing values in key model input columns
    key_cols = ["T_cpu", "T_gpu", "pwm2", "pwm4", "pwm5", "P_cpu", "P_gpu", "T_amb"]
    before_drop = len(df)
    df = df.dropna(subset=key_cols)
    dropped = before_drop - len(df)
    if dropped > 0:
        logger.info(f"Dropped {dropped} rows with missing values in key columns")

    # 5. Normalize PWM to 0-100 range (from 0-255)
    for col in df.columns:
        if col.startswith("pwm"):
            df[col] = df[col] / 2.55

    # 6. Add aggregate throttling-aware features
    # CPU busy percentage (indicates throttling when high temp but low busy %)
    busy_cols = [c for c in df.columns if c.startswith("cpu_busy_pct_core")]
    if busy_cols:
        df["cpu_busy_pct"] = df[busy_cols].mean(axis=1)
        logger.info(f"Added cpu_busy_pct feature (mean of {len(busy_cols)} cores)")

    # CPU frequency (drops when throttling)
    mhz_cols = [c for c in df.columns if c.startswith("cpu_bzy_mhz_core")]
    if mhz_cols:
        df["cpu_total_mhz"] = df[mhz_cols].sum(axis=1)
        logger.info(f"Added cpu_total_mhz feature (sum of {len(mhz_cols)} cores)")

    # 6a. Apply data balancing strategies
    logger.info(f"Before balancing: {len(df)} samples")

    # Remove near-duplicates first (before other strategies)
    dedup_cfg = balance_cfg.get("remove_duplicates", {})
    if dedup_cfg.get("enabled", False):
        df = remove_near_duplicates(
            df,
            temp_precision=dedup_cfg.get("temp_precision", 0.5),
            pwm_precision=dedup_cfg.get("pwm_precision", 5),
        )

    # Downsample steady-state data
    steady_cfg = balance_cfg.get("downsample_steady_state", {})
    if steady_cfg.get("enabled", False):
        df = downsample_steady_state(
            df,
            transient_threshold=steady_cfg.get("transient_threshold", 0.3),
            steady_ratio=steady_cfg.get("steady_state_ratio", 0.3),
        )

    # Stratified sampling (usually mutually exclusive with above)
    strat_cfg = balance_cfg.get("stratified_sampling", {})
    if strat_cfg.get("enabled", False):
        df = stratified_sample(
            df,
            temp_bins=strat_cfg.get("temp_bins", [0, 60, 70, 80, 85, 90, 95]),
            pwm_bins=strat_cfg.get("pwm_bins", [0, 100, 200, 300, 500, 800]),
            max_per_bin=strat_cfg.get("max_samples_per_bin", 500),
        )

    # Add lag features for dynamics modeling
    if lag_cfg.get("enabled", False):
        df = add_lag_features(df, lags=lag_cfg.get("lags", [1, 2, 5]))

    logger.info(f"After balancing: {len(df)} samples")

    # 7. Generate EDA plots from filtered data
    plots_dir = Path("data/processed/plots")
    generate_eda_plots(df.copy(), plots_dir)

    # 7b. Generate time-series specific plots if episode data exists
    if "episode_id" in df.columns:
        generate_timeseries_plots(df.copy(), plots_dir)

    # 8. Split into train/val
    # For time-series data: split by episode to preserve temporal continuity
    # For equilibrium data: random split
    if "episode_id" in df.columns:
        episodes = df["episode_id"].unique()
        np.random.seed(42)
        np.random.shuffle(episodes)
        split_idx = int(len(episodes) * (1 - test_size))
        train_episodes = episodes[:split_idx]
        val_episodes = episodes[split_idx:]
        train_df = df[df["episode_id"].isin(train_episodes)].copy()
        val_df = df[df["episode_id"].isin(val_episodes)].copy()
        logger.info(
            f"Split by episode: {len(train_episodes)} train, {len(val_episodes)} val episodes"
        )
    else:
        from sklearn.model_selection import train_test_split
        train_df, val_df = train_test_split(df, test_size=test_size, random_state=42)

    # 9. Save processed CSVs
    train_path = output_dir / "train.csv"
    val_path = output_dir / "val.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    logger.info(f"Saved {len(train_df)} training samples to {train_path}")
    logger.info(f"Saved {len(val_df)} validation samples to {val_path}")
