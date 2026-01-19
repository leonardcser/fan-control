"""
Plotting utilities for thermal data visualization.

This module contains all plotting functions for visualizing thermal data
collected during model fitting.
"""

import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from scipy.interpolate import griddata
from tqdm import tqdm

# Suppress warnings for cleaner output during data collection
warnings.filterwarnings("ignore")

# Use non-interactive backend for batch plotting
matplotlib.use("Agg")

# Set style
sns.set_theme(style="darkgrid")
plt.rcParams["figure.figsize"] = (14, 10)

# Load config for device information
CONFIG_PATH = Path(__file__).parent.parent.parent.parent / "config.yaml"
with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)


def plot_correlation_matrix(df: pd.DataFrame) -> plt.Figure:
    """Plot correlation matrix for all key variables"""
    fig, ax = plt.subplots(figsize=(12, 10))

    # Key variables for physics model (only include those that exist)
    # Get PWM columns from config
    pwm_cols = list(CONFIG["devices"].keys())
    key_vars = pwm_cols + ["P_cpu", "P_gpu", "T_amb", "T_cpu", "T_gpu"]
    key_vars = [v for v in key_vars if v in df.columns]

    # Compute correlation matrix
    corr = df[key_vars].corr()

    # Create heatmap
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=1,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )

    ax.set_title(
        "Correlation Matrix - Thermal System Variables",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()

    return fig


def plot_temp_vs_fans(df: pd.DataFrame) -> plt.Figure:
    """Plot component temperatures vs fan speeds"""
    # Get fans from config that exist in data
    all_fans = [
        (dev_id, f"{dev_info['name']}\n({dev_id})")
        for dev_id, dev_info in CONFIG["devices"].items()
    ]
    fans_available = [(f, n) for f, n in all_fans if f in df.columns]
    fans, fan_names = zip(*fans_available) if fans_available else ([], [])

    fig, axes = plt.subplots(2, max(len(fans), 1), figsize=(5 * len(fans), 10))

    # Handle case with single or multiple fans
    if len(fans) == 1:
        axes = axes.reshape(2, 1)
    elif axes.ndim == 1:
        axes = axes.reshape(2, -1)

    # CPU temperature vs each fan
    for i, (fan, name) in enumerate(zip(fans, fan_names)):
        axes[0, i].scatter(
            df[fan], df["T_cpu"], alpha=0.6, s=50, c=df["P_cpu"], cmap="viridis"
        )
        axes[0, i].set_xlabel(f"{name} PWM", fontsize=11)
        axes[0, i].set_ylabel("CPU Temp (°C)", fontsize=11)
        axes[0, i].set_title(f"CPU Temp vs {name}", fontsize=12, fontweight="bold")
        axes[0, i].grid(True, alpha=0.3)

        # Add colorbar for power
        if i == 3:
            sm = plt.cm.ScalarMappable(
                cmap="viridis",
                norm=plt.Normalize(vmin=df["P_cpu"].min(), vmax=df["P_cpu"].max()),
            )
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=axes[0, i])
            cbar.set_label("CPU Power (W)", fontsize=10)

    # GPU temperature vs each fan
    for i, (fan, name) in enumerate(zip(fans, fan_names)):
        axes[1, i].scatter(
            df[fan], df["T_gpu"], alpha=0.6, s=50, c=df["P_gpu"], cmap="plasma"
        )
        axes[1, i].set_xlabel(f"{name} PWM", fontsize=11)
        axes[1, i].set_ylabel("GPU Temp (°C)", fontsize=11)
        axes[1, i].set_title(f"GPU Temp vs {name}", fontsize=12, fontweight="bold")
        axes[1, i].grid(True, alpha=0.3)

        # Add colorbar for power
        if i == 3:
            sm = plt.cm.ScalarMappable(
                cmap="plasma",
                norm=plt.Normalize(vmin=df["P_gpu"].min(), vmax=df["P_gpu"].max()),
            )
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=axes[1, i])
            cbar.set_label("GPU Power (W)", fontsize=10)

    plt.tight_layout()
    return fig


def plot_temp_vs_power(df: pd.DataFrame) -> plt.Figure:
    """Plot temperatures vs power consumption"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # CPU temperature vs CPU power
    scatter1 = axes[0].scatter(
        df["P_cpu"], df["T_cpu"], c=df["pwm2"], s=100, alpha=0.6, cmap="coolwarm"
    )
    axes[0].set_xlabel("CPU Power (W)", fontsize=12)
    axes[0].set_ylabel("CPU Temperature (°C)", fontsize=12)
    axes[0].set_title(
        "CPU: Temperature vs Power\n(color = radiator fan speed)",
        fontsize=13,
        fontweight="bold",
    )
    axes[0].grid(True, alpha=0.3)
    cbar1 = plt.colorbar(scatter1, ax=axes[0])
    cbar1.set_label("Radiator Fan PWM", fontsize=10)

    # GPU temperature vs GPU power
    scatter2 = axes[1].scatter(
        df["P_gpu"], df["T_gpu"], c=df["pwm4"], s=100, alpha=0.6, cmap="coolwarm"
    )
    axes[1].set_xlabel("GPU Power (W)", fontsize=12)
    axes[1].set_ylabel("GPU Temperature (°C)", fontsize=12)
    axes[1].set_title(
        "GPU: Temperature vs Power\n(color = bottom intake speed)",
        fontsize=13,
        fontweight="bold",
    )
    axes[1].grid(True, alpha=0.3)
    cbar2 = plt.colorbar(scatter2, ax=axes[1])
    cbar2.set_label("Bottom Intake PWM", fontsize=10)

    plt.tight_layout()
    return fig


def plot_pump_radiator_interaction(df: pd.DataFrame) -> plt.Figure:
    """Plot fan interaction effects on component temperature"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Get fan interaction configuration
    interaction_cfg = CONFIG.get("plot_config", {}).get("interaction_plot", {})

    if not interaction_cfg.get("enabled", False):
        msg = "Fan interaction plot not enabled in config"
        axes[0].text(0.5, 0.5, msg, ha="center", va="center", fontsize=14)
        axes[0].set_title("Fan Interaction Effect", fontsize=13, fontweight="bold")
        axes[1].text(0.5, 0.5, msg, ha="center", va="center", fontsize=14)
        axes[1].set_title("Fan Interaction Effect", fontsize=13, fontweight="bold")
        plt.tight_layout()
        return fig

    fan1 = interaction_cfg.get("fan1")
    fan2 = interaction_cfg.get("fan2")
    component = interaction_cfg.get("component", "cpu")
    temp_col = f"T_{component}"

    # Check if fans exist in data
    if fan1 not in df.columns or fan2 not in df.columns:
        msg = f"Fan interaction data not available\n({fan1} or {fan2} missing)"
        axes[0].text(0.5, 0.5, msg, ha="center", va="center", fontsize=14)
        axes[0].set_title("Fan Interaction Effect", fontsize=13, fontweight="bold")
        axes[1].text(0.5, 0.5, msg, ha="center", va="center", fontsize=14)
        axes[1].set_title("Fan Interaction Effect", fontsize=13, fontweight="bold")
        plt.tight_layout()
        return fig

    # Get device names from config
    fan1_name = CONFIG["devices"][fan1]["name"]
    fan2_name = CONFIG["devices"][fan2]["name"]

    # Plot 1: temp vs fan1, colored by fan2
    scatter1 = axes[0].scatter(
        df[fan1], df[temp_col], c=df[fan2], s=100, alpha=0.6, cmap="viridis"
    )
    axes[0].set_xlabel(f"{fan1_name} ({fan1})", fontsize=12)
    axes[0].set_ylabel(f"{component.upper()} Temperature (°C)", fontsize=12)
    axes[0].set_title(
        f"Fan Interaction Effect\n{component.upper()} Temp vs {fan1_name} (color = {fan2_name})",
        fontsize=13,
        fontweight="bold",
    )
    axes[0].grid(True, alpha=0.3)
    cbar1 = plt.colorbar(scatter1, ax=axes[0])
    cbar1.set_label(f"{fan2_name} ({fan2})", fontsize=10)

    # Plot 2: temp vs fan2, colored by fan1
    scatter2 = axes[1].scatter(
        df[fan2], df[temp_col], c=df[fan1], s=100, alpha=0.6, cmap="plasma"
    )
    axes[1].set_xlabel(f"{fan2_name} ({fan2})", fontsize=12)
    axes[1].set_ylabel(f"{component.upper()} Temperature (°C)", fontsize=12)
    axes[1].set_title(
        f"Fan Interaction Effect\n{component.upper()} Temp vs {fan2_name} (color = {fan1_name})",
        fontsize=13,
        fontweight="bold",
    )
    axes[1].grid(True, alpha=0.3)
    cbar2 = plt.colorbar(scatter2, ax=axes[1])
    cbar2.set_label(f"{fan1_name} ({fan1})", fontsize=10)

    plt.tight_layout()
    return fig


def plot_thermal_resistance(df: pd.DataFrame) -> plt.Figure:
    """Plot effective thermal resistance: (T - T_amb) / Power"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Get plot config
    resistance_cfg = CONFIG.get("plot_config", {}).get("thermal_resistance", {})
    cpu_cfg = resistance_cfg.get("cpu", {"x_axis": "pwm2", "color_by": None})

    # CPU thermal resistance
    df_cpu = df[df["P_cpu"] > 0].copy()
    df_cpu["R_cpu"] = (df_cpu["T_cpu"] - df_cpu["T_amb"]) / df_cpu["P_cpu"]

    x_axis = cpu_cfg.get("x_axis", "pwm2")
    color_by = cpu_cfg.get("color_by")

    if x_axis in df_cpu.columns:
        x_name = CONFIG["devices"][x_axis]["name"]

        if color_by and color_by in df_cpu.columns:
            color_data = df_cpu[color_by]
            color_name = CONFIG["devices"][color_by]["name"]
            title_suffix = f"(color = {color_name})"
            cbar_label = f"{color_name} ({color_by})"
        else:
            color_data = df_cpu["P_cpu"]
            title_suffix = "(color = CPU power)"
            cbar_label = "CPU Power (W)"

        scatter1 = axes[0].scatter(
            df_cpu[x_axis],
            df_cpu["R_cpu"],
            c=color_data,
            s=100,
            alpha=0.6,
            cmap="coolwarm",
        )
        axes[0].set_xlabel(f"{x_name} ({x_axis})", fontsize=12)
        axes[0].set_ylabel("CPU Thermal Resistance (°C/W)", fontsize=12)
        axes[0].set_title(
            f"CPU Thermal Resistance vs {x_name}\n{title_suffix}",
            fontsize=13,
            fontweight="bold",
        )
        axes[0].grid(True, alpha=0.3)
        cbar1 = plt.colorbar(scatter1, ax=axes[0])
        cbar1.set_label(cbar_label, fontsize=10)
    else:
        axes[0].text(
            0.5,
            0.5,
            f"Fan {x_axis} not available",
            ha="center",
            va="center",
            fontsize=12,
        )
        axes[0].set_title("CPU Thermal Resistance", fontsize=13, fontweight="bold")

    # GPU thermal resistance
    gpu_cfg = resistance_cfg.get("gpu", {"x_axis": "pwm4", "color_by": None})
    df_gpu = df[df["P_gpu"] > 0].copy()

    if len(df_gpu) > 0:
        df_gpu["R_gpu"] = (df_gpu["T_gpu"] - df_gpu["T_amb"]) / df_gpu["P_gpu"]

        x_axis_gpu = gpu_cfg.get("x_axis", "pwm4")
        color_by_gpu = gpu_cfg.get("color_by")

        if x_axis_gpu in df_gpu.columns:
            x_name_gpu = CONFIG["devices"][x_axis_gpu]["name"]

            if color_by_gpu and color_by_gpu in df_gpu.columns:
                color_data_gpu = df_gpu[color_by_gpu]
                color_name_gpu = CONFIG["devices"][color_by_gpu]["name"]
                title_suffix_gpu = f"(color = {color_name_gpu})"
                cbar_label_gpu = f"{color_name_gpu} ({color_by_gpu})"
            else:
                color_data_gpu = df_gpu["P_gpu"]
                title_suffix_gpu = "(color = GPU power)"
                cbar_label_gpu = "GPU Power (W)"

            scatter2 = axes[1].scatter(
                df_gpu[x_axis_gpu],
                df_gpu["R_gpu"],
                c=color_data_gpu,
                s=100,
                alpha=0.6,
                cmap="coolwarm",
            )
            axes[1].set_xlabel(f"{x_name_gpu} ({x_axis_gpu})", fontsize=12)
            axes[1].set_ylabel("GPU Thermal Resistance (°C/W)", fontsize=12)
            axes[1].set_title(
                f"GPU Thermal Resistance vs {x_name_gpu}\n{title_suffix_gpu}",
                fontsize=13,
                fontweight="bold",
            )
            axes[1].grid(True, alpha=0.3)
            cbar2 = plt.colorbar(scatter2, ax=axes[1])
            cbar2.set_label(cbar_label_gpu, fontsize=10)
        else:
            axes[1].text(
                0.5,
                0.5,
                f"Fan {x_axis_gpu} not available",
                ha="center",
                va="center",
                fontsize=12,
            )
            axes[1].set_title("GPU Thermal Resistance", fontsize=13, fontweight="bold")
    else:
        axes[1].text(
            0.5,
            0.5,
            "No GPU load data available",
            ha="center",
            va="center",
            fontsize=14,
        )
        axes[1].set_title("GPU Thermal Resistance", fontsize=13, fontweight="bold")

    plt.tight_layout()
    return fig


def plot_pairwise_interactions(df: pd.DataFrame) -> plt.Figure:
    """Create pairplot for key variables"""
    # Get all PWM columns from config plus key metrics
    pwm_cols = [dev_id for dev_id in CONFIG["devices"].keys() if dev_id in df.columns]
    key_vars = pwm_cols + ["P_cpu", "T_cpu"]
    key_vars = [v for v in key_vars if v in df.columns]

    # Sample if too many points
    df_plot = df[key_vars].copy()
    if len(df_plot) > 100:
        df_plot = df_plot.sample(100, random_state=42)

    g = sns.pairplot(df_plot, diag_kind="kde", plot_kws={"alpha": 0.6, "s": 40})
    g.figure.suptitle(
        "Pairwise Variable Interactions", y=1.01, fontsize=16, fontweight="bold"
    )

    return g.figure


def plot_cooling_effectiveness(df: pd.DataFrame) -> plt.Figure:
    """Plot cooling effectiveness: how much temperature changes per PWM unit"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Calculate normalized temperature (remove ambient)
    df_work = df.copy()
    df_work["T_cpu_norm"] = df_work["T_cpu"] - df_work["T_amb"]
    df_work["T_gpu_norm"] = df_work["T_gpu"] - df_work["T_amb"]

    # Get cooling effectiveness config
    effectiveness_cfg = CONFIG.get("plot_config", {}).get("cooling_effectiveness", {})
    cpu_fans = effectiveness_cfg.get("cpu_fans", ["pwm2"])
    gpu_fans = effectiveness_cfg.get("gpu_fans", ["pwm4", "pwm5"])

    # CPU cooling effectiveness - plot first fan
    if len(cpu_fans) > 0 and cpu_fans[0] in df_work.columns:
        cpu_fan = cpu_fans[0]
        cpu_fan_name = CONFIG["devices"][cpu_fan]["name"]

        df_sorted = df_work.sort_values(cpu_fan)
        axes[0, 0].scatter(
            df_sorted[cpu_fan],
            df_sorted["T_cpu_norm"],
            c=df_sorted["P_cpu"],
            s=60,
            alpha=0.6,
            cmap="viridis",
        )
        axes[0, 0].set_xlabel(f"{cpu_fan_name} ({cpu_fan})", fontsize=12)
        axes[0, 0].set_ylabel("CPU ΔT above Ambient (°C)", fontsize=12)
        axes[0, 0].set_title(
            f"CPU Cooling Effectiveness: {cpu_fan_name}\n(Diminishing returns at high speeds)",
            fontsize=13,
            fontweight="bold",
        )
        axes[0, 0].grid(True, alpha=0.3)
        cbar = plt.colorbar(
            plt.cm.ScalarMappable(
                cmap="viridis",
                norm=plt.Normalize(
                    vmin=df_work["P_cpu"].min(), vmax=df_work["P_cpu"].max()
                ),
            ),
            ax=axes[0, 0],
        )
        cbar.set_label("CPU Power (W)", fontsize=10)
    else:
        axes[0, 0].text(
            0.5,
            0.5,
            "CPU fan data not available",
            ha="center",
            va="center",
            fontsize=12,
        )
        axes[0, 0].set_title(
            "CPU Cooling Effectiveness", fontsize=13, fontweight="bold"
        )

    # CPU cooling effectiveness - plot second fan if available
    if len(cpu_fans) > 1 and cpu_fans[1] in df_work.columns:
        cpu_fan2 = cpu_fans[1]
        cpu_fan2_name = CONFIG["devices"][cpu_fan2]["name"]

        df_sorted = df_work.sort_values(cpu_fan2)
        axes[0, 1].scatter(
            df_sorted[cpu_fan2],
            df_sorted["T_cpu_norm"],
            c=df_sorted[cpu_fans[0]],
            s=60,
            alpha=0.6,
            cmap="plasma",
        )
        axes[0, 1].set_xlabel(f"{cpu_fan2_name} ({cpu_fan2})", fontsize=12)
        cpu_fan1_name = CONFIG["devices"][cpu_fans[0]]["name"]

        axes[0, 1].set_ylabel("CPU ΔT above Ambient (°C)", fontsize=12)
        axes[0, 1].set_title(
            f"CPU Cooling Effectiveness: {cpu_fan2_name}\n(color = {cpu_fan1_name})",
            fontsize=13,
            fontweight="bold",
        )
        axes[0, 1].grid(True, alpha=0.3)
        cbar = plt.colorbar(
            plt.cm.ScalarMappable(
                cmap="plasma",
                norm=plt.Normalize(
                    vmin=df_work[cpu_fans[0]].min(), vmax=df_work[cpu_fans[0]].max()
                ),
            ),
            ax=axes[0, 1],
        )
        cbar.set_label(f"{cpu_fan1_name} ({cpu_fans[0]})", fontsize=10)
    else:
        axes[0, 1].text(
            0.5,
            0.5,
            "Second CPU fan not configured",
            ha="center",
            va="center",
            fontsize=12,
        )
        axes[0, 1].set_title(
            "CPU Cooling Effectiveness", fontsize=13, fontweight="bold"
        )

    # GPU cooling effectiveness - plot first fan
    if len(gpu_fans) > 0 and gpu_fans[0] in df_work.columns:
        gpu_fan = gpu_fans[0]
        gpu_fan_name = CONFIG["devices"][gpu_fan]["name"]

        df_sorted = df_work.sort_values(gpu_fan)
        axes[1, 0].scatter(
            df_sorted[gpu_fan],
            df_sorted["T_gpu_norm"],
            c=df_sorted["P_gpu"],
            s=60,
            alpha=0.6,
            cmap="viridis",
        )
        axes[1, 0].set_xlabel(f"{gpu_fan_name} ({gpu_fan})", fontsize=12)
        axes[1, 0].set_ylabel("GPU ΔT above Ambient (°C)", fontsize=12)
        axes[1, 0].set_title(
            f"GPU Cooling Effectiveness: {gpu_fan_name}\n(Direct airflow to GPU)",
            fontsize=13,
            fontweight="bold",
        )
        axes[1, 0].grid(True, alpha=0.3)
        cbar = plt.colorbar(
            plt.cm.ScalarMappable(
                cmap="viridis",
                norm=plt.Normalize(
                    vmin=df_work["P_gpu"].min(), vmax=df_work["P_gpu"].max()
                ),
            ),
            ax=axes[1, 0],
        )
        cbar.set_label("GPU Power (W)", fontsize=10)
    else:
        axes[1, 0].text(
            0.5,
            0.5,
            "GPU fan data not available",
            ha="center",
            va="center",
            fontsize=12,
        )
        axes[1, 0].set_title(
            "GPU Cooling Effectiveness", fontsize=13, fontweight="bold"
        )

    # GPU cooling effectiveness - plot second fan if available
    if len(gpu_fans) > 1 and gpu_fans[1] in df_work.columns:
        gpu_fan2 = gpu_fans[1]
        gpu_fan2_name = CONFIG["devices"][gpu_fan2]["name"]

        df_sorted = df_work.sort_values(gpu_fan2)
        axes[1, 1].scatter(
            df_sorted[gpu_fan2],
            df_sorted["T_gpu_norm"],
            c=df_sorted[gpu_fans[0]],
            s=60,
            alpha=0.6,
            cmap="plasma",
        )
        gpu_fan1_name = CONFIG["devices"][gpu_fans[0]]["name"]

        axes[1, 1].set_xlabel(f"{gpu_fan2_name} ({gpu_fan2})", fontsize=12)
        axes[1, 1].set_ylabel("GPU ΔT above Ambient (°C)", fontsize=12)
        axes[1, 1].set_title(
            f"GPU Cooling Effectiveness: {gpu_fan2_name}\n(color = {gpu_fan1_name})",
            fontsize=13,
            fontweight="bold",
        )
        axes[1, 1].grid(True, alpha=0.3)
        cbar = plt.colorbar(
            plt.cm.ScalarMappable(
                cmap="plasma",
                norm=plt.Normalize(
                    vmin=df_work[gpu_fans[0]].min(), vmax=df_work[gpu_fans[0]].max()
                ),
            ),
            ax=axes[1, 1],
        )
        cbar.set_label(f"{gpu_fan1_name} ({gpu_fans[0]})", fontsize=10)
    else:
        axes[1, 1].text(
            0.5,
            0.5,
            "Second GPU fan not configured",
            ha="center",
            va="center",
            fontsize=12,
        )
        axes[1, 1].set_title(
            "GPU Cooling Effectiveness", fontsize=13, fontweight="bold"
        )

    plt.tight_layout()
    return fig


def plot_ambient_normalized(df: pd.DataFrame) -> plt.Figure:
    """Plot ambient-normalized temperatures to remove ambient variation effects"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Calculate normalized temperatures
    df_work = df.copy()
    df_work["dT_cpu"] = df_work["T_cpu"] - df_work["T_amb"]
    df_work["dT_gpu"] = df_work["T_gpu"] - df_work["T_amb"]

    # CPU: dT vs Power (should be more linear than raw temp)
    axes[0, 0].scatter(
        df_work["P_cpu"],
        df_work["dT_cpu"],
        c=df_work["pwm2"],
        s=80,
        alpha=0.6,
        cmap="coolwarm",
    )
    axes[0, 0].set_xlabel("CPU Power (W)", fontsize=12)
    axes[0, 0].set_ylabel("CPU ΔT = T_cpu - T_ambient (°C)", fontsize=12)
    axes[0, 0].set_title(
        "CPU Temperature Rise vs Power\n(Ambient-normalized, color = radiator fan)",
        fontsize=13,
        fontweight="bold",
    )
    axes[0, 0].grid(True, alpha=0.3)
    cbar = plt.colorbar(
        plt.cm.ScalarMappable(
            cmap="coolwarm",
            norm=plt.Normalize(vmin=df_work["pwm2"].min(), vmax=df_work["pwm2"].max()),
        ),
        ax=axes[0, 0],
    )
    cbar.set_label("Radiator Fan PWM", fontsize=10)

    # GPU: dT vs Power
    axes[0, 1].scatter(
        df_work["P_gpu"],
        df_work["dT_gpu"],
        c=df_work["pwm4"],
        s=80,
        alpha=0.6,
        cmap="coolwarm",
    )
    axes[0, 1].set_xlabel("GPU Power (W)", fontsize=12)
    axes[0, 1].set_ylabel("GPU ΔT = T_gpu - T_ambient (°C)", fontsize=12)
    axes[0, 1].set_title(
        "GPU Temperature Rise vs Power\n(Ambient-normalized, color = bottom intake)",
        fontsize=13,
        fontweight="bold",
    )
    axes[0, 1].grid(True, alpha=0.3)
    cbar = plt.colorbar(
        plt.cm.ScalarMappable(
            cmap="coolwarm",
            norm=plt.Normalize(vmin=df_work["pwm4"].min(), vmax=df_work["pwm4"].max()),
        ),
        ax=axes[0, 1],
    )
    cbar.set_label("Bottom Intake PWM", fontsize=10)

    # CPU: Thermal resistance R = dT / P
    df_cpu = df_work[df_work["P_cpu"] > 0].copy()
    if len(df_cpu) > 0:
        df_cpu["R_cpu"] = df_cpu["dT_cpu"] / df_cpu["P_cpu"]

        if "pwm7" in df_cpu.columns:
            axes[1, 0].scatter(
                df_cpu["pwm2"],
                df_cpu["R_cpu"],
                c=df_cpu["pwm7"],
                s=80,
                alpha=0.6,
                cmap="viridis",
            )
            cbar = plt.colorbar(
                plt.cm.ScalarMappable(
                    cmap="viridis",
                    norm=plt.Normalize(
                        vmin=df_cpu["pwm7"].min(), vmax=df_cpu["pwm7"].max()
                    ),
                ),
                ax=axes[1, 0],
            )
            cbar.set_label("Pump Speed (pwm7)", fontsize=10)
            title_suffix = "(Lower is better, color = pump speed)"
        else:
            axes[1, 0].scatter(
                df_cpu["pwm2"],
                df_cpu["R_cpu"],
                c=df_cpu["P_cpu"],
                s=80,
                alpha=0.6,
                cmap="viridis",
            )
            cbar = plt.colorbar(
                plt.cm.ScalarMappable(
                    cmap="viridis",
                    norm=plt.Normalize(
                        vmin=df_cpu["P_cpu"].min(), vmax=df_cpu["P_cpu"].max()
                    ),
                ),
                ax=axes[1, 0],
            )
            cbar.set_label("CPU Power (W)", fontsize=10)
            title_suffix = "(Lower is better, color = CPU power)"

        axes[1, 0].set_xlabel("Radiator Fan Speed (pwm2)", fontsize=12)
        axes[1, 0].set_ylabel("CPU Thermal Resistance R = ΔT/P (°C/W)", fontsize=12)
        axes[1, 0].set_title(
            f"CPU Thermal Resistance vs Radiator Fan\n{title_suffix}",
            fontsize=13,
            fontweight="bold",
        )
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(
            0.5,
            0.5,
            "Insufficient CPU load data",
            ha="center",
            va="center",
            fontsize=12,
        )

    # GPU: Thermal resistance
    df_gpu = df_work[df_work["P_gpu"] > 0].copy()
    if len(df_gpu) > 0:
        df_gpu["R_gpu"] = df_gpu["dT_gpu"] / df_gpu["P_gpu"]

        axes[1, 1].scatter(
            df_gpu["pwm4"],
            df_gpu["R_gpu"],
            c=df_gpu["pwm5"],
            s=80,
            alpha=0.6,
            cmap="viridis",
        )
        axes[1, 1].set_xlabel("Bottom Intake Speed (pwm4)", fontsize=12)
        axes[1, 1].set_ylabel("GPU Thermal Resistance R = ΔT/P (°C/W)", fontsize=12)
        axes[1, 1].set_title(
            "GPU Thermal Resistance vs Bottom Intake\n(Lower is better, color = front/rear fans)",
            fontsize=13,
            fontweight="bold",
        )
        axes[1, 1].grid(True, alpha=0.3)
        cbar = plt.colorbar(
            plt.cm.ScalarMappable(
                cmap="viridis",
                norm=plt.Normalize(
                    vmin=df_gpu["pwm5"].min(), vmax=df_gpu["pwm5"].max()
                ),
            ),
            ax=axes[1, 1],
        )
        cbar.set_label("Front/Rear Fan PWM", fontsize=10)
    else:
        axes[1, 1].text(
            0.5,
            0.5,
            "No GPU load data available",
            ha="center",
            va="center",
            fontsize=14,
        )

    plt.tight_layout()
    return fig


def plot_3d_surfaces(df: pd.DataFrame) -> plt.Figure:
    """Create 3D surface plots showing temperature as function of two variables"""
    import mpl_toolkits.mplot3d  # noqa: F401

    fig = plt.figure(figsize=(18, 12))

    # Get surface plot config
    surface_cfg = CONFIG.get("plot_config", {}).get("surface_3d", {})
    cpu_cfg = surface_cfg.get("cpu", {"x_axis": "pwm2", "y_axis": "P_cpu"})

    # CPU surface plot
    ax1 = fig.add_subplot(2, 2, 1, projection="3d")

    x_var = cpu_cfg.get("x_axis", "pwm2")
    y_var = cpu_cfg.get("y_axis", "P_cpu")

    if x_var not in df.columns or y_var not in df.columns:
        ax1.text2D(
            0.5,
            0.5,
            f"Variables {x_var} or {y_var} not available",
            ha="center",
            va="center",
            fontsize=12,
        )
        ax1.set_title(
            f"CPU Temperature Surface\nf({x_var}, {y_var})",
            fontsize=12,
            fontweight="bold",
        )
    else:
        # Get variable labels
        x_label = (
            CONFIG["devices"][x_var]["name"] if x_var in CONFIG["devices"] else x_var
        )
        y_label = (
            CONFIG["devices"][y_var]["name"] if y_var in CONFIG["devices"] else y_var
        )

        # Create grid for interpolation
        x_grid = np.linspace(df[x_var].min(), df[x_var].max(), 30)
        y_grid = np.linspace(df[y_var].min(), df[y_var].max(), 30)
        x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)

        # Interpolate CPU temperature (use linear to avoid extrapolation artifacts)
        points = df[[x_var, y_var]].values
        values = df["T_cpu"].values
        T_cpu_interp = griddata(points, values, (x_mesh, y_mesh), method="linear")

        surf1 = ax1.plot_surface(
            x_mesh, y_mesh, T_cpu_interp, cmap="viridis", alpha=0.8, edgecolor="none"
        )
        ax1.scatter(df[x_var], df[y_var], df["T_cpu"], c="red", s=10, alpha=0.5)
        ax1.set_xlabel(f"{x_label} ({x_var})", fontsize=10)
        ax1.set_ylabel(f"{y_label} ({y_var})", fontsize=10)
        ax1.set_zlabel("CPU Temp (°C)", fontsize=10)
        ax1.set_title(
            f"CPU Temperature Surface\nf({x_var}, {y_var})",
            fontsize=12,
            fontweight="bold",
        )
        fig.colorbar(surf1, ax=ax1, shrink=0.5)

    # CPU: T_cpu as f(pwm, P_cpu) - Alternative view (power dependency)
    ax2 = fig.add_subplot(2, 2, 2, projection="3d")

    # Use first available fan for this plot
    fan_for_power_plot = next(
        (k for k in CONFIG["devices"].keys() if k in df.columns), "pwm2"
    )

    P_cpu_grid = np.linspace(df["P_cpu"].min(), df["P_cpu"].max(), 30)
    fan_grid = np.linspace(
        df[fan_for_power_plot].min(), df[fan_for_power_plot].max(), 30
    )
    P_cpu_mesh, fan_mesh = np.meshgrid(P_cpu_grid, fan_grid)

    fan_name = CONFIG["devices"][fan_for_power_plot]["name"]

    points2 = df[["P_cpu", fan_for_power_plot]].values
    values2 = df["T_cpu"].values
    T_cpu_interp2 = griddata(points2, values2, (P_cpu_mesh, fan_mesh), method="linear")

    surf2 = ax2.plot_surface(
        P_cpu_mesh,
        fan_mesh,
        T_cpu_interp2,
        cmap="plasma",
        alpha=0.8,
        edgecolor="none",
    )
    ax2.scatter(
        df["P_cpu"], df[fan_for_power_plot], df["T_cpu"], c="red", s=10, alpha=0.5
    )
    ax2.set_xlabel("CPU Power (W)", fontsize=10)
    ax2.set_ylabel(f"{fan_name} ({fan_for_power_plot})", fontsize=10)
    ax2.set_zlabel("CPU Temp (°C)", fontsize=10)
    ax2.set_title(
        f"CPU Temperature Surface\nf(P_cpu, {fan_for_power_plot})",
        fontsize=12,
        fontweight="bold",
    )
    fig.colorbar(surf2, ax=ax2, shrink=0.5)

    # GPU surface plot
    gpu_cfg = surface_cfg.get("gpu", {"x_axis": "pwm4", "y_axis": "P_gpu"})
    ax3 = fig.add_subplot(2, 2, 3, projection="3d")

    x_var_gpu = gpu_cfg.get("x_axis", "pwm4")
    y_var_gpu = gpu_cfg.get("y_axis", "P_gpu")

    if x_var_gpu not in df.columns or y_var_gpu not in df.columns:
        ax3.text2D(
            0.5,
            0.5,
            f"Variables {x_var_gpu} or {y_var_gpu} not available",
            ha="center",
            va="center",
            fontsize=12,
        )
        ax3.set_title(
            f"GPU Temperature Surface\nf({x_var_gpu}, {y_var_gpu})",
            fontsize=12,
            fontweight="bold",
        )
    else:
        # Get variable labels
        x_label_gpu = (
            CONFIG["devices"][x_var_gpu]["name"]
            if x_var_gpu in CONFIG["devices"]
            else x_var_gpu
        )
        y_label_gpu = (
            CONFIG["devices"][y_var_gpu]["name"]
            if y_var_gpu in CONFIG["devices"]
            else y_var_gpu
        )

        # Create grid for interpolation
        x_grid_gpu = np.linspace(df[x_var_gpu].min(), df[x_var_gpu].max(), 30)
        y_grid_gpu = np.linspace(df[y_var_gpu].min(), df[y_var_gpu].max(), 30)
        x_mesh_gpu, y_mesh_gpu = np.meshgrid(x_grid_gpu, y_grid_gpu)

        # Interpolate GPU temperature (use linear to avoid extrapolation artifacts)
        points3 = df[[x_var_gpu, y_var_gpu]].values
        values3 = df["T_gpu"].values
        T_gpu_interp = griddata(
            points3, values3, (x_mesh_gpu, y_mesh_gpu), method="linear"
        )

        surf3 = ax3.plot_surface(
            x_mesh_gpu,
            y_mesh_gpu,
            T_gpu_interp,
            cmap="viridis",
            alpha=0.8,
            edgecolor="none",
        )
        ax3.scatter(df[x_var_gpu], df[y_var_gpu], df["T_gpu"], c="red", s=10, alpha=0.5)
        ax3.set_xlabel(f"{x_label_gpu} ({x_var_gpu})", fontsize=10)
        ax3.set_ylabel(f"{y_label_gpu} ({y_var_gpu})", fontsize=10)
        ax3.set_zlabel("GPU Temp (°C)", fontsize=10)
        ax3.set_title(
            f"GPU Temperature Surface\nf({x_var_gpu}, {y_var_gpu})",
            fontsize=12,
            fontweight="bold",
        )
        fig.colorbar(surf3, ax=ax3, shrink=0.5)

    # GPU: T_gpu as f(pwm, P_gpu) - Alternative view (power dependency)
    ax4 = fig.add_subplot(2, 2, 4, projection="3d")

    # Use first available fan for GPU power plot
    fan_for_gpu_power_plot = next(
        (k for k in CONFIG["devices"].keys() if k in df.columns), "pwm4"
    )

    P_gpu_grid = np.linspace(df["P_gpu"].min(), df["P_gpu"].max(), 30)
    fan_gpu_grid = np.linspace(
        df[fan_for_gpu_power_plot].min(), df[fan_for_gpu_power_plot].max(), 30
    )
    P_gpu_mesh, fan_gpu_mesh = np.meshgrid(P_gpu_grid, fan_gpu_grid)

    points4 = df[["P_gpu", fan_for_gpu_power_plot]].values
    T_gpu_interp2 = griddata(
        points4, values3, (P_gpu_mesh, fan_gpu_mesh), method="linear"
    )

    fan_gpu_name = CONFIG["devices"][fan_for_gpu_power_plot]["name"]

    surf4 = ax4.plot_surface(
        P_gpu_mesh,
        fan_gpu_mesh,
        T_gpu_interp2,
        cmap="plasma",
        alpha=0.8,
        edgecolor="none",
    )
    ax4.scatter(
        df["P_gpu"], df[fan_for_gpu_power_plot], df["T_gpu"], c="red", s=10, alpha=0.5
    )
    ax4.set_xlabel("GPU Power (W)", fontsize=10)
    ax4.set_ylabel(f"{fan_gpu_name} ({fan_for_gpu_power_plot})", fontsize=10)
    ax4.set_zlabel("GPU Temp (°C)", fontsize=10)
    ax4.set_title(
        f"GPU Temperature Surface\nf(P_gpu, {fan_for_gpu_power_plot})",
        fontsize=12,
        fontweight="bold",
    )
    fig.colorbar(surf4, ax=ax4, shrink=0.5)

    plt.tight_layout()
    return fig


def generate_all_plots(
    data: pd.DataFrame | Path, output_dir: Path, quiet: bool = False
) -> None:
    """
    Generate all thermal data visualization plots.

    Args:
        data: Path to the CSV data file OR a pandas DataFrame
        output_dir: Directory to save plots
        quiet: If True, suppress console output
    """
    if not quiet:
        source = f"from {data}" if isinstance(data, Path) else "from DataFrame"
        print(f"\nGenerating plots {source}...")

    # Load data if path provided
    if isinstance(data, Path):
        try:
            df = pd.read_csv(data)
        except Exception as e:
            if not quiet:
                print(f"  ✗ Failed to load data: {e}")
            return
    else:
        df = data

    if len(df) < 2:
        if not quiet:
            print(f"  ⚠ Not enough data points ({len(df)}) to generate plots")
        return

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots
    plot_functions = [
        ("correlation_matrix", plot_correlation_matrix),
        ("temp_vs_fans", plot_temp_vs_fans),
        ("temp_vs_power", plot_temp_vs_power),
        ("pump_radiator_interaction", plot_pump_radiator_interaction),
        ("thermal_resistance", plot_thermal_resistance),
        ("pairwise_interactions", plot_pairwise_interactions),
        ("cooling_effectiveness", plot_cooling_effectiveness),
        ("ambient_normalized", plot_ambient_normalized),
        ("3d_surfaces", plot_3d_surfaces),
    ]

    for name, func in (
        pbar := tqdm(
            plot_functions,
            desc="Generating plots",
            leave=False,
            disable=quiet,
            unit="plot",
            bar_format="{desc}: {percentage:3.0f}% |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]",
        )
    ):
        try:
            pbar.set_postfix_str(f"{name}.png")
            fig = func(df)
            output_path = output_dir / f"{name}.png"
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close(fig)  # Close figure to free memory
        except Exception as e:
            if not quiet:
                tqdm.write(f"  ✗ {name}: {e}")

    if not quiet:
        print(f"  Plots saved to {output_dir}")
