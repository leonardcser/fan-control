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
from scipy.interpolate import griddata
from tqdm import tqdm

# Suppress warnings for cleaner output during data collection
warnings.filterwarnings("ignore")

# Use non-interactive backend for batch plotting
matplotlib.use("Agg")

# Set style
sns.set_theme(style="darkgrid")
plt.rcParams["figure.figsize"] = (14, 10)


def plot_correlation_matrix(df: pd.DataFrame) -> plt.Figure:
    """Plot correlation matrix for all key variables"""
    fig, ax = plt.subplots(figsize=(12, 10))

    # Key variables for physics model
    key_vars = [
        "pwm2",
        "pwm4",
        "pwm5",
        "pwm7",
        "P_cpu",
        "P_gpu",
        "T_amb",
        "T_cpu",
        "T_gpu",
    ]

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
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    fans = ["pwm2", "pwm4", "pwm5", "pwm7"]
    fan_names = [
        "Radiator Fan\n(pwm2)",
        "Bottom Intake\n(pwm4)",
        "Front/Rear Fans\n(pwm5)",
        "Pump\n(pwm7)",
    ]

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
    """Plot the pump-radiator coupling effect on CPU temperature"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # CPU temp vs radiator fan, colored by pump speed
    scatter1 = axes[0].scatter(
        df["pwm2"], df["T_cpu"], c=df["pwm7"], s=100, alpha=0.6, cmap="viridis"
    )
    axes[0].set_xlabel("Radiator Fan Speed (pwm2)", fontsize=12)
    axes[0].set_ylabel("CPU Temperature (°C)", fontsize=12)
    axes[0].set_title(
        "Pump-Radiator Coupling Effect\nCPU Temp vs Radiator Fan (color = pump speed)",
        fontsize=13,
        fontweight="bold",
    )
    axes[0].grid(True, alpha=0.3)
    cbar1 = plt.colorbar(scatter1, ax=axes[0])
    cbar1.set_label("Pump Speed (pwm7)", fontsize=10)

    # CPU temp vs pump speed, colored by radiator fan
    scatter2 = axes[1].scatter(
        df["pwm7"], df["T_cpu"], c=df["pwm2"], s=100, alpha=0.6, cmap="plasma"
    )
    axes[1].set_xlabel("Pump Speed (pwm7)", fontsize=12)
    axes[1].set_ylabel("CPU Temperature (°C)", fontsize=12)
    axes[1].set_title(
        "Pump-Radiator Coupling Effect\nCPU Temp vs Pump Speed (color = radiator fan)",
        fontsize=13,
        fontweight="bold",
    )
    axes[1].grid(True, alpha=0.3)
    cbar2 = plt.colorbar(scatter2, ax=axes[1])
    cbar2.set_label("Radiator Fan Speed (pwm2)", fontsize=10)

    plt.tight_layout()
    return fig


def plot_thermal_resistance(df: pd.DataFrame) -> plt.Figure:
    """Plot effective thermal resistance: (T - T_amb) / Power"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # CPU thermal resistance
    df_cpu = df[df["P_cpu"] > 0].copy()
    df_cpu["R_cpu"] = (df_cpu["T_cpu"] - df_cpu["T_amb"]) / df_cpu["P_cpu"]

    scatter1 = axes[0].scatter(
        df_cpu["pwm2"],
        df_cpu["R_cpu"],
        c=df_cpu["pwm7"],
        s=100,
        alpha=0.6,
        cmap="coolwarm",
    )
    axes[0].set_xlabel("Radiator Fan Speed (pwm2)", fontsize=12)
    axes[0].set_ylabel("CPU Thermal Resistance (°C/W)", fontsize=12)
    axes[0].set_title(
        "CPU Thermal Resistance vs Radiator Fan\n(color = pump speed)",
        fontsize=13,
        fontweight="bold",
    )
    axes[0].grid(True, alpha=0.3)
    cbar1 = plt.colorbar(scatter1, ax=axes[0])
    cbar1.set_label("Pump Speed (pwm7)", fontsize=10)

    # GPU thermal resistance
    df_gpu = df[df["P_gpu"] > 0].copy()
    if len(df_gpu) > 0:
        df_gpu["R_gpu"] = (df_gpu["T_gpu"] - df_gpu["T_amb"]) / df_gpu["P_gpu"]

        scatter2 = axes[1].scatter(
            df_gpu["pwm4"],
            df_gpu["R_gpu"],
            c=df_gpu["pwm5"],
            s=100,
            alpha=0.6,
            cmap="coolwarm",
        )
        axes[1].set_xlabel("Bottom Intake Speed (pwm4)", fontsize=12)
        axes[1].set_ylabel("GPU Thermal Resistance (°C/W)", fontsize=12)
        axes[1].set_title(
            "GPU Thermal Resistance vs Bottom Intake\n(color = front/rear fan speed)",
            fontsize=13,
            fontweight="bold",
        )
        axes[1].grid(True, alpha=0.3)
        cbar2 = plt.colorbar(scatter2, ax=axes[1])
        cbar2.set_label("Front/Rear Fan Speed (pwm5)", fontsize=10)
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
    key_vars = ["pwm2", "pwm4", "pwm5", "pwm7", "P_cpu", "T_cpu"]

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

    # CPU cooling effectiveness vs radiator fan (pwm2)
    df_sorted = df_work.sort_values("pwm2")
    axes[0, 0].scatter(
        df_sorted["pwm2"],
        df_sorted["T_cpu_norm"],
        c=df_sorted["P_cpu"],
        s=60,
        alpha=0.6,
        cmap="viridis",
    )
    axes[0, 0].set_xlabel("Radiator Fan Speed (pwm2)", fontsize=12)
    axes[0, 0].set_ylabel("CPU ΔT above Ambient (°C)", fontsize=12)
    axes[0, 0].set_title(
        "CPU Cooling Effectiveness: Radiator Fan\n(Diminishing returns at high speeds)",
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

    # CPU cooling effectiveness vs pump (pwm7)
    df_sorted = df_work.sort_values("pwm7")
    axes[0, 1].scatter(
        df_sorted["pwm7"],
        df_sorted["T_cpu_norm"],
        c=df_sorted["pwm2"],
        s=60,
        alpha=0.6,
        cmap="plasma",
    )
    axes[0, 1].set_xlabel("Pump Speed (pwm7)", fontsize=12)
    axes[0, 1].set_ylabel("CPU ΔT above Ambient (°C)", fontsize=12)
    axes[0, 1].set_title(
        "CPU Cooling Effectiveness: Pump Speed\n(color = radiator fan speed)",
        fontsize=13,
        fontweight="bold",
    )
    axes[0, 1].grid(True, alpha=0.3)
    cbar = plt.colorbar(
        plt.cm.ScalarMappable(
            cmap="plasma",
            norm=plt.Normalize(vmin=df_work["pwm2"].min(), vmax=df_work["pwm2"].max()),
        ),
        ax=axes[0, 1],
    )
    cbar.set_label("Radiator Fan PWM", fontsize=10)

    # GPU cooling effectiveness vs bottom intake (pwm4)
    df_sorted = df_work.sort_values("pwm4")
    axes[1, 0].scatter(
        df_sorted["pwm4"],
        df_sorted["T_gpu_norm"],
        c=df_sorted["P_gpu"],
        s=60,
        alpha=0.6,
        cmap="viridis",
    )
    axes[1, 0].set_xlabel("Bottom Intake Speed (pwm4)", fontsize=12)
    axes[1, 0].set_ylabel("GPU ΔT above Ambient (°C)", fontsize=12)
    axes[1, 0].set_title(
        "GPU Cooling Effectiveness: Bottom Intake\n(Direct airflow to GPU)",
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

    # GPU cooling effectiveness vs front/rear fans (pwm5)
    df_sorted = df_work.sort_values("pwm5")
    axes[1, 1].scatter(
        df_sorted["pwm5"],
        df_sorted["T_gpu_norm"],
        c=df_sorted["pwm4"],
        s=60,
        alpha=0.6,
        cmap="plasma",
    )
    axes[1, 1].set_xlabel("Front/Rear Fan Speed (pwm5)", fontsize=12)
    axes[1, 1].set_ylabel("GPU ΔT above Ambient (°C)", fontsize=12)
    axes[1, 1].set_title(
        "GPU Cooling Effectiveness: Front/Rear Fans\n(color = bottom intake speed)",
        fontsize=13,
        fontweight="bold",
    )
    axes[1, 1].grid(True, alpha=0.3)
    cbar = plt.colorbar(
        plt.cm.ScalarMappable(
            cmap="plasma",
            norm=plt.Normalize(vmin=df_work["pwm4"].min(), vmax=df_work["pwm4"].max()),
        ),
        ax=axes[1, 1],
    )
    cbar.set_label("Bottom Intake PWM", fontsize=10)

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
    df_cpu["R_cpu"] = df_cpu["dT_cpu"] / df_cpu["P_cpu"]

    axes[1, 0].scatter(
        df_cpu["pwm2"],
        df_cpu["R_cpu"],
        c=df_cpu["pwm7"],
        s=80,
        alpha=0.6,
        cmap="viridis",
    )
    axes[1, 0].set_xlabel("Radiator Fan Speed (pwm2)", fontsize=12)
    axes[1, 0].set_ylabel("CPU Thermal Resistance R = ΔT/P (°C/W)", fontsize=12)
    axes[1, 0].set_title(
        "CPU Thermal Resistance vs Radiator Fan\n(Lower is better, color = pump speed)",
        fontsize=13,
        fontweight="bold",
    )
    axes[1, 0].grid(True, alpha=0.3)
    cbar = plt.colorbar(
        plt.cm.ScalarMappable(
            cmap="viridis",
            norm=plt.Normalize(vmin=df_cpu["pwm7"].min(), vmax=df_cpu["pwm7"].max()),
        ),
        ax=axes[1, 0],
    )
    cbar.set_label("Pump Speed (pwm7)", fontsize=10)

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

    # CPU: T_cpu as f(pwm2, pwm7) - Pump-Radiator interaction
    ax1 = fig.add_subplot(2, 2, 1, projection="3d")

    # Create grid for interpolation
    pwm2_grid = np.linspace(df["pwm2"].min(), df["pwm2"].max(), 30)
    pwm7_grid = np.linspace(df["pwm7"].min(), df["pwm7"].max(), 30)
    pwm2_mesh, pwm7_mesh = np.meshgrid(pwm2_grid, pwm7_grid)

    # Interpolate CPU temperature
    points = df[["pwm2", "pwm7"]].values
    values = df["T_cpu"].values
    T_cpu_interp = griddata(points, values, (pwm2_mesh, pwm7_mesh), method="cubic")

    surf1 = ax1.plot_surface(
        pwm2_mesh, pwm7_mesh, T_cpu_interp, cmap="viridis", alpha=0.8, edgecolor="none"
    )
    ax1.scatter(df["pwm2"], df["pwm7"], df["T_cpu"], c="red", s=10, alpha=0.5)
    ax1.set_xlabel("Radiator Fan (pwm2)", fontsize=10)
    ax1.set_ylabel("Pump Speed (pwm7)", fontsize=10)
    ax1.set_zlabel("CPU Temp (°C)", fontsize=10)
    ax1.set_title(
        "CPU Temperature Surface\nf(pwm2, pwm7)", fontsize=12, fontweight="bold"
    )
    fig.colorbar(surf1, ax=ax1, shrink=0.5)

    # CPU: T_cpu as f(pwm2, P_cpu) - Power dependency
    ax2 = fig.add_subplot(2, 2, 2, projection="3d")

    P_cpu_grid = np.linspace(df["P_cpu"].min(), df["P_cpu"].max(), 30)
    pwm2_grid2 = np.linspace(df["pwm2"].min(), df["pwm2"].max(), 30)
    P_cpu_mesh, pwm2_mesh2 = np.meshgrid(P_cpu_grid, pwm2_grid2)

    points2 = df[["P_cpu", "pwm2"]].values
    T_cpu_interp2 = griddata(points2, values, (P_cpu_mesh, pwm2_mesh2), method="cubic")

    surf2 = ax2.plot_surface(
        P_cpu_mesh,
        pwm2_mesh2,
        T_cpu_interp2,
        cmap="plasma",
        alpha=0.8,
        edgecolor="none",
    )
    ax2.scatter(df["P_cpu"], df["pwm2"], df["T_cpu"], c="red", s=10, alpha=0.5)
    ax2.set_xlabel("CPU Power (W)", fontsize=10)
    ax2.set_ylabel("Radiator Fan (pwm2)", fontsize=10)
    ax2.set_zlabel("CPU Temp (°C)", fontsize=10)
    ax2.set_title(
        "CPU Temperature Surface\nf(P_cpu, pwm2)", fontsize=12, fontweight="bold"
    )
    fig.colorbar(surf2, ax=ax2, shrink=0.5)

    # GPU: T_gpu as f(pwm4, pwm5) - Fan interaction
    ax3 = fig.add_subplot(2, 2, 3, projection="3d")

    pwm4_grid = np.linspace(df["pwm4"].min(), df["pwm4"].max(), 30)
    pwm5_grid = np.linspace(df["pwm5"].min(), df["pwm5"].max(), 30)
    pwm4_mesh, pwm5_mesh = np.meshgrid(pwm4_grid, pwm5_grid)

    points3 = df[["pwm4", "pwm5"]].values
    values3 = df["T_gpu"].values
    T_gpu_interp = griddata(points3, values3, (pwm4_mesh, pwm5_mesh), method="cubic")

    surf3 = ax3.plot_surface(
        pwm4_mesh, pwm5_mesh, T_gpu_interp, cmap="viridis", alpha=0.8, edgecolor="none"
    )
    ax3.scatter(df["pwm4"], df["pwm5"], df["T_gpu"], c="red", s=10, alpha=0.5)
    ax3.set_xlabel("Bottom Intake (pwm4)", fontsize=10)
    ax3.set_ylabel("Front/Rear Fans (pwm5)", fontsize=10)
    ax3.set_zlabel("GPU Temp (°C)", fontsize=10)
    ax3.set_title(
        "GPU Temperature Surface\nf(pwm4, pwm5)", fontsize=12, fontweight="bold"
    )
    fig.colorbar(surf3, ax=ax3, shrink=0.5)

    # GPU: T_gpu as f(pwm4, P_gpu) - Power dependency
    ax4 = fig.add_subplot(2, 2, 4, projection="3d")

    P_gpu_grid = np.linspace(df["P_gpu"].min(), df["P_gpu"].max(), 30)
    pwm4_grid2 = np.linspace(df["pwm4"].min(), df["pwm4"].max(), 30)
    P_gpu_mesh, pwm4_mesh2 = np.meshgrid(P_gpu_grid, pwm4_grid2)

    points4 = df[["P_gpu", "pwm4"]].values
    T_gpu_interp2 = griddata(points4, values3, (P_gpu_mesh, pwm4_mesh2), method="cubic")

    surf4 = ax4.plot_surface(
        P_gpu_mesh,
        pwm4_mesh2,
        T_gpu_interp2,
        cmap="plasma",
        alpha=0.8,
        edgecolor="none",
    )
    ax4.scatter(df["P_gpu"], df["pwm4"], df["T_gpu"], c="red", s=10, alpha=0.5)
    ax4.set_xlabel("GPU Power (W)", fontsize=10)
    ax4.set_ylabel("Bottom Intake (pwm4)", fontsize=10)
    ax4.set_zlabel("GPU Temp (°C)", fontsize=10)
    ax4.set_title(
        "GPU Temperature Surface\nf(P_gpu, pwm4)", fontsize=12, fontweight="bold"
    )
    fig.colorbar(surf4, ax=ax4, shrink=0.5)

    plt.tight_layout()
    return fig


def generate_all_plots(data_path: Path, output_dir: Path, quiet: bool = False) -> None:
    """
    Generate all thermal data visualization plots.

    Args:
        data_path: Path to the CSV data file
        output_dir: Directory to save plots
        quiet: If True, suppress console output
    """
    if not quiet:
        print(f"\nGenerating plots from {data_path}...")

    # Load data
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        if not quiet:
            print(f"  ✗ Failed to load data: {e}")
        return

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
            bar_format="{desc}: {percentage:3.0f}% |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {postfix}]",
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
