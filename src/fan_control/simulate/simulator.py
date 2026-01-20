import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from pathlib import Path
from typing import Dict, List
import logging

from ..control.optimizer import Optimizer
from ..fit.train import ThermalModel

logger = logging.getLogger(__name__)

# Color palette for consistent fan visualization
FAN_COLORS = ["blue", "green", "purple", "cyan", "orange", "red", "brown", "pink"]


def run_simulation(model_path: Path, config: Dict, output_dir: Path):
    """
    Run a simulation of the optimizer against increasing loads.
    """
    # Create optimizer
    optimizer = Optimizer(model_path, config)

    # Load targets from controller config
    targets = config["controller"]["targets"]

    # Generate Synthetic Scenario (0 to 100 seconds/steps)
    # Scenario:
    # 0-30s: CPU ramps 20W -> 180W (GPU Idle)
    # 30-60s: GPU ramps 20W -> 300W (CPU Idle)
    # 60-90s: Both ramp Max (Stress test)
    # 90-100s: Cooldown

    steps = 100

    p_cpu = np.zeros(steps)
    p_gpu = np.zeros(steps)

    # Phase 1: CPU Ramp
    p_cpu[0:30] = np.linspace(20, 180, 30)
    p_gpu[0:30] = 20

    # Phase 2: GPU Ramp
    p_cpu[30:60] = 30
    p_gpu[30:60] = np.linspace(20, 300, 30)

    # Phase 3: Combined Stress
    p_cpu[60:90] = np.linspace(30, 150, 30)
    p_gpu[60:90] = np.linspace(30, 250, 30)

    # Phase 4: Cooldown
    p_cpu[90:] = 20
    p_gpu[90:] = 20

    t_amb = 25.0

    results = []

    print(f"Running simulation ({steps} steps)...")

    # Initialize previous state for continuity (Start at IDLE/MINIMUMS)
    # This is more realistic than starting at 50%
    last_pwms = {}
    for name in optimizer.pwm_names:
        dev_cfg = config["devices"][name]
        last_pwms[name] = float(dev_cfg["min_pwm"])

    import time as time_module

    for t in range(steps):
        state = {"P_cpu": p_cpu[t], "P_gpu": p_gpu[t], "T_amb": t_amb}

        # Optimize
        t_start = time_module.perf_counter()
        pwms = optimizer.optimize(state, targets, initial_guess=last_pwms)
        t_end = time_module.perf_counter()
        opt_time = (t_end - t_start) * 1000.0

        # Update state for next iteration
        last_pwms = {k: float(v) for k, v in pwms.items()}

        # Predict result temperature at this optimal point
        full_input = state.copy()
        full_input.update(pwms)
        input_df = pd.DataFrame([full_input])
        t_cpu_pred, t_gpu_pred = optimizer.model.predict(input_df)

        res = {
            "time": t,
            "P_cpu": state["P_cpu"],
            "P_gpu": state["P_gpu"],
            "T_pred_cpu": t_cpu_pred[0],
            "T_pred_gpu": t_gpu_pred[0],
            "opt_time_ms": opt_time,
            **pwms,
        }
        results.append(res)

    df = pd.DataFrame(results)

    # Save CSV
    df.to_csv(output_dir / "simulation_results.csv", index=False)
    print(f"Simulation data saved to {output_dir}/simulation_results.csv")

    # Generate Plots
    generate_simulation_plots(df, targets, output_dir)

    # Generate Model Analysis Plots
    generate_model_analysis_plots(optimizer.model, config, output_dir)


def generate_simulation_plots(df: pd.DataFrame, targets: Dict, output_dir: Path):
    """Plot simulation results."""
    sns.set_theme(style="whitegrid")

    # Generate fan colors and labels dynamically from available PWM columns
    pwm_cols = sorted([col for col in df.columns if col.startswith("pwm")])
    fan_colors = {
        pwm: FAN_COLORS[i % len(FAN_COLORS)] for i, pwm in enumerate(pwm_cols)
    }
    fan_labels_short = {pwm: pwm.upper() for pwm in pwm_cols}
    fan_labels_long = {pwm: f"{pwm.upper()}" for pwm in pwm_cols}

    # 1. CPU Focus Plot
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Left Axis: Power & Temp
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Power (W) / Temp (°C)")

    l1 = ax1.plot(
        df["time"], df["P_cpu"], label="CPU Power (W)", color="orange", linestyle="--"
    )
    l2 = ax1.plot(
        df["time"],
        df["T_pred_cpu"],
        label="CPU Temp",
        color="red",
        linewidth=2,
    )
    l3 = ax1.axhline(
        targets["T_cpu"], label="Target CPU Temp", color="red", linestyle=":", alpha=0.5
    )

    # Right Axis: PWM
    ax2 = ax1.twinx()
    ax2.set_ylabel("Fan PWM (0-100)")
    ax2.set_ylim(0, 105)

    # Plot relevant fans for CPU
    fan_lines = []
    cpu_fan_priority = [col for col in pwm_cols if col in df.columns]

    for pwm_name in cpu_fan_priority[:4]:  # Limit to first 4 fans to avoid clutter
        l = ax2.plot(
            df["time"],
            df[pwm_name],
            label=f"{fan_labels_long[pwm_name]}",
            color=fan_colors[pwm_name],
            alpha=0.7,
            linewidth=1.5,
        )
        fan_lines.extend(l)

    # Legend
    lines = l1 + l2 + [l3] + fan_lines
    labs = [l.get_label() for l in lines]
    ax1.legend(lines, labs, loc="upper left", fontsize=8)

    plt.title("Simulation: CPU Thermal Response")
    plt.tight_layout()
    plt.savefig(output_dir / "sim_cpu_response.png")
    plt.close()

    # 2. GPU Focus Plot
    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Power (W) / Temp (°C)")

    l1 = ax1.plot(
        df["time"], df["P_gpu"], label="GPU Power (W)", color="green", linestyle="--"
    )
    l2 = ax1.plot(
        df["time"],
        df["T_pred_gpu"],
        label="GPU Temp",
        color="darkgreen",
        linewidth=2,
    )
    l3 = ax1.axhline(
        targets["T_gpu"],
        label="Target GPU Temp",
        color="darkgreen",
        linestyle=":",
        alpha=0.5,
    )

    ax2 = ax1.twinx()
    ax2.set_ylabel("Fan PWM (0-100)")
    ax2.set_ylim(0, 105)

    fan_lines = []
    gpu_fan_priority = [col for col in pwm_cols if col in df.columns]

    for pwm_name in gpu_fan_priority[:4]:  # Limit to first 4 fans to avoid clutter
        l = ax2.plot(
            df["time"],
            df[pwm_name],
            label=f"{fan_labels_long[pwm_name]}",
            color=fan_colors[pwm_name],
            alpha=0.7,
            linewidth=1.5,
        )
        fan_lines.extend(l)

    lines = l1 + l2 + [l3] + fan_lines
    labs = [l.get_label() for l in lines]
    ax1.legend(lines, labs, loc="upper left", fontsize=8)

    plt.title("Simulation: GPU Thermal Response")
    plt.tight_layout()
    plt.savefig(output_dir / "sim_gpu_response.png")
    plt.close()

    # 3. Combined / All Fans
    plt.figure(figsize=(12, 6))

    for pwm_name in pwm_cols:
        if pwm_name in df.columns:
            plt.plot(
                df["time"],
                df[pwm_name],
                label=f"{fan_labels_short[pwm_name]}",
                color=fan_colors[pwm_name],
                linewidth=1.5,
            )

    plt.ylabel("PWM")
    plt.xlabel("Time")
    plt.title("Simulation: All Fan Curves")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "sim_all_fans.png")
    plt.close()

    # 4. Timing Plot
    plt.figure(figsize=(12, 4))
    plt.plot(
        df["time"],
        df["opt_time_ms"],
        color="purple",
        label="Optimization Time",
        linewidth=1.5,
    )
    plt.ylabel("Time (ms)")
    plt.xlabel("Time Step")
    plt.title("Controller Latency Analysis")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "sim_timing.png")
    plt.close()

    avg_time = df["opt_time_ms"].mean()
    print(f"Average Optimization Time: {avg_time:.2f} ms")
    print(f"Plots saved to {output_dir}/")


def generate_model_analysis_plots(model: ThermalModel, config: Dict, output_dir: Path):
    """
    Generate analysis plots to understand what the model has learned.
    Includes PWM impact, partial dependence, and fan efficiency plots.
    """
    print(f"\nGenerating model analysis plots...")

    # Plot 1: PWM Impact
    plot_pwm_impact(model, config, output_dir)

    # Plot 2: Partial Dependence
    plot_partial_dependence(model, config, output_dir)

    # Plot 3: Fan Efficiency Comparison
    plot_fan_efficiency(model, config, output_dir)

    print(f"Model analysis plots saved to {output_dir}/")


def plot_pwm_impact(model: ThermalModel, config: Dict, output_dir: Path):
    """
    Plot the impact of each PWM channel on CPU and GPU temperatures.
    Varies each fan individually while holding others constant at their median.
    """
    devices = config["devices"]
    pwm_names = list(devices.keys())

    # Define representative operating conditions
    # Medium load scenario
    base_state = {
        "P_cpu": 100.0,  # Medium CPU load
        "P_gpu": 150.0,  # Medium GPU load
        "T_amb": 25.0,  # Room temperature
    }

    # Set baseline fan speeds (median of their range)
    baseline_pwms = {}
    for pwm_name in pwm_names:
        min_pwm = devices[pwm_name]["min_pwm"]
        baseline_pwms[pwm_name] = (min_pwm + 100.0) / 2.0

    # Create figure with subplots for CPU and GPU
    fig, (ax_cpu, ax_gpu) = plt.subplots(1, 2, figsize=(16, 6))

    # Generate colors for each fan
    fan_colors = {
        pwm: FAN_COLORS[i % len(FAN_COLORS)] for i, pwm in enumerate(pwm_names)
    }

    # For each fan, vary it and plot temperature response
    for pwm_name in pwm_names:
        min_pwm = devices[pwm_name]["min_pwm"]
        max_pwm = 100.0

        # Create sweep of PWM values
        pwm_values = np.linspace(min_pwm, max_pwm, 50)

        temps_cpu = []
        temps_gpu = []

        for pwm_val in pwm_values:
            # Create input with this fan at pwm_val, others at baseline
            input_state = base_state.copy()
            input_state.update(baseline_pwms)
            input_state[pwm_name] = pwm_val

            # Predict temperatures
            input_df = pd.DataFrame([input_state])
            t_cpu_pred, t_gpu_pred = model.predict(input_df)

            temps_cpu.append(t_cpu_pred[0])
            temps_gpu.append(t_gpu_pred[0])

        # Plot CPU impact
        ax_cpu.plot(
            pwm_values,
            temps_cpu,
            label=pwm_name.upper(),
            color=fan_colors[pwm_name],
            linewidth=2,
            marker="o",
            markersize=2,
        )

        # Plot GPU impact
        ax_gpu.plot(
            pwm_values,
            temps_gpu,
            label=pwm_name.upper(),
            color=fan_colors[pwm_name],
            linewidth=2,
            marker="o",
            markersize=2,
        )

    # Format CPU plot
    ax_cpu.set_xlabel("PWM (%)", fontsize=12)
    ax_cpu.set_ylabel("CPU Temperature (°C)", fontsize=12)
    ax_cpu.set_title(
        f"PWM Impact on CPU Temperature\n(P_cpu={base_state['P_cpu']}W, P_gpu={base_state['P_gpu']}W)",
        fontsize=13,
    )
    ax_cpu.legend(loc="best")
    ax_cpu.grid(True, alpha=0.3)

    # Format GPU plot
    ax_gpu.set_xlabel("PWM (%)", fontsize=12)
    ax_gpu.set_ylabel("GPU Temperature (°C)", fontsize=12)
    ax_gpu.set_title(
        f"PWM Impact on GPU Temperature\n(P_cpu={base_state['P_cpu']}W, P_gpu={base_state['P_gpu']}W)",
        fontsize=13,
    )
    ax_gpu.legend(loc="best")
    ax_gpu.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "model_pwm_impact.png", dpi=150)
    plt.close()

    print(f"✓ PWM Impact plot generated")


def plot_partial_dependence(model: ThermalModel, config: Dict, output_dir: Path):
    """
    Plot partial dependence for each feature to show how predicted temperature
    changes as each feature varies independently.
    """
    devices = config["devices"]
    pwm_names = list(devices.keys())

    # Define baseline state
    base_state = {
        "P_cpu": 100.0,
        "P_gpu": 150.0,
        "T_amb": 25.0,
    }

    # Set baseline fan speeds
    baseline_pwms = {}
    for pwm_name in pwm_names:
        min_pwm = devices[pwm_name]["min_pwm"]
        baseline_pwms[pwm_name] = (min_pwm + 100.0) / 2.0

    # Define features to analyze with their ranges
    features_to_plot = [
        ("P_cpu", np.linspace(10, 200, 50), "CPU Power (W)"),
        ("P_gpu", np.linspace(10, 350, 50), "GPU Power (W)"),
        ("T_amb", np.linspace(15, 35, 50), "Ambient Temp (°C)"),
    ]

    # Add PWM features
    for pwm_name in pwm_names:
        min_pwm = devices[pwm_name]["min_pwm"]
        features_to_plot.append(
            (pwm_name, np.linspace(min_pwm, 100, 50), f"{pwm_name.upper()} (%)")
        )

    # Calculate grid dimensions
    n_features = len(features_to_plot)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

    for idx, (feature_name, feature_range, feature_label) in enumerate(
        features_to_plot
    ):
        ax = axes[idx]

        temps_cpu = []
        temps_gpu = []

        for feature_val in feature_range:
            # Create input with this feature varied
            input_state = base_state.copy()
            input_state.update(baseline_pwms)
            input_state[feature_name] = feature_val

            # Predict temperatures
            input_df = pd.DataFrame([input_state])
            t_cpu_pred, t_gpu_pred = model.predict(input_df)

            temps_cpu.append(t_cpu_pred[0])
            temps_gpu.append(t_gpu_pred[0])

        # Plot both CPU and GPU on same axis
        ax.plot(feature_range, temps_cpu, label="CPU", color="red", linewidth=2)
        ax.plot(feature_range, temps_gpu, label="GPU", color="green", linewidth=2)

        ax.set_xlabel(feature_label, fontsize=10)
        ax.set_ylabel("Temperature (°C)", fontsize=10)
        ax.set_title(f"Partial Dependence: {feature_label}", fontsize=11)
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / "model_partial_dependence.png", dpi=150)
    plt.close()

    print(f"✓ Partial Dependence plots generated")


def plot_fan_efficiency(model: ThermalModel, config: Dict, output_dir: Path):
    """
    Plot fan efficiency: cooling effectiveness per PWM unit.
    Calculates temperature reduction per 10% PWM increase for each fan.
    """
    devices = config["devices"]
    pwm_names = list(devices.keys())

    # Define test conditions
    base_state = {
        "P_cpu": 100.0,
        "P_gpu": 150.0,
        "T_amb": 25.0,
    }

    # Set baseline fan speeds
    baseline_pwms = {}
    for pwm_name in pwm_names:
        min_pwm = devices[pwm_name]["min_pwm"]
        baseline_pwms[pwm_name] = (min_pwm + 100.0) / 2.0

    # Calculate efficiency for each fan
    efficiencies_cpu = []
    efficiencies_gpu = []

    for pwm_name in pwm_names:
        min_pwm = devices[pwm_name]["min_pwm"]

        # Measure temperature at low and high PWM
        pwm_low = min_pwm + 10.0  # Start 10% above minimum
        pwm_high = min(min_pwm + 30.0, 100.0)  # 20% increase, capped at 100%

        # Get temps at low PWM
        input_state_low = base_state.copy()
        input_state_low.update(baseline_pwms)
        input_state_low[pwm_name] = pwm_low
        df_low = pd.DataFrame([input_state_low])
        t_cpu_low, t_gpu_low = model.predict(df_low)

        # Get temps at high PWM
        input_state_high = base_state.copy()
        input_state_high.update(baseline_pwms)
        input_state_high[pwm_name] = pwm_high
        df_high = pd.DataFrame([input_state_high])
        t_cpu_high, t_gpu_high = model.predict(df_high)

        # Calculate efficiency (temperature reduction per 10% PWM increase)
        pwm_delta = pwm_high - pwm_low
        cpu_efficiency = -(t_cpu_high[0] - t_cpu_low[0]) / pwm_delta * 10.0
        gpu_efficiency = -(t_gpu_high[0] - t_gpu_low[0]) / pwm_delta * 10.0

        efficiencies_cpu.append(cpu_efficiency)
        efficiencies_gpu.append(gpu_efficiency)

    # Create bar chart
    x = np.arange(len(pwm_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    bars_cpu = ax.bar(
        x - width / 2,
        efficiencies_cpu,
        width,
        label="CPU Cooling",
        color="red",
        alpha=0.7,
    )
    bars_gpu = ax.bar(
        x + width / 2,
        efficiencies_gpu,
        width,
        label="GPU Cooling",
        color="green",
        alpha=0.7,
    )

    # Add value labels on bars
    for bars in [bars_cpu, bars_gpu]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_xlabel("Fan", fontsize=12)
    ax.set_ylabel("Cooling Efficiency (°C per 10% PWM)", fontsize=12)
    ax.set_title(
        "Fan Cooling Efficiency Comparison\n(Higher = More effective cooling per PWM increase)",
        fontsize=13,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([name.upper() for name in pwm_names])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(output_dir / "model_fan_efficiency.png", dpi=150)
    plt.close()

    print(f"✓ Fan Efficiency comparison generated")
