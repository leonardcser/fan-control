import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
import logging

from ..control.optimizer import Optimizer
from ..fit.train import ThermalModel

logger = logging.getLogger(__name__)


def run_simulation(model_path: Path, config: Dict, output_dir: Path):
    """
    Run a simulation of the optimizer against increasing loads.
    """
    optimizer = Optimizer(model_path, config)

    # Define Targets
    targets = {"T_cpu": 75.0, "T_gpu": 70.0}

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
        dev_cfg = config["devices"].get(name, {})
        last_pwms[name] = float(dev_cfg.get("min_pwm", 20))

    import time as time_module

    for t in range(steps):
        state = {"P_cpu": p_cpu[t], "P_gpu": p_gpu[t], "T_amb": t_amb}

        # Optimize
        t_start = time_module.perf_counter()
        pwms = optimizer.optimize(state, targets, initial_guess=last_pwms)
        t_end = time_module.perf_counter()

        # Update state for next iteration
        last_pwms = {k: float(v) for k, v in pwms.items()}

        # Predict result temperature at this optimal point
        full_input = state.copy()
        full_input.update(pwms)
        input_df = pd.DataFrame([full_input])

        # Feature engineering inside predict
        t_cpu_pred, t_gpu_pred = optimizer.model.predict(input_df)

        res = {
            "time": t,
            "P_cpu": state["P_cpu"],
            "P_gpu": state["P_gpu"],
            "T_pred_cpu": t_cpu_pred[0],
            "T_pred_gpu": t_gpu_pred[0],
            "opt_time_ms": (t_end - t_start) * 1000.0,
            **pwms,
        }
        results.append(res)

    df = pd.DataFrame(results)

    # Save CSV
    df.to_csv(output_dir / "simulation_results.csv", index=False)
    print(f"Simulation data saved to {output_dir}/simulation_results.csv")

    # Generate Plots
    generate_simulation_plots(df, targets, output_dir)


def generate_simulation_plots(df: pd.DataFrame, targets: Dict, output_dir: Path):
    """Plot simulation results."""
    sns.set_theme(style="whitegrid")

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
        label="Predicted CPU Temp",
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

    # Plot relevant fans for CPU (pwm2=Rad, pwm7=Pump, pwm4=Intake, pwm5=Exhaust)
    l4 = ax2.plot(
        df["time"], df["pwm2"], label="Rad Fan (pwm2)", color="blue", alpha=0.7
    )
    l5 = ax2.plot(df["time"], df["pwm7"], label="Pump (pwm7)", color="cyan", alpha=0.7)

    # Legend
    lines = l1 + l2 + [l3] + l4 + l5
    labs = [l.get_label() for l in lines]
    ax1.legend(lines, labs, loc="upper left")

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
        label="Predicted GPU Temp",
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

    l4 = ax2.plot(
        df["time"], df["pwm4"], label="Intake (pwm4)", color="blue", alpha=0.7
    )
    l5 = ax2.plot(
        df["time"], df["pwm5"], label="Exhaust (pwm5)", color="purple", alpha=0.7
    )

    lines = l1 + l2 + [l3] + l4 + l5
    labs = [l.get_label() for l in lines]
    ax1.legend(lines, labs, loc="upper left")

    plt.title("Simulation: GPU Thermal Response")
    plt.tight_layout()
    plt.savefig(output_dir / "sim_gpu_response.png")
    plt.close()

    # 3. Combined / All Fans
    plt.figure(figsize=(12, 6))
    plt.plot(df["time"], df["pwm2"], label="Rad Fan")
    plt.plot(df["time"], df["pwm4"], label="Intake")
    plt.plot(df["time"], df["pwm5"], label="Exhaust")
    plt.plot(df["time"], df["pwm7"], label="Pump")
    plt.ylabel("PWM")
    plt.xlabel("Time")
    plt.title("Simulation: All Fan Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "sim_all_fans.png")
    plt.close()

    # 4. Timing Plot
    plt.figure(figsize=(12, 4))
    plt.plot(df["time"], df["opt_time_ms"], color="purple", label="Optimization Time")
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
