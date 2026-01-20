"""
Validation controller that logs predicted vs actual temperatures.
Used to evaluate model accuracy during live operation.
"""

import csv
import logging
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import yaml

from ..fit.train import ThermalModel
from ..hardware import HardwareController

logger = logging.getLogger(__name__)


class ValidatingController:
    """
    Control loop that logs model predictions alongside actual temperatures.
    Generates comparison plots on shutdown.
    """

    def __init__(self, config_path: Path, model_path: Path, output_dir: Path):
        self.running = False
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load Config
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        # Initialize Hardware
        hw_cfg = self.config["hardware"]
        self.hw = HardwareController(
            hwmon_device_name=hw_cfg["hwmon_device_name"],
            cpu_sensor_name=hw_cfg["cpu_sensor_name"],
            cpu_sensor_label=hw_cfg["cpu_sensor_label"],
            ambient_config=self.config["ambient"],
        )

        # Load thermal model
        self.model = ThermalModel.load(model_path)
        self.feature_order = self.model.feature_names_in_

        # Get PWM names from config devices
        self.pwm_names = list(self.config["devices"].keys())

        # Control interval
        self.interval = self.config["controller"]["interval"]

        # CSV logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = self.output_dir / f"validation_{timestamp}.csv"
        self.records: List[Dict] = []

    def start(self):
        """Start the validation loop."""
        self.running = True

        # Enable manual mode for all controlled fans
        for pwm_name in self.pwm_names:
            dev_cfg = self.config["devices"][pwm_name]
            pwm_num = dev_cfg["pwm_number"]
            logger.info(f"Enabling manual control for {pwm_name} (pwm{pwm_num})")
            self.hw.enable_manual_control(pwm_num)

        logger.info(f"Starting validation loop (Interval: {self.interval}s)")
        logger.info(f"Logging to: {self.csv_path}")

        # Handle graceful shutdown
        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

        # Initialize CSV with header
        self._init_csv()

        try:
            while self.running:
                self._tick()
                time.sleep(self.interval)
        except Exception as e:
            logger.error(f"Validation loop crashed: {e}")
            self._emergency_mode()
            raise

    def _init_csv(self):
        """Initialize CSV file with header."""
        fieldnames = [
            "timestamp",
            "P_cpu",
            "P_gpu",
            "T_amb",
            "T_cpu_actual",
            "T_gpu_actual",
            "T_cpu_predicted",
            "T_gpu_predicted",
            "T_cpu_error",
            "T_gpu_error",
        ]
        # Add PWM columns
        for name in self.pwm_names:
            fieldnames.append(name)

        with open(self.csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    def _tick(self):
        """Single validation iteration."""
        state = self._read_sensors()
        if not state:
            logger.warning("Sensor read failed. Skipping tick.")
            return

        # Read current PWM values
        pwm_values = {}
        for name in self.pwm_names:
            dev_cfg = self.config["devices"][name]
            pwm_num = dev_cfg["pwm_number"]
            raw_pwm = self.hw.get_pwm_value(pwm_num)
            pwm_values[name] = (raw_pwm / 2.55) if raw_pwm is not None else 50.0

        # Build feature vector for prediction
        features = np.zeros(len(self.feature_order))
        for i, feat_name in enumerate(self.feature_order):
            if feat_name in state:
                features[i] = state[feat_name]
            elif feat_name in pwm_values:
                features[i] = pwm_values[feat_name]

        # Get model prediction
        t_cpu_pred, t_gpu_pred = self.model.predict_numpy(features)
        t_cpu_pred = float(t_cpu_pred[0])
        t_gpu_pred = float(t_gpu_pred[0])

        # Calculate errors
        t_cpu_error = t_cpu_pred - state["T_cpu"]
        t_gpu_error = t_gpu_pred - state["T_gpu"]

        # Build record
        record = {
            "timestamp": datetime.now().isoformat(),
            "P_cpu": state["P_cpu"],
            "P_gpu": state["P_gpu"],
            "T_amb": state["T_amb"],
            "T_cpu_actual": state["T_cpu"],
            "T_gpu_actual": state["T_gpu"],
            "T_cpu_predicted": t_cpu_pred,
            "T_gpu_predicted": t_gpu_pred,
            "T_cpu_error": t_cpu_error,
            "T_gpu_error": t_gpu_error,
        }
        for name in self.pwm_names:
            record[name] = pwm_values[name]

        self.records.append(record)

        # Append to CSV
        self._append_csv(record)

        logger.info(
            f"CPU: {state['T_cpu']:.1f}°C (pred: {t_cpu_pred:.1f}°C, err: {t_cpu_error:+.1f}°C) | "
            f"GPU: {state['T_gpu']:.1f}°C (pred: {t_gpu_pred:.1f}°C, err: {t_gpu_error:+.1f}°C)"
        )

    def _append_csv(self, record: Dict):
        """Append a single record to CSV."""
        fieldnames = list(record.keys())
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(record)

    def _read_sensors(self) -> Optional[Dict[str, float]]:
        """Read current system state."""
        try:
            t_cpu = self.hw.get_cpu_temp()
            t_gpu = self.hw.get_gpu_temp()
            if t_cpu is None or t_gpu is None:
                return None
            return {
                "P_cpu": self.hw.get_cpu_power() or 0.0,
                "P_gpu": self.hw.get_gpu_power() or 0.0,
                "T_amb": self.hw.get_ambient_temp() or 25.0,
                "T_cpu": t_cpu,
                "T_gpu": t_gpu,
            }
        except Exception as e:
            logger.error(f"Sensor read error: {e}")
            return None

    def _emergency_mode(self):
        """Set all fans to 100%."""
        logger.warning("Applying EMERGENCY FAN SPEEDS (100%)")
        pwm_nums = [
            self.config["devices"][name]["pwm_number"] for name in self.pwm_names
        ]
        self.hw.set_all_fans_max(pwm_nums)

    def _restore_auto_control(self):
        """Restore fans to automatic control."""
        logger.info("Restoring fans to automatic control")
        for pwm_name in self.pwm_names:
            pwm_num = self.config["devices"][pwm_name]["pwm_number"]
            self.hw.enable_auto_control(pwm_num)
            logger.info(f"Restored {pwm_name} (pwm{pwm_num}) to auto control")

    def _generate_plots(self):
        """Generate prediction vs actual comparison plots."""
        if not self.records:
            logger.warning("No records to plot")
            return

        logger.info("Generating validation plots...")

        # Convert records to arrays
        t_cpu_actual = np.array([r["T_cpu_actual"] for r in self.records])
        t_cpu_pred = np.array([r["T_cpu_predicted"] for r in self.records])
        t_gpu_actual = np.array([r["T_gpu_actual"] for r in self.records])
        t_gpu_pred = np.array([r["T_gpu_predicted"] for r in self.records])
        timestamps = np.arange(len(self.records))

        # Calculate metrics
        cpu_rmse = np.sqrt(np.mean((t_cpu_pred - t_cpu_actual) ** 2))
        gpu_rmse = np.sqrt(np.mean((t_gpu_pred - t_gpu_actual) ** 2))
        cpu_mae = np.mean(np.abs(t_cpu_pred - t_cpu_actual))
        gpu_mae = np.mean(np.abs(t_gpu_pred - t_gpu_actual))

        # Plot 1: Time series comparison
        _fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

        # CPU subplot
        axes[0].plot(timestamps, t_cpu_actual, label="Actual", color="blue", linewidth=1.5)
        axes[0].plot(
            timestamps, t_cpu_pred, label="Predicted", color="red", linestyle="--", linewidth=1.5
        )
        axes[0].fill_between(
            timestamps,
            t_cpu_actual,
            t_cpu_pred,
            alpha=0.3,
            color="gray",
            label="Error",
        )
        axes[0].set_ylabel("CPU Temperature (°C)")
        axes[0].set_title(f"CPU Temperature: Predicted vs Actual (RMSE={cpu_rmse:.2f}°C, MAE={cpu_mae:.2f}°C)")
        axes[0].legend(loc="upper right")
        axes[0].grid(True, alpha=0.3)

        # GPU subplot
        axes[1].plot(timestamps, t_gpu_actual, label="Actual", color="blue", linewidth=1.5)
        axes[1].plot(
            timestamps, t_gpu_pred, label="Predicted", color="red", linestyle="--", linewidth=1.5
        )
        axes[1].fill_between(
            timestamps,
            t_gpu_actual,
            t_gpu_pred,
            alpha=0.3,
            color="gray",
            label="Error",
        )
        axes[1].set_xlabel("Sample")
        axes[1].set_ylabel("GPU Temperature (°C)")
        axes[1].set_title(f"GPU Temperature: Predicted vs Actual (RMSE={gpu_rmse:.2f}°C, MAE={gpu_mae:.2f}°C)")
        axes[1].legend(loc="upper right")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = self.output_dir / "validation_timeseries.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        logger.info(f"Saved time series plot to {plot_path}")

        # Plot 2: Scatter plot (predicted vs actual)
        _fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # CPU scatter
        axes[0].scatter(t_cpu_actual, t_cpu_pred, alpha=0.5, s=10)
        min_val = min(t_cpu_actual.min(), t_cpu_pred.min())
        max_val = max(t_cpu_actual.max(), t_cpu_pred.max())
        axes[0].plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect fit")
        axes[0].set_xlabel("Actual CPU Temp (°C)")
        axes[0].set_ylabel("Predicted CPU Temp (°C)")
        axes[0].set_title(f"CPU: RMSE={cpu_rmse:.2f}°C")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # GPU scatter
        axes[1].scatter(t_gpu_actual, t_gpu_pred, alpha=0.5, s=10, color="green")
        min_val = min(t_gpu_actual.min(), t_gpu_pred.min())
        max_val = max(t_gpu_actual.max(), t_gpu_pred.max())
        axes[1].plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect fit")
        axes[1].set_xlabel("Actual GPU Temp (°C)")
        axes[1].set_ylabel("Predicted GPU Temp (°C)")
        axes[1].set_title(f"GPU: RMSE={gpu_rmse:.2f}°C")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        scatter_path = self.output_dir / "validation_scatter.png"
        plt.savefig(scatter_path, dpi=150)
        plt.close()
        logger.info(f"Saved scatter plot to {scatter_path}")

        # Print summary
        print(f"\n{'='*60}")
        print("VALIDATION SUMMARY")
        print(f"{'='*60}")
        print(f"Samples collected: {len(self.records)}")
        print(f"CPU - RMSE: {cpu_rmse:.2f}°C, MAE: {cpu_mae:.2f}°C")
        print(f"GPU - RMSE: {gpu_rmse:.2f}°C, MAE: {gpu_mae:.2f}°C")
        print(f"CSV: {self.csv_path}")
        print(f"Plots: {self.output_dir}")
        print(f"{'='*60}\n")

    def _shutdown(self, _signum, _frame):
        """Graceful shutdown handler."""
        logger.info("Shutdown signal received. Stopping...")
        self.running = False
        self._restore_auto_control()
        self._generate_plots()
        sys.exit(0)


def validate_mode(args) -> None:
    """
    Run the validation controller to compare model predictions vs actual temps.
    """
    print("\n" + "=" * 70)
    print("FAN CONTROL - MODEL VALIDATION")
    print("=" * 70 + "\n")

    config_path = Path(args.config)
    run_dir = Path(args.run)

    # Locate model file
    model_path = run_dir / "fit" / "thermal_model.pkl"

    if not model_path.exists():
        print(f"Model not found at: {model_path}")
        print("  Please run 'fan-control repro' first.")
        sys.exit(1)

    # Output directory for validation results
    output_dir = run_dir / "validation"

    print(f"Config: {config_path}")
    print(f"Model: {model_path}")
    print(f"Output: {output_dir}")
    print("\nStarting validation... (Press Ctrl+C to stop and generate plots)")

    try:
        controller = ValidatingController(config_path, model_path, output_dir)
        controller.start()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"\nFatal error: {e}")
        sys.exit(1)
