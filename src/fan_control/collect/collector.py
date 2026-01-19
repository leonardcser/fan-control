"""Data collection for thermal model fitting."""

import csv
import re
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from ..hardware import HardwareController
from ..load import LoadOrchestrator
from ..safety import SafetyMonitor, AbortPointError
from .models import MeasurementPoint, SafetyCheck, TestPoint
from ..plot.plotting import generate_all_plots
from ..utils import drop_privileges


def _parse_cpu_cores(cpu_load_flags: str) -> int:
    """
    Extract core count from cpu_load_flags (e.g., '--cpu 6' -> 6).

    Args:
        cpu_load_flags: stress flags string

    Returns:
        Number of cores, or 0 if not specified (idle)
    """
    if not cpu_load_flags:
        return 0
    match = re.search(r"--cpu\s+(\d+)", cpu_load_flags)
    return int(match.group(1)) if match else 0


class TemperatureWindow:
    """Maintains a sliding window of temperature readings for equilibration detection."""

    def __init__(self, window_duration: float, check_interval: float):
        self.window_duration = window_duration
        self.check_interval = check_interval
        self.max_samples = int(window_duration / check_interval) + 1

        # Separate queues for CPU and GPU
        self.cpu_temps = deque(maxlen=self.max_samples)
        self.gpu_temps = deque(maxlen=self.max_samples)
        self.timestamps = deque(maxlen=self.max_samples)

    def add_reading(
        self, cpu_temp: Optional[float], gpu_temp: Optional[float], timestamp: float
    ) -> None:
        """Add a temperature reading to the window."""
        self.cpu_temps.append(cpu_temp)
        self.gpu_temps.append(gpu_temp)
        self.timestamps.append(timestamp)

    def is_full(self) -> bool:
        """Check if window has enough data for stability check."""
        return len(self.timestamps) >= self.max_samples

    def check_equilibration(self, threshold: float) -> Tuple[bool, dict]:
        """
        Check if temperatures have equilibrated.

        Returns:
            (is_equilibrated, details_dict)
        """
        if not self.is_full():
            return False, {
                "cpu_range": None,
                "gpu_range": None,
                "num_samples": len(self.timestamps),
            }

        # Filter out None values
        cpu_valid = [t for t in self.cpu_temps if t is not None]
        gpu_valid = [t for t in self.gpu_temps if t is not None]

        # Calculate ranges (max - min)
        cpu_range = max(cpu_valid) - min(cpu_valid) if cpu_valid else None
        gpu_range = max(gpu_valid) - min(gpu_valid) if gpu_valid else None

        # Check stability
        cpu_stable = cpu_range is not None and cpu_range < threshold
        gpu_stable = gpu_range is not None and gpu_range < threshold

        # Equilibrated if both stable (or one missing but other stable)
        is_equilibrated = (
            (cpu_stable and gpu_stable)
            or (cpu_range is None and gpu_stable)
            or (gpu_range is None and cpu_stable)
        )

        return is_equilibrated, {
            "cpu_range": cpu_range,
            "gpu_range": gpu_range,
            "cpu_stable": cpu_stable,
            "gpu_stable": gpu_stable,
            "num_samples": len(self.timestamps),
        }


class DataCollector:
    """Collect thermal data for model fitting."""

    def __init__(
        self,
        hardware: HardwareController,
        load_orchestrator: LoadOrchestrator,
        safety: SafetyMonitor,
        config: Dict,
    ):
        self.hardware = hardware
        self.load_orchestrator = load_orchestrator
        self.safety = safety
        self.config = config

        # Extract configuration
        self.data_config = config["data_collection"]
        self.safety_config = config["safety"]
        self.hw_config = config["hardware"]
        self.devices = config["devices"]
        self.output_config = config.get("output", {})
        self.load_stabilization_time = self.data_config.get(
            "load_stabilization_time", 10
        )
        self.sampling_seed = self.data_config.get("sampling_seed", 42)

        # Dynamic equilibration config
        self.eq_check_interval = self.data_config.get(
            "equilibration_check_interval", 1.0
        )
        self.eq_window = self.data_config["equilibration_window"]
        self.eq_threshold = self.data_config.get("equilibration_threshold", 0.5)
        self.eq_max_wait = self.data_config.get("equilibration_max_wait", 120)
        self.eq_min_wait = self.data_config.get("equilibration_min_wait", 10)

        # Data storage
        self.measurements: List[MeasurementPoint] = []

    def generate_test_points_for_load(
        self, cpu_load_flags: str, gpu_load_flags: str, description: str
    ):
        """
        Generator that yields test points for a specific load level.

        Uses a two-phase sampling approach:
        1. Boundary sweep: All fans at same level, stepping down from high to low
        2. Biased LHS: Latin Hypercube samples biased toward lower fan speeds

        This allows replacing skipped/aborted points to maintain target sample count.

        Args:
            cpu_load_flags: CPU load flags (stress)
            gpu_load_flags: GPU load flags (gpu_load.py)
            description: Load description

        Yields:
            TestPoint instances
        """
        from scipy.stats import qmc

        device_keys = list(self.devices.keys())
        num_devices = len(device_keys)

        # Collect PWM ranges for each device
        levels = [np.array(self.data_config[f"{k}_levels"]) for k in device_keys]

        # Parse cpu_cores once for this load level
        cpu_cores = _parse_cpu_cores(cpu_load_flags)

        # === Phase 1: Boundary Sweep ===
        # Start with all fans at max, then sweep one fan at a time from max to min
        # This isolates each fan's contribution to cooling
        sweep_steps = self.data_config.get("boundary_sweep_steps", [100, 70, 50, 30, 0])

        # Get min/max for each device from configured levels
        device_ranges = {}
        for key in device_keys:
            lvl = self.data_config[f"{key}_levels"]
            device_ranges[key] = {"min": min(lvl), "max": max(lvl)}

        # Helper to convert percentage of range to PWM value
        def pct_to_pwm(key: str, pct: float) -> int:
            r = device_ranges[key]
            value = r["min"] + (pct / 100.0) * (r["max"] - r["min"])
            speed = int(round(value))
            # Clamp to 0 if below min_pwm (fan would stall)
            min_pwm = self.devices[key].get("min_pwm", 0)
            if 0 < speed < min_pwm:
                speed = 0
            return speed

        # Baseline: all fans at max
        baseline_pwm = {key: device_ranges[key]["max"] for key in device_keys}
        yield TestPoint(
            pwm_values=baseline_pwm.copy(),
            cpu_load_flags=cpu_load_flags,
            gpu_load_flags=gpu_load_flags,
            cpu_cores=cpu_cores,
            description=description,
        )

        # Sweep each fan independently (others stay at max)
        for sweep_key in device_keys:
            for step_pct in sweep_steps:
                # Skip 100% as it's already covered by baseline
                if step_pct == 100:
                    continue

                pwm_map = baseline_pwm.copy()
                pwm_map[sweep_key] = pct_to_pwm(sweep_key, step_pct)

                yield TestPoint(
                    pwm_values=pwm_map,
                    cpu_load_flags=cpu_load_flags,
                    gpu_load_flags=gpu_load_flags,
                    cpu_cores=cpu_cores,
                    description=description,
                )

        # === Phase 2: Biased LHS ===
        # Fill remaining samples with LHS biased toward lower fan speeds
        alpha = self.data_config.get("sampling_bias_alpha", 2.0)
        beta_param = self.data_config.get("sampling_bias_beta", 5.0)
        batch_size = self.data_config.get("num_samples_per_load", 25)
        batch_num = 0

        while True:
            seed = self.sampling_seed + batch_num
            rng = np.random.default_rng(seed=seed)

            # Generate LHS samples in [0,1] for stratification
            sampler = qmc.LatinHypercube(d=num_devices, scramble=True, seed=seed)
            uniform_samples = sampler.random(n=batch_size)

            # Generate Beta-distributed samples for bias toward lower values
            # Beta(2, 5) has mode at ~0.2, skewing samples toward lower fan speeds
            biased_samples = rng.beta(alpha, beta_param, size=uniform_samples.shape)

            # Combine LHS stratification with Beta marginals via rank matching
            # This preserves good space coverage while biasing toward low values
            for dim in range(num_devices):
                uniform_ranks = np.argsort(np.argsort(uniform_samples[:, dim]))
                sorted_biased = np.sort(biased_samples[:, dim])
                biased_samples[:, dim] = sorted_biased[uniform_ranks]

            # Scale to PWM ranges for each device
            scaled_samples = np.zeros_like(biased_samples)
            for i, lvl in enumerate(levels):
                scaled_samples[:, i] = lvl.min() + biased_samples[:, i] * (
                    lvl.max() - lvl.min()
                )

            for sample in scaled_samples:
                pwm_map = {}
                for key, val in zip(device_keys, sample):
                    speed = int(round(val))
                    min_pwm = self.devices[key].get("min_pwm", 0)
                    if 0 < speed < min_pwm:
                        speed = 0
                    pwm_map[key] = speed

                yield TestPoint(
                    pwm_values=pwm_map,
                    cpu_load_flags=cpu_load_flags,
                    gpu_load_flags=gpu_load_flags,
                    cpu_cores=cpu_cores,
                    description=description,
                )

            batch_num += 1

    def is_safe_test_point(
        self, point: TestPoint, current_power: Optional[tuple] = None
    ) -> SafetyCheck:
        """
        Check if a test point is safe to execute.

        Implements safety constraints from PHYSICS_MODEL.md to avoid overheating.

        Args:
            point: Test point to check
            current_power: Optional (cpu_power, gpu_power) if already measured

        Returns:
            SafetyCheck result
        """
        cpu_power, gpu_power = current_power or (None, None)

        # Check minimum total cooling effort
        total_cooling = sum(point.pwm_values.values())
        min_effort = self.safety_config.get("min_total_cooling_effort", 100)

        if total_cooling < min_effort:
            return SafetyCheck(
                safe=False,
                reason=f"Total cooling effort {total_cooling}% < minimum {min_effort}%",
            )

        # Check power-dependent minimums (if power is known)
        if cpu_power is not None or gpu_power is not None:
            for constraint in self.safety_config.get("power_dependent_minimums", []):
                component = constraint.get("component", "cpu")
                threshold = constraint.get("power_threshold", 0)

                # Select which power to check based on the component
                relevant_power = cpu_power if component == "cpu" else gpu_power

                if relevant_power and relevant_power > threshold:
                    for key, min_val in constraint.items():
                        if key.endswith("_min") and key != "power_threshold":
                            device_id = key.replace("_min", "")
                            if (
                                device_id in point.pwm_values
                                and point.pwm_values[device_id] < min_val
                            ):
                                return SafetyCheck(
                                    safe=False,
                                    reason=f"{component.upper()} power {relevant_power:.1f}W > {threshold}W requires {device_id} >= {min_val}%",
                                    cpu_power=cpu_power if component == "cpu" else None,
                                    gpu_power=gpu_power if component == "gpu" else None,
                                )

        return SafetyCheck(safe=True)

    def collect_measurement(self, point: TestPoint) -> Optional[MeasurementPoint]:
        """
        Execute a single test point and collect measurement.

        Args:
            point: Test point to measure

        Returns:
            MeasurementPoint if successful, None if aborted
        """
        measurement_duration = self.data_config["measurement_duration"]
        sample_interval = self.data_config["sample_interval"]

        # Measure power
        cpu_power = self.hardware.get_cpu_power()
        gpu_power = self.hardware.get_gpu_power()

        # Safety check with current power
        safety_check = self.is_safe_test_point(
            point, current_power=(cpu_power, gpu_power)
        )
        if not safety_check.safe:
            tqdm.write(f"✗ Unsafe: {safety_check.reason}")
            return None

        # Set fan speeds
        for device_id, speed in point.pwm_values.items():
            pwm_num = self.devices[device_id]["pwm_number"]
            if not self.hardware.set_fan_speed(pwm_num, speed):
                tqdm.write(f"✗ Failed to set {device_id} (PWM{pwm_num})")
                return None

        # Wait for equilibration
        try:
            actual_stabilization_time, eq_info = self.wait_for_equilibration()
        except AbortPointError:
            return None

        # Measure over duration
        measurements = {
            "cpu_temps": [],
            "gpu_temps": [],
            "cpu_powers": [],
            "gpu_powers": [],
            "ambient_temps": [],
        }

        num_samples = int(measurement_duration / sample_interval)
        for _ in range(num_samples):
            measurements["cpu_temps"].append(self.hardware.get_cpu_temp())
            measurements["gpu_temps"].append(self.hardware.get_gpu_temp())
            measurements["cpu_powers"].append(self.hardware.get_cpu_power())
            measurements["gpu_powers"].append(self.hardware.get_gpu_power())
            measurements["ambient_temps"].append(self.hardware.get_ambient_temp())

            time.sleep(sample_interval)

        # Average measurements
        cpu_temp_avg = float(
            np.nanmean([t for t in measurements["cpu_temps"] if t is not None])
        )
        gpu_temp_avg = float(
            np.nanmean([t for t in measurements["gpu_temps"] if t is not None])
        )
        cpu_power_avg = float(
            np.nanmean([p for p in measurements["cpu_powers"] if p is not None])
        )
        gpu_power_avg = float(
            np.nanmean([p for p in measurements["gpu_powers"] if p is not None])
        )
        ambient_temps = [t for t in measurements["ambient_temps"] if t is not None]
        ambient_temp_avg = float(np.nanmean(ambient_temps)) if ambient_temps else None

        # Convert PWM percentage to 0-255 for storage
        pwm_values_raw = {
            k: int(round(v * 255 / 100)) for k, v in point.pwm_values.items()
        }

        # Create measurement point
        measurement = MeasurementPoint(
            timestamp=time.time(),
            pwm_values=pwm_values_raw,
            P_cpu=cpu_power_avg,
            P_gpu=gpu_power_avg,
            T_amb=ambient_temp_avg,
            T_cpu=cpu_temp_avg,
            T_gpu=gpu_temp_avg,
            cpu_load_flags=point.cpu_load_flags,
            gpu_load_flags=point.gpu_load_flags,
            cpu_cores=point.cpu_cores,
            stabilization_time=actual_stabilization_time,
            equilibrated=eq_info["equilibrated"],
            equilibration_reason=eq_info.get("reason"),
            description=point.description,
        )

        return measurement

    def wait_for_equilibration(self) -> Tuple[float, dict]:
        """
        Wait for temperature equilibration using dynamic detection.

        Returns:
            (actual_wait_time, equilibration_info)
        """
        start_time = time.time()

        # Initialize temperature window
        window = TemperatureWindow(
            window_duration=self.eq_window, check_interval=self.eq_check_interval
        )

        eq_info = {
            "equilibrated": False,
            "reason": None,
            "final_cpu_range": None,
            "final_gpu_range": None,
        }

        with tqdm(
            total=self.eq_max_wait,
            desc="Equilibrating",
            unit="s",
            leave=False,
            bar_format="{desc}: {percentage:3.0f}% |{bar}| {n_fmt}/{total_fmt}s [{elapsed}<{remaining}{postfix}]",
        ) as pbar:
            pbar.set_postfix_str("Initializing...")
            while True:
                elapsed = time.time() - start_time
                time.sleep(self.eq_check_interval)

                # Safety check
                try:
                    self.safety.check_safety()
                except AbortPointError as e:
                    tqdm.write(f"\n✗ ABORT: {e}")
                    tqdm.write(
                        "  Restoring auto fan control and aborting this test point"
                    )
                    self.safety._apply_emergency_speeds()
                    raise

                # Read temperatures
                cpu_temp = self.hardware.get_cpu_temp()
                gpu_temp = self.hardware.get_gpu_temp()
                window.add_reading(cpu_temp, gpu_temp, time.time())

                # Update progress bar
                cpu_str = f"{cpu_temp:.1f}°C" if cpu_temp else "N/A"
                gpu_str = f"{gpu_temp:.1f}°C" if gpu_temp else "N/A"
                _, details = window.check_equilibration(self.eq_threshold)

                status = (
                    "Stable"
                    if details.get("cpu_stable") and details.get("gpu_stable")
                    else "Unstable"
                )
                pbar.set_postfix_str(f"CPU: {cpu_str}, GPU: {gpu_str} ({status})")
                pbar.n = min(int(elapsed), self.eq_max_wait)
                pbar.refresh()

                # Check timeout
                if elapsed >= self.eq_max_wait:
                    eq_info["equilibrated"] = False
                    eq_info["reason"] = f"timeout_after_{self.eq_max_wait}s"
                    eq_info["final_cpu_range"] = details["cpu_range"]
                    eq_info["final_gpu_range"] = details["gpu_range"]
                    break

                # Check equilibration (only after minimum wait)
                if elapsed >= self.eq_min_wait:
                    is_equilibrated, details = window.check_equilibration(
                        self.eq_threshold
                    )

                    if is_equilibrated:
                        eq_info["equilibrated"] = True
                        eq_info["reason"] = "equilibrated"
                        eq_info["final_cpu_range"] = details["cpu_range"]
                        eq_info["final_gpu_range"] = details["gpu_range"]
                        break

        return time.time() - start_time, eq_info

    def run_collection(self, output_path: Path) -> None:
        """
        Run full data collection campaign.

        Args:
            output_path: Path to output CSV file
        """
        load_levels = self.data_config["load_levels"]
        num_samples_per_load = self.data_config.get("num_samples_per_load", 25)
        total_target = num_samples_per_load * len(load_levels)

        print("\n" + "=" * 80)
        print("STARTING DATA COLLECTION")
        print("=" * 80)
        print(f"Target: {num_samples_per_load} measurements per load level")
        print(f"Load levels: {len(load_levels)}")
        print(f"Total target measurements: {total_target}")

        total_successful = 0
        total_skipped = 0

        with tqdm(
            total=total_target,
            desc="Total Progress",
            unit="pt",
            bar_format="{desc}: {percentage:3.0f}% |{bar}| {n_fmt}/{total_fmt} points [{elapsed}<{remaining}]",
        ) as main_pbar:
            # Process each load level
            for load_idx, load in enumerate(load_levels, 1):
                cpu_load = load.get("cpu_load", "")
                gpu_load = load.get("gpu_load", "")
                description = load["description"]

                tqdm.write(f"\n{'=' * 80}")
                tqdm.write(f"Load Level {load_idx}/{len(load_levels)}: {description}")
                tqdm.write(f"  CPU: '{cpu_load}' | GPU: '{gpu_load}'")
                tqdm.write(f"{'=' * 80}")

                # Set load for the entire group
                if not self.load_orchestrator.set_workload(cpu_load, gpu_load):
                    tqdm.write("✗ Failed to set load for this level")
                    tqdm.write(
                        f"  Skipping all {num_samples_per_load} planned measurements"
                    )
                    total_skipped += num_samples_per_load
                    main_pbar.update(num_samples_per_load)
                    continue

                # Create generator for this load level
                point_generator = self.generate_test_points_for_load(
                    cpu_load, gpu_load, description
                )

                # Collect measurements until we reach target
                successful_for_load = 0
                attempted_for_load = 0
                skipped_for_load = 0

                for point in point_generator:
                    # Stop when we have enough successful measurements
                    if successful_for_load >= num_samples_per_load:
                        break

                    attempted_for_load += 1

                    # Check if point is safe (pre-check without power)
                    safety_check = self.is_safe_test_point(point)
                    if not safety_check.safe:
                        skipped_for_load += 1
                        tqdm.write(
                            f"\n[Load {load_idx}: {successful_for_load}/{num_samples_per_load}] "
                            f"Skipping unsafe point (attempt {attempted_for_load}): {safety_check.reason}"
                        )
                        continue

                    # Collect measurement
                    measurement = self.collect_measurement(point)

                    if measurement:
                        self.measurements.append(measurement)
                        successful_for_load += 1
                        total_successful += 1
                        main_pbar.update(1)

                        # Save incrementally
                        self.save_measurements(output_path)
                    else:
                        skipped_for_load += 1

                total_skipped += skipped_for_load

        # Final save
        self.save_measurements(output_path, generate_plots=False)

        # Final summary
        print("\n" + "=" * 80)
        print("DATA COLLECTION COMPLETE")
        print("=" * 80)
        print(f"Target measurements: {num_samples_per_load * len(load_levels)}")
        print(f"Successful measurements: {total_successful}")
        print(f"Skipped/Aborted: {total_skipped}")
        print(f"Total points collected: {len(self.measurements)}")
        print(f"Data saved to: {output_path}")

        if len(self.measurements) >= 2:
            plots_dir = output_path.parent / "plots"
            print(f"Plots saved to: {plots_dir}")

        print("=" * 80 + "\n")

    def save_measurements(
        self, output_path: Path, generate_plots: bool = False
    ) -> None:
        """
        Save measurements to CSV file and generate plots.

        Args:
            output_path: Path to output CSV file
            generate_plots: Whether to generate plots (default: False)
        """
        device_keys = list(self.devices.keys())

        # Save CSV (drop privileges to ensure proper ownership)
        with drop_privileges():
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w", newline="") as f:
                writer = csv.DictWriter(
                    f, fieldnames=MeasurementPoint.csv_header(device_keys)
                )
                writer.writeheader()

                for measurement in self.measurements:
                    writer.writerow(measurement.to_dict())

        # Generate plots
        if generate_plots and len(self.measurements) >= 2:
            plots_dir = output_path.parent / "plots"
            try:
                with drop_privileges():
                    generate_all_plots(output_path, plots_dir, quiet=True)
            except Exception as e:
                # Don't fail data collection if plotting fails
                print(f"  ⚠ Plot generation failed: {e}")
