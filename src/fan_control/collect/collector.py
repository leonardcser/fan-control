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

    def check_equilibration(self, cpu_threshold: float, gpu_threshold: float) -> Tuple[bool, dict]:
        """
        Check if temperatures have equilibrated.

        Args:
            cpu_threshold: Maximum temperature range for CPU equilibration (°C)
            gpu_threshold: Maximum temperature range for GPU equilibration (°C)

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

        # Check stability with separate thresholds
        cpu_stable = cpu_range is not None and cpu_range < cpu_threshold
        gpu_stable = gpu_range is not None and gpu_range < gpu_threshold

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
        self.output_config = config["output"]
        self.load_stabilization_time = self.data_config[
            "load_stabilization_time"
        ]
        self.sampling_seed = self.data_config["sampling_seed"]

        # Dynamic equilibration config
        self.eq_check_interval = self.data_config[
            "equilibration_check_interval"
        ]
        self.eq_window = self.data_config["equilibration_window"]

        # Separate thresholds for CPU and GPU
        eq_threshold_cfg = self.data_config["equilibration_threshold"]
        self.eq_threshold_cpu = eq_threshold_cfg["cpu"]
        self.eq_threshold_gpu = eq_threshold_cfg["gpu"]

        self.eq_max_wait = self.data_config["equilibration_max_wait"]
        self.eq_min_wait = self.data_config["equilibration_min_wait"]

        # Phase control configuration
        self.enable_single_fan_sweep = self.data_config["enable_single_fan_sweep"]
        self.enable_max_fan_sweep = self.data_config["enable_max_fan_sweep"]

        # Data storage
        self.measurements: List[MeasurementPoint] = []

    def generate_test_points_for_load(
        self, cpu_load_flags: str, gpu_load_flags: str, description: str
    ):
        """
        Generator that yields test points for a specific load level.

        Uses a two-phase sampling approach (togglable):
        1. Single fan sweep (optional): Each fan varies from min to max, others at min
        2. Contrast sampling (always): Latin Hypercube samples biased toward lower fan speeds

        This allows replacing skipped/aborted points to maintain target sample count.

        Args:
            cpu_load_flags: CPU load flags (stress)
            gpu_load_flags: GPU load flags (gpu_load.py)
            description: Load description

        Yields:
            TestPoint instances
        """
        device_keys = list(self.devices.keys())
        num_devices = len(device_keys)

        # Collect PWM ranges for each device
        levels = [np.array(self.data_config[f"{k}_levels"]) for k in device_keys]

        # Parse cpu_cores once for this load level
        cpu_cores = _parse_cpu_cores(cpu_load_flags)

        # Get min/max for each device from configured levels (needed by Phase 1 and 2)
        device_ranges = {}
        for key in device_keys:
            lvl = self.data_config[f"{key}_levels"]
            device_ranges[key] = {"min": min(lvl), "max": max(lvl)}

        # Helper to convert percentage of range to PWM value (shared by Phase 1 and 2)
        def pct_to_pwm(key: str, pct: float) -> int:
            r = device_ranges[key]
            value = r["min"] + (pct / 100.0) * (r["max"] - r["min"])
            speed = int(round(value))
            # Clamp to 0 if below stall_pwm (fan would stall)
            stall_pwm = self.devices[key]["stall_pwm"]
            if 0 < speed < stall_pwm:
                speed = 0
            return speed

        # === Phase 1: Single Fan Sweep (Optional) ===
        # Run each fan individually at varying speeds (min to max), others at their minimum configured level
        # This identifies each fan's cooling contribution while maintaining baseline cooling
        if self.enable_single_fan_sweep:
            num_repeats = self.data_config["single_fan_sweep_repeats"]

            # First, generate unique points (deduplicated)
            unique_points = []
            seen_points = set()
            for sweep_key in device_keys:
                for step_pct in self.data_config["single_fan_sweep_steps"]:
                    pwm_map = {}
                    for key in device_keys:
                        if key == sweep_key:
                            pwm_map[key] = pct_to_pwm(key, step_pct)
                        else:
                            pwm_map[key] = device_ranges[key]["min"]

                    point_tuple = tuple(pwm_map[k] for k in device_keys)
                    if point_tuple not in seen_points:
                        seen_points.add(point_tuple)
                        unique_points.append(pwm_map)

            # Then yield each unique point num_repeats times
            for _repeat in range(num_repeats):
                for pwm_map in unique_points:
                    yield TestPoint(
                        pwm_values=pwm_map.copy(),
                        cpu_load_flags=cpu_load_flags,
                        gpu_load_flags=gpu_load_flags,
                        cpu_cores=cpu_cores,
                        description=description,
                    )

        # === Phase 1b: Max Fan Sweep (Optional) ===
        # Run each fan individually at varying speeds (max to min), others at their maximum (100%)
        # This identifies each fan's cooling contribution when other fans provide maximum cooling
        if self.enable_max_fan_sweep:
            num_repeats = self.data_config["max_fan_sweep_repeats"]
            max_sweep_steps = self.data_config["max_fan_sweep_steps"]

            # First, generate unique points (deduplicated)
            unique_points = []
            seen_points = set()
            for sweep_key in device_keys:
                for step_pct in max_sweep_steps:
                    pwm_map = {}
                    for key in device_keys:
                        if key == sweep_key:
                            pwm_map[key] = pct_to_pwm(key, step_pct)
                        else:
                            # Others at max (100%)
                            pwm_map[key] = device_ranges[key]["max"]

                    point_tuple = tuple(pwm_map[k] for k in device_keys)
                    if point_tuple not in seen_points:
                        seen_points.add(point_tuple)
                        unique_points.append(pwm_map)

            # Then yield each unique point num_repeats times
            for _repeat in range(num_repeats):
                for pwm_map in unique_points:
                    yield TestPoint(
                        pwm_values=pwm_map.copy(),
                        cpu_load_flags=cpu_load_flags,
                        gpu_load_flags=gpu_load_flags,
                        cpu_cores=cpu_cores,
                        description=description,
                    )

        # === Phase 2: Contrast Sampling ===
        # Generate samples with contrast: 1-2 fans high, rest low
        # This ensures we capture each fan's individual contribution
        contrast_config = self.data_config["contrast_sampling"]
        num_high_fans_choices = contrast_config["num_high_fans"]
        high_range = contrast_config["high_range"]
        low_range = contrast_config["low_range"]

        batch_size = self.data_config["num_samples_per_load"]
        batch_num = 0

        while True:
            seed = self.sampling_seed + batch_num
            rng = np.random.default_rng(seed=seed)

            for _ in range(batch_size):
                # Randomly choose how many fans will be "high" (1 or 2)
                num_high = rng.choice(num_high_fans_choices)

                # Randomly select which fans will be high
                high_fan_indices = rng.choice(
                    num_devices, size=num_high, replace=False
                )

                pwm_map = {}
                for i, key in enumerate(device_keys):
                    lvl = levels[i]
                    lvl_min, lvl_max = lvl.min(), lvl.max()

                    if i in high_fan_indices:
                        # High fan: uniform sample from high_range (as % of device range)
                        pct = rng.uniform(high_range[0], high_range[1]) / 100.0
                    else:
                        # Low fan: beta-biased sample from low_range
                        # Beta(2, 5) biases toward lower end of the range
                        beta_val = rng.beta(2.0, 5.0)
                        low_pct_range = (low_range[1] - low_range[0]) / 100.0
                        pct = (low_range[0] / 100.0) + beta_val * low_pct_range

                    # Convert percentage to actual PWM value
                    speed = int(round(lvl_min + pct * (lvl_max - lvl_min)))

                    # Clamp: if below stall_pwm but > 0, set to 0 (fan would stall)
                    stall_pwm = self.devices[key]["stall_pwm"]
                    if 0 < speed < stall_pwm:
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
        min_effort = self.safety_config["min_total_cooling_effort"]

        if total_cooling < min_effort:
            return SafetyCheck(
                safe=False,
                reason=f"Total cooling effort {total_cooling}% < minimum {min_effort}%",
            )

        # Check power-dependent minimums (if power is known)
        if cpu_power is not None or gpu_power is not None:
            for constraint in self.safety_config["power_dependent_minimums"]:
                component = constraint["component"]
                threshold = constraint["power_threshold"]

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
        cpu_power_data = self.hardware.get_cpu_power()
        cpu_power = cpu_power_data['package'] if cpu_power_data else None
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
            "cpu_core_metrics": [],  # List of dicts with per-core metrics, one per sample
            "gpu_powers": [],
            "gpu_fan_speeds": [],
            "ambient_temps": [],
        }

        num_samples = int(measurement_duration / sample_interval)
        for _ in range(num_samples):
            measurements["cpu_temps"].append(self.hardware.get_cpu_temp())
            measurements["gpu_temps"].append(self.hardware.get_gpu_temp())

            # Get CPU power and performance metrics (package and per-core)
            cpu_power_data = self.hardware.get_cpu_power()
            if cpu_power_data:
                measurements["cpu_powers"].append(cpu_power_data['package'])
                measurements["cpu_core_metrics"].append(cpu_power_data['cores'])
            else:
                measurements["cpu_powers"].append(None)
                measurements["cpu_core_metrics"].append(None)

            measurements["gpu_powers"].append(self.hardware.get_gpu_power())
            measurements["gpu_fan_speeds"].append(self.hardware.get_gpu_fan_speed())
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

        # Average GPU fan speed
        gpu_fan_speeds = [s for s in measurements["gpu_fan_speeds"] if s is not None]
        gpu_fan_speed_avg = int(round(np.nanmean(gpu_fan_speeds))) if gpu_fan_speeds else None

        # Average per-core metrics
        cpu_core_power_avg = None
        cpu_avg_mhz_avg = None
        cpu_bzy_mhz_avg = None
        cpu_busy_pct_avg = None

        valid_core_samples = [cm for cm in measurements["cpu_core_metrics"] if cm is not None]
        if valid_core_samples:
            # Aggregate all core measurements across samples
            core_power_sums = {}
            core_avg_mhz_sums = {}
            core_bzy_mhz_sums = {}
            core_busy_pct_sums = {}
            core_counts = {}

            for sample in valid_core_samples:
                for core_num, metrics in sample.items():
                    if core_num not in core_counts:
                        core_power_sums[core_num] = 0
                        core_avg_mhz_sums[core_num] = 0
                        core_bzy_mhz_sums[core_num] = 0
                        core_busy_pct_sums[core_num] = 0
                        core_counts[core_num] = 0

                    core_power_sums[core_num] += metrics['power']
                    core_avg_mhz_sums[core_num] += metrics['avg_mhz']
                    core_bzy_mhz_sums[core_num] += metrics['bzy_mhz']
                    core_busy_pct_sums[core_num] += metrics['busy_pct']
                    core_counts[core_num] += 1

            # Calculate averages
            cpu_core_power_avg = {
                core_num: core_power_sums[core_num] / core_counts[core_num]
                for core_num in core_counts
            }
            cpu_avg_mhz_avg = {
                core_num: core_avg_mhz_sums[core_num] / core_counts[core_num]
                for core_num in core_counts
            }
            cpu_bzy_mhz_avg = {
                core_num: core_bzy_mhz_sums[core_num] / core_counts[core_num]
                for core_num in core_counts
            }
            cpu_busy_pct_avg = {
                core_num: core_busy_pct_sums[core_num] / core_counts[core_num]
                for core_num in core_counts
            }

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
            T_cpu=cpu_temp_avg,
            T_gpu=gpu_temp_avg,
            cpu_load_flags=point.cpu_load_flags,
            gpu_load_flags=point.gpu_load_flags,
            cpu_cores=point.cpu_cores,
            stabilization_time=actual_stabilization_time,
            P_cpu_cores=cpu_core_power_avg,
            cpu_avg_mhz=cpu_avg_mhz_avg,
            cpu_bzy_mhz=cpu_bzy_mhz_avg,
            cpu_busy_pct=cpu_busy_pct_avg,
            T_amb=ambient_temp_avg,
            gpu_fan_speed=gpu_fan_speed_avg,
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
                        f"  Setting fans to full speed and cooling down for {self.safety.abort_cooldown_time}s..."
                    )
                    self.safety._apply_abort_speeds()
                    # Wait for cooldown at full speed
                    time.sleep(self.safety.abort_cooldown_time)
                    raise

                # Read temperatures
                cpu_temp = self.hardware.get_cpu_temp()
                gpu_temp = self.hardware.get_gpu_temp()
                window.add_reading(cpu_temp, gpu_temp, time.time())

                # Update progress bar
                cpu_str = f"{cpu_temp:.1f}°C" if cpu_temp else "N/A"
                gpu_str = f"{gpu_temp:.1f}°C" if gpu_temp else "N/A"
                _, details = window.check_equilibration(self.eq_threshold_cpu, self.eq_threshold_gpu)

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
                        self.eq_threshold_cpu, self.eq_threshold_gpu
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
        num_samples_per_load = self.data_config["num_samples_per_load"]
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
                cpu_load = load["cpu_load"]
                gpu_load = load["gpu_load"]
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

        # Determine number of cores from measurements
        num_cores = 12  # default
        for measurement in self.measurements:
            # Check any of the per-core fields to determine core count
            if measurement.P_cpu_cores:
                num_cores = max(measurement.P_cpu_cores.keys()) + 1
                break
            elif measurement.cpu_avg_mhz:
                num_cores = max(measurement.cpu_avg_mhz.keys()) + 1
                break

        # Save CSV (drop privileges to ensure proper ownership)
        with drop_privileges():
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w", newline="") as f:
                writer = csv.DictWriter(
                    f, fieldnames=MeasurementPoint.csv_header(device_keys, num_cores)
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
