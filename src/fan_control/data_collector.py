"""Data collection for thermal model fitting."""

import csv
import itertools
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .data import MeasurementPoint, SafetyCheck, TestPoint
from .hardware import HardwareController
from .load import LoadOrchestrator
from .plotting import generate_all_plots
from .safety import SafetyMonitor, SkipPointError
from .utils import drop_privileges


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

        # Check if using new equilibration config
        if "equilibration_window" in self.data_config:
            self.use_dynamic_equilibration = True
            self.eq_check_interval = self.data_config.get(
                "equilibration_check_interval", 1.0
            )
            self.eq_window = self.data_config.get("equilibration_window", 10.0)
            self.eq_threshold = self.data_config.get("equilibration_threshold", 0.5)
            self.eq_max_wait = self.data_config.get("equilibration_max_wait", 120)
            self.eq_min_wait = self.data_config.get("equilibration_min_wait", 10)
        else:
            # Fall back to old fixed stabilization
            self.use_dynamic_equilibration = False
            self.stabilization_time = self.data_config.get("stabilization_time", 45)

        # Data storage
        self.measurements: List[MeasurementPoint] = []

    def generate_test_points(self) -> List[TestPoint]:
        """
        Generate test points using configured sampling strategy.

        Returns:
            List of test points to measure
        """
        strategy = self.data_config.get("sampling_strategy", "latin_hypercube")
        num_samples_per_load = self.data_config.get("num_samples_per_load", 25)

        device_keys = list(self.devices.keys())
        num_devices = len(device_keys)

        # Collect levels for each device dynamically
        levels = [np.array(self.data_config[f"{k}_levels"]) for k in device_keys]
        load_levels = self.data_config["load_levels"]

        test_points = []

        if strategy == "latin_hypercube":
            # Latin Hypercube Sampling - good coverage with fewer points
            from scipy.stats import qmc

            sampler = qmc.LatinHypercube(d=num_devices, seed=self.sampling_seed)
            samples = sampler.random(n=num_samples_per_load)

            # Scale to PWM ranges
            scaled_samples = qmc.scale(
                samples,
                l_bounds=[lvl.min() for lvl in levels],
                u_bounds=[lvl.max() for lvl in levels],
            )

            # For each load level, create test points from samples
            for load in load_levels:
                for sample in scaled_samples:
                    pwm_map = {
                        key: int(round(val)) for key, val in zip(device_keys, sample)
                    }
                    test_points.append(
                        TestPoint(
                            pwm_values=pwm_map,
                            cpu_percent=load["cpu_percent"],
                            gpu_percent=load["gpu_percent"],
                            description=load["description"],
                        )
                    )

        elif strategy == "random":
            # Random sampling
            rng = np.random.default_rng(seed=self.sampling_seed)

            for load in load_levels:
                for _ in range(num_samples_per_load):
                    pwm_map = {
                        key: int(rng.choice(lvl))
                        for key, lvl in zip(device_keys, levels)
                    }
                    test_points.append(
                        TestPoint(
                            pwm_values=pwm_map,
                            cpu_percent=load["cpu_percent"],
                            gpu_percent=load["gpu_percent"],
                            description=load["description"],
                        )
                    )

        elif strategy == "grid":
            # Grid sampling (recursive to handle dynamic number of fans)
            import itertools

            for load in load_levels:
                for pwm_values in itertools.product(*levels):
                    pwm_map = {key: val for key, val in zip(device_keys, pwm_values)}
                    test_points.append(
                        TestPoint(
                            pwm_values=pwm_map,
                            cpu_percent=load["cpu_percent"],
                            gpu_percent=load["gpu_percent"],
                            description=load["description"],
                        )
                    )

        return test_points

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

        pwm_str = ", ".join([f"{k}={v}%" for k, v in point.pwm_values.items()])
        print(f"\n{'=' * 80}")
        print(f"Test Point: {point.description}")
        print(f"  Load: CPU {point.cpu_percent}%, GPU {point.gpu_percent}%")
        print(f"  PWM: {pwm_str}")
        print(f"{'=' * 80}")

        # Measure power
        cpu_power = self.hardware.get_cpu_power()
        gpu_power = self.hardware.get_gpu_power()

        print(
            f"  Power: CPU {cpu_power:.1f}W, GPU {gpu_power:.1f}W"
            if cpu_power and gpu_power
            else "  Power: Not available"
        )

        # Safety check with current power
        safety_check = self.is_safe_test_point(
            point, current_power=(cpu_power, gpu_power)
        )
        if not safety_check.safe:
            print(f"✗ Test point unsafe: {safety_check.reason}")
            print("  Skipping this test point")
            return None

        # Set fan speeds
        print("Setting fan speeds...")
        for device_id, speed in point.pwm_values.items():
            pwm_num = self.devices[device_id]["pwm_number"]
            if not self.hardware.set_fan_speed(pwm_num, speed):
                print(f"✗ Failed to set {device_id} (PWM{pwm_num})")
                return None

        # Wait for stabilization/equilibration
        if self.use_dynamic_equilibration:
            try:
                actual_stabilization_time, eq_info = self.wait_for_equilibration()
            except SkipPointError:
                return None
        else:
            try:
                actual_stabilization_time = self._wait_fixed_stabilization()
                eq_info = {
                    "method": "fixed",
                    "equilibrated": True,
                    "reason": f"fixed_{self.stabilization_time}s",
                }
            except SkipPointError:
                return None

        # Measure over duration
        print(f"Measuring for {measurement_duration}s...")
        measurements = {
            "cpu_temps": [],
            "gpu_temps": [],
            "cpu_powers": [],
            "gpu_powers": [],
            "ambient_temps": [],
        }

        for _ in range(int(measurement_duration / sample_interval)):
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

        print(f"  Averaged: CPU {cpu_temp_avg:.1f}°C, GPU {gpu_temp_avg:.1f}°C")
        print(f"  Power: CPU {cpu_power_avg:.1f}W, GPU {gpu_power_avg:.1f}W")
        if ambient_temp_avg:
            print(f"  Ambient: {ambient_temp_avg:.1f}°C")

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
            cpu_load_target=point.cpu_percent,
            gpu_load_target=point.gpu_percent,
            stabilization_time=actual_stabilization_time,
            equilibration_method=eq_info["method"],
            equilibrated=eq_info["equilibrated"],
            equilibration_reason=eq_info.get("reason"),
            description=point.description,
        )

        print("✓ Measurement complete")
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
            "method": "dynamic",
            "equilibrated": False,
            "reason": None,
            "final_cpu_range": None,
            "final_gpu_range": None,
        }

        print(f"Waiting for thermal equilibrium (max {self.eq_max_wait}s)...")
        print(f"  Criteria: ΔT < {self.eq_threshold}°C over {self.eq_window}s window")

        iteration = 0

        while True:
            elapsed = time.time() - start_time
            time.sleep(self.eq_check_interval)
            iteration += 1

            # Safety check
            try:
                self.safety.check_safety()
            except SkipPointError as e:
                print(f"\n✗ SKIP: {e}")
                print("  Restoring auto fan control and skipping this test point")
                self.safety._apply_emergency_speeds()
                raise

            # Read temperatures
            cpu_temp = self.hardware.get_cpu_temp()
            gpu_temp = self.hardware.get_gpu_temp()
            window.add_reading(cpu_temp, gpu_temp, time.time())

            # Log progress every 10 seconds
            if iteration % 10 == 0 or iteration == 1:
                cpu_str = f"{cpu_temp:.1f}°C" if cpu_temp else "N/A"
                gpu_str = f"{gpu_temp:.1f}°C" if gpu_temp else "N/A"
                print(f"  {int(elapsed):3d}s: CPU {cpu_str}, GPU {gpu_str}")

            # Check timeout
            if elapsed >= self.eq_max_wait:
                eq_info["equilibrated"] = False
                eq_info["reason"] = f"timeout_after_{self.eq_max_wait}s"
                _, details = window.check_equilibration(self.eq_threshold)
                eq_info["final_cpu_range"] = details["cpu_range"]
                eq_info["final_gpu_range"] = details["gpu_range"]
                print(f"\n  ⚠ Timeout reached ({self.eq_max_wait}s)")
                if details["cpu_range"] is not None:
                    print(f"  Final CPU range: {details['cpu_range']:.2f}°C")
                if details["gpu_range"] is not None:
                    print(f"  Final GPU range: {details['gpu_range']:.2f}°C")
                break

            # Check equilibration (only after minimum wait)
            if elapsed >= self.eq_min_wait:
                is_equilibrated, details = window.check_equilibration(self.eq_threshold)

                if is_equilibrated:
                    eq_info["equilibrated"] = True
                    eq_info["reason"] = "equilibrated"
                    eq_info["final_cpu_range"] = details["cpu_range"]
                    eq_info["final_gpu_range"] = details["gpu_range"]
                    print(f"\n  ✓ Equilibrium reached after {elapsed:.1f}s")
                    if details["cpu_range"] is not None:
                        print(f"  CPU range: {details['cpu_range']:.2f}°C")
                    if details["gpu_range"] is not None:
                        print(f"  GPU range: {details['gpu_range']:.2f}°C")
                    break

        return time.time() - start_time, eq_info

    def _wait_fixed_stabilization(self) -> float:
        """Legacy fixed-time stabilization wait (backward compatibility)."""
        stabilization_time = self.stabilization_time
        print(f"Waiting {stabilization_time}s for thermal equilibrium (fixed)...")
        start = time.time()

        for i in range(stabilization_time):
            time.sleep(1)

            try:
                self.safety.check_safety()
            except SkipPointError as e:
                print(f"\n✗ SKIP: {e}")
                print("  Restoring auto fan control and skipping this test point")
                self.safety._apply_emergency_speeds()
                raise

            cpu_temp = self.hardware.get_cpu_temp()
            gpu_temp = self.hardware.get_gpu_temp()

            if (i + 1) % 10 == 0 or i == 0:
                cpu_str = f"{cpu_temp:.1f}°C" if cpu_temp else "N/A"
                gpu_str = f"{gpu_temp:.1f}°C" if gpu_temp else "N/A"
                print(f"  {i + 1:3d}s: CPU {cpu_str}, GPU {gpu_str}")

        return time.time() - start

    def run_collection(self, output_path: Path) -> None:
        """
        Run full data collection campaign.

        Args:
            output_path: Path to output CSV file
        """
        # Generate test points
        print("\nGenerating test points...")
        test_points = self.generate_test_points()
        print(f"Generated {len(test_points)} test points")

        # Filter safe test points (pre-check without power)
        safe_points = []
        for point in test_points:
            check = self.is_safe_test_point(point)
            if check.safe:
                safe_points.append(point)
            else:
                print(f"Skipping unsafe point: {check.reason}")

        print(f"\n{len(safe_points)} safe test points to collect")

        # Collect measurements
        print("\n" + "=" * 80)
        print("STARTING DATA COLLECTION")
        print("=" * 80)

        successful = 0
        aborted = 0

        # Sort points by load to group them effectively
        safe_points.sort(key=lambda p: (p.cpu_percent, p.gpu_percent))
        total_points = len(safe_points)
        processed_count = 0

        # Group by load level
        for (cpu_load, gpu_load), group_iter in itertools.groupby(
            safe_points, key=lambda p: (p.cpu_percent, p.gpu_percent)
        ):
            points_in_group = list(group_iter)
            print(
                f"\nSetting Load Group: CPU {cpu_load}%, GPU {gpu_load}% ({len(points_in_group)} points)"
            )

            # Set load for the entire group
            if not self.load_orchestrator.set_workload(cpu_load, gpu_load):
                print("✗ Failed to set load for group")
                aborted += len(points_in_group)
                processed_count += len(points_in_group)
                continue

            # Process points in this group
            for point in points_in_group:
                processed_count += 1
                print(f"\n[{processed_count}/{total_points}]")

                measurement = self.collect_measurement(point)

                if measurement:
                    self.measurements.append(measurement)
                    successful += 1

                    # Save incrementally
                    self.save_measurements(output_path)
                else:
                    aborted += 1

        # Final save and plot
        self.save_measurements(output_path, generate_plots=True)

        # Final summary
        print("\n" + "=" * 80)
        print("DATA COLLECTION COMPLETE")
        print("=" * 80)
        print(f"Successful measurements: {successful}")
        print(f"Aborted measurements: {aborted}")
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
