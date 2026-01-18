"""Data collection for thermal model fitting."""

import csv
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .data import MeasurementPoint, SafetyCheck, TestPoint
from .hardware import HardwareController
from .load import LoadOrchestrator
from .safety import SafetyMonitor, SkipPointError


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
        self.load_stabilization_time = self.data_config.get(
            "load_stabilization_time", 10
        )
        self.sampling_seed = self.data_config.get("sampling_seed", 42)

        # Data storage
        self.measurements: List[MeasurementPoint] = []

    def generate_test_points(self) -> List[TestPoint]:
        """
        Generate test points using configured sampling strategy.

        Returns:
            List of test points to measure
        """
        strategy = self.data_config.get("sampling_strategy", "latin_hypercube")
        num_samples = self.data_config.get("num_samples", 50)

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
            samples = sampler.random(n=num_samples)

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
                for _ in range(num_samples // len(load_levels)):
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
        stabilization_time = self.data_config["stabilization_time"]
        measurement_duration = self.data_config["measurement_duration"]
        sample_interval = self.data_config["sample_interval"]

        pwm_str = ", ".join([f"{k}={v}%" for k, v in point.pwm_values.items()])
        print(f"\n{'=' * 80}")
        print(f"Test Point: {point.description}")
        print(f"  Load: CPU {point.cpu_percent}%, GPU {point.gpu_percent}%")
        print(f"  PWM: {pwm_str}")
        print(f"{'=' * 80}")

        # Set load
        print("Setting load...")
        if not self.load_orchestrator.set_workload(
            point.cpu_percent, point.gpu_percent
        ):
            print("✗ Failed to set load")
            return None

        # Wait a bit for load to stabilize
        time.sleep(self.load_stabilization_time)

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

        # Wait for stabilization
        print(f"Waiting {stabilization_time}s for thermal equilibrium...")
        start_stabilization = time.time()

        for i in range(stabilization_time):
            time.sleep(1)

            # Check safety monitor
            try:
                self.safety.check_safety()
            except SkipPointError as e:
                print(f"\n✗ SKIP: {e}")
                print("  Restoring auto fan control and skipping this test point")
                self.safety._apply_emergency_speeds()
                return None
            except Exception as e:
                print(f"\n✗ ABORT: Unexpected error during safety check: {e}")
                raise

            # Get current temperatures for display
            cpu_temp = self.hardware.get_cpu_temp()
            gpu_temp = self.hardware.get_gpu_temp()

            # Print progress
            if (i + 1) % 10 == 0 or i == 0:
                cpu_str = f"{cpu_temp:.1f}°C" if cpu_temp else "N/A"
                gpu_str = f"{gpu_temp:.1f}°C" if gpu_temp else "N/A"
                print(f"  {i + 1:3d}s: CPU {cpu_str}, GPU {gpu_str}")

        actual_stabilization_time = time.time() - start_stabilization

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
            description=point.description,
        )

        print("✓ Measurement complete")
        return measurement

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

        for i, point in enumerate(safe_points):
            print(f"\n[{i + 1}/{len(safe_points)}]")

            measurement = self.collect_measurement(point)

            if measurement:
                self.measurements.append(measurement)
                successful += 1

                # Save incrementally
                self.save_measurements(output_path)
            else:
                aborted += 1

        # Final summary
        print("\n" + "=" * 80)
        print("DATA COLLECTION COMPLETE")
        print("=" * 80)
        print(f"Successful measurements: {successful}")
        print(f"Aborted measurements: {aborted}")
        print(f"Total points collected: {len(self.measurements)}")
        print(f"Data saved to: {output_path}")
        print("=" * 80 + "\n")

    def save_measurements(self, output_path: Path) -> None:
        """
        Save measurements to CSV file.

        Args:
            output_path: Path to output CSV file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        device_keys = list(self.devices.keys())

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=MeasurementPoint.csv_header(device_keys)
            )
            writer.writeheader()

            for measurement in self.measurements:
                writer.writerow(measurement.to_dict())
