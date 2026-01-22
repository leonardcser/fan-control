"""Hardware control for fans and temperature sensors."""

import glob
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import requests


class HardwareError(Exception):
    """Hardware access error."""

    pass


class HardwareController:
    """Control fans and read temperatures."""

    def __init__(
        self,
        hwmon_device_name: str,
        cpu_sensor_name: str,
        cpu_sensor_label: str,
        max_retries: int = 3,
        retry_delay: float = 0.1,
        command_timeout: int = 5,
        ambient_timeout: int = 300,
        ambient_config: Optional[Dict] = None,
    ):
        self.hwmon_path: Optional[str] = None
        self.hwmon_device_name = hwmon_device_name
        self.cpu_sensor_name = cpu_sensor_name
        self.cpu_sensor_label = cpu_sensor_label
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.command_timeout = command_timeout
        self.ambient_timeout = ambient_timeout
        self.ambient_config = ambient_config or {}

        # Ambient cache
        self.last_ambient_temp: Optional[float] = None
        self.last_ambient_time: float = 0

        self._find_hwmon_device()

    def get_ambient_temp(self) -> Optional[float]:
        """Fetch ambient temperature from configured source (e.g. Home Assistant)."""
        if not self.ambient_config:
            return None

        source = self.ambient_config["source"]
        if not source:
            return None

        if source == "home-assistant":
            try:
                url = f"{self.ambient_config['ha_url'].rstrip('/')}/api/states/{self.ambient_config['ha_entity_id']}"
                token = os.environ.get(self.ambient_config["ha_token_env"])

                if not token:
                    print("Error: HA_TOKEN not found in environment", file=sys.stderr)
                    return None

                headers = {
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                }

                # Using a session would be better for repeated calls
                response = requests.get(
                    url, headers=headers, timeout=self.command_timeout
                )
                response.raise_for_status()
                data = response.json()

                temp = float(data["state"])
                self.last_ambient_temp = temp
                self.last_ambient_time = time.time()
                return temp

            except Exception as e:
                print(
                    f"Warning: Failed to fetch ambient temp from HA: {e}",
                    file=sys.stderr,
                )

                # Check for timeout
                if self.last_ambient_temp is not None:
                    if time.time() - self.last_ambient_time > self.ambient_timeout:
                        print(
                            f"Error: Ambient data is stale (>{self.ambient_timeout}s). Stopping.",
                            file=sys.stderr,
                        )
                        return None
                    return self.last_ambient_temp

                return None

        return None

    def _find_hwmon_device(self) -> None:
        """Find the hwmon device by name."""
        for hwmon_path in glob.glob("/sys/class/hwmon/hwmon*"):
            name_file = Path(hwmon_path) / "name"
            try:
                with open(name_file, "r") as f:
                    if f.read().strip() == self.hwmon_device_name:
                        self.hwmon_path = hwmon_path
                        return
            except IOError:
                continue

        raise HardwareError(f"Could not find hwmon device: {self.hwmon_device_name}")

    def get_cpu_temp(self) -> Optional[float]:
        """Get CPU temperature from sensor."""
        try:
            result = subprocess.run(
                ["sensors", self.cpu_sensor_name],
                capture_output=True,
                text=True,
                check=True,
                timeout=self.command_timeout,
            )

            for line in result.stdout.split("\n"):
                if self.cpu_sensor_label in line:
                    match = re.search(r"([+-]?\d+\.?\d*)\s*°C", line)
                    if match:
                        return float(match.group(1))

            return None

        except (
            subprocess.CalledProcessError,
            FileNotFoundError,
            subprocess.TimeoutExpired,
        ):
            return None

    def get_gpu_temp(self) -> Optional[float]:
        """Get GPU temperature from nvidia-smi."""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=self.command_timeout,
            )
            temp_str = result.stdout.strip()
            if temp_str:
                return float(temp_str)
            return None
        except (
            subprocess.CalledProcessError,
            FileNotFoundError,
            ValueError,
            subprocess.TimeoutExpired,
        ):
            return None

    def set_pwm_mode(self, pwm_num: int, mode: int) -> bool:
        """Set PWM control mode (e.g., 1 for manual, 5 for auto)."""
        if not self.hwmon_path:
            raise HardwareError("hwmon device not found")

        pwm_enable_path = Path(self.hwmon_path) / f"pwm{pwm_num}_enable"

        for attempt in range(self.max_retries):
            try:
                # Read current mode
                with open(pwm_enable_path, "r") as f:
                    current_mode = f.read().strip()

                if current_mode != str(mode):
                    with open(pwm_enable_path, "w") as f:
                        f.write(str(mode))
                    time.sleep(0.5)

                    # Verify
                    with open(pwm_enable_path, "r") as f:
                        new_mode = f.read().strip()

                    if new_mode == str(mode):
                        return True
                    else:
                        if attempt < self.max_retries - 1:
                            time.sleep(self.retry_delay * (2**attempt))
                            continue
                        return False
                else:
                    return True

            except IOError as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2**attempt))
                    continue
                print(
                    f"Error setting mode {mode} for pwm{pwm_num}: {e}",
                    file=sys.stderr,
                )
                return False

        return False

    def enable_manual_control(self, pwm_num: int) -> bool:
        """Enable manual PWM control (mode 1)."""
        return self.set_pwm_mode(pwm_num, 1)

    def enable_auto_control(self, pwm_num: int) -> bool:
        """Restore automatic PWM control (mode 5)."""
        return self.set_pwm_mode(pwm_num, 5)

    def set_fan_speed(self, pwm_num: int, percentage: int) -> bool:
        """Set fan speed (0-100 percentage)."""
        if not self.hwmon_path:
            raise HardwareError("hwmon device not found")

        if not 0 <= percentage <= 100:
            raise ValueError(f"Fan speed must be 0-100, got {percentage}")

        pwm_value = int(round(percentage * 255 / 100))
        pwm_path = Path(self.hwmon_path) / f"pwm{pwm_num}"

        for attempt in range(self.max_retries):
            try:
                with open(pwm_path, "w") as f:
                    f.write(str(pwm_value))
                return True

            except IOError as e:
                if e.errno == 16 and attempt < self.max_retries - 1:  # EBUSY
                    delay = self.retry_delay * (2**attempt)
                    time.sleep(delay)
                    continue

                if attempt == self.max_retries - 1:
                    print(
                        f"Error setting PWM{pwm_num} after {attempt + 1} attempts: {e}",
                        file=sys.stderr,
                    )
                    return False

        return False

    def set_all_fans_max(self, fan_pwms: list[int]) -> None:
        """Set all fans to maximum speed."""
        for pwm_num in fan_pwms:
            self.set_fan_speed(pwm_num, 100)

    def get_pwm_value(self, pwm_num: int) -> Optional[int]:
        """Read current PWM value (0-255) for a fan."""
        if not self.hwmon_path:
            raise HardwareError("hwmon device not found")

        pwm_path = Path(self.hwmon_path) / f"pwm{pwm_num}"
        try:
            with open(pwm_path, "r") as f:
                return int(f.read().strip())
        except (IOError, ValueError):
            return None

    def get_cpu_power(self) -> Optional[Dict]:
        """
        Get CPU power draw and performance metrics using turbostat.
        Includes per-core power, frequency, and utilization.
        Requires root permissions.

        Returns:
            Dict with:
                'package': total watts
                'cores': dict mapping core num to dict with:
                    'power': watts
                    'avg_mhz': average frequency
                    'bzy_mhz': busy frequency
                    'busy_pct': busy percentage
            or None if unavailable
        """
        try:
            result = subprocess.run(
                [
                    "turbostat",
                    "--quiet",
                    "--num_iterations",
                    "1",
                    "--interval",
                    "0.1",
                    "--show",
                    "Core,CPU,Avg_MHz,Busy%,Bzy_MHz,CorWatt,PkgWatt",
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=self.command_timeout,
            )

            # Parse output to extract per-core metrics and package power
            # Output format:
            # Core  CPU  Avg_MHz  Busy%  Bzy_MHz  CorWatt  PkgWatt
            # -     -    225      4.49   5009     13.34    60.66     <- summary line
            # 0     0    207      4.62   4482     1.29     60.50     <- individual cores/CPUs
            # 0     12   174      3.28   5290     ...      ...
            lines = result.stdout.strip().split("\n")

            if len(lines) < 3:  # Need header + summary + at least one core
                return None

            # lines[0] is header, lines[1] is summary with package power
            summary_line = lines[1].split()
            if len(summary_line) < 7:
                return None

            try:
                pkg_watt = float(summary_line[6])  # PkgWatt column
            except ValueError:
                return None

            # Parse per-core metrics from remaining lines (skip header and summary)
            # Aggregate by physical core (average frequencies/busy%, sum power from both threads)
            core_metrics = {}
            for line in lines[2:]:
                parts = line.split()
                if len(parts) < 6:
                    continue

                try:
                    core_num = int(parts[0])
                    avg_mhz = float(parts[2])
                    busy_pct = float(parts[3])
                    bzy_mhz = float(parts[4])
                    cor_watt = float(parts[5])

                    if core_num not in core_metrics:
                        core_metrics[core_num] = {
                            'power': 0,
                            'avg_mhz': [],
                            'bzy_mhz': [],
                            'busy_pct': []
                        }

                    # Sum power from both threads
                    core_metrics[core_num]['power'] += cor_watt
                    # Collect frequency/utilization from both threads for averaging
                    core_metrics[core_num]['avg_mhz'].append(avg_mhz)
                    core_metrics[core_num]['bzy_mhz'].append(bzy_mhz)
                    core_metrics[core_num]['busy_pct'].append(busy_pct)

                except (ValueError, IndexError):
                    continue

            # Average frequency/utilization across threads for each core
            core_data = {}
            for core_num, metrics in core_metrics.items():
                core_data[core_num] = {
                    'power': metrics['power'],
                    'avg_mhz': sum(metrics['avg_mhz']) / len(metrics['avg_mhz']),
                    'bzy_mhz': sum(metrics['bzy_mhz']) / len(metrics['bzy_mhz']),
                    'busy_pct': sum(metrics['busy_pct']) / len(metrics['busy_pct'])
                }

            return {
                'package': pkg_watt,
                'cores': core_data
            }

        except (
            subprocess.CalledProcessError,
            FileNotFoundError,
            subprocess.TimeoutExpired,
        ):
            return None

    def get_gpu_power(self) -> Optional[float]:
        """
        Get GPU power draw from nvidia-smi.

        Returns:
            Power in Watts, or None if unavailable
        """
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=power.draw",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=self.command_timeout,
            )

            power_str = result.stdout.strip()
            if power_str:
                # Handle multiple GPUs by taking the first one
                return float(power_str.split("\n")[0])
            return None
        except (
            subprocess.CalledProcessError,
            FileNotFoundError,
            ValueError,
            subprocess.TimeoutExpired,
        ):
            return None

    def get_gpu_fan_speed(self) -> Optional[int]:
        """
        Get GPU fan speed from nvidia-smi.

        Returns:
            Fan speed as percentage (0-100), or None if unavailable
        """
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=fan.speed",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=self.command_timeout,
            )

            speed_str = result.stdout.strip()
            if speed_str:
                # Handle multiple GPUs by taking the first one
                return int(speed_str.split("\n")[0])
            return None
        except (
            subprocess.CalledProcessError,
            FileNotFoundError,
            ValueError,
            subprocess.TimeoutExpired,
        ):
            return None
