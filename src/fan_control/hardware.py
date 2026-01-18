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
        source = self.ambient_config.get("source")
        if not source:
            return None

        if source == "home-assistant":
            try:
                url = f"{self.ambient_config['ha_url'].rstrip('/')}/api/states/{self.ambient_config['ha_entity_id']}"
                token = os.environ.get(
                    self.ambient_config.get("ha_token_env", "HA_TOKEN")
                )

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

    def get_cpu_power(self) -> Optional[float]:
        """
        Get CPU power draw using turbostat.
        Requires root permissions.

        Returns:
            Power in Watts, or None if unavailable
        """
        try:
            result = subprocess.run(
                [
                    "turbostat",
                    "--Summary",
                    "--num_iterations",
                    "1",
                    "--interval",
                    "0.1",
                    "--show",
                    "PkgWatt",
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=self.command_timeout,
            )

            # turbostat output contains headers and then the value
            # We look for the line after the header
            lines = result.stdout.strip().split("\n")
            if len(lines) >= 2:
                # The last line should be the value if we used --num_iterations 1
                try:
                    return float(lines[-1])
                except ValueError:
                    pass

            return None

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
