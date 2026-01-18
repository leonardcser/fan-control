"""Safety monitoring and emergency shutdown."""

import signal
from typing import Dict

from .hardware import HardwareController
from .load import LoadOrchestrator


class SafetyError(Exception):
    """Fatal safety limit exceeded."""

    pass


class SkipPointError(Exception):
    """Non-fatal temperature limit exceeded, skip test point."""

    pass


class SafetyMonitor:
    """Monitor temperatures and enforce safety limits."""

    def __init__(
        self,
        hardware: HardwareController,
        load_orchestrator: LoadOrchestrator,
        device_pwms: Dict[str, int],
        safety_config: dict,
    ):
        self.hardware = hardware
        self.load_orchestrator = load_orchestrator
        self.device_pwms = device_pwms
        self.emergency_active = False

        # Load safety limits from config
        self.emergency_cpu_temp = safety_config["emergency_cpu_temp"]
        self.emergency_gpu_temp = safety_config["emergency_gpu_temp"]
        self.skip_cpu_temp = safety_config["skip_cpu_temp"]
        self.skip_gpu_temp = safety_config["skip_gpu_temp"]

        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def check_safety(self) -> None:
        """
        Check temperature safety limits.

        Raises:
            SafetyError: If fatal safety limits are exceeded
            SkipPointError: If soft skip limits are exceeded
        """
        current_temps = {
            "cpu": self.hardware.get_cpu_temp(),
            "gpu": self.hardware.get_gpu_temp(),
        }

        # Check emergency (fatal) limits
        if current_temps["cpu"] and current_temps["cpu"] > self.emergency_cpu_temp:
            raise SafetyError(
                f"CPU temperature {current_temps['cpu']:.1f}°C exceeds emergency limit {self.emergency_cpu_temp}°C"
            )

        if current_temps["gpu"] and current_temps["gpu"] > self.emergency_gpu_temp:
            raise SafetyError(
                f"GPU temperature {current_temps['gpu']:.1f}°C exceeds emergency limit {self.emergency_gpu_temp}°C"
            )

        # Check skip (soft) thresholds
        if current_temps["cpu"] and current_temps["cpu"] > self.skip_cpu_temp:
            raise SkipPointError(
                f"CPU temperature {current_temps['cpu']:.1f}°C exceeds skip threshold {self.skip_cpu_temp}°C"
            )

        if current_temps["gpu"] and current_temps["gpu"] > self.skip_gpu_temp:
            raise SkipPointError(
                f"GPU temperature {current_temps['gpu']:.1f}°C exceeds skip threshold {self.skip_gpu_temp}°C"
            )

    def _apply_emergency_speeds(self) -> None:
        """Set fans to auto control (mode 5) for safety."""
        print("Setting all fans to auto control mode...")
        for _, pwm_num in self.device_pwms.items():
            self.hardware.set_pwm_mode(pwm_num, 5)

    def emergency_shutdown(self, reason: str) -> None:
        """Full emergency shutdown."""
        print(f"\n{'=' * 80}")
        print(f"EMERGENCY SHUTDOWN: {reason}")
        print(f"{'=' * 80}\n")

        self.emergency_active = True

        # Set fans to emergency levels (auto mode)
        self._apply_emergency_speeds()

        # Stop all load generation
        print("Stopping all load generation...")
        self.load_orchestrator.stop_all()

    def _signal_handler(self, signum, frame) -> None:
        """Handle interrupt signals."""
        print(f"\n\nReceived signal {signum}, performing emergency shutdown...")
        self.emergency_shutdown(f"Interrupted by signal {signum}")
        import sys

        sys.exit(1)
