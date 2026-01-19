import time
import logging
import signal
import sys
from pathlib import Path
from typing import Dict, Optional
import yaml

from ..hardware import HardwareController
from .optimizer import Optimizer

logger = logging.getLogger(__name__)


class FanController:
    """
    Main control loop.
    Reads sensors -> Optimizes -> Sets fans.
    """

    def __init__(self, config_path: Path, model_path: Path):
        self.running = False

        # Load Config
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        # Initialize Hardware
        hw_cfg = self.config["hardware"]
        self.hw = HardwareController(
            hwmon_device_name=hw_cfg["hwmon_device_name"],
            cpu_sensor_name=hw_cfg["cpu_sensor_name"],
            cpu_sensor_label=hw_cfg["cpu_sensor_label"],
            ambient_config=self.config.get("ambient"),
        )

        # Initialize Optimizer
        self.optimizer = Optimizer(model_path, self.config)

        # Default Targets (can be overridden)
        self.targets = {
            "T_cpu": 75.0,  # Target CPU Temp
            "T_gpu": 70.0,  # Target GPU Temp
        }

        # Loop settings
        self.interval = 5.0  # Seconds

        # State tracking for optimization continuity
        self.last_pwms = {name: 50.0 for name in self.optimizer.pwm_names}

    def start(self):
        """Start the control loop."""
        self.running = True

        # Enable manual mode for all controlled fans
        for pwm_name in self.optimizer.pwm_names:
            dev_cfg = self.config["devices"][pwm_name]
            pwm_num = dev_cfg["pwm_number"]
            logger.info(f"Enabling manual control for {pwm_name} (pwm{pwm_num})")
            self.hw.enable_manual_control(pwm_num)

        logger.info(f"Starting control loop (Interval: {self.interval}s)")
        logger.info(f"Targets: {self.targets}")

        # Handle graceful shutdown
        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

        try:
            while self.running:
                self._tick()
                time.sleep(self.interval)
        except Exception as e:
            logger.error(f"Control loop crashed: {e}")
            self._emergency_mode()
            raise

    def _tick(self):
        """Single control iteration."""
        # 1. Read Sensors
        t_start = time.time()

        state = self._read_sensors()
        if not state:
            logger.warning("Sensor read failed. Skipping tick.")
            return

        # 2. Check Safety
        if self._check_safety(state):
            return  # Safety triggered, skip optimization

        # 3. Optimize
        optimal_pwms = self.optimizer.optimize(
            state, self.targets, initial_guess=self.last_pwms
        )

        # Update state
        self.last_pwms = {k: float(v) for k, v in optimal_pwms.items()}

        # 4. Apply Fan Speeds
        self._apply_speeds(optimal_pwms)

        duration = time.time() - t_start
        logger.info(
            f"Tick: CPU={state['T_cpu']}°C, GPU={state['T_gpu']}°C | Opt: {optimal_pwms} | Time: {duration:.2f}s"
        )

    def _read_sensors(self) -> Optional[Dict[str, float]]:
        """Read current system state."""
        try:
            return {
                "P_cpu": self.hw.get_cpu_power() or 0.0,
                "P_gpu": self.hw.get_gpu_power() or 0.0,
                "T_amb": self.hw.get_ambient_temp() or 25.0,  # Fallback to 25C
                "T_cpu": self.hw.get_cpu_temp(),
                "T_gpu": self.hw.get_gpu_temp(),
                # We need current PWMs? Not strictly for the optimizer,
                # but good for logging. The optimizer builds its own DataFrame.
            }
        except Exception as e:
            logger.error(f"Sensor read error: {e}")
            return None

    def _check_safety(self, state: Dict) -> bool:
        """
        Check for overheating.
        Returns True if safety limits exceeded (and emergency fans applied).
        """
        safety_cfg = self.config["safety"]
        triggered = False

        if state["T_cpu"] > safety_cfg["emergency_cpu_temp"]:
            logger.critical(
                f"EMERGENCY: CPU Temp {state['T_cpu']}°C > Limit {safety_cfg['emergency_cpu_temp']}°C"
            )
            triggered = True

        if state["T_gpu"] > safety_cfg["emergency_gpu_temp"]:
            logger.critical(
                f"EMERGENCY: GPU Temp {state['T_gpu']}°C > Limit {safety_cfg['emergency_gpu_temp']}°C"
            )
            triggered = True

        if triggered:
            self._emergency_mode()
            return True

        return False

    def _emergency_mode(self):
        """Set all fans to 100%."""
        logger.warning("Applying EMERGENCY FAN SPEEDS (100%)")
        pwm_nums = [
            self.config["devices"][name]["pwm_number"]
            for name in self.optimizer.pwm_names
        ]
        self.hw.set_all_fans_max(pwm_nums)

    def _apply_speeds(self, pwms: Dict[str, int]):
        """Set hardware PWM values."""
        for name, value in pwms.items():
            pwm_num = self.config["devices"][name]["pwm_number"]
            self.hw.set_fan_speed(pwm_num, value)

    def _shutdown(self, signum, frame):
        """Graceful shutdown handler."""
        logger.info("Shutdown signal received. Stopping...")
        self.running = False
        # Optional: Set fans to safe defaults on exit?
        # For now, let's leave them as-is or set to a safe 50%?
        # Better safe:
        self._emergency_mode()
        sys.exit(0)
