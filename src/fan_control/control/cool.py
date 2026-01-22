import sys
import time
import logging
import yaml
from pathlib import Path

from ..core.hardware import HardwareController

logger = logging.getLogger(__name__)


def cool_mode(args) -> None:
    """
    Set all fans to max speed until interrupted, then restore auto control.
    """
    print("\n" + "=" * 70)
    print("FAN CONTROL - COOL DOWN MODE")
    print("=" * 70 + "\n")

    config_path = Path(args.config)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Initialize Hardware
    hw_cfg = config["hardware"]
    hw = HardwareController(
        hwmon_device_name=hw_cfg["hwmon_device_name"],
        cpu_sensor_name=hw_cfg["cpu_sensor_name"],
        cpu_sensor_label=hw_cfg["cpu_sensor_label"],
        ambient_config=config["ambient"],
    )

    pwm_nums = [dev["pwm_number"] for dev in config["devices"].values()]

    print(f"Config: {config_path}")
    print(f"Controlled PWMs: {pwm_nums}")
    print(
        "\nSetting all fans to 100%... (Press Ctrl+C to stop and restore auto control)"
    )

    try:
        # Enable manual control and set to 100%
        for pwm_num in pwm_nums:
            logger.info(f"Setting pwm{pwm_num} to manual mode and 100% speed")
            hw.enable_manual_control(pwm_num)
            hw.set_fan_speed(pwm_num, 100)

        while True:
            # Read temps just to show status
            cpu_temp = hw.get_cpu_temp()
            gpu_temp = hw.get_gpu_temp()
            cpu_str = f"{cpu_temp}°C" if cpu_temp is not None else "N/A"
            gpu_str = f"{gpu_temp}°C" if gpu_temp is not None else "N/A"
            print(
                f"\rStatus: CPU={cpu_str}, GPU={gpu_str} | All fans at 100%   ",
                end="",
                flush=True,
            )
            time.sleep(2)

    except KeyboardInterrupt:
        print("\n\nRestoring automatic control...")
        for pwm_num in pwm_nums:
            logger.info(f"Setting pwm{pwm_num} to auto mode (mode 5)")
            hw.set_pwm_mode(pwm_num, 5)
        print("Done. Goodbye!")
    except Exception as e:
        logger.error(f"Error in cool mode: {e}")
        sys.exit(1)
