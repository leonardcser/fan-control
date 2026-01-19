"""Command-line interface for thermal data collection."""

import os
import sys
import shutil
from datetime import datetime
from pathlib import Path

import yaml

from .collector import DataCollector
from ..hardware import HardwareController
from ..load import LoadOrchestrator
from ..safety import SafetyMonitor, SafetyError
from ..utils import drop_privileges


def check_prerequisites() -> bool:
    """Check if all prerequisites are met."""
    checks_passed = True

    print("Running pre-flight checks...\n")

    # Check root
    if os.geteuid() != 0:
        print("✗ Must run as root (use sudo)")
        checks_passed = False
    else:
        print("✓ Running as root")

    # Check required tools
    required_tools = {
        "stress": "CPU stress testing",
        "gpu-burn": "GPU stress testing",
        "sensors": "Temperature monitoring",
        "nvidia-smi": "GPU monitoring",
        "turbostat": "CPU power monitoring",
    }

    for tool, purpose in required_tools.items():
        if shutil.which(tool):
            print(f"✓ {tool} found ({purpose})")
        else:
            print(f"✗ {tool} not found ({purpose})")
            checks_passed = False

    print()
    return checks_passed


def collect_mode(args) -> None:
    """Data collection mode - collect thermal measurements."""
    print("\n" + "=" * 80)
    print("THERMAL DATA COLLECTION")
    print("=" * 80 + "\n")

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"✗ Config file not found: {config_path}")
        sys.exit(1)

    try:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f"✗ Error parsing config file: {e}")
        sys.exit(1)

    # Override output directory if specified
    if hasattr(args, "output") and args.output:
        cfg["output"]["directory"] = args.output

    # Pre-flight checks
    if not check_prerequisites():
        print("\nPre-flight checks failed. Please resolve issues before proceeding.")
        sys.exit(1)

    print("✓ All pre-flight checks passed\n")

    # Initialize hardware
    print("Initializing hardware...")
    try:
        hw_cfg = cfg.get("hardware", {})
        ambient_cfg = cfg.get("ambient", {})

        # Validate HA Token if needed
        if ambient_cfg.get("source") == "home-assistant":
            token_env = ambient_cfg.get("ha_token_env", "HA_TOKEN")
            if not os.environ.get(token_env):
                print(
                    f"✗ Error: Ambient mode enabled but {token_env} is not set in environment"
                )
                sys.exit(1)
            print(
                f"✓ Ambient correction enabled via Home Assistant ({ambient_cfg['ha_entity_id']})"
            )

        hardware = HardwareController(
            hwmon_device_name=hw_cfg["hwmon_device_name"],
            cpu_sensor_name=hw_cfg["cpu_sensor_name"],
            cpu_sensor_label=hw_cfg["cpu_sensor_label"],
            max_retries=hw_cfg.get("max_retries", 3),
            retry_delay=hw_cfg.get("retry_delay", 0.1),
            command_timeout=hw_cfg.get("command_timeout", 5),
            ambient_timeout=hw_cfg.get("ambient_timeout", 300),
            ambient_config=ambient_cfg,
        )
        print(f"✓ Found hwmon device at: {hardware.hwmon_path}\n")
    except Exception as e:
        print(f"✗ Hardware initialization failed: {e}")
        sys.exit(1)

    # Initialize load orchestrator
    load_orchestrator = LoadOrchestrator(
        load_stabilization_time=cfg.get("data_collection", {}).get(
            "load_stabilization_time", 10
        ),
    )

    # Get device PWM mappings for safety monitor
    device_pwms = {
        dev_id: dev_cfg["pwm_number"] for dev_id, dev_cfg in cfg["devices"].items()
    }

    # Initialize safety monitor
    safety = SafetyMonitor(
        hardware,
        load_orchestrator,
        device_pwms,
        cfg["safety"],
    )

    try:
        # Enable manual control for all devices
        print("\nEnabling manual control...")
        for dev_cfg in cfg["devices"].values():
            if not hardware.enable_manual_control(dev_cfg["pwm_number"]):
                print(f"✗ Failed to enable manual control for {dev_cfg['name']}")
                raise Exception("Failed to enable manual control")
            print(f"✓ Manual control enabled for {dev_cfg['name']}")

        # Initialize data collector
        collector = DataCollector(
            hardware,
            load_orchestrator,
            safety,
            cfg,
        )

        # Create output path (drop privileges for file creation)
        output_dir = Path(cfg["output"]["directory"])
        with drop_privileges():
            output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"fan_control_{timestamp}"
        run_dir = output_dir / run_name

        with drop_privileges():
            run_dir.mkdir(parents=True, exist_ok=True)

        filename = "data.csv"
        output_path = run_dir / filename

        print(f"\nData will be saved to: {output_path}")

        # Run data collection
        collector.run_collection(output_path)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        safety.emergency_shutdown("User interrupt")
        sys.exit(1)

    except SafetyError as e:
        print(f"\n\nSAFETY HALT: {e}")
        safety.emergency_shutdown(str(e))
        sys.exit(1)

    except Exception as e:
        print(f"\n\nError during data collection: {e}")
        import traceback

        traceback.print_exc()
        safety.emergency_shutdown(str(e))
        sys.exit(1)

    finally:
        # Cleanup
        print("\nCleaning up...")
        load_orchestrator.stop_all()
        # Restore auto control as a safe default after collection
        print("Restoring auto fan control...")
        for pwm_num in device_pwms.values():
            hardware.set_pwm_mode(pwm_num, 5)
