"""Data models for thermal data collection."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class MeasurementPoint:
    """
    Single steady-state measurement point for thermal model fitting.

    Corresponds to the data collection schema in PHYSICS_MODEL.md.
    Each row represents one steady-state measurement after thermal equilibrium.
    """

    timestamp: float  # Unix timestamp

    # Control variables (PWM values: 0-255)
    pwm_values: dict[str, int]  # Map: {"pwm2": 128, "pwm4": 100, ...}

    # Power measurements (Watts)
    P_cpu: float  # CPU power draw
    P_gpu: float  # GPU power draw

    # Temperature measurements (°C)
    T_amb: Optional[float]  # Ambient temperature
    T_cpu: float  # CPU temperature
    T_gpu: float  # GPU temperature

    # Load settings (flags)
    cpu_load_flags: str  # Stress flags (e.g. "--cpu 6")
    gpu_load_flags: str  # gpu_load.py flags (e.g. "--load 50 --memory 30")
    cpu_cores: int  # Number of CPU cores under load (parsed from cpu_load_flags)

    # Metadata
    stabilization_time: float  # Time waited for equilibrium (seconds)

    # Equilibration tracking
    equilibrated: bool = True  # Was equilibrium reached?
    equilibration_reason: Optional[str] = (
        None  # "equilibrated", "timeout_after_120s", etc.
    )

    description: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for CSV export."""
        d = {
            "timestamp": self.timestamp,
            "P_cpu": self.P_cpu,
            "P_gpu": self.P_gpu,
            "T_amb": self.T_amb,
            "T_cpu": self.T_cpu,
            "T_gpu": self.T_gpu,
            "cpu_load_flags": self.cpu_load_flags,
            "gpu_load_flags": self.gpu_load_flags,
            "cpu_cores": self.cpu_cores,
            "stabilization_time": self.stabilization_time,
            "equilibrated": self.equilibrated,
            "equilibration_reason": self.equilibration_reason,
            "description": self.description,
        }
        d.update(self.pwm_values)
        return d

    @staticmethod
    def csv_header(device_keys: list[str]) -> list[str]:
        """Get CSV header columns."""
        return (
            [
                "timestamp",
            ]
            + device_keys
            + [
                "P_cpu",
                "P_gpu",
                "T_amb",
                "T_cpu",
                "T_gpu",
                "cpu_load_flags",
                "gpu_load_flags",
                "cpu_cores",
                "stabilization_time",
                "equilibrated",
                "equilibration_reason",
                "description",
            ]
        )


@dataclass
class TestPoint:
    """A test configuration to measure."""

    pwm_values: dict[str, int]  # 0-100 percentage
    cpu_load_flags: str  # stress flags
    gpu_load_flags: str  # gpu_load.py flags
    cpu_cores: int = 0  # Number of CPU cores under load
    description: str = ""


@dataclass
class SafetyCheck:
    """Result of a safety check."""

    safe: bool
    reason: str = ""
    cpu_temp: Optional[float] = None
    gpu_temp: Optional[float] = None
    cpu_power: Optional[float] = None
    gpu_power: Optional[float] = None
