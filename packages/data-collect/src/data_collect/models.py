"""Data models for time-series thermal data collection."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TimeSeriesSample:
    """Single 1-second sample for MPC model training."""

    timestamp: float  # Unix timestamp
    episode_id: str  # Groups samples into sequences
    sample_index: int  # Index within episode (0, 1, 2, ...)

    # State (temperatures)
    T_cpu: float
    T_gpu: float

    # Control inputs (PWM 0-255)
    pwm2: int
    pwm4: int
    pwm5: int

    # Disturbances - aggregate power
    P_cpu: float
    P_gpu: float
    T_amb: Optional[float]

    # Per-core CPU metrics (all kept for detailed modeling)
    P_cpu_cores: Optional[dict[int, float]] = None  # Per-core power (watts)
    cpu_avg_mhz: Optional[dict[int, float]] = None  # Per-core avg frequency
    cpu_bzy_mhz: Optional[dict[int, float]] = None  # Per-core busy frequency
    cpu_busy_pct: Optional[dict[int, float]] = None  # Per-core busy percentage

    # GPU metrics
    gpu_fan_speed: Optional[int] = None

    # Metadata
    load_description: str = ""
    cpu_load_flags: str = ""
    gpu_load_flags: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for CSV export."""
        d = {
            "timestamp": self.timestamp,
            "episode_id": self.episode_id,
            "sample_index": self.sample_index,
            "T_cpu": self.T_cpu,
            "T_gpu": self.T_gpu,
            "pwm2": self.pwm2,
            "pwm4": self.pwm4,
            "pwm5": self.pwm5,
            "P_cpu": self.P_cpu,
            "P_gpu": self.P_gpu,
            "T_amb": self.T_amb,
        }

        # Add per-core power columns
        if self.P_cpu_cores:
            for core_num, power in self.P_cpu_cores.items():
                d[f"P_cpu_core{core_num}"] = power

        # Add per-core frequency columns
        if self.cpu_avg_mhz:
            for core_num, freq in self.cpu_avg_mhz.items():
                d[f"cpu_avg_mhz_core{core_num}"] = freq

        if self.cpu_bzy_mhz:
            for core_num, freq in self.cpu_bzy_mhz.items():
                d[f"cpu_bzy_mhz_core{core_num}"] = freq

        if self.cpu_busy_pct:
            for core_num, pct in self.cpu_busy_pct.items():
                d[f"cpu_busy_pct_core{core_num}"] = pct

        d["gpu_fan_speed"] = self.gpu_fan_speed
        d["load_description"] = self.load_description
        d["cpu_load_flags"] = self.cpu_load_flags
        d["gpu_load_flags"] = self.gpu_load_flags

        return d

    @staticmethod
    def csv_header(num_cores: int = 12) -> list[str]:
        """Get CSV header columns.

        Args:
            num_cores: Number of CPU cores for per-core columns (default: 12)
        """
        # Generate per-core column names
        core_power_cols = [f"P_cpu_core{i}" for i in range(num_cores)]
        core_avg_mhz_cols = [f"cpu_avg_mhz_core{i}" for i in range(num_cores)]
        core_bzy_mhz_cols = [f"cpu_bzy_mhz_core{i}" for i in range(num_cores)]
        core_busy_pct_cols = [f"cpu_busy_pct_core{i}" for i in range(num_cores)]

        return (
            [
                "timestamp",
                "episode_id",
                "sample_index",
                "T_cpu",
                "T_gpu",
                "pwm2",
                "pwm4",
                "pwm5",
                "P_cpu",
                "P_gpu",
                "T_amb",
            ]
            + core_power_cols
            + core_avg_mhz_cols
            + core_bzy_mhz_cols
            + core_busy_pct_cols
            + [
                "gpu_fan_speed",
                "load_description",
                "cpu_load_flags",
                "gpu_load_flags",
            ]
        )


@dataclass
class Episode:
    """Metadata for a collection episode."""

    episode_id: str
    start_timestamp: float
    end_timestamp: float
    load_description: str
    num_samples: int
    aborted: bool = False
    abort_reason: Optional[str] = None


@dataclass
class PWMStep:
    """A planned PWM change in the schedule."""

    time_offset: float  # Seconds from episode start
    pwm_values: dict[str, int] = field(default_factory=dict)  # Target PWM values (0-255)


