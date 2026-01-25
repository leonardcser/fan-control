"""Time-series data collection for MPC thermal model training."""

import csv
import time
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

from .hardware import HardwareController
from .load import LoadOrchestrator
from .models import Episode, EpisodeType, PWMStep, TimeSeriesSample
from .safety import AbortPointError, SafetyMonitor
from .utils import drop_privileges


class PWMScheduleGenerator:
    """Generates PWM step sequences for episodes."""

    def __init__(self, devices: dict, schedule_config: dict, data_config: dict):
        """Initialize schedule generator.

        Args:
            devices: Device configuration dict from config.yaml
            schedule_config: pwm_schedule config (min_hold_time, max_hold_time)
            data_config: data_collection config (contains pwmX_levels)
        """
        self.devices = devices
        self.min_hold = schedule_config["min_hold_time"]
        self.max_hold = schedule_config["max_hold_time"]

        # Store discrete PWM levels per device (converted to 0-255 scale)
        self.pwm_levels: dict[str, list[int]] = {}
        self.pwm_min: dict[str, int] = {}  # Minimum PWM for isolation modes
        for device_key in devices.keys():
            levels = data_config[f"{device_key}_levels"]
            # Convert percentage (0-100) to PWM (0-255)
            pwm_values = [int(round(lvl * 255 / 100)) for lvl in levels]
            self.pwm_levels[device_key] = pwm_values
            self.pwm_min[device_key] = min(pwm_values)

    def _sample_pwm(
        self,
        rng: np.random.Generator,
        varying_fans: list[str],
        fixed_mode: Optional[str] = None,
    ) -> dict[str, int]:
        """Generate PWM values by sampling from discrete levels.

        Args:
            rng: Random number generator
            varying_fans: List of device keys that should vary
            fixed_mode: How to set non-varying fans ("minimum" or None for random)

        Returns:
            Dict mapping device keys to PWM values (0-255)
        """
        pwm_values = {}
        for device_key, levels in self.pwm_levels.items():
            if device_key in varying_fans:
                # Sample from discrete levels
                pwm = int(rng.choice(levels))
            elif fixed_mode == "minimum":
                # Fixed fans use minimum value
                pwm = self.pwm_min[device_key]
            else:
                # Random selection for non-isolation modes
                pwm = int(rng.choice(levels))

            # Apply stall threshold check
            pwm = self._apply_stall_check(device_key, pwm)
            pwm_values[device_key] = pwm
        return pwm_values

    def _apply_stall_check(self, device_key: str, pwm_val: int) -> int:
        """Apply stall threshold check to a PWM value.

        Args:
            device_key: Device key (e.g., 'pwm2')
            pwm_val: PWM value to check

        Returns:
            Adjusted PWM value (0 if below stall threshold)
        """
        stall_pwm = int(round(self.devices[device_key]["stall_pwm"] * 255 / 100))
        if 0 < pwm_val < stall_pwm:
            return 0
        return pwm_val

    def _step_change(
        self,
        current_pwm: dict[str, int],
        rng: np.random.Generator,
        varying_fans: list[str],
    ) -> dict[str, int]:
        """Generate a step change from current PWM (modify exactly 1 fan).

        Args:
            current_pwm: Current PWM values
            rng: Random number generator
            varying_fans: List of device keys that can be changed

        Returns:
            New PWM values with exactly one fan changed
        """
        new_pwm = current_pwm.copy()

        # Always change exactly 1 fan from the varying set
        fan_to_change = str(rng.choice(varying_fans))
        levels = self.pwm_levels[fan_to_change]
        current_val = current_pwm[fan_to_change]

        # Pre-compute effective levels after stall check
        effective_levels = [self._apply_stall_check(fan_to_change, lvl) for lvl in levels]
        # Deduplicate while preserving order
        seen = set()
        unique_effective = []
        for lvl in effective_levels:
            if lvl not in seen:
                seen.add(lvl)
                unique_effective.append(lvl)

        # Filter to levels different from current value
        other_levels = [lvl for lvl in unique_effective if lvl != current_val]
        if not other_levels:
            # All levels same as current (edge case)
            other_levels = unique_effective

        new_val = int(rng.choice(other_levels))
        new_pwm[fan_to_change] = new_val

        return new_pwm

    def generate(
        self,
        episode_duration: float,
        seed: int,
        mode: str,
        varying_fans: list[str],
    ) -> list[PWMStep]:
        """Generate random step schedule for an episode.

        Args:
            episode_duration: Episode length in seconds
            seed: Random seed for reproducibility
            mode: Episode mode ("single_isolation", "pair_isolation", "full_sequential")
            varying_fans: List of device keys that should vary during episode

        Returns:
            List of PWMStep objects with time offsets and PWM values
        """
        rng = np.random.default_rng(seed)
        steps = []
        t = 0.0

        # Determine fixed_mode based on episode type
        fixed_mode = "minimum" if mode in ("single_isolation", "pair_isolation") else None

        # Initial PWM values
        current_pwm = self._sample_pwm(rng, varying_fans, fixed_mode)
        steps.append(PWMStep(time_offset=0.0, pwm_values=current_pwm))

        while t < episode_duration:
            hold_time = rng.uniform(self.min_hold, self.max_hold)
            t += hold_time
            if t >= episode_duration:
                break

            # Step change: modify exactly 1 fan from varying set
            new_pwm = self._step_change(current_pwm, rng, varying_fans)
            steps.append(PWMStep(time_offset=t, pwm_values=new_pwm))
            current_pwm = new_pwm

        return steps


class EpisodeCollector:
    """Collect time-series episodes for MPC model training."""

    def __init__(
        self,
        hardware: HardwareController,
        load_orchestrator: LoadOrchestrator,
        safety: SafetyMonitor,
        config: dict,
    ):
        self.hardware = hardware
        self.load_orchestrator = load_orchestrator
        self.safety = safety
        self.config = config

        # Extract configuration
        self.data_config = config["data_collection"]
        self.devices = config["devices"]

        # Time-series parameters
        self.sample_interval = self.data_config["sample_interval"]
        self.episode_duration = self.data_config["episode_duration"]
        self.sampling_seed = self.data_config["sampling_seed"]

        # Parse episode types
        self.episode_types: list[EpisodeType] = [
            EpisodeType(mode=et["mode"], repeats_per_load=et["repeats_per_load"])
            for et in self.data_config["episode_types"]
        ]

        # Device keys for generating fan combinations
        self.device_keys = list(self.devices.keys())

        # PWM schedule generator
        self.schedule_generator = PWMScheduleGenerator(
            self.devices,
            self.data_config["pwm_schedule"],
            self.data_config,
        )

        # Data storage
        self.samples: list[TimeSeriesSample] = []
        self.episodes: list[Episode] = []

        # Episode counter for seed variation
        self._episode_counter = 0

    def _apply_pwm(self, pwm_values: dict[str, int]) -> bool:
        """Apply PWM values to all devices.

        Args:
            pwm_values: Dict mapping device keys to PWM values (0-255)

        Returns:
            True if all succeeded, False otherwise
        """
        for device_key, pwm_value in pwm_values.items():
            pwm_num = self.devices[device_key]["pwm_number"]
            # Convert 0-255 to percentage for hardware API
            pct = int(round(pwm_value * 100 / 255))
            if not self.hardware.set_fan_speed(pwm_num, pct):
                tqdm.write(f"Failed to set {device_key} (PWM{pwm_num}) to {pct}%")
                return False
        return True

    def _collect_sample(
        self,
        episode_id: str,
        sample_index: int,
        pwm_values: dict[str, int],
        load_config: dict,
    ) -> TimeSeriesSample:
        """Collect a single instantaneous sample.

        Args:
            episode_id: Episode identifier
            sample_index: Index within episode
            pwm_values: Current PWM values (0-255)
            load_config: Load configuration dict

        Returns:
            TimeSeriesSample with all measurements
        """
        timestamp = time.time()

        # Read temperatures
        T_cpu = self.hardware.get_cpu_temp()
        T_gpu = self.hardware.get_gpu_temp()

        # Read power and per-core metrics
        cpu_power_data = self.hardware.get_cpu_power()
        P_cpu = cpu_power_data["package"] if cpu_power_data else 0.0
        P_cpu_cores = None
        cpu_avg_mhz = None
        cpu_bzy_mhz = None
        cpu_busy_pct = None

        if cpu_power_data and cpu_power_data.get("cores"):
            cores_data = cpu_power_data["cores"]
            P_cpu_cores = {k: v["power"] for k, v in cores_data.items()}
            cpu_avg_mhz = {k: v["avg_mhz"] for k, v in cores_data.items()}
            cpu_bzy_mhz = {k: v["bzy_mhz"] for k, v in cores_data.items()}
            cpu_busy_pct = {k: v["busy_pct"] for k, v in cores_data.items()}

        P_gpu = self.hardware.get_gpu_power() or 0.0
        T_amb = self.hardware.get_ambient_temp()
        gpu_fan_speed = self.hardware.get_gpu_fan_speed()

        return TimeSeriesSample(
            timestamp=timestamp,
            episode_id=episode_id,
            sample_index=sample_index,
            T_cpu=T_cpu or 0.0,
            T_gpu=T_gpu or 0.0,
            pwm2=pwm_values.get("pwm2", 0),
            pwm4=pwm_values.get("pwm4", 0),
            pwm5=pwm_values.get("pwm5", 0),
            P_cpu=P_cpu,
            P_gpu=P_gpu,
            T_amb=T_amb,
            P_cpu_cores=P_cpu_cores,
            cpu_avg_mhz=cpu_avg_mhz,
            cpu_bzy_mhz=cpu_bzy_mhz,
            cpu_busy_pct=cpu_busy_pct,
            gpu_fan_speed=gpu_fan_speed,
            load_description=load_config["description"],
            cpu_load_flags=load_config["cpu_load"],
            gpu_load_flags=load_config["gpu_load"],
        )

    def run_episode(
        self,
        load_config: dict,
        episode_id: str,
        seed: int,
        mode: str,
        varying_fans: list[str],
    ) -> tuple[Episode, list[TimeSeriesSample]]:
        """Run single episode: continuous 1Hz sampling with PWM steps.

        No equilibration - captures transient dynamics.

        Args:
            load_config: Load configuration dict
            episode_id: Unique episode identifier
            seed: Random seed for PWM schedule generation
            mode: Episode mode for PWM variation strategy
            varying_fans: List of device keys that vary during this episode

        Returns:
            Tuple of (Episode metadata, list of samples)
        """
        schedule = self.schedule_generator.generate(
            self.episode_duration, seed, mode, varying_fans
        )
        samples: list[TimeSeriesSample] = []
        start_time = time.time()
        current_step_idx = 0
        sample_index = 0
        aborted = False
        abort_reason: Optional[str] = None

        # Apply initial PWM
        self._apply_pwm(schedule[0].pwm_values)

        # Progress bar for episode
        expected_samples = int(self.episode_duration / self.sample_interval)
        with tqdm(
            total=expected_samples,
            desc=f"Episode {episode_id[:20]}",
            unit="sample",
            leave=False,
            bar_format="{desc}: {percentage:3.0f}% |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        ) as pbar:
            while (elapsed := time.time() - start_time) < self.episode_duration:
                loop_start = time.time()

                # Check for scheduled PWM step
                if (
                    current_step_idx + 1 < len(schedule)
                    and elapsed >= schedule[current_step_idx + 1].time_offset
                ):
                    current_step_idx += 1
                    self._apply_pwm(schedule[current_step_idx].pwm_values)
                    pwm_str = ", ".join(
                        f"{k}={v}"
                        for k, v in schedule[current_step_idx].pwm_values.items()
                    )
                    tqdm.write(f"  PWM step at {elapsed:.1f}s: {pwm_str}")

                # Safety check
                try:
                    self.safety.check_safety()
                except AbortPointError as e:
                    tqdm.write(f"\nABORT: {e}")
                    tqdm.write(
                        f"  Setting fans to full speed and cooling for {self.safety.abort_cooldown_time}s..."
                    )
                    self.safety._apply_abort_speeds()
                    time.sleep(self.safety.abort_cooldown_time)
                    aborted = True
                    abort_reason = str(e)
                    break

                # Sample immediately (no waiting!)
                sample = self._collect_sample(
                    episode_id,
                    sample_index,
                    schedule[current_step_idx].pwm_values,
                    load_config,
                )
                samples.append(sample)
                sample_index += 1
                pbar.update(1)

                # Update status
                pbar.set_postfix_str(
                    f"CPU:{sample.T_cpu:.1f}C GPU:{sample.T_gpu:.1f}C"
                )

                # Maintain sample rate
                sleep_time = max(0, self.sample_interval - (time.time() - loop_start))
                time.sleep(sleep_time)

        end_time = time.time()

        episode = Episode(
            episode_id=episode_id,
            start_timestamp=start_time,
            end_timestamp=end_time,
            load_description=load_config["description"],
            num_samples=len(samples),
            aborted=aborted,
            abort_reason=abort_reason,
        )

        return episode, samples

    def _get_fan_combinations(self, mode: str) -> list[list[str]]:
        """Get fan combinations for an episode type.

        Args:
            mode: Episode mode

        Returns:
            List of fan combinations (each is a list of device keys)
        """
        if mode == "single_isolation":
            # One fan at a time: [[pwm2], [pwm4], [pwm5]]
            return [[dk] for dk in self.device_keys]
        elif mode == "pair_isolation":
            # Pairs of fans: [[pwm2, pwm4], [pwm2, pwm5], [pwm4, pwm5]]
            from itertools import combinations

            return [list(combo) for combo in combinations(self.device_keys, 2)]
        else:  # full_sequential
            # All fans vary together
            return [self.device_keys]

    def _calculate_total_episodes(self, num_loads: int) -> int:
        """Calculate total episodes across all types and loads."""
        total = 0
        for et in self.episode_types:
            combinations = self._get_fan_combinations(et.mode)
            total += len(combinations) * et.repeats_per_load * num_loads
        return total

    def run_collection(self, output_path: Path) -> None:
        """Run full campaign: iterate over episode types and load levels.

        Args:
            output_path: Path to output CSV file
        """
        load_levels = self.data_config["load_levels"]
        total_episodes = self._calculate_total_episodes(len(load_levels))

        # Calculate episodes per load for display
        episodes_per_load = sum(
            len(self._get_fan_combinations(et.mode)) * et.repeats_per_load
            for et in self.episode_types
        )

        print("\n" + "=" * 80)
        print("STARTING TIME-SERIES DATA COLLECTION")
        print("=" * 80)
        print(f"Episode duration: {self.episode_duration}s")
        print(f"Episode types: {len(self.episode_types)}")
        for et in self.episode_types:
            combos = len(self._get_fan_combinations(et.mode))
            print(f"  - {et.mode}: {et.repeats_per_load} repeats × {combos} combinations")
        print(f"Episodes per load: {episodes_per_load}")
        print(f"Load levels: {len(load_levels)}")
        print(f"Total episodes: {total_episodes}")
        print(f"Sample interval: {self.sample_interval}s")
        print(f"Expected samples per episode: ~{int(self.episode_duration / self.sample_interval)}")

        with tqdm(
            total=total_episodes,
            desc="Total Progress",
            unit="ep",
            bar_format="{desc}: {percentage:3.0f}% |{bar}| {n_fmt}/{total_fmt} episodes [{elapsed}<{remaining}]",
        ) as main_pbar:
            # Process each load level
            for load_idx, load in enumerate(load_levels, 1):
                description = load["description"]

                tqdm.write(f"\n{'=' * 80}")
                tqdm.write(
                    f"Load Level {load_idx}/{len(load_levels)}: {description}"
                )
                tqdm.write(f"  CPU: '{load['cpu_load']}' | GPU: '{load['gpu_load']}'")
                tqdm.write(f"{'=' * 80}")

                # Set load for the entire group (LoadOrchestrator handles stabilization wait)
                if not self.load_orchestrator.set_workload(
                    load["cpu_load"], load["gpu_load"]
                ):
                    tqdm.write(f"Failed to set load for: {description}")
                    main_pbar.update(episodes_per_load)
                    continue

                # Run episodes for each episode type
                for et in self.episode_types:
                    fan_combinations = self._get_fan_combinations(et.mode)

                    for varying_fans in fan_combinations:
                        fans_str = "+".join(varying_fans)
                        tqdm.write(f"\n  Episode type: {et.mode}, varying: {fans_str}")

                        for rep in range(et.repeats_per_load):
                            episode_id = f"{description}_{et.mode}_{fans_str}_{rep}_{int(time.time())}"
                            seed = self.sampling_seed + self._episode_counter
                            self._episode_counter += 1

                            tqdm.write(
                                f"    Starting episode {rep + 1}/{et.repeats_per_load}: {episode_id}"
                            )

                            episode, samples = self.run_episode(
                                load, episode_id, seed, et.mode, varying_fans
                            )
                            self.episodes.append(episode)
                            self.samples.extend(samples)

                            # Incremental save
                            self.save_samples(output_path)

                            main_pbar.update(1)

                            if episode.aborted:
                                tqdm.write(
                                    f"    Episode aborted: {episode.abort_reason}"
                                )

        # Final summary
        print("\n" + "=" * 80)
        print("DATA COLLECTION COMPLETE")
        print("=" * 80)
        print(f"Total episodes: {len(self.episodes)}")
        print(f"Completed episodes: {sum(1 for e in self.episodes if not e.aborted)}")
        print(f"Aborted episodes: {sum(1 for e in self.episodes if e.aborted)}")
        print(f"Total samples: {len(self.samples)}")
        print(f"Data saved to: {output_path}")
        print("=" * 80 + "\n")

    def save_samples(self, output_path: Path) -> None:
        """Save samples to CSV file.

        Args:
            output_path: Path to output CSV file
        """
        # Determine number of cores from samples
        num_cores = 12  # default
        for sample in self.samples:
            if sample.P_cpu_cores:
                num_cores = max(sample.P_cpu_cores.keys()) + 1
                break

        # Save CSV (drop privileges to ensure proper ownership)
        with drop_privileges():
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w", newline="") as f:
                writer = csv.DictWriter(
                    f, fieldnames=TimeSeriesSample.csv_header(num_cores)
                )
                writer.writeheader()

                for sample in self.samples:
                    writer.writerow(sample.to_dict())
