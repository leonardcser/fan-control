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

    def generate_step_response(
        self,
        transition: dict,
    ) -> list[PWMStep]:
        """Generate schedule for a step response episode.

        Args:
            transition: Dict with 'from', 'to', 'settle', 'observe' keys
                - from: Dict of {pwm2: val, pwm4: val, pwm5: val} percentages
                - to: Dict of {pwm2: val, pwm4: val, pwm5: val} percentages
                - settle: Seconds to hold at 'from' level before step
                - observe: Seconds to observe after step to 'to' level

        Returns:
            List of PWMStep objects
        """
        steps = []

        # Convert percentages to 0-255 scale
        from_pwm = {
            k: self._apply_stall_check(k, int(round(v * 255 / 100)))
            for k, v in transition["from"].items()
        }
        to_pwm = {
            k: self._apply_stall_check(k, int(round(v * 255 / 100)))
            for k, v in transition["to"].items()
        }

        # Start at 'from' level
        steps.append(PWMStep(time_offset=0.0, pwm_values=from_pwm))

        # Step to 'to' level after settle time
        settle_time = transition["settle"]
        steps.append(PWMStep(time_offset=settle_time, pwm_values=to_pwm))

        return steps

    def generate_prbs(
        self,
        duration: float,
        level_low: dict,
        level_high: dict,
        hold_times: list[int],
        seed: int,
    ) -> list[PWMStep]:
        """Generate PRBS (Pseudo-Random Binary Sequence) schedule.

        Args:
            duration: Total episode duration in seconds
            level_low: Low PWM state {pwm2: val, ...} in percentages
            level_high: High PWM state {pwm2: val, ...} in percentages
            hold_times: List of possible hold durations to randomly select from
            seed: Random seed for reproducibility

        Returns:
            List of PWMStep objects with alternating levels
        """
        rng = np.random.default_rng(seed)
        steps = []

        # Convert percentages to 0-255 scale
        low_pwm = {
            k: self._apply_stall_check(k, int(round(v * 255 / 100)))
            for k, v in level_low.items()
        }
        high_pwm = {
            k: self._apply_stall_check(k, int(round(v * 255 / 100)))
            for k, v in level_high.items()
        }

        t = 0.0
        current_level = "low"
        steps.append(PWMStep(time_offset=0.0, pwm_values=low_pwm))

        while t < duration:
            hold = int(rng.choice(hold_times))
            t += hold
            if t >= duration:
                break

            # Toggle level
            if current_level == "low":
                steps.append(PWMStep(time_offset=t, pwm_values=high_pwm))
                current_level = "high"
            else:
                steps.append(PWMStep(time_offset=t, pwm_values=low_pwm))
                current_level = "low"

        return steps

    def generate_staircase(
        self,
        levels: list[int],
        hold_per_step: float,
        fan_scale: dict,
    ) -> list[PWMStep]:
        """Generate staircase schedule for nonlinearity identification.

        Args:
            levels: List of PWM percentages to step through (e.g., [20, 40, 60, 80, 100, 80, 60, 40, 20])
            hold_per_step: Seconds to hold at each level
            fan_scale: Dict of {pwm2: scale, ...} where scale is 0.0-1.0 multiplier

        Returns:
            List of PWMStep objects stepping through levels
        """
        steps = []
        t = 0.0

        for level in levels:
            # Apply fan scaling and convert to 0-255
            pwm_values = {}
            for device_key, scale in fan_scale.items():
                pct = level * scale
                pwm_val = int(round(pct * 255 / 100))
                pwm_values[device_key] = self._apply_stall_check(device_key, pwm_val)

            steps.append(PWMStep(time_offset=t, pwm_values=pwm_values))
            t += hold_per_step

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

        # Parse episode types (skip disabled ones)
        self.episode_types: list[EpisodeType] = []
        for et in self.data_config["episode_types"]:
            if not et.get("enabled", True):
                continue
            mode = et["mode"]
            repeats = et.get("repeats_per_load", 1)
            # Store full config for new episode types
            config = {k: v for k, v in et.items() if k not in ("mode", "repeats_per_load", "enabled")}
            self.episode_types.append(EpisodeType(mode=mode, repeats_per_load=repeats, config=config))

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

    def run_scheduled_episode(
        self,
        load_config: dict,
        episode_id: str,
        schedule: list[PWMStep],
        total_duration: float,
    ) -> tuple[Episode, list[TimeSeriesSample]]:
        """Run an episode with a pre-generated PWM schedule.

        Used for step_response, prbs, and staircase episodes.

        Args:
            load_config: Load configuration dict
            episode_id: Unique episode identifier
            schedule: Pre-generated list of PWMStep objects
            total_duration: Total episode duration in seconds

        Returns:
            Tuple of (Episode metadata, list of samples)
        """
        samples: list[TimeSeriesSample] = []
        start_time = time.time()
        current_step_idx = 0
        sample_index = 0
        aborted = False
        abort_reason: Optional[str] = None

        # Apply initial PWM
        self._apply_pwm(schedule[0].pwm_values)

        # Progress bar for episode
        expected_samples = int(total_duration / self.sample_interval)
        with tqdm(
            total=expected_samples,
            desc=f"Episode {episode_id[:30]}",
            unit="sample",
            leave=False,
            bar_format="{desc}: {percentage:3.0f}% |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        ) as pbar:
            while (elapsed := time.time() - start_time) < total_duration:
                loop_start = time.time()

                # Check for scheduled PWM step
                if (
                    current_step_idx + 1 < len(schedule)
                    and elapsed >= schedule[current_step_idx + 1].time_offset
                ):
                    current_step_idx += 1
                    self._apply_pwm(schedule[current_step_idx].pwm_values)
                    pwm_str = ", ".join(
                        f"{k}={int(v * 100 / 255)}%"
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

                # Sample immediately
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

    def _calculate_total_episodes(self, load_levels: list[dict]) -> int:
        """Calculate total episodes across all types and loads."""
        total = 0
        load_descriptions = [ld["description"] for ld in load_levels]

        for et in self.episode_types:
            if et.mode in ("single_isolation", "pair_isolation", "full_sequential"):
                # Old episode types run on all loads
                combinations = self._get_fan_combinations(et.mode)
                total += len(combinations) * et.repeats_per_load * len(load_levels)

            elif et.mode == "step_response":
                # step_response runs specific transitions on target loads
                target_loads = et.config.get("target_loads", load_descriptions)
                transitions = et.config.get("transitions", [])
                matching_loads = [tl for tl in target_loads if tl in load_descriptions]
                total += len(matching_loads) * len(transitions)

            elif et.mode == "prbs":
                # prbs runs one long episode per target load
                target_loads = et.config.get("target_loads", load_descriptions)
                matching_loads = [tl for tl in target_loads if tl in load_descriptions]
                total += len(matching_loads)

            elif et.mode == "staircase":
                # staircase runs one episode per target load
                target_loads = et.config.get("target_loads", load_descriptions)
                matching_loads = [tl for tl in target_loads if tl in load_descriptions]
                total += len(matching_loads)

        return total

    def run_collection(self, output_path: Path) -> None:
        """Run full campaign: iterate over episode types and load levels.

        Args:
            output_path: Path to output CSV file
        """
        load_levels = self.data_config["load_levels"]
        load_descriptions = [ld["description"] for ld in load_levels]
        total_episodes = self._calculate_total_episodes(load_levels)

        print("\n" + "=" * 80)
        print("STARTING TIME-SERIES DATA COLLECTION")
        print("=" * 80)
        print(f"Default episode duration: {self.episode_duration}s")
        print(f"Episode types: {len(self.episode_types)}")
        for et in self.episode_types:
            if et.mode in ("single_isolation", "pair_isolation", "full_sequential"):
                combos = len(self._get_fan_combinations(et.mode))
                print(f"  - {et.mode}: {et.repeats_per_load} repeats × {combos} combinations (all loads)")
            elif et.mode == "step_response":
                transitions = len(et.config.get("transitions", []))
                target_loads = et.config.get("target_loads", [])
                print(f"  - {et.mode}: {transitions} transitions × {len(target_loads)} loads")
            elif et.mode == "prbs":
                duration = et.config.get("duration", 600)
                target_loads = et.config.get("target_loads", [])
                print(f"  - {et.mode}: {duration}s episodes × {len(target_loads)} loads")
            elif et.mode == "staircase":
                levels = len(et.config.get("levels", []))
                target_loads = et.config.get("target_loads", [])
                print(f"  - {et.mode}: {levels} levels × {len(target_loads)} loads")
        print(f"Load levels: {len(load_levels)}")
        print(f"Total episodes: {total_episodes}")
        print(f"Sample interval: {self.sample_interval}s")

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
                    # Skip appropriate number of episodes
                    for et in self.episode_types:
                        if et.mode in ("single_isolation", "pair_isolation", "full_sequential"):
                            main_pbar.update(len(self._get_fan_combinations(et.mode)) * et.repeats_per_load)
                        elif et.mode == "step_response" and description in et.config.get("target_loads", []):
                            main_pbar.update(len(et.config.get("transitions", [])))
                        elif et.mode in ("prbs", "staircase") and description in et.config.get("target_loads", []):
                            main_pbar.update(1)
                    continue

                # Run episodes for each episode type
                for et in self.episode_types:
                    if et.mode in ("single_isolation", "pair_isolation", "full_sequential"):
                        # Original episode types
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
                                    tqdm.write(f"    Episode aborted: {episode.abort_reason}")

                    elif et.mode == "step_response":
                        # Step response episodes - only for target loads
                        target_loads = et.config.get("target_loads", load_descriptions)
                        if description not in target_loads:
                            continue

                        transitions = et.config.get("transitions", [])
                        tqdm.write(f"\n  Episode type: {et.mode} ({len(transitions)} transitions)")

                        for t_idx, transition in enumerate(transitions):
                            from_pwm = transition["from"]
                            to_pwm = transition["to"]
                            settle = transition["settle"]
                            observe = transition["observe"]
                            total_duration = settle + observe

                            # Create descriptive episode ID
                            from_str = f"pwm{int(sum(from_pwm.values()) / len(from_pwm))}"
                            to_str = f"pwm{int(sum(to_pwm.values()) / len(to_pwm))}"
                            episode_id = f"{description}_step_{from_str}_to_{to_str}_{int(time.time())}"

                            tqdm.write(f"    Transition {t_idx + 1}/{len(transitions)}: {from_str} → {to_str} (settle={settle}s, observe={observe}s)")

                            # Generate schedule
                            schedule = self.schedule_generator.generate_step_response(transition)

                            episode, samples = self.run_scheduled_episode(
                                load, episode_id, schedule, total_duration
                            )
                            self.episodes.append(episode)
                            self.samples.extend(samples)

                            self.save_samples(output_path)
                            main_pbar.update(1)

                            if episode.aborted:
                                tqdm.write(f"    Episode aborted: {episode.abort_reason}")

                    elif et.mode == "prbs":
                        # PRBS episodes - only for target loads
                        target_loads = et.config.get("target_loads", load_descriptions)
                        if description not in target_loads:
                            continue

                        duration = et.config.get("duration", 600)
                        level_low = et.config.get("level_low", {"pwm2": 25, "pwm4": 0, "pwm5": 0})
                        level_high = et.config.get("level_high", {"pwm2": 100, "pwm4": 80, "pwm5": 100})
                        hold_times = et.config.get("hold_times", [10, 15, 20, 30, 45])

                        episode_id = f"{description}_prbs_{int(time.time())}"
                        seed = self.sampling_seed + self._episode_counter
                        self._episode_counter += 1

                        tqdm.write(f"\n  Episode type: {et.mode} (duration={duration}s)")
                        tqdm.write(f"    Starting PRBS episode: {episode_id}")

                        # Generate schedule
                        schedule = self.schedule_generator.generate_prbs(
                            duration, level_low, level_high, hold_times, seed
                        )

                        episode, samples = self.run_scheduled_episode(
                            load, episode_id, schedule, duration
                        )
                        self.episodes.append(episode)
                        self.samples.extend(samples)

                        self.save_samples(output_path)
                        main_pbar.update(1)

                        if episode.aborted:
                            tqdm.write(f"    Episode aborted: {episode.abort_reason}")

                    elif et.mode == "staircase":
                        # Staircase episodes - only for target loads
                        target_loads = et.config.get("target_loads", load_descriptions)
                        if description not in target_loads:
                            continue

                        levels = et.config.get("levels", [20, 40, 60, 80, 100, 80, 60, 40, 20])
                        hold_per_step = et.config.get("hold_per_step", 45)
                        fan_scale = et.config.get("fan_scale", {"pwm2": 1.0, "pwm4": 0.8, "pwm5": 1.0})
                        total_duration = len(levels) * hold_per_step

                        episode_id = f"{description}_staircase_{int(time.time())}"

                        tqdm.write(f"\n  Episode type: {et.mode} ({len(levels)} levels × {hold_per_step}s = {total_duration}s)")
                        tqdm.write(f"    Starting staircase episode: {episode_id}")

                        # Generate schedule
                        schedule = self.schedule_generator.generate_staircase(
                            levels, hold_per_step, fan_scale
                        )

                        episode, samples = self.run_scheduled_episode(
                            load, episode_id, schedule, total_duration
                        )
                        self.episodes.append(episode)
                        self.samples.extend(samples)

                        self.save_samples(output_path)
                        main_pbar.update(1)

                        if episode.aborted:
                            tqdm.write(f"    Episode aborted: {episode.abort_reason}")

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
