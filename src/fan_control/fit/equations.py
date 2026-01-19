"""
Physics model equations for thermal prediction.

CONDUCTANCE-BASED THERMAL MODEL:
- Models heat transfer as a serial resistance + convection conductance.
- R_total = R_serial + 1 / (G_base + G_active)
- G_active = Σ (k_fan · pwm_fan) + (k_pump · pwm_pump)
- T_component = T_amb + Power × R_total

Physical Interpretation:
- R_serial: Irreducible thermal resistance (die-to-IHS, TIM, block base).
- G_base: Baseline natural convection/radiation conductance.
- k_fan: Effectiveness of fan in increasing convection conductance.
- k_pump: Effectiveness of pump in increasing convection conductance.

This model is superior to the linear model because:
1. It enforces diminishing returns (increasing airflow has less effect at high speeds).
2. It prevents unphysical negative resistance predictions.
3. It has a physical asymptote (R_serial) representing the cooling limit.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


@dataclass
class FanConfig:
    """Configuration for a single fan's contribution to cooling."""

    param: str  # Parameter name (e.g., 'k_rad_fan')
    pump_coupled: bool = False  # Not used in simplified conductance model
    coupling_param: Optional[str] = None


@dataclass
class ComponentModelConfig:
    """Configuration for a component's thermal model (CPU or GPU)."""

    base_resistance: str  # Maps to R_serial
    fans: Dict[str, FanConfig]  # Map: device_name -> FanConfig
    baseline: str  # Maps to G_base
    pump_effect: Optional[Dict[str, str]] = None  # Maps to k_pump

    def get_param_names(self) -> List[str]:
        """Get all parameter names used in this component's model."""
        params = [self.base_resistance, self.baseline]

        if self.pump_effect:
            params.append(self.pump_effect["param"])

        for fan_config in self.fans.values():
            params.append(fan_config.param)
            if fan_config.pump_coupled and fan_config.coupling_param:
                params.append(fan_config.coupling_param)

        return params


@dataclass
class ModelConfig:
    """
    Complete thermal model configuration.
    """

    cpu: ComponentModelConfig
    gpu: ComponentModelConfig
    initial_guesses: Dict[str, float]
    bounds: Dict[str, List[float]]

    def validate(self) -> None:
        """Validate that all parameters have initial guesses and bounds."""
        all_params = set(self.cpu.get_param_names() + self.gpu.get_param_names())

        missing_guesses = all_params - set(self.initial_guesses.keys())
        if missing_guesses:
            raise ValueError(
                f"Missing initial_guesses for parameters: {missing_guesses}"
            )

        missing_bounds = all_params - set(self.bounds.keys())
        if missing_bounds:
            raise ValueError(f"Missing bounds for parameters: {missing_bounds}")

        for param, bound in self.bounds.items():
            if not isinstance(bound, list) or len(bound) != 2:
                raise ValueError(
                    f"Bounds for '{param}' must be [lower, upper], got: {bound}"
                )
            if bound[0] >= bound[1]:
                raise ValueError(
                    f"Invalid bounds for '{param}': lower >= upper ({bound})"
                )

    @classmethod
    def from_config_dict(cls, config: Dict[str, Any]) -> "ModelConfig":
        """Parse ModelConfig from config.yaml dictionary."""
        if "model" not in config:
            raise ValueError(
                "Config missing 'model' section. Add model structure, "
                "initial_guesses, and bounds to config.yaml"
            )

        model_cfg = config["model"]

        # Parse CPU model
        cpu_struct = model_cfg["structure"]["cpu"]
        cpu_fans = {}
        for device, fan_cfg in cpu_struct.get("fans", {}).items():
            cpu_fans[device] = FanConfig(
                param=fan_cfg["param"],
                pump_coupled=fan_cfg.get("pump_coupled", False),
                coupling_param=fan_cfg.get("coupling_param"),
            )

        cpu_pump_effect = None
        if "pump_effect" in cpu_struct:
            cpu_pump_effect = {
                "device": cpu_struct["pump_effect"]["device"],
                "param": cpu_struct["pump_effect"]["param"],
            }

        cpu_config = ComponentModelConfig(
            base_resistance=cpu_struct["base_resistance"],
            fans=cpu_fans,
            baseline=cpu_struct["baseline"],
            pump_effect=cpu_pump_effect,
        )

        # Parse GPU model
        gpu_struct = model_cfg["structure"]["gpu"]
        gpu_fans = {}
        for device, fan_cfg in gpu_struct.get("fans", {}).items():
            gpu_fans[device] = FanConfig(param=fan_cfg["param"])

        gpu_config = ComponentModelConfig(
            base_resistance=gpu_struct["base_resistance"],
            fans=gpu_fans,
            baseline=gpu_struct["baseline"],
        )

        model_config = cls(
            cpu=cpu_config,
            gpu=gpu_config,
            initial_guesses=model_cfg["initial_guesses"],
            bounds=model_cfg["bounds"],
        )

        model_config.validate()
        return model_config


def build_cpu_predictor(model_config: ModelConfig) -> Callable:
    """
    Build CPU temperature prediction function from model configuration.

    Returns: predict(pwm_dict, P_cpu, T_amb, params) -> T_cpu

    Model: R_total = R_serial + 1 / (G_base + Σ k·pwm)
    """
    cpu_cfg = model_config.cpu

    def predict(
        pwm_dict: Dict[str, float], P_cpu: float, T_amb: float, params: Dict[str, float]
    ) -> float:
        # 1. Serial Resistance (Irreducible)
        R_serial = params[cpu_cfg.base_resistance]

        # 2. Conductance Accumulator
        G_total = params[cpu_cfg.baseline]  # Start with baseline conductance

        # Pump Contribution to Conductance
        if cpu_cfg.pump_effect:
            pump_device = cpu_cfg.pump_effect["device"]
            k_pump = params[cpu_cfg.pump_effect["param"]]
            pwm_pump = pwm_dict.get(pump_device, 0.0)
            G_total += k_pump * pwm_pump

        # Fan Contributions to Conductance
        for device, fan_config in cpu_cfg.fans.items():
            pwm_fan = pwm_dict.get(device, 0.0)
            k_fan = params[fan_config.param]
            G_total += k_fan * pwm_fan

        # Prevent division by zero
        G_total = max(G_total, 1e-6)

        # 3. Total Resistance
        R_total = R_serial + (1.0 / G_total)

        # 4. Temperature Prediction
        return T_amb + P_cpu * R_total

    return predict


def build_gpu_predictor(model_config: ModelConfig) -> Callable:
    """
    Build GPU temperature prediction function from model configuration.

    Returns: predict(pwm_dict, P_gpu, T_amb, params) -> T_gpu

    Model: R_total = R_serial + 1 / (G_base + Σ k·pwm)
    """
    gpu_cfg = model_config.gpu

    def predict(
        pwm_dict: Dict[str, float], P_gpu: float, T_amb: float, params: Dict[str, float]
    ) -> float:
        # 1. Serial Resistance (Irreducible)
        R_serial = params[gpu_cfg.base_resistance]

        # 2. Conductance Accumulator
        G_total = params[gpu_cfg.baseline]

        # Fan Contributions to Conductance
        for device, fan_config in gpu_cfg.fans.items():
            pwm_fan = pwm_dict.get(device, 0.0)
            k_fan = params[fan_config.param]
            G_total += k_fan * pwm_fan

        # Prevent division by zero
        G_total = max(G_total, 1e-6)

        # 3. Total Resistance
        R_total = R_serial + (1.0 / G_total)

        # 4. Temperature Prediction
        return T_amb + P_gpu * R_total

    return predict


def predict_temp(
    component: str,
    model_config: ModelConfig,
    pwm_dict: Dict[str, float],
    power: float,
    T_amb: float,
    params: Dict[str, float],
) -> float:
    """
    Generic temperature predictor.
    """
    if component == "cpu":
        predictor = build_cpu_predictor(model_config)
    elif component == "gpu":
        predictor = build_gpu_predictor(model_config)
    else:
        raise ValueError(f"Unknown component: {component}")

    return predictor(pwm_dict, power, T_amb, params)