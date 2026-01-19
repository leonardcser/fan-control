"""
Physics model equations for thermal prediction.

SIMPLIFIED LINEAR MODEL based on data analysis:
- T_cpu = T_amb + P_cpu × (R_base - k_pump·pwm7 - k_fan2·pwm2 - k_fan4·pwm4 - k_fan5·pwm5 - R_min)
- T_gpu = T_amb + P_gpu × (R_base_gpu - k_fan4·pwm4 - k_fan5·pwm5 - R_min)

Data shows cooling effects are small (~1°C across full range), so linear approximation is appropriate.
The original complex 1/conductance model had numerical stability issues and didn't match observations.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


@dataclass
class FanConfig:
    """Configuration for a single fan's contribution to cooling."""

    param: str  # Parameter name (e.g., 'a', 'b', 'c')
    pump_coupled: bool = False  # Whether this fan couples with pump
    coupling_param: Optional[str] = None  # Coupling parameter (e.g., 'h')


@dataclass
class ComponentModelConfig:
    """Configuration for a component's thermal model (CPU or GPU)."""

    base_resistance: str  # Parameter name for base resistance
    fans: Dict[str, FanConfig]  # Map: device_name -> FanConfig
    baseline: str  # Baseline conductance parameter
    pump_effect: Optional[Dict[str, str]] = None  # Pump parameters if applicable

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

    Parsed from config.yaml's 'model' section. Defines which devices affect
    which components and what parameters to fit.
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

        # Validate bounds format
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

    Returns a function: predict(pwm_dict, P_cpu, T_amb, params_dict) -> T_cpu

    SIMPLIFIED LINEAR MODEL:
    T_cpu = T_amb + P_cpu × (R_base - cooling_reduction)
    where cooling_reduction = k_pump·pwm + k_fan1·pwm1 + ... + baseline
    """
    cpu_cfg = model_config.cpu

    def predict(
        pwm_dict: Dict[str, float], P_cpu: float, T_amb: float, params: Dict[str, float]
    ) -> float:
        """Predict CPU temperature given PWM values, power, and ambient temp."""
        # Base resistance
        R_base = params[cpu_cfg.base_resistance]

        # Cooling reduction (linear sum of all cooling effects)
        cooling_reduction = 0.0

        # Pump effect (reduces thermal resistance)
        if cpu_cfg.pump_effect:
            pump_device = cpu_cfg.pump_effect["device"]
            k_pump = params[cpu_cfg.pump_effect["param"]]
            pwm_pump = pwm_dict.get(pump_device, 0.0)
            cooling_reduction += k_pump * pwm_pump

        # Fan effects (each fan reduces thermal resistance linearly)
        for device, fan_config in cpu_cfg.fans.items():
            pwm_fan = pwm_dict.get(device, 0.0)
            k_fan = params[fan_config.param]
            cooling_reduction += k_fan * pwm_fan
            # Note: pump_coupled is ignored in simplified model

        # Baseline reduction (always present)
        cooling_reduction += params[cpu_cfg.baseline]

        # Total thermal resistance
        R_total = R_base - cooling_reduction

        # Ensure resistance stays positive
        R_total = max(R_total, 0.01)

        # Temperature prediction
        T_cpu = T_amb + P_cpu * R_total
        return T_cpu

    return predict


def build_gpu_predictor(model_config: ModelConfig) -> Callable:
    """
    Build GPU temperature prediction function from model configuration.

    Returns a function: predict(pwm_dict, P_gpu, T_amb, params_dict) -> T_gpu

    SIMPLIFIED LINEAR MODEL:
    T_gpu = T_amb + P_gpu × (R_base - cooling_reduction)
    where cooling_reduction = k_fan1·pwm1 + k_fan2·pwm2 + ... + baseline
    """
    gpu_cfg = model_config.gpu

    def predict(
        pwm_dict: Dict[str, float], P_gpu: float, T_amb: float, params: Dict[str, float]
    ) -> float:
        """Predict GPU temperature given PWM values, power, and ambient temp."""
        # Base resistance
        R_base = params[gpu_cfg.base_resistance]

        # Cooling reduction (linear sum of all cooling effects)
        cooling_reduction = 0.0

        # Fan effects (each fan reduces thermal resistance linearly)
        for device, fan_config in gpu_cfg.fans.items():
            pwm_fan = pwm_dict.get(device, 0.0)
            k_fan = params[fan_config.param]
            cooling_reduction += k_fan * pwm_fan

        # Baseline reduction (always present)
        cooling_reduction += params[gpu_cfg.baseline]

        # Total thermal resistance
        R_total = R_base - cooling_reduction

        # Ensure resistance stays positive
        R_total = max(R_total, 0.01)

        # Temperature prediction
        T_gpu = T_amb + P_gpu * R_total
        return T_gpu

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
    Generic temperature predictor for either component.

    Args:
        component: 'cpu' or 'gpu'
        model_config: Model configuration
        pwm_dict: PWM values (normalized 0-1) for all devices
        power: Component power (W)
        T_amb: Ambient temperature (°C)
        params: Fitted parameters dictionary

    Returns:
        Predicted temperature (°C)
    """
    if component == "cpu":
        predictor = build_cpu_predictor(model_config)
    elif component == "gpu":
        predictor = build_gpu_predictor(model_config)
    else:
        raise ValueError(f"Unknown component: {component}")

    return predictor(pwm_dict, power, T_amb, params)
