"""
Simulation stage for DVC pipeline.
Runs simulation scenario and generates metrics.
"""

import json
import logging
from pathlib import Path
import yaml

from ..core.simulator import run_simulation

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


if __name__ == "__main__":
    # Load config
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    # Load params
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    models_dir = Path("models")
    metrics_dir = Path("metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model_path = models_dir / "thermal_model.pkl"
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}")
        raise FileNotFoundError(f"Model not found at {model_path}")

    logger.info(f"Loaded model from {model_path}")

    # Run simulation
    logger.info("Starting simulation...")
    run_simulation(model_path, config, params, metrics_dir)
    logger.info("Simulation completed")

    # Generate simulation metrics (summary of simulation results)
    metrics = {
        "sim": {
            "status": "completed"
        }
    }

    metrics_path = metrics_dir / "sim_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved simulation metrics to {metrics_path}")
