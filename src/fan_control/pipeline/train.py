"""
Training stage for DVC pipeline.
Trains a dynamic thermal model using the model registry.
"""

import json
import logging
from pathlib import Path

import pandas as pd
import yaml

from fan_control.models import get_model

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


if __name__ == "__main__":
    # Load parameters
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    model_config = params["model"]
    model_type = model_config["type"]

    data_dir = Path("data/processed")
    output_dir = Path(f"out/models/{model_type}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load training data
    train_path = data_dir / "train.csv"
    if not train_path.exists():
        logger.error(f"Training data not found at {train_path}")
        raise FileNotFoundError(f"Training data not found at {train_path}")

    train_df = pd.read_csv(train_path)
    logger.info(f"Loaded {len(train_df)} training samples from {train_path}")

    # Create model using registry
    logger.info(f"Creating {model_type} model...")
    model = get_model(model_type, model_config)

    # Train model
    logger.info("Training model...")
    metrics = model.train(train_df)

    # Save model
    model.save(output_dir)

    # Save training metrics
    metrics_dir = Path("out/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)

    train_metrics = {
        "model_type": model_type,
        "train": metrics,
    }

    metrics_path = metrics_dir / "train_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(train_metrics, f, indent=2)

    logger.info(f"Saved training metrics to {metrics_path}")
    logger.info(f"Model saved to {output_dir}")
