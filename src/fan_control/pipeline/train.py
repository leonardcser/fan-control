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
    train_output_dir = Path("out/train")
    model_output_dir = train_output_dir / "models" / model_type
    logs_dir = train_output_dir / "logs"

    model_output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Load training data
    train_path = data_dir / "train.csv"
    if not train_path.exists():
        logger.error(f"Training data not found at {train_path}")
        raise FileNotFoundError(f"Training data not found at {train_path}")

    train_df = pd.read_csv(train_path)
    logger.info(f"Loaded {len(train_df)} training samples from {train_path}")

    # Load validation data
    val_path = data_dir / "val.csv"
    val_df = None
    if val_path.exists():
        val_df = pd.read_csv(val_path)
        logger.info(f"Loaded {len(val_df)} validation samples from {val_path}")
    else:
        logger.warning(f"Validation data not found at {val_path}, training without validation")

    # Add log directory to config for models that support it
    model_config["log_dir"] = str(logs_dir)

    # Create model using registry
    logger.info(f"Creating {model_type} model...")
    model = get_model(model_type, model_config)

    # Train model
    logger.info("Training model...")
    metrics = model.train(train_df, val_df=val_df)

    # Save model
    model.save(model_output_dir)

    # Save training metrics
    train_metrics = {
        "model_type": model_type,
        "train": metrics,
    }

    metrics_path = train_output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(train_metrics, f, indent=2)

    logger.info(f"Saved training metrics to {metrics_path}")
    logger.info(f"Model saved to {model_output_dir}")
