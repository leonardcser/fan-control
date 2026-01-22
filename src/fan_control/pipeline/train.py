"""
Training stage for DVC pipeline.
Trains the ThermalModel on processed data and saves model + metrics.
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
import yaml

from ..core.train import ThermalModel

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


if __name__ == "__main__":
    # Load parameters from params.yaml
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    # Get ML model configuration from params
    ml_config = params.get("ml_model", {})

    # Update hyperparameters from params.yaml if provided
    if "model" in params:
        model_params = params["model"]
        if "n_estimators" in model_params:
            ml_config["hyperparameters"]["n_estimators"] = model_params["n_estimators"]
        if "learning_rate" in model_params:
            ml_config["hyperparameters"]["learning_rate"] = model_params["learning_rate"]
        if "max_depth" in model_params:
            ml_config["hyperparameters"]["max_depth"] = model_params["max_depth"]

    data_dir = Path("data/processed")
    output_dir = Path("out/models")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load processed data
    train_path = data_dir / "train.csv"
    if not train_path.exists():
        logger.error(f"Training data not found at {train_path}")
        raise FileNotFoundError(f"Training data not found at {train_path}")

    train_df = pd.read_csv(train_path)
    logger.info(f"Loaded {len(train_df)} training samples from {train_path}")

    # 2. Initialize and train model
    model = ThermalModel(ml_config)
    model.train(train_df)

    # Check if at least one model was trained
    if model.cpu_model is None and model.gpu_model is None:
        logger.error("No models were trained (insufficient data).")
        raise ValueError("No models trained - insufficient data")

    # 3. Save model
    model_path = output_dir / "thermal_model.pkl"
    model.save(model_path)

    # 4. Calculate training metrics
    t_cpu_pred, t_gpu_pred = model.predict(train_df)

    metrics = {
        "train": {}
    }

    if model.cpu_model:
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        rmse_cpu = np.sqrt(mean_squared_error(train_df["T_cpu"], t_cpu_pred))
        r2_cpu = r2_score(train_df["T_cpu"], t_cpu_pred)
        mae_cpu = mean_absolute_error(train_df["T_cpu"], t_cpu_pred)
        metrics["train"]["cpu"] = {"rmse": float(rmse_cpu), "r2": float(r2_cpu), "mae": float(mae_cpu)}
        logger.info(f"CPU (train): RMSE={rmse_cpu:.2f}°C, R2={r2_cpu:.4f}, MAE={mae_cpu:.2f}°C")

    if model.gpu_model:
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        rmse_gpu = np.sqrt(mean_squared_error(train_df["T_gpu"], t_gpu_pred))
        r2_gpu = r2_score(train_df["T_gpu"], t_gpu_pred)
        mae_gpu = mean_absolute_error(train_df["T_gpu"], t_gpu_pred)
        metrics["train"]["gpu"] = {"rmse": float(rmse_gpu), "r2": float(r2_gpu), "mae": float(mae_gpu)}
        logger.info(f"GPU (train): RMSE={rmse_gpu:.2f}°C, R2={r2_gpu:.4f}, MAE={mae_gpu:.2f}°C")

    # Save metrics
    metrics_dir = Path("out/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / "train_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved training metrics to {metrics_path}")
