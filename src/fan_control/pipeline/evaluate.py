"""
Evaluation stage for DVC pipeline.
Evaluates model on validation data and generates metrics.
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from ..core.train import ThermalModel

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


if __name__ == "__main__":
    data_dir = Path("data/processed")
    models_dir = Path("models")
    metrics_dir = Path("metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load validation data
    val_path = data_dir / "val.csv"
    if not val_path.exists():
        logger.error(f"Validation data not found at {val_path}")
        raise FileNotFoundError(f"Validation data not found at {val_path}")

    val_df = pd.read_csv(val_path)
    logger.info(f"Loaded {len(val_df)} validation samples from {val_path}")

    # 2. Load model
    model_path = models_dir / "thermal_model.pkl"
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}")
        raise FileNotFoundError(f"Model not found at {model_path}")

    model = ThermalModel.load(model_path)
    logger.info(f"Loaded model from {model_path}")

    # 3. Evaluate
    t_cpu_pred, t_gpu_pred = model.predict(val_df)

    metrics = {
        "val": {}
    }

    print("\n=== Validation Results ===")

    if model.cpu_model:
        rmse_cpu = np.sqrt(mean_squared_error(val_df["T_cpu"], t_cpu_pred))
        r2_cpu = r2_score(val_df["T_cpu"], t_cpu_pred)
        mae_cpu = mean_absolute_error(val_df["T_cpu"], t_cpu_pred)
        metrics["val"]["cpu"] = {"rmse": float(rmse_cpu), "r2": float(r2_cpu), "mae": float(mae_cpu)}
        print(f"CPU: RMSE={rmse_cpu:.2f}°C, R2={r2_cpu:.4f}, MAE={mae_cpu:.2f}°C")
        logger.info(f"CPU (val): RMSE={rmse_cpu:.2f}°C, R2={r2_cpu:.4f}, MAE={mae_cpu:.2f}°C")
    else:
        logger.warning("CPU model not trained (skipped)")

    if model.gpu_model:
        rmse_gpu = np.sqrt(mean_squared_error(val_df["T_gpu"], t_gpu_pred))
        r2_gpu = r2_score(val_df["T_gpu"], t_gpu_pred)
        mae_gpu = mean_absolute_error(val_df["T_gpu"], t_gpu_pred)
        metrics["val"]["gpu"] = {"rmse": float(rmse_gpu), "r2": float(r2_gpu), "mae": float(mae_gpu)}
        print(f"GPU: RMSE={rmse_gpu:.2f}°C, R2={r2_gpu:.4f}, MAE={mae_gpu:.2f}°C")
        logger.info(f"GPU (val): RMSE={rmse_gpu:.2f}°C, R2={r2_gpu:.4f}, MAE={mae_gpu:.2f}°C")
    else:
        logger.warning("GPU model not trained (skipped)")

    # 4. Save metrics
    metrics_path = metrics_dir / "val_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved validation metrics to {metrics_path}")
