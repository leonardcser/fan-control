import pandas as pd
import numpy as np
import joblib
import json
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class ThermalModel:
    """
    Wrapper for ML-based thermal models (CPU and GPU).
    Handles feature engineering, training, and prediction.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.features = config["features"]
        self.hyperparams = config.get("hyperparameters", {})
        self.cpu_model = GradientBoostingRegressor(**self.hyperparams)
        self.gpu_model = GradientBoostingRegressor(**self.hyperparams)
        self.feature_names_ = None  # Set after training

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply physics-inspired feature engineering."""
        df_eng = df.copy()

        # 1. Inverse PWM (Resistance is proportional to 1/Flow)
        # Add epsilon to prevent division by zero
        for pwm in ["pwm2", "pwm4", "pwm5", "pwm7"]:
            if pwm in df.columns:
                df_eng[f"inv_{pwm}"] = 1.0 / (df[pwm] + 10.0)

        # 2. Power-Flow Interactions (Heat Flux)
        # Power * Resistance ~ Delta T
        if "P_cpu" in df.columns:
            if "inv_pwm2" in df_eng.columns:
                df_eng["P_cpu_pwm2"] = df_eng["P_cpu"] * df_eng["inv_pwm2"]
            if "inv_pwm7" in df_eng.columns:
                df_eng["P_cpu_pwm7"] = df_eng["P_cpu"] * df_eng["inv_pwm7"]

        if "P_gpu" in df.columns:
            if "inv_pwm4" in df_eng.columns:
                df_eng["P_gpu_pwm4"] = df_eng["P_gpu"] * df_eng["inv_pwm4"]

        return df_eng

    def _get_feature_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract and engineer features for the model."""
        df_eng = self._engineer_features(df)

        # Base features
        cols = [f for f in self.features if f in df_eng.columns]

        # Add engineered features if they exist
        eng_cols = [c for c in df_eng.columns if c.startswith("inv_") or "_pwm" in c]
        # Filter eng_cols to ensure we don't duplicate or include targets
        eng_cols = [
            c for c in eng_cols if c not in cols and c not in ["T_cpu", "T_gpu"]
        ]

        final_cols = cols + eng_cols

        # Save feature names during training
        if self.feature_names_ is None:
            self.feature_names_ = final_cols

        # During prediction, ensure we have the same columns
        return df_eng[self.feature_names_]

    def train(self, df: pd.DataFrame):
        """Train both CPU and GPU models."""
        X = self._get_feature_matrix(df)

        if "T_cpu" in df.columns:
            logger.info("Training CPU Model...")
            self.cpu_model.fit(X, df["T_cpu"])

        if "T_gpu" in df.columns:
            logger.info("Training GPU Model...")
            self.gpu_model.fit(X, df["T_gpu"])

    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Predict temperatures. Returns (T_cpu_pred, T_gpu_pred)."""
        X = self._get_feature_matrix(df)

        t_cpu = self.cpu_model.predict(X)
        t_gpu = self.gpu_model.predict(X)

        return t_cpu, t_gpu

    def save(self, path: Path):
        """Save the trained model to disk."""
        joblib.dump(self, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: Path) -> "ThermalModel":
        """Load a trained model from disk."""
        return joblib.load(path)


def train_model(config: Dict[str, Any], data_path: Path, output_dir: Path):
    """
    Main training routine.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    csv_files = sorted(data_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_path}")

    df_list = []
    for f in csv_files:
        try:
            df_list.append(pd.read_csv(f))
            logger.info(f"Loaded {len(df_list[-1])} rows from {f.name}")
        except Exception as e:
            logger.error(f"Failed to load {f}: {e}")

    full_df = pd.concat(df_list, ignore_index=True)
    logger.info(f"Total training samples: {len(full_df)}")

    # 2. Filter valid data (Equilibrated only ideally, but we'll use all for now)
    # df = full_df[full_df['equilibrated'] == True].copy()
    df = full_df.copy()  # Use all data for robustness

    # 3. Initialize and Train Model
    ml_config = config["ml_model"]
    model = ThermalModel(ml_config)

    # Split for validation metrics
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    model.train(train_df)

    # 4. Validation
    logger.info("Validating model...")
    t_cpu_pred, t_gpu_pred = model.predict(val_df)

    metrics = {}

    # CPU Metrics
    rmse_cpu = np.sqrt(mean_squared_error(val_df["T_cpu"], t_cpu_pred))
    r2_cpu = r2_score(val_df["T_cpu"], t_cpu_pred)
    mae_cpu = mean_absolute_error(val_df["T_cpu"], t_cpu_pred)
    metrics["cpu"] = {"rmse": rmse_cpu, "r2": r2_cpu, "mae": mae_cpu}

    # GPU Metrics
    rmse_gpu = np.sqrt(mean_squared_error(val_df["T_gpu"], t_gpu_pred))
    r2_gpu = r2_score(val_df["T_gpu"], t_gpu_pred)
    mae_gpu = mean_absolute_error(val_df["T_gpu"], t_gpu_pred)
    metrics["gpu"] = {"rmse": rmse_gpu, "r2": r2_gpu, "mae": mae_gpu}

    print("\n=== Validation Results ===")
    print(f"CPU: RMSE={rmse_cpu:.2f}°C, R2={r2_cpu:.4f}, MAE={mae_cpu:.2f}°C")
    print(f"GPU: RMSE={rmse_gpu:.2f}°C, R2={r2_gpu:.4f}, MAE={mae_gpu:.2f}°C")

    # 5. Save Artifacts
    model_path = output_dir / "thermal_model.pkl"
    model.save(model_path)

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # 6. Generate Plots
    plot_validation(val_df, t_cpu_pred, t_gpu_pred, output_dir)


def plot_validation(
    df: pd.DataFrame, pred_cpu: np.ndarray, pred_gpu: np.ndarray, output_dir: Path
):
    """Generate Predicted vs Actual plots."""
    sns.set_theme(style="whitegrid")

    # CPU Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(df["T_cpu"], pred_cpu, alpha=0.6, label="Data")
    min_val = min(df["T_cpu"].min(), pred_cpu.min())
    max_val = max(df["T_cpu"].max(), pred_cpu.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect Fit")
    plt.xlabel("Actual CPU Temp (°C)")
    plt.ylabel("Predicted CPU Temp (°C)")
    plt.title("CPU Temperature: Predicted vs Actual (Gradient Boosting)")
    plt.legend()
    plt.savefig(output_dir / "val_cpu_pred_vs_actual.png")
    plt.close()

    # GPU Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(df["T_gpu"], pred_gpu, alpha=0.6, color="green", label="Data")
    min_val = min(df["T_gpu"].min(), pred_gpu.min())
    max_val = max(df["T_gpu"].max(), pred_gpu.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect Fit")
    plt.xlabel("Actual GPU Temp (°C)")
    plt.ylabel("Predicted GPU Temp (°C)")
    plt.title("GPU Temperature: Predicted vs Actual (Gradient Boosting)")
    plt.legend()
    plt.savefig(output_dir / "val_gpu_pred_vs_actual.png")
    plt.close()
