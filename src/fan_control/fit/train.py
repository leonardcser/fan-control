import pandas as pd
import numpy as np
import joblib
import json
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple, Any, List

# HistGradientBoostingRegressor supports monotonic constraints natively
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class ThermalModel:
    """
    Advanced Monotonic Thermal Model (GBM).
    Predicts Temperature directly from Power, Ambient, and Fan Speeds.
    Enforces physics via monotonic constraints:
    - Increasing Power -> Increases Temp
    - Increasing Fan Speed -> Decreases Temp

    Replaces the previous 2-stage Resistance model with a single non-linear
    model that better captures saturation and complex interactions.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.features = config["features"]  # Expected raw features from config

        # Hyperparameters
        self.hyperparams = config["hyperparameters"]

        # Map generic GBM params to HistGradientBoostingRegressor params
        # Default to robust settings for small-to-medium datasets
        self.hgb_params = {
            "max_iter": self.hyperparams["n_estimators"],
            "learning_rate": self.hyperparams["learning_rate"],
            "max_depth": self.hyperparams["max_depth"],
            "random_state": self.hyperparams["random_state"],
            "early_stopping": True,
            "scoring": "loss",
            "validation_fraction": 0.1,
            "n_iter_no_change": 10,
        }

        self.cpu_model = None
        self.gpu_model = None

        # Store feature names used for training to ensure consistency
        self.feature_names_in_: List[str] = []

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Feature engineering placeholder.
        Currently no engineered features - GBM learns interactions directly.
        """
        return df.copy()

    def _get_monotonic_constraints(self, feature_names: List[str]) -> List[int]:
        """
        Generate constraint vector for HistGradientBoostingRegressor.
        1: Monotonic increasing (Power, Amb)
        -1: Monotonic decreasing (Fans)
        0: No constraint
        """
        cst = []
        for name in feature_names:
            if name.startswith("pwm"):
                # Increasing Fan Speed -> Decreases Temp
                cst.append(-1)
            elif name.startswith("P_"):
                # Increasing Power -> Increases Temp
                cst.append(1)
            elif name == "T_amb":
                # Higher Ambient -> Higher Temp
                cst.append(1)
            else:
                cst.append(0)
        return cst

    def train(self, df: pd.DataFrame):
        """Train Monotonic models for CPU and GPU."""
        df_eng = self._engineer_features(df)

        # Determine features to use from config
        train_features = list(self.features)

        # Verify columns exist
        train_features = [f for f in train_features if f in df_eng.columns]
        self.feature_names_in_ = train_features

        cst = self._get_monotonic_constraints(train_features)
        logger.info(f"Training Features: {train_features}")
        logger.info(f"Monotonic Constraints: {cst}")

        # --- Train CPU Model ---
        if "T_cpu" in df.columns and "P_cpu" in df.columns:
            logger.info("Training CPU Model (Monotonic GBM)...")
            # Filter valid power for training to avoid noise at 0W
            mask = df["P_cpu"] > 10.0
            X_cpu = df_eng[mask][train_features]
            y_cpu = df_eng[mask]["T_cpu"]

            self.cpu_model = HistGradientBoostingRegressor(
                monotonic_cst=cst, **self.hgb_params
            )
            self.cpu_model.fit(X_cpu, y_cpu)

        # --- Train GPU Model ---
        if "T_gpu" in df.columns and "P_gpu" in df.columns:
            logger.info("Training GPU Model (Monotonic GBM)...")
            mask = df["P_gpu"] > 10.0
            X_gpu = df_eng[mask][train_features]
            y_gpu = df_eng[mask]["T_gpu"]

            self.gpu_model = HistGradientBoostingRegressor(
                monotonic_cst=cst, **self.hgb_params
            )
            self.gpu_model.fit(X_gpu, y_gpu)

    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Predict temperatures."""
        if self.cpu_model is None and self.gpu_model is None:
            # Return zeros if not trained (or raise error)
            logger.warning("Models not trained, returning zeros")
            return np.zeros(len(df)), np.zeros(len(df))

        df_eng = self._engineer_features(df)

        # Handle missing columns safely
        missing = [col for col in self.feature_names_in_ if col not in df_eng.columns]
        if missing:
            logger.warning(f"Missing features for prediction: {missing}")
            # Add missing columns with defaults (safest is to fail, but here we patch)
            for col in missing:
                df_eng[col] = 0

        X = df_eng[self.feature_names_in_]

        t_cpu = np.zeros(len(df))
        if self.cpu_model:
            t_cpu = self.cpu_model.predict(X)

        t_gpu = np.zeros(len(df))
        if self.gpu_model:
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

    # 2. Filter valid data
    # Filter out very low power points for training stability
    df = full_df[(full_df["P_cpu"] > 10.0) | (full_df["P_gpu"] > 10.0)].copy()

    # Filter out thermally-saturated data (thermal throttling region)
    # When CPU hits thermal limit, fans appear ineffective. Remove these points.
    ml_config = config["ml_model"]
    filter_cfg = ml_config["thermal_saturation_filter"]
    max_cpu_temp = filter_cfg["max_cpu_temp"]
    max_gpu_temp = filter_cfg["max_gpu_temp"]

    initial_count = len(df)
    df = df[(df["T_cpu"] < max_cpu_temp) & (df["T_gpu"] < max_gpu_temp)]
    filtered_count = initial_count - len(df)
    if filtered_count > 0:
        logger.info(
            f"Filtered {filtered_count} thermal-throttling samples "
            f"(T_cpu >= {max_cpu_temp} or T_gpu >= {max_gpu_temp})"
        )

    # Normalize PWM to 0-100 range (from 0-255)
    for col in df.columns:
        if col.startswith("pwm"):
            df[col] = df[col] / 2.55

    # 3. Initialize and Train Model
    ml_config = config["ml_model"]
    model = ThermalModel(ml_config)

    # Split for validation metrics
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    # Hint for linter
    train_df = pd.DataFrame(train_df)
    val_df = pd.DataFrame(val_df)

    model.train(train_df)

    # 4. Validation
    logger.info("Validating model...")
    t_cpu_pred, t_gpu_pred = model.predict(val_df)

    metrics = {}

    # CPU Metrics
    if model.cpu_model:
        rmse_cpu = np.sqrt(mean_squared_error(val_df["T_cpu"], t_cpu_pred))
        r2_cpu = r2_score(val_df["T_cpu"], t_cpu_pred)
        mae_cpu = mean_absolute_error(val_df["T_cpu"], t_cpu_pred)
        metrics["cpu"] = {"rmse": rmse_cpu, "r2": r2_cpu, "mae": mae_cpu}
        print(f"CPU: RMSE={rmse_cpu:.2f}°C, R2={r2_cpu:.4f}, MAE={mae_cpu:.2f}°C")

    # GPU Metrics
    if model.gpu_model:
        rmse_gpu = np.sqrt(mean_squared_error(val_df["T_gpu"], t_gpu_pred))
        r2_gpu = r2_score(val_df["T_gpu"], t_gpu_pred)
        mae_gpu = mean_absolute_error(val_df["T_gpu"], t_gpu_pred)
        metrics["gpu"] = {"rmse": rmse_gpu, "r2": r2_gpu, "mae": mae_gpu}
        print(f"GPU: RMSE={rmse_gpu:.2f}°C, R2={r2_gpu:.4f}, MAE={mae_gpu:.2f}°C")

    print("\n=== Validation Results (Monotonic GBM) ===")

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
    if np.any(pred_cpu):
        plt.figure(figsize=(10, 6))
        plt.scatter(df["T_cpu"], pred_cpu, alpha=0.6, label="Data")
        min_val = min(df["T_cpu"].min(), pred_cpu.min())
        max_val = max(df["T_cpu"].max(), pred_cpu.max())
        plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect Fit")
        plt.xlabel("Actual CPU Temp (°C)")
        plt.ylabel("Predicted CPU Temp (°C)")
        plt.title("CPU Temperature: Predicted vs Actual (Monotonic GBM)")
        plt.legend()
        plt.savefig(output_dir / "val_cpu_pred_vs_actual.png")
        plt.close()

    # GPU Plot
    if np.any(pred_gpu):
        plt.figure(figsize=(10, 6))
        plt.scatter(df["T_gpu"], pred_gpu, alpha=0.6, color="green", label="Data")
        min_val = min(df["T_gpu"].min(), pred_gpu.min())
        max_val = max(df["T_gpu"].max(), pred_gpu.max())
        plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect Fit")
        plt.xlabel("Actual GPU Temp (°C)")
        plt.ylabel("Predicted GPU Temp (°C)")
        plt.title("GPU Temperature: Predicted vs Actual (Monotonic GBM)")
        plt.legend()
        plt.savefig(output_dir / "val_gpu_pred_vs_actual.png")
        plt.close()
