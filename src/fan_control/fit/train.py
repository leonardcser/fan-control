import pandas as pd
import numpy as np
import joblib
import json
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple, Any
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class ThermalModel:
    """
    Wrapper for ML-based thermal models (CPU and GPU).
    Handles feature engineering, training, and prediction.

    NEW STRATEGY: Target Resistance (R) instead of Temperature (T)
    T = T_amb + P * R_pred(features)
    This forces the fan effect to scale with Power.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.features = config["features"]
        self.hyperparams = config.get("hyperparameters", {})

        # 1. Base Models (Context: Power, Ambient -> Baseline Resistance)
        self.cpu_base_model = GradientBoostingRegressor(**self.hyperparams)
        self.gpu_base_model = GradientBoostingRegressor(**self.hyperparams)

        # 2. Fan Models (Control: Fan Speed -> Resistance Reduction)
        # We force positive coefficients because 1/PWM should increase Resistance
        self.cpu_fan_model = LinearRegression(positive=True)
        self.gpu_fan_model = LinearRegression(positive=True)

        self.feature_names_ = None  # Set after training

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply physics-inspired feature engineering."""
        df_eng = df.copy()

        # 0. CPU cores and heat flux density
        # Heat concentrated in fewer cores = higher local temperature
        if "cpu_cores" in df.columns:
            # Ensure cpu_cores is at least 1 for computation (0 = idle)
            cores = df["cpu_cores"].clip(lower=1)
            df_eng["cpu_cores"] = df["cpu_cores"]

            # Heat flux density: power per active core (W/core)
            # Higher density = harder to cool
            if "P_cpu" in df.columns:
                df_eng["heat_flux_density"] = df["P_cpu"] / cores

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

        # 3. Fan-Fan Interactions (e.g. Pump * Fan)
        # R_total ~ 1/(Flow_water * Flow_air) => inv_pwm_water * inv_pwm_air
        if "inv_pwm2" in df_eng.columns and "inv_pwm7" in df_eng.columns:
            df_eng["interact_pwm2_pwm7"] = df_eng["inv_pwm2"] * df_eng["inv_pwm7"]

        return df_eng

    def train(self, df: pd.DataFrame):
        """Train 2-stage models (Base + Residual) for CPU and GPU."""
        df_eng = self._engineer_features(df)

        # Define feature sets

        fan_feats = [c for c in df_eng.columns if c.startswith("inv_")]

        if "T_cpu" in df.columns and "P_cpu" in df.columns and "T_amb" in df.columns:
            logger.info("Training CPU Model (2-Stage)...")

            # 1. Prepare Data
            mask = df["P_cpu"] > 10.0
            data = df_eng[mask].copy()

            # Calculate Target Resistance
            delta_T = data["T_cpu"] - data["T_amb"]
            R_cpu = delta_T / data["P_cpu"]

            # 2. Train Base Model (P_cpu, T_amb, heat_flux_density -> R_cpu)
            # Include heat_flux_density if available (captures core count effect)
            base_features = ["P_cpu", "T_amb"]
            if "heat_flux_density" in data.columns:
                base_features.append("heat_flux_density")
            X_base = data[base_features]
            self.cpu_base_model.fit(X_base, R_cpu)
            self._cpu_base_features = base_features  # Store for prediction

            # 3. Calculate Residuals
            R_base_pred = self.cpu_base_model.predict(X_base)
            R_resid = R_cpu - R_base_pred

            # 4. Train Fan Model (inv_pwm -> R_resid)
            # Only use configured fan features
            cpu_fan_feats = [
                f
                for f in fan_feats
                if f in self.features or f.replace("inv_", "") in self.features
            ]

            # Include interactions if components are in features
            if "interact_pwm2_pwm7" in df_eng.columns:
                # Check if base features are in self.features
                if "pwm2" in self.features and "pwm7" in self.features:
                    cpu_fan_feats.append("interact_pwm2_pwm7")

            # Fallback if self.features uses raw pwm names
            if not cpu_fan_feats:
                cpu_fan_feats = [
                    f for f in fan_feats if f.replace("inv_", "") in self.features
                ]

            X_fan = data[cpu_fan_feats]
            self.cpu_fan_model.fit(X_fan, R_resid)

            logger.info(
                f"CPU Fan Coefficients: {dict(zip(cpu_fan_feats, self.cpu_fan_model.coef_))}"
            )

        if "T_gpu" in df.columns and "P_gpu" in df.columns and "T_amb" in df.columns:
            logger.info("Training GPU Model (2-Stage)...")

            mask = df["P_gpu"] > 10.0
            data = df_eng[mask].copy()

            delta_T = data["T_gpu"] - data["T_amb"]
            R_gpu = delta_T / data["P_gpu"]

            X_base = data[["P_gpu", "T_amb"]]
            self.gpu_base_model.fit(X_base, R_gpu)

            R_base_pred = self.gpu_base_model.predict(X_base)
            R_resid = R_gpu - R_base_pred

            X_fan = data[fan_feats]
            self.gpu_fan_model.fit(X_fan, R_resid)

            logger.info(
                f"GPU Fan Coefficients: {dict(zip(fan_feats, self.gpu_fan_model.coef_))}"
            )

        # Store feature names for prediction consistency
        self.feature_names_ = {
            "base": [
                "P_cpu",
                "P_gpu",
                "T_amb",
            ],  # Superset, handled by column selection
            "fan": fan_feats,
        }

    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Predict temperatures using 2-stage model."""
        df_eng = self._engineer_features(df)
        fan_feats = [c for c in df_eng.columns if c.startswith("inv_")]

        # --- CPU Prediction ---
        t_cpu = np.zeros(len(df))
        if hasattr(self, "cpu_base_model"):  # Check if trained
            # 1. Base Resistance (use stored features from training)
            base_features = getattr(self, "_cpu_base_features", ["P_cpu", "T_amb"])
            X_base = df_eng[base_features]
            r_base = self.cpu_base_model.predict(X_base)

            X_fan = df_eng[self.cpu_fan_model.feature_names_in_]
            r_resid = self.cpu_fan_model.predict(X_fan)

            # 3. Total R -> Temperature
            r_total = r_base + r_resid
            t_cpu = df["T_amb"] + df["P_cpu"] * r_total

        # --- GPU Prediction ---
        t_gpu = np.zeros(len(df))
        if hasattr(self, "gpu_base_model"):
            X_base = df_eng[["P_gpu", "T_amb"]]
            r_base = self.gpu_base_model.predict(X_base)

            X_fan = df_eng[fan_feats]
            r_resid = self.gpu_fan_model.predict(X_fan)

            r_total = r_base + r_resid
            t_gpu = df["T_amb"] + df["P_gpu"] * r_total

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
    plt.title("CPU Temperature: Predicted vs Actual (GBM on Resistance)")
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
    plt.title("GPU Temperature: Predicted vs Actual (GBM on Resistance)")
    plt.legend()
    plt.savefig(output_dir / "val_gpu_pred_vs_actual.png")
    plt.close()
