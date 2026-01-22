"""
Preprocessing stage for DVC pipeline.
Loads raw CSVs, filters, normalizes, and splits into train/val sets.
"""

import pandas as pd
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
import yaml

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


if __name__ == "__main__":
    # Load parameters from params.yaml
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    preprocess_params = params.get("preprocess", {})
    min_power_cfg = preprocess_params.get("min_power", {"cpu": 10.0, "gpu": 5.0})
    max_temp_cfg = preprocess_params.get("max_temp", {"cpu": 95.0, "gpu": 85.0})
    test_size = preprocess_params.get("test_size", 0.2)

    # Load config for ML parameters if needed
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    data_dir = Path("data/raw")
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load all CSV files from data/raw
    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        logger.error(f"No CSV files found in {data_dir}")
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    df_list = []
    for f in csv_files:
        try:
            df_list.append(pd.read_csv(f))
            logger.info(f"Loaded {len(df_list[-1])} rows from {f.name}")
        except Exception as e:
            logger.error(f"Failed to load {f}: {e}")

    full_df = pd.concat(df_list, ignore_index=True)
    logger.info(f"Total samples after loading: {len(full_df)}")

    # 2. Filter by minimum power
    min_cpu_power = min_power_cfg.get("cpu", 10.0)
    min_gpu_power = min_power_cfg.get("gpu", 5.0)

    df = full_df[(full_df["P_cpu"] > min_cpu_power) | (full_df["P_gpu"] > min_gpu_power)].copy()
    logger.info(f"Filtered to {len(df)} samples with P_cpu > {min_cpu_power}W or P_gpu > {min_gpu_power}W")

    # 3. Filter out thermally-saturated data (thermal throttling)
    max_cpu_temp = max_temp_cfg.get("cpu", 95.0)
    max_gpu_temp = max_temp_cfg.get("gpu", 85.0)

    initial_count = len(df)
    df = df[(df["T_cpu"] < max_cpu_temp) & (df["T_gpu"] < max_gpu_temp)]
    filtered_count = initial_count - len(df)
    if filtered_count > 0:
        logger.info(
            f"Filtered {filtered_count} thermal-throttling samples "
            f"(T_cpu >= {max_cpu_temp} or T_gpu >= {max_gpu_temp})"
        )

    # 4. Normalize PWM to 0-100 range (from 0-255)
    for col in df.columns:
        if col.startswith("pwm"):
            df[col] = df[col] / 2.55

    # 5. Split into train/val
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=42)

    # 6. Save processed CSVs
    train_path = output_dir / "train.csv"
    val_path = output_dir / "val.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    logger.info(f"Saved {len(train_df)} training samples to {train_path}")
    logger.info(f"Saved {len(val_df)} validation samples to {val_path}")
