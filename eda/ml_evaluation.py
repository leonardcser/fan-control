import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import os
import glob

# 1. Load Data
search_path = "data/fan_control_20260119_040722/*.csv"
files = sorted(glob.glob(search_path))

print(f"Searching: {search_path}")
df_list = []
for f in files:
    try:
        data = pd.read_csv(f)
        data["source_file"] = os.path.basename(f)
        df_list.append(data)
    except Exception as e:
        print(f"Error loading {f}: {e}")

if not df_list:
    print("No data loaded!")
    exit(1)

df = pd.concat(df_list, ignore_index=True)
print(f"Total merged samples: {len(df)}")

# Label regimes
df["regime"] = np.where(
    df["source_file"].str.contains("094800"), "HighFlux", "Standard"
)

# 2. Feature Engineering
# Inverse PWM
for pwm in ["pwm2", "pwm4", "pwm5", "pwm7"]:
    df[f"inv_{pwm}"] = 1.0 / (df[pwm] + 10.0)

# Interaction terms
df["P_cpu_pwm2"] = df["P_cpu"] * df["inv_pwm2"]
df["P_cpu_pwm7"] = df["P_cpu"] * df["inv_pwm7"]

# Feature selection
feature_cols = [
    "P_cpu",
    "T_amb",
    "pwm2",
    "pwm4",
    "pwm5",
    "pwm7",
    "inv_pwm2",
    "inv_pwm4",
    "inv_pwm5",
    "inv_pwm7",
    "P_cpu_pwm2",
    "P_cpu_pwm7",
]
X = df[feature_cols]
y = df["T_cpu"]

# 3. Model Definition
models = {
    "Linear Regression": LinearRegression(),
    "Ridge (Alpha 1.0)": Ridge(alpha=1.0),
    "Polynomial (Deg 2)": make_pipeline(PolynomialFeatures(2), LinearRegression()),
    "SVR (RBF)": make_pipeline(StandardScaler(), SVR(kernel="rbf", C=100, epsilon=0.1)),
    "KNN (k=5)": make_pipeline(
        StandardScaler(), KNeighborsRegressor(n_neighbors=5, weights="distance")
    ),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42
    ),
    "XGBoost": xgb.XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42, n_jobs=1
    ),
    "MLP (Neural Net)": make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42),
    ),
}

print("\n--- Cross-Validation Evaluation (5-Fold) ---")
results = []

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    # Negative MSE is the score, so negate it for RMSE
    cv_scores = cross_val_score(
        model, X, y, cv=kf, scoring="neg_root_mean_squared_error"
    )
    rmse_scores = -cv_scores
    avg_rmse = np.mean(rmse_scores)
    std_rmse = np.std(rmse_scores)

    # Train on full set for final R2 check
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)

    # Check max error specifically on HighFlux samples
    high_flux_idx = df["regime"] == "HighFlux"
    if high_flux_idx.any():
        y_hf = y[high_flux_idx]
        y_hf_pred = model.predict(X[high_flux_idx])
        max_hf_error = np.max(np.abs(y_hf - y_hf_pred))
        hf_rmse = np.sqrt(mean_squared_error(y_hf, y_hf_pred))
    else:
        max_hf_error = 0.0
        hf_rmse = 0.0

    results.append(
        {
            "Model": name,
            "CV RMSE": avg_rmse,
            "Std RMSE": std_rmse,
            "Full R2": r2,
            "HighFlux RMSE": hf_rmse,
            "Max HighFlux Error": max_hf_error,
        }
    )

# Convert to DataFrame for display
results_df = pd.DataFrame(results).sort_values(by="CV RMSE")
print(results_df.to_string(index=False, float_format=lambda x: "{:.4f}".format(x)))

print("\n--- Best Model Details ---")
best_model_name = results_df.iloc[0]["Model"]
print(f"Winner: {best_model_name}")

best_model = models[best_model_name]
# Fit again on standard split to show specific failure cases if any
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
best_model.fit(X_train, y_train)
y_pred_test = best_model.predict(X_test)
print(f"Test Set RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_test)):.4f}")
