import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# Reproduce prepared dataset
df = pd.read_csv("data/one_meter.csv", parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

df["hour_of_day"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.day_of_week
df["lag_1"]   = df["electricity_kwh"].shift(1)
df["lag_2"]   = df["electricity_kwh"].shift(2)
df["lag_3"]   = df["electricity_kwh"].shift(3)
df["lag_24"]  = df["electricity_kwh"].shift(24)
df["lag_168"] = df["electricity_kwh"].shift(168)
df = df.dropna().reset_index(drop=True)

# Train/test split
FEATURES = ["hour_of_day", "day_of_week", "lag_1", "lag_2", "lag_3", "lag_24", "lag_168"]
TARGET   = "electricity_kwh"

test_cutoff = df["timestamp"].max() - pd.Timedelta(weeks=2)
train = df[df["timestamp"] <= test_cutoff].copy()
test  = df[df["timestamp"] >  test_cutoff].copy()

X_train, y_train = train[FEATURES], train[TARGET]
X_test,  y_test  = test[FEATURES],  test[TARGET]

# Train model
model = GradientBoostingRegressor(random_state=42)
model.fit(X_train, y_train)

# --- Model predictions ---
y_pred = model.predict(X_test)

# --- Baseline: predict using lag_1 (previous hour's consumption) ---
y_baseline = test["lag_1"].values

# --- Metrics ---
model_mae  = mean_absolute_error(y_test, y_pred)
model_rmse = root_mean_squared_error(y_test, y_pred)

base_mae   = mean_absolute_error(y_test, y_baseline)
base_rmse  = root_mean_squared_error(y_test, y_baseline)

print("=== Evaluation on 2-week test set (336 hours) ===")
print(f"\n{'':30s} {'MAE (kWh)':>12} {'RMSE (kWh)':>12}")
print("-" * 56)
print(f"{'Baseline (lag_1 = prev hour)':30s} {base_mae:>12.3f} {base_rmse:>12.3f}")
print(f"{'GradientBoostingRegressor':30s} {model_mae:>12.3f} {model_rmse:>12.3f}")
print("-" * 56)
print(f"{'Improvement':30s} {base_mae - model_mae:>+12.3f} {base_rmse - model_rmse:>+12.3f}")

print("\n=== Error distribution (model) ===")
errors = pd.Series(np.abs(y_pred - y_test.values))
print(errors.describe().round(3).to_string())

print("\n=== Worst 5 predictions ===")
test_copy = test.copy()
test_copy["predicted"] = y_pred
test_copy["abs_error"] = np.abs(y_pred - y_test.values)
worst = test_copy.nlargest(5, "abs_error")[["timestamp", "electricity_kwh", "predicted", "abs_error"]]
print(worst.round(2).to_string(index=False))
