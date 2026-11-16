import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, saves to file
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import joblib

# ── Reproduce prepared dataset ───────────────────────────────────────────────
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

FEATURES = ["hour_of_day", "day_of_week", "lag_1", "lag_2", "lag_3", "lag_24", "lag_168"]
TARGET   = "electricity_kwh"

test_cutoff = df["timestamp"].max() - pd.Timedelta(weeks=2)
train = df[df["timestamp"] <= test_cutoff].copy()
test  = df[df["timestamp"] >  test_cutoff].copy()

X_train, y_train = train[FEATURES], train[TARGET]
X_test,  y_test  = test[FEATURES],  test[TARGET]

# ── Train model ───────────────────────────────────────────────────────────────
model = GradientBoostingRegressor(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ── Plot actual vs predicted ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))

ax.plot(test["timestamp"], y_test.values, label="Actual", color="#1f77b4", linewidth=1.2)
ax.plot(test["timestamp"], y_pred,        label="Predicted", color="#ff7f0e",
        linewidth=1.0, linestyle="--", alpha=0.85)

ax.set_title("Actual vs Predicted — 2-week test window (METER_01)", fontsize=13)
ax.set_xlabel("Date")
ax.set_ylabel("Electricity (kWh)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
fig.autofmt_xdate()
ax.legend()
ax.grid(True, alpha=0.3)

plot_path = "models/actual_vs_predicted.png"
fig.tight_layout()
fig.savefig(plot_path, dpi=150)
plt.close()
print(f"Plot saved → {plot_path}")

# ── Save model and feature order ──────────────────────────────────────────────
model_path   = "models/gbr_model.joblib"
feature_path = "models/feature_order.json"

joblib.dump(model, model_path)
print(f"Model saved → {model_path}")

with open(feature_path, "w") as f:
    json.dump(FEATURES, f, indent=2)
print(f"Feature order saved → {feature_path}")

# ── Final summary ─────────────────────────────────────────────────────────────
mae  = mean_absolute_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)

print("\n=== Final model performance on test set ===")
print(f"  MAE  : {mae:.3f} kWh")
print(f"  RMSE : {rmse:.3f} kWh")

print("\n=== Saved artefacts ===")
import os
for p in [model_path, feature_path, plot_path]:
    size = os.path.getsize(p)
    print(f"  {p}  ({size:,} bytes)")
