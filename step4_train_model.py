import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

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

X_train = train[FEATURES]
y_train = train[TARGET]

# Train the model
model = GradientBoostingRegressor(random_state=42)
model.fit(X_train, y_train)

print("=== Model trained ===")
print(f"Algorithm      : GradientBoostingRegressor")
print(f"random_state   : 42")
print(f"Training rows  : {len(X_train)}")
print(f"Features used  : {FEATURES}")

print("\n=== Model parameters (defaults) ===")
params = model.get_params()
for k, v in params.items():
    print(f"  {k}: {v}")

print("\n=== Feature importances ===")
importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
for feat, imp in importances.items():
    bar = "█" * int(imp * 50)
    print(f"  {feat:<12} {imp:.4f}  {bar}")
