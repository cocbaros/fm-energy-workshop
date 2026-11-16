import pandas as pd

# Load and reproduce the prepared dataset from Step 2
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

# --- Split: hold back last 2 weeks (336 hours) as test set ---
FEATURES = ["hour_of_day", "day_of_week", "lag_1", "lag_2", "lag_3", "lag_24", "lag_168"]
TARGET   = "electricity_kwh"

test_cutoff = df["timestamp"].max() - pd.Timedelta(weeks=2)
train = df[df["timestamp"] <= test_cutoff].copy()
test  = df[df["timestamp"] >  test_cutoff].copy()

X_train = train[FEATURES]
y_train = train[TARGET]
X_test  = test[FEATURES]
y_test  = test[TARGET]

print("=== Split summary ===")
print(f"Total rows      : {len(df)}")
print(f"Test cutoff     : {test_cutoff}")
print(f"Train rows      : {len(train)}  ({train['timestamp'].min().date()} → {train['timestamp'].max().date()})")
print(f"Test rows       : {len(test)}   ({test['timestamp'].min().date()} → {test['timestamp'].max().date()})")

print("\n=== Feature matrix shapes ===")
print(f"X_train : {X_train.shape}")
print(f"X_test  : {X_test.shape}")

print("\n=== No data leakage check ===")
overlap = set(train["timestamp"]) & set(test["timestamp"])
print(f"Overlapping timestamps between train and test: {len(overlap)}")
