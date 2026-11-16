import pandas as pd

# Load the data
df = pd.read_csv("data/one_meter.csv", parse_dates=["timestamp"])

# Sort by time (already sorted, but be explicit)
df = df.sort_values("timestamp").reset_index(drop=True)

# --- Time features ---
df["hour_of_day"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.day_of_week  # Monday=0, Sunday=6

# --- Lag features ---
df["lag_1"]   = df["electricity_kwh"].shift(1)    # 1 hour ago
df["lag_2"]   = df["electricity_kwh"].shift(2)    # 2 hours ago
df["lag_3"]   = df["electricity_kwh"].shift(3)    # 3 hours ago
df["lag_24"]  = df["electricity_kwh"].shift(24)   # same hour yesterday
df["lag_168"] = df["electricity_kwh"].shift(168)  # same hour last week

# --- Drop rows that don't have full history (first 168 rows) ---
df = df.dropna().reset_index(drop=True)

print("=== Shape after feature engineering ===")
print(f"{df.shape[0]} rows, {df.shape[1]} columns")

print("\n=== Columns ===")
print(df.columns.tolist())

print("\n=== First 3 rows ===")
print(df[["timestamp", "electricity_kwh", "hour_of_day", "day_of_week",
          "lag_1", "lag_2", "lag_3", "lag_24", "lag_168"]].head(3).to_string())

print("\n=== Last 3 rows ===")
print(df[["timestamp", "electricity_kwh", "hour_of_day", "day_of_week",
          "lag_1", "lag_2", "lag_3", "lag_24", "lag_168"]].tail(3).to_string())

print("\n=== Missing values after drop ===")
print(df.isnull().sum())

print(f"\nRows dropped (first 168 lacking full history): 168")
print(f"Rows remaining for modelling: {df.shape[0]}")
