import pandas as pd

# Load the data
df = pd.read_csv("data/one_meter.csv", parse_dates=["timestamp"])

print("=== Shape ===")
print(f"{df.shape[0]} rows, {df.shape[1]} columns")

print("\n=== Columns & types ===")
print(df.dtypes)

print("\n=== First 5 rows ===")
print(df.head())

print("\n=== Last 5 rows ===")
print(df.tail())

print("\n=== Basic statistics for electricity_kwh ===")
print(df["electricity_kwh"].describe().round(2))

print("\n=== Missing values ===")
print(df.isnull().sum())

print("\n=== Unique meter IDs ===")
print(df["meter_id"].unique())

print("\n=== Unique building IDs ===")
print(df["building_id"].unique())

print("\n=== Date range ===")
print(f"From : {df['timestamp'].min()}")
print(f"To   : {df['timestamp'].max()}")
total_hours = int((df["timestamp"].max() - df["timestamp"].min()).total_seconds() / 3600) + 1
print(f"Expected hours in range : {total_hours}")
print(f"Actual rows             : {df.shape[0]}")
print(f"Missing hours           : {total_hours - df.shape[0]}")

print("\n=== Hour spacing check (any gaps > 1 hour?) ===")
gaps = df["timestamp"].diff().dropna()
irregular = gaps[gaps != pd.Timedelta(hours=1)]
if irregular.empty:
    print("No gaps found — all readings are exactly 1 hour apart.")
else:
    print(f"{len(irregular)} irregular gaps found:")
    print(irregular)
