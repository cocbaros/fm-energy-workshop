"""
Create meter_readings_source table in Neon and upload three_meters.csv.
Uses the Neon HTTP SQL API to avoid port-5432 firewall issues.
"""

import csv
import json
import os
import re
import sys
from pathlib import Path

import urllib3
import requests
from dotenv import load_dotenv

# Corporate proxy uses a self-signed cert in the chain; suppress the warning.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")

raw_url = os.environ["DATABASE_URL"].strip().strip('"')
# Strip channel_binding=require (Neon HTTP API does not need it)
conn_str = re.sub(r"[&?]channel_binding=require", "", raw_url)

# Derive HTTP SQL endpoint from the host in the connection string
host_match = re.search(r"@([^/]+)/", conn_str)
if not host_match:
    sys.exit("Cannot parse host from DATABASE_URL")
host = host_match.group(1)
SQL_URL = f"https://{host}/sql"

HEADERS = {
    "Content-Type": "application/json",
    "Neon-Connection-String": conn_str,
}

CSV_PATH = BASE_DIR / "data" / "three_meters.csv"
BATCH_SIZE = 500


# ── Helpers ───────────────────────────────────────────────────────────────────
def run_sql(query: str, params: list | None = None):
    payload = {"query": query}
    if params:
        payload["params"] = params
    resp = requests.post(SQL_URL, headers=HEADERS, json=payload, timeout=30, verify=False)
    resp.raise_for_status()
    return resp.json()


# ── Create table ──────────────────────────────────────────────────────────────
print("Creating table meter_readings_source …")
run_sql("""
CREATE TABLE IF NOT EXISTS meter_readings_source (
    meter_id       TEXT        NOT NULL,
    timestamp      TIMESTAMPTZ NOT NULL,
    building_id    TEXT        NOT NULL,
    electricity_kwh NUMERIC(10, 4) NOT NULL,
    CONSTRAINT meter_readings_source_pk PRIMARY KEY (meter_id, timestamp)
)
""")
print("  Table ready.")

# ── Load CSV and upload in batches ────────────────────────────────────────────
print(f"Reading {CSV_PATH} …")
with open(CSV_PATH, newline="", encoding="utf-8") as fh:
    rows = list(csv.DictReader(fh))

print(f"  {len(rows)} rows to upload.")

uploaded = 0
for start in range(0, len(rows), BATCH_SIZE):
    batch = rows[start : start + BATCH_SIZE]

    # Build a multi-row INSERT with positional parameters
    placeholders = []
    params = []
    for i, row in enumerate(batch):
        base = i * 4
        placeholders.append(f"(${base+1}, ${base+2}, ${base+3}, ${base+4})")
        params.extend([
            row["meter_id"],
            row["timestamp"],
            row["building_id"],
            row["electricity_kwh"],
        ])

    sql = (
        "INSERT INTO meter_readings_source "
        "(meter_id, timestamp, building_id, electricity_kwh) VALUES "
        + ", ".join(placeholders)
        + " ON CONFLICT (meter_id, timestamp) DO NOTHING"
    )

    run_sql(sql, params)
    uploaded += len(batch)
    print(f"  Uploaded {uploaded}/{len(rows)} rows …", end="\r")

print(f"\nDone. {uploaded} rows sent.")

# ── Summary query ─────────────────────────────────────────────────────────────
print("\nSummary per meter:")
result = run_sql("""
SELECT
    meter_id,
    COUNT(*)                     AS row_count,
    MIN(timestamp)::TEXT         AS earliest,
    MAX(timestamp)::TEXT         AS latest
FROM meter_readings_source
GROUP BY meter_id
ORDER BY meter_id
""")

rows_out = result.get("rows", [])
if rows_out:
    # Header
    print(f"  {'meter_id':<12} {'rows':>6}  {'earliest':<25} {'latest':<25}")
    print(f"  {'-'*12} {'-'*6}  {'-'*25} {'-'*25}")
    for r in rows_out:
        print(f"  {r['meter_id']:<12} {r['row_count']:>6}  {r['earliest']:<25} {r['latest']:<25}")
else:
    print("  (no rows returned)")
