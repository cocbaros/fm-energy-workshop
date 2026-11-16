"""
FM Energy Dashboard — v2 (live simulation)
-------------------------------------------
Starts with January 2016 data for all three meters.
Every real minute it fetches the next recorded hour from Neon (one reading
per meter) and appends it to the display, updating all charts live.
When the last February record is consumed, the refresh stops.
Press "Restart simulation" to reset to January and begin again.

Run from the project root:
    ../vibe/Scripts/streamlit.exe run dashboard/app_v2.py --server.headless true
"""

import os
import re
import time
import urllib3
import requests
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FM Energy Dashboard v2",
    page_icon="⚡",
    layout="wide",
)

# ── Environment ───────────────────────────────────────────────────────────────
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

raw_url = os.environ["DATABASE_URL"].strip().strip('"')
conn_str = re.sub(r"[&?]channel_binding=require", "", raw_url)
host = re.search(r"@([^/]+)/", conn_str).group(1)
SQL_URL = f"https://{host}/sql"
HEADERS = {
    "Content-Type": "application/json",
    "Neon-Connection-String": conn_str,
}

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Simulation boundaries
SIM_START = pd.Timestamp("2016-01-01 00:00:00", tz="UTC")  # first timestamp shown
SIM_SEED_END = pd.Timestamp("2016-01-31 23:00:00", tz="UTC")  # last seed timestamp
SIM_DATA_END = pd.Timestamp("2016-02-29 23:00:00", tz="UTC")  # last available timestamp

TICK_SECONDS = 60  # one real minute per simulated hour


# ── Neon helpers ──────────────────────────────────────────────────────────────
def run_sql(query: str, params: list | None = None) -> dict:
    payload = {"query": query}
    if params:
        payload["params"] = params
    resp = requests.post(SQL_URL, headers=HEADERS, json=payload,
                         timeout=60, verify=False)
    resp.raise_for_status()
    return resp.json()


def _to_df(rows: list) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["meter_id", "timestamp", "building_id", "electricity_kwh"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["electricity_kwh"] = df["electricity_kwh"].astype(float)
    return df


@st.cache_data(ttl=600, show_spinner="Loading January seed data …")
def fetch_seed() -> pd.DataFrame:
    result = run_sql(
        """
        SELECT meter_id,
               timestamp AT TIME ZONE 'UTC' AS timestamp,
               building_id,
               electricity_kwh
        FROM   meter_readings_source
        WHERE  timestamp >= $1
          AND  timestamp <= $2
        ORDER  BY timestamp, meter_id
        """,
        ["2016-01-01 00:00:00+00", "2016-01-31 23:00:00+00"],
    )
    return _to_df(result["rows"])


def fetch_next_hour(after_ts: pd.Timestamp) -> pd.DataFrame:
    """Return the three meter rows for the first hour strictly after after_ts."""
    ts_str = after_ts.strftime("%Y-%m-%d %H:%M:%S+00")
    result = run_sql(
        """
        SELECT meter_id,
               timestamp AT TIME ZONE 'UTC' AS timestamp,
               building_id,
               electricity_kwh
        FROM   meter_readings_source
        WHERE  timestamp = (
                   SELECT MIN(timestamp)
                   FROM   meter_readings_source
                   WHERE  timestamp > $1
               )
        ORDER  BY meter_id
        """,
        [ts_str],
    )
    return _to_df(result["rows"])


# ── Chart builders ────────────────────────────────────────────────────────────
METER_COLOURS = {
    "METER_01": "#00b4d8",
    "METER_02": "#f77f00",
    "METER_03": "#06d6a0",
}
TOTAL_COLOUR = "#9b5de5"

_COMMON_LAYOUT = dict(
    xaxis=dict(showgrid=False, tickformat="%d %b"),
    yaxis=dict(title="kWh", gridcolor="#2a2a2a"),
    plot_bgcolor="#0e1117",
    paper_bgcolor="#0e1117",
    font=dict(color="#fafafa"),
    margin=dict(l=50, r=20, t=40, b=40),
)


def meter_chart(df_meter: pd.DataFrame, meter_id: str, height: int = 280) -> go.Figure:
    colour = METER_COLOURS.get(meter_id, "#888888")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_meter["timestamp"],
        y=df_meter["electricity_kwh"],
        mode="lines",
        line=dict(color=colour, width=1.5),
        name=meter_id,
        hovertemplate="%{x|%d %b %H:%M}<br>%{y:.2f} kWh<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=meter_id, font=dict(size=14)),
        height=height,
        **_COMMON_LAYOUT,
    )
    return fig


def total_chart(df_total: pd.DataFrame, height: int = 320) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_total["timestamp"],
        y=df_total["electricity_kwh"],
        mode="lines",
        fill="tozeroy",
        line=dict(color=TOTAL_COLOUR, width=2),
        fillcolor="rgba(155,93,229,0.15)",
        name="Total",
        hovertemplate="%{x|%d %b %H:%M}<br>%{y:.2f} kWh<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Total — all meters combined", font=dict(size=14)),
        height=height,
        **_COMMON_LAYOUT,
    )
    return fig


# ── Session-state initialisation ──────────────────────────────────────────────
def init_state():
    """Populate session state from the seed data on first run or after restart."""
    seed = fetch_seed()
    st.session_state["df"] = seed.copy()
    st.session_state["last_ts"] = seed["timestamp"].max()
    st.session_state["tick_due"] = time.time() + TICK_SECONDS
    st.session_state["finished"] = False
    st.session_state["initialised"] = True


if not st.session_state.get("initialised"):
    init_state()

# ── Restart button ────────────────────────────────────────────────────────────
st.title("⚡ FM Energy Dashboard — Live Simulation")
st.caption(
    "Historical hourly readings replayed in real time · "
    "source: `meter_readings_source` · 1 real minute = 1 simulated hour"
)

if st.button("🔄 Restart simulation"):
    # Clear seed cache so a fresh DB fetch happens on restart
    fetch_seed.clear()
    init_state()
    st.rerun()

# ── Advance simulation if a tick is due ──────────────────────────────────────
finished = st.session_state["finished"]
now = time.time()
tick_due = st.session_state["tick_due"]

if not finished and now >= tick_due:
    last_ts = st.session_state["last_ts"]
    if last_ts >= SIM_DATA_END:
        st.session_state["finished"] = True
        finished = True
    else:
        new_rows = fetch_next_hour(last_ts)
        if not new_rows.empty:
            st.session_state["df"] = pd.concat(
                [st.session_state["df"], new_rows], ignore_index=True
            )
            st.session_state["last_ts"] = new_rows["timestamp"].max()
            if st.session_state["last_ts"] >= SIM_DATA_END:
                st.session_state["finished"] = True
                finished = True
        st.session_state["tick_due"] = time.time() + TICK_SECONDS

# ── Status bar ────────────────────────────────────────────────────────────────
df: pd.DataFrame = st.session_state["df"]
last_ts: pd.Timestamp = st.session_state["last_ts"]

status_col, next_col = st.columns([3, 1])
with status_col:
    if finished:
        st.success("✅ Simulation complete — all February data loaded. Press Restart to replay.")
    else:
        secs_left = max(0, int(st.session_state["tick_due"] - time.time()))
        st.info(
            f"🕐 Showing data up to **{last_ts.strftime('%d %b %Y %H:%M UTC')}** · "
            f"next fetch in **{secs_left}s**"
        )

# ── Charts ────────────────────────────────────────────────────────────────────
meters = sorted(df["meter_id"].unique())

cols = st.columns(3)
for col, meter_id in zip(cols, meters):
    with col:
        st.plotly_chart(
            meter_chart(df[df["meter_id"] == meter_id], meter_id),
            use_container_width=True,
        )

df_total = (
    df.groupby("timestamp", as_index=False)["electricity_kwh"]
    .sum()
    .sort_values("timestamp")
)
st.plotly_chart(total_chart(df_total), use_container_width=True)

# ── Stats strip ───────────────────────────────────────────────────────────────
st.divider()
stat_cols = st.columns(len(meters) + 1)
for col, meter_id in zip(stat_cols, meters):
    sub = df[df["meter_id"] == meter_id]
    with col:
        st.metric(
            label=meter_id,
            value=f"{sub['electricity_kwh'].sum():,.0f} kWh",
            delta=f"{len(sub):,} readings",
            delta_color="off",
        )
with stat_cols[-1]:
    st.metric(
        label="Combined total",
        value=f"{df['electricity_kwh'].sum():,.0f} kWh",
        delta=(
            f"{df['timestamp'].min().strftime('%d %b')} – "
            f"{last_ts.strftime('%d %b %Y')}"
        ),
        delta_color="off",
    )

# ── Placeholders ──────────────────────────────────────────────────────────────
st.divider()
pred_col, llm_col = st.columns(2)

with pred_col:
    st.subheader("🔮 Prediction")
    st.info("Next-24-hour forecasts will appear here.", icon="🚧")

with llm_col:
    st.subheader("🤖 LLM Analysis")
    st.info("AI-generated energy usage analysis will appear here.", icon="🚧")

# ── Auto-rerun while simulation is running ────────────────────────────────────
if not finished:
    secs_left = max(1, int(st.session_state["tick_due"] - time.time()))
    # Rerun 1 second before the tick is due so the countdown stays accurate,
    # but at most every 5 seconds to avoid hammering the browser.
    sleep_ms = min(5000, max(1000, (secs_left - 1) * 1000))
    time.sleep(sleep_ms / 1000)
    st.rerun()
