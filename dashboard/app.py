"""
FM Energy Dashboard
-------------------
Historical hourly electricity readings from three meters, served from Neon
PostgreSQL via the HTTP SQL API.

Run from the project root:
    ../vibe/Scripts/streamlit.exe run dashboard/app.py --server.headless true
"""

import os
import re
import urllib3
import requests
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FM Energy Dashboard",
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


# ── Data fetching ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner="Fetching historical readings …")
def load_data() -> pd.DataFrame:
    payload = {
        "query": """
            SELECT
                meter_id,
                timestamp AT TIME ZONE 'UTC' AS timestamp,
                building_id,
                electricity_kwh
            FROM meter_readings_source
            ORDER BY timestamp
        """
    }
    resp = requests.post(SQL_URL, headers=HEADERS, json=payload,
                         timeout=60, verify=False)  # 60 s — allows for Neon cold-start
    resp.raise_for_status()
    result = resp.json()
    df = pd.DataFrame(result["rows"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["electricity_kwh"] = df["electricity_kwh"].astype(float)
    return df


# ── Colour palette (one per meter) ───────────────────────────────────────────
METER_COLOURS = {
    "METER_01": "#00b4d8",
    "METER_02": "#f77f00",
    "METER_03": "#06d6a0",
}
TOTAL_COLOUR = "#9b5de5"


def line_chart(df_meter: pd.DataFrame, meter_id: str) -> go.Figure:
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
        xaxis=dict(showgrid=False, tickformat="%d %b"),
        yaxis=dict(title="kWh", gridcolor="#2a2a2a"),
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
        margin=dict(l=50, r=20, t=40, b=40),
        height=280,
    )
    return fig


def total_chart(df_total: pd.DataFrame) -> go.Figure:
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
        xaxis=dict(showgrid=False, tickformat="%d %b"),
        yaxis=dict(title="kWh", gridcolor="#2a2a2a"),
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
        margin=dict(l=50, r=20, t=40, b=40),
        height=320,
    )
    return fig


# ── Layout ────────────────────────────────────────────────────────────────────
st.title("⚡ FM Energy Dashboard")
st.caption("Historical hourly electricity readings · source: `meter_readings_source`")

df = load_data()

meters = sorted(df["meter_id"].unique())

# Row 1 — three individual meters side by side
cols = st.columns(3)
for col, meter_id in zip(cols, meters):
    with col:
        st.plotly_chart(
            line_chart(df[df["meter_id"] == meter_id], meter_id),
            use_container_width=True,
        )

# Row 2 — combined total
df_total = (
    df.groupby("timestamp", as_index=False)["electricity_kwh"]
    .sum()
    .sort_values("timestamp")
)
st.plotly_chart(total_chart(df_total), use_container_width=True)

# ── Stats strip ──────────────────────────────────────────────────────────────
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
        delta=f"{df['timestamp'].min().strftime('%d %b')} – {df['timestamp'].max().strftime('%d %b %Y')}",
        delta_color="off",
    )

# ── Placeholders ─────────────────────────────────────────────────────────────
st.divider()
pred_col, llm_col = st.columns(2)

with pred_col:
    st.subheader("🔮 Prediction")
    st.info("Next-24-hour forecasts will appear here.", icon="🚧")

with llm_col:
    st.subheader("🤖 LLM Analysis")
    st.info("AI-generated energy usage analysis will appear here.", icon="🚧")
