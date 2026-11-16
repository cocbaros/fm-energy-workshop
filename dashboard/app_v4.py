"""
FM Energy Dashboard — v4 (live simulation + prediction + LLM analysis)
------------------------------------------------------------------------
Identical to v3, with the LLM Analysis placeholder replaced by a live panel.

The fixed prompt is:
  "Give me a summary of the energy usage for METER_02 for the past 2 days"

When Send is pressed:
  - The latest 48 hourly readings for METER_02 are taken from session state.
  - Total, average, min and max kWh are calculated in Python.
  - A JSON context block is built and attached to the prompt as the user
    message sent to OpenAI.
  - The response is stored in session state and survives simulation ticks.
  - A counter tracks how many LLM requests have been made this session.

OPENAI_API_KEY and OPENAI_MODEL are read from .env and never displayed.

Run from the project root:
    ../vibe/Scripts/streamlit.exe run dashboard/app_v4.py --server.headless true
"""

import json
import os
import re
import time
from pathlib import Path

import joblib
import openai
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import urllib3
from dotenv import load_dotenv

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FM Energy Dashboard v4",
    page_icon="⚡",
    layout="wide",
)

# ── Environment & paths ───────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
load_dotenv(BASE_DIR / ".env")

raw_url = os.environ["DATABASE_URL"].strip().strip('"')
conn_str = re.sub(r"[&?]channel_binding=require", "", raw_url)
host = re.search(r"@([^/]+)/", conn_str).group(1)
SQL_URL = f"https://{host}/sql"
NEON_HEADERS = {
    "Content-Type": "application/json",
    "Neon-Connection-String": conn_str,
}

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"].strip().strip('"')
OPENAI_MODEL   = os.environ.get("OPENAI_MODEL", "gpt-4o-mini").strip().strip('"')

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

MODEL_PATH        = BASE_DIR / "models" / "gbr_model.joblib"
FEATURE_ORDER_PATH = BASE_DIR / "models" / "feature_order.json"

LLM_METER   = "METER_02"
LLM_WINDOW  = 48          # hours
FIXED_PROMPT = f"Give me a summary of the energy usage for {LLM_METER} for the past 2 days"

LAG_MAX      = 168
SIM_DATA_END = pd.Timestamp("2016-02-29 23:00:00", tz="UTC")
TICK_SECONDS = 60

# ── Cached resources ──────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    with open(FEATURE_ORDER_PATH) as f:
        features = json.load(f)
    return model, features


@st.cache_resource
def get_openai_client():
    return openai.OpenAI(api_key=OPENAI_API_KEY)


# ── Neon helpers ──────────────────────────────────────────────────────────────
def run_sql(query: str, params: list | None = None) -> dict:
    payload = {"query": query}
    if params:
        payload["params"] = params
    resp = requests.post(SQL_URL, headers=NEON_HEADERS, json=payload,
                         timeout=60, verify=False)
    resp.raise_for_status()
    return resp.json()


def _to_df(rows: list) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(
            columns=["meter_id", "timestamp", "building_id", "electricity_kwh"]
        )
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


# ── Prediction helper ─────────────────────────────────────────────────────────
def predict_next_hour(meter_series: pd.Series, target_ts: pd.Timestamp,
                      model, features: list) -> tuple[float, str] | tuple[None, str]:
    n = len(meter_series)
    if n < LAG_MAX:
        return None, f"Need {LAG_MAX - n} more reading(s) (minimum {LAG_MAX} required for lag_168)."
    vals = meter_series.values
    row = {
        "hour_of_day": target_ts.hour,
        "day_of_week": target_ts.day_of_week,
        "lag_1":   vals[-1],
        "lag_2":   vals[-2],
        "lag_3":   vals[-3],
        "lag_24":  vals[-24],
        "lag_168": vals[-168],
    }
    X = pd.DataFrame([row])[features]
    return float(model.predict(X)[0]), ""


# ── LLM analysis helper ───────────────────────────────────────────────────────
def build_llm_payload(df: pd.DataFrame, meter_id: str, window: int) -> tuple[str, dict]:
    """
    Take the latest `window` readings for `meter_id` from df.
    Return (json_string, stats_dict).
    """
    sub = (
        df[df["meter_id"] == meter_id]
        .sort_values("timestamp")
        .tail(window)
        .reset_index(drop=True)
    )
    readings = [
        {"timestamp": row["timestamp"].strftime("%Y-%m-%d %H:%M UTC"),
         "electricity_kwh": round(row["electricity_kwh"], 4)}
        for _, row in sub.iterrows()
    ]
    kwh = sub["electricity_kwh"]
    stats = {
        "meter":      meter_id,
        "from":       sub["timestamp"].iloc[0].strftime("%Y-%m-%d %H:%M UTC"),
        "to":         sub["timestamp"].iloc[-1].strftime("%Y-%m-%d %H:%M UTC"),
        "n_readings": len(sub),
        "total_kwh":  round(float(kwh.sum()), 4),
        "avg_kwh":    round(float(kwh.mean()), 4),
        "min_kwh":    round(float(kwh.min()), 4),
        "max_kwh":    round(float(kwh.max()), 4),
        "readings":   readings,
    }
    return json.dumps(stats, indent=2), stats


def call_openai(prompt: str, json_context: str) -> str:
    client = get_openai_client()
    user_message = (
        f"{prompt}\n\n"
        f"Here is the data:\n```json\n{json_context}\n```"
    )
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an energy analyst. "
                    "Answer concisely based only on the data provided."
                ),
            },
            {"role": "user", "content": user_message},
        ],
    )
    return response.choices[0].message.content.strip()


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
    seed = fetch_seed()
    st.session_state["df"]           = seed.copy()
    st.session_state["last_ts"]      = seed["timestamp"].max()
    st.session_state["tick_due"]     = time.time() + TICK_SECONDS
    st.session_state["finished"]     = False
    st.session_state["predictions"]  = {}
    st.session_state["llm_response"] = None   # {"text": str, "as_of": ts, "json": str}
    st.session_state["llm_count"]    = 0
    st.session_state["initialised"]  = True


if not st.session_state.get("initialised"):
    init_state()

# ── Title & restart ───────────────────────────────────────────────────────────
st.title("⚡ FM Energy Dashboard — Live Simulation")
st.caption(
    "Historical hourly readings replayed in real time · "
    "source: `meter_readings_source` · 1 real minute = 1 simulated hour"
)

if st.button("🔄 Restart simulation"):
    fetch_seed.clear()
    init_state()
    st.rerun()

# ── Advance simulation tick ───────────────────────────────────────────────────
finished = st.session_state["finished"]
now = time.time()

if not finished and now >= st.session_state["tick_due"]:
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
            st.session_state["predictions"] = {}
        st.session_state["tick_due"] = time.time() + TICK_SECONDS

# ── Status bar ────────────────────────────────────────────────────────────────
df: pd.DataFrame = st.session_state["df"]
last_ts: pd.Timestamp = st.session_state["last_ts"]

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

# ── Prediction + LLM panels ───────────────────────────────────────────────────
st.divider()
pred_col, llm_col = st.columns(2)

# ── Prediction ────────────────────────────────────────────────────────────────
with pred_col:
    st.subheader("🔮 Prediction")

    if st.button("Predict next hour", type="primary"):
        model, features = load_model()
        target_ts = last_ts + pd.Timedelta(hours=1)
        results = {}
        for meter_id in meters:
            series = (
                df[df["meter_id"] == meter_id]
                .sort_values("timestamp")["electricity_kwh"]
                .reset_index(drop=True)
            )
            kwh, err = predict_next_hour(series, target_ts, model, features)
            results[meter_id] = {"kwh": kwh, "err": err, "target_ts": target_ts}
        st.session_state["predictions"] = results

    preds = st.session_state.get("predictions", {})
    if preds:
        sample = next(iter(preds.values()))
        st.caption(f"Target: **{sample['target_ts'].strftime('%d %b %Y %H:%M UTC')}**")
        pcols = st.columns(len(meters))
        for col, meter_id in zip(pcols, meters):
            p = preds[meter_id]
            with col:
                if p["kwh"] is not None:
                    st.metric(label=meter_id, value=f"{p['kwh']:.2f} kWh")
                else:
                    st.warning(f"**{meter_id}**\n\n{p['err']}", icon="⚠️")
    else:
        st.caption("Press the button to predict the next recorded hour for each meter.")

# ── LLM Analysis ──────────────────────────────────────────────────────────────
with llm_col:
    st.subheader("🤖 LLM Analysis")

    # Read-only prompt display
    st.text_input(
        label="Prompt",
        value=FIXED_PROMPT,
        disabled=True,
        label_visibility="collapsed",
    )

    # How many METER_02 readings are available
    m02_count = len(df[df["meter_id"] == LLM_METER])
    if m02_count < LLM_WINDOW:
        st.caption(
            f"⚠️ Only {m02_count} reading(s) available for {LLM_METER}; "
            f"{LLM_WINDOW} needed. Send will use all available readings."
        )

    send_clicked = st.button("Send", type="primary", key="llm_send")

    if send_clicked:
        with st.spinner("Calling OpenAI …"):
            try:
                json_str, _ = build_llm_payload(df, LLM_METER, LLM_WINDOW)
                response_text = call_openai(FIXED_PROMPT, json_str)
                st.session_state["llm_response"] = {
                    "text":   response_text,
                    "as_of":  last_ts,
                    "json":   json_str,
                }
                st.session_state["llm_count"] = st.session_state.get("llm_count", 0) + 1
            except Exception as exc:
                st.error(f"OpenAI call failed: {exc}")

    llm = st.session_state.get("llm_response")
    if llm:
        st.caption(
            f"Based on readings up to **{llm['as_of'].strftime('%d %b %Y %H:%M UTC')}** · "
            f"LLM requests this session: **{st.session_state['llm_count']}**"
        )
        st.markdown(llm["text"])
        with st.expander("Data sent to the LLM"):
            st.code(llm["json"], language="json")

# ── Auto-rerun while simulation is live ───────────────────────────────────────
if not finished:
    secs_left = max(1, int(st.session_state["tick_due"] - time.time()))
    sleep_ms = min(5000, max(1000, (secs_left - 1) * 1000))
    time.sleep(sleep_ms / 1000)
    st.rerun()
