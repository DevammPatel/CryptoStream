"""CryptoStream live dashboard (Streamlit).

Shows, per symbol: price + VWAP candles, technical indicators, the ML model's
next-move prediction (via the FastAPI service), and the LLM market summary.

Run:  streamlit run dashboard/app.py
"""
from __future__ import annotations

import os
import sys

import duckdb
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

DUCKDB_PATH = os.getenv("DUCKDB_PATH", os.path.join(REPO, "data", "warehouse", "crypto.duckdb"))
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="CryptoStream", page_icon="📈", layout="wide")


@st.cache_data(ttl=10)
def load_symbols() -> list[str]:
    if not os.path.exists(DUCKDB_PATH):
        return []
    con = duckdb.connect(DUCKDB_PATH, read_only=True)
    try:
        tbl = con.execute(
            "select count(*) from information_schema.tables "
            "where table_name in ('int_technical_features','ohlcv_10s')"
        ).fetchone()[0]
        if not tbl:
            return []
        try:
            df = con.execute("select distinct symbol from intermediate.int_technical_features").df()
        except Exception:
            df = con.execute("select distinct symbol from raw.ohlcv_10s").df()
    finally:
        con.close()
    return sorted(df["symbol"].tolist())


@st.cache_data(ttl=10)
def load_bars(symbol: str, limit: int = 300) -> pd.DataFrame:
    con = duckdb.connect(DUCKDB_PATH, read_only=True)
    try:
        try:
            df = con.execute(
                "select window_start, open, high, low, close, vwap, volume, "
                "volatility_30, px_vs_sma30, order_flow_imbalance, rel_volume "
                "from intermediate.int_technical_features where symbol = ? "
                "order by window_start desc limit ?",
                [symbol, limit],
            ).df()
        except Exception:
            df = con.execute(
                "select window_start, open, high, low, close, vwap, volume "
                "from raw.ohlcv_10s where symbol = ? order by window_start desc limit ?",
                [symbol, limit],
            ).df()
    finally:
        con.close()
    return df.sort_values("window_start").reset_index(drop=True)


def get_prediction(symbol: str) -> dict | None:
    try:
        r = requests.get(f"{API_URL}/predict/{symbol}", timeout=3)
        if r.ok:
            return r.json()
    except requests.RequestException:
        return None
    return None


def get_summary() -> str:
    from llm.market_agent import generate_market_summary

    return generate_market_summary()


# ------------------------------ UI ------------------------------
st.title("📈 CryptoStream — Real-Time Crypto Analytics & Prediction")

symbols = load_symbols()
if not symbols:
    st.warning(
        "No data yet. Run `python scripts/generate_sample_data.py` (offline demo) "
        "or start the live stack with `make up`."
    )
    st.stop()

with st.sidebar:
    st.header("Controls")
    symbol = st.selectbox("Symbol", symbols, index=0)
    st.caption(f"Serving API: {API_URL}")
    if st.button("🔄 Refresh"):
        st.cache_data.clear()
        st.rerun()

bars = load_bars(symbol)
latest = bars.iloc[-1]

# KPI row
c1, c2, c3, c4 = st.columns(4)
c1.metric("Last price", f"${latest['close']:,.2f}")
if "px_vs_sma30" in bars:
    c2.metric("vs SMA-30", f"{latest['px_vs_sma30']*100:+.2f}%")
    c3.metric("Volatility (30)", f"{latest['volatility_30']*100:.3f}%")
    c4.metric("Order-flow imbalance", f"{latest['order_flow_imbalance']:+.2f}")

# Prediction banner
pred = get_prediction(symbol)
if pred:
    arrow = "🟢 ▲ UP" if pred["direction"] == "up" else "🔴 ▼ DOWN"
    st.subheader(f"Model call: {arrow}  ·  P(up)={pred['prob_up']:.2f}  ·  confidence={pred['confidence']:.2f}")
else:
    st.info("Prediction service unavailable — train the model and start the API (`make api`).")

# Candles + VWAP
fig = go.Figure()
fig.add_trace(
    go.Candlestick(
        x=bars["window_start"], open=bars["open"], high=bars["high"],
        low=bars["low"], close=bars["close"], name="Price",
    )
)
if "vwap" in bars:
    fig.add_trace(go.Scatter(x=bars["window_start"], y=bars["vwap"], name="VWAP", line=dict(width=1)))
fig.update_layout(height=430, margin=dict(l=10, r=10, t=30, b=10), xaxis_rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)

# Volume + LLM summary
left, right = st.columns([2, 1])
with left:
    vfig = go.Figure(go.Bar(x=bars["window_start"], y=bars["volume"], name="Volume"))
    vfig.update_layout(height=220, margin=dict(l=10, r=10, t=30, b=10), title="Volume")
    st.plotly_chart(vfig, use_container_width=True)
with right:
    st.markdown("### 🧠 Market summary")
    try:
        st.markdown(get_summary())
    except Exception as exc:
        st.caption(f"Summary unavailable: {exc}")

st.caption("Data: Binance public WebSocket · Pipeline: Redpanda → Spark → MinIO → dbt/DuckDB → LightGBM → FastAPI")
