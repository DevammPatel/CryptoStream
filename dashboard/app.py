"""CryptoStream live dashboard (Streamlit).

Shows, per symbol: price + VWAP candles, technical indicators, the ML model's
next-move prediction (via the FastAPI service), and the market summary that Dagster
saves after each refresh.

Bars come from the API's live feed and refresh every 5 seconds. When the feed isn't
available (offline demo, or the API is down) they come from the DuckDB warehouse.

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
    if not os.path.exists(DUCKDB_PATH):
        return pd.DataFrame()
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


def api_get(path: str):
    try:
        r = requests.get(f"{API_URL}{path}", timeout=3)
        if r.ok:
            return r.json()
    except requests.RequestException:
        return None
    return None


def load_live_bars(symbol: str) -> pd.DataFrame | None:
    """Bars the API holds in memory, including the one still forming."""
    rows = api_get(f"/bars/{symbol}?limit=300")
    if not rows:
        return None
    df = pd.DataFrame(rows).astype({"px_vs_sma30": float, "volatility_30": float, "rel_volume": float})
    df["window_start"] = pd.to_datetime(df["window_start"], unit="ms", utc=True)
    return df


def available_symbols() -> list[str]:
    live = (api_get("/symbols") or {}).get("symbols", [])
    return sorted(set(load_symbols()) | set(live))


@st.cache_data(ttl=30)
def get_summary() -> dict:
    """The summary Dagster saved after its last refresh.

    The page never calls the LLM itself. Before Dagster's first run it shows the free
    offline template instead.
    """
    from llm.market_agent import build_market_summary, load_saved_summary

    return load_saved_summary() or {**build_market_summary(use_llm=False), "built_here": True}


def summary_caption(record: dict) -> str:
    if record.get("built_here"):
        return "Offline template built by this page: Dagster hasn't saved a summary yet."
    written = pd.Timestamp(record["generated_at"])
    minutes = (pd.Timestamp.now(tz="UTC") - written).total_seconds() / 60
    by = "the LLM" if record["source"] == "llm" else "the offline template"
    note = f"Written by {by} at {written:%H:%M} UTC ({minutes:.0f} min ago); Dagster refreshes it every 15 minutes."
    if record.get("llm_error"):
        note += f" The LLM call failed: {record['llm_error']}"
    return note


@st.fragment(run_every="5s")
def wait_for_data() -> None:
    st.warning(
        "No data yet. Run `python scripts/generate_sample_data.py` (offline demo) "
        "or start the live stack with `make up`."
    )
    if available_symbols():
        st.rerun()


@st.fragment(run_every="5s")
def live_view(symbol: str) -> None:
    """Everything that moves. Only this part reruns on the timer."""
    bars = load_live_bars(symbol)
    source = "Live stream"
    if bars is None:
        bars = load_bars(symbol)
        source = "Warehouse (refreshes every 15 minutes)"
    if bars.empty:
        st.info("Waiting for bars.")
        return
    latest = bars.iloc[-1]
    st.caption(f"Source: {source}")

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Last price", f"${latest['close']:,.2f}")
    if "px_vs_sma30" in bars:
        c2.metric("vs SMA-30", f"{latest['px_vs_sma30']*100:+.2f}%")
        c3.metric("Volatility (30)", f"{latest['volatility_30']*100:.3f}%")
        c4.metric("Order-flow imbalance", f"{latest['order_flow_imbalance']:+.2f}")

    # Prediction banner
    pred = api_get(f"/predict/{symbol}")
    if pred:
        arrow = "🟢 ▲ UP" if pred["direction"] == "up" else "🔴 ▼ DOWN"
        st.subheader(f"Model call: {arrow}  ·  P(up)={pred['prob_up']:.2f}  ·  confidence={pred['confidence']:.2f}")
        if pred.get("as_of"):
            age = (pd.Timestamp.now(tz="UTC") - pd.Timestamp(pred["as_of"])).total_seconds()
            st.caption(f"Based on the {pred['source']} bar from {age:.0f}s ago")
        track = (api_get("/model/live") or {}).get("current_model", {})
        if track.get("n"):
            st.caption(
                f"Live track record of this model: {track['n']:,} calls on bars it never trained on, "
                f"right {track['hit_rate']:.1%} of the time (always calling the more common side: "
                f"{track['hit_rate_baseline']:.1%}); Brier {track['brier']:.4f} vs {track['brier_baseline']:.4f}."
            )
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

    vfig = go.Figure(go.Bar(x=bars["window_start"], y=bars["volume"], name="Volume"))
    vfig.update_layout(height=220, margin=dict(l=10, r=10, t=30, b=10), title="Volume")
    st.plotly_chart(vfig, use_container_width=True)


# ------------------------------ UI ------------------------------
st.title("📈 CryptoStream — Real-Time Crypto Analytics & Prediction")

symbols = available_symbols()
if not symbols:
    wait_for_data()
    st.stop()

with st.sidebar:
    st.header("Controls")
    symbol = st.selectbox("Symbol", symbols, index=0)
    st.caption(f"Serving API: {API_URL}")
    if st.button("🔄 Refresh"):
        st.cache_data.clear()
        st.rerun()

live_view(symbol)

st.markdown("### 🧠 Market summary")
try:
    summary = get_summary()
    st.markdown(summary["text"])
    st.caption(summary_caption(summary))
except Exception as exc:
    st.caption(f"Summary unavailable: {exc}")

st.caption("Data: Binance public WebSocket · Pipeline: Redpanda → Spark → SeaweedFS → dbt/DuckDB → LightGBM → FastAPI")
