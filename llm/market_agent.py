"""LLM market-summary agent.

Turns the latest per-symbol snapshot from the warehouse, plus the served model's call
on each symbol, into a short market read. Works fully offline with a template; if
OPENAI_API_KEY is set, an LLM writes the narrative from the same facts (and only
those facts: nothing is looked up or retrieved).

Dagster builds the summary after each refresh and saves it to market_summary.json next
to the warehouse. The dashboard only reads that file, so the LLM runs once per refresh,
never once per page view.

Usage:
    from llm.market_agent import build_market_summary
    print(build_market_summary()["text"])
"""
from __future__ import annotations

import json
import os
import textwrap

import duckdb
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DUCKDB = os.path.join(HERE, "data", "warehouse", "crypto.duckdb")

# Volatility is judged against the symbol's own median, since a fixed level means
# very different things for BTC and SOL.
VOL_HIGH, VOL_LOW = 1.5, 0.67
# A model call within this distance of 50/50 is reported as no lean at all.
LEAN = 0.02
FLOW = 0.05  # order-flow imbalance beyond this counts as net buying or selling


def _duckdb_path() -> str:
    return os.getenv("DUCKDB_PATH", DEFAULT_DUCKDB)


def summary_path() -> str:
    return os.getenv("SUMMARY_PATH", os.path.join(os.path.dirname(_duckdb_path()), "market_summary.json"))


def _model_dir() -> str:
    return os.getenv("MODEL_DIR", os.path.join(HERE, "ml", "models"))


def load_snapshot() -> list[dict]:
    if not os.path.exists(_duckdb_path()):
        return []
    con = duckdb.connect(_duckdb_path(), read_only=True)
    try:
        exists = con.execute(
            "select count(*) from information_schema.tables "
            "where table_schema='marts' and table_name='mart_symbol_snapshot'"
        ).fetchone()[0]
        if not exists:
            return []
        rows = con.execute(
            "select symbol, as_of, last_price, last_return, px_vs_sma30, volatility_30, typical_volatility_30, "
            "order_flow_imbalance, rel_volume from marts.mart_symbol_snapshot order by symbol"
        ).df()
    finally:
        con.close()
    return rows.to_dict("records")


def load_model_calls(symbols: list[str]) -> tuple[dict[str, dict], dict]:
    """The served model's P(up) on each symbol's newest warehouse bar, and its metadata.

    Returns nothing when no model is trained yet, or when this process lacks the ML
    packages (the dashboard image), so the summary still works without the calls.
    """
    model_path = os.path.join(_model_dir(), "direction_lgbm.pkl")
    if not os.path.exists(model_path):
        return {}, {}
    try:
        import joblib

        from ml.features import latest_feature_row
    except ImportError:
        return {}, {}
    payload = joblib.load(model_path)
    calls = {}
    for symbol in symbols:
        feats = latest_feature_row(symbol)
        if feats is None:
            continue
        x = pd.DataFrame([{c: feats[c] for c in payload["features"]}], columns=payload["features"])
        calls[symbol] = {"prob_up": float(payload["model"].predict_proba(x)[:, 1][0]), "as_of": feats["as_of"]}
    meta = {}
    meta_path = os.path.join(_model_dir(), "model_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as fh:
            meta = json.load(fh)
    return calls, meta


def _name(symbol: str) -> str:
    return symbol.upper().replace("USDT", "")


def _describe(row: dict) -> str:
    ret = (row.get("last_return") or 0) * 100
    trend = "above" if (row.get("px_vs_sma30") or 0) > 0 else "below"
    vol = row.get("volatility_30") or 0
    typical = row.get("typical_volatility_30") or 0
    ofi = row.get("order_flow_imbalance") or 0
    flow = "net buying" if ofi > FLOW else "net selling" if ofi < -FLOW else "balanced flow"
    if typical and vol > VOL_HIGH * typical:
        vol_word = "elevated"
    elif typical and vol < VOL_LOW * typical:
        vol_word = "subdued"
    else:
        vol_word = "normal"
    momentum = "bullish" if trend == "above" else "bearish"
    return (
        f"{_name(row['symbol'])}: last move {ret:+.2f}%, trading {trend} its 30-bar average ({momentum} momentum), "
        f"{vol_word} volatility, order flow shows {flow}."
    )


def _describe_call(call: dict) -> str:
    p = call["prob_up"]
    if abs(p - 0.5) < LEAN:
        return f"Model: no clear lean (P(up) {p:.2f} over the next 30s)."
    side = "up" if p > 0.5 else "down"
    return f"Model: a slight lean {side} (P(up) {p:.2f} over the next 30s)."


def _tone(snapshot: list[dict]) -> str:
    """Constructive when most symbols trade above their average and few see net selling."""
    n = len(snapshot)
    above = sum(1 for r in snapshot if (r.get("px_vs_sma30") or 0) > 0)
    buying = sum(1 for r in snapshot if (r.get("order_flow_imbalance") or 0) > FLOW)
    selling = sum(1 for r in snapshot if (r.get("order_flow_imbalance") or 0) < -FLOW)
    if above > n / 2 and selling <= n / 2:
        return "constructive"
    if above < n / 2 and buying <= n / 2:
        return "cautious"
    return "mixed"


def _edge_note(meta: dict) -> str | None:
    auc = (meta.get("metrics") or {}).get("test_auc")
    if auc is None:
        return None
    return (
        f"The model's edge is small (held-out AUC {auc:.3f}, where 0.5 is guessing), "
        "so its calls are slight leans, not forecasts."
    )


def _facts(snapshot: list[dict], calls: dict[str, dict]) -> list[str]:
    facts = []
    for row in snapshot:
        line = _describe(row)
        if row["symbol"] in calls:
            line += " " + _describe_call(calls[row["symbol"]])
        facts.append(line)
    return facts


def _template_summary(snapshot: list[dict], calls: dict[str, dict] | None = None, meta: dict | None = None) -> str:
    if not snapshot:
        return "No market data available yet. Start the pipeline or run the sample-data generator."
    calls = calls or {}
    header = f"Market read ({len(snapshot)} assets): tone is {_tone(snapshot)}."
    body = "\n".join(f"- {line}" for line in _facts(snapshot, calls))
    note = _edge_note(meta or {}) if calls else None
    return f"{header}\n\n{body}" + (f"\n\n{note}" if note else "")


def _llm_summary(snapshot: list[dict], calls: dict[str, dict], meta: dict) -> str:
    """An LLM narrative over exactly the facts the template would show."""
    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )
    facts = "\n".join(_facts(snapshot, calls))
    note = _edge_note(meta) if calls else None
    prompt = textwrap.dedent(
        """
        You are a concise crypto market analyst. Using ONLY the facts below, write a
        3-4 sentence market summary for a trading dashboard. Be specific and neutral, do
        not invent numbers, and do not give financial advice. Describe the model's calls
        as slight leans, never as predictions of what will happen.

        Facts:
        """
    ).strip()
    prompt += "\n" + facts + (f"\n{note}" if note else "")
    resp = client.chat.completions.create(
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=220,
    )
    return resp.choices[0].message.content.strip()


def build_market_summary(use_llm: bool = True) -> dict:
    """The summary text plus what it was built from.

    Uses the LLM when use_llm is set and OPENAI_API_KEY is configured; otherwise, or if
    the LLM call fails, the offline template.
    """
    snapshot = load_snapshot()
    calls, meta = load_model_calls([r["symbol"] for r in snapshot]) if snapshot else ({}, {})
    as_of = [pd.Timestamp(r["as_of"]) for r in snapshot if r.get("as_of") is not None]
    record = {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "data_as_of": max(as_of).isoformat() if as_of else None,
        "calls": calls,
        "model_trained_through": meta.get("trained_through"),
        "source": "template",
        "text": _template_summary(snapshot, calls, meta),
    }
    if snapshot and use_llm and os.getenv("OPENAI_API_KEY"):
        try:
            record["text"], record["source"] = _llm_summary(snapshot, calls, meta), "llm"
        except Exception as exc:  # keep the offline summary rather than nothing
            record["llm_error"] = f"{type(exc).__name__}: {exc}"
    return record


def save_market_summary(record: dict, path: str | None = None) -> str:
    """Write then rename, so the dashboard never reads half a file."""
    path = path or summary_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as fh:
        json.dump(record, fh, indent=2)
    os.replace(tmp, path)
    return path


def load_saved_summary(path: str | None = None) -> dict | None:
    path = path or summary_path()
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def generate_market_summary() -> str:
    """Just the text, built now."""
    return build_market_summary()["text"]


if __name__ == "__main__":
    summary = build_market_summary()
    print(summary["text"])
    print(f"\n(saved to {save_market_summary(summary)}, source: {summary['source']})")
