"""LLM market-summary agent.

Turns the latest per-symbol feature snapshot (from DuckDB) + the model's prediction
into a short, natural-language market read. Works fully offline with a deterministic
template summariser; if OPENAI_API_KEY is set, it upgrades to an LLM-written narrative.

Usage:
    from llm.market_agent import generate_market_summary
    print(generate_market_summary())
"""
from __future__ import annotations

import os
import textwrap

import duckdb

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DUCKDB = os.path.join(HERE, "data", "warehouse", "crypto.duckdb")


def _duckdb_path() -> str:
    return os.getenv("DUCKDB_PATH", DEFAULT_DUCKDB)


def load_snapshot() -> list[dict]:
    con = duckdb.connect(_duckdb_path(), read_only=True)
    try:
        exists = con.execute(
            "select count(*) from information_schema.tables "
            "where table_schema='marts' and table_name='mart_symbol_snapshot'"
        ).fetchone()[0]
        if not exists:
            return []
        rows = con.execute(
            "select symbol, last_price, last_return, px_vs_sma30, volatility_30, "
            "order_flow_imbalance, rel_volume from marts.mart_symbol_snapshot order by symbol"
        ).df()
    finally:
        con.close()
    return rows.to_dict("records")


def _describe(row: dict) -> str:
    sym = row["symbol"].upper().replace("USDT", "")
    ret = (row.get("last_return") or 0) * 100
    trend = "above" if (row.get("px_vs_sma30") or 0) > 0 else "below"
    vol = row.get("volatility_30") or 0
    ofi = row.get("order_flow_imbalance") or 0
    flow = "net buying" if ofi > 0.05 else "net selling" if ofi < -0.05 else "balanced flow"
    vol_word = "elevated" if vol > 0.004 else "subdued"
    momentum = "bullish" if trend == "above" else "bearish"
    return (
        f"{sym}: last move {ret:+.2f}%, trading {trend} its 30-bar average ({momentum} momentum), "
        f"{vol_word} volatility, order flow shows {flow}."
    )


def _template_summary(snapshot: list[dict]) -> str:
    if not snapshot:
        return "No market data available yet. Start the pipeline or run the sample-data generator."
    lines = [_describe(r) for r in snapshot]
    up = sum(1 for r in snapshot if (r.get("px_vs_sma30") or 0) > 0)
    tone = "broadly constructive" if up > len(snapshot) / 2 else "cautious"
    header = f"Market read ({len(snapshot)} assets) — tone is {tone}."
    return header + "\n\n" + "\n".join(f"- {ln}" for ln in lines)


def _llm_summary(snapshot: list[dict]) -> str:
    """Upgrade to an LLM narrative when an API key is configured."""
    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )
    facts = "\n".join(_describe(r) for r in snapshot)
    prompt = textwrap.dedent(
        f"""
        You are a concise crypto market analyst. Using ONLY the facts below, write a
        3-4 sentence market summary for a trading dashboard. Be specific, neutral, and
        do not invent numbers or give financial advice.

        Facts:
        {facts}
        """
    ).strip()
    resp = client.chat.completions.create(
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=220,
    )
    return resp.choices[0].message.content.strip()


def generate_market_summary() -> str:
    snapshot = load_snapshot()
    if not snapshot:
        return _template_summary(snapshot)
    if os.getenv("OPENAI_API_KEY"):
        try:
            return _llm_summary(snapshot)
        except Exception as exc:  # graceful fallback keeps the demo alive
            return _template_summary(snapshot) + f"\n\n(LLM unavailable: {exc}; using offline summary.)"
    return _template_summary(snapshot)


if __name__ == "__main__":
    print(generate_market_summary())
