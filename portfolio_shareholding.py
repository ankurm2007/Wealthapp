"""Indian shareholding pattern data (Promoter / FII / DII / Public) via Screener.in."""

from __future__ import annotations

import re
from datetime import datetime

import pandas as pd
import streamlit as st

import portfolio_earnings as pearn
import portfolio_finance_context as pfctx
import screener_client as screener

SHAREHOLDING_PATTERNS = (
    r"\bshareholding\b",
    r"\bshare.?holding\b",
    r"\bpromoter\b.*\b(holding|stake|share)\b",
    r"\b(holding|stake|share)\b.*\bpromoter\b",
    r"\bfii\b",
    r"\bdii\b",
    r"\binstitutional\b.*\b(holding|ownership|stake)\b",
    r"\bownership\b.*\b(pattern|breakdown|structure)\b",
    r"\bwho owns\b",
    r"\bpublic holding\b",
    r"\bretail holding\b",
)


def is_shareholding_question(question: str) -> bool:
    text = question.lower()
    if pearn.is_earnings_question(question) and not any(
        re.search(p, text) for p in (r"\bshareholding\b", r"\bpromoter\b", r"\bfii\b", r"\bdii\b")
    ):
        return False
    return any(re.search(pattern, text) for pattern in SHAREHOLDING_PATTERNS)


def _from_institutional_row(inst_row: pd.Series) -> dict | None:
    mapping = {
        "promoters": "Promoter %",
        "fiis": "FII %",
        "diis": "DII %",
        "public": "Public %",
    }
    latest: dict = {"period": "Uploaded export"}
    has_data = False
    for key, col in mapping.items():
        val = inst_row.get(col)
        if pd.notna(val):
            latest[key] = float(val)
            has_data = True
        else:
            latest[key] = None
    if not has_data:
        return None
    pledged = inst_row.get("Pledged %")
    if pd.notna(pledged):
        latest["pledged_pct"] = float(pledged)
    return latest


@st.cache_data(ttl="12h", show_spinner=False)
def fetch_shareholding_pattern(symbol: str) -> dict:
    try:
        html = screener.fetch_company_page(symbol)
        parsed = screener.parse_shareholding_pattern(html)
    except Exception as exc:
        return {"ok": False, "symbol": symbol, "error": f"Could not reach Screener.in: {exc}"}

    if not parsed:
        return {
            "ok": False,
            "symbol": symbol,
            "error": "Shareholding table not found on Screener.in for this symbol.",
        }

    return {
        "ok": True,
        "symbol": symbol,
        "source": "Screener.in",
        "url": screener.screener_url(symbol),
        "quarters": parsed["quarters"],
        "latest": parsed["latest"],
        "prior": parsed["prior"],
        "qoq_pp": parsed["qoq_pp"],
        "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


def build_shareholding_context(
    data: dict,
    merged_row: dict | None = None,
    inst_row: dict | None = None,
) -> str:
    if inst_row and not data.get("ok"):
        lines = [
            f"Shareholding pattern ({data.get('symbol', 'symbol')}, uploaded Screener/Trendlyne export):"
        ]
        for field, label in (
            ("promoters", "Promoter"),
            ("fiis", "FII"),
            ("diis", "DII"),
            ("public", "Public"),
        ):
            val = inst_row.get(field)
            if val is not None:
                lines.append(f"- {label}: {val:.2f}%")
        if inst_row.get("pledged_pct") is not None:
            lines.append(f"- Promoter pledged: {inst_row['pledged_pct']:.2f}%")
        if merged_row:
            lines.append(
                f"Portfolio position: {data.get('symbol')} at {merged_row.get('Weight %', 0):.2f}% weight."
            )
        return "\n".join(lines)

    if not data.get("ok"):
        return f"Shareholding data for {data.get('symbol', 'symbol')}: {data.get('error', 'unavailable')}"

    latest = data["latest"]
    period = latest.get("period", "latest quarter")
    lines = [
        f"Shareholding pattern ({data['symbol']}, source: {data.get('source')}, period: {period}, "
        f"fetched {data.get('fetched_at', '—')}):",
        f"- Promoter: {latest.get('promoters'):.2f}%" if latest.get("promoters") is not None else "- Promoter: —",
        f"- FII: {latest.get('fiis'):.2f}%" if latest.get("fiis") is not None else "- FII: —",
        f"- DII: {latest.get('diis'):.2f}%" if latest.get("diis") is not None else "- DII: —",
        f"- Public: {latest.get('public'):.2f}%" if latest.get("public") is not None else "- Public: —",
    ]
    if latest.get("government") is not None:
        lines.append(f"- Government: {latest['government']:.2f}%")
    if latest.get("shareholders") is not None:
        lines.append(f"- Shareholders count: {latest['shareholders']:,}")

    qoq = data.get("qoq_pp") or {}
    if qoq:
        parts = []
        for key, label in (("promoters", "Promoter"), ("fiis", "FII"), ("diis", "DII"), ("public", "Public")):
            change = qoq.get(key)
            if change is not None:
                parts.append(f"{label} {change:+.2f} pp")
        if parts:
            prior = data.get("prior") or {}
            lines.append(f"- QoQ change vs {prior.get('period', 'prior quarter')}: " + ", ".join(parts))

    lines.append("Recent quarters (Promoter / FII / DII / Public %):")
    for row in data.get("quarters", [])[-5:]:
        lines.append(
            f"- {row.get('period')}: "
            f"{row.get('promoters', 0):.2f}% / "
            f"{row.get('fiis', 0):.2f}% / "
            f"{row.get('diis', 0):.2f}% / "
            f"{row.get('public', 0):.2f}%"
        )

    if merged_row:
        lines.append(
            f"Portfolio position: {data['symbol']} at {merged_row.get('Weight %', 0):.2f}% weight, "
            f"{merged_row.get('Return %', 0):+.2f}% return."
        )
    return "\n".join(lines)


def context_for_symbol(
    symbol: str,
    merged: pd.DataFrame,
    inst_df: pd.DataFrame | None = None,
) -> str:
    merged_row = None
    if symbol in merged["Symbol"].values:
        merged_row = merged.loc[merged["Symbol"] == symbol].iloc[0].to_dict()

    inst_row = None
    if inst_df is not None and not inst_df.empty and symbol in inst_df["Symbol"].values:
        inst = inst_df.loc[inst_df["Symbol"] == symbol].iloc[0]
        inst_row = _from_institutional_row(inst)
        if inst_row and "pledged_pct" not in inst_row and "Pledged %" in inst.columns:
            pledged = inst.get("Pledged %")
            if pd.notna(pledged):
                inst_row["pledged_pct"] = float(pledged)

    data = fetch_shareholding_pattern(symbol)
    ctx = build_shareholding_context(data, merged_row, inst_row)
    if inst_row and inst_row.get("pledged_pct") is not None and "pledged" not in ctx.lower():
        ctx += f"\n- Promoter pledged (upload): {inst_row['pledged_pct']:.2f}%"
    return ctx


def context_for_question(
    question: str,
    merged: pd.DataFrame,
    inst_df: pd.DataFrame | None = None,
) -> str:
    if not is_shareholding_question(question):
        return ""
    symbol = pfctx.resolve_symbol_from_question(question, merged["Symbol"].tolist())
    if not symbol:
        return ""
    return context_for_symbol(symbol, merged, inst_df)
