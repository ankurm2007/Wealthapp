"""Quarterly earnings / results data for portfolio holdings (Screener.in + FMP + Yahoo fallback)."""

from __future__ import annotations

import re
from datetime import datetime

import pandas as pd
import streamlit as st

import portfolio_forensics as pforensic
import screener_client as screener
import stock_analyzer as san
import yahoo_client as yahoo

FMP_INCOME_URL = "https://financialmodelingprep.com/stable/income-statement"

INCOME_ROWS = {
    "revenue": ("Total Revenue", "Operating Revenue", "Revenue"),
    "gross_profit": ("Gross Profit",),
    "operating_income": ("Operating Income", "EBIT"),
    "ebitda": ("EBITDA", "Normalized EBITDA"),
    "net_income": ("Net Income", "Net Income Common Stockholders"),
    "eps": ("Diluted EPS", "Basic EPS"),
}

EARNINGS_PATTERNS = (
    r"\bquarterly\b",
    r"\bquarter\b",
    r"\bq[1-4]\b",
    r"\bresults?\b",
    r"\bearnings\b",
    r"\beps\b",
    r"\bnet profit\b",
    r"\brevenue\b.*\b(quarter|results|earnings)\b",
    r"\b(quarter|results|earnings)\b.*\brevenue\b",
    r"\bprofit after tax\b",
    r"\bp&l\b.*\b(quarter|results)\b",
)


def is_earnings_question(question: str) -> bool:
    text = question.lower()
    return any(re.search(pattern, text) for pattern in EARNINGS_PATTERNS)


def _fmt_cr(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"₹{value / 1e7:,.0f} cr"


def _pick_row(df: pd.DataFrame, names: tuple[str, ...]) -> pd.Series | None:
    for name in names:
        if name in df.index:
            return df.loc[name]
    return None


def _pct_change(current: float | None, prior: float | None) -> float | None:
    if current is None or prior in (None, 0) or pd.isna(current) or pd.isna(prior):
        return None
    return round((float(current) / float(prior) - 1) * 100, 1)


def _same_quarter_year_ago(d1: str, d2: str) -> bool:
    try:
        a = datetime.strptime(d1, "%Y-%m-%d")
        b = datetime.strptime(d2, "%Y-%m-%d")
    except ValueError:
        return False
    return a.month == b.month and a.year - b.year == 1


def _enrich_quarter_changes(records: list[dict]) -> list[dict]:
    for i, row in enumerate(records):
        prev = records[i + 1] if i + 1 < len(records) else None
        if prev:
            row["revenue_qoq_pct"] = _pct_change(row.get("revenue"), prev.get("revenue"))
            row["net_income_qoq_pct"] = _pct_change(row.get("net_income"), prev.get("net_income"))
            row["eps_qoq_pct"] = _pct_change(row.get("eps"), prev.get("eps"))

        yoy = next((r for r in records[i + 1 :] if _same_quarter_year_ago(row["date"], r["date"])), None)
        if yoy:
            row["revenue_yoy_pct"] = _pct_change(row.get("revenue"), yoy.get("revenue"))
            row["net_income_yoy_pct"] = _pct_change(row.get("net_income"), yoy.get("net_income"))
            row["eps_yoy_pct"] = _pct_change(row.get("eps"), yoy.get("eps"))
    return records


def _quarter_records_from_yahoo(df: pd.DataFrame, limit: int = 6) -> list[dict]:
    cols = list(df.columns[:limit])
    records: list[dict] = []
    for col in cols:
        period = pd.Timestamp(col)
        row: dict = {"date": period.strftime("%Y-%m-%d"), "period_label": period.strftime("%b %Y")}
        for key, names in INCOME_ROWS.items():
            series = _pick_row(df, names)
            if series is not None and col in series.index:
                val = series[col]
                row[key] = float(val) if pd.notna(val) else None
            else:
                row[key] = None
        records.append(row)
    return _enrich_quarter_changes(records)


def _fetch_screener_quarters(symbol: str, limit: int = 6) -> list[dict]:
    try:
        html = screener.fetch_company_page(symbol)
        records = screener.parse_quarterly_results(html, limit=limit)
        return _enrich_quarter_changes(records)
    except Exception:
        return []


def _fetch_yahoo_quarters(symbol: str, limit: int = 6) -> list[dict]:
    df = yahoo.safe_quarterly_income(san.to_yahoo_ticker(symbol))
    if df.empty:
        return []
    return _quarter_records_from_yahoo(df, limit=limit)


def _fetch_fmp_quarters(symbol: str, api_key: str, limit: int = 6) -> list[dict]:
    if not api_key:
        return []
    rows = pforensic._fmp_get(FMP_INCOME_URL, symbol, api_key, limit=limit, period="quarter")
    if not rows:
        return []
    records: list[dict] = []
    for item in rows:
        records.append(
            {
                "date": str(item.get("date") or item.get("fillingDate") or "")[:10],
                "period_label": str(item.get("period") or item.get("calendarYear") or item.get("date") or ""),
                "revenue": item.get("revenue"),
                "gross_profit": item.get("grossProfit"),
                "operating_income": item.get("operatingIncome"),
                "ebitda": item.get("ebitda"),
                "net_income": item.get("netIncome"),
                "eps": item.get("eps") or item.get("epsdiluted"),
            }
        )
    return _enrich_quarter_changes(records)


@st.cache_data(ttl="6h", show_spinner=False)
def fetch_quarterly_earnings(symbol: str, fmp_api_key: str = "", limit: int = 6) -> dict:
    quarters = _fetch_screener_quarters(symbol, limit=limit)
    source = "Screener.in"
    if not quarters and fmp_api_key:
        quarters = _fetch_fmp_quarters(symbol, fmp_api_key, limit=limit)
        source = "FMP"
    if not quarters:
        quarters = _fetch_yahoo_quarters(symbol, limit=limit)
        source = "Yahoo Finance"
    if not quarters:
        return {
            "ok": False,
            "symbol": symbol,
            "error": "No quarterly results found (Screener.in / FMP / Yahoo unavailable).",
        }

    return {
        "ok": True,
        "symbol": symbol,
        "source": source,
        "quarters": quarters,
        "latest_event": None,
        "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


def build_earnings_context(data: dict, merged_row: dict | None = None) -> str:
    if not data.get("ok"):
        return f"Quarterly earnings data for {data.get('symbol', 'symbol')}: {data.get('error', 'unavailable')}"

    lines = [
        f"Quarterly earnings / results ({data['symbol']}, source: {data.get('source', '—')}, "
        f"fetched {data.get('fetched_at', '—')}):",
    ]
    latest = data["quarters"][0]
    eps_txt = f"{latest['eps']:.2f}" if latest.get("eps") is not None else "—"
    lines.append(
        f"Latest quarter ({latest.get('period_label', latest['date'])}): "
        f"Revenue {_fmt_cr(latest.get('revenue'))}, "
        f"Net profit {_fmt_cr(latest.get('net_income'))}, "
        f"EBITDA {_fmt_cr(latest.get('ebitda'))}, "
        f"Operating profit {_fmt_cr(latest.get('operating_income'))}, "
        f"EPS {eps_txt}"
    )
    if latest.get("revenue_qoq_pct") is not None:
        lines.append(
            f"- QoQ: revenue {latest['revenue_qoq_pct']:+.1f}%, "
            f"net profit {latest.get('net_income_qoq_pct', 0):+.1f}%, "
            f"EPS {latest.get('eps_qoq_pct', 0):+.1f}%"
        )
    if latest.get("revenue_yoy_pct") is not None:
        lines.append(
            f"- YoY (same quarter prior year): revenue {latest['revenue_yoy_pct']:+.1f}%, "
            f"net profit {latest.get('net_income_yoy_pct', 0):+.1f}%, "
            f"EPS {latest.get('eps_yoy_pct', 0):+.1f}%"
        )

    lines.append("Recent quarters (revenue / net profit / EPS):")
    for row in data["quarters"][:5]:
        eps_row = f"{row['eps']:.2f}" if row.get("eps") is not None else "—"
        lines.append(
            f"- {row.get('period_label', row['date'])}: "
            f"rev {_fmt_cr(row.get('revenue'))}, "
            f"PAT {_fmt_cr(row.get('net_income'))}, "
            f"EPS {eps_row}"
        )

    if merged_row:
        lines.append(
            f"Portfolio position: {data['symbol']} at {merged_row.get('Weight %', 0):.2f}% weight, "
            f"{merged_row.get('Return %', 0):+.2f}% return, P&L ₹{merged_row.get('P&L', 0):,.0f}."
        )
    return "\n".join(lines)
