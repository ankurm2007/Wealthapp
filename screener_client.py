"""Shared Screener.in fetch + table parsing for Indian equity data."""

from __future__ import annotations

from urllib.parse import quote

import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup

SCREENER_BASE = "https://www.screener.in/company/{symbol}/consolidated/"
USER_AGENT = "Mozilla/5.0 (compatible; Wealthapp/1.0)"


def screener_url(symbol: str) -> str:
    slug = quote(symbol.strip().upper(), safe="")
    return SCREENER_BASE.format(symbol=slug)


@st.cache_data(ttl="12h", show_spinner=False)
def fetch_company_page(symbol: str) -> str:
    url = screener_url(symbol)
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=20)
    resp.raise_for_status()
    return resp.text


def _parse_number(value: str, *, as_crore_rupees: bool = True) -> float | None:
    clean = str(value).strip().replace(",", "")
    if not clean or clean in {"-", "—"}:
        return None
    if clean.endswith("%"):
        return float(clean[:-1])
    try:
        num = float(clean)
    except ValueError:
        return None
    if as_crore_rupees:
        return num * 1e7
    return num


def _period_to_date(period: str) -> str:
    try:
        return pd.to_datetime(period, format="%b %Y").strftime("%Y-%m-%d")
    except Exception:
        return period


def _parse_section_table(html: str, section_id: str) -> dict | None:
    soup = BeautifulSoup(html, "html.parser")
    section = soup.find("section", id=section_id)
    if section is None:
        return None
    table = section.find("table")
    if table is None:
        return None

    header_cells = [c.get_text(strip=True) for c in table.find("tr").find_all(["th", "td"])]
    periods = [c for c in header_cells[1:] if c]
    if not periods:
        return None

    rows: dict[str, list] = {}
    for tr in table.find_all("tr")[1:]:
        cells = [c.get_text(strip=True) for c in tr.find_all(["th", "td"])]
        if not cells:
            continue
        label = cells[0].strip()
        rows[label] = cells[1 : 1 + len(periods)]

    return {"periods": periods, "rows": rows}


def _match_row(rows: dict[str, list], *labels: str) -> list | None:
    normalized = {k.strip().lower().replace("+", ""): v for k, v in rows.items()}
    for label in labels:
        key = label.lower().replace("+", "")
        if key in normalized:
            return normalized[key]
        for row_key, values in normalized.items():
            if key in row_key or row_key.startswith(key):
                return values
    return None


def parse_quarterly_results(html: str, limit: int = 6) -> list[dict]:
    parsed = _parse_section_table(html, "quarters")
    if not parsed:
        return []

    periods = parsed["periods"][-limit:]
    rows = parsed["rows"]
    sales = _match_row(rows, "Sales+", "Sales") or []
    net_profit = _match_row(rows, "Net Profit+", "Net Profit") or []
    op_profit = _match_row(rows, "Operating Profit") or []
    eps = _match_row(rows, "EPS in Rs", "EPS") or []

    records: list[dict] = []
    offset = len(parsed["periods"]) - len(periods)
    for idx, period in enumerate(periods):
        i = offset + idx
        records.append(
            {
                "date": _period_to_date(period),
                "period_label": period,
                "revenue": _parse_number(sales[i]) if i < len(sales) else None,
                "net_income": _parse_number(net_profit[i]) if i < len(net_profit) else None,
                "operating_income": _parse_number(op_profit[i]) if i < len(op_profit) else None,
                "ebitda": None,
                "eps": _parse_number(eps[i], as_crore_rupees=False) if i < len(eps) else None,
            }
        )
    return list(reversed(records))


def parse_shareholding_pattern(html: str, limit: int = 6) -> dict | None:
    parsed = _parse_section_table(html, "shareholding")
    if not parsed:
        return None

    row_map = {
        "promoters": ("Promoters+", "Promoter"),
        "fiis": ("FIIs+", "FII"),
        "diis": ("DIIs+", "DII"),
        "government": ("Government+", "Government"),
        "public": ("Public+", "Public"),
        "shareholders": ("No. of Shareholders", "Shareholders"),
    }

    quarters: list[dict] = []
    for idx, period in enumerate(parsed["periods"]):
        row: dict = {"period": period}
        for key, labels in row_map.items():
            values = _match_row(parsed["rows"], *labels) or []
            val = values[idx] if idx < len(values) else None
            if key == "shareholders":
                clean = str(val or "").replace(",", "")
                row[key] = int(clean) if clean.isdigit() else None
            else:
                row[key] = _parse_number(str(val or ""), as_crore_rupees=False) if val else None
        quarters.append(row)

    if not quarters:
        return None

    trimmed = quarters[-limit:]
    latest = trimmed[-1]
    prior = trimmed[-2] if len(trimmed) >= 2 else None
    qoq: dict[str, float] = {}
    if prior:
        for key in ("promoters", "fiis", "diis", "government", "public"):
            cur = latest.get(key)
            prev = prior.get(key)
            if cur is not None and prev is not None:
                qoq[key] = round(cur - prev, 2)

    return {
        "periods": [q["period"] for q in trimmed],
        "quarters": trimmed,
        "latest": latest,
        "prior": prior,
        "qoq_pp": qoq,
    }
