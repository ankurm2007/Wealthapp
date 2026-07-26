"""Forensic quality checks: D/E, cash flow vs profit, FMP scores."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pandas as pd
import requests
import streamlit as st

import stock_analyzer as san

FMP_SCORE_URL = "https://financialmodelingprep.com/stable/score"
FMP_INCOME_URL = "https://financialmodelingprep.com/stable/income-statement"
FMP_CASH_URL = "https://financialmodelingprep.com/stable/cash-flow-statement"

DE_WARN = 1.0
DE_FAIL = 2.0
OCF_RATIO_WARN = 0.7
OCF_RATIO_FAIL = 0.4
PIOTROSKI_WARN = 5
PIOTROSKI_FAIL = 3


def _normalize_de_ratio(value: float) -> float:
    """Yahoo/FMP may return D/E as ratio or scaled value."""
    v = float(value)
    if v > 20:  # likely percentage-style (e.g. 36.6 meaning 0.37)
        return v / 100.0
    return v


def _position_cols_forensics(enriched: pd.DataFrame, merged: pd.DataFrame) -> pd.DataFrame:
    if "Weight %" in enriched.columns:
        return enriched.copy()
    return enriched.merge(merged[["Symbol", "Weight %"]], on="Symbol", how="left")


def _fmp_symbol(symbol: str) -> list[str]:
    base = san.to_yahoo_ticker(symbol).replace(".NS", "")
    return list(dict.fromkeys([f"{base}.NS", f"{base}.BO", base]))


def _fmp_get(url: str, symbol: str, api_key: str, limit: int = 1, period: str | None = None) -> list[dict]:
    for candidate in _fmp_symbol(symbol):
        try:
            params: dict = {"symbol": candidate, "apikey": api_key, "limit": limit}
            if period:
                params["period"] = period
            resp = requests.get(
                url,
                params=params,
                timeout=12,
            )
            if resp.status_code != 200:
                continue
            data = resp.json()
            if isinstance(data, list) and data:
                return data
        except Exception:
            continue
    return []


def fetch_fmp_scores(symbol: str, api_key: str) -> dict | None:
    rows = _fmp_get(FMP_SCORE_URL, symbol, api_key)
    return rows[0] if rows else None


def fetch_ocf_vs_profit(symbol: str, api_key: str) -> dict | None:
    income = _fmp_get(FMP_INCOME_URL, symbol, api_key, limit=1)
    cash = _fmp_get(FMP_CASH_URL, symbol, api_key, limit=1)
    if not income or not cash:
        return None
    inc = income[0]
    cf = cash[0]
    net_income = inc.get("netIncome") or inc.get("netIncomeRatio")
    ocf = cf.get("operatingCashFlow") or cf.get("netCashProvidedByOperatingActivities")
    if net_income is None or ocf is None:
        return None
    try:
        net_income = float(net_income)
        ocf = float(ocf)
    except (TypeError, ValueError):
        return None
    if net_income == 0:
        return None
    ratio = ocf / net_income
    return {
        "net_income": net_income,
        "operating_cash_flow": ocf,
        "ocf_to_profit": round(ratio, 2),
        "period": inc.get("date") or inc.get("calendarYear"),
    }


@st.cache_data(ttl="12h", show_spinner=False)
def fetch_forensic_snapshot(symbols: tuple[str, ...], api_key: str) -> dict[str, dict]:
    if not api_key:
        return {}

    def _one(sym: str) -> tuple[str, dict]:
        row: dict[str, Any] = {}
        scores = fetch_fmp_scores(sym, api_key)
        if scores:
            row["piotroski"] = scores.get("piotroskiScore") or scores.get("PiotroskiScore")
            row["altman_z"] = scores.get("altmanZScore") or scores.get("AltmanZScore")
        ocf = fetch_ocf_vs_profit(sym, api_key)
        if ocf:
            row.update(ocf)
        return sym, row

    out: dict[str, dict] = {}
    workers = min(4, max(1, len(symbols)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for sym, data in pool.map(_one, symbols):
            if data:
                out[sym] = data
    return out


def run_forensic_checks(
    merged: pd.DataFrame,
    enriched: pd.DataFrame | None,
    fmp_snapshot: dict[str, dict] | None,
    *,
    api_key: str = "",
) -> list[dict]:
    checks: list[dict] = []

    def add(status: str, name: str, headline: str, detail: str = "", table: pd.DataFrame | None = None):
        checks.append(
            {"status": status, "name": name, "headline": headline, "detail": detail, "table": table}
        )

    # --- Debt / equity from Yahoo enrichment ---
    if enriched is not None and "Debt/Equity" in enriched.columns:
        data = _position_cols_forensics(enriched, merged)
        heavy = data[(data["Weight %"] >= 3) & pd.notna(data["Debt/Equity"])].copy()
        heavy["D/E ratio"] = heavy["Debt/Equity"].apply(_normalize_de_ratio)

        fail_de = heavy[heavy["D/E ratio"] >= DE_FAIL]
        warn_de = heavy[(heavy["D/E ratio"] >= DE_WARN) & (heavy["D/E ratio"] < DE_FAIL)]
        if not fail_de.empty:
            tbl = fail_de[["Symbol", "Weight %", "D/E ratio"]].head(5).round(2)
            add(
                "fail",
                "high_leverage",
                f"{len(fail_de)} holding(s) with D/E ≥ {DE_FAIL:.1f}.",
                "High leverage amplifies downside in rate or earnings shocks.",
                tbl,
            )
        elif not warn_de.empty:
            tbl = warn_de[["Symbol", "Weight %", "D/E ratio"]].head(5).round(2)
            add(
                "warn",
                "high_leverage",
                f"{len(warn_de)} holding(s) with D/E between {DE_WARN:.1f} and {DE_FAIL:.1f}.",
                "Review balance sheet before adding size.",
                tbl,
            )
        else:
            add("pass", "high_leverage", "No material D/E flags on weighted holdings.")

    # --- FMP: OCF vs profit & Piotroski ---
    if not api_key:
        add(
            "warn",
            "fmp_data",
            "FMP key not configured — Piotroski and OCF checks skipped.",
            "Add [fmp] api_key in secrets.toml for forensic scores.",
        )
        return checks

    if not fmp_snapshot:
        add("warn", "fmp_data", "FMP forensic data not loaded yet.", "Load forensic data in Insights.")
        return checks

    ocf_rows = []
    pio_rows = []
    for sym in merged["Symbol"]:
        snap = fmp_snapshot.get(sym, {})
        if not snap:
            continue
        wt = float(merged.loc[merged["Symbol"] == sym, "Weight %"].iloc[0])
        if "ocf_to_profit" in snap:
            ocf_rows.append(
                {
                    "Symbol": sym,
                    "Weight %": wt,
                    "OCF / Net profit": snap["ocf_to_profit"],
                    "Net income": snap.get("net_income"),
                    "OCF": snap.get("operating_cash_flow"),
                }
            )
        if snap.get("piotroski") is not None:
            pio_rows.append({"Symbol": sym, "Weight %": wt, "Piotroski F-Score": snap["piotroski"]})

    if ocf_rows:
        ocf_df = pd.DataFrame(ocf_rows)
        weak = ocf_df[ocf_df["OCF / Net profit"] < OCF_RATIO_WARN]
        critical = ocf_df[ocf_df["OCF / Net profit"] < OCF_RATIO_FAIL]
        if not critical.empty:
            add(
                "fail",
                "earnings_quality",
                f"{len(critical)} name(s) with OCF well below reported profit.",
                "Cash conversion weak — earnings quality concern.",
                critical.head(5),
            )
        elif not weak.empty:
            add(
                "warn",
                "earnings_quality",
                f"{len(weak)} name(s) with OCF below ~70% of net profit.",
                "Profit may not be fully backed by cash flow.",
                weak.head(5),
            )
        else:
            add("pass", "earnings_quality", "OCF vs profit looks acceptable on covered names.")

    if pio_rows:
        pio_df = pd.DataFrame(pio_rows)
        low = pio_df[pio_df["Piotroski F-Score"] <= PIOTROSKI_FAIL]
        mid = pio_df[(pio_df["Piotroski F-Score"] > PIOTROSKI_FAIL) & (pio_df["Piotroski F-Score"] <= PIOTROSKI_WARN)]
        if not low.empty:
            add(
                "fail",
                "piotroski_score",
                f"{len(low)} name(s) with Piotroski F-Score ≤ {PIOTROSKI_FAIL}.",
                "Weak financial strength by FMP composite score.",
                low.head(5),
            )
        elif not mid.empty:
            add(
                "warn",
                "piotroski_score",
                f"{len(mid)} name(s) with middling Piotroski scores (4–5).",
                "Not distressed, but not high-quality either.",
                mid.head(5),
            )
        else:
            add("pass", "piotroski_score", "Piotroski scores look healthy on covered names.")
    else:
        add(
            "warn",
            "piotroski_score",
            "No Piotroski scores returned from FMP for your symbols.",
            "FMP coverage for NSE names can be limited — use Screener.in export as backup.",
        )

    return checks


def build_forensic_context(fmp_snapshot: dict[str, dict] | None, merged: pd.DataFrame) -> str:
    if not fmp_snapshot:
        return ""
    lines = ["FMP forensic snapshot:"]
    for sym in merged.head(10)["Symbol"]:
        snap = fmp_snapshot.get(sym)
        if not snap:
            continue
        parts = [sym]
        if snap.get("piotroski") is not None:
            parts.append(f"Piotroski {snap['piotroski']}")
        if snap.get("ocf_to_profit") is not None:
            parts.append(f"OCF/profit {snap['ocf_to_profit']:.2f}")
        if snap.get("altman_z") is not None:
            parts.append(f"Altman Z {snap['altman_z']}")
        lines.append("- " + " · ".join(parts))
    return "\n".join(lines)
