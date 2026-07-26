"""Nifty and sector benchmark helpers for Insights analysis."""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

import yahoo_client as yahoo

# Approximate Nifty 50 sector weights mapped to Yahoo/our labels.
# Sources are indicative (index reconstitution changes weights over time).
NIFTY50_SECTOR_WEIGHTS = {
    "Financials": 35.0,
    "Technology": 13.0,
    "Energy": 11.0,
    "Consumer": 9.0,
    "FMCG": 8.0,
    "Industrials": 6.0,
    "Healthcare": 5.0,
    "Materials": 4.0,
    "Utilities": 3.0,
    "Telecom & Media": 3.0,
    "Real Estate": 1.5,
    "ETF / Index": 0.0,
    "Equity (uncategorized)": 0.0,
    "Unknown": 0.0,
}


def _flatten_close(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=float)
    if isinstance(df.columns, pd.MultiIndex):
        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
    else:
        close = df["Close"]
    return close.astype(float).dropna()


@st.cache_data(ttl="6h", show_spinner=False)
def fetch_index_history(ticker: str = "^NSEI", days: int = 90) -> pd.DataFrame:
    end = datetime.now()
    start = end - timedelta(days=days + 10)
    df = yahoo.safe_download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        progress=False,
        auto_adjust=True,
    )
    if df is None or df.empty:
        df = yahoo.safe_download(
            "NIFTYBEES.NS",
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=True,
        )
    if df is None or df.empty:
        return pd.DataFrame()

    close = _flatten_close(df)
    out = pd.DataFrame({"date": close.index, "value": close.values})
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    return out.tail(days).reset_index(drop=True)


def period_return(series: pd.Series) -> float | None:
    clean = series.dropna()
    if len(clean) < 2:
        return None
    start = float(clean.iloc[0])
    end = float(clean.iloc[-1])
    if start == 0:
        return None
    return (end / start - 1.0) * 100


@st.cache_data(ttl="6h", show_spinner=False)
def fetch_portfolio_proxy_return(symbols: tuple[str, ...], weights: tuple[float, ...], days: int = 30) -> dict:
    """
    Approximate portfolio return by weight-averaging Yahoo price returns.
    """
    if not symbols or not weights or sum(weights) <= 0:
        return {"ok": False, "return_pct": None, "series": pd.DataFrame(), "coverage": 0}

    end = datetime.now()
    start = end - timedelta(days=days + 10)
    frames = []
    used_weight = 0.0

    for symbol, weight in zip(symbols, weights):
        ticker = f"{symbol}.NS"
        hist = yahoo.safe_download(
            ticker,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=True,
        )
        close = _flatten_close(hist)
        if close.empty:
            continue
        normalized = close / close.iloc[0]
        frames.append(normalized.rename(symbol) * weight)
        used_weight += weight

    if not frames or used_weight <= 0:
        return {"ok": False, "return_pct": None, "series": pd.DataFrame(), "coverage": 0}

    combined = pd.concat(frames, axis=1).ffill().dropna(how="all")
    portfolio = combined.sum(axis=1) / used_weight
    series = pd.DataFrame({"date": portfolio.index, "value": portfolio.values})
    series["date"] = pd.to_datetime(series["date"]).dt.tz_localize(None)
    series = series.tail(days).reset_index(drop=True)
    return {
        "ok": True,
        "return_pct": period_return(series["value"]),
        "series": series,
        "coverage": round(used_weight, 1),
    }


def build_nifty_vs_portfolio(merged: pd.DataFrame, days: int = 30) -> dict:
    symbols = tuple(merged["Symbol"].tolist())
    weights = tuple(float(w) for w in merged["Weight %"].tolist())
    portfolio = fetch_portfolio_proxy_return(symbols, weights, days=days)
    nifty = fetch_index_history("^NSEI", days=days)

    nifty_ret = period_return(nifty["value"]) if not nifty.empty else None
    port_ret = portfolio.get("return_pct")

    comparison = pd.DataFrame(
        {
            "Benchmark": ["Your portfolio", "Nifty 50"],
            "Return %": [
                round(port_ret, 2) if port_ret is not None else None,
                round(nifty_ret, 2) if nifty_ret is not None else None,
            ],
        }
    )

    # Align series for overlay chart (normalize to 100).
    chart_df = pd.DataFrame()
    if portfolio.get("ok") and not portfolio["series"].empty and not nifty.empty:
        p = portfolio["series"].copy()
        n = nifty.copy()
        p["Portfolio"] = p["value"] / p["value"].iloc[0] * 100
        n["Nifty 50"] = n["value"] / n["value"].iloc[0] * 100
        chart_df = (
            p[["date", "Portfolio"]]
            .merge(n[["date", "Nifty 50"]], on="date", how="inner")
            .melt("date", var_name="Series", value_name="Indexed")
        )

    alpha = None
    if port_ret is not None and nifty_ret is not None:
        alpha = round(port_ret - nifty_ret, 2)

    return {
        "comparison": comparison,
        "chart_df": chart_df,
        "portfolio_return": port_ret,
        "nifty_return": nifty_ret,
        "alpha": alpha,
        "coverage": portfolio.get("coverage", 0),
        "days": days,
    }


def build_sector_vs_nifty(sector_df: pd.DataFrame) -> pd.DataFrame:
    if sector_df is None or sector_df.empty:
        return pd.DataFrame()

    rows = []
    for _, row in sector_df.iterrows():
        sector = row["Sector"]
        if sector in ("Unknown", "Equity (uncategorized)"):
            continue
        nifty_w = NIFTY50_SECTOR_WEIGHTS.get(sector, 0.0)
        port_w = float(row["Weight %"])
        rows.append(
            {
                "Sector": sector,
                "Your portfolio %": round(port_w, 2),
                "Nifty 50 %": round(nifty_w, 2),
                "Active weight %": round(port_w - nifty_w, 2),
            }
        )

    # Include Nifty sectors you don't hold (active underweight).
    held = {r["Sector"] for r in rows}
    for sector, nifty_w in NIFTY50_SECTOR_WEIGHTS.items():
        if sector in held or nifty_w <= 0:
            continue
        rows.append(
            {
                "Sector": sector,
                "Your portfolio %": 0.0,
                "Nifty 50 %": round(nifty_w, 2),
                "Active weight %": round(-nifty_w, 2),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("Your portfolio %", ascending=False).reset_index(drop=True)
