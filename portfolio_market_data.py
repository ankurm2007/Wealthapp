"""Yahoo Finance market data for Indian equities."""

from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import streamlit as st

# Renamed / merged NSE symbols that Yahoo will not resolve under the old ticker.
SYMBOL_ALIASES: dict[str, list[str]] = {
    "ZOMATO": ["ETERNAL"],
    "MACROTECH": ["LODHA"],
    "L&TFH": ["LTF"],
    "LTFH": ["LTF"],
    "LT FINANCE": ["LTF"],
    "L&T FINANCE": ["LTF"],
}

ETF_KEYWORDS = (
    "ETF",
    "BEES",
    "NIFTY",
    "SENSEX",
    "INDEX",
    "GOLD BEES",
    "LIQUID BEES",
    "JUNIOR BEES",
    "BANK BEES",
    "IT BEES",
)

SECTOR_LABELS = {
    "Consumer Defensive": "FMCG",
    "Consumer Cyclical": "Consumer",
    "Financial Services": "Financials",
    "Basic Materials": "Materials",
    "Communication Services": "Telecom & Media",
    "Real Estate": "Real Estate",
}


def portfolio_symbols_key(symbols: list[str]) -> tuple[str, ...]:
    return tuple(sorted(set(symbols)))


def _clean_symbol(symbol: str) -> str:
    return re.sub(r"\s+", "", symbol.strip().upper())


def _symbol_bases(symbol: str) -> list[str]:
    clean = _clean_symbol(symbol)
    bases = [clean]

    for alias in SYMBOL_ALIASES.get(clean, []):
        bases.append(_clean_symbol(alias))

    if "-" in clean:
        bases.append(clean.split("-")[0])

    # Drop duplicate bases while preserving order.
    return list(dict.fromkeys(base for base in bases if base))


def _ticker_candidates(symbol: str) -> list[str]:
    tickers: list[str] = []
    for base in _symbol_bases(symbol):
        tickers.extend([f"{base}.NS", f"{base}.BO"])
    return list(dict.fromkeys(tickers))


def _normalize_sector_label(sector: str) -> str:
    return SECTOR_LABELS.get(sector, sector)


def _infer_sector_industry(info: dict, symbol: str) -> tuple[str, str]:
    sector = info.get("sector")
    industry = info.get("industry")

    if sector:
        return _normalize_sector_label(sector), industry or "Unknown"

    if industry:
        return _normalize_sector_label(industry), industry

    name = (info.get("longName") or info.get("shortName") or symbol).upper()
    sym = _clean_symbol(symbol)

    if sym.startswith("SGB") or "SOVEREIGN GOLD" in name:
        return "Gold / SGB", "Sovereign Gold Bond"

    if any(keyword in name or keyword in sym for keyword in ETF_KEYWORDS):
        return "ETF / Index", "Exchange Traded Fund"

    if "REIT" in name:
        return "Real Estate", "REIT"

    if "INVIT" in name:
        return "Infrastructure", "InvIT"

    if info.get("regularMarketPrice"):
        return "Equity (uncategorized)", "Sector not provided by Yahoo"

    return "Unknown", "Unknown"


def _extract_info(info: dict, symbol: str, ticker: str) -> dict | None:
    if not info or info.get("regularMarketPrice") is None:
        return None

    sector, industry = _infer_sector_industry(info, symbol)
    high_52 = info.get("fiftyTwoWeekHigh")
    low_52 = info.get("fiftyTwoWeekLow")
    range_52 = f"₹{low_52:,.0f} – ₹{high_52:,.0f}" if high_52 and low_52 else None
    roe = info.get("returnOnEquity")
    div_yield = info.get("dividendYield")

    return {
        "Symbol": symbol,
        "Company": info.get("longName") or info.get("shortName") or symbol,
        "Sector": sector,
        "Industry": industry,
        "Yahoo price": info.get("regularMarketPrice"),
        "P/E": info.get("trailingPE"),
        "Forward P/E": info.get("forwardPE"),
        "P/B": info.get("priceToBook"),
        "Beta": info.get("beta"),
        "ROE %": round(roe * 100, 1) if roe is not None else None,
        "Debt/Equity": info.get("debtToEquity"),
        "Div yield %": round(div_yield * 100, 2) if div_yield is not None else None,
        "Market cap": info.get("marketCap"),
        "52w high": high_52,
        "52w low": low_52,
        "52-week range": range_52,
        "Yahoo ticker": ticker,
        "Data source": "Yahoo Finance",
    }


def _lookup_yahoo_info(symbol: str) -> dict:
    import yahoo_client as yahoo

    tickers = _ticker_candidates(symbol)
    for ticker in tickers:
        for attempt in range(2):
            info = yahoo.safe_ticker_info(ticker)
            parsed = _extract_info(info, symbol, ticker) if info else None
            if parsed:
                return parsed
            if attempt == 0:
                time.sleep(0.35)
            continue

    return {
        "Symbol": symbol,
        "Company": symbol,
        "Sector": "Unknown",
        "Industry": "Unknown",
        "Yahoo price": None,
        "P/E": None,
        "Market cap": None,
        "52-week range": None,
        "Yahoo ticker": None,
        "Data source": "Not found",
    }


@st.cache_data(ttl="6h", show_spinner=False)
def fetch_market_info_for_symbols(symbols: tuple[str, ...], _lookup_version: int = 4) -> pd.DataFrame:
    del _lookup_version
    if not symbols:
        return pd.DataFrame()
    workers = min(2, len(symbols))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        rows = list(executor.map(_lookup_yahoo_info, symbols))
    return pd.DataFrame(rows)


def enrich_holdings(merged: pd.DataFrame) -> pd.DataFrame:
    import symbol_resolver as sym

    merged_resolved = sym.normalize_portfolio_symbols(merged)
    symbols = portfolio_symbols_key(merged_resolved["Symbol"].tolist())
    market_df = fetch_market_info_for_symbols(symbols)
    enriched = merged_resolved.merge(market_df, on="Symbol", how="left")
    enriched["Sector"] = enriched["Sector"].fillna("Unknown")
    enriched["Industry"] = enriched["Industry"].fillna("Unknown")
    enriched["Company"] = enriched["Company"].fillna(enriched["Symbol"])
    if "Company Name" in enriched.columns:
        enriched["Company"] = enriched["Company"].where(
            enriched["Company"].notna() & (enriched["Company"] != enriched["Symbol"]),
            enriched["Company Name"],
        )
    return enriched


def sector_summary(enriched: pd.DataFrame, total_current: float) -> pd.DataFrame:
    grouped = (
        enriched.groupby("Sector", as_index=False)
        .agg(
            Current_Value=("Current Value", "sum"),
            Invested_Value=("Invested Value", "sum"),
            PnL=("P&L", "sum"),
            Holdings=("Symbol", "count"),
            Stocks=("Symbol", lambda values: ", ".join(values)),
        )
        .rename(
            columns={
                "Current_Value": "Current Value",
                "Invested_Value": "Invested Value",
                "PnL": "P&L",
            }
        )
        .sort_values("Current Value", ascending=False)
        .reset_index(drop=True)
    )
    grouped["Weight %"] = grouped["Current Value"] / total_current * 100
    grouped["Return %"] = (grouped["P&L"] / grouped["Invested Value"].replace(0, pd.NA)) * 100
    return grouped


def format_market_snapshot(info: dict) -> str | None:
    if info.get("Data source") != "Yahoo Finance":
        return None

    parts = [
        f"- Company: {info.get('Company', '—')}",
        f"- Sector: {info.get('Sector', 'Unknown')} / {info.get('Industry', 'Unknown')}",
        f"- Yahoo price: ₹{info['Yahoo price']:,.2f}" if info.get("Yahoo price") else None,
        f"- Trailing P/E: {info['P/E']:.2f}" if info.get("P/E") else None,
        f"- 52-week range: {info['52-week range']}" if info.get("52-week range") else None,
    ]
    text = "\n".join(part for part in parts if part)
    return text or None


def coverage_stats(enriched: pd.DataFrame) -> dict:
    found = int((enriched["Data source"] == "Yahoo Finance").sum())
    total = len(enriched)
    sector_mapped = int((enriched["Sector"] != "Unknown").sum())
    unknown_sector = enriched.loc[enriched["Sector"] == "Unknown", "Symbol"].tolist()
    not_found = enriched.loc[enriched["Data source"] == "Not found", "Symbol"].tolist()
    return {
        "found": found,
        "total": total,
        "missing": total - found,
        "coverage_pct": round(found / total * 100, 1) if total else 0,
        "sector_mapped": sector_mapped,
        "sector_mapped_pct": round(sector_mapped / total * 100, 1) if total else 0,
        "unknown_sector_symbols": unknown_sector,
        "not_found_symbols": not_found,
    }


def clear_market_cache() -> None:
    for key in ("market_symbols_key", "market_enriched", "market_sectors", "market_coverage"):
        st.session_state.pop(key, None)
    fetch_market_info_for_symbols.clear()


def get_cached_market_context(merged: pd.DataFrame, summary: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict] | None:
    key = portfolio_symbols_key(merged["Symbol"].tolist())
    if (
        st.session_state.get("market_symbols_key") == key
        and "market_enriched" in st.session_state
        and "market_sectors" in st.session_state
        and "market_coverage" in st.session_state
    ):
        return (
            st.session_state.market_enriched,
            st.session_state.market_sectors,
            st.session_state.market_coverage,
        )
    return None


def fetch_and_cache_market_context(merged: pd.DataFrame, summary: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    key = portfolio_symbols_key(merged["Symbol"].tolist())
    enriched = enrich_holdings(merged)
    sectors = sector_summary(enriched, summary["total_current"])
    coverage = coverage_stats(enriched)
    st.session_state.market_symbols_key = key
    st.session_state.market_enriched = enriched
    st.session_state.market_sectors = sectors
    st.session_state.market_coverage = coverage
    return enriched, sectors, coverage
