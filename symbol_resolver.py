"""Map Groww company names, tickers, and ISINs to NSE trading symbols."""

from __future__ import annotations

import re
from functools import lru_cache
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

# Normalized company name -> NSE symbol
NAME_TO_SYMBOL: dict[str, str] = {
    "RELIANCE INDUSTRIES": "RELIANCE",
    "RELIANCE INDUSTRIES LTD": "RELIANCE",
    "ADANI ENTERPRISES": "ADANIENT",
    "ADANI ENTERPRISES LIMITED": "ADANIENT",
    "BHARAT ELECTRONICS": "BEL",
    "BHARAT ELECTRONICS LTD": "BEL",
    "GUJARAT MINERAL DEV CORP": "GMDCLTD",
    "GUJARAT MINERAL DEVELOPMENT CORPORATION": "GMDCLTD",
    "GMDC": "GMDCLTD",
    "TATA CONSULTANCY SERV LT": "TCS",
    "TATA CONSULTANCY SERVICES": "TCS",
    "TATA CONSULTANCY SERVICES LTD": "TCS",
    "ADANI POWER": "ADANIPOWER",
    "ADANI POWER LTD": "ADANIPOWER",
    "AMARA RAJA ENERGY MOB LTD": "ARE&M",
    "AMARA RAJA ENERGY MOBILITIES": "ARE&M",
    "AJANTA PHARMA": "AJANTPHARM",
    "AJANTA PHARMA LIMITED": "AJANTPHARM",
    "NATCO PHARMA LTD": "NATCOPHARM",
    "NATCO PHARMA": "NATCOPHARM",
    "INDIAN OIL CORP LTD": "IOC",
    "INDIAN OIL CORPORATION": "IOC",
    "JIO FIN SERVICES LTD": "JIOFIN",
    "JIO FINANCIAL SERVICES": "JIOFIN",
    "APOLLO MICRO SYSTEMS LTD": "APOLLO",
    "APOLLO MICRO SYSTEMS": "APOLLO",
    "DATA PATTERNS INDIA LTD": "DATAPATTNS",
    "DATA PATTERNS (INDIA)": "DATAPATTNS",
    "HDFC BANK LTD": "HDFCBANK",
    "HDFC BANK": "HDFCBANK",
    "YES BANK": "YESBANK",
    "YES BANK LIMITED": "YESBANK",
    "POWER FIN CORP LTD": "PFC",
    "POWER FINANCE CORPORATION": "PFC",
    "DLF": "DLF",
    "DLF LIMITED": "DLF",
    "INFOSYS": "INFY",
    "INFOSYS LIMITED": "INFY",
    "ITC LTD": "ITC",
    "ITC": "ITC",
    "STATE BANK OF INDIA": "SBIN",
    "HINDUSTAN UNILEVER": "HINDUNILVR",
    "ICICI BANK": "ICICIBANK",
    "KOTAK MAHINDRA BANK": "KOTAKBANK",
    "AXIS BANK": "AXISBANK",
    "BANK OF BARODA": "BANKBARODA",
    "WIPRO": "WIPRO",
    "HCL TECHNOLOGIES": "HCLTECH",
    "BHARTI AIRTEL": "BHARTIARTL",
    "MARUTI SUZUKI": "MARUTI",
    "SUN PHARMACEUTICAL": "SUNPHARMA",
    "TITAN COMPANY": "TITAN",
    "LARSEN & TOUBRO": "LT",
    "NTPC": "NTPC",
    "POWER GRID CORPORATION": "POWERGRID",
    "COAL INDIA": "COALINDIA",
    "OIL & NATURAL GAS CORPORATION": "ONGC",
    "TATA STEEL": "TATASTEEL",
    "BAJAJ FINANCE": "BAJFINANCE",
    "MAHINDRA & MAHINDRA": "M&M",
}

ETF_NAME_TO_SYMBOL: dict[str, str] = {
    "NIP IND ETF NIFTY BEES": "NIFTYBEES",
    "NIP IND ETF GOLD BEES": "GOLDBEES",
    "NIP IND ETF LONGTERM GILT": "LTGILTBEES",
    "NIP IND ETF CONSUMPTION": "CONSUMBEES",
    "NIP IND ETF BANK BEES": "BANKBEES",
    "NIP IND ETF IT BEES": "ITBEES",
    "NIP IND ETF JUNIOR BEES": "JUNIORBEES",
    "NIP IND ETF LIQUID BEES": "LIQUIDBEES",
}

TICKER_PATTERN = re.compile(r"^[A-Z0-9&.\-]{1,20}$")
ISIN_PATTERN = re.compile(r"^IN[A-Z0-9]{10}$")

NSE_EQUITY_URL = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
NSE_ETF_URL = "https://nsearchives.nseindia.com/content/equities/eq_etfseclist.csv"
CACHE_DIR = Path(__file__).parent / "data"
ISIN_CACHE_PATH = CACHE_DIR / "isin_to_symbol.csv"


def normalize_label(label: str) -> str:
    text = label.upper().strip()
    text = text.replace(".", "")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(
        r"\b(LTD|LIMITED|LTD|INC|CORP|CORPORATION|PVT|PRIVATE|CO)\b",
        "",
        text,
    )
    text = re.sub(r"\s+", " ", text).strip(" ,")
    return text


def looks_like_isin(label: str) -> bool:
    clean = label.strip().upper().replace(".NS", "").replace(".BO", "").replace(".BSE", "")
    return bool(ISIN_PATTERN.match(clean))


def looks_like_nse_ticker(label: str) -> bool:
    clean = label.strip().upper()
    if " " in clean or looks_like_isin(clean):
        return False
    # ISINs wrongly treated as tickers are 12 chars starting with IN
    if clean.startswith("IN") and len(clean) == 12:
        return False
    return bool(TICKER_PATTERN.match(clean))


def _download_isin_map() -> pd.DataFrame:
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "*/*",
        "Referer": "https://www.nseindia.com/",
    }
    rows: list[dict] = []

    equity = pd.read_csv(StringIO(requests.get(NSE_EQUITY_URL, headers=headers, timeout=30).text))
    equity.columns = [c.strip() for c in equity.columns]
    for _, row in equity.iterrows():
        isin = str(row.get("ISIN NUMBER", "")).strip().upper()
        symbol = str(row.get("SYMBOL", "")).strip().upper()
        name = str(row.get("NAME OF COMPANY", "")).strip()
        if isin and symbol and isin != "NAN":
            rows.append({"isin": isin, "symbol": symbol, "name": name})

    etf = pd.read_csv(StringIO(requests.get(NSE_ETF_URL, headers=headers, timeout=30).text))
    etf.columns = [c.strip() for c in etf.columns]
    for _, row in etf.iterrows():
        isin = str(row.get("ISINNumber", "")).strip().upper()
        symbol = str(row.get("Symbol", "")).strip().upper()
        name = str(row.get("SecurityName", "")).strip()
        if isin and symbol and isin != "NAN":
            rows.append({"isin": isin, "symbol": symbol, "name": name})

    df = pd.DataFrame(rows).drop_duplicates(subset=["isin"], keep="first")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(ISIN_CACHE_PATH, index=False)
    return df


@lru_cache(maxsize=1)
def load_isin_map() -> dict[str, str]:
    """Load ISIN → NSE symbol map (cached on disk + memory)."""
    try:
        if ISIN_CACHE_PATH.exists():
            df = pd.read_csv(ISIN_CACHE_PATH)
        else:
            df = _download_isin_map()
        return {
            str(row["isin"]).strip().upper(): str(row["symbol"]).strip().upper()
            for _, row in df.iterrows()
            if pd.notna(row.get("isin")) and pd.notna(row.get("symbol"))
        }
    except Exception:
        return {}


def resolve_isin(isin: str) -> str | None:
    clean = isin.strip().upper().replace(".NS", "").replace(".BO", "").replace(".BSE", "")
    mapping = load_isin_map()
    if clean in mapping:
        return mapping[clean]
    # Retry once with a fresh download if cache miss looks complete but symbol missing.
    try:
        load_isin_map.cache_clear()
        if ISIN_CACHE_PATH.exists():
            ISIN_CACHE_PATH.unlink()
        mapping = load_isin_map()
        return mapping.get(clean)
    except Exception:
        return None


def resolve_nse_symbol(label: str) -> str:
    raw = label.strip().upper()
    if not raw:
        return raw

    # Strip accidental Yahoo suffix before detection.
    bare = raw.replace(".NS", "").replace(".BO", "").replace(".BSE", "")

    if looks_like_isin(bare):
        resolved = resolve_isin(bare)
        if resolved:
            return resolved
        return bare  # keep ISIN visible rather than inventing INE....NS

    if looks_like_nse_ticker(bare):
        return bare

    normalized = normalize_label(raw)

    if normalized in NAME_TO_SYMBOL:
        return NAME_TO_SYMBOL[normalized]

    for etf_name, symbol in ETF_NAME_TO_SYMBOL.items():
        if etf_name in raw or etf_name in normalized:
            return symbol

    for key, symbol in sorted(NAME_TO_SYMBOL.items(), key=lambda item: len(item[0]), reverse=True):
        if key in normalized or normalized in key:
            return symbol

    # Last resort: compact to a pseudo-ticker (still may fail on Yahoo).
    return re.sub(r"[^A-Z0-9&]", "", normalized)[:20] or bare


def normalize_portfolio_symbols(portfolio_df: pd.DataFrame) -> pd.DataFrame:
    df = portfolio_df.copy()
    if "Company Name" not in df.columns:
        df["Company Name"] = df["Symbol"]
    # Prefetch ISIN map once for the batch.
    load_isin_map()
    df["Symbol"] = df["Symbol"].apply(lambda value: resolve_nse_symbol(str(value)))
    return df
