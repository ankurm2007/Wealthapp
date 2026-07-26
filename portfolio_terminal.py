"""OpenBB-powered research terminal layer (free Bloomberg-style data for Indian equities)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd
import streamlit as st

import stock_analyzer as san

OPENBB_AVAILABLE = False
_obb = None
_OPENBB_IMPORT_ERROR: str | None = None

try:
    from openbb import obb as _obb

    OPENBB_AVAILABLE = True
except ImportError:
    _OPENBB_IMPORT_ERROR = "OpenBB is not installed."
except (PermissionError, OSError) as exc:
    _OPENBB_IMPORT_ERROR = (
        "OpenBB cannot run in this environment (read-only filesystem). "
        "Install and use it locally instead."
    )
except Exception as exc:
    _OPENBB_IMPORT_ERROR = f"OpenBB failed to load: {exc}"


@dataclass
class TerminalMetrics:
    symbol: str
    name: str | None = None
    pe_ratio: float | None = None
    forward_pe: float | None = None
    peg_ratio: float | None = None
    price_to_book: float | None = None
    dividend_yield: float | None = None
    profit_margin: float | None = None
    operating_margin: float | None = None
    revenue_growth: float | None = None
    earnings_growth: float | None = None
    return_on_equity: float | None = None
    debt_to_equity: float | None = None
    beta: float | None = None
    market_cap: float | None = None
    price_return_1y: float | None = None
    enterprise_to_ebitda: float | None = None


def is_available() -> bool:
    return OPENBB_AVAILABLE


def unavailability_reason() -> str:
    if OPENBB_AVAILABLE:
        return ""
    if _OPENBB_IMPORT_ERROR:
        return _OPENBB_IMPORT_ERROR
    return "OpenBB is not installed."


def _yahoo_symbol(symbol: str) -> str:
    return san.to_yahoo_ticker(symbol)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fetch_metrics(symbol: str) -> TerminalMetrics | None:
    if not OPENBB_AVAILABLE or _obb is None:
        return None
    yahoo = _yahoo_symbol(symbol)
    try:
        out = _obb.equity.fundamental.metrics(yahoo, provider="yfinance")
        if not out.results:
            return None
        row = out.results[0]
        return TerminalMetrics(
            symbol=symbol,
            name=getattr(row, "name", None),
            pe_ratio=_safe_float(getattr(row, "pe_ratio", None)),
            forward_pe=_safe_float(getattr(row, "forward_pe", None)),
            peg_ratio=_safe_float(getattr(row, "peg_ratio", None)),
            price_to_book=_safe_float(getattr(row, "price_to_book", None)),
            dividend_yield=_safe_float(getattr(row, "dividend_yield", None)),
            profit_margin=_safe_float(getattr(row, "profit_margin", None)),
            operating_margin=_safe_float(getattr(row, "operating_margin", None)),
            revenue_growth=_safe_float(getattr(row, "revenue_growth", None)),
            earnings_growth=_safe_float(getattr(row, "earnings_growth", None)),
            return_on_equity=_safe_float(getattr(row, "return_on_equity", None)),
            debt_to_equity=_safe_float(getattr(row, "debt_to_equity", None)),
            beta=_safe_float(getattr(row, "beta", None)),
            market_cap=_safe_float(getattr(row, "market_cap", None)),
            price_return_1y=_safe_float(getattr(row, "price_return_1y", None)),
            enterprise_to_ebitda=_safe_float(getattr(row, "enterprise_to_ebitda", None)),
        )
    except Exception:
        return None


def fetch_profile(symbol: str) -> dict:
    if not OPENBB_AVAILABLE or _obb is None:
        return {}
    try:
        out = _obb.equity.profile(_yahoo_symbol(symbol), provider="yfinance")
        if not out.results:
            return {}
        row = out.results[0]
        return {
            "name": getattr(row, "name", symbol),
            "sector": getattr(row, "sector", None),
            "industry": getattr(row, "industry", None),
            "description": (getattr(row, "long_description", None) or "")[:400],
            "website": getattr(row, "website", None),
            "employees": getattr(row, "employees", None),
        }
    except Exception:
        return {}


def fetch_market_pulse() -> dict:
    """Nifty 50 index snapshot for macro context."""
    if not OPENBB_AVAILABLE or _obb is None:
        return {}
    try:
        out = _obb.equity.price.quote("^NSEI", provider="yfinance")
        if not out.results:
            return {}
        row = out.results[0]
        return {
            "name": getattr(row, "name", "NIFTY 50"),
            "prev_close": _safe_float(getattr(row, "prev_close", None)),
            "year_high": _safe_float(getattr(row, "year_high", None)),
            "year_low": _safe_float(getattr(row, "year_low", None)),
            "ma_50d": _safe_float(getattr(row, "ma_50d", None)),
            "ma_200d": _safe_float(getattr(row, "ma_200d", None)),
        }
    except Exception:
        return {}


def fetch_news_headlines(symbol: str, limit: int = 3) -> list[str]:
    if not OPENBB_AVAILABLE or _obb is None:
        return []
    try:
        out = _obb.news.company(symbol, provider="yfinance", limit=limit)
        if not out.results:
            return []
        headlines = []
        for item in out.results[:limit]:
            title = getattr(item, "title", None) or getattr(item, "headline", None)
            if title:
                headlines.append(str(title).strip())
        return headlines
    except Exception:
        return []


@st.cache_data(ttl="4h", show_spinner=False)
def build_terminal_snapshot(symbols: tuple[str, ...], news_for: tuple[str, ...]) -> dict:
    """Fetch OpenBB metrics for holdings + market pulse + news."""
    if not OPENBB_AVAILABLE:
        return {"available": False, "metrics": [], "pulse": {}, "news": {}}

    metrics: list[dict] = []
    with ThreadPoolExecutor(max_workers=min(4, max(1, len(symbols)))) as pool:
        rows = list(pool.map(fetch_metrics, symbols))
    for row in rows:
        if row:
            metrics.append(asdict(row))

    pulse = fetch_market_pulse()
    news: dict[str, list[str]] = {}
    for sym in news_for[:5]:
        headlines = fetch_news_headlines(sym, limit=2)
        if headlines:
            news[sym] = headlines

    return {
        "available": True,
        "metrics": metrics,
        "pulse": pulse,
        "news": news,
    }


def metrics_dataframe(snapshot: dict, merged: pd.DataFrame) -> pd.DataFrame:
    if not snapshot.get("metrics"):
        return pd.DataFrame()
    df = pd.DataFrame(snapshot["metrics"])
    if "symbol" in df.columns:
        df = df.rename(columns={"symbol": "Symbol"})
    if not merged.empty and "Symbol" in merged.columns:
        weights = merged[["Symbol", "Weight %", "Return %"]].drop_duplicates("Symbol")
        df = df.merge(weights, on="Symbol", how="left")
    if "Weight %" in df.columns:
        return df.sort_values("Weight %", ascending=False, na_position="last")
    return df


def _fmt_pct(value: float | None, scale: float = 100) -> str:
    if value is None:
        return "—"
    return f"{value * scale:+.1f}%" if scale != 1 else f"{value:+.1f}%"


def build_terminal_context(snapshot: dict, merged: pd.DataFrame) -> str:
    if not snapshot.get("available"):
        reason = unavailability_reason()
        return f"OpenBB terminal data unavailable. {reason}"

    lines = ["OpenBB research terminal (free data layer, yfinance provider):"]

    pulse = snapshot.get("pulse") or {}
    if pulse.get("prev_close"):
        lines.append(
            f"- Nifty 50: last close {pulse['prev_close']:,.0f}, "
            f"52w range {pulse.get('year_low', 0):,.0f}–{pulse.get('year_high', 0):,.0f}, "
            f"50d MA {pulse.get('ma_50d', 0):,.0f}, 200d MA {pulse.get('ma_200d', 0):,.0f}"
        )

    lines.append("- Holding fundamentals (OpenBB metrics):")
    df = metrics_dataframe(snapshot, merged)
    for _, row in df.head(12).iterrows():
        parts = [f"  · {row['Symbol']} ({row.get('Weight %', 0):.1f}% wt)"]
        if pd.notna(row.get("pe_ratio")):
            parts.append(f"P/E {row['pe_ratio']:.1f}")
        if pd.notna(row.get("forward_pe")):
            parts.append(f"fwd {row['forward_pe']:.1f}")
        if pd.notna(row.get("peg_ratio")):
            parts.append(f"PEG {row['peg_ratio']:.2f}")
        if pd.notna(row.get("profit_margin")):
            parts.append(f"margin {row['profit_margin']*100:.1f}%")
        if pd.notna(row.get("revenue_growth")):
            parts.append(f"rev gr {_fmt_pct(row['revenue_growth'])}")
        if pd.notna(row.get("earnings_growth")):
            parts.append(f"earn gr {_fmt_pct(row['earnings_growth'])}")
        if pd.notna(row.get("return_on_equity")):
            parts.append(f"ROE {_fmt_pct(row['return_on_equity'])}")
        if pd.notna(row.get("debt_to_equity")):
            parts.append(f"D/E {row['debt_to_equity']:.1f}")
        if pd.notna(row.get("price_return_1y")):
            parts.append(f"1y price {_fmt_pct(row['price_return_1y'])}")
        lines.append(" · ".join(parts))

    news = snapshot.get("news") or {}
    if news:
        lines.append("- Recent headlines (top holdings):")
        for sym, headlines in news.items():
            for headline in headlines:
                lines.append(f"  · {sym}: {headline[:140]}")

    return "\n".join(lines)
