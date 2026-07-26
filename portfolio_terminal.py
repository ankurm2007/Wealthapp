"""OpenBB-powered research terminal layer (free Bloomberg-style data for Indian equities)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd
import streamlit as st

import stock_analyzer as san

_obb = None
_OPENBB_TRIED = False
_OPENBB_IMPORT_ERROR: str | None = None


def _ensure_openbb() -> bool:
    """Lazy-load OpenBB so app startup is not blocked by a heavy import."""
    global _obb, _OPENBB_TRIED, _OPENBB_IMPORT_ERROR
    if _OPENBB_TRIED:
        return _obb is not None
    _OPENBB_TRIED = True
    try:
        from openbb import obb as obb_mod

        _obb = obb_mod
        return True
    except ImportError:
        _OPENBB_IMPORT_ERROR = "OpenBB is not installed."
    except (PermissionError, OSError):
        _OPENBB_IMPORT_ERROR = (
            "OpenBB cannot run in this environment (read-only filesystem). "
            "Install and use it locally instead."
        )
    except Exception as exc:
        _OPENBB_IMPORT_ERROR = f"OpenBB failed to load: {exc}"
    _obb = None
    return False


def is_available() -> bool:
    return _ensure_openbb()


def unavailability_reason() -> str:
    _ensure_openbb()
    if _obb is not None:
        return ""
    if _OPENBB_IMPORT_ERROR:
        return _OPENBB_IMPORT_ERROR
    return "OpenBB is not installed."


OPENBB_AVAILABLE = False  # legacy alias; prefer is_available()


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


@dataclass
class TechnicalMetrics:
    symbol: str
    yahoo_ticker: str | None = None
    last_close: float | None = None
    rsi_14: float | None = None
    sma_50: float | None = None
    momentum_20d_pct: float | None = None
    price_vs_sma50_pct: float | None = None
    source: str | None = None
    error: str | None = None


# Asset types that typically lack reliable equity OHLC on OpenBB/Yahoo.
NON_EQUITY_ASSET_TYPES = frozenset(
    {
        "gold",
        "sgb",
        "cash",
        "mutual_fund",
        "mf",
        "debt",
        "bond",
        "fd",
        "fixed_deposit",
        "commodity",
        "crypto",
    }
)


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
    if not _ensure_openbb():
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
    if not _ensure_openbb():
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
    if not _ensure_openbb():
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
    if not _ensure_openbb():
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


def _rsi_wilder(close: pd.Series, period: int = 14) -> float | None:
    if close is None or len(close) < period + 1:
        return None
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    last_gain = float(avg_gain.iloc[-1])
    last_loss = float(avg_loss.iloc[-1])
    if last_loss == 0:
        return 100.0 if last_gain > 0 else 50.0
    rs = last_gain / last_loss
    return round(100.0 - (100.0 / (1.0 + rs)), 2)


def _close_series_from_openbb(yahoo: str, lookback_days: int = 120) -> pd.Series | None:
    """Pull daily closes via OpenBB equity.price.historical (yfinance provider)."""
    if not _ensure_openbb():
        return None
    try:
        end = pd.Timestamp.today().normalize()
        start = end - pd.Timedelta(days=lookback_days + 10)
        out = _obb.equity.price.historical(
            yahoo,
            start_date=start.strftime("%Y-%m-%d"),
            end_date=end.strftime("%Y-%m-%d"),
            provider="yfinance",
        )
        df = out.to_dataframe() if hasattr(out, "to_dataframe") else None
        if df is None or df.empty:
            if getattr(out, "results", None):
                df = pd.DataFrame([r.model_dump() if hasattr(r, "model_dump") else vars(r) for r in out.results])
            else:
                return None
        # Normalize column names across OpenBB versions
        colmap = {c.lower(): c for c in df.columns}
        close_col = colmap.get("close") or colmap.get("adj_close") or colmap.get("adj close")
        if not close_col:
            return None
        close = pd.to_numeric(df[close_col], errors="coerce").dropna()
        return close if len(close) >= 15 else None
    except Exception:
        return None


def _close_series_from_yahoo(yahoo: str, lookback_days: int = 120) -> pd.Series | None:
    """Fallback OHLCV via existing Yahoo helper."""
    try:
        df = san.fetch_price_data(yahoo, days=lookback_days)
        if df is None or df.empty or "Close" not in df.columns:
            return None
        close = pd.to_numeric(df["Close"], errors="coerce").dropna()
        return close if len(close) >= 15 else None
    except Exception:
        return None


def _metrics_from_close(symbol: str, yahoo: str, close: pd.Series, source: str) -> TechnicalMetrics:
    last_close = _safe_float(close.iloc[-1])
    rsi = _rsi_wilder(close, 14)
    sma_50 = round(float(close.tail(50).mean()), 2) if len(close) >= 50 else (
        round(float(close.mean()), 2) if len(close) >= 20 else None
    )
    momentum = None
    if len(close) >= 21 and float(close.iloc[-21]) != 0:
        momentum = round((float(close.iloc[-1]) / float(close.iloc[-21]) - 1.0) * 100.0, 2)
    vs_sma = None
    if last_close is not None and sma_50:
        vs_sma = round((last_close / sma_50 - 1.0) * 100.0, 2)
    return TechnicalMetrics(
        symbol=symbol,
        yahoo_ticker=yahoo,
        last_close=round(last_close, 2) if last_close is not None else None,
        rsi_14=rsi,
        sma_50=sma_50,
        momentum_20d_pct=momentum,
        price_vs_sma50_pct=vs_sma,
        source=source,
    )


def is_enrichable_asset(asset_type: str | None) -> bool:
    """Return False for non-standard assets that should skip market enrichment."""
    if not asset_type:
        return True
    key = str(asset_type).strip().lower().replace(" ", "_").replace("-", "_")
    return key not in NON_EQUITY_ASSET_TYPES and "mutual" not in key and "fund" not in key


def fetch_technicals(symbol: str, asset_type: str | None = None) -> TechnicalMetrics:
    """
    Fetch RSI-14, SMA-50, and 20d momentum for a symbol.
    Prefers OpenBB historicals; falls back to Yahoo. Never raises.
    """
    if not is_enrichable_asset(asset_type):
        return TechnicalMetrics(
            symbol=symbol,
            error=f"skipped non-equity asset_type={asset_type}",
        )

    try:
        yahoo = _yahoo_symbol(symbol)
    except Exception as exc:
        return TechnicalMetrics(symbol=symbol, error=f"ticker resolve failed: {exc}")

    close = _close_series_from_openbb(yahoo)
    source = "openbb"
    if close is None:
        close = _close_series_from_yahoo(yahoo)
        source = "yahoo"
    if close is None:
        return TechnicalMetrics(
            symbol=symbol,
            yahoo_ticker=yahoo,
            error="no price history",
        )
    try:
        return _metrics_from_close(symbol, yahoo, close, source)
    except Exception as exc:
        return TechnicalMetrics(symbol=symbol, yahoo_ticker=yahoo, error=str(exc)[:160])


@st.cache_data(ttl="2h", show_spinner=False)
def enrich_symbols_technicals(
    symbols: tuple[str, ...],
    asset_types: tuple[str | None, ...] = (),
) -> dict:
    """
    Batch-enrich unique symbols with technical metrics.
    Returns {available, metrics: list[dict], warnings: list[str]}.
    """
    warnings: list[str] = []
    if not symbols:
        return {"available": False, "metrics": [], "warnings": ["No symbols to enrich."]}

    type_map: dict[str, str | None] = {}
    if asset_types and len(asset_types) == len(symbols):
        type_map = dict(zip(symbols, asset_types))

    metrics: list[dict] = []
    workers = min(6, max(1, len(symbols)))

    def _one(sym: str) -> TechnicalMetrics:
        return fetch_technicals(sym, type_map.get(sym))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(_one, symbols))

    for row in rows:
        payload = asdict(row)
        metrics.append(payload)
        if row.error:
            warnings.append(f"{row.symbol}: {row.error}")

    ok = sum(1 for m in metrics if m.get("rsi_14") is not None or m.get("sma_50") is not None)
    if ok == 0 and not _ensure_openbb():
        warnings.append(unavailability_reason() or "OpenBB unavailable; Yahoo fallback also failed.")

    return {
        "available": ok > 0,
        "metrics": metrics,
        "warnings": warnings,
        "coverage": {"ok": ok, "total": len(symbols)},
    }


@st.cache_data(ttl="4h", show_spinner=False)
def build_terminal_snapshot(symbols: tuple[str, ...], news_for: tuple[str, ...]) -> dict:
    """Fetch OpenBB metrics for holdings + market pulse + news."""
    if not _ensure_openbb():
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
