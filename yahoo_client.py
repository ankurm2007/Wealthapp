"""Quiet, defensive wrappers around yfinance (Yahoo often rate-limits / 401s)."""

from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd

_CONFIGURED = False


def configure_yfinance() -> None:
    """Suppress noisy Yahoo 401/crumb errors in the UI and logs."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True

    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    logging.getLogger("peewee").setLevel(logging.CRITICAL)

    try:
        import yfinance as yf

        yf.config.debug.hide_exceptions = True
        yf.config.debug.logging = False
        yf.config.network.retries = 1
    except Exception:
        pass


def safe_download(
    tickers: str | list[str],
    *,
    start: str | None = None,
    end: str | None = None,
    period: str | None = None,
    auto_adjust: bool = True,
    progress: bool = False,
) -> pd.DataFrame:
    configure_yfinance()
    try:
        import yfinance as yf

        kwargs: dict = {
            "progress": progress,
            "auto_adjust": auto_adjust,
            "threads": False,
        }
        if period:
            kwargs["period"] = period
        else:
            kwargs["start"] = start
            kwargs["end"] = end or datetime.now().strftime("%Y-%m-%d")
        return yf.download(tickers, **kwargs)
    except Exception:
        return pd.DataFrame()


def safe_ticker_info(ticker: str) -> dict | None:
    configure_yfinance()
    try:
        import yfinance as yf

        info = yf.Ticker(ticker).info
        if not info or info.get("regularMarketPrice") is None:
            return None
        return info
    except Exception:
        return None


def safe_quarterly_income(ticker: str) -> pd.DataFrame:
    configure_yfinance()
    try:
        import yfinance as yf

        df = yf.Ticker(ticker).quarterly_income_stmt
        if df is None or df.empty:
            return pd.DataFrame()
        return df
    except Exception:
        return pd.DataFrame()
