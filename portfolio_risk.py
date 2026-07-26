"""Portfolio risk metrics: beta, Sharpe ratio, and correlation."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st

import stock_analyzer as san
import yahoo_client as yahoo

DEFAULT_LOOKBACK_DAYS = 252
DEFAULT_RISK_FREE_RATE = 0.07  # ~India 10Y govt yield assumption (annual)
TRADING_DAYS = 252


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
def fetch_daily_returns(
    tickers: tuple[str, ...],
    days: int = DEFAULT_LOOKBACK_DAYS,
) -> pd.DataFrame:
    """Daily % returns for symbols + Nifty benchmark."""
    end = datetime.now()
    start = end - timedelta(days=days + 30)
    frames: dict[str, pd.Series] = {}

    for ticker in tickers:
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
        ret = close.pct_change().dropna()
        label = ticker.replace(".NS", "").replace("^", "NIFTY")
        frames[label] = ret

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, axis=1).dropna(how="all").tail(days)
    return out


def compute_portfolio_risk(
    merged: pd.DataFrame,
    *,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    top_n_corr: int = 10,
) -> dict:
    """
    Compute weighted portfolio beta, Sharpe ratio, and correlation matrix
    for top holdings by weight.
    """
    symbols = merged.head(top_n_corr)["Symbol"].tolist()
    yahoo_symbols = [san.to_yahoo_ticker(s) for s in symbols]
    tickers = tuple(yahoo_symbols + ["^NSEI"])

    returns = fetch_daily_returns(tickers, days=lookback_days)
    if returns.empty or len(returns) < 20:
        return {"ok": False, "error": "Insufficient price history for risk math."}

    nifty_col = next((c for c in returns.columns if "NIFTY" in c.upper() or c == "NSEI"), None)
    stock_cols = [c for c in returns.columns if c != nifty_col]

    # Map return columns back to portfolio symbols
    sym_map: dict[str, str] = {}
    for sym in symbols:
        base = sym.upper()
        for col in stock_cols:
            if col.upper() == base or col.upper().replace(".NS", "") == base:
                sym_map[sym] = col
                break

    weights: dict[str, float] = {}
    total_w = 0.0
    for _, row in merged.iterrows():
        sym = row["Symbol"]
        if sym in sym_map:
            weights[sym] = float(row["Weight %"])
            total_w += weights[sym]

    if total_w <= 0:
        return {"ok": False, "error": "No overlapping holdings with price data."}

    # Normalize weights over available names
    for sym in weights:
        weights[sym] /= total_w

    port_returns = pd.Series(0.0, index=returns.index)
    stock_betas: list[dict] = []
    for sym, col in sym_map.items():
        w = weights.get(sym, 0)
        port_returns += returns[col] * w
        if nifty_col and nifty_col in returns.columns:
            aligned = pd.concat([returns[col], returns[nifty_col]], axis=1).dropna()
            if len(aligned) >= 20:
                cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])
                var_m = cov[1, 1]
                beta = float(cov[0, 1] / var_m) if var_m > 0 else None
            else:
                beta = None
        else:
            beta = None
        stock_betas.append({"Symbol": sym, "Weight %": weights[sym] * total_w, "Beta vs Nifty": beta})

    portfolio_beta = None
    if nifty_col and nifty_col in returns.columns:
        aligned = pd.concat([port_returns, returns[nifty_col]], axis=1).dropna()
        if len(aligned) >= 20:
            cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])
            var_m = cov[1, 1]
            portfolio_beta = float(cov[0, 1] / var_m) if var_m > 0 else None

    # Sharpe (annualized)
    daily_rf = (1 + risk_free_rate) ** (1 / TRADING_DAYS) - 1
    excess = port_returns - daily_rf
    sharpe = None
    if excess.std() > 0:
        sharpe = float(excess.mean() / excess.std() * np.sqrt(TRADING_DAYS))

    ann_return = float(port_returns.mean() * TRADING_DAYS * 100)
    ann_vol = float(port_returns.std() * np.sqrt(TRADING_DAYS) * 100)

    corr_cols = [sym_map[s] for s in symbols if s in sym_map][:top_n_corr]
    corr = returns[corr_cols].corr() if len(corr_cols) >= 2 else pd.DataFrame()
    # Rename columns to symbols
    if not corr.empty:
        rename = {v: k for k, v in sym_map.items()}
        corr = corr.rename(columns=rename, index=rename)

    return {
        "ok": True,
        "portfolio_beta": round(portfolio_beta, 2) if portfolio_beta is not None else None,
        "sharpe_ratio": round(sharpe, 2) if sharpe is not None else None,
        "annualized_return_pct": round(ann_return, 1),
        "annualized_vol_pct": round(ann_vol, 1),
        "risk_free_rate_pct": risk_free_rate * 100,
        "lookback_days": lookback_days,
        "stock_betas": pd.DataFrame(stock_betas),
        "correlation": corr,
        "coverage_symbols": len(sym_map),
        "total_symbols": len(symbols),
    }


def build_risk_context(risk: dict) -> str:
    if not risk.get("ok"):
        return f"Portfolio risk metrics: {risk.get('error', 'unavailable')}"

    lines = [
        "Portfolio risk metrics:",
        f"- Portfolio beta vs Nifty: {risk.get('portfolio_beta', '—')}",
        f"- Sharpe ratio (~{risk['lookback_days']}d, rf {risk['risk_free_rate_pct']:.0f}%): "
        f"{risk.get('sharpe_ratio', '—')}",
        f"- Annualized return / vol: {risk.get('annualized_return_pct', 0):+.1f}% / "
        f"{risk.get('annualized_vol_pct', 0):.1f}%",
        f"- Price coverage: {risk.get('coverage_symbols')}/{risk.get('total_symbols')} top holdings",
    ]
    betas = risk.get("stock_betas")
    if isinstance(betas, pd.DataFrame) and not betas.empty:
        lines.append("- Stock betas vs Nifty (top names):")
        for _, row in betas.head(8).iterrows():
            b = row.get("Beta vs Nifty")
            b_txt = f"{b:.2f}" if pd.notna(b) else "—"
            lines.append(f"  · {row['Symbol']}: β {b_txt} ({row['Weight %']:.1f}% wt)")
    return "\n".join(lines)
