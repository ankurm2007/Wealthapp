#!/usr/bin/env python3
"""
Multi-agent stock analysis pipeline combining yfinance, Alpha Vantage, Groq, and Gemini.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime, timedelta

import pandas as pd
import requests
import yahoo_client as yahoo
from dotenv import load_dotenv

load_dotenv()

TICKERS = ["RELIANCE.NS", "GMDCLTD.NS", "BEL.NS"]
ALPHA_VANTAGE_RSI_URL = "https://www.alphavantage.co/query"
GROQ_MODEL = "llama-3.3-70b-versatile"
GEMINI_MODEL = "gemini-2.5-pro"

# Common broker / shorthand labels that differ from Yahoo NSE tickers.
YAHOO_TICKER_ALIASES = {
    "GMDC": "GMDCLTD",
    "GMDCLTD": "GMDCLTD",
}


def get_env_key(name: str) -> str | None:
    value = os.getenv(name, "").strip()
    if not value or value.startswith("paste_") or "your_" in value.lower():
        return None
    return value


def resolve_api_keys(overrides: dict[str, str] | None = None) -> dict[str, str | None]:
    """Resolve keys from env, with optional Streamlit/secret overrides."""
    overrides = overrides or {}

    def _clean(value: str | None) -> str | None:
        if not value:
            return None
        value = value.strip()
        if not value or value.startswith("paste_") or "your_" in value.lower():
            return None
        return value

    groq = _clean(overrides.get("groq")) or get_env_key("GROQ_API_KEY")
    xai = _clean(overrides.get("xai")) or get_env_key("XAI_API_KEY")
    # Common mistake: xAI key pasted into GROQ slot.
    if groq and groq.startswith("xai-"):
        if not xai:
            xai = groq
        groq = None
    return {
        "alpha": _clean(overrides.get("alpha")) or get_env_key("ALPHA_VANTAGE_API_KEY"),
        "groq": groq,
        "xai": xai,
        "gemini": _clean(overrides.get("gemini")) or get_env_key("GEMINI_API_KEY"),
    }


def to_yahoo_ticker(symbol: str) -> str:
    """Normalize a portfolio/NSE symbol to a Yahoo Finance ticker."""
    clean = symbol.strip().upper()
    for suffix in (".NS", ".BO", ".BSE"):
        if clean.endswith(suffix):
            clean = clean[: -len(suffix)]
            break

    # Resolve ISINs (Groww exports) to NSE symbols before adding .NS
    try:
        import symbol_resolver as sym

        if sym.looks_like_isin(clean):
            resolved = sym.resolve_isin(clean) or sym.resolve_nse_symbol(clean)
            if resolved and not sym.looks_like_isin(resolved):
                clean = resolved
        else:
            clean = sym.resolve_nse_symbol(clean)
    except Exception:
        pass

    clean = YAHOO_TICKER_ALIASES.get(clean, clean)
    return f"{clean}.NS"


def alpha_vantage_symbol_candidates(yahoo_ticker: str) -> list[str]:
    """Build Alpha Vantage symbol variants for Indian listings."""
    base = yahoo_ticker.upper().replace(".NS", "").replace(".BSE", "").replace(".BO", "")
    return list(dict.fromkeys([f"{base}.BSE", f"{base}.NS", f"{base}.BO", base]))


def _flatten_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    return df


def fetch_price_data(ticker: str, days: int = 30) -> pd.DataFrame | None:
    """Fetch recent OHLCV history from Yahoo Finance."""
    end = datetime.now()
    start = end - timedelta(days=days + 5)
    df = yahoo.safe_download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        progress=False,
        auto_adjust=False,
    )
    if df is None or df.empty:
        return None

    df = _flatten_ohlcv_columns(df).tail(days).copy()
    df.index = pd.to_datetime(df.index)
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        return None
    return df


def summarize_price_trend(df: pd.DataFrame) -> dict:
    """Summarize recent price action for downstream LLM prompts."""
    close = df["Close"].astype(float)
    first_close = float(close.iloc[0])
    last_close = float(close.iloc[-1])
    pct_change = ((last_close - first_close) / first_close) * 100 if first_close else 0.0
    avg_volume = float(df["Volume"].astype(float).mean())
    high_30d = float(df["High"].astype(float).max())
    low_30d = float(df["Low"].astype(float).min())
    return {
        "start_close": round(first_close, 2),
        "end_close": round(last_close, 2),
        "pct_change_30d": round(pct_change, 2),
        "avg_volume": round(avg_volume, 0),
        "high_30d": round(high_30d, 2),
        "low_30d": round(low_30d, 2),
        "trend": "up" if pct_change > 0 else "down" if pct_change < 0 else "flat",
    }


def fetch_rsi_alpha_vantage(ticker: str, api_key: str) -> float | None:
    """Fetch latest 14-day RSI from Alpha Vantage."""
    for symbol in alpha_vantage_symbol_candidates(ticker):
        try:
            params = {
                "function": "RSI",
                "symbol": symbol,
                "interval": "daily",
                "time_period": 14,
                "series_type": "close",
                "apikey": api_key,
            }
            response = requests.get(ALPHA_VANTAGE_RSI_URL, params=params, timeout=20)
            response.raise_for_status()
            payload = response.json()

            if "Note" in payload:
                print(f"[WARN] Alpha Vantage rate limit for {ticker} ({symbol}): {payload['Note']}")
                return None
            if "Information" in payload:
                print(f"[WARN] Alpha Vantage info for {ticker} ({symbol}): {payload['Information']}")
                return None
            if "Error Message" in payload:
                print(f"[WARN] Alpha Vantage error for {ticker} ({symbol}): {payload['Error Message']}")
                continue

            rsi_block = payload.get("Technical Analysis: RSI")
            if not rsi_block:
                print(f"[WARN] Alpha Vantage RSI block missing for {ticker} using symbol {symbol}.")
                continue

            latest_date = sorted(rsi_block.keys())[-1]
            rsi_value = float(rsi_block[latest_date]["RSI"])
            print(f"[INFO] Alpha Vantage RSI for {ticker} resolved via {symbol}: {rsi_value:.2f}")
            return rsi_value
        except Exception as exc:
            print(f"[WARN] Alpha Vantage RSI request failed for {ticker} ({symbol}): {exc}")
            continue

    print(f"[WARN] Could not fetch Alpha Vantage RSI for {ticker} with any symbol variant.")
    return None


def dummy_news_headlines(ticker: str) -> list[str]:
    """Return sample headlines for sentiment testing."""
    base = ticker.replace(".NS", "").replace(".BSE", "")
    return [
        f"{base} reports steady quarterly revenue growth amid stable demand.",
        f"Analysts remain divided on {base} valuation after recent market volatility.",
        f"{base} faces margin pressure from input costs but maintains market share.",
        f"Foreign portfolio investors increase exposure to {base} in the latest session.",
        f"Regulatory headlines create near-term uncertainty for {base} investors.",
    ]


def _extract_json_object(text: str) -> dict | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
    return None


def fetch_groq_sentiment(ticker: str, headlines: list[str], api_key: str) -> float | None:
    """Score headline sentiment using Groq or xAI. Returns float in [-1.0, 1.0]."""
    try:
        headline_text = "\n".join(f"- {headline}" for headline in headlines)
        prompt = (
            "You are a financial sentiment engine.\n"
            f"Evaluate the overall sentiment of these news headlines for stock {ticker}.\n"
            "Return ONLY valid JSON with this exact schema:\n"
            '{"sentiment_score": <float between -1.0 and 1.0>}\n'
            "Where -1.0 is very bearish and 1.0 is very bullish.\n\n"
            f"Headlines:\n{headline_text}"
        )
        messages = [
            {
                "role": "system",
                "content": "Respond with JSON only. No markdown. No commentary.",
            },
            {"role": "user", "content": prompt},
        ]

        if api_key.startswith("xai-"):
            from openai import OpenAI

            client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
            completion = client.chat.completions.create(
                model="grok-2-latest",
                messages=messages,
                temperature=0.0,
            )
            provider = "xAI"
        else:
            from groq import Groq

            client = Groq(api_key=api_key)
            completion = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                temperature=0.0,
            )
            provider = "Groq"

        content = completion.choices[0].message.content or ""
        parsed = _extract_json_object(content)
        if not parsed or "sentiment_score" not in parsed:
            print(f"[WARN] {provider} sentiment response malformed for {ticker}: {content}")
            return None

        score = float(parsed["sentiment_score"])
        score = max(-1.0, min(1.0, score))
        print(f"[INFO] {provider} sentiment for {ticker}: {score:.2f}")
        return score
    except Exception as exc:
        print(f"[WARN] Sentiment failed for {ticker}: {exc}")
        return None


def fetch_gemini_analysis(
    ticker: str,
    price_summary: dict,
    rsi: float | None,
    sentiment: float | None,
    api_key: str,
) -> str | None:
    """Generate final 3-bullet risk/opportunity assessment using Gemini (flash-first)."""
    try:
        import portfolio_ai as pai

        rsi_text = f"{rsi:.2f}" if rsi is not None else "Unavailable"
        sentiment_text = f"{sentiment:.2f}" if sentiment is not None else "Unavailable"
        prompt = (
            f"Review Indian stock {ticker}. Output a dashboard — not an essay.\n\n"
            "## Verdict\nOne sentence.\n\n"
            "## Signals\n| Signal | Value | Read |\n|---|---|---|\n"
            "Fill rows from inputs below.\n\n"
            "## Actions\n| Action (RISK/OPPORTUNITY/WATCH) | One-line reason |\n|---|---|\n"
            "Exactly 3 rows.\n\n"
            f"Price (30d): {json.dumps(price_summary)}\n"
            f"RSI: {rsi_text}\n"
            f"Sentiment (-1 to +1): {sentiment_text}\n"
        )
        text, provider = pai.generate_ai_text(prompt, gemini_key=api_key)
        print(f"[INFO] Deep analysis via {provider}")
        return text
    except Exception as exc:
        print(f"[WARN] Gemini analysis failed for {ticker}: {exc}")
        return None


def print_section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def analyze_stock(
    ticker: str,
    alpha_key: str | None = None,
    groq_key: str | None = None,
    gemini_key: str | None = None,
    *,
    quiet: bool = False,
) -> dict:
    """
    Run the full multi-agent pipeline for one ticker.

    Returns a serializable result dict for CLI or Streamlit display.
    """
    yahoo_ticker = to_yahoo_ticker(ticker)
    warnings: list[str] = []
    result: dict = {
        "input_symbol": ticker,
        "yahoo_ticker": yahoo_ticker,
        "price_summary": None,
        "rsi": None,
        "sentiment": None,
        "headlines": [],
        "gemini_analysis": None,
        "warnings": warnings,
        "ok": False,
    }

    if not quiet:
        print_section(f"Analyzing {yahoo_ticker}")

    price_df = fetch_price_data(yahoo_ticker)
    if price_df is None:
        msg = f"yfinance returned no data for {yahoo_ticker}."
        warnings.append(msg)
        if not quiet:
            print(f"[WARN] Skipping {yahoo_ticker} because price data is unavailable.")
        return result

    price_summary = summarize_price_trend(price_df)
    result["price_summary"] = price_summary
    if not quiet:
        print("[Layer 1] yfinance price summary:")
        for key, value in price_summary.items():
            print(f"  - {key}: {value}")

    rsi = None
    if alpha_key:
        rsi = fetch_rsi_alpha_vantage(yahoo_ticker, alpha_key)
        if rsi is None:
            warnings.append("Alpha Vantage RSI unavailable (rate limit, symbol mismatch, or API error).")
            if not quiet:
                print(f"[WARN] RSI layer skipped for {yahoo_ticker}.")
    else:
        warnings.append("ALPHA_VANTAGE_API_KEY missing. RSI layer skipped.")
        if not quiet:
            print("[WARN] ALPHA_VANTAGE_API_KEY missing. RSI layer skipped.")
    result["rsi"] = rsi

    sentiment = None
    headlines = dummy_news_headlines(yahoo_ticker)
    result["headlines"] = headlines
    if groq_key:
        sentiment = fetch_groq_sentiment(yahoo_ticker, headlines, groq_key)
        if sentiment is None:
            warnings.append("Groq sentiment unavailable.")
            if not quiet:
                print(f"[WARN] Sentiment layer skipped for {yahoo_ticker}.")
    else:
        warnings.append("GROQ_API_KEY missing. Sentiment layer skipped.")
        if not quiet:
            print("[WARN] GROQ_API_KEY missing. Sentiment layer skipped.")
    result["sentiment"] = sentiment

    analysis = None
    if gemini_key:
        analysis = fetch_gemini_analysis(yahoo_ticker, price_summary, rsi, sentiment, gemini_key)
        if analysis is None:
            warnings.append("Gemini analysis unavailable.")
            if not quiet:
                print(f"[WARN] Gemini analysis skipped for {yahoo_ticker}.")
    else:
        warnings.append("GEMINI_API_KEY missing. Gemini analysis skipped.")
        if not quiet:
            print("[WARN] GEMINI_API_KEY missing. Gemini analysis skipped.")
    result["gemini_analysis"] = analysis
    result["ok"] = True

    if not quiet:
        print("\n[Final Assessment]")
        print(analysis if analysis else "No final assessment generated.")

    return result


def format_analysis_markdown(result: dict) -> str:
    """Format a pipeline result for Streamlit / chat display."""
    ticker = result.get("yahoo_ticker") or result.get("input_symbol") or "Unknown"
    lines = [f"### Multi-agent analysis: `{ticker}`", ""]

    price = result.get("price_summary")
    if price:
        lines.extend(
            [
                "**Price trend (30d · yfinance)**",
                f"- Trend: **{price['trend']}** ({price['pct_change_30d']:+.2f}%)",
                f"- Close: ₹{price['start_close']:,.2f} → ₹{price['end_close']:,.2f}",
                f"- 30d range: ₹{price['low_30d']:,.2f} – ₹{price['high_30d']:,.2f}",
                f"- Avg volume: {price['avg_volume']:,.0f}",
                "",
            ]
        )
    else:
        lines.append("_Price data unavailable._\n")

    rsi = result.get("rsi")
    lines.append("**Technical (Alpha Vantage)**")
    lines.append(f"- 14-day RSI: **{rsi:.2f}**" if rsi is not None else "- 14-day RSI: _unavailable_")
    lines.append("")

    sentiment = result.get("sentiment")
    lines.append("**Sentiment (Groq)**")
    if sentiment is not None:
        label = "bullish" if sentiment > 0.15 else "bearish" if sentiment < -0.15 else "neutral"
        lines.append(f"- Score: **{sentiment:+.2f}** ({label})")
    else:
        lines.append("- Score: _unavailable_")
    lines.append("")

    analysis = result.get("gemini_analysis")
    lines.append("**Deep analysis (Gemini)**")
    lines.append(analysis if analysis else "_No assessment generated._")

    warnings = result.get("warnings") or []
    if warnings:
        lines.extend(["", "**Notes**"])
        for warning in warnings:
            lines.append(f"- {warning}")

    return "\n".join(lines)


def main() -> int:
    keys = resolve_api_keys()
    missing = [
        name
        for name, value in [
            ("ALPHA_VANTAGE_API_KEY", keys["alpha"]),
            ("GROQ_API_KEY", keys["groq"]),
            ("GEMINI_API_KEY", keys["gemini"]),
        ]
        if not value
    ]
    if missing:
        print(f"[WARN] Missing env keys: {', '.join(missing)}. Related layers will be skipped.")

    for index, ticker in enumerate(TICKERS):
        analyze_stock(ticker, keys["alpha"], keys["groq"], keys["gemini"])
        if index < len(TICKERS) - 1:
            # Alpha Vantage free tier is strict; small pause reduces burst failures.
            time.sleep(2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
