"""AI research layer for portfolio briefing, chat, and stock deep-dives."""

from __future__ import annotations

import json
import logging
from typing import Any

import pandas as pd

import portfolio_ai_context as paictx
import portfolio_market_data as pmd
import portfolio_terminal as pterm
import stock_analyzer as san

logger = logging.getLogger(__name__)


SYSTEM_ANALYST = """You are an Indian equity portfolio analyst. Write like a research dashboard, not a story or essay.

Format rules (strict):
- No narrative filler ("it's worth noting", "overall", "in conclusion", "the market environment").
- No motivational or storytelling language.
- Lead every response with ## Verdict — exactly ONE factual sentence with numbers.
- Prefer markdown TABLES over paragraphs when comparing stocks, sectors, or metrics.
- Bullets: max 12 words per line; start with the number or symbol.
- Bold only symbols, weights %, returns %, and P&L ₹.
- Label actions explicitly: TRIM | HOLD | ADD | WATCH | REVIEW (one word, then reason).
- If data is missing, say "Missing: <item>" — do not guess.
- Educational analysis only; one brief disclaimer line at the end if giving actions.
- Use ## section headers; keep each section to one table OR up to 5 bullets."""


BRIEFING_PROMPT = """Produce a concise portfolio dashboard memo using ONLY the data below.

Strict format — use these exact sections:

## Verdict
One sentence: portfolio health, total return %, vs Nifty if known, #1 risk.

## Key metrics
Markdown table with columns: Metric | Value | Signal
Include at least: total value, overall return %, top-3 weight %, vs Nifty alpha, concentration (HHI label).

## Strengths
Markdown table: Symbol or theme | Metric | Why it matters
Max 3 rows.

## Risks
Markdown table: Risk | Symbol(s) | Number | Severity (High/Med/Low)
Max 4 rows.

## Actions (30–90 days)
Markdown table: Priority | Action (TRIM/HOLD/ADD/WATCH) | Symbol | Current wt → Target wt | One-line reason
Max 5 rows, ordered by priority.

## Review list
Markdown table: Symbol | Trigger metric | Bull one-liner | Bear one-liner
Max 4 symbols.

No prose blocks outside tables except the Verdict line.

Portfolio data:
{context}
"""


CHAT_PROMPT = """Answer using ONLY the data below. Dashboard style — not an essay.

Format:
## Verdict
One sentence with the direct answer and key numbers.

## Data
Markdown table(s) with the numbers that support the verdict.
- Earnings questions → table: Period | Revenue | Profit | EPS | QoQ/YoY
- Shareholding → table: Holder | % | Change (pp)
- Portfolio → table: Symbol | Weight % | Return % | P&L ₹

## Actions (only if user asks what to do)
Markdown table: Priority | Action | Symbol | Reason (with number)

Rules:
- Max 5 bullets total outside tables.
- No storytelling or generic market commentary.
- Do not claim data is missing when it appears below.

Question: {question}
{history}

Portfolio data:
{context}
{earnings_block}
"""


ENGINE_SYSTEM = """You are a Quantitative Portfolio Analyst for Indian equities and ETFs.
Use ONLY the JSON payload provided. Do not invent prices, RSI, weights, or tickers.
If a metric is null or a symbol was skipped, say so briefly — do not guess.
Educational analysis only; end with one short disclaimer line.
Write in markdown with exactly these three ## section headers (no others):
## Key Portfolio Drivers & Technical Health
## Risk Exposure & Concentration Flags
## Actionable Tactical Recommendations
Prefer short bullets and compact markdown tables. Bold symbols, weights %, returns %, and RSI values."""


ENGINE_PROMPT = """Analyze this portfolio JSON as a Quantitative Portfolio Analyst.

Return exactly these three sections:

## Key Portfolio Drivers & Technical Health
What is driving P&L and technical posture (RSI-14, SMA-50, 20d momentum) across enrichable holdings.

## Risk Exposure & Concentration Flags
Concentration, single-name / sector weight risk, and weak or overbought technicals.

## Actionable Tactical Recommendations
Concrete TRIM | HOLD | ADD | WATCH ideas with numbers from the payload (weight, RSI, momentum). Max 5 rows.

Portfolio JSON:
{payload_json}
"""


HOLDING_PROMPT = """Produce a structured holding note — tables first, no essay.

## Verdict
One sentence: HOLD / TRIM / ADD / WATCH plus weight % and return %.

## Position
| Field | Value |
|---|---|
| Weight | {weight:.2f}% |
| P&L | ₹{pl:,.0f} ({return_pct:+.2f}%) |
| Current / Invested | ₹{current:,.2f} / ₹{invested:,.2f} |

## Fundamentals
Markdown table from Yahoo data (P/E, sector, 52w range, etc.) — omit rows with no data.

## Technical (30d)
Markdown table or 3 bullets max from price/RSI data.

## Bull vs bear
| Case | Evidence (metric) |
|---|---|
| Bull | one line |
| Bear | one line |

## Action
| Action | Trigger |
|---|---|
| TRIM/HOLD/ADD/WATCH | specific number trigger |

Holding: {symbol} · Portfolio total ₹{total:,.2f} ({port_return:+.2f}%)

Yahoo / market:
{market}

30d / technicals:
{technical}

Portfolio context:
{portfolio_context}
"""


# Prefer free-tier / lite models first. Pro often returns quota limit: 0.
GEMINI_MODELS = (
    "gemini-flash-lite-latest",
    "gemini-flash-latest",
    "gemini-2.5-flash-lite",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.5-flash",
    "gemini-3.6-flash",
    "gemini-2.5-pro",
)

XAI_MODELS = (
    "grok-3-mini",
    "grok-3-mini-fast",
    "grok-4-latest",
    "grok-4.5",
    "grok-2-latest",
)


def build_rich_context(
    merged: pd.DataFrame,
    summary: dict,
    metrics: dict,
    portfolio_df: pd.DataFrame | None = None,
    *,
    include_technicals: bool = True,
    terminal_snapshot: dict | None = None,
    risk: dict | None = None,
    fmp_snapshot: dict | None = None,
    inst_df: pd.DataFrame | None = None,
    fmp_api_key: str = "",
) -> str:
    if terminal_snapshot is None and pterm.is_available():
        symbols = tuple(merged["Symbol"].tolist())
        news_symbols = tuple(merged.head(5)["Symbol"].tolist())
        terminal_snapshot = pterm.build_terminal_snapshot(symbols, news_symbols)
    return paictx.build_deep_analysis_context(
        merged,
        summary,
        metrics,
        portfolio_df,
        include_technicals=include_technicals,
        include_fundamentals=True,
        terminal_snapshot=terminal_snapshot,
        risk=risk,
        fmp_snapshot=fmp_snapshot,
        inst_df=inst_df,
        fmp_api_key=fmp_api_key,
    )


def detect_key_provider(api_key: str) -> str:
    """Guess provider from key prefix."""
    key = (api_key or "").strip()
    if not key or key.startswith("paste_") or "your_" in key.lower():
        return "none"
    if key.startswith("xai-"):
        return "xai"
    if key.startswith("gsk_"):
        return "groq"
    if key.startswith("AIza") or key.startswith("AQ."):
        return "gemini"
    if key.startswith("sk-"):
        return "openai"
    return "unknown"


def _call_gemini(prompt: str, api_key: str, system: str = SYSTEM_ANALYST) -> tuple[str, str]:
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    errors: list[str] = []
    for model in GEMINI_MODELS:
        try:
            client = genai.GenerativeModel(
                model,
                system_instruction=system,
            )
            response = client.generate_content(prompt)
            text = (response.text or "").strip()
            if text:
                return text, f"Gemini ({model})"
            errors.append(f"{model}: empty response")
        except Exception as exc:
            message = str(exc)
            if any(token in message for token in ("429", "quota", "404", "not found", "NOT_FOUND")):
                errors.append(f"{model}: {message[:160]}")
                continue
            errors.append(f"{model}: {message[:160]}")
            continue
    raise RuntimeError(" | ".join(errors) if errors else "Gemini failed with no details.")


def _call_groq(
    prompt: str,
    api_key: str,
    model: str = "llama-3.3-70b-versatile",
    system: str = SYSTEM_ANALYST,
) -> str:
    from groq import Groq

    if detect_key_provider(api_key) == "xai":
        raise RuntimeError(
            "This key starts with 'xai-' (xAI/Grok), not Groq. "
            "Get a Groq key from https://console.groq.com (usually starts with gsk_)."
        )

    client = Groq(api_key=api_key)
    completion = client.chat.completions.create(
        model=model,
        temperature=0.25,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    )
    text = completion.choices[0].message.content or ""
    if not text.strip():
        raise RuntimeError("Groq returned an empty response.")
    return text.strip()


def _call_xai(prompt: str, api_key: str, system: str = SYSTEM_ANALYST) -> tuple[str, str]:
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
    errors: list[str] = []
    chat_messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    for model in XAI_MODELS:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=chat_messages,
                temperature=0.25,
            )
            text = response.choices[0].message.content or ""
            if text.strip():
                return text.strip(), f"xAI ({model})"
            errors.append(f"{model}: empty response")
        except Exception as exc:
            errors.append(f"{model}: {str(exc)[:160]}")
            continue
    raise RuntimeError(" | ".join(errors) if errors else "xAI failed with no details.")


def _call_openai(
    prompt: str,
    api_key: str,
    model: str = "gpt-4o-mini",
    messages: list[dict] | None = None,
    system: str = SYSTEM_ANALYST,
) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    chat_messages = [{"role": "system", "content": system}]
    if messages:
        for msg in messages[-8:]:
            chat_messages.append({"role": msg["role"], "content": msg["content"]})
    chat_messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(
        model=model,
        messages=chat_messages,
        temperature=0.25,
    )
    text = response.choices[0].message.content or ""
    if not text.strip():
        raise RuntimeError("OpenAI returned an empty response.")
    return text.strip()


def generate_ai_text(
    prompt: str,
    *,
    gemini_key: str = "",
    groq_key: str = "",
    xai_key: str = "",
    openai_key: str = "",
    openai_model: str = "gpt-4o-mini",
    messages: list[dict] | None = None,
    system: str = SYSTEM_ANALYST,
) -> tuple[str, str]:
    errors: list[str] = []

    if not xai_key and detect_key_provider(groq_key) == "xai":
        xai_key = groq_key
        groq_key = ""

    if gemini_key and detect_key_provider(gemini_key) != "none":
        try:
            return _call_gemini(prompt, gemini_key, system=system)
        except Exception as exc:
            errors.append(f"Gemini: {exc}")

    if xai_key and detect_key_provider(xai_key) != "none":
        try:
            return _call_xai(prompt, xai_key, system=system)
        except Exception as exc:
            errors.append(f"xAI: {exc}")

    if groq_key and detect_key_provider(groq_key) != "none":
        try:
            return _call_groq(prompt, groq_key, system=system), "Groq"
        except Exception as exc:
            errors.append(f"Groq: {exc}")

    if openai_key and detect_key_provider(openai_key) != "none":
        try:
            return (
                _call_openai(
                    prompt, openai_key, openai_model, messages=messages, system=system
                ),
                "OpenAI",
            )
        except Exception as exc:
            errors.append(f"OpenAI: {exc}")

    detail = "; ".join(errors) if errors else "No AI API keys configured."
    raise RuntimeError(detail)


def generate_portfolio_briefing(
    merged: pd.DataFrame,
    summary: dict,
    metrics: dict,
    portfolio_df: pd.DataFrame,
    *,
    gemini_key: str = "",
    groq_key: str = "",
    xai_key: str = "",
    openai_key: str = "",
    terminal_snapshot: dict | None = None,
    risk: dict | None = None,
    fmp_snapshot: dict | None = None,
    inst_df: pd.DataFrame | None = None,
    fmp_api_key: str = "",
) -> dict[str, str]:
    context = build_rich_context(
        merged,
        summary,
        metrics,
        portfolio_df,
        include_technicals=True,
        terminal_snapshot=terminal_snapshot,
        risk=risk,
        fmp_snapshot=fmp_snapshot,
        inst_df=inst_df,
        fmp_api_key=fmp_api_key,
    )
    prompt = BRIEFING_PROMPT.format(context=context)
    text, provider = generate_ai_text(
        prompt,
        gemini_key=gemini_key,
        groq_key=groq_key,
        xai_key=xai_key,
        openai_key=openai_key,
    )
    return {"text": text, "provider": provider}


def answer_portfolio_question(
    question: str,
    messages: list[dict],
    merged: pd.DataFrame,
    summary: dict,
    metrics: dict,
    portfolio_df: pd.DataFrame | None = None,
    *,
    gemini_key: str = "",
    groq_key: str = "",
    xai_key: str = "",
    openai_key: str = "",
    openai_model: str = "gpt-4o-mini",
    risk: dict | None = None,
    fmp_snapshot: dict | None = None,
    inst_df: pd.DataFrame | None = None,
    fmp_api_key: str = "",
    earnings_context: str = "",
) -> dict[str, str]:
    context = build_rich_context(
        merged,
        summary,
        metrics,
        portfolio_df,
        include_technicals=True,
        risk=risk,
        fmp_snapshot=fmp_snapshot,
        inst_df=inst_df,
        fmp_api_key=fmp_api_key,
    )
    history_block = ""
    if messages:
        recent = messages[-6:]
        history_block = "\n\nRecent conversation:\n" + "\n".join(
            f"{m['role'].upper()}: {m['content'][:800]}" for m in recent
        )

    earnings_block = f"\n\n{earnings_context}" if earnings_context else ""
    prompt = CHAT_PROMPT.format(
        question=question,
        history=history_block,
        context=context,
        earnings_block=earnings_block,
    )
    text, provider = generate_ai_text(
        prompt,
        gemini_key=gemini_key,
        groq_key=groq_key,
        xai_key=xai_key,
        openai_key=openai_key,
        openai_model=openai_model,
        messages=messages if openai_key and not (gemini_key or groq_key or xai_key) else None,
    )
    return {"text": text, "provider": provider}


def _format_market_block(market: dict) -> str:
    if not market or market.get("Data source") != "Yahoo Finance":
        return str(market or "No Yahoo data")
    lines = []
    for key in (
        "Company",
        "Sector",
        "Industry",
        "Yahoo price",
        "P/E",
        "Forward P/E",
        "P/B",
        "Beta",
        "ROE %",
        "Debt/Equity",
        "Div yield %",
        "Market cap",
        "52-week range",
        "52w high",
        "52w low",
    ):
        val = market.get(key)
        if val is not None and val != "":
            if key == "Market cap" and isinstance(val, (int, float)):
                lines.append(f"- {key}: ₹{val/1e7:,.0f} Cr approx")
            elif isinstance(val, float) and key not in ("Yahoo price", "52w high", "52w low"):
                lines.append(f"- {key}: {val:.2f}")
            else:
                lines.append(f"- {key}: {val}")
    return "\n".join(lines) if lines else str(market)


def research_holding_deep(
    symbol: str,
    merged: pd.DataFrame,
    summary: dict,
    metrics: dict,
    *,
    portfolio_df: pd.DataFrame | None = None,
    gemini_key: str = "",
    groq_key: str = "",
    xai_key: str = "",
    openai_key: str = "",
    alpha_key: str = "",
    include_technicals: bool = True,
) -> dict[str, Any]:
    row = merged[merged["Symbol"] == symbol]
    if row.empty:
        raise ValueError(f"{symbol} is not in the current portfolio.")

    holding = row.iloc[0].to_dict()
    market_rows = pmd.fetch_market_info_for_symbols((symbol,))
    market = market_rows.iloc[0].to_dict() if not market_rows.empty else {}

    technical: dict[str, Any] = {}
    if include_technicals:
        try:
            tech = san.analyze_stock(
                symbol,
                alpha_key=alpha_key or None,
                groq_key=None,
                gemini_key=None,
                quiet=True,
            )
            technical = {
                "yahoo_ticker": tech.get("yahoo_ticker"),
                "price_summary": tech.get("price_summary"),
                "rsi": tech.get("rsi"),
            }
        except Exception as exc:
            technical = {"error": str(exc)}

    portfolio_context = build_rich_context(
        merged,
        summary,
        metrics,
        portfolio_df,
        include_technicals=False,
    )

    prompt = HOLDING_PROMPT.format(
        symbol=symbol,
        weight=float(holding.get("Weight %", 0)),
        current=float(holding.get("Current Value", 0)),
        invested=float(holding.get("Invested Value", 0)),
        pl=float(holding.get("P&L", 0)),
        return_pct=float(holding.get("Return %", 0)),
        platforms=holding.get("Platforms", "—"),
        total=float(summary.get("total_current", 0)),
        port_return=float(metrics.get("overall_return_pct", 0)),
        market=_format_market_block(market),
        technical=technical,
        portfolio_context=portfolio_context[:12000],
    )
    text, provider = generate_ai_text(
        prompt,
        gemini_key=gemini_key,
        groq_key=groq_key,
        xai_key=xai_key,
        openai_key=openai_key,
    )
    return {
        "text": text,
        "provider": provider,
        "symbol": symbol,
        "market": market,
        "technical": technical,
    }


# ---------------------------------------------------------------------------
# Generic Portfolio Analysis Engine (dynamic holdings → OpenBB → LLM)
# ---------------------------------------------------------------------------

_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "symbol": ("symbol", "Symbol", "ticker", "Ticker", "tradingsymbol"),
    "quantity": ("quantity", "Quantity", "qty", "Qty"),
    "buy_price": ("buy_price", "Buy Price", "average_price", "avg_price", "Avg Price"),
    "current_price": ("current_price", "Current Price", "ltp", "LTP", "last_price", "Last Price"),
    "asset_type": ("asset_type", "Asset Type", "asset", "Asset", "instrument_type", "Type"),
}


def _pick_column(df: pd.DataFrame, canonical: str) -> str | None:
    for name in _COLUMN_ALIASES.get(canonical, ()):
        if name in df.columns:
            return name
    lower_map = {str(c).strip().lower(): c for c in df.columns}
    for name in _COLUMN_ALIASES.get(canonical, ()):
        hit = lower_map.get(name.lower())
        if hit is not None:
            return hit
    return None


def normalize_holdings(holdings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Accept any dynamic holdings DataFrame and return a canonical schema:
    symbol, quantity, buy_price, current_price, asset_type (+ derived value cols).
    Does not hardcode tickers.
    """
    if holdings_df is None or holdings_df.empty:
        raise ValueError("Holdings DataFrame is empty.")

    mapping: dict[str, str] = {}
    for canonical in ("symbol", "quantity", "buy_price", "current_price", "asset_type"):
        src = _pick_column(holdings_df, canonical)
        if src:
            mapping[src] = canonical

    if "symbol" not in mapping.values():
        raise ValueError("Holdings DataFrame must include a symbol/ticker column.")

    out = holdings_df.rename(columns=mapping).copy()
    for col in ("quantity", "buy_price", "current_price"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        else:
            out[col] = pd.NA

    if "asset_type" not in out.columns:
        out["asset_type"] = "equity"
    else:
        out["asset_type"] = out["asset_type"].fillna("equity").astype(str)

    out["symbol"] = out["symbol"].astype(str).str.strip().str.upper()
    out = out[out["symbol"].ne("") & out["symbol"].ne("NAN")].copy()
    if out.empty:
        raise ValueError("No valid symbols found in holdings.")

    # Prefer existing Title-Case value columns when present (app portfolio schema).
    if "Invested Value" in holdings_df.columns and "invested_value" not in out.columns:
        out["invested_value"] = pd.to_numeric(holdings_df["Invested Value"], errors="coerce")
    else:
        out["invested_value"] = out["quantity"] * out["buy_price"]

    if "Current Value" in holdings_df.columns and "current_value" not in out.columns:
        out["current_value"] = pd.to_numeric(holdings_df["Current Value"], errors="coerce")
    else:
        out["current_value"] = out["quantity"] * out["current_price"]

    if "P&L" in holdings_df.columns:
        out["pnl"] = pd.to_numeric(holdings_df["P&L"], errors="coerce")
    else:
        out["pnl"] = out["current_value"] - out["invested_value"]

    if "Return %" in holdings_df.columns:
        out["return_pct"] = pd.to_numeric(holdings_df["Return %"], errors="coerce")
    else:
        out["return_pct"] = (out["pnl"] / out["invested_value"].replace(0, pd.NA)) * 100

    total = float(out["current_value"].fillna(0).sum())
    out["weight_pct"] = (
        out["current_value"].fillna(0) / total * 100 if total else 0.0
    )
    return out


def extract_unique_symbols(holdings: pd.DataFrame) -> list[str]:
    """Unique ticker symbols from a normalized (or raw) holdings frame — never hardcoded."""
    if "symbol" in holdings.columns:
        series = holdings["symbol"]
    else:
        col = _pick_column(holdings, "symbol")
        if not col:
            return []
        series = holdings[col]
    return list(dict.fromkeys(str(s).strip().upper() for s in series if pd.notna(s) and str(s).strip()))


def build_analysis_payload(
    holdings: pd.DataFrame,
    technicals: dict | None = None,
) -> dict[str, Any]:
    """Structured JSON payload: portfolio stats + OpenBB/Yahoo technical metrics."""
    df = holdings if "weight_pct" in holdings.columns else normalize_holdings(holdings)
    total_invested = float(df["invested_value"].fillna(0).sum())
    total_current = float(df["current_value"].fillna(0).sum())
    total_pnl = total_current - total_invested
    overall_return = (total_pnl / total_invested * 100) if total_invested else 0.0

    ranked = df.sort_values("current_value", ascending=False)
    by_return = df.dropna(subset=["return_pct"]).sort_values("return_pct", ascending=False)

    def _row_brief(row: pd.Series) -> dict[str, Any]:
        return {
            "symbol": row["symbol"],
            "asset_type": row.get("asset_type", "equity"),
            "weight_pct": round(float(row.get("weight_pct") or 0), 2),
            "return_pct": round(float(row["return_pct"]), 2) if pd.notna(row.get("return_pct")) else None,
            "pnl": round(float(row["pnl"]), 0) if pd.notna(row.get("pnl")) else None,
            "current_value": round(float(row.get("current_value") or 0), 0),
        }

    top_gainers = [_row_brief(r) for _, r in by_return.head(5).iterrows()]
    top_losers = [_row_brief(r) for _, r in by_return.tail(5).iloc[::-1].iterrows()]
    top_weights = [_row_brief(r) for _, r in ranked.head(8).iterrows()]

    allocation_by_type: dict[str, float] = {}
    for asset, group in df.groupby(df["asset_type"].str.lower()):
        allocation_by_type[str(asset)] = round(float(group["weight_pct"].sum()), 2)

    tech_rows = (technicals or {}).get("metrics") or []
    tech_by_symbol = {m["symbol"]: m for m in tech_rows if m.get("symbol")}

    holdings_detail = []
    for _, row in ranked.iterrows():
        sym = row["symbol"]
        tech = tech_by_symbol.get(sym, {})
        holdings_detail.append(
            {
                **_row_brief(row),
                "quantity": float(row["quantity"]) if pd.notna(row.get("quantity")) else None,
                "buy_price": float(row["buy_price"]) if pd.notna(row.get("buy_price")) else None,
                "current_price": float(row["current_price"]) if pd.notna(row.get("current_price")) else None,
                "technicals": {
                    "yahoo_ticker": tech.get("yahoo_ticker"),
                    "rsi_14": tech.get("rsi_14"),
                    "sma_50": tech.get("sma_50"),
                    "momentum_20d_pct": tech.get("momentum_20d_pct"),
                    "price_vs_sma50_pct": tech.get("price_vs_sma50_pct"),
                    "last_close": tech.get("last_close"),
                    "source": tech.get("source"),
                    "error": tech.get("error"),
                },
            }
        )

    top3_weight = round(sum(h["weight_pct"] for h in top_weights[:3]), 2)
    return {
        "portfolio": {
            "holding_count": int(len(df)),
            "total_invested": round(total_invested, 0),
            "total_current": round(total_current, 0),
            "total_pnl": round(total_pnl, 0),
            "overall_return_pct": round(overall_return, 2),
            "top3_weight_pct": top3_weight,
            "allocation_by_asset_type": allocation_by_type,
        },
        "top_weights": top_weights,
        "top_gainers": top_gainers,
        "top_losers": top_losers,
        "holdings": holdings_detail,
        "market_enrichment": {
            "available": bool((technicals or {}).get("available")),
            "coverage": (technicals or {}).get("coverage") or {},
            "warnings": (technicals or {}).get("warnings") or [],
        },
    }


def analyze_portfolio_engine(
    holdings_df: pd.DataFrame,
    *,
    gemini_key: str = "",
    groq_key: str = "",
    xai_key: str = "",
    openai_key: str = "",
    max_symbols: int = 40,
) -> dict[str, Any]:
    """
    End-to-end engine: normalize holdings → OpenBB/Yahoo technicals → LLM insights.
    Individual API failures become warnings; the function returns a result dict
    instead of crashing the Streamlit app.
    """
    warnings: list[str] = []
    result: dict[str, Any] = {
        "ok": False,
        "text": "",
        "provider": "",
        "payload": {},
        "technicals": {},
        "warnings": warnings,
    }

    try:
        holdings = normalize_holdings(holdings_df)
    except Exception as exc:
        logger.warning("Engine normalize failed: %s", exc)
        warnings.append(f"normalize: {exc}")
        result["warnings"] = warnings
        return result

    symbols = extract_unique_symbols(holdings)
    # Align asset_type per unique symbol (first occurrence)
    type_by_symbol: dict[str, str] = {}
    for _, row in holdings.iterrows():
        sym = row["symbol"]
        if sym not in type_by_symbol:
            type_by_symbol[sym] = str(row.get("asset_type") or "equity")

    enrich_symbols = symbols[:max_symbols]
    if len(symbols) > max_symbols:
        warnings.append(f"Technical enrichment limited to top {max_symbols} unique symbols.")

    asset_types = tuple(type_by_symbol.get(s) for s in enrich_symbols)
    try:
        technicals = pterm.enrich_symbols_technicals(tuple(enrich_symbols), asset_types)
    except Exception as exc:
        logger.warning("OpenBB technical enrichment failed: %s", exc)
        warnings.append(f"openbb: {exc}")
        technicals = {"available": False, "metrics": [], "warnings": [str(exc)]}

    for w in technicals.get("warnings") or []:
        warnings.append(w)

    try:
        payload = build_analysis_payload(holdings, technicals)
    except Exception as exc:
        logger.warning("Payload assembly failed: %s", exc)
        warnings.append(f"payload: {exc}")
        result["technicals"] = technicals
        result["warnings"] = warnings
        return result

    result["payload"] = payload
    result["technicals"] = technicals

    try:
        payload_json = json.dumps(payload, default=str, indent=2)
        # Keep prompt size bounded for free-tier models
        if len(payload_json) > 28000:
            slim = {
                "portfolio": payload["portfolio"],
                "top_weights": payload["top_weights"],
                "top_gainers": payload["top_gainers"],
                "top_losers": payload["top_losers"],
                "holdings": payload["holdings"][:20],
                "market_enrichment": payload["market_enrichment"],
            }
            payload_json = json.dumps(slim, default=str, indent=2)
            warnings.append("Payload truncated to top 20 holdings for LLM context limit.")

        prompt = ENGINE_PROMPT.format(payload_json=payload_json)
        text, provider = generate_ai_text(
            prompt,
            gemini_key=gemini_key,
            groq_key=groq_key,
            xai_key=xai_key,
            openai_key=openai_key,
            system=ENGINE_SYSTEM,
        )
        result["ok"] = True
        result["text"] = text
        result["provider"] = provider
    except Exception as exc:
        logger.warning("LLM portfolio analysis failed: %s", exc)
        warnings.append(f"llm: {exc}")
        result["text"] = ""
        result["provider"] = ""

    result["warnings"] = warnings
    return result
