"""AI research layer for portfolio briefing, chat, and stock deep-dives."""

from __future__ import annotations

from typing import Any

import pandas as pd

import portfolio_ai_context as paictx
import portfolio_market_data as pmd
import portfolio_terminal as pterm
import stock_analyzer as san


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


def _call_gemini(prompt: str, api_key: str) -> tuple[str, str]:
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    errors: list[str] = []
    for model in GEMINI_MODELS:
        try:
            client = genai.GenerativeModel(
                model,
                system_instruction=SYSTEM_ANALYST,
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


def _call_groq(prompt: str, api_key: str, model: str = "llama-3.3-70b-versatile") -> str:
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
            {"role": "system", "content": SYSTEM_ANALYST},
            {"role": "user", "content": prompt},
        ],
    )
    text = completion.choices[0].message.content or ""
    if not text.strip():
        raise RuntimeError("Groq returned an empty response.")
    return text.strip()


def _call_xai(prompt: str, api_key: str) -> tuple[str, str]:
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
    errors: list[str] = []
    chat_messages = [
        {"role": "system", "content": SYSTEM_ANALYST},
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
) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    chat_messages = [{"role": "system", "content": SYSTEM_ANALYST}]
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
) -> tuple[str, str]:
    errors: list[str] = []

    if not xai_key and detect_key_provider(groq_key) == "xai":
        xai_key = groq_key
        groq_key = ""

    if gemini_key and detect_key_provider(gemini_key) != "none":
        try:
            return _call_gemini(prompt, gemini_key)
        except Exception as exc:
            errors.append(f"Gemini: {exc}")

    if xai_key and detect_key_provider(xai_key) != "none":
        try:
            return _call_xai(prompt, xai_key)
        except Exception as exc:
            errors.append(f"xAI: {exc}")

    if groq_key and detect_key_provider(groq_key) != "none":
        try:
            return _call_groq(prompt, groq_key), "Groq"
        except Exception as exc:
            errors.append(f"Groq: {exc}")

    if openai_key and detect_key_provider(openai_key) != "none":
        try:
            return _call_openai(prompt, openai_key, openai_model, messages=messages), "OpenAI"
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
