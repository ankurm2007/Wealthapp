"""Earnings call transcript auditor powered by Gemini."""

from __future__ import annotations

import portfolio_ai as pai

AUDIT_PROMPT = """Forensic review of an earnings call transcript for Indian stock {symbol}.
Investor position: {weight:.1f}% weight, {return_pct:+.1f}% return.

Use ONLY the transcript. Dashboard format — no essay.

## Verdict
One sentence: credible / cautious / red-flag — with one supporting fact.

## Tone
| Aspect | Rating (High/Med/Low) | Evidence |
|---|---|---|

## Red flags
| Flag | Quote or paraphrase |
|---|---|
Max 3 rows.

## Positive signals
| Signal | Evidence |
|---|---|
Max 3 rows.

## Position impact
| Field | Take |
|---|---|
| For {weight:.1f}% weight | one line |

## Follow-ups
| Check before next results |
|---|
Max 3 rows.

Portfolio context:
{portfolio_excerpt}

TRANSCRIPT:
{transcript}
"""


def audit_transcript(
    transcript: str,
    symbol: str,
    merged_row: dict,
    portfolio_excerpt: str,
    *,
    gemini_key: str = "",
    groq_key: str = "",
    xai_key: str = "",
    openai_key: str = "",
) -> dict[str, str]:
    text = (transcript or "").strip()
    if len(text) < 200:
        raise ValueError("Transcript too short — paste at least a few paragraphs.")

    prompt = AUDIT_PROMPT.format(
        symbol=symbol,
        weight=float(merged_row.get("Weight %", 0)),
        return_pct=float(merged_row.get("Return %", 0)),
        portfolio_excerpt=portfolio_excerpt[:4000],
        transcript=text[:120000],
    )

    result, provider = pai.generate_ai_text(
        prompt,
        gemini_key=gemini_key,
        groq_key=groq_key,
        xai_key=xai_key,
        openai_key=openai_key,
    )
    return {"text": result, "provider": provider, "symbol": symbol}
