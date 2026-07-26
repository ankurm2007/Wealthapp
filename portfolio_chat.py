"""Portfolio Q&A: AI-first research with rule-based fallback."""

from __future__ import annotations

import re

import pandas as pd

import portfolio_ai as pai
import portfolio_agent as pagt
import portfolio_finance_context as pfctx
import portfolio_market_data as pmd


SUGGESTED_QUESTIONS = {
    "Deep portfolio memo": "Write a deep portfolio memo: scorecard, rebalancing plan, and which holdings to review this month with specific weights and returns.",
    "What should I trim?": "Based on concentration, P&L, and sector tilt vs Nifty, which positions should I trim first and to what weight band?",
    "Am I beating Nifty?": "Compare my 30-day return vs Nifty, explain alpha, and whether stock picks or sector bets drove the gap.",
    "Valuation risks": "Which holdings look expensive on P/E or near 52-week highs relative to their weight in my book?",
    "Research largest holding": "Do a deep research note on my largest holding — fundamentals, portfolio role, and hold/trim/add with triggers.",
    "Quarterly results": "Summarize the latest quarterly results for my largest holding — revenue, profit, EPS, QoQ/YoY — and what it means for my position.",
    "Shareholding pattern": "What is the shareholding pattern for my largest holding — promoter, FII, DII, public — and how has it changed?",
    "Heavy losers": "Which heavy losers (5%+ weight, negative return) need a thesis review vs a cut, with numbers?",
}


def _match(question: str, *patterns: str) -> bool:
    text = question.lower()
    return any(re.search(pattern, text) for pattern in patterns)


def try_rule_based_answer(question: str, merged: pd.DataFrame, metrics: dict) -> str | None:
    if _match(question, r"\bmulti-?agent\b", r"\bhow do i run deep\b"):
        return (
            "Use the panels above:\n"
            "1. **AI portfolio briefing** — full portfolio memo\n"
            "2. **Deep holding research** — Gemini/Groq note on one stock\n"
            "3. **Multi-agent analyzer** — yfinance + RSI + sentiment + Gemini\n"
            "4. **Research chat** — agent mode picks live tools (earnings, shareholding, Nifty) then answers.\n"
            "Set Gemini / Groq keys in secrets or `.env` for the best answers."
        )

    if _match(question, r"\bhelp\b", r"\bwhat can you\b"):
        return (
            "I can analyze your portfolio with AI. Try:\n"
            "- Full portfolio briefing\n"
            "- What should I trim / rebalance?\n"
            "- Quarterly results or shareholding for any holding\n"
            "- Am I beating Nifty?\n\n"
            "Complex chat questions use **agent mode** (Groq plans tools → live data → answer)."
        )

    # Keep a few fast deterministic answers for offline/no-key mode.
    if _match(question, r"\btop holdings by weight\b", r"\bbiggest holdings\b"):
        rows = merged.head(10)
        lines = ["**Top holdings by weight:**", ""]
        for _, row in rows.iterrows():
            lines.append(
                f"- **{row['Symbol']}**: {row['Weight %']:.2f}% "
                f"(₹{row['Current Value']:,.2f}, {row['Return %']:+.2f}% return)"
            )
        return "\n".join(lines)

    if _match(question, r"\bsector", r"\bindustry\b") and _match(question, r"\bbreakdown\b|\ballocation\b"):
        cached = pmd.get_cached_market_context(merged, {"total_current": metrics["total_current"]})
        if cached is None:
            return "Sector data is not loaded yet. Open **Insights → Analysis** and click **Load market data**."
        _, sectors, _ = cached
        lines = ["**Sector allocation:**", ""]
        for _, row in sectors.iterrows():
            lines.append(
                f"- **{row['Sector']}**: {row['Weight %']:.1f}% "
                f"(₹{row['Current Value']:,.0f}, {row['Return %']:+.1f}%) — {row['Stocks']}"
            )
        return "\n".join(lines)

    return None


def _extract_symbol(question: str, symbols: list[str]) -> str | None:
    return pfctx.resolve_symbol_from_question(question, symbols)


def _research_context_for_question(
    question: str,
    merged: pd.DataFrame,
    fmp_api_key: str = "",
    inst_df: pd.DataFrame | None = None,
) -> str:
    return pfctx.build_live_finance_context(
        question,
        merged,
        fmp_api_key=fmp_api_key,
        inst_df=inst_df,
    )


def _format_agent_response(result: pagt.AgentResult) -> str:
    tools = ", ".join(result.tools_used) if result.tools_used else "none"
    symbol_note = f" · symbol **{result.symbol}**" if result.symbol else ""
    header = f"*Agent research via {result.provider}* (tools: {tools}{symbol_note})"
    return f"{header}\n\n{result.text}"


def _agent_context(
    question: str,
    messages: list[dict],
    merged: pd.DataFrame,
    summary: dict,
    metrics: dict,
    *,
    portfolio_df: pd.DataFrame | None,
    gemini_key: str,
    groq_key: str,
    xai_key: str,
    openai_key: str,
    openai_model: str,
    risk: dict | None,
    fmp_snapshot: dict | None,
    inst_df: pd.DataFrame | None,
    fmp_api_key: str,
) -> pagt.AgentContext:
    return pagt.AgentContext(
        question=question,
        messages=messages,
        merged=merged,
        summary=summary,
        metrics=metrics,
        portfolio_df=portfolio_df,
        risk=risk,
        fmp_snapshot=fmp_snapshot,
        inst_df=inst_df,
        fmp_api_key=fmp_api_key,
        gemini_key=gemini_key,
        groq_key=groq_key,
        xai_key=xai_key,
        openai_key=openai_key,
        openai_model=openai_model,
    )


def get_response(
    question: str,
    messages: list[dict],
    merged: pd.DataFrame,
    summary: dict,
    metrics: dict,
    *,
    portfolio_df: pd.DataFrame | None = None,
    gemini_key: str = "",
    groq_key: str = "",
    xai_key: str = "",
    openai_key: str = "",
    openai_model: str = "gpt-4o-mini",
    api_key: str = "",  # backward-compatible OpenAI key alias
    risk: dict | None = None,
    fmp_snapshot: dict | None = None,
    inst_df: pd.DataFrame | None = None,
    fmp_api_key: str = "",
) -> str:
    openai_key = openai_key or api_key
    has_ai = bool(gemini_key or groq_key or xai_key or openai_key)

    # Prefer AI whenever keys exist.
    if has_ai:
        try:
            # Deep research shortcut for "research SYMBOL" / largest holding.
            if _match(question, r"\bresearch\b", r"\bdeep dive\b", r"\banalys[e|is]\b"):
                symbol = _extract_symbol(question, merged["Symbol"].tolist())
                if not symbol and _match(question, r"\blargest\b"):
                    symbol = metrics.get("largest_symbol")
                if symbol:
                    result = pai.research_holding_deep(
                        symbol,
                        merged,
                        summary,
                        metrics,
                        portfolio_df=portfolio_df,
                        gemini_key=gemini_key,
                        groq_key=groq_key,
                        xai_key=xai_key,
                        openai_key=openai_key,
                    )
                    return f"*AI research via {result['provider']}*\n\n{result['text']}"

            if _match(question, r"\bbriefing\b", r"\bfull (ai )?portfolio\b", r"\bexecutive summary\b"):
                result = pai.generate_portfolio_briefing(
                    merged,
                    summary,
                    metrics,
                    portfolio_df if portfolio_df is not None else merged,
                    gemini_key=gemini_key,
                    groq_key=groq_key,
                    xai_key=xai_key,
                    openai_key=openai_key,
                    risk=risk,
                    fmp_snapshot=fmp_snapshot,
                    inst_df=inst_df,
                    fmp_api_key=fmp_api_key,
                )
                return f"*AI briefing via {result['provider']}*\n\n{result['text']}"

            if pagt.should_use_agent(question, merged):
                agent_ctx = _agent_context(
                    question,
                    messages,
                    merged,
                    summary,
                    metrics,
                    portfolio_df=portfolio_df,
                    gemini_key=gemini_key,
                    groq_key=groq_key,
                    xai_key=xai_key,
                    openai_key=openai_key,
                    openai_model=openai_model,
                    risk=risk,
                    fmp_snapshot=fmp_snapshot,
                    inst_df=inst_df,
                    fmp_api_key=fmp_api_key,
                )
                agent_result = pagt.run_agent(agent_ctx)
                return _format_agent_response(agent_result)

            result = pai.answer_portfolio_question(
                question,
                messages,
                merged,
                summary,
                metrics,
                portfolio_df=portfolio_df,
                gemini_key=gemini_key,
                groq_key=groq_key,
                xai_key=xai_key,
                openai_key=openai_key,
                openai_model=openai_model,
                risk=risk,
                fmp_snapshot=fmp_snapshot,
                inst_df=inst_df,
                fmp_api_key=fmp_api_key,
                earnings_context=_research_context_for_question(
                    question, merged, fmp_api_key, inst_df
                ),
            )
            return f"*Answered by {result['provider']}*\n\n{result['text']}"
        except Exception as exc:
            local = try_rule_based_answer(question, merged, metrics)
            if local:
                return f"{local}\n\n*AI unavailable: {exc}*"
            return f"AI research failed: {exc}"

    local = try_rule_based_answer(question, merged, metrics)
    if local:
        return local

    # Minimal offline fallback for common questions.
    if _match(question, r"\bsummar", r"\boverview\b"):
        return (
            f"Portfolio value **₹{metrics['total_current']:,.0f}** "
            f"({metrics['overall_return_pct']:+.1f}%). "
            f"Largest holding: **{metrics['largest_symbol']}** "
            f"({metrics['largest_weight']:.1f}%). "
            "Add a Gemini or Groq key for full AI research."
        )

    return (
        "AI research is not configured yet. Add `[gemini]`, `[xai]` (`xai-...`), "
        "and/or `[groq]` (`gsk_...`) in `.streamlit/secrets.toml` (or `.env`), then ask again.\n\n"
        "Meanwhile you can use the multi-agent analyzer panel above."
    )
