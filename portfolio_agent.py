"""Agentic portfolio research: plan tools with Groq, execute locally, synthesize with existing LLMs."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

import portfolio_ai as pai
import portfolio_benchmarks as pbench
import portfolio_earnings as pearn
import portfolio_finance_context as pfctx
import portfolio_forensics as pforensic
import portfolio_shareholding as pshare

AGENT_TOOL_NAMES = (
    "get_top_holdings",
    "get_concentration_summary",
    "get_nifty_comparison",
    "get_heavy_losers",
    "get_risk_summary",
    "get_portfolio_position",
    "get_earnings",
    "get_shareholding",
    "get_forensics",
)

MAX_TOOLS_PER_QUESTION = 5

AGENT_ROUTE_PATTERNS = (
    r"\bearnings\b",
    r"\bshareholding\b",
    r"\bpromoter\b",
    r"\bfii\b",
    r"\bdii\b",
    r"\btrim\b",
    r"\brebalanc",
    r"\bwhat should i\b",
    r"\bheavy loser",
    r"\bbeating nifty\b",
    r"\bvs nifty\b",
    r"\btoo big\b",
    r"\bconcentration\b",
    r"\bquarterly\b",
    r"\bresults\b",
    r"\bcompare\b",
    r"\bforensic\b",
    r"\bpiotroski\b",
    r"\bvaluation\b",
    r"\bhold\b.*\b(sell|cut|add|exit)\b",
    r"\b(and|also|plus)\b",
)

FAST_CHAT_PATTERNS = (
    r"^\s*top \d+ holdings\b",
    r"^\s*biggest holdings\b",
    r"^\s*sector (breakdown|allocation)\b",
)

PLAN_PROMPT = """You plan data fetches for an Indian equity portfolio assistant.
Return ONLY valid JSON (no markdown):
{{"tools": ["tool1", "..."], "symbol": "NSE_TICKER_OR_null", "n": 5}}

Rules:
- Pick 1–5 tools from this list only: {tool_list}
- symbol must be one of the portfolio symbols when a single stock is needed, else null
- For portfolio-wide coach questions (trim, rebalance, Nifty), prefer concentration + nifty + heavy_losers
- For earnings/results questions, include get_earnings (+ get_portfolio_position)
- For shareholding/promoter/FII, include get_shareholding (+ get_portfolio_position)
- n is optional (default 5) for get_top_holdings

Portfolio symbols: {symbols}
Largest holding: {largest}
User question: {question}
"""


SYNTHESIS_PROMPT = """Answer using ONLY the snapshot and tool results. Dashboard format — not an essay.

## Verdict
One sentence — direct answer with numbers.

## Data
Markdown table(s) built from tool results (weights, returns, earnings, shareholding, Nifty alpha).
Max 2 tables.

## Actions (if user asked what to do)
Markdown table: Priority | Action (TRIM/HOLD/ADD/WATCH) | Symbol | Reason

Rules:
- No storytelling or filler sentences.
- Max 3 bullets outside tables.
- Say "Missing: …" for gaps — do not invent data.
- One-line educational disclaimer only if giving actions.

Question: {question}
{history}

Portfolio snapshot:
{snapshot}

Tool results:
{tool_blocks}
"""


@dataclass
class AgentContext:
    question: str
    messages: list[dict]
    merged: pd.DataFrame
    summary: dict
    metrics: dict
    portfolio_df: pd.DataFrame | None = None
    risk: dict | None = None
    fmp_snapshot: dict | None = None
    inst_df: pd.DataFrame | None = None
    fmp_api_key: str = ""
    gemini_key: str = ""
    groq_key: str = ""
    xai_key: str = ""
    openai_key: str = ""
    openai_model: str = "gpt-4o-mini"


@dataclass
class AgentResult:
    text: str
    provider: str
    tools_used: list[str] = field(default_factory=list)
    symbol: str | None = None


def _match(question: str, *patterns: str) -> bool:
    text = question.lower()
    return any(re.search(pattern, text) for pattern in patterns)


def should_use_agent(question: str, merged: pd.DataFrame) -> bool:
    if _match(question, *FAST_CHAT_PATTERNS):
        return False
    if _match(question, r"\bbriefing\b", r"\bfull (ai )?portfolio\b", r"\bexecutive summary\b"):
        return False
    if _match(question, r"\bresearch\b", r"\bdeep dive\b") and _match(
        question, r"\blargest\b|\bholding\b|[A-Z]{3,}"
    ):
        return False
    if _match(question, *AGENT_ROUTE_PATTERNS):
        return True
    symbols = merged["Symbol"].tolist()
    symbol = pfctx.resolve_symbol_from_question(question, symbols)
    if symbol and pfctx.is_stock_finance_question(question):
        return True
    if len(question.split()) >= 12:
        return True
    return False


def _portfolio_snapshot(ctx: AgentContext) -> str:
    m = ctx.metrics
    largest_return = float(ctx.merged.iloc[0]["Return %"]) if not ctx.merged.empty else 0.0
    lines = [
        f"- Total value: ₹{m['total_current']:,.0f} ({m['overall_return_pct']:+.1f}% overall)",
        f"- Holdings: {m['holding_count']} | Concentration: {m['concentration_label']} (HHI {m['hhi']:.3f})",
        f"- Top 3 weight: {m['top3_weight']:.1f}% | Top 5: {m['top5_weight']:.1f}%",
        f"- Largest: {m['largest_symbol']} ({m['largest_weight']:.1f}%, {largest_return:+.1f}% return)",
        f"- In profit / loss: {m['in_profit']} / {m['in_loss']}",
    ]
    if m.get("concentrated_symbols"):
        lines.append(f"- Positions ≥15% weight: {', '.join(m['concentrated_symbols'][:5])}")
    return "\n".join(lines)


def _resolve_symbol(ctx: AgentContext, symbol_hint: str | None = None) -> str | None:
    symbols = ctx.merged["Symbol"].tolist()
    if symbol_hint:
        hint = symbol_hint.strip().upper()
        for sym in symbols:
            if sym.upper() == hint:
                return sym
    return pfctx.resolve_symbol_from_question(ctx.question, symbols)


def _tool_get_top_holdings(ctx: AgentContext, n: int = 5) -> str:
    rows = ctx.merged.head(max(1, min(int(n), 15)))
    lines = [f"Top {len(rows)} holdings by weight:", ""]
    for _, row in rows.iterrows():
        lines.append(
            f"- **{row['Symbol']}**: {row['Weight %']:.2f}% "
            f"(₹{row['Current Value']:,.0f}, {row['Return %']:+.1f}% return, P&L ₹{row['P&L']:,.0f})"
        )
    return "\n".join(lines)


def _tool_get_concentration_summary(ctx: AgentContext) -> str:
    m = ctx.metrics
    lines = [
        "Concentration summary:",
        f"- HHI: {m['hhi']:.3f} ({m['concentration_label']})",
        f"- Top 3 weight: {m['top3_weight']:.1f}%",
        f"- Top 5 weight: {m['top5_weight']:.1f}%",
        f"- Weighted avg return: {m['weighted_return']:+.1f}%",
    ]
    if m.get("concentrated_symbols"):
        lines.append(f"- Single-name ≥15%: {', '.join(m['concentrated_symbols'])}")
    return "\n".join(lines)


def _tool_get_nifty_comparison(ctx: AgentContext) -> str:
    try:
        bench = pbench.build_nifty_vs_portfolio(ctx.merged, days=30)
    except Exception as exc:
        return f"Nifty comparison unavailable: {exc}"
    port = bench.get("portfolio_return")
    nifty = bench.get("nifty_return")
    alpha = bench.get("alpha")
    lines = ["30-day vs Nifty 50:"]
    if port is not None:
        lines.append(f"- Portfolio (price proxy): {port:+.2f}%")
    if nifty is not None:
        lines.append(f"- Nifty 50: {nifty:+.2f}%")
    if alpha is not None:
        lines.append(f"- Alpha (portfolio − Nifty): {alpha:+.2f} pp")
    lines.append(f"- Price coverage: {bench.get('coverage', 0):.0f}% of weights")
    return "\n".join(lines)


def _tool_get_heavy_losers(ctx: AgentContext, min_weight_pct: float = 5.0) -> str:
    losers = ctx.merged[
        (ctx.merged["Return %"] < 0) & (ctx.merged["Weight %"] >= min_weight_pct)
    ].sort_values("Weight %", ascending=False)
    if losers.empty:
        return f"No holdings with weight ≥{min_weight_pct:.0f}% and negative return."
    lines = [f"Heavy losers (weight ≥{min_weight_pct:.0f}%, negative return):", ""]
    for _, row in losers.head(8).iterrows():
        lines.append(
            f"- **{row['Symbol']}**: {row['Weight %']:.1f}% weight, "
            f"{row['Return %']:+.1f}% return, P&L ₹{row['P&L']:,.0f}"
        )
    return "\n".join(lines)


def _tool_get_risk_summary(ctx: AgentContext) -> str:
    if not ctx.risk:
        return "Risk metrics not loaded. User can compute them under Insights → Analysis → Portfolio risk."
    import portfolio_risk as prisk

    return prisk.build_risk_context(ctx.risk)


def _tool_get_portfolio_position(ctx: AgentContext, symbol: str) -> str:
    if symbol not in ctx.merged["Symbol"].values:
        return f"{symbol} is not in the current portfolio."
    row = ctx.merged.loc[ctx.merged["Symbol"] == symbol].iloc[0]
    return (
        f"Portfolio position — **{symbol}**:\n"
        f"- Weight: {row['Weight %']:.2f}%\n"
        f"- Current value: ₹{row['Current Value']:,.0f}\n"
        f"- Invested: ₹{row['Invested Value']:,.0f}\n"
        f"- P&L: ₹{row['P&L']:,.0f} ({row['Return %']:+.2f}%)\n"
        f"- Platforms: {row.get('Platforms', '—')}"
    )


def _tool_get_earnings(ctx: AgentContext, symbol: str) -> str:
    row = None
    if symbol in ctx.merged["Symbol"].values:
        row = ctx.merged.loc[ctx.merged["Symbol"] == symbol].iloc[0].to_dict()
    data = pearn.fetch_quarterly_earnings(symbol, ctx.fmp_api_key)
    block = pearn.build_earnings_context(data, row)
    return block or f"No quarterly earnings data found for {symbol} (Screener/FMP/Yahoo)."


def _tool_get_shareholding(ctx: AgentContext, symbol: str) -> str:
    block = pshare.context_for_symbol(symbol, ctx.merged, ctx.inst_df)
    return block or f"No shareholding pattern found for {symbol} on Screener.in."


def _tool_get_forensics(ctx: AgentContext, symbol: str) -> str:
    snap = (ctx.fmp_snapshot or {}).get(symbol)
    if not snap and ctx.fmp_api_key:
        fetched = pforensic.fetch_forensic_snapshot((symbol,), ctx.fmp_api_key)
        snap = fetched.get(symbol)
    if not snap:
        return f"No FMP forensic data for {symbol}. FMP coverage for NSE names can be limited."
    parts = [f"Forensics — **{symbol}**:"]
    if snap.get("piotroski") is not None:
        parts.append(f"- Piotroski score: {snap['piotroski']}")
    if snap.get("ocf_to_profit") is not None:
        parts.append(f"- OCF / net profit: {snap['ocf_to_profit']:.2f}")
    if snap.get("altman_z") is not None:
        parts.append(f"- Altman Z: {snap['altman_z']}")
    return "\n".join(parts)


def execute_tool(name: str, ctx: AgentContext, symbol: str | None, plan: dict[str, Any]) -> str:
    n = int(plan.get("n") or 5)
    if name == "get_top_holdings":
        return _tool_get_top_holdings(ctx, n=n)
    if name == "get_concentration_summary":
        return _tool_get_concentration_summary(ctx)
    if name == "get_nifty_comparison":
        return _tool_get_nifty_comparison(ctx)
    if name == "get_heavy_losers":
        return _tool_get_heavy_losers(ctx)
    if name == "get_risk_summary":
        return _tool_get_risk_summary(ctx)
    if not symbol:
        return f"Tool `{name}` needs a portfolio symbol; none was resolved."
    if name == "get_portfolio_position":
        return _tool_get_portfolio_position(ctx, symbol)
    if name == "get_earnings":
        return _tool_get_earnings(ctx, symbol)
    if name == "get_shareholding":
        return _tool_get_shareholding(ctx, symbol)
    if name == "get_forensics":
        return _tool_get_forensics(ctx, symbol)
    return f"Unknown tool: {name}"


def _parse_json_plan(text: str) -> dict[str, Any] | None:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


def heuristic_plan(ctx: AgentContext, symbol: str | None) -> dict[str, Any]:
    tools: list[str] = []
    question = ctx.question

    if _match(
        question,
        r"\btrim\b",
        r"\brebalanc",
        r"\bwhat should i\b",
        r"\bconcentration\b",
        r"\btoo big\b",
    ):
        tools.extend(["get_concentration_summary", "get_top_holdings", "get_nifty_comparison"])

    if _match(question, r"\bheavy loser", r"\bloser", r"\bloss\b"):
        tools.append("get_heavy_losers")

    if _match(question, r"\bnifty\b", r"\balpha\b", r"\bbeating\b"):
        tools.append("get_nifty_comparison")

    if _match(question, r"\brisk\b", r"\bbeta\b", r"\bsharpe\b"):
        tools.append("get_risk_summary")

    if symbol:
        tools.append("get_portfolio_position")
        fetch_earnings, fetch_shareholding = pfctx.finance_data_needs(question, symbol)
        if fetch_earnings or _match(question, r"\bearnings\b", r"\bresults\b", r"\bquarter"):
            tools.append("get_earnings")
        if fetch_shareholding or _match(question, r"\bshareholding\b", r"\bpromoter\b", r"\bfii\b"):
            tools.append("get_shareholding")
        if _match(question, r"\bforensic\b", r"\bpiotroski\b", r"\bocf\b"):
            tools.append("get_forensics")

    if not tools:
        tools = ["get_top_holdings", "get_concentration_summary"]

    deduped = [t for t in tools if t in AGENT_TOOL_NAMES]
    return {"tools": list(dict.fromkeys(deduped))[:MAX_TOOLS_PER_QUESTION], "symbol": symbol}


def plan_tools(ctx: AgentContext, symbol: str | None) -> dict[str, Any]:
    if ctx.groq_key and pai.detect_key_provider(ctx.groq_key) == "groq":
        prompt = PLAN_PROMPT.format(
            tool_list=", ".join(AGENT_TOOL_NAMES),
            symbols=", ".join(ctx.merged["Symbol"].tolist()[:40]),
            largest=ctx.metrics.get("largest_symbol", "—"),
            question=ctx.question,
        )
        try:
            raw = pai._call_groq(prompt, ctx.groq_key)
            parsed = _parse_json_plan(raw)
            if parsed and isinstance(parsed.get("tools"), list):
                tools = [t for t in parsed["tools"] if t in AGENT_TOOL_NAMES][:MAX_TOOLS_PER_QUESTION]
                if tools:
                    sym = parsed.get("symbol")
                    if sym in ("null", "None", "", None):
                        sym = symbol
                    return {"tools": tools, "symbol": sym or symbol, "n": parsed.get("n", 5)}
        except Exception:
            pass
    return heuristic_plan(ctx, symbol)


def run_tools(ctx: AgentContext, plan: dict[str, Any]) -> tuple[list[str], str]:
    symbol = _resolve_symbol(ctx, plan.get("symbol"))
    blocks: list[str] = []
    used: list[str] = []
    for name in plan.get("tools", [])[:MAX_TOOLS_PER_QUESTION]:
        if name not in AGENT_TOOL_NAMES:
            continue
        result = execute_tool(name, ctx, symbol, plan)
        used.append(name)
        blocks.append(f"### Tool: {name}\n{result}")
    return used, "\n\n".join(blocks)


def synthesize(ctx: AgentContext, tool_blocks: str) -> tuple[str, str]:
    history_block = ""
    if ctx.messages:
        recent = ctx.messages[-4:]
        history_block = "\n\nRecent conversation:\n" + "\n".join(
            f"{m['role'].upper()}: {m['content'][:600]}" for m in recent
        )
    prompt = SYNTHESIS_PROMPT.format(
        question=ctx.question,
        history=history_block,
        snapshot=_portfolio_snapshot(ctx),
        tool_blocks=tool_blocks or "No tool data was collected.",
    )
    return pai.generate_ai_text(
        prompt,
        gemini_key=ctx.gemini_key,
        groq_key=ctx.groq_key,
        xai_key=ctx.xai_key,
        openai_key=ctx.openai_key,
        openai_model=ctx.openai_model,
    )


def run_agent(ctx: AgentContext) -> AgentResult:
    symbol = _resolve_symbol(ctx, None)
    plan = plan_tools(ctx, symbol)
    tools_used, tool_blocks = run_tools(ctx, plan)
    text, provider = synthesize(ctx, tool_blocks)
    resolved = _resolve_symbol(ctx, plan.get("symbol")) or symbol
    return AgentResult(
        text=text,
        provider=f"{provider} (agent)",
        tools_used=tools_used,
        symbol=resolved,
    )
