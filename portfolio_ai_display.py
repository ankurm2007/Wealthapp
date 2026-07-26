"""Structured, scannable rendering for AI research output."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd
import streamlit as st

META_LINE = re.compile(r"^\*(.+?)\*\s*$")
SECTION_HEADER = re.compile(r"^##\s+(.+)$", re.MULTILINE)


def split_meta_and_body(text: str) -> tuple[str | None, str]:
    lines = (text or "").strip().splitlines()
    if lines and META_LINE.match(lines[0].strip()):
        return lines[0].strip().strip("*"), "\n".join(lines[1:]).strip()
    return None, (text or "").strip()


def split_sections(body: str) -> list[tuple[str, str]]:
    if not body:
        return []
    matches = list(SECTION_HEADER.finditer(body))
    if not matches:
        return []
    sections: list[tuple[str, str]] = []
    for idx, match in enumerate(matches):
        title = match.group(1).strip()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(body)
        content = body[start:end].strip()
        if content:
            sections.append((title, content))
    return sections


def _section_tone(title: str) -> str:
    lower = title.lower()
    if any(k in lower for k in ("risk", "attention", "loser", "problem", "underestimat")):
        return "risk"
    if any(k in lower for k in ("action", "rebalanc", "trim", "plan", "do ", "next step")):
        return "action"
    if any(k in lower for k in ("working", "strength", "positive", "bull")):
        return "positive"
    if any(k in lower for k in ("verdict", "summary", "snapshot", "scorecard", "metrics")):
        return "headline"
    return "neutral"


def _render_section(title: str, content: str) -> None:
    with st.container(border=True):
        st.markdown(f"**{title}**")
        st.markdown(content)


def render_ai_response(text: str, *, show_meta: bool = True) -> None:
    """Render research markdown in quiet bordered sections."""
    meta, body = split_meta_and_body(text)
    if show_meta and meta:
        st.caption(meta)

    sections = split_sections(body)
    if not sections:
        st.markdown(body or "_No content._")
        return

    headline = sections[0]
    if _section_tone(headline[0]) == "headline" or "verdict" in headline[0].lower():
        with st.container(border=True):
            st.markdown(f"**{headline[0]}**")
            st.markdown(headline[1])
        sections = sections[1:]

    for title, content in sections:
        _render_section(title, content)


def render_chat_assistant(content: str) -> None:
    render_ai_response(content, show_meta=True)


def _fmt_pct(value: Any, signed: bool = True) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    try:
        num = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{num:+.1f}%" if signed else f"{num:.1f}%"


def _signal_badge(signal: str) -> str:
    s = (signal or "").upper()
    if s == "BULLISH":
        return ":green-badge[BULLISH]"
    if s == "BEARISH":
        return ":red-badge[BEARISH]"
    return ":gray-badge[NEUTRAL]"


def _severity_badge(severity: str) -> str:
    s = (severity or "").lower()
    if s == "high":
        return ":red-badge[High]"
    if s == "med" or s == "medium":
        return ":orange-badge[Med]"
    return ":green-badge[Low]"


def _action_badge(action: str) -> str:
    a = (action or "").upper()
    colors = {
        "TRIM": "red",
        "ADD": "green",
        "HOLD": "blue",
        "WATCH": "orange",
    }
    color = colors.get(a, "gray")
    return f":{color}-badge[{a or '—'}]"


def _render_scorecard_kpis(scorecard: dict[str, Any], coverage: dict[str, Any], provider: str) -> None:
    health = scorecard.get("health_score")
    risk = scorecard.get("risk_score")
    ret = scorecard.get("overall_return_pct")
    with st.container(horizontal=True):
        st.metric(
            "Health score",
            f"{health}/100" if health is not None else "—",
            scorecard.get("health_label"),
            border=True,
            help="Composite portfolio health score from returns, concentration, and technicals",
        )
        st.metric(
            "Risk score",
            f"{risk}/100" if risk is not None else "—",
            scorecard.get("risk_label"),
            border=True,
            delta_color="inverse",
            help="Higher means more portfolio risk from concentration and volatility signals",
        )
        st.metric(
            "Concentration",
            scorecard.get("concentration") or "—",
            f"Top 3 holdings · {scorecard.get('top3_weight_pct', 0):.0f}%",
            border=True,
            delta_color="off",
            help="How much of the portfolio sits in the largest positions",
        )
        st.metric(
            "Portfolio return",
            _fmt_pct(ret),
            f"{scorecard.get('winners', 0)} gainers · {scorecard.get('losers', 0)} losers",
            border=True,
            help="Overall unrealised return across holdings",
        )

    with st.container(horizontal=True):
        st.metric(
            "Average RSI (14)",
            scorecard.get("avg_rsi") if scorecard.get("avg_rsi") is not None else "—",
            border=True,
            help="Average 14-day Relative Strength Index across holdings",
        )
        sma = scorecard.get("pct_above_sma50")
        st.metric(
            "% above 50-day SMA",
            f"{sma:.0f}%" if sma is not None else "—",
            border=True,
            help="Share of holdings trading above their 50-day simple moving average",
        )
        st.metric(
            "Average 20-day momentum",
            _fmt_pct(scorecard.get("avg_momentum_20d")),
            border=True,
            help="Average price change over the last 20 trading days",
        )
        st.metric(
            "RSI extremes",
            f"{scorecard.get('overbought_count', 0)} overbought · {scorecard.get('oversold_count', 0)} oversold",
            border=True,
            delta_color="off",
            help="Count of holdings with RSI above 70 (overbought) or below 30 (oversold)",
        )

    bits = []
    if provider and provider != "—":
        bits.append(f"Model: {provider}")
    if coverage:
        bits.append(f"Technicals loaded: {coverage.get('ok', 0)}/{coverage.get('total', 0)} holdings")
    if bits:
        st.caption(" · ".join(bits) + " · scores are calculated from holdings; notes below are model-written")


def _movers_frame(rows: list[dict], *, losers: bool = False) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["Symbol", "Weight %", "Return %", "P&L ₹"])
    data = [
        {
            "Symbol": r.get("symbol"),
            "Weight %": r.get("weight_pct"),
            "Return %": r.get("return_pct"),
            "P&L ₹": r.get("pnl"),
        }
        for r in rows
    ]
    df = pd.DataFrame(data)
    if losers and not df.empty and "Return %" in df.columns:
        df = df.sort_values("Return %", ascending=True)
    return df


def _render_objective_tables(scorecard: dict[str, Any]) -> None:
    left, right = st.columns(2)
    mover_cfg = {
        "Weight %": st.column_config.NumberColumn(format="%.1f%%"),
        "Return %": st.column_config.NumberColumn(format="%+.1f%%"),
        "P&L ₹": st.column_config.NumberColumn(format="₹%d"),
    }
    with left:
        with st.container(border=True):
            st.markdown("**Top gainers**")
            st.caption("Holdings with the highest unrealised return %")
            st.dataframe(
                _movers_frame(scorecard.get("top_gainers") or []),
                hide_index=True,
                column_config=mover_cfg,
            )
    with right:
        with st.container(border=True):
            st.markdown("**Top losers**")
            st.caption("Holdings with the lowest unrealised return %")
            st.dataframe(
                _movers_frame(scorecard.get("top_losers") or [], losers=True),
                hide_index=True,
                column_config=mover_cfg,
            )

    tech = scorecard.get("technical_table") or []
    if tech:
        with st.container(border=True):
            st.markdown("**Technical indicators by holding**")
            st.caption("Sorted by portfolio weight — RSI, distance from 50-day SMA, 20-day momentum")
            tdf = pd.DataFrame(tech)
            if "Weight %" in tdf.columns:
                tdf = tdf.sort_values("Weight %", ascending=False, na_position="last")
            st.dataframe(
                tdf,
                hide_index=True,
                column_config={
                    "Weight %": st.column_config.ProgressColumn(
                        "Weight %",
                        format="%.1f%%",
                        min_value=0,
                        max_value=max(float(tdf["Weight %"].max() or 1), 1),
                    ),
                    "Return %": st.column_config.NumberColumn(format="%+.1f%%"),
                    "P&L ₹": st.column_config.NumberColumn(format="₹%d"),
                    "RSI-14": st.column_config.NumberColumn(format="%.1f"),
                    "vs SMA50 %": st.column_config.NumberColumn(format="%+.1f%%"),
                    "Mom 20d %": st.column_config.NumberColumn(format="%+.1f%%"),
                },
            )


def _insight_drivers_df(insight: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for item in insight.get("drivers") or []:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "Symbol": item.get("symbol"),
                "Signal": str(item.get("signal") or "NEUTRAL").upper(),
                "Metric": item.get("metric"),
                "Note": item.get("note"),
            }
        )
    return pd.DataFrame(rows)


def _insight_risks_df(insight: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for item in insight.get("risk_flags") or []:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "Flag": item.get("flag"),
                "Severity": item.get("severity"),
                "Symbol": item.get("symbol"),
                "Metric": item.get("metric"),
                "Note": item.get("note"),
            }
        )
    return pd.DataFrame(rows)


def _insight_actions_df(insight: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for item in insight.get("actions") or []:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "Priority": item.get("priority"),
                "Action": str(item.get("action") or "").upper(),
                "Symbol": item.get("symbol"),
                "Weight %": item.get("weight_pct"),
                "Trigger": item.get("trigger"),
                "Reason": item.get("reason"),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty and "Priority" in df.columns:
        df = df.sort_values("Priority", ascending=True, na_position="last")
    return df


def _render_insight_panels(insight: dict[str, Any]) -> None:
    verdict = (insight.get("verdict") or "").strip()
    if verdict:
        st.info(verdict, icon=":material/verified:")

    drivers = _insight_drivers_df(insight)
    risks = _insight_risks_df(insight)
    actions = _insight_actions_df(insight)

    with st.container(border=True):
        st.markdown("**Key return drivers**")
        st.caption("Holdings and technical signals most influencing portfolio performance")
        if drivers.empty:
            st.caption("No driver notes were returned for this run.")
        else:
            for _, row in drivers.iterrows():
                st.markdown(
                    f"{_signal_badge(str(row.get('Signal')))} **{row.get('Symbol')}** · "
                    f"{row.get('Metric') or '—'} — {row.get('Note') or ''}"
                )
            with st.expander("Drivers table", expanded=False):
                st.dataframe(drivers, hide_index=True)

    with st.container(border=True):
        st.markdown("**Main risk exposures**")
        st.caption("Concentration, drawdown, and other portfolio risk flags")
        if risks.empty:
            st.caption("No risk flags were returned for this run.")
        else:
            for _, row in risks.iterrows():
                st.markdown(
                    f"{_severity_badge(str(row.get('Severity')))} **{row.get('Flag')}** · "
                    f"`{row.get('Symbol')}` · {row.get('Metric') or '—'} — {row.get('Note') or ''}"
                )
            with st.expander("Risk flags table", expanded=False):
                st.dataframe(risks, hide_index=True)

    with st.container(border=True):
        st.markdown("**Suggested actions**")
        st.caption("Prioritised hold / trim / add ideas with triggers")
        if actions.empty:
            st.caption("No actions were returned for this run.")
        else:
            for _, row in actions.iterrows():
                wt = row.get("Weight %")
                wt_txt = f"{float(wt):.1f}%" if isinstance(wt, (int, float)) else "—"
                st.markdown(
                    f"**#{row.get('Priority') or '—'}** {_action_badge(str(row.get('Action')))} "
                    f"**{row.get('Symbol')}** ({wt_txt}) · trigger `{row.get('Trigger') or '—'}` — "
                    f"{row.get('Reason') or ''}"
                )
            with st.expander("Actions table", expanded=False):
                st.dataframe(
                    actions,
                    hide_index=True,
                    column_config={
                        "Weight %": st.column_config.NumberColumn(format="%.1f%%"),
                    },
                )

    st.caption("For education only — not personalised advice.")


def render_engine_analysis(result: dict, *, show_payload: bool = False) -> None:
    """Render Portfolio Analysis Engine as scorecards + tables (not essay prose)."""
    provider = result.get("provider") or "—"
    warnings = result.get("warnings") or []
    coverage = (result.get("technicals") or {}).get("coverage") or {}
    scorecard = result.get("scorecard") or {}
    insight = result.get("insight")

    if not result.get("ok") and not scorecard:
        st.warning(
            "Analysis incomplete — market enrichment and/or data prep failed. "
            "Check warnings and API keys."
        )
        if warnings:
            with st.expander(f"Data warnings ({len(warnings)})", expanded=True):
                for w in warnings[:40]:
                    st.caption(w)
        return

    if scorecard:
        _render_scorecard_kpis(scorecard, coverage, provider)
        _render_objective_tables(scorecard)
    else:
        with st.container(horizontal=True):
            st.metric("Provider", provider, border=True)
            if coverage:
                st.metric(
                    "Technicals coverage",
                    f"{coverage.get('ok', 0)}/{coverage.get('total', 0)}",
                    border=True,
                )

    if insight:
        st.divider()
        _render_insight_panels(insight)
    elif result.get("text"):
        st.divider()
        st.warning("Model returned unstructured text — showing raw output.")
        with st.container(border=True):
            st.markdown(result["text"])
    elif result.get("ok"):
        st.info("Objective scorecard ready. LLM insight unavailable — see warnings.")

    if warnings:
        with st.expander(
            f"Data warnings ({len(warnings)})",
            icon=":material/warning:",
            expanded=False,
        ):
            for w in warnings[:40]:
                st.caption(w)
            if len(warnings) > 40:
                st.caption(f"…and {len(warnings) - 40} more")

    if show_payload and result.get("payload"):
        with st.expander("Analysis payload (JSON)", icon=":material/data_object:", expanded=False):
            st.json(result["payload"])
