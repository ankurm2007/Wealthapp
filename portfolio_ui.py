"""Shared layout, formatting, and visual patterns for the wealth dashboard."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import streamlit as st

# Palettes mirror .streamlit/config.toml chartCategoricalColors per mode.
_LIGHT = {
    "indigo": "#2563EB",
    "green": "#16A34A",
    "cyan": "#0EA5E9",
    "red": "#E11D48",
    "amber": "#EA580C",
    "violet": "#7C3AED",
    "teal": "#2563EB",
    "slate": "#64748B",
}
_DARK = _LIGHT  # App is locked to light theme; keep palette consistent.

# Legacy module-level constants (light values) kept for direct imports.
CHART_BLUE = _LIGHT["indigo"]
CHART_GREEN = _LIGHT["green"]
CHART_VIOLET = _LIGHT["violet"]
CHART_RED = _LIGHT["red"]
CHART_AMBER = _LIGHT["amber"]
CHART_CYAN = _LIGHT["cyan"]
CHART_ORANGE = _LIGHT["amber"]
CHART_SLATE = _LIGHT["slate"]

CHART_LINE_INVESTED = CHART_SLATE
CHART_LINE_CURRENT = CHART_GREEN
CHART_BAR_CHANGE = CHART_BLUE
CHART_BAR_GROWTH = CHART_VIOLET
CHART_AREA_ZERODHA = CHART_AMBER
CHART_AREA_GROWW = CHART_CYAN

VIEW_GUIDES = {
    "Portfolio": {
        "label": "Portfolio overview",
        "caption": "Current market value, decision snapshot, broker split, and the full holdings table",
        "do_next": "Start with Decision snapshot below — then open Insights before changing a large position",
        "icon": "account_balance_wallet",
    },
    "Insights": {
        "label": "Portfolio analysis",
        "caption": "Allocation, performance drivers, Nifty comparison, and portfolio-level risk metrics",
        "do_next": "Begin on Scorecard, then open Allocation if a few stocks dominate weight",
        "icon": "insights",
    },
    "Research": {
        "label": "Stock research",
        "caption": "Company fundamentals for your holdings, written research notes, and Q&A on the portfolio",
        "do_next": "Load Fundamentals first, then use Notes for a written summary",
        "icon": "menu_book",
    },
    "Trends": {
        "label": "Wealth trends",
        "caption": "Daily history — one point per day after close; Groww finalises T+1 as-of",
        "do_next": "Refresh the portfolio once a day to keep the trend line growing",
        "icon": "timeline",
    },
}


def is_dark() -> bool:
    try:
        return st.context.theme.type == "dark"
    except Exception:
        return False


def palette() -> dict[str, str]:
    """Accent colors matched to the active light/dark theme."""
    return _DARK if is_dark() else _LIGHT


def chart_colors() -> dict[str, str]:
    """Named series colors for native Streamlit charts."""
    p = palette()
    return {
        "invested": p["slate"],
        "current": p["green"],
        "change": p["indigo"],
        "growth": p["violet"],
        "zerodha": p["amber"],
        "groww": p["cyan"],
    }


def format_inr(value: float, *, decimals: int = 0) -> str:
    if decimals == 0:
        return f"₹{value:,.0f}"
    return f"₹{value:,.{decimals}f}"


def format_inr_compact(value: float) -> str:
    """Indian-style short form: ₹1.24 Cr, ₹8.5 L, ₹42,000."""
    sign = "-" if value < 0 else ""
    amount = abs(value)
    if amount >= 1_00_00_000:
        return f"{sign}₹{amount / 1_00_00_000:,.2f} Cr"
    if amount >= 1_00_000:
        return f"{sign}₹{amount / 1_00_000:,.2f} L"
    if amount >= 1_000:
        return f"{sign}₹{amount / 1_000:,.1f} K"
    return f"{sign}₹{amount:,.0f}"


def format_pct(value: float, *, decimals: int = 1, signed: bool = True) -> str:
    if signed:
        return f"{value:+.{decimals}f}%"
    return f"{value:.{decimals}f}%"


def tone_badge(value: float, text: str) -> str:
    """Inline markdown badge colored by sign."""
    color = "green" if value >= 0 else "red"
    return f":{color}-badge[{text}]"


def section(
    title: str,
    caption: str | None = None,
    *,
    icon: str | None = None,
    badge: str | None = None,
) -> None:
    """Quiet section title — content first, no icon noise."""
    del icon  # kept for call-site compatibility; headings stay text-only
    st.space("small")
    heading = f"### {title}"
    if badge:
        heading = f"{heading} &nbsp;{badge}"
    st.markdown(heading)
    if caption:
        st.caption(caption)


def hero(
    title: str,
    caption: str,
    *,
    icon: str,
    headline: str | None = None,
    headline_label: str | None = None,
    badges: Sequence[str] = (),
    stats: Sequence[tuple[str, str]] = (),
    do_next: str | None = None,
) -> None:
    """Top-of-page banner: lead with the figure that matters."""
    del icon
    with st.container(border=True):
        left, right = st.columns([3, 2], vertical_alignment="center")
        with left:
            if headline:
                st.caption(headline_label or title)
                st.markdown(f"# {headline}")
            else:
                st.markdown(f"# {title}")
            st.caption(caption)
            if badges:
                st.markdown(" &nbsp;".join(badges))
            if do_next:
                st.caption(f":blue-badge[Suggested next step] {do_next}")
        with right:
            if stats:
                with st.container(horizontal=True, gap="small"):
                    for label, value in stats:
                        st.metric(label, value, border=True)


def page_header(view: str, summary: Mapping[str, Any] | None = None) -> None:
    info = VIEW_GUIDES.get(view, VIEW_GUIDES["Portfolio"])

    if view != "Portfolio" or not summary:
        hero(
            info["label"],
            info["caption"],
            icon=info["icon"],
            do_next=info.get("do_next"),
        )
        return

    invested = float(summary.get("total_invested") or 0)
    current = float(summary.get("total_current") or 0)
    pl = float(summary.get("pl_amount") or 0)
    ret = (pl / invested * 100) if invested else 0.0
    n = int(summary.get("holding_count") or 0)
    realized = float(summary.get("realized_total") or 0)
    economic = summary.get("economic_pl")
    has_realized = summary.get("has_realized_pnl")

    badges = [tone_badge(ret, f"{format_pct(ret)} unrealised return")]
    if has_realized:
        badges.append(tone_badge(realized, f"{format_inr_compact(realized)} booked P&L"))
        if economic is not None:
            badges.append(tone_badge(float(economic), f"{format_inr_compact(float(economic))} economic"))
    badges.append(f":gray-badge[{n} holdings]")

    stats = [
        ("Total invested", format_inr_compact(invested)),
        ("Unrealised P&L", format_inr_compact(pl)),
    ]
    if has_realized:
        stats.append(("Booked P&L", format_inr_compact(realized)))

    hero(
        info["label"],
        info["caption"],
        icon=info["icon"],
        headline=format_inr(current),
        headline_label="Total portfolio value",
        badges=badges,
        stats=stats,
        do_next=info.get("do_next"),
    )


def kpi_row(items: Iterable[Any]) -> None:
    """Row of bordered metrics for scannable KPIs."""
    with st.container(horizontal=True):
        for item in items:
            if isinstance(item, Mapping):
                delta_color = item.get("delta_color", "normal")
                st.metric(
                    item["label"],
                    item["value"],
                    item.get("delta"),
                    delta_color=delta_color,
                    delta_arrow=item.get(
                        "delta_arrow", "off" if delta_color == "off" else "auto"
                    ),
                    help=item.get("help"),
                    border=True,
                    chart_data=item.get("chart"),
                    chart_type=item.get("chart_type", "line"),
                )
            else:
                label, value, delta = item
                st.metric(label, value, delta, border=True)


def empty_state(
    title: str,
    message: str,
    *,
    icon: str = "info",
    hint: str | None = None,
    steps: Sequence[str] = (),
) -> None:
    with st.container(border=True, horizontal_alignment="center"):
        st.space("small")
        st.markdown(f"### :material/{icon}: {title}", text_alignment="center")
        st.markdown(message, text_alignment="center")
        if steps:
            st.space("small")
            for i, step in enumerate(steps, start=1):
                st.markdown(f"**{i}.** {step}", text_alignment="center")
        elif hint:
            st.caption(hint, text_alignment="center")
        st.space("small")


def sidebar_block(title: str, *, icon: str = "tune") -> None:
    del icon
    st.sidebar.markdown(f"##### {title}")


def sidebar_howto() -> None:
    """Compact how-to at top of sidebar."""
    st.sidebar.markdown("##### How to load your portfolio")
    st.sidebar.caption("1 · Connect Zerodha (live) and/or upload Groww (T+1 as-of)")
    st.sidebar.caption("2 · Upload Realised P&L so booked losses from sells are visible")
    st.sidebar.caption(
        "3 · Refresh — today's history point auto-saves after 3:30 PM IST"
    )
    st.sidebar.space("small")


def status_banner(message: str, *, kind: str = "info") -> None:
    icons = {
        "info": "info",
        "success": "check_circle",
        "warning": "warning",
        "error": "error",
    }
    icon = icons.get(kind, "info")
    renderer = {
        "info": st.info,
        "success": st.success,
        "warning": st.warning,
        "error": st.error,
    }.get(kind, st.info)
    renderer(message, icon=f":material/{icon}:")


# Soft theme accents for diagnosis finding chips (match config.toml)
_DIAGNOSIS_ACCENT = {
    "blue": ("#2563EB", "#EFF6FF"),
    "violet": ("#7C3AED", "#F5F3FF"),
    "red": ("#E11D48", "#FFF1F2"),
    "orange": ("#EA580C", "#FFF7ED"),
    "gray": ("#64748B", "#F8FAFC"),
    "green": ("#16A34A", "#F0FDF4"),
    "yellow": ("#CA8A04", "#FEFCE8"),
    "primary": ("#2563EB", "#EFF6FF"),
}


def _escape_html(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def render_diagnosis_brief(brief: Mapping[str, Any]) -> None:
    """
    Review-ready portfolio diagnosis with calmer visual hierarchy:
    verdict → KPI strip → accent finding cards → priority list.
    """
    health = brief.get("health", "mixed")
    banner_kind = {
        "healthy": "success",
        "mixed": "warning",
        "needs attention": "error",
    }.get(health, "info")
    badge = {
        "healthy": ":green-badge[Looking good]",
        "mixed": ":orange-badge[Mixed]",
        "needs attention": ":red-badge[Needs attention]",
    }.get(health, ":blue-badge[Overview]")

    # Verdict — callout + plain "picture this" summary
    status_banner(
        f"{badge}  {brief.get('headline', '')}",
        kind=banner_kind,
    )
    if brief.get("summary"):
        st.markdown(brief["summary"])

    # KPI strip — everyday labels
    kpis = list(brief.get("kpis") or [])
    if kpis:
        st.space("small")
        st.caption("At a glance")
        with st.container(horizontal=True):
            for kpi in kpis:
                st.metric(
                    kpi["label"],
                    kpi["value"],
                    border=True,
                    help=kpi.get("help"),
                )

    # Findings — soft accent cards (2-up)
    findings = list(brief.get("findings") or [])
    if findings:
        st.space("small")
        st.markdown("##### What stands out")
        st.caption("Read the big number first — the note below explains what it means")

        for row_start in range(0, len(findings), 2):
            pair = findings[row_start : row_start + 2]
            cols = st.columns(2, gap="medium")
            for col, item in zip(cols, pair):
                color = str(item.get("color") or "blue")
                accent, soft = _DIAGNOSIS_ACCENT.get(color, _DIAGNOSIS_ACCENT["blue"])
                title = _escape_html(item.get("title", ""))
                metric = _escape_html(item.get("metric", ""))
                text = _escape_html(item.get("text", ""))
                with col:
                    st.html(
                        f"""
<div style="
  height:100%;
  min-height:7.5rem;
  padding:0.9rem 1rem 1rem 1rem;
  border-radius:10px;
  border:1px solid {accent}33;
  border-left:4px solid {accent};
  background:linear-gradient(180deg, {soft} 0%, #FFFFFF 72%);
  box-sizing:border-box;
">
  <div style="
    font-size:0.78rem;
    font-weight:600;
    letter-spacing:0.02em;
    color:{accent};
    margin-bottom:0.35rem;
  ">{title}</div>
  <div style="
    font-family:'Literata',Georgia,serif;
    font-size:1.45rem;
    font-weight:600;
    line-height:1.2;
    color:#1E293B;
    margin-bottom:0.45rem;
  ">{metric}</div>
  <div style="
    font-size:0.92rem;
    line-height:1.5;
    color:#334155;
  ">{text}</div>
</div>
"""
                    )

    # Priorities — plain next steps
    priorities = list(brief.get("priorities") or [])
    st.space("small")
    st.markdown("##### Do this next")
    st.caption("Clear decisions before you buy or sell anything large")
    with st.container(border=True):
        if not priorities:
            st.markdown(
                ":green-badge[Nothing urgent] No red or amber flags from the automated checks."
            )
            return

        actions = [p for p in priorities if p.get("severity") == "fail"]
        monitors = [p for p in priorities if p.get("severity") == "warn"]

        if actions:
            st.markdown("**Fix these first**")
            for item in actions:
                detail = item.get("detail") or ""
                line = f":red-badge[Fix] **{item.get('title', '')}**"
                if detail:
                    line = f"{line} — {detail}"
                st.markdown(line)

        if monitors:
            if actions:
                st.space("small")
            st.markdown("**Keep an eye on**")
            for item in monitors:
                detail = item.get("detail") or ""
                line = f":orange-badge[Watch] **{item.get('title', '')}**"
                if detail:
                    line = f"{line} — {detail}"
                st.markdown(line)
