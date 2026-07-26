"""Shared layout, formatting, and visual patterns for the wealth dashboard."""

from __future__ import annotations

import streamlit as st

# Align with .streamlit/config.toml chartCategoricalColors
CHART_BLUE = "#2563EB"
CHART_GREEN = "#059669"
CHART_VIOLET = "#7C3AED"
CHART_RED = "#DC2626"
CHART_AMBER = "#D97706"
CHART_CYAN = "#0891B2"
CHART_ORANGE = "#EA580C"
CHART_SLATE = "#64748B"

CHART_LINE_INVESTED = CHART_SLATE
CHART_LINE_CURRENT = CHART_GREEN
CHART_BAR_CHANGE = CHART_BLUE
CHART_BAR_GROWTH = CHART_VIOLET
CHART_AREA_ZERODHA = CHART_ORANGE
CHART_AREA_GROWW = CHART_CYAN


def format_inr(value: float, *, decimals: int = 0) -> str:
    if decimals == 0:
        return f"₹{value:,.0f}"
    return f"₹{value:,.{decimals}f}"


def format_pct(value: float, *, decimals: int = 1, signed: bool = True) -> str:
    if signed:
        return f"{value:+.{decimals}f}%"
    return f"{value:.{decimals}f}%"


def section(title: str, caption: str | None = None, *, icon: str = "analytics") -> None:
    st.space("small")
    st.markdown(f"### :material/{icon}: {title}")
    if caption:
        st.caption(caption)


def page_header(view: str, summary: dict | None = None) -> None:
    meta = {
        "Portfolio": {
            "title": "Portfolio",
            "caption": "Combined wealth, platform split, and holdings",
            "icon": "account_balance_wallet",
            "badge": None,
        },
        "Trends": {
            "title": "Trends",
            "caption": "Snapshot history and growth over time",
            "icon": "timeline",
            "badge": None,
        },
        "Insights": {
            "title": "Insights",
            "caption": "Allocation, benchmarks, risk, and AI research",
            "icon": "insights",
            "badge": None,
        },
    }
    info = meta.get(view, meta["Portfolio"])
    with st.container(border=True):
        head_left, head_right = st.columns([5, 2])
        with head_left:
            st.markdown(f"## :material/{info['icon']}: {info['title']}")
            st.caption(info["caption"])
        with head_right:
            if summary and view == "Portfolio":
                ret = (
                    summary["pl_amount"] / summary["total_invested"] * 100
                    if summary.get("total_invested")
                    else 0
                )
                tone = "green" if ret >= 0 else "red"
                st.markdown(f":{tone}-badge[{format_pct(ret)} total return]")
                st.caption(f"{format_inr(summary['total_current'])} current value")


def kpi_row(items: list[tuple[str, str, str | None]]) -> None:
    """Render a horizontal row of bordered metrics: (label, value, delta)."""
    with st.container(horizontal=True):
        for label, value, delta in items:
            st.metric(label, value, delta, border=True)


def empty_state(
    title: str,
    message: str,
    *,
    icon: str = "info",
    hint: str | None = None,
) -> None:
    with st.container(border=True):
        st.markdown(f"#### :material/{icon}: {title}")
        st.markdown(message)
        if hint:
            st.caption(hint)


def sidebar_block(title: str, *, icon: str = "tune") -> None:
    st.markdown(f":material/{icon}: **{title}**")


def status_banner(message: str, *, kind: str = "info") -> None:
    icons = {
        "info": "info",
        "success": "check_circle",
        "warning": "warning",
        "error": "error",
    }
    icon = icons.get(kind, "info")
    with st.container(border=True):
        st.markdown(f":material/{icon}: {message}")
