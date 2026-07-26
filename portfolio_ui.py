"""Shared layout, formatting, and visual patterns for the wealth dashboard."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import streamlit as st

# Palettes mirror .streamlit/config.toml chartCategoricalColors per mode.
_LIGHT = {
    "indigo": "#4F46E5",
    "green": "#047857",
    "cyan": "#0E7490",
    "red": "#BE123C",
    "amber": "#B45309",
    "violet": "#6D28D9",
    "teal": "#0F766E",
    "slate": "#94A3B8",
}
_DARK = {
    "indigo": "#818CF8",
    "green": "#34D399",
    "cyan": "#38BDF8",
    "red": "#FB7185",
    "amber": "#FBBF24",
    "violet": "#A78BFA",
    "teal": "#2DD4BF",
    "slate": "#94A3B8",
}

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
    icon: str = "analytics",
    badge: str | None = None,
) -> None:
    st.space("small")
    heading = f"### :material/{icon}: {title}"
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
) -> None:
    """Top-of-page banner with an oversized headline figure."""
    with st.container(border=True):
        left, right = st.columns([3, 2], vertical_alignment="center")
        with left:
            if headline:
                st.caption(f":material/{icon}: {headline_label or title}")
                st.markdown(f"# {headline}")
            else:
                st.markdown(f"# :material/{icon}: {title}")
            st.caption(caption)
            if badges:
                st.markdown(" &nbsp;".join(badges))
        with right:
            if stats:
                with st.container(horizontal=True, gap="small"):
                    for label, value in stats:
                        st.metric(label, value, border=True)


def page_header(view: str, summary: Mapping[str, Any] | None = None) -> None:
    meta = {
        "Portfolio": {
            "caption": "Combined wealth, platform split, and holdings",
            "icon": "account_balance_wallet",
        },
        "Trends": {
            "caption": "Snapshot history and growth over time",
            "icon": "timeline",
        },
        "Insights": {
            "caption": "Allocation, benchmarks, risk, and AI research",
            "icon": "insights",
        },
    }
    info = meta.get(view, meta["Portfolio"])

    if view != "Portfolio" or not summary:
        hero(view, info["caption"], icon=info["icon"])
        return

    invested = float(summary.get("total_invested") or 0)
    current = float(summary.get("total_current") or 0)
    pl = float(summary.get("pl_amount") or 0)
    ret = (pl / invested * 100) if invested else 0.0

    hero(
        view,
        info["caption"],
        icon=info["icon"],
        headline=format_inr(current),
        headline_label="Total wealth",
        badges=[
            tone_badge(ret, f"{format_pct(ret)} return"),
            f":blue-badge[{summary.get('holding_count', 0)} holdings]",
        ],
        stats=[
            ("Invested", format_inr_compact(invested)),
            ("Unrealised P&L", format_inr_compact(pl)),
        ],
    )


def kpi_row(items: Iterable[Any]) -> None:
    """Row of bordered metrics.

    Each item is either a `(label, value, delta)` tuple or a mapping supporting
    `label`, `value`, `delta`, `delta_color`, `help`, `chart`, and `chart_type`.
    """
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
) -> None:
    with st.container(border=True, horizontal_alignment="center"):
        st.space("small")
        st.markdown(f"### :material/{icon}:", text_alignment="center")
        st.markdown(f"#### {title}", text_alignment="center")
        st.markdown(message, text_alignment="center")
        if hint:
            st.caption(hint, text_alignment="center")
        st.space("small")


def sidebar_block(title: str, *, icon: str = "tune") -> None:
    st.sidebar.markdown(f"##### :material/{icon}: {title}")


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
