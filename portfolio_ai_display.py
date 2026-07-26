"""Structured, scannable rendering for AI research output."""

from __future__ import annotations

import re

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
    tone = _section_tone(title)
    with st.container(border=True):
        if tone == "headline":
            st.markdown(f"**:blue[{title}]**")
        elif tone == "risk":
            st.markdown(f"**:red[{title}]**")
        elif tone == "action":
            st.markdown(f"**:violet[{title}]**")
        elif tone == "positive":
            st.markdown(f"**:green[{title}]**")
        else:
            st.markdown(f"**{title}**")
        st.markdown(content)


def render_ai_response(text: str, *, show_meta: bool = True) -> None:
    """Render AI markdown in bordered sections with color cues."""
    meta, body = split_meta_and_body(text)
    if show_meta and meta:
        st.caption(f":material/smart_toy: {meta}")

    sections = split_sections(body)
    if not sections:
        st.markdown(body or "_No content._")
        return

    headline = sections[0]
    if _section_tone(headline[0]) == "headline" or "verdict" in headline[0].lower():
        with st.container(border=True):
            st.markdown(f"**:blue[{headline[0]}]**")
            st.markdown(headline[1])
        sections = sections[1:]

    for title, content in sections:
        _render_section(title, content)


def render_chat_assistant(content: str) -> None:
    render_ai_response(content, show_meta=True)


ENGINE_SECTION_ORDER = (
    "Key Portfolio Drivers & Technical Health",
    "Risk Exposure & Concentration Flags",
    "Actionable Tactical Recommendations",
)


def _match_engine_section(title: str) -> str | None:
    lower = title.lower()
    for canonical in ENGINE_SECTION_ORDER:
        if canonical.lower() in lower or lower in canonical.lower():
            return canonical
    if "driver" in lower or "technical" in lower:
        return ENGINE_SECTION_ORDER[0]
    if "risk" in lower or "concentration" in lower:
        return ENGINE_SECTION_ORDER[1]
    if "action" in lower or "tactical" in lower or "recommend" in lower:
        return ENGINE_SECTION_ORDER[2]
    return None


def render_engine_analysis(result: dict, *, show_payload: bool = False) -> None:
    """Render Portfolio Analysis Engine output as three structured sections."""
    provider = result.get("provider") or "—"
    warnings = result.get("warnings") or []
    coverage = (result.get("technicals") or {}).get("coverage") or {}

    with st.container(horizontal=True):
        st.metric("Provider", provider, border=True)
        if coverage:
            st.metric(
                "Technicals coverage",
                f"{coverage.get('ok', 0)}/{coverage.get('total', 0)}",
                border=True,
            )
        portfolio = (result.get("payload") or {}).get("portfolio") or {}
        if portfolio.get("overall_return_pct") is not None:
            st.metric(
                "Portfolio return",
                f"{portfolio['overall_return_pct']:+.1f}%",
                border=True,
            )

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

    if not result.get("ok") or not result.get("text"):
        st.warning(
            "Analysis incomplete — market enrichment and/or LLM call failed. "
            "Check warnings above and API keys in secrets."
        )
        if show_payload and result.get("payload"):
            with st.expander("Raw analysis payload", expanded=False):
                st.json(result["payload"])
        return

    st.caption(f":material/smart_toy: Quantitative portfolio analysis · {provider}")

    sections = split_sections(result["text"])
    by_key: dict[str, str] = {}
    extras: list[tuple[str, str]] = []
    for title, content in sections:
        key = _match_engine_section(title)
        if key and key not in by_key:
            by_key[key] = content
        else:
            extras.append((title, content))

    icons = {
        ENGINE_SECTION_ORDER[0]: "blue",
        ENGINE_SECTION_ORDER[1]: "red",
        ENGINE_SECTION_ORDER[2]: "violet",
    }

    for canonical in ENGINE_SECTION_ORDER:
        content = by_key.get(canonical)
        if not content:
            continue
        color = icons[canonical]
        with st.container(border=True):
            st.markdown(f"**:{color}[{canonical}]**")
            st.markdown(content)

    # Fallback if the model ignored ## headers
    if not by_key:
        with st.container(border=True):
            st.markdown(result["text"])

    for title, content in extras:
        with st.expander(title, expanded=False):
            st.markdown(content)

    if show_payload and result.get("payload"):
        with st.expander("Analysis payload (JSON)", icon=":material/data_object:", expanded=False):
            st.json(result["payload"])
