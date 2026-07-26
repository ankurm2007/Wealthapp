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
