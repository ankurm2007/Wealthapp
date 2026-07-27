"""Writable data paths — Streamlit Cloud repo mount is read-only; use /tmp there."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

REPO_DATA = Path(__file__).resolve().parent / "data"
CLOUD_DATA = Path("/tmp/wealthapp_data")


def is_streamlit_cloud() -> bool:
    """Detect Streamlit Community Cloud / sharing runtime."""
    if os.environ.get("STREAMLIT_RUNTIME_ENV", "").lower() == "cloud":
        return True
    if Path("/mount/src").exists():
        return True
    if Path("/home/adminuser").exists():
        return True
    return False


def _dir_is_writable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True
    except OSError:
        return False


@lru_cache(maxsize=1)
def data_dir() -> Path:
    """Prefer repo data/ locally; fall back to /tmp on Cloud."""
    if _dir_is_writable(REPO_DATA):
        return REPO_DATA
    CLOUD_DATA.mkdir(parents=True, exist_ok=True)
    return CLOUD_DATA


def data_file(name: str) -> Path:
    return data_dir() / name
