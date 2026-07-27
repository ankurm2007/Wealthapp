"""Persist and query portfolio snapshot history.

Zerodha is live (API). Groww files usually arrive T+1, so history is sleeve-aware:
- Today's Zerodha updates from the live API refresh.
- A Groww upload defaults to *yesterday's* as-of date (finalises yesterday).
- Today's Groww sleeve is carried forward from the latest known Groww until a
  newer Groww file is applied.
"""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo

import pandas as pd

DB_PATH = Path(__file__).parent / "data" / "portfolio_history.db"
IST = ZoneInfo("Asia/Kolkata")

MARKET_CLOSE_IST = time(15, 30)

# Guardrails on *composed* daily totals vs prior day
MIN_VALUE_RATIO_VS_PRIOR = 0.45
MAX_VALUE_RATIO_VS_PRIOR = 2.75

SNAPSHOT_COLS = [
    "snapshot_date",
    "total_invested",
    "total_current",
    "pl_amount",
    "zerodha_current",
    "groww_current",
    "zerodha_invested",
    "groww_invested",
    "holding_count",
    "created_at",
]


def today_ist() -> date:
    return datetime.now(IST).date()


def now_ist() -> datetime:
    return datetime.now(IST)


def yesterday_ist() -> date:
    return today_ist() - timedelta(days=1)


def is_after_market_close(when: datetime | None = None) -> bool:
    when = when or now_ist()
    return when.timetz().replace(tzinfo=None) >= MARKET_CLOSE_IST


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot_date TEXT UNIQUE NOT NULL,
                total_invested REAL NOT NULL,
                total_current REAL NOT NULL,
                pl_amount REAL NOT NULL,
                zerodha_current REAL NOT NULL DEFAULT 0,
                groww_current REAL NOT NULL DEFAULT 0,
                zerodha_invested REAL NOT NULL DEFAULT 0,
                groww_invested REAL NOT NULL DEFAULT 0,
                holding_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            )
            """
        )
        # Migrate older DBs that pre-date sleeve invested columns.
        existing = {
            row[1] for row in conn.execute("PRAGMA table_info(snapshots)").fetchall()
        }
        if "zerodha_invested" not in existing:
            conn.execute(
                "ALTER TABLE snapshots ADD COLUMN zerodha_invested REAL NOT NULL DEFAULT 0"
            )
        if "groww_invested" not in existing:
            conn.execute(
                "ALTER TABLE snapshots ADD COLUMN groww_invested REAL NOT NULL DEFAULT 0"
            )


def _row_to_dict(row: tuple | None, columns: list[str]) -> dict[str, Any] | None:
    if row is None:
        return None
    return dict(zip(columns, row))


def get_snapshot(snapshot_date: date | None = None) -> dict[str, Any] | None:
    init_db()
    day = (snapshot_date or today_ist()).isoformat()
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            f"SELECT {', '.join(SNAPSHOT_COLS)} FROM snapshots WHERE snapshot_date = ? LIMIT 1",
            (day,),
        ).fetchone()
    data = _row_to_dict(row, SNAPSHOT_COLS)
    return _backfill_sleeve_invested(data) if data else None


def get_previous_snapshot(before: date | None = None) -> dict[str, Any] | None:
    init_db()
    day = (before or today_ist()).isoformat()
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            f"""
            SELECT {', '.join(SNAPSHOT_COLS)} FROM snapshots
            WHERE snapshot_date < ?
            ORDER BY snapshot_date DESC
            LIMIT 1
            """,
            (day,),
        ).fetchone()
    data = _row_to_dict(row, SNAPSHOT_COLS)
    return _backfill_sleeve_invested(data) if data else None


def _backfill_sleeve_invested(row: dict[str, Any]) -> dict[str, Any]:
    """Estimate sleeve invested for rows saved before those columns existed."""
    z_inv = float(row.get("zerodha_invested") or 0)
    g_inv = float(row.get("groww_invested") or 0)
    if z_inv > 0 or g_inv > 0:
        return row
    total_inv = float(row.get("total_invested") or 0)
    total_cur = float(row.get("total_current") or 0)
    z_cur = float(row.get("zerodha_current") or 0)
    g_cur = float(row.get("groww_current") or 0)
    if total_cur > 0 and total_inv > 0:
        row["zerodha_invested"] = total_inv * (z_cur / total_cur)
        row["groww_invested"] = total_inv * (g_cur / total_cur)
    return row


def _summary_as_candidate(summary: Mapping[str, float | int]) -> dict[str, float | int]:
    z_cur = float(summary.get("zerodha_current") or 0)
    g_cur = float(summary.get("groww_current") or 0)
    z_inv = float(summary.get("zerodha_invested") or 0)
    g_inv = float(summary.get("groww_invested") or 0)
    total_current = float(summary.get("total_current") or (z_cur + g_cur))
    total_invested = float(summary.get("total_invested") or (z_inv + g_inv))
    return {
        "total_invested": total_invested,
        "total_current": total_current,
        "pl_amount": float(summary.get("pl_amount") or (total_current - total_invested)),
        "zerodha_current": z_cur,
        "groww_current": g_cur,
        "zerodha_invested": z_inv,
        "groww_invested": g_inv,
        "holding_count": int(summary.get("holding_count") or 0),
    }


def _broker_flags(row: Mapping[str, Any]) -> tuple[bool, bool]:
    return float(row.get("zerodha_current") or 0) > 0, float(row.get("groww_current") or 0) > 0


def compose_dual_source_summary(
    live: Mapping[str, float | int],
    *,
    had_zerodha: bool,
    had_groww: bool,
    groww_as_of: date | None = None,
) -> dict[str, Any]:
    """
    Build sleeve-aware totals for history.

    - Zerodha: take live sleeve when fetched; else keep today's saved / prior sleeve.
    - Groww: if a file was uploaded, that sleeve belongs to groww_as_of (default yesterday).
      Today's Groww sleeve uses that value when as-of is yesterday or today; otherwise carry.
    """
    today = today_ist()
    groww_as_of = groww_as_of or yesterday_ist()
    existing_today = get_snapshot(today)
    previous = get_previous_snapshot(today)

    live_z_cur = float(live.get("zerodha_current") or 0)
    live_g_cur = float(live.get("groww_current") or 0)
    live_z_inv = float(live.get("zerodha_invested") or 0)
    live_g_inv = float(live.get("groww_invested") or 0)

    live_z_n = int(live.get("zerodha_holdings") or 0)
    live_g_n = int(live.get("groww_holdings") or 0)

    def _sleeve(source: Mapping[str, Any] | None, kind: str) -> tuple[float, float]:
        if not source:
            return 0.0, 0.0
        if kind == "zerodha":
            return float(source.get("zerodha_current") or 0), float(
                source.get("zerodha_invested") or 0
            )
        return float(source.get("groww_current") or 0), float(
            source.get("groww_invested") or 0
        )

    # --- Zerodha (live API) ---
    z_n = 0
    if had_zerodha and live_z_cur > 0:
        z_cur, z_inv = live_z_cur, live_z_inv
        z_n = live_z_n
        z_note = "live"
    else:
        z_cur, z_inv = _sleeve(existing_today, "zerodha")
        if z_cur <= 0:
            z_cur, z_inv = _sleeve(previous, "zerodha")
        z_note = "carried" if z_cur > 0 else "missing"
        # Holdings count isn't stored per sleeve in DB — leave 0 when carried.

    # --- Groww (usually T+1 file) ---
    # Never invent a missing calendar day. Only patch Groww onto an as-of day
    # that already exists (e.g. tomorrow: as-of=27 when 27 already has Zerodha).
    groww_patch: dict[str, Any] | None = None
    g_n = 0
    as_of_row = get_snapshot(groww_as_of) if groww_as_of < today else None
    if had_groww and live_g_cur > 0:
        if groww_as_of < today and as_of_row is None:
            # File is for a day we never started — do not back-create it, and do
            # not write Groww into today's history yet (keeps day-1 Zerodha-only).
            g_cur, g_inv = _sleeve(existing_today, "groww")
            if g_cur <= 0:
                g_cur, g_inv = _sleeve(previous, "groww")
            g_note = (
                f"file held back (as-of {groww_as_of.isoformat()} not in history yet)"
            )
        else:
            g_cur, g_inv = live_g_cur, live_g_inv
            g_n = live_g_n
            g_note = f"file as-of {groww_as_of.isoformat()}"
            if groww_as_of < today and as_of_row is not None:
                groww_patch = {
                    "date": groww_as_of,
                    "groww_current": g_cur,
                    "groww_invested": g_inv,
                    "groww_holdings": g_n,
                }
    else:
        g_cur, g_inv = _sleeve(existing_today, "groww")
        if g_cur <= 0:
            g_cur, g_inv = _sleeve(previous, "groww")
        g_note = "carried" if g_cur > 0 else "missing"

    # Count only sleeves actually present in this composed snapshot.
    holding_count = z_n + g_n
    if holding_count <= 0:
        # Fallback for carried days (no per-sleeve counts in older rows).
        holding_count = int(live.get("holding_count") or 0)
        if g_cur <= 0 and z_cur > 0 and live_g_n > 0:
            # Live mix included Groww but history is Zerodha-only — don't keep Groww count.
            holding_count = live_z_n or holding_count
        if existing_today and holding_count <= 0:
            holding_count = int(existing_today.get("holding_count") or 0)

    total_current = z_cur + g_cur
    total_invested = z_inv + g_inv
    composed = {
        "total_invested": total_invested,
        "total_current": total_current,
        "pl_amount": total_current - total_invested,
        "zerodha_current": z_cur,
        "groww_current": g_cur,
        "zerodha_invested": z_inv,
        "groww_invested": g_inv,
        "holding_count": holding_count,
        "_meta": {
            "zerodha": z_note,
            "groww": g_note,
            "groww_as_of": groww_as_of.isoformat(),
            "groww_patch": groww_patch,
        },
    }
    return composed


def evaluate_snapshot_quality(
    summary: Mapping[str, float | int],
    *,
    existing: Mapping[str, Any] | None = None,
    previous: Mapping[str, Any] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Quality gate for a *composed* daily candidate."""
    candidate = _summary_as_candidate(summary)
    phase = "closing" if is_after_market_close() else "intraday"

    if force:
        return {
            "ok": True,
            "reason": f"Forced save ({phase}).",
            "phase": phase,
            "candidate": candidate,
        }

    if candidate["total_current"] <= 0:
        return {
            "ok": False,
            "reason": "Composed current value is zero — not saved.",
            "phase": phase,
            "candidate": candidate,
        }
    if candidate["holding_count"] <= 0 and candidate["total_current"] <= 0:
        return {
            "ok": False,
            "reason": "No holdings / value to save.",
            "phase": phase,
            "candidate": candidate,
        }

    cand_z, cand_g = _broker_flags(candidate)
    if existing:
        ex_z, ex_g = _broker_flags(existing)
        # After composition, sleeves should be carried — these fire only if carry failed.
        if ex_z and not cand_z:
            return {
                "ok": False,
                "reason": "Would drop Zerodha sleeve with no carry available — kept earlier.",
                "phase": phase,
                "candidate": candidate,
            }
        if ex_g and not cand_g:
            return {
                "ok": False,
                "reason": "Would drop Groww sleeve with no carry available — kept earlier.",
                "phase": phase,
                "candidate": candidate,
            }

    if previous:
        prior_value = float(previous.get("total_current") or 0)
        if prior_value > 0:
            ratio = candidate["total_current"] / prior_value
            if ratio < MIN_VALUE_RATIO_VS_PRIOR:
                return {
                    "ok": False,
                    "reason": (
                        f"Composed value is only {ratio:.0%} of {previous.get('snapshot_date')} "
                        "— likely incomplete. Not saved."
                    ),
                    "phase": phase,
                    "candidate": candidate,
                }
            if ratio > MAX_VALUE_RATIO_VS_PRIOR:
                return {
                    "ok": False,
                    "reason": (
                        f"Composed value is {ratio:.0%} of {previous.get('snapshot_date')} "
                        "— looks abnormal. Not saved."
                    ),
                    "phase": phase,
                    "candidate": candidate,
                }

    label = "closing" if phase == "closing" else "intraday"
    action = "Updated" if existing else "Saved"
    return {
        "ok": True,
        "reason": f"{action} {label} snapshot (sleeve-aware).",
        "phase": phase,
        "candidate": candidate,
        "provisional": not (cand_z and cand_g),
    }


def save_snapshot(
    *,
    total_invested: float,
    total_current: float,
    pl_amount: float,
    zerodha_current: float = 0.0,
    groww_current: float = 0.0,
    zerodha_invested: float = 0.0,
    groww_invested: float = 0.0,
    holding_count: int = 0,
    snapshot_date: date | None = None,
) -> str:
    init_db()
    snapshot_date = snapshot_date or today_ist()
    created_at = now_ist().isoformat(timespec="seconds")

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO snapshots (
                snapshot_date, total_invested, total_current, pl_amount,
                zerodha_current, groww_current, zerodha_invested, groww_invested,
                holding_count, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(snapshot_date) DO UPDATE SET
                total_invested = excluded.total_invested,
                total_current = excluded.total_current,
                pl_amount = excluded.pl_amount,
                zerodha_current = excluded.zerodha_current,
                groww_current = excluded.groww_current,
                zerodha_invested = excluded.zerodha_invested,
                groww_invested = excluded.groww_invested,
                holding_count = excluded.holding_count,
                created_at = excluded.created_at
            """,
            (
                snapshot_date.isoformat(),
                total_invested,
                total_current,
                pl_amount,
                zerodha_current,
                groww_current,
                zerodha_invested,
                groww_invested,
                holding_count,
                created_at,
            ),
        )

    return snapshot_date.isoformat()


def _pick_zerodha_sleeve(
    *sources: Mapping[str, Any] | None,
) -> tuple[float, float]:
    """First non-zero Zerodha sleeve among sources (existing → prior → live)."""
    for source in sources:
        if not source:
            continue
        z_cur = float(source.get("zerodha_current") or 0)
        if z_cur > 0:
            return z_cur, float(source.get("zerodha_invested") or 0)
    return 0.0, 0.0


def _apply_groww_patch(
    patch: Mapping[str, Any],
    *,
    live_zerodha: Mapping[str, Any] | None = None,
    live_holding_count: int = 0,
) -> str | None:
    """
    Update Groww on an as-of day that already exists. Never create a new day.

    Zerodha stays from the existing row (live Zerodha only fills if that sleeve
    was empty).
    """
    day: date = patch["date"]
    existing = get_snapshot(day)
    if existing is None:
        return None

    z_cur, z_inv = _pick_zerodha_sleeve(existing, live_zerodha)
    if z_cur <= 0:
        return None

    g_cur = float(patch["groww_current"])
    g_inv = float(patch["groww_invested"])
    total_current = z_cur + g_cur
    total_invested = z_inv + g_inv
    holding_count = max(
        int(existing.get("holding_count") or 0),
        int(live_holding_count or 0),
    )
    return save_snapshot(
        total_invested=total_invested,
        total_current=total_current,
        pl_amount=total_current - total_invested,
        zerodha_current=z_cur,
        groww_current=g_cur,
        zerodha_invested=z_inv,
        groww_invested=g_inv,
        holding_count=holding_count,
        snapshot_date=day,
    )


def save_summary_snapshot(
    summary: Mapping[str, float | int],
    *,
    snapshot_date: date | None = None,
    force: bool = False,
    had_zerodha: bool = True,
    had_groww: bool = True,
    groww_as_of: date | None = None,
) -> dict[str, Any]:
    """
    Compose sleeve-aware totals, optionally finalise Groww on T+1 as-of day,
    then save today when quality checks pass.
    """
    day = snapshot_date or today_ist()
    composed = compose_dual_source_summary(
        summary,
        had_zerodha=had_zerodha,
        had_groww=had_groww,
        groww_as_of=groww_as_of,
    )
    meta = dict(composed.pop("_meta", {}))
    notes: list[str] = []

    patch = meta.get("groww_patch")
    patched_day = None
    if patch:
        patched_day = _apply_groww_patch(
            patch,
            live_zerodha={
                "zerodha_current": composed.get("zerodha_current"),
                "zerodha_invested": composed.get("zerodha_invested"),
            },
            live_holding_count=int(composed.get("holding_count") or 0),
        )
        if patched_day:
            notes.append(f"Groww finalised on {patched_day} (T+1; existing day updated).")
        else:
            notes.append(
                "Groww as-of day not updated (missing day is never invented) — "
                "history keeps prior sleeves."
            )

    existing = get_snapshot(day)
    previous = get_previous_snapshot(day)
    verdict = evaluate_snapshot_quality(
        composed, existing=existing, previous=previous, force=force
    )
    if not verdict["ok"]:
        return {
            "saved": False,
            "date": day.isoformat() if existing else None,
            "reason": " ".join(notes + [verdict["reason"]]).strip(),
            "phase": verdict["phase"],
            "provisional": False,
            "meta": meta,
            "patched_day": patched_day,
        }

    cand = verdict["candidate"]
    saved_date = save_snapshot(
        total_invested=float(cand["total_invested"]),
        total_current=float(cand["total_current"]),
        pl_amount=float(cand["pl_amount"]),
        zerodha_current=float(cand["zerodha_current"]),
        groww_current=float(cand["groww_current"]),
        zerodha_invested=float(cand["zerodha_invested"]),
        groww_invested=float(cand["groww_invested"]),
        holding_count=int(cand["holding_count"]),
        snapshot_date=day,
    )
    z_label = meta.get("zerodha", "?")
    g_label = meta.get("groww", "?")
    notes.append(
        f"{verdict['reason']} Zerodha={z_label}; Groww={g_label}."
    )
    return {
        "saved": True,
        "date": saved_date,
        "reason": " ".join(notes),
        "phase": verdict["phase"],
        "provisional": bool(verdict.get("provisional")),
        "meta": meta,
        "patched_day": patched_day,
    }


def has_snapshot_for(snapshot_date: date | None = None) -> bool:
    return get_snapshot(snapshot_date) is not None


def load_snapshots() -> pd.DataFrame:
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(
            f"""
            SELECT
                snapshot_date AS date,
                total_invested,
                total_current,
                pl_amount,
                zerodha_current,
                groww_current,
                zerodha_invested,
                groww_invested,
                holding_count,
                created_at
            FROM snapshots
            ORDER BY snapshot_date
            """,
            conn,
        )

    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"])
    return df


def get_monthly_summary(df: pd.DataFrame | None = None) -> pd.DataFrame:
    history = load_snapshots() if df is None else df.copy()
    if history.empty:
        return history

    history = history.sort_values("date").set_index("date")
    monthly = history.resample("ME").last().dropna(subset=["total_current"]).copy()
    monthly["monthly_change"] = monthly["total_current"].diff()
    monthly["monthly_growth_pct"] = monthly["total_current"].pct_change() * 100
    return monthly.reset_index()


def get_last_snapshot_date() -> date | None:
    df = load_snapshots()
    if df.empty:
        return None
    return df["date"].max().date()


def history_status(*, lookback_days: int = 14) -> dict:
    df = load_snapshots()
    today = today_ist()
    after_close = is_after_market_close()
    if df.empty:
        return {
            "count": 0,
            "today_saved": False,
            "last_date": None,
            "missing_recent": lookback_days,
            "caption": "No daily history yet — refresh to save Zerodha live + carried Groww.",
            "after_close": after_close,
        }

    dates = {ts.date() for ts in df["date"]}
    last_date = max(dates)
    today_saved = today in dates
    window_start = today - timedelta(days=lookback_days - 1)
    expected = {d for d in (window_start + timedelta(days=i) for i in range(lookback_days)) if d <= today}
    missing = len(expected - dates)

    phase = "after close" if after_close else "intraday"
    if today_saved:
        caption = f"Today saved ({phase}) · {len(dates)} day(s) · Groww is T+1-aware"
    else:
        caption = f"Today not saved · last {last_date.strftime('%d %b')}"
    if missing and lookback_days <= 30:
        caption += f" · {missing} gap(s) in last {lookback_days}d"

    return {
        "count": len(dates),
        "today_saved": today_saved,
        "last_date": last_date,
        "missing_recent": missing,
        "caption": caption,
        "after_close": after_close,
    }
