"""Persist and query portfolio snapshot history.

Design (one row = one calendar day in IST):

1. Zerodha sleeve updates from the live API on *today* only.
2. Groww files are T+1: an upload with as-of date D updates Groww on day D.
   If D is missing (e.g. Cloud /tmp wipe), D is created using live Zerodha carry-back
   — never a Groww-only half-row.
3. Today's Groww sleeve is carried from the latest known Groww when no same-day file.
4. Auto-save for *today* runs after 3:30 PM IST, or when today already has a row
   (intraday refresh), or on the first-ever snapshot. Overnight refreshes can still
   finalise yesterday's Groww without inventing a twin "today" row.
"""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo

import pandas as pd

import app_paths

DB_PATH = app_paths.data_file("portfolio_history.db")
IST = ZoneInfo("Asia/Kolkata")

MARKET_CLOSE_IST = time(15, 30)

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
    "zerodha_holdings",
    "groww_holdings",
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
                zerodha_holdings INTEGER NOT NULL DEFAULT 0,
                groww_holdings INTEGER NOT NULL DEFAULT 0,
                holding_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            )
            """
        )
        existing = {
            row[1] for row in conn.execute("PRAGMA table_info(snapshots)").fetchall()
        }
        migrations = {
            "zerodha_invested": "REAL NOT NULL DEFAULT 0",
            "groww_invested": "REAL NOT NULL DEFAULT 0",
            "zerodha_holdings": "INTEGER NOT NULL DEFAULT 0",
            "groww_holdings": "INTEGER NOT NULL DEFAULT 0",
        }
        for col, ddl in migrations.items():
            if col not in existing:
                conn.execute(f"ALTER TABLE snapshots ADD COLUMN {col} {ddl}")


def _row_to_dict(row: tuple | None, columns: list[str]) -> dict[str, Any] | None:
    if row is None:
        return None
    return dict(zip(columns, row))


def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    """Fill missing sleeve invested/holdings and keep totals consistent."""
    z_cur = float(row.get("zerodha_current") or 0)
    g_cur = float(row.get("groww_current") or 0)
    z_inv = float(row.get("zerodha_invested") or 0)
    g_inv = float(row.get("groww_invested") or 0)
    z_n = int(row.get("zerodha_holdings") or 0)
    g_n = int(row.get("groww_holdings") or 0)
    total_inv = float(row.get("total_invested") or 0)
    total_cur = float(row.get("total_current") or 0)

    if z_inv <= 0 and g_inv <= 0 and total_cur > 0 and total_inv > 0:
        z_inv = total_inv * (z_cur / total_cur)
        g_inv = total_inv * (g_cur / total_cur)

    # Older rows only had combined holding_count — split proportionally if needed.
    combined = int(row.get("holding_count") or 0)
    if z_n <= 0 and g_n <= 0 and combined > 0:
        if z_cur > 0 and g_cur <= 0:
            z_n = combined
        elif g_cur > 0 and z_cur <= 0:
            g_n = combined
        elif z_cur > 0 and g_cur > 0:
            z_n = max(1, round(combined * (z_cur / (z_cur + g_cur))))
            g_n = max(0, combined - z_n)

    row["zerodha_current"] = z_cur
    row["groww_current"] = g_cur
    row["zerodha_invested"] = z_inv
    row["groww_invested"] = g_inv
    row["zerodha_holdings"] = z_n
    row["groww_holdings"] = g_n
    row["holding_count"] = z_n + g_n
    row["total_current"] = z_cur + g_cur
    row["total_invested"] = z_inv + g_inv
    row["pl_amount"] = row["total_current"] - row["total_invested"]
    return row


def get_snapshot(snapshot_date: date | None = None) -> dict[str, Any] | None:
    init_db()
    day = (snapshot_date or today_ist()).isoformat()
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            f"SELECT {', '.join(SNAPSHOT_COLS)} FROM snapshots WHERE snapshot_date = ? LIMIT 1",
            (day,),
        ).fetchone()
    data = _row_to_dict(row, SNAPSHOT_COLS)
    return _normalize_row(data) if data else None


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
    return _normalize_row(data) if data else None


def _sleeve(source: Mapping[str, Any] | None, kind: str) -> tuple[float, float, int]:
    if not source:
        return 0.0, 0.0, 0
    if kind == "zerodha":
        return (
            float(source.get("zerodha_current") or 0),
            float(source.get("zerodha_invested") or 0),
            int(source.get("zerodha_holdings") or 0),
        )
    return (
        float(source.get("groww_current") or 0),
        float(source.get("groww_invested") or 0),
        int(source.get("groww_holdings") or 0),
    )


def _broker_flags(row: Mapping[str, Any]) -> tuple[bool, bool]:
    return float(row.get("zerodha_current") or 0) > 0, float(row.get("groww_current") or 0) > 0


def _pack(
    *,
    z_cur: float,
    z_inv: float,
    z_n: int,
    g_cur: float,
    g_inv: float,
    g_n: int,
) -> dict[str, float | int]:
    total_current = z_cur + g_cur
    total_invested = z_inv + g_inv
    return {
        "total_invested": total_invested,
        "total_current": total_current,
        "pl_amount": total_current - total_invested,
        "zerodha_current": z_cur,
        "groww_current": g_cur,
        "zerodha_invested": z_inv,
        "groww_invested": g_inv,
        "zerodha_holdings": int(z_n),
        "groww_holdings": int(g_n),
        "holding_count": int(z_n) + int(g_n),
    }


def should_autosave_today(*, force: bool = False) -> tuple[bool, str]:
    """Decide whether Refresh may write/update today's history row."""
    if force:
        return True, "forced"
    if get_snapshot(today_ist()) is not None:
        return True, "update-existing"
    if get_previous_snapshot(today_ist()) is None:
        return True, "bootstrap-first-day"
    if is_after_market_close():
        return True, "after-close"
    return False, "before-close-skip-today"


def apply_groww_to_as_of_day(
    *,
    as_of: date,
    groww_current: float,
    groww_invested: float,
    groww_holdings: int,
    live_zerodha: Mapping[str, Any] | None = None,
) -> str | None:
    """
    Write Groww onto as-of day.

    - If the day exists: keep its Zerodha sleeve, replace Groww.
    - If the day is missing: create it using live Zerodha as carry-back (needed after
      Cloud /tmp history wipes). Refuses Groww-only half-rows with no Zerodha.
    """
    existing = get_snapshot(as_of)
    z_cur, z_inv, z_n = _sleeve(existing, "zerodha")
    if z_cur <= 0 and live_zerodha:
        z_cur = float(live_zerodha.get("zerodha_current") or 0)
        z_inv = float(live_zerodha.get("zerodha_invested") or 0)
        z_n = int(live_zerodha.get("zerodha_holdings") or 0)

    if z_cur <= 0:
        return None

    packed = _pack(
        z_cur=z_cur,
        z_inv=z_inv,
        z_n=z_n,
        g_cur=float(groww_current),
        g_inv=float(groww_invested),
        g_n=int(groww_holdings),
    )
    return save_snapshot(**packed, snapshot_date=as_of)


def compose_today_summary(
    live: Mapping[str, float | int],
    *,
    had_zerodha: bool,
    had_groww: bool,
    groww_as_of: date | None = None,
) -> dict[str, Any]:
    """Compose sleeve-aware totals for *today* (no past-day writes)."""
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

    if had_zerodha and live_z_cur > 0:
        z_cur, z_inv, z_n = live_z_cur, live_z_inv, live_z_n
        z_note = "live"
    else:
        z_cur, z_inv, z_n = _sleeve(existing_today, "zerodha")
        if z_cur <= 0:
            z_cur, z_inv, z_n = _sleeve(previous, "zerodha")
        z_note = "carried" if z_cur > 0 else "missing"

    if had_groww and live_g_cur > 0 and groww_as_of == today:
        g_cur, g_inv, g_n = live_g_cur, live_g_inv, live_g_n
        g_note = "file as-of today"
    elif had_groww and live_g_cur > 0 and groww_as_of < today:
        # File belongs to a past day — carry that sleeve into today only when
        # that past day exists (or was just patched). Otherwise keep prior carry.
        if get_snapshot(groww_as_of) is not None:
            g_cur, g_inv, g_n = live_g_cur, live_g_inv, live_g_n
            g_note = f"carried from file as-of {groww_as_of.isoformat()}"
        else:
            g_cur, g_inv, g_n = _sleeve(existing_today, "groww")
            if g_cur <= 0:
                g_cur, g_inv, g_n = _sleeve(previous, "groww")
            g_note = (
                f"file held back (as-of {groww_as_of.isoformat()} not in history)"
            )
    else:
        g_cur, g_inv, g_n = _sleeve(existing_today, "groww")
        if g_cur <= 0:
            g_cur, g_inv, g_n = _sleeve(previous, "groww")
        g_note = "carried" if g_cur > 0 else "missing"

    packed = _pack(
        z_cur=z_cur, z_inv=z_inv, z_n=z_n, g_cur=g_cur, g_inv=g_inv, g_n=g_n
    )
    packed["_meta"] = {
        "zerodha": z_note,
        "groww": g_note,
        "groww_as_of": groww_as_of.isoformat(),
    }
    return packed


# Back-compat alias used by older call sites / tests.
def compose_dual_source_summary(
    live: Mapping[str, float | int],
    *,
    had_zerodha: bool,
    had_groww: bool,
    groww_as_of: date | None = None,
) -> dict[str, Any]:
    composed = compose_today_summary(
        live,
        had_zerodha=had_zerodha,
        had_groww=had_groww,
        groww_as_of=groww_as_of,
    )
    # Older callers expected groww_patch in meta; compute without applying.
    today = today_ist()
    groww_as_of = groww_as_of or yesterday_ist()
    meta = dict(composed.get("_meta") or {})
    if (
        had_groww
        and float(live.get("groww_current") or 0) > 0
        and groww_as_of < today
        and get_snapshot(groww_as_of) is not None
    ):
        meta["groww_patch"] = {
            "date": groww_as_of,
            "groww_current": float(live.get("groww_current") or 0),
            "groww_invested": float(live.get("groww_invested") or 0),
            "groww_holdings": int(live.get("groww_holdings") or 0),
        }
    else:
        meta["groww_patch"] = None
    composed["_meta"] = meta
    return composed


def evaluate_snapshot_quality(
    summary: Mapping[str, float | int],
    *,
    existing: Mapping[str, Any] | None = None,
    previous: Mapping[str, Any] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    candidate = {
        "total_invested": float(summary.get("total_invested") or 0),
        "total_current": float(summary.get("total_current") or 0),
        "pl_amount": float(summary.get("pl_amount") or 0),
        "zerodha_current": float(summary.get("zerodha_current") or 0),
        "groww_current": float(summary.get("groww_current") or 0),
        "zerodha_invested": float(summary.get("zerodha_invested") or 0),
        "groww_invested": float(summary.get("groww_invested") or 0),
        "zerodha_holdings": int(summary.get("zerodha_holdings") or 0),
        "groww_holdings": int(summary.get("groww_holdings") or 0),
        "holding_count": int(summary.get("holding_count") or 0),
    }
    phase = "closing" if is_after_market_close() else "intraday"

    if force:
        return {
            "ok": True,
            "reason": f"Forced save ({phase}).",
            "phase": phase,
            "candidate": candidate,
            "provisional": False,
        }

    if candidate["total_current"] <= 0:
        return {
            "ok": False,
            "reason": "Composed current value is zero — not saved.",
            "phase": phase,
            "candidate": candidate,
        }

    cand_z, cand_g = _broker_flags(candidate)
    if existing:
        ex_z, ex_g = _broker_flags(existing)
        if ex_z and not cand_z:
            return {
                "ok": False,
                "reason": "Would drop Zerodha sleeve — kept earlier snapshot.",
                "phase": phase,
                "candidate": candidate,
            }
        if ex_g and not cand_g:
            return {
                "ok": False,
                "reason": "Would drop Groww sleeve — kept earlier snapshot.",
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
                        f"Composed value is only {ratio:.0%} of "
                        f"{previous.get('snapshot_date')} — likely incomplete."
                    ),
                    "phase": phase,
                    "candidate": candidate,
                }
            if ratio > MAX_VALUE_RATIO_VS_PRIOR:
                return {
                    "ok": False,
                    "reason": (
                        f"Composed value is {ratio:.0%} of "
                        f"{previous.get('snapshot_date')} — looks abnormal."
                    ),
                    "phase": phase,
                    "candidate": candidate,
                }

    label = "closing" if phase == "closing" else "intraday"
    action = "Updated" if existing else "Saved"
    return {
        "ok": True,
        "reason": f"{action} {label} snapshot.",
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
    zerodha_holdings: int = 0,
    groww_holdings: int = 0,
    holding_count: int = 0,
    snapshot_date: date | None = None,
) -> str:
    init_db()
    snapshot_date = snapshot_date or today_ist()
    created_at = now_ist().isoformat(timespec="seconds")
    if holding_count <= 0:
        holding_count = int(zerodha_holdings) + int(groww_holdings)

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO snapshots (
                snapshot_date, total_invested, total_current, pl_amount,
                zerodha_current, groww_current, zerodha_invested, groww_invested,
                zerodha_holdings, groww_holdings, holding_count, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(snapshot_date) DO UPDATE SET
                total_invested = excluded.total_invested,
                total_current = excluded.total_current,
                pl_amount = excluded.pl_amount,
                zerodha_current = excluded.zerodha_current,
                groww_current = excluded.groww_current,
                zerodha_invested = excluded.zerodha_invested,
                groww_invested = excluded.groww_invested,
                zerodha_holdings = excluded.zerodha_holdings,
                groww_holdings = excluded.groww_holdings,
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
                int(zerodha_holdings),
                int(groww_holdings),
                int(holding_count),
                created_at,
            ),
        )

    return snapshot_date.isoformat()


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
    1) Optionally finalise Groww on an existing as-of day (T+1).
    2) Compose today's sleeves.
    3) Save today only when allowed (EOD / existing / bootstrap / force).
    """
    day = snapshot_date or today_ist()
    groww_as_of = groww_as_of or (yesterday_ist() if had_groww else today_ist())
    notes: list[str] = []
    patched_day = None

    live_g_cur = float(summary.get("groww_current") or 0)
    if had_groww and live_g_cur > 0 and groww_as_of < day:
        live_z = {
            "zerodha_current": summary.get("zerodha_current"),
            "zerodha_invested": summary.get("zerodha_invested"),
            "zerodha_holdings": summary.get("zerodha_holdings"),
        }
        existed = get_snapshot(groww_as_of) is not None
        patched_day = apply_groww_to_as_of_day(
            as_of=groww_as_of,
            groww_current=live_g_cur,
            groww_invested=float(summary.get("groww_invested") or 0),
            groww_holdings=int(summary.get("groww_holdings") or 0),
            live_zerodha=live_z if had_zerodha else None,
        )
        if patched_day:
            action = "updated" if existed else "created"
            notes.append(
                f"Groww {action} on {patched_day} "
                f"(Zerodha sleeve kept/carried; T+1 as-of)."
            )
        else:
            notes.append(
                f"Groww as-of {groww_as_of.isoformat()} not written — need Zerodha "
                "connected so that day can be created/updated with both sleeves."
            )

    composed = compose_today_summary(
        summary,
        had_zerodha=had_zerodha,
        had_groww=had_groww,
        groww_as_of=groww_as_of,
    )
    meta = dict(composed.pop("_meta", {}))

    allow, allow_reason = should_autosave_today(force=force)
    if day != today_ist():
        # Explicit non-today saves only via force path / internal tools.
        allow = force
        allow_reason = "explicit-day" if force else "refusing-non-today"

    if not allow:
        return {
            "saved": False,
            "date": None,
            "reason": " ".join(
                notes
                + [
                    "Today's history point not written yet "
                    f"({allow_reason}; auto-save after 3:30 PM IST, "
                    "or use Force-save)."
                ]
            ).strip(),
            "phase": "intraday",
            "provisional": True,
            "meta": meta,
            "patched_day": patched_day,
            "skipped_today": True,
        }

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
            "skipped_today": False,
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
        zerodha_holdings=int(cand["zerodha_holdings"]),
        groww_holdings=int(cand["groww_holdings"]),
        holding_count=int(cand["holding_count"]),
        snapshot_date=day,
    )
    notes.append(
        f"{verdict['reason']} Zerodha={meta.get('zerodha')}; Groww={meta.get('groww')}."
    )
    return {
        "saved": True,
        "date": saved_date,
        "reason": " ".join(notes),
        "phase": verdict["phase"],
        "provisional": bool(verdict.get("provisional")),
        "meta": meta,
        "patched_day": patched_day,
        "skipped_today": False,
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
                zerodha_holdings,
                groww_holdings,
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
    # Normalize display totals from sleeves when possible.
    for idx in df.index:
        row = df.loc[idx].to_dict()
        norm = _normalize_row(row)
        for key in (
            "total_invested",
            "total_current",
            "pl_amount",
            "zerodha_holdings",
            "groww_holdings",
            "holding_count",
            "zerodha_invested",
            "groww_invested",
        ):
            df.at[idx, key] = norm[key]
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


def delete_snapshot(snapshot_date: date) -> bool:
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            "DELETE FROM snapshots WHERE snapshot_date = ?",
            (snapshot_date.isoformat(),),
        )
        conn.commit()
        return cur.rowcount > 0


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
            "caption": (
                "No daily history yet — after 3:30 PM IST, Refresh saves day 1 "
                "(or Force-save anytime)."
            ),
            "after_close": after_close,
        }

    dates = {ts.date() for ts in df["date"]}
    last_date = max(dates)
    today_saved = today in dates
    window_start = today - timedelta(days=lookback_days - 1)
    expected = {
        d
        for d in (window_start + timedelta(days=i) for i in range(lookback_days))
        if d <= today
    }
    missing = len(expected - dates)

    if today_saved:
        caption = f"Today saved · {len(dates)} day(s) in history"
    elif after_close:
        caption = f"Today not saved yet · last {last_date.strftime('%d %b')}"
    else:
        caption = (
            f"Before close — today's point waits until 3:30 PM IST "
            f"· last {last_date.strftime('%d %b')}"
        )
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
