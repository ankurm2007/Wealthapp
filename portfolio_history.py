"""Persist and query portfolio snapshot history."""

from __future__ import annotations

import sqlite3
from datetime import date, datetime
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).parent / "data" / "portfolio_history.db"


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
                holding_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            )
            """
        )


def save_snapshot(
    *,
    total_invested: float,
    total_current: float,
    pl_amount: float,
    zerodha_current: float = 0.0,
    groww_current: float = 0.0,
    holding_count: int = 0,
    snapshot_date: date | None = None,
) -> str:
    init_db()
    snapshot_date = snapshot_date or date.today()
    created_at = datetime.now().isoformat(timespec="seconds")

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO snapshots (
                snapshot_date, total_invested, total_current, pl_amount,
                zerodha_current, groww_current, holding_count, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(snapshot_date) DO UPDATE SET
                total_invested = excluded.total_invested,
                total_current = excluded.total_current,
                pl_amount = excluded.pl_amount,
                zerodha_current = excluded.zerodha_current,
                groww_current = excluded.groww_current,
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
                holding_count,
                created_at,
            ),
        )

    return snapshot_date.isoformat()


def load_snapshots() -> pd.DataFrame:
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(
            """
            SELECT
                snapshot_date AS date,
                total_invested,
                total_current,
                pl_amount,
                zerodha_current,
                groww_current,
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
