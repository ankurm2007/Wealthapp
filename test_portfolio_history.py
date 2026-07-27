"""Regression tests for sleeve-aware portfolio history."""

from __future__ import annotations

import tempfile
import unittest
from datetime import date, datetime, time
from pathlib import Path
from unittest import mock
from zoneinfo import ZoneInfo

import portfolio_history as ph

IST = ZoneInfo("Asia/Kolkata")


class HistoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "t.db"
        self._db_patch = mock.patch.object(ph, "DB_PATH", self.db)
        self._db_patch.start()
        # Freeze "today" to 28 Jul 2026 afternoon after close by default.
        self.today = date(2026, 7, 28)
        self.yesterday = date(2026, 7, 27)
        self._time_patch = mock.patch.object(ph, "today_ist", return_value=self.today)
        self._time_patch.start()
        self._now = mock.patch.object(
            ph,
            "now_ist",
            return_value=datetime(2026, 7, 28, 16, 0, tzinfo=IST),
        )
        self._now.start()
        ph.init_db()

    def tearDown(self) -> None:
        self._now.stop()
        self._time_patch.stop()
        self._db_patch.stop()
        self.tmp.cleanup()

    def _live(self, **over):
        base = {
            "zerodha_current": 1_750_000.0,
            "zerodha_invested": 1_860_000.0,
            "zerodha_holdings": 12,
            "groww_current": 6_300_000.0,
            "groww_invested": 6_000_000.0,
            "groww_holdings": 17,
            "holding_count": 29,
        }
        base.update(over)
        base["total_current"] = base["zerodha_current"] + base["groww_current"]
        base["total_invested"] = base["zerodha_invested"] + base["groww_invested"]
        base["pl_amount"] = base["total_current"] - base["total_invested"]
        return base

    def test_never_invents_missing_as_of_day(self):
        # Only today bootstrap with Zerodha — Groww as-of yesterday missing.
        r = ph.save_summary_snapshot(
            self._live(groww_current=0, groww_invested=0, groww_holdings=0),
            had_zerodha=True,
            had_groww=False,
            force=True,
        )
        self.assertTrue(r["saved"])
        self.assertIsNone(ph.get_snapshot(self.yesterday))

        r2 = ph.save_summary_snapshot(
            self._live(),
            had_zerodha=True,
            had_groww=True,
            groww_as_of=self.yesterday,
            force=True,
        )
        self.assertIsNone(r2.get("patched_day"))
        self.assertIsNone(ph.get_snapshot(self.yesterday))
        today = ph.get_snapshot(self.today)
        # Groww held back because yesterday not in history.
        self.assertEqual(today["groww_current"], 0.0)

    def test_groww_finalises_existing_yesterday_without_touching_zerodha(self):
        ph.save_snapshot(
            total_invested=1_860_000,
            total_current=1_750_000,
            pl_amount=-110_000,
            zerodha_current=1_750_000,
            groww_current=0,
            zerodha_invested=1_860_000,
            groww_invested=0,
            zerodha_holdings=12,
            groww_holdings=0,
            snapshot_date=self.yesterday,
        )
        r = ph.save_summary_snapshot(
            self._live(),
            had_zerodha=True,
            had_groww=True,
            groww_as_of=self.yesterday,
            force=True,
        )
        self.assertEqual(r["patched_day"], self.yesterday.isoformat())
        y = ph.get_snapshot(self.yesterday)
        self.assertAlmostEqual(y["zerodha_current"], 1_750_000)
        self.assertAlmostEqual(y["groww_current"], 6_300_000)
        self.assertEqual(y["zerodha_holdings"], 12)
        self.assertEqual(y["groww_holdings"], 17)
        self.assertEqual(y["holding_count"], 29)
        t = ph.get_snapshot(self.today)
        self.assertAlmostEqual(t["groww_current"], 6_300_000)
        self.assertEqual(t["holding_count"], 29)

    def test_before_close_skips_today_but_can_patch_yesterday(self):
        ph.save_snapshot(
            total_invested=1_860_000,
            total_current=1_750_000,
            pl_amount=-110_000,
            zerodha_current=1_750_000,
            groww_current=0,
            zerodha_invested=1_860_000,
            groww_invested=0,
            zerodha_holdings=12,
            groww_holdings=0,
            snapshot_date=self.yesterday,
        )
        with mock.patch.object(
            ph,
            "now_ist",
            return_value=datetime(2026, 7, 28, 0, 45, tzinfo=IST),
        ):
            r = ph.save_summary_snapshot(
                self._live(),
                had_zerodha=True,
                had_groww=True,
                groww_as_of=self.yesterday,
                force=False,
            )
        self.assertFalse(r["saved"])
        self.assertTrue(r.get("skipped_today"))
        self.assertEqual(r["patched_day"], self.yesterday.isoformat())
        self.assertIsNone(ph.get_snapshot(self.today))
        y = ph.get_snapshot(self.yesterday)
        self.assertAlmostEqual(y["groww_current"], 6_300_000)
        self.assertEqual(y["zerodha_holdings"], 12)

    def test_no_identical_twin_from_overnight_groww_upload(self):
        ph.save_snapshot(
            total_invested=1_860_000,
            total_current=1_750_000,
            pl_amount=-110_000,
            zerodha_current=1_750_000,
            groww_current=0,
            zerodha_invested=1_860_000,
            groww_invested=0,
            zerodha_holdings=12,
            groww_holdings=0,
            snapshot_date=self.yesterday,
        )
        with mock.patch.object(
            ph,
            "now_ist",
            return_value=datetime(2026, 7, 28, 0, 45, tzinfo=IST),
        ):
            ph.save_summary_snapshot(
                self._live(),
                had_zerodha=True,
                had_groww=True,
                groww_as_of=self.yesterday,
            )
        df = ph.load_snapshots()
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["date"].date(), self.yesterday)


if __name__ == "__main__":
    unittest.main()
