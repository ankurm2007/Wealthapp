"""Unit tests for day P&L helpers."""

from __future__ import annotations

import unittest

import day_pnl as dp


class DayPnlTests(unittest.TestCase):
    def test_day_book_move_needs_prior(self):
        out = dp.day_book_move({"total_current": 100.0}, None)
        self.assertIsNone(out["day_book_move"])
        self.assertIn("prior", (out.get("note") or "").lower())

    def test_day_book_move_subtracts_prior(self):
        out = dp.day_book_move(
            {"total_current": 1_100_000},
            {"total_current": 1_000_000, "snapshot_date": "2026-07-27"},
        )
        self.assertAlmostEqual(out["day_book_move"], 100_000)
        self.assertEqual(out["prior_date"], "2026-07-27")
        self.assertAlmostEqual(out["prior_total"], 1_000_000)

    def test_closed_position_realised_from_buy_sell_value(self):
        pos = {
            "tradingsymbol": "ABC",
            "quantity": 0,
            "buy_quantity": 10,
            "sell_quantity": 10,
            "buy_value": 1000,
            "sell_value": 900,
            "realised": 0,
            "unrealised": 0,
            "pnl": -100,
            "multiplier": 1,
        }
        self.assertAlmostEqual(dp.position_realised(pos), -100)

    def test_summarize_day_positions_sums_booked(self):
        payload = {
            "day": [
                {
                    "tradingsymbol": "AAA",
                    "product": "CNC",
                    "quantity": 0,
                    "buy_quantity": 5,
                    "sell_quantity": 5,
                    "buy_value": 500,
                    "sell_value": 450,
                    "realised": 0,
                    "unrealised": 0,
                    "pnl": -50,
                    "multiplier": 1,
                },
                {
                    "tradingsymbol": "BBB",
                    "product": "MIS",
                    "quantity": 0,
                    "buy_quantity": 2,
                    "sell_quantity": 2,
                    "buy_value": 200,
                    "sell_value": 260,
                    "realised": 0,
                    "unrealised": 0,
                    "pnl": 60,
                    "multiplier": 1,
                },
            ]
        }
        summary = dp.summarize_day_positions(payload)
        self.assertAlmostEqual(summary["today_booked"], 10)
        self.assertEqual(len(summary["positions"]), 2)

    def test_trades_estimated_pnl_from_holding_avg(self):
        trades = [
            {
                "tradingsymbol": "XYZ",
                "transaction_type": "SELL",
                "quantity": 10,
                "average_price": 90,
                "product": "CNC",
                "order_id": "1",
                "exchange": "NSE",
            }
        ]
        rows = dp.trades_to_rows(trades, {"XYZ": 100.0})
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["estimated_pnl"], -100.0)

    def test_build_day_pnl_falls_back_to_trade_estimates(self):
        out = dp.build_day_pnl(
            live_summary={"total_current": 2_000_000},
            prior_snapshot={"total_current": 1_900_000, "snapshot_date": "2026-07-27"},
            positions_payload={"day": []},
            trades=[
                {
                    "tradingsymbol": "XYZ",
                    "transaction_type": "SELL",
                    "quantity": 5,
                    "average_price": 80,
                    "product": "CNC",
                }
            ],
            holdings_avg={"XYZ": 100.0},
        )
        self.assertTrue(out["available"])
        self.assertAlmostEqual(out["day_book_move"], 100_000)
        self.assertAlmostEqual(out["today_booked"], -100.0)
        self.assertEqual(out["sell_count"], 1)

    def test_combine_avoids_double_count_when_imported(self):
        combo = dp.combine_day_economic(
            -50_000,
            imported_realized=-20_000,
            today_booked=-5_000,
            has_imported=True,
        )
        self.assertFalse(combo["include_today_in_economic"])
        self.assertAlmostEqual(combo["economic"], -70_000)

    def test_combine_uses_today_when_no_import(self):
        combo = dp.combine_day_economic(
            -50_000,
            imported_realized=0,
            today_booked=-5_000,
            has_imported=False,
        )
        self.assertTrue(combo["include_today_in_economic"])
        self.assertAlmostEqual(combo["economic"], -55_000)


if __name__ == "__main__":
    unittest.main()
