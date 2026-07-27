"""Day P&L from yesterday's history snapshot + today's Zerodha activity.

- Day book move: live total_current − prior snapshot total_current
  (book-value change; excludes cash and Groww lag).
- Today booked (Zerodha): derived from positions day activity + sell trades.
  Console CSV remains the source for cumulative historical booked P&L.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")


def empty_day_pnl() -> dict[str, Any]:
    return {
        "available": False,
        "day_book_move": None,
        "prior_date": None,
        "prior_total": None,
        "live_total": None,
        "today_booked": 0.0,
        "today_unrealised_day": 0.0,
        "trade_count": 0,
        "sell_count": 0,
        "trades": [],
        "positions": [],
        "source": None,
        "note": None,
        "fetched_at": None,
        "error": None,
    }


def day_book_move(
    live_summary: Mapping[str, Any] | None,
    prior_snapshot: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Live book value vs prior history row."""
    live_total = float((live_summary or {}).get("total_current") or 0)
    if prior_snapshot is None:
        return {
            "day_book_move": None,
            "prior_date": None,
            "prior_total": None,
            "live_total": live_total if live_total else None,
            "note": "Need a prior history snapshot (yesterday) for day move.",
        }
    prior_total = float(prior_snapshot.get("total_current") or 0)
    prior_date = prior_snapshot.get("snapshot_date") or prior_snapshot.get("date")
    if hasattr(prior_date, "isoformat"):
        prior_date = prior_date.isoformat()
    elif prior_date is not None:
        prior_date = str(prior_date)[:10]
    return {
        "day_book_move": live_total - prior_total,
        "prior_date": prior_date,
        "prior_total": prior_total,
        "live_total": live_total,
        "note": (
            "Book-value change vs prior snapshot "
            "(excludes cash; Groww may lag T+1)."
        ),
    }


def _f(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def position_realised(pos: Mapping[str, Any]) -> float:
    """Best-effort realised for one position row.

    Kite's `realised` field is legacy/unreliable; fall back to closed-leg math.
    """
    api_r = _f(pos.get("realised"))
    if api_r != 0:
        return api_r

    qty = int(_f(pos.get("quantity")))
    buy_value = _f(pos.get("buy_value"))
    sell_value = _f(pos.get("sell_value"))
    mult = _f(pos.get("multiplier"), 1.0) or 1.0

    if qty == 0 and (buy_value or sell_value):
        return sell_value - buy_value

    sell_qty = _f(pos.get("sell_quantity"))
    buy_price = _f(pos.get("buy_price"))
    sell_price = _f(pos.get("sell_price"))
    if sell_qty and buy_price and sell_price:
        return (sell_price - buy_price) * sell_qty * mult

    # Partial close: total pnl minus MTM on remaining
    last = _f(pos.get("last_price"))
    avg = _f(pos.get("average_price"))
    if sell_qty and qty != 0 and last and avg:
        total_pnl = (sell_value - buy_value) + qty * last * mult
        unrealised = qty * (last - avg) * mult
        return total_pnl - unrealised

    return 0.0


def summarize_day_positions(positions_payload: Mapping[str, Any] | None) -> dict[str, Any]:
    """Aggregate today's booked from the `day` positions list."""
    day_rows = list((positions_payload or {}).get("day") or [])
    positions_out: list[dict[str, Any]] = []
    booked = 0.0
    unrealised_day = 0.0

    for pos in day_rows:
        sell_qty = _f(pos.get("sell_quantity"))
        buy_qty = _f(pos.get("buy_quantity"))
        if sell_qty <= 0 and buy_qty <= 0:
            continue
        realised = position_realised(pos)
        unrealised = _f(pos.get("unrealised"))
        if unrealised == 0 and _f(pos.get("quantity")) != 0:
            last = _f(pos.get("last_price"))
            avg = _f(pos.get("average_price"))
            mult = _f(pos.get("multiplier"), 1.0) or 1.0
            qty = _f(pos.get("quantity"))
            if last and avg:
                unrealised = qty * (last - avg) * mult
        if sell_qty > 0 or realised != 0:
            booked += realised
        unrealised_day += unrealised
        positions_out.append(
            {
                "symbol": pos.get("tradingsymbol") or "",
                "product": pos.get("product") or "",
                "quantity": int(_f(pos.get("quantity"))),
                "buy_quantity": int(buy_qty),
                "sell_quantity": int(sell_qty),
                "buy_value": _f(pos.get("buy_value")),
                "sell_value": _f(pos.get("sell_value")),
                "realised": realised,
                "unrealised": unrealised,
                "pnl": _f(pos.get("pnl")),
            }
        )

    return {
        "today_booked": booked,
        "today_unrealised_day": unrealised_day,
        "positions": positions_out,
    }


def trades_to_rows(
    trades: list[Mapping[str, Any]] | None,
    holdings_avg: Mapping[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Flatten today's trades; estimate sell P&L when holding avg is known."""
    holdings_avg = holdings_avg or {}
    rows: list[dict[str, Any]] = []
    for trade in trades or []:
        side = str(trade.get("transaction_type") or "").upper()
        symbol = str(trade.get("tradingsymbol") or "")
        qty = int(_f(trade.get("quantity")))
        price = _f(trade.get("average_price"))
        value = qty * price
        est_pnl = None
        if side == "SELL" and symbol in holdings_avg and holdings_avg[symbol] > 0:
            est_pnl = (price - holdings_avg[symbol]) * qty
        rows.append(
            {
                "symbol": symbol,
                "side": side,
                "quantity": qty,
                "price": price,
                "value": value,
                "product": trade.get("product") or "",
                "order_id": str(trade.get("order_id") or ""),
                "exchange": trade.get("exchange") or "",
                "estimated_pnl": est_pnl,
            }
        )
    return rows


def build_day_pnl(
    *,
    live_summary: Mapping[str, Any] | None,
    prior_snapshot: Mapping[str, Any] | None,
    positions_payload: Mapping[str, Any] | None = None,
    trades: list[Mapping[str, Any]] | None = None,
    holdings_avg: Mapping[str, float] | None = None,
    source: str = "zerodha",
) -> dict[str, Any]:
    """Compose the session day-P&L payload (no network)."""
    out = empty_day_pnl()
    move = day_book_move(live_summary, prior_snapshot)
    out.update(move)

    pos_summary = summarize_day_positions(positions_payload)
    trade_rows = trades_to_rows(trades, holdings_avg)
    sells = [r for r in trade_rows if r["side"] == "SELL"]

    # If positions realised is ~0 but we can estimate from sells + holding avg, use that.
    booked = float(pos_summary["today_booked"])
    if abs(booked) < 1e-9:
        est = [r["estimated_pnl"] for r in sells if r.get("estimated_pnl") is not None]
        if est:
            booked = float(sum(est))

    out["available"] = True
    out["today_booked"] = booked
    out["today_unrealised_day"] = float(pos_summary["today_unrealised_day"])
    out["positions"] = pos_summary["positions"]
    out["trades"] = trade_rows
    out["trade_count"] = len(trade_rows)
    out["sell_count"] = len(sells)
    out["source"] = source
    out["fetched_at"] = datetime.now(IST).isoformat(timespec="seconds")
    if out.get("note") is None:
        out["note"] = "Today booked from Zerodha day positions / trades."
    return out


def combine_day_economic(
    unrealized: float,
    *,
    imported_realized: float = 0.0,
    today_booked: float = 0.0,
    has_imported: bool = False,
) -> dict[str, float | bool]:
    """
    Economic P&L without double-counting Console CSV and live today booked.

    If an imported realised file is loaded, economic = unrealised + imported only
    (today live shown separately). Otherwise economic = unrealised + today booked.
    """
    unrealized = float(unrealized)
    imported = float(imported_realized or 0)
    today = float(today_booked or 0)
    if has_imported:
        return {
            "unrealized": unrealized,
            "imported_realized": imported,
            "today_booked": today,
            "include_today_in_economic": False,
            "economic": unrealized + imported,
        }
    return {
        "unrealized": unrealized,
        "imported_realized": 0.0,
        "today_booked": today,
        "include_today_in_economic": True,
        "economic": unrealized + today,
    }
