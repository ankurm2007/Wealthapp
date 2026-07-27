"""Booked (realised) P&L from broker P&L / trade reports.

Open holdings only show unrealised mark-to-market. After you sell at a loss,
that loss disappears from holdings — so reports look artificially green unless
we ingest Console / Groww P&L exports.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, BinaryIO, Mapping
from zoneinfo import ZoneInfo

import pandas as pd

IST = ZoneInfo("Asia/Kolkata")
STORE_PATH = Path(__file__).parent / "data" / "realized_pnl.json"

# Column name aliases → canonical fields
_SYMBOL_ALIASES = ("symbol", "tradingsymbol", "trading symbol", "stock name", "scrip", "ticker", "company")
_QTY_ALIASES = ("quantity", "qty", "qty.", "shares")
_REALIZED_ALIASES = (
    "realized p&l",
    "realised p&l",
    "realized pnl",
    "realised pnl",
    "realized profit",
    "realised profit",
    "net realised p&l",
    "net realized p&l",
    "realized",
    "realised",
)
_BUY_VAL_ALIASES = ("buy value", "buy_value", "buyvalue", "purchase value")
_SELL_VAL_ALIASES = ("sell value", "sell_value", "sellvalue")
_BUY_DATE_ALIASES = ("buy date", "buy_date", "purchase date")
_SELL_DATE_ALIASES = ("sell date", "sell_date", "exit date")
_ISIN_ALIASES = ("isin",)


def _norm_col(name: object) -> str:
    text = str(name or "").strip().lower()
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text)
    return text


def _find_col(columns: list[str], aliases: tuple[str, ...]) -> str | None:
    normalized = {_norm_col(c): c for c in columns}
    for alias in aliases:
        if alias in normalized:
            return normalized[alias]
    # soft contains match
    for alias in aliases:
        for norm, original in normalized.items():
            if alias in norm:
                return original
    return None


def _to_number(value: object) -> float:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text or text in {"-", "—", "NA", "N/A", "null"}:
        return 0.0
    neg = text.startswith("(") and text.endswith(")")
    text = text.replace(",", "").replace("₹", "").replace("Rs.", "").replace("Rs", "")
    text = text.replace("(", "").replace(")", "").strip()
    try:
        number = float(text)
    except ValueError:
        return 0.0
    return -abs(number) if neg else number


def _read_table(uploaded: BinaryIO | bytes, filename: str = "") -> pd.DataFrame:
    raw = uploaded.read() if hasattr(uploaded, "read") else uploaded
    name = (filename or getattr(uploaded, "name", "") or "").lower()
    bio = BytesIO(raw if isinstance(raw, (bytes, bytearray)) else bytes(raw))

    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(bio)

    # Zerodha / Groww CSVs sometimes have a few preamble rows.
    bio.seek(0)
    try:
        preview = pd.read_csv(bio, header=None, nrows=15, dtype=str)
    except Exception:
        bio.seek(0)
        return pd.read_csv(bio)

    header_row = 0
    for idx, row in preview.iterrows():
        joined = " ".join(str(x).lower() for x in row.tolist() if pd.notna(x))
        if any(key in joined for key in ("symbol", "isin", "realis", "stock name", "quantity")):
            header_row = int(idx)
            break
    bio.seek(0)
    return pd.read_csv(bio, header=header_row)


def parse_realized_report(
    uploaded: BinaryIO | bytes,
    *,
    filename: str = "",
    source: str = "Broker P&L",
) -> dict[str, Any]:
    """
    Parse Zerodha Console / Groww P&L exports into booked (realised) rows.

    Returns summary + line items. Raises ValueError if no realised column found.
    """
    df = _read_table(uploaded, filename=filename)
    if df.empty:
        raise ValueError("The P&L file is empty.")

    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    symbol_col = _find_col(list(df.columns), _SYMBOL_ALIASES)
    realized_col = _find_col(list(df.columns), _REALIZED_ALIASES)
    qty_col = _find_col(list(df.columns), _QTY_ALIASES)
    buy_val_col = _find_col(list(df.columns), _BUY_VAL_ALIASES)
    sell_val_col = _find_col(list(df.columns), _SELL_VAL_ALIASES)
    buy_date_col = _find_col(list(df.columns), _BUY_DATE_ALIASES)
    sell_date_col = _find_col(list(df.columns), _SELL_DATE_ALIASES)
    isin_col = _find_col(list(df.columns), _ISIN_ALIASES)

    if realized_col is None and buy_val_col and sell_val_col:
        df["__realized__"] = df[sell_val_col].map(_to_number) - df[buy_val_col].map(_to_number)
        realized_col = "__realized__"

    if realized_col is None:
        raise ValueError(
            "Could not find a Realised P&L column. Download "
            "Console → Reports → P&L (Realised) or Groww Tax/P&L export."
        )

    rows: list[dict[str, Any]] = []
    for _, raw in df.iterrows():
        pnl = _to_number(raw.get(realized_col))
        symbol = str(raw.get(symbol_col) or "").strip() if symbol_col else ""
        if not symbol and isin_col:
            symbol = str(raw.get(isin_col) or "").strip()
        # Skip blank / total / header junk
        lower = symbol.lower()
        if not symbol or lower in {"symbol", "total", "grand total", "nan"}:
            # Still keep anonymous realised lines if P&L is non-zero (rare totals).
            if abs(pnl) < 1e-9:
                continue
            symbol = symbol or "UNKNOWN"

        # Prefer closed trades: non-zero realised, or sell value present.
        sell_val = _to_number(raw.get(sell_val_col)) if sell_val_col else None
        if abs(pnl) < 1e-9 and (sell_val is None or abs(sell_val) < 1e-9):
            continue

        rows.append(
            {
                "symbol": symbol.upper(),
                "isin": str(raw.get(isin_col) or "").strip() if isin_col else "",
                "quantity": _to_number(raw.get(qty_col)) if qty_col else 0.0,
                "buy_value": _to_number(raw.get(buy_val_col)) if buy_val_col else 0.0,
                "sell_value": _to_number(raw.get(sell_val_col)) if sell_val_col else 0.0,
                "realized_pnl": pnl,
                "buy_date": str(raw.get(buy_date_col) or "").strip() if buy_date_col else "",
                "sell_date": str(raw.get(sell_date_col) or "").strip() if sell_date_col else "",
            }
        )

    if not rows:
        raise ValueError(
            "No realised (booked) P&L rows found. In Console choose "
            "P&L → Realised (not Unrealised) for the date range of your sells."
        )

    items = pd.DataFrame(rows)
    realized_total = float(items["realized_pnl"].sum())
    booked_losses = float(items.loc[items["realized_pnl"] < 0, "realized_pnl"].sum())
    booked_gains = float(items.loc[items["realized_pnl"] > 0, "realized_pnl"].sum())
    loss_rows = items[items["realized_pnl"] < 0].sort_values("realized_pnl")
    gain_rows = items[items["realized_pnl"] > 0].sort_values("realized_pnl", ascending=False)

    payload = {
        "source": source,
        "filename": filename or getattr(uploaded, "name", "") or "",
        "imported_at": datetime.now(IST).isoformat(timespec="seconds"),
        "row_count": int(len(items)),
        "realized_total": realized_total,
        "booked_gains": booked_gains,
        "booked_losses": booked_losses,  # negative or zero
        "loss_count": int((items["realized_pnl"] < 0).sum()),
        "gain_count": int((items["realized_pnl"] > 0).sum()),
        "items": items.to_dict(orient="records"),
        "top_losses": loss_rows.head(10).to_dict(orient="records"),
        "top_gains": gain_rows.head(10).to_dict(orient="records"),
    }
    return payload


def empty_realized_state() -> dict[str, Any]:
    return {
        "source": "",
        "filename": "",
        "imported_at": "",
        "row_count": 0,
        "realized_total": 0.0,
        "booked_gains": 0.0,
        "booked_losses": 0.0,
        "loss_count": 0,
        "gain_count": 0,
        "items": [],
        "top_losses": [],
        "top_gains": [],
        "manual_note": "",
    }


def save_realized_state(state: Mapping[str, Any]) -> None:
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STORE_PATH.write_text(json.dumps(dict(state), indent=2), encoding="utf-8")


def load_realized_state() -> dict[str, Any]:
    if not STORE_PATH.exists():
        return empty_realized_state()
    try:
        data = json.loads(STORE_PATH.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return empty_realized_state()
        base = empty_realized_state()
        base.update(data)
        return base
    except Exception:
        return empty_realized_state()


def clear_realized_state() -> None:
    if STORE_PATH.exists():
        STORE_PATH.unlink()


def combine_economic_pnl(unrealized: float, realized_state: dict[str, Any] | None) -> dict[str, float]:
    """Open-book unrealised + booked realised = fuller economic P&L."""
    realized = float((realized_state or {}).get("realized_total") or 0)
    booked_losses = float((realized_state or {}).get("booked_losses") or 0)
    booked_gains = float((realized_state or {}).get("booked_gains") or 0)
    return {
        "unrealized": float(unrealized),
        "realized": realized,
        "booked_losses": booked_losses,
        "booked_gains": booked_gains,
        "economic": float(unrealized) + realized,
    }
