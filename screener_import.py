"""Parse Screener.in and Trendlyne CSV/XLSX exports for institutional data."""

from __future__ import annotations

import re

import pandas as pd
import streamlit as st

import symbol_resolver as sym

SCREENER_HINTS = ("screener", "promoter", "fii holding", "dii holding", "pledged")
TRENDLYNE_HINTS = ("trendlyne", "shareholding", "mf holding", "fii", "dii")

COLUMN_ALIASES: dict[str, list[str]] = {
    "Symbol": ["symbol", "stock", "name", "company", "scrip", "ticker"],
    "Promoter %": ["promoter holding", "promoter", "promoter %", "promoter holding %"],
    "Pledged %": ["pledged", "promoter pledged", "pledged %", "promoter shares pledged"],
    "FII %": ["fii holding", "fii", "fii %", "fii holding %", "foreign institutional"],
    "DII %": ["dii holding", "dii", "dii %", "dii holding %", "domestic institutional"],
    "Debt/Equity": ["debt to equity", "debt/equity", "d/e", "debt equity"],
    "ROE %": ["roe", "return on equity", "roe %"],
    "OPM %": ["opm", "operating profit margin", "opm %"],
}


def _norm_col(name: str) -> str:
    return re.sub(r"\s+", " ", str(name).strip().lower())


def _match_column(cols: list[str], aliases: list[str]) -> str | None:
    normalized = {_norm_col(c): c for c in cols}
    for alias in aliases:
        key = _norm_col(alias)
        if key in normalized:
            return normalized[key]
    for col in cols:
        nc = _norm_col(col)
        for alias in aliases:
            if _norm_col(alias) in nc:
                return col
    return None


def _map_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping: dict[str, str] = {}
    for target, aliases in COLUMN_ALIASES.items():
        src = _match_column(list(df.columns), aliases)
        if src:
            mapping[src] = target
    if "Symbol" not in mapping.values():
        return pd.DataFrame()
    out = df.rename(columns=mapping)
    keep = [c for c in COLUMN_ALIASES if c in out.columns]
    out = out[keep].copy()
    out["Symbol"] = out["Symbol"].astype(str).str.strip()
    out = sym.normalize_portfolio_symbols(out)
    for col in keep:
        if col == "Symbol":
            continue
        out[col] = pd.to_numeric(
            out[col].astype(str).str.replace("%", "", regex=False).str.replace(",", "", regex=False),
            errors="coerce",
        )
    return out.dropna(subset=["Symbol"]).drop_duplicates("Symbol")


def detect_export_format(df: pd.DataFrame) -> str:
    header = " ".join(_norm_col(c) for c in df.columns)
    if any(h in header for h in TRENDLYNE_HINTS):
        return "trendlyne"
    if any(h in header for h in SCREENER_HINTS):
        return "screener"
    if _match_column(list(df.columns), COLUMN_ALIASES["Promoter %"]):
        return "screener"
    return "unknown"


def parse_institutional_file(uploaded_file) -> tuple[pd.DataFrame, str]:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        raw = pd.read_csv(uploaded_file)
    else:
        raw = pd.read_excel(uploaded_file)

    fmt = detect_export_format(raw)
    parsed = _map_columns(raw)
    if parsed.empty:
        uploaded_file.seek(0)
        if name.endswith(".csv"):
            raw2 = pd.read_csv(uploaded_file)
        else:
            raw2 = pd.read_excel(uploaded_file)
        parsed = _map_columns(raw2)
        fmt = detect_export_format(raw2) if not parsed.empty else "unknown"

    return parsed, fmt


def merge_institutional_data(merged: pd.DataFrame, inst_df: pd.DataFrame) -> pd.DataFrame:
    if inst_df is None or inst_df.empty:
        return merged
    return merged.merge(inst_df, on="Symbol", how="left", suffixes=("", "_inst"))


def run_institutional_checks(merged: pd.DataFrame, inst_df: pd.DataFrame | None) -> list[dict]:
    checks: list[dict] = []
    if inst_df is None or inst_df.empty:
        return checks

    data = merge_institutional_data(merged, inst_df)

    def add(status, name, headline, detail="", table=None):
        checks.append({"status": status, "name": name, "headline": headline, "detail": detail, "table": table})

    if "Pledged %" in data.columns:
        pledged = data[(data["Weight %"] >= 3) & (data["Pledged %"].fillna(0) >= 20)]
        if not pledged.empty:
            add(
                "fail",
                "promoter_pledge",
                f"{len(pledged)} holding(s) with promoter pledge ≥ 20%.",
                "High pledge increases governance / liquidity risk.",
                pledged[["Symbol", "Weight %", "Pledged %"]].head(5),
            )
        elif (data["Pledged %"].fillna(0) >= 10).any():
            warn = data[data["Pledged %"].fillna(0) >= 10]
            add(
                "warn",
                "promoter_pledge",
                f"{len(warn)} name(s) with promoter pledge ≥ 10%.",
                "",
                warn[["Symbol", "Weight %", "Pledged %"]].head(5),
            )
        else:
            add("pass", "promoter_pledge", "Promoter pledge levels look manageable.")

    if "FII %" in data.columns:
        low_fii = data[(data["Weight %"] >= 5) & (data["FII %"].fillna(0) < 5)]
        if len(low_fii) >= 3:
            add(
                "warn",
                "fii_interest",
                f"{len(low_fii)} weighted names with FII holding under 5%.",
                "Low FII interest can mean less institutional oversight / liquidity.",
                low_fii[["Symbol", "Weight %", "FII %"]].head(5),
            )

    return checks


def build_institutional_context(inst_df: pd.DataFrame | None, merged: pd.DataFrame) -> str:
    if inst_df is None or inst_df.empty:
        return ""
    data = merge_institutional_data(merged, inst_df)
    lines = ["Institutional data (Screener/Trendlyne export):"]
    cols = [c for c in ["Promoter %", "Pledged %", "FII %", "DII %", "Debt/Equity"] if c in data.columns]
    for _, row in data.head(12).iterrows():
        if row["Weight %"] < 2:
            continue
        parts = [f"{row['Symbol']} ({row['Weight %']:.1f}% wt)"]
        for col in cols:
            val = row.get(col)
            if pd.notna(val):
                parts.append(f"{col.replace(' %', '')} {val:.1f}%")
        lines.append("- " + " · ".join(parts))
    return "\n".join(lines)
