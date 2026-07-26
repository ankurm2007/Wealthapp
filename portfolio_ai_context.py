"""Structured financial context for AI portfolio research."""

from __future__ import annotations

import pandas as pd

import portfolio_analysis as pan
import portfolio_benchmarks as pbench
import portfolio_forensics as pforensic
import portfolio_history as ph
import portfolio_market_data as pmd
import portfolio_research as presearch
import portfolio_risk as prisk
import portfolio_terminal as pterm
import screener_import as screener
import stock_analyzer as san


def _safe_pct(num: float | None, den: float | None) -> float | None:
    if num is None or den in (None, 0):
        return None
    return round((num / den - 1) * 100, 1)


def _resolve_market(
    merged: pd.DataFrame,
    summary: dict,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, dict | None]:
    cached = pmd.get_cached_market_context(merged, summary)
    if cached is not None:
        return cached
    try:
        enriched = pmd.enrich_holdings(merged)
        sectors = pmd.sector_summary(enriched, summary["total_current"])
        coverage = pmd.coverage_stats(enriched)
        return enriched, sectors, coverage
    except Exception:
        return None, None, None


def _holding_fundamentals_row(row: pd.Series) -> str:
    parts = [
        f"{row['Symbol']} ({row.get('Weight %', 0):.1f}% wt)",
        f"ret {row.get('Return %', 0):+.1f}%",
        f"P&L ₹{row.get('P&L', 0):,.0f}",
    ]
    if row.get("Sector") and row.get("Sector") != "Unknown":
        parts.append(f"sector {row['Sector']}")
    if pd.notna(row.get("P/E")):
        parts.append(f"P/E {row['P/E']:.1f}")
    if pd.notna(row.get("Forward P/E")):
        parts.append(f"fwd P/E {row['Forward P/E']:.1f}")
    if pd.notna(row.get("P/B")):
        parts.append(f"P/B {row['P/B']:.1f}")
    if pd.notna(row.get("Beta")):
        parts.append(f"β {row['Beta']:.2f}")
    if pd.notna(row.get("ROE %")):
        parts.append(f"ROE {row['ROE %']:.0f}%")
    price = row.get("Yahoo price") or row.get("Current Value")
    high = row.get("52w high")
    low = row.get("52w low")
    if price and high:
        from_high = _safe_pct(price, high)
        if from_high is not None:
            parts.append(f"{from_high:+.0f}% from 52w high")
    if price and low:
        from_low = _safe_pct(price, low)
        if from_low is not None:
            parts.append(f"{from_low:+.0f}% from 52w low")
    if row.get("52-week range"):
        parts.append(f"52w {row['52-week range']}")
    return " · ".join(parts)


def _position_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize position columns after enrich/merge (handles _x/_y suffix collisions)."""
    out = df.copy()
    for col in ("Weight %", "Return %", "P&L", "Platforms"):
        if col in out.columns:
            continue
        for suffix in ("", "_x", "_y", "_pos"):
            alt = f"{col}{suffix}" if suffix else col
            if alt in out.columns:
                out[col] = out[alt]
                break
    return out


def _enriched_holdings_block(enriched: pd.DataFrame, merged: pd.DataFrame) -> str:
    data = _position_cols(enriched if "Weight %" in enriched.columns else enriched.merge(
        merged[["Symbol", "Weight %", "Return %", "P&L", "Platforms"]],
        on="Symbol",
        how="left",
    ))
    lines = ["Holdings with fundamentals (Yahoo Finance where available):"]
    for _, row in data.iterrows():
        lines.append(f"- {_holding_fundamentals_row(row)}")
    return "\n".join(lines)


def _valuation_signals(enriched: pd.DataFrame, merged: pd.DataFrame) -> str:
    if "Weight %" in enriched.columns:
        data = _position_cols(enriched)
    else:
        data = _position_cols(
            enriched.merge(merged[["Symbol", "Weight %", "Return %"]], on="Symbol", how="left")
        )
    pe_rows = data[pd.notna(data["P/E"]) & (data["P/E"] > 0)]
    lines = ["Valuation & positioning signals:"]

    if not pe_rows.empty and pe_rows["Weight %"].sum() > 0:
        weighted_pe = (pe_rows["P/E"] * pe_rows["Weight %"]).sum() / pe_rows["Weight %"].sum()
        lines.append(f"- Weighted avg trailing P/E (where known): {weighted_pe:.1f}x")
        expensive = pe_rows.nlargest(3, "P/E")
        for _, r in expensive.iterrows():
            wt = float(r.get("Weight %", 0) or 0)
            ret = float(r.get("Return %", 0) or 0)
            lines.append(
                f"- High P/E name: {r['Symbol']} at {r['P/E']:.1f}x "
                f"({wt:.1f}% weight, {ret:+.1f}% return)"
            )

    for _, row in data.iterrows():
        price = row.get("Yahoo price")
        high = row.get("52w high")
        low = row.get("52w low")
        if not price or not high or not low:
            continue
        near_high = price >= high * 0.95
        near_low = price <= low * 1.05
        if near_high and row.get("Weight %", 0) >= 3:
            lines.append(
                f"- Near 52w high: {row['Symbol']} ({row['Weight %']:.1f}% wt) — "
                f"price ₹{price:,.0f}, high ₹{high:,.0f}"
            )
        if near_low and row.get("Weight %", 0) >= 3:
            lines.append(
                f"- Near 52w low: {row['Symbol']} ({row['Weight %']:.1f}% wt) — "
                f"price ₹{price:,.0f}, low ₹{low:,.0f}"
            )

    heavy_losers = merged[(merged["Weight %"] >= 5) & (merged["Return %"] < -10)]
    for _, row in heavy_losers.iterrows():
        lines.append(
            f"- Heavy loser: {row['Symbol']} {row['Weight %']:.1f}% wt, "
            f"{row['Return %']:+.1f}% return, P&L ₹{row['P&L']:,.0f}"
        )

    return "\n".join(lines)


def _pnl_attribution(merged: pd.DataFrame) -> str:
    total_pl = merged["P&L"].sum()
    lines = ["P&L attribution (rupee impact):"]
    for _, row in merged.nlargest(5, "P&L").iterrows():
        share = row["P&L"] / total_pl * 100 if total_pl else 0
        lines.append(
            f"- Gainer {row['Symbol']}: ₹{row['P&L']:,.0f} "
            f"({share:.0f}% of total P&L, {row['Weight %']:.1f}% wt)"
        )
    for _, row in merged.nsmallest(3, "P&L").iterrows():
        if row["P&L"] >= 0:
            continue
        lines.append(
            f"- Drag {row['Symbol']}: ₹{row['P&L']:,.0f} "
            f"({row['Weight %']:.1f}% wt, {row['Return %']:+.1f}%)"
        )
    return "\n".join(lines)


def _benchmark_block(merged: pd.DataFrame, sector_df: pd.DataFrame | None) -> str:
    lines = ["Benchmark context:"]
    try:
        bench = pbench.build_nifty_vs_portfolio(merged, days=30)
        if bench.get("portfolio_return") is not None and bench.get("nifty_return") is not None:
            lines.append(
                f"- 30d return: portfolio {bench['portfolio_return']:+.1f}% vs "
                f"Nifty 50 {bench['nifty_return']:+.1f}% "
                f"(alpha {bench.get('alpha', 0):+.1f}%, coverage ~{bench.get('coverage', 0):.0f}%)"
            )
    except Exception:
        lines.append("- Nifty comparison unavailable.")

    if sector_df is not None and not sector_df.empty:
        vs = pbench.build_sector_vs_nifty(sector_df)
        if not vs.empty:
            lines.append("- Sector active weights vs Nifty 50 (you − index):")
            for _, row in vs.head(6).iterrows():
                if row["Sector"] in ("Unknown", "Equity (uncategorized)"):
                    continue
                lines.append(
                    f"  · {row['Sector']}: you {row['Your portfolio %']:.1f}% vs "
                    f"Nifty {row['Nifty 50 %']:.1f}% → active {row['Active weight %']:+.1f}%"
                )
    return "\n".join(lines)


def _quadrant_block(merged: pd.DataFrame) -> str:
    q = presearch.quadrant_labels(merged)
    lines = ["Position map (weight × return):"]
    for label, symbols in q.items():
        if symbols:
            lines.append(f"- {label}: {', '.join(symbols)}")
        else:
            lines.append(f"- {label}: none")
    return "\n".join(lines)


def _checks_block(checks: list[dict]) -> str:
    if not checks:
        return ""
    lines = ["Automated risk checks (prioritize FAIL then WARN):"]
    for status in ("fail", "warn", "pass"):
        subset = [c for c in checks if c["status"] == status]
        for check in subset[:6 if status != "pass" else 3]:
            line = f"- [{status.upper()}] {check['headline']}"
            if check.get("detail"):
                line += f" — {check['detail']}"
            lines.append(line)
    return "\n".join(lines)


def _platform_block(portfolio_df: pd.DataFrame) -> str:
    plat = portfolio_df.groupby("Owner", as_index=False)["Current Value"].sum()
    total = plat["Current Value"].sum()
    lines = ["Broker split:"]
    for _, row in plat.iterrows():
        wt = row["Current Value"] / total * 100 if total else 0
        lines.append(f"- {row['Owner']}: ₹{row['Current Value']:,.0f} ({wt:.1f}%)")
    return "\n".join(lines)


def _history_block() -> str:
    history = ph.load_snapshots()
    if history.empty or len(history) < 2:
        return ""
    first = history.iloc[0]
    last = history.iloc[-1]
    change = last["total_current"] - first["total_current"]
    pct = change / first["total_current"] * 100 if first["total_current"] else 0
    return (
        f"Snapshot history ({len(history)} saves, {first['date'].date()} → {last['date'].date()}):\n"
        f"- Value change: ₹{change:,.0f} ({pct:+.1f}%)"
    )


def _top_holdings_technicals(merged: pd.DataFrame, limit: int = 5) -> str:
    lines = ["Recent price action (30d, top holdings by weight):"]
    for symbol in merged.head(limit)["Symbol"]:
        ticker = san.to_yahoo_ticker(symbol)
        df = san.fetch_price_data(ticker, days=35)
        if df is None or df.empty:
            lines.append(f"- {symbol}: price data unavailable")
            continue
        summary = san.summarize_price_trend(df)
        lines.append(
            f"- {symbol}: {summary['trend']} {summary['pct_change_30d']:+.1f}% "
            f"(₹{summary['start_close']:,.0f} → ₹{summary['end_close']:,.0f}), "
            f"30d range ₹{summary['low_30d']:,.0f}–₹{summary['high_30d']:,.0f}"
        )
    return "\n".join(lines)


def build_deep_analysis_context(
    merged: pd.DataFrame,
    summary: dict,
    metrics: dict,
    portfolio_df: pd.DataFrame | None = None,
    *,
    include_technicals: bool = True,
    include_fundamentals: bool = True,
    terminal_snapshot: dict | None = None,
    risk: dict | None = None,
    fmp_snapshot: dict | None = None,
    inst_df: pd.DataFrame | None = None,
    fmp_api_key: str = "",
) -> str:
    """Assemble a data-rich prompt context for AI research."""
    enriched, sector_df, coverage = _resolve_market(merged, summary)

    checks: list[dict] = []
    if portfolio_df is not None:
        checks = pan.run_portfolio_checks(merged, portfolio_df, summary, metrics)
        if sector_df is not None:
            checks.extend(pan.run_sector_checks(sector_df))

    sections = [
        pan.build_portfolio_context(merged, summary, metrics),
        "",
        _pnl_attribution(merged),
        "",
        _quadrant_block(merged),
    ]

    if portfolio_df is not None:
        sections.extend(["", _platform_block(portfolio_df)])

    if sector_df is not None and not sector_df.empty:
        sector_lines = ["Sector allocation:"]
        for _, row in sector_df.iterrows():
            sector_lines.append(
                f"- {row['Sector']}: {row['Weight %']:.1f}% wt, "
                f"{row['Return %']:+.1f}% ret, stocks: {row['Stocks']}"
            )
        sections.extend(["", "\n".join(sector_lines)])

    sections.extend(["", _benchmark_block(merged, sector_df)])

    if include_fundamentals and enriched is not None:
        sections.extend(["", _enriched_holdings_block(enriched, merged)])
        sections.extend(["", _valuation_signals(enriched, merged)])
        if coverage:
            sections.append(
                f"\nMarket data coverage: {coverage.get('found', 0)}/{coverage.get('total', 0)} "
                f"stocks on Yahoo, {coverage.get('sector_mapped', 0)} with sector tags."
            )

    if checks:
        sections.extend(["", _checks_block(checks)])

    forensic_checks = pforensic.run_forensic_checks(
        merged,
        enriched,
        fmp_snapshot,
        api_key=fmp_api_key,
    )
    if forensic_checks:
        sections.extend(["", _checks_block(forensic_checks)])

    inst_checks = screener.run_institutional_checks(merged, inst_df)
    if inst_checks:
        sections.extend(["", _checks_block(inst_checks)])

    if risk:
        sections.extend(["", prisk.build_risk_context(risk)])

    forensic_ctx = pforensic.build_forensic_context(fmp_snapshot, merged)
    if forensic_ctx:
        sections.extend(["", forensic_ctx])

    inst_ctx = screener.build_institutional_context(inst_df, merged)
    if inst_ctx:
        sections.extend(["", inst_ctx])

    history = _history_block()
    if history:
        sections.extend(["", history])

    if include_technicals:
        sections.extend(["", _top_holdings_technicals(merged)])

    if terminal_snapshot and terminal_snapshot.get("available"):
        sections.extend(["", pterm.build_terminal_context(terminal_snapshot, merged)])

    return "\n".join(sections)
