"""Portfolio analysis helpers for the Insights tab."""

from __future__ import annotations

from typing import Literal

import pandas as pd

CheckStatus = Literal["pass", "warn", "fail"]


def merge_holdings(portfolio_df: pd.DataFrame, total_current: float) -> pd.DataFrame:
    grouped = (
        portfolio_df.groupby("Symbol", as_index=False)
        .agg(
            {
                "Quantity": "sum",
                "Invested Value": "sum",
                "Current Value": "sum",
                "P&L": "sum",
                "Owner": lambda owners: ", ".join(sorted(set(owners))),
            }
        )
        .rename(columns={"Owner": "Platforms"})
        .sort_values("Current Value", ascending=False)
        .reset_index(drop=True)
    )

    grouped["Return %"] = (grouped["P&L"] / grouped["Invested Value"].replace(0, pd.NA)) * 100
    grouped["Weight %"] = grouped["Current Value"] / total_current * 100
    grouped["Cumulative weight %"] = grouped["Weight %"].cumsum()
    grouped["P&L contribution %"] = grouped["P&L"] / grouped["P&L"].sum() * 100 if grouped["P&L"].sum() else 0
    return grouped


def compute_metrics(merged: pd.DataFrame, summary: dict) -> dict:
    weights = merged["Weight %"] / 100
    hhi = float((weights**2).sum())
    in_profit = int((merged["P&L"] > 0).sum())
    in_loss = int((merged["P&L"] < 0).sum())
    flat = int((merged["P&L"] == 0).sum())
    top3_weight = float(merged.head(3)["Weight %"].sum())
    top5_weight = float(merged.head(5)["Weight %"].sum())
    concentrated = merged[merged["Weight %"] >= 15]

    weighted_return = (
        (merged["Return %"] * merged["Weight %"]).sum() / merged["Weight %"].sum()
        if merged["Weight %"].sum()
        else 0.0
    )

    best = merged.loc[merged["Return %"].idxmax()] if not merged.empty else None
    worst = merged.loc[merged["Return %"].idxmin()] if not merged.empty else None
    largest = merged.iloc[0] if not merged.empty else None
    top_pl = merged.loc[merged["P&L"].idxmax()] if not merged.empty else None
    worst_pl = merged.loc[merged["P&L"].idxmin()] if not merged.empty else None

    if hhi >= 0.25:
        concentration_label = "High"
    elif hhi >= 0.15:
        concentration_label = "Moderate"
    else:
        concentration_label = "Diversified"

    return {
        "holding_count": len(merged),
        "in_profit": in_profit,
        "in_loss": in_loss,
        "flat": flat,
        "hhi": hhi,
        "concentration_label": concentration_label,
        "top3_weight": top3_weight,
        "top5_weight": top5_weight,
        "concentrated_count": len(concentrated),
        "concentrated_symbols": concentrated["Symbol"].tolist(),
        "weighted_return": weighted_return,
        "best_symbol": best["Symbol"] if best is not None else None,
        "best_return": float(best["Return %"]) if best is not None else None,
        "worst_symbol": worst["Symbol"] if worst is not None else None,
        "worst_return": float(worst["Return %"]) if worst is not None else None,
        "largest_symbol": largest["Symbol"] if largest is not None else None,
        "largest_weight": float(largest["Weight %"]) if largest is not None else None,
        "top_pl_symbol": top_pl["Symbol"] if top_pl is not None else None,
        "top_pl_amount": float(top_pl["P&L"]) if top_pl is not None else None,
        "worst_pl_symbol": worst_pl["Symbol"] if worst_pl is not None else None,
        "worst_pl_amount": float(worst_pl["P&L"]) if worst_pl is not None else None,
        "total_invested": summary["total_invested"],
        "total_current": summary["total_current"],
        "pl_amount": summary["pl_amount"],
        "overall_return_pct": (
            summary["pl_amount"] / summary["total_invested"] * 100 if summary["total_invested"] else 0
        ),
    }


def format_weight_table(merged: pd.DataFrame) -> pd.DataFrame:
    display = merged.copy()
    numeric_cols = [
        "Quantity",
        "Invested Value",
        "Current Value",
        "P&L",
        "Return %",
        "Weight %",
        "Cumulative weight %",
        "P&L contribution %",
    ]
    for col in numeric_cols:
        if col in display.columns:
            display[col] = display[col].round(2)
    return display[
        [
            "Symbol",
            "Platforms",
            "Quantity",
            "Invested Value",
            "Current Value",
            "Weight %",
            "Cumulative weight %",
            "P&L",
            "Return %",
            "P&L contribution %",
        ]
    ]


def build_portfolio_context(merged: pd.DataFrame, summary: dict, metrics: dict) -> str:
    lines = [
        "Family portfolio snapshot:",
        f"- Total invested: ₹{summary['total_invested']:,.2f}",
        f"- Current value: ₹{summary['total_current']:,.2f}",
        f"- Overall P&L: ₹{summary['pl_amount']:,.2f} ({metrics['overall_return_pct']:.2f}%)",
        f"- Holdings: {metrics['holding_count']} stocks ({metrics['in_profit']} in profit, {metrics['in_loss']} in loss)",
        f"- Concentration: {metrics['concentration_label']} (HHI {metrics['hhi']:.3f})",
        f"- Top 3 holdings weight: {metrics['top3_weight']:.1f}%",
        f"- Top 5 holdings weight: {metrics['top5_weight']:.1f}%",
        "",
        "Full holdings by weight:",
    ]

    for _, row in merged.iterrows():
        lines.append(
            f"- {row['Symbol']}: {row['Weight %']:.2f}% weight, "
            f"₹{row['Current Value']:,.2f} value, {row['Return %']:.2f}% return, "
            f"P&L ₹{row['P&L']:,.2f}, platforms: {row['Platforms']}"
        )

    if metrics["concentrated_symbols"]:
        lines.extend(
            [
                "",
                "Concentration flags (>15% weight): "
                + ", ".join(metrics["concentrated_symbols"]),
            ]
        )

    return "\n".join(lines)


def build_portfolio_context_with_sectors(
    merged: pd.DataFrame,
    summary: dict,
    metrics: dict,
    sector_df: pd.DataFrame | None = None,
) -> str:
    context = build_portfolio_context(merged, summary, metrics)
    if sector_df is None or sector_df.empty:
        return context

    lines = [context, "", "Sector allocation:"]
    for _, row in sector_df.iterrows():
        lines.append(
            f"- {row['Sector']}: {row['Weight %']:.1f}% weight, "
            f"{row['Return %']:+.1f}% return, stocks: {row['Stocks']}"
        )
    return "\n".join(lines)


def run_portfolio_checks(
    merged: pd.DataFrame,
    portfolio_df: pd.DataFrame,
    summary: dict,
    metrics: dict,
) -> list[dict]:
    checks: list[dict] = []

    def add_check(
        name: str,
        status: CheckStatus,
        headline: str,
        detail: str = "",
        table: pd.DataFrame | None = None,
    ) -> None:
        checks.append(
            {
                "name": name,
                "status": status,
                "headline": headline,
                "detail": detail,
                "table": table,
            }
        )

    concentrated = merged[merged["Weight %"] >= 15]
    if concentrated.empty:
        add_check(
            "single_stock_concentration",
            "pass",
            "No single stock exceeds 15% of the portfolio.",
        )
    else:
        add_check(
            "single_stock_concentration",
            "fail" if concentrated["Weight %"].max() >= 25 else "warn",
            f"{len(concentrated)} stock(s) exceed 15% weight.",
            "Large single-stock positions increase downside if one name falls sharply.",
            concentrated[["Symbol", "Current Value", "Weight %", "Return %"]].reset_index(drop=True),
        )

    top3 = metrics["top3_weight"]
    if top3 >= 65:
        add_check(
            "top3_concentration",
            "fail",
            f"Top 3 holdings are {top3:.1f}% of the portfolio.",
            "Most of your wealth is tied to just three names.",
            merged.head(3)[["Symbol", "Weight %", "Current Value", "Return %"]],
        )
    elif top3 >= 50:
        add_check(
            "top3_concentration",
            "warn",
            f"Top 3 holdings are {top3:.1f}% of the portfolio.",
            "Consider whether you want more names in the mix.",
            merged.head(3)[["Symbol", "Weight %", "Current Value", "Return %"]],
        )
    else:
        add_check(
            "top3_concentration",
            "pass",
            f"Top 3 holdings are {top3:.1f}% — reasonably spread.",
        )

    top5 = metrics["top5_weight"]
    if top5 >= 80:
        add_check(
            "top5_concentration",
            "warn",
            f"Top 5 holdings account for {top5:.1f}% of value.",
            "A handful of stocks drive most of your portfolio.",
            merged.head(5)[["Symbol", "Weight %", "Cumulative weight %"]],
        )
    else:
        add_check(
            "top5_concentration",
            "pass",
            f"Top 5 holdings account for {top5:.1f}% of value.",
        )

    platform_weights = (
        portfolio_df.groupby("Owner")["Current Value"].sum() / summary["total_current"] * 100
    )
    top_platform = platform_weights.idxmax()
    top_platform_weight = float(platform_weights.max())
    platform_table = platform_weights.reset_index()
    platform_table.columns = ["Platform", "Weight %"]
    platform_table["Weight %"] = platform_table["Weight %"].round(2)
    if top_platform_weight >= 90:
        add_check(
            "platform_concentration",
            "warn",
            f"{top_platform_weight:.1f}% of wealth sits on {top_platform}.",
            "Platform outage or sync issues could hide part of your portfolio view.",
            platform_table,
        )
    else:
        add_check(
            "platform_concentration",
            "pass",
            f"Platform split looks balanced (largest: {top_platform} at {top_platform_weight:.1f}%).",
        )

    heavy_losers = merged[(merged["Weight %"] >= 5) & (merged["Return %"] < -10)].sort_values("Return %")
    if not heavy_losers.empty:
        add_check(
            "heavy_losers",
            "warn",
            f"{len(heavy_losers)} meaningful position(s) are down more than 10%.",
            "These names have both size and negative returns — review your thesis.",
            heavy_losers[["Symbol", "Weight %", "P&L", "Return %"]],
        )
    else:
        add_check(
            "heavy_losers",
            "pass",
            "No large positions (>5% weight) are down more than 10%.",
        )

    large_loss_positions = merged[(merged["Weight %"] >= 10) & (merged["P&L"] < 0)]
    if not large_loss_positions.empty:
        add_check(
            "large_loss_positions",
            "fail",
            f"{len(large_loss_positions)} position(s) above 10% weight are in loss.",
            "Big positions in the red can drag overall portfolio returns.",
            large_loss_positions[["Symbol", "Weight %", "P&L", "Return %"]],
        )
    else:
        add_check(
            "large_loss_positions",
            "pass",
            "No 10%+ positions are currently in loss.",
        )

    total_loss = merged.loc[merged["P&L"] < 0, "P&L"].sum()
    loss_ratio = abs(total_loss) / summary["total_invested"] * 100 if summary["total_invested"] else 0
    if loss_ratio >= 20:
        add_check(
            "unrealized_loss_ratio",
            "fail",
            f"Unrealized losses are {loss_ratio:.1f}% of total invested capital.",
            f"Combined red positions: ₹{total_loss:,.2f}.",
        )
    elif loss_ratio >= 10:
        add_check(
            "unrealized_loss_ratio",
            "warn",
            f"Unrealized losses are {loss_ratio:.1f}% of invested capital.",
            f"Combined red positions: ₹{total_loss:,.2f}.",
        )
    else:
        add_check(
            "unrealized_loss_ratio",
            "pass",
            f"Unrealized losses are {loss_ratio:.1f}% of invested capital.",
        )

    overlap = portfolio_df.groupby("Symbol")["Owner"].nunique()
    duplicated = overlap[overlap > 1].index.tolist()
    if duplicated:
        dup_rows = merged[merged["Symbol"].isin(duplicated)][["Symbol", "Platforms", "Weight %", "Return %"]]
        add_check(
            "duplicate_holdings",
            "warn",
            f"{len(duplicated)} stock(s) appear on more than one platform.",
            "Same names across brokers can make true weight easy to miss.",
            dup_rows,
        )
    else:
        add_check(
            "duplicate_holdings",
            "pass",
            "No duplicate symbols across Zerodha and Groww.",
        )

    tiny_positions = merged[merged["Weight %"] < 2]
    if len(tiny_positions) >= 8:
        add_check(
            "fragmentation",
            "warn",
            f"{len(tiny_positions)} holdings are each under 2% weight.",
            "Many tiny positions add tracking effort without moving the needle.",
            tiny_positions[["Symbol", "Weight %", "Current Value"]].head(10),
        )
    else:
        add_check(
            "fragmentation",
            "pass",
            f"Only {len(tiny_positions)} holding(s) are under 2% weight.",
        )

    if metrics["holding_count"] < 5:
        add_check(
            "diversification_count",
            "warn",
            f"Only {metrics['holding_count']} stocks in the portfolio.",
            "Very small portfolios can be volatile if one or two names move.",
        )
    else:
        add_check(
            "diversification_count",
            "pass",
            f"{metrics['holding_count']} stocks — adequate breadth for a direct-equity book.",
        )

    if metrics["in_loss"] > metrics["in_profit"] and metrics["in_loss"] >= 3:
        add_check(
            "win_rate",
            "warn",
            f"More losers ({metrics['in_loss']}) than winners ({metrics['in_profit']}).",
            "Review whether weak names still fit your long-term plan.",
        )
    else:
        add_check(
            "win_rate",
            "pass",
            f"Winners ({metrics['in_profit']}) outnumber or match losers ({metrics['in_loss']}).",
        )

    hhi = metrics["hhi"]
    if hhi >= 0.25:
        add_check(
            "hhi_concentration",
            "fail",
            f"Portfolio concentration index (HHI) is {hhi:.3f} — high.",
            "Higher HHI means returns depend on fewer stocks.",
        )
    elif hhi >= 0.15:
        add_check(
            "hhi_concentration",
            "warn",
            f"Portfolio concentration index (HHI) is {hhi:.3f} — moderate.",
        )
    else:
        add_check(
            "hhi_concentration",
            "pass",
            f"Portfolio concentration index (HHI) is {hhi:.3f} — diversified.",
        )

    return checks


def run_sector_checks(sector_df: pd.DataFrame) -> list[dict]:
    checks: list[dict] = []
    if sector_df.empty:
        return checks

    known = sector_df[sector_df["Sector"] != "Unknown"]
    if known.empty:
        checks.append(
            {
                "name": "sector_data",
                "status": "warn",
                "headline": "Sector data unavailable for all holdings.",
                "detail": "Yahoo Finance could not map sectors. Indian symbols may need NSE tickers.",
                "table": None,
            }
        )
        return checks

    top = known.iloc[0]
    if top["Weight %"] >= 45:
        checks.append(
            {
                "name": "sector_concentration",
                "status": "fail",
                "headline": f"{top['Sector']} is {top['Weight %']:.1f}% of the portfolio.",
                "detail": "Heavy sector concentration adds macro/sector risk on top of stock risk.",
                "table": known.head(4)[["Sector", "Weight %", "Holdings", "Return %"]],
            }
        )
    elif top["Weight %"] >= 30:
        checks.append(
            {
                "name": "sector_concentration",
                "status": "warn",
                "headline": f"{top['Sector']} is {top['Weight %']:.1f}% of the portfolio.",
                "detail": "You are materially overweight in one sector.",
                "table": known.head(4)[["Sector", "Weight %", "Holdings", "Return %"]],
            }
        )
    else:
        checks.append(
            {
                "name": "sector_concentration",
                "status": "pass",
                "headline": f"Largest sector ({top['Sector']}) is {top['Weight %']:.1f}% — within comfort range.",
                "detail": "",
                "table": None,
            }
        )

    unknown_weight = sector_df.loc[sector_df["Sector"] == "Unknown", "Weight %"].sum()
    if unknown_weight >= 10:
        checks.append(
            {
                "name": "sector_data",
                "status": "warn",
                "headline": f"{unknown_weight:.1f}% of portfolio has unknown sector mapping.",
                "detail": "Some symbols may be ETFs, delisted names, or missing on Yahoo Finance.",
                "table": None,
            }
        )

    return checks


def summarize_checks(checks: list[dict]) -> dict:
    return {
        "pass": sum(1 for check in checks if check["status"] == "pass"),
        "warn": sum(1 for check in checks if check["status"] == "warn"),
        "fail": sum(1 for check in checks if check["status"] == "fail"),
    }


def prepare_insights_table(merged: pd.DataFrame) -> pd.DataFrame:
    display = format_weight_table(merged).copy()
    display["Return %"] = display["Return %"].map(lambda x: f"{x:+.2f}%" if pd.notna(x) else "—")
    return display
