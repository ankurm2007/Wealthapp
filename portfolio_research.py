"""Auto-generated research briefs and findings for Insights."""

from __future__ import annotations

import pandas as pd


CHECK_TITLES = {
    "single_stock_concentration": "Single-stock limit",
    "top3_concentration": "Top 3 cluster",
    "top5_concentration": "Top 5 cluster",
    "platform_concentration": "Broker balance",
    "heavy_losers": "Heavy losers",
    "large_loss_positions": "Large red positions",
    "unrealized_loss_ratio": "Unrealized loss ratio",
    "duplicate_holdings": "Duplicate symbols",
    "fragmentation": "Tiny positions",
    "diversification_count": "Stock count",
    "win_rate": "Win rate",
    "hhi_concentration": "Concentration index",
    "sector_concentration": "Sector concentration",
    "sector_data": "Sector data coverage",
    "high_leverage": "Debt / equity",
    "earnings_quality": "Cash vs profit",
    "piotroski_score": "Piotroski F-Score",
    "fmp_data": "FMP forensic data",
    "promoter_pledge": "Promoter pledge",
    "fii_interest": "FII interest",
}


def build_research_brief(
    merged: pd.DataFrame,
    summary: dict,
    metrics: dict,
    checks: list[dict],
    sector_df: pd.DataFrame | None = None,
    benchmark: dict | None = None,
) -> dict:
    """
    Build a clearer, structured research brief for the Analysis tab.
    Returns a dict with headline, summary paragraphs, and key takeaways.
    """
    pl = summary["pl_amount"]
    ret = metrics["overall_return_pct"]
    health = "healthy" if ret >= 0 and metrics["concentration_label"] != "High" else "mixed"
    if ret < -5 or metrics["concentration_label"] == "High":
        health = "needs attention"

    headline = {
        "healthy": "Your portfolio looks broadly healthy",
        "mixed": "Your portfolio has strengths, with a few things to watch",
        "needs attention": "Your portfolio needs closer attention",
    }[health]

    one_liner = (
        f"You have invested **₹{summary['total_invested']:,.0f}**, now worth "
        f"**₹{summary['total_current']:,.0f}** "
        f"({ret:+.1f}% overall, P&L ₹{pl:+,.0f}). "
        f"**{metrics['in_profit']}** stocks are up and **{metrics['in_loss']}** are down "
        f"across **{metrics['holding_count']}** holdings."
    )

    concentration = (
        f"Your biggest stock is **{metrics['largest_symbol']}** "
        f"({metrics['largest_weight']:.1f}% of the book). "
        f"Top 3 holdings make up **{metrics['top3_weight']:.1f}%** — "
        f"concentration looks **{metrics['concentration_label'].lower()}**."
    )

    performance = (
        f"Best performer: **{metrics['best_symbol']}** ({metrics['best_return']:+.1f}%). "
        f"Biggest drag: **{metrics['worst_pl_symbol']}** "
        f"(₹{metrics['worst_pl_amount']:,.0f}, {metrics['worst_return']:+.1f}%)."
    )

    sector_line = "Load market data to see sector exposure."
    sector_heavy = False
    if sector_df is not None and not sector_df.empty:
        known = sector_df[sector_df["Sector"] != "Unknown"]
        if not known.empty:
            top = known.iloc[0]
            sector_heavy = float(top["Weight %"]) >= 30
            sector_line = (
                f"Largest sector: **{top['Sector']}** at **{top['Weight %']:.1f}%** "
                f"({int(top['Holdings'])} stocks)."
            )

    market_line = None
    if benchmark and benchmark.get("portfolio_return") is not None and benchmark.get("nifty_return") is not None:
        alpha = benchmark.get("alpha")
        vs = "ahead of" if alpha and alpha > 0 else "behind" if alpha and alpha < 0 else "in line with"
        market_line = (
            f"Over ~{benchmark.get('days', 30)} days, your portfolio returned "
            f"**{benchmark['portfolio_return']:+.1f}%** vs Nifty 50 "
            f"**{benchmark['nifty_return']:+.1f}%** — you are {vs} the market "
            f"by **{alpha:+.1f}%**."
        )

    failed = [c for c in checks if c["status"] == "fail"]
    warned = [c for c in checks if c["status"] == "warn"]
    if failed:
        action = "Fix first: " + ", ".join(CHECK_TITLES.get(c["name"], c["name"]) for c in failed[:3]) + "."
        action_tone = "negative"
    elif warned:
        action = "Watch next: " + ", ".join(CHECK_TITLES.get(c["name"], c["name"]) for c in warned[:3]) + "."
        action_tone = "neutral"
    else:
        action = "No urgent red flags from automated checks."
        action_tone = "positive"

    takeaways = [
        {"tone": "neutral", "title": "In one line", "body": one_liner},
        {"tone": "neutral" if metrics["concentration_label"] != "High" else "negative", "title": "Concentration", "body": concentration},
        {"tone": "neutral", "title": "Performance", "body": performance},
        {"tone": "negative" if sector_heavy else "neutral", "title": "Sectors", "body": sector_line},
        {"tone": action_tone, "title": "What to do", "body": action},
    ]
    if market_line:
        takeaways.insert(
            1,
            {
                "tone": "positive" if (benchmark.get("alpha") or 0) >= 0 else "negative",
                "title": "Vs Nifty",
                "body": market_line,
            },
        )

    return {
        "headline": headline,
        "health": health,
        "takeaways": takeaways,
    }


def quadrant_labels(merged: pd.DataFrame) -> dict:
    stars = merged[(merged["Weight %"] >= 5) & (merged["Return %"] >= 0)]
    heavy_losers = merged[(merged["Weight %"] >= 5) & (merged["Return %"] < 0)]
    hidden_gems = merged[(merged["Weight %"] < 5) & (merged["Return %"] >= 10)]
    noise = merged[(merged["Weight %"] < 2) & (merged["Return %"] < 0)]
    return {
        "Core winners": stars["Symbol"].tolist()[:4],
        "Core losers": heavy_losers["Symbol"].tolist()[:4],
        "Small winners": hidden_gems["Symbol"].tolist()[:4],
        "Low-impact losers": noise["Symbol"].tolist()[:4],
    }
