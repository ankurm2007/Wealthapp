"""Auto-generated research briefs and findings for Insights."""

from __future__ import annotations

import pandas as pd


CHECK_TITLES = {
    "single_stock_concentration": "One stock is too large",
    "top3_concentration": "Top 3 stocks dominate",
    "top5_concentration": "Top 5 stocks dominate",
    "platform_concentration": "Most money sits on one broker",
    "heavy_losers": "A large stock is down sharply",
    "large_loss_positions": "A big holding is in loss",
    "unrealized_loss_ratio": "Losses are large vs invested capital",
    "duplicate_holdings": "Same stock held on more than one broker",
    "fragmentation": "Many tiny holdings",
    "diversification_count": "Number of stocks",
    "win_rate": "Share of stocks in profit",
    "hhi_concentration": "Overall concentration",
    "sector_concentration": "One industry is too large",
    "sector_data": "Industry data incomplete",
    "high_leverage": "High company debt",
    "earnings_quality": "Cash vs reported profit",
    "piotroski_score": "Financial strength score",
    "fmp_data": "Company data incomplete",
    "promoter_pledge": "Promoter share pledge",
    "fii_interest": "Foreign investor ownership",
}

# Plain titles people can scan without jargon
THEME_META = {
    "capital": {"color": "blue", "title": "Money up vs money down"},
    "contribution": {"color": "violet", "title": "Who drives your P&L"},
    "drawdown": {"color": "red", "title": "Big stocks that are down"},
    "structure": {"color": "orange", "title": "Too many tiny stocks"},
    "asymmetry": {"color": "gray", "title": "Winners vs losers"},
    "satellite": {"color": "green", "title": "Small stocks doing well"},
    "sector": {"color": "yellow", "title": "Industry mix"},
    "relative": {"color": "primary", "title": "You vs Nifty"},
}


def _inr_compact(value: float) -> str:
    sign = "-" if value < 0 else ""
    amount = abs(float(value))
    if amount >= 1_00_00_000:
        return f"{sign}₹{amount / 1_00_00_000:,.2f} Cr"
    if amount >= 1_00_000:
        return f"{sign}₹{amount / 1_00_000:,.2f} L"
    if amount >= 1_000:
        return f"{sign}₹{amount / 1_000:,.1f} K"
    return f"{sign}₹{amount:,.0f}"


def _names(items: list[str], *, limit: int = 3) -> str:
    shown = items[:limit]
    if not shown:
        return ""
    if len(items) > limit:
        return ", ".join(shown) + f" (+{len(items) - limit} more)"
    return ", ".join(shown)


def _finding(theme: str, metric: str, text: str, *, tone: str = "neutral") -> dict:
    meta = THEME_META[theme]
    return {
        "theme": theme,
        "color": meta["color"],
        "title": meta["title"],
        "metric": metric,
        "text": text,
        "tone": tone,
    }


def build_research_brief(
    merged: pd.DataFrame,
    summary: dict,
    metrics: dict,
    checks: list[dict],
    sector_df: pd.DataFrame | None = None,
    benchmark: dict | None = None,
) -> dict:
    """Build a plain-language portfolio diagnosis for Insights."""
    df = merged.copy()
    if df.empty:
        return {
            "headline": "No holdings to analyse yet",
            "health": "mixed",
            "summary": "Load your portfolio first, then come back here for a plain-English read.",
            "kpis": [],
            "findings": [],
            "priorities": [],
            "stats": {},
        }

    ret = float(metrics.get("overall_return_pct") or 0)
    total_current = float(summary.get("total_current") or 0)
    total_pl = float(summary.get("pl_amount") or 0)
    conc = metrics.get("concentration_label") or "n/a"

    winners = df[df["P&L"] > 0]
    losers = df[df["P&L"] < 0]
    capital_in_profit = float(winners["Current Value"].sum()) if not winners.empty else 0.0
    capital_in_loss = float(losers["Current Value"].sum()) if not losers.empty else 0.0
    capital_profit_pct = (capital_in_profit / total_current * 100) if total_current else 0.0
    capital_loss_pct = (capital_in_loss / total_current * 100) if total_current else 0.0

    gains = float(winners["P&L"].sum()) if not winners.empty else 0.0
    losses_abs = abs(float(losers["P&L"].sum())) if not losers.empty else 0.0
    top_gain_share = 0.0
    top_gain_names: list[str] = []
    if gains > 0 and not winners.empty:
        top_g = winners.nlargest(min(2, len(winners)), "P&L")
        top_gain_share = float(top_g["P&L"].sum() / gains * 100)
        top_gain_names = top_g["Symbol"].astype(str).tolist()

    top_loss_share = 0.0
    top_loss_names: list[str] = []
    if losses_abs > 0 and not losers.empty:
        top_l = losers.nsmallest(min(2, len(losers)), "P&L")
        top_loss_share = float(abs(top_l["P&L"].sum()) / losses_abs * 100)
        top_loss_names = top_l["Symbol"].astype(str).tolist()

    tiny = df[df["Weight %"] < 2]
    tiny_count = int(len(tiny))
    tiny_weight = float(tiny["Weight %"].sum()) if not tiny.empty else 0.0

    core_losers = df[(df["Weight %"] >= 5) & (df["Return %"] < 0)].sort_values("P&L")
    core_loser_weight = float(core_losers["Weight %"].sum()) if not core_losers.empty else 0.0
    core_loser_pnl = float(core_losers["P&L"].sum()) if not core_losers.empty else 0.0
    core_loser_names = core_losers["Symbol"].astype(str).head(3).tolist()

    small_winners = df[(df["Weight %"] < 5) & (df["Return %"] >= 15)].sort_values(
        "Return %", ascending=False
    )

    avg_win = float(winners["Return %"].mean()) if not winners.empty else 0.0
    avg_loss = float(losers["Return %"].mean()) if not losers.empty else 0.0

    alpha = None
    days = 30
    port_ret = None
    nifty_ret = None
    if (
        benchmark
        and benchmark.get("portfolio_return") is not None
        and benchmark.get("nifty_return") is not None
    ):
        alpha = float(benchmark.get("alpha") or 0)
        days = int(benchmark.get("days") or 30)
        port_ret = float(benchmark["portfolio_return"])
        nifty_ret = float(benchmark["nifty_return"])

    health = "mixed"
    if (
        ret >= 0
        and capital_profit_pct >= 55
        and conc != "High"
        and core_loser_weight < 20
    ):
        health = "healthy"
    if (
        ret < -5
        or conc == "High"
        or capital_loss_pct >= 55
        or core_loser_weight >= 25
    ):
        health = "needs attention"

    headline = {
        "healthy": "Looking steady — no urgent fixes needed right now",
        "mixed": "Mixed picture — a few spots need a clear decision",
        "needs attention": "Needs attention — fix big losers and concentration before buying more",
    }[health]

    # Short picture for the reader
    summary_bits = [
        f"Picture this: of every ₹100 in your portfolio today, about "
        f"₹{capital_profit_pct:.0f} is in stocks that are up, and "
        f"₹{capital_loss_pct:.0f} is in stocks that are down."
    ]
    if core_loser_weight >= 10 and core_loser_names:
        summary_bits.append(
            f"The main weight dragging you is in {_names(core_loser_names)} "
            f"({core_loser_weight:.0f}% of the portfolio)."
        )
    elif top_gain_share >= 65 and top_gain_names:
        summary_bits.append(
            f"Most of your gains sit in just {_names(top_gain_names)}."
        )
    if alpha is not None:
        if alpha > 1:
            summary_bits.append(f"Over ~{days} days you are ahead of Nifty.")
        elif alpha < -1:
            summary_bits.append(f"Over ~{days} days you are behind Nifty.")
        else:
            summary_bits.append(f"Over ~{days} days you are roughly in line with Nifty.")
    exec_summary = " ".join(summary_bits)

    kpis = [
        {
            "label": "Money in profit",
            "value": f"{capital_profit_pct:.0f}%",
            "help": f"{_inr_compact(capital_in_profit)} of current value is in stocks that are up",
            "tone": "positive" if capital_profit_pct >= 55 else "neutral",
        },
        {
            "label": "Money in loss",
            "value": f"{capital_loss_pct:.0f}%",
            "help": f"{_inr_compact(capital_in_loss)} of current value is in stocks that are down",
            "tone": "negative" if capital_loss_pct >= 45 else "neutral",
        },
        {
            "label": "Stuck in big losers",
            "value": f"{core_loser_weight:.0f}%",
            "help": (
                f"{_inr_compact(core_loser_pnl)} loss in large holdings (5%+ each)"
                if core_loser_pnl
                else "No large holding (5%+) is currently in loss"
            ),
            "tone": "negative"
            if core_loser_weight >= 20
            else "positive"
            if core_loser_weight == 0
            else "neutral",
        },
        {
            "label": f"Vs Nifty (~{days}d)",
            "value": f"{alpha:+.1f}%" if alpha is not None else "—",
            "help": (
                f"Your portfolio {port_ret:+.1f}% · Nifty {nifty_ret:+.1f}%"
                if alpha is not None
                else "Nifty comparison unavailable"
            ),
            "tone": (
                "positive"
                if alpha is not None and alpha > 1
                else "negative"
                if alpha is not None and alpha < -1
                else "neutral"
            ),
        },
    ]

    findings: list[dict] = []

    # Money up vs down
    pl_plain = (
        f"Overall you are {_inr_compact(total_pl)} "
        f"({'up' if total_pl >= 0 else 'down'} {abs(ret):.1f}% from what you invested)."
    )
    findings.append(
        _finding(
            "capital",
            f"{capital_profit_pct:.0f}% up · {capital_loss_pct:.0f}% down",
            (
                f"{_inr_compact(capital_in_profit)} is sitting in stocks that made money; "
                f"{_inr_compact(capital_in_loss)} is sitting in stocks that lost money. "
                f"{pl_plain} Spread of money across stocks looks {str(conc).lower()}."
            ),
            tone="positive"
            if capital_profit_pct >= 55
            else "negative"
            if capital_loss_pct >= 55
            else "neutral",
        )
    )

    # Who drives P&L
    if top_gain_names or top_loss_names:
        bits = []
        if top_gain_names:
            bits.append(
                f"Your profit story is mostly {_names(top_gain_names)} "
                f"— they make up {top_gain_share:.0f}% of all paper gains."
            )
        if top_loss_names:
            bits.append(
                f"Your loss story is mostly {_names(top_loss_names)} "
                f"— they make up {top_loss_share:.0f}% of all paper losses."
            )
        if top_gain_share >= 70 or top_loss_share >= 70:
            bits.append("If those few names reverse, your whole P&L can swing hard.")
        findings.append(
            _finding(
                "contribution",
                f"{top_gain_share:.0f}% of gains · {top_loss_share:.0f}% of losses",
                " ".join(bits),
                tone="negative"
                if top_gain_share >= 70 or top_loss_share >= 70
                else "neutral",
            )
        )

    # Big stocks that are down
    if not core_losers.empty:
        findings.append(
            _finding(
                "drawdown",
                f"{core_loser_weight:.0f}% of portfolio",
                (
                    f"These large holdings (each 5%+) are currently in loss: "
                    f"{_names(core_loser_names)}. Together they pull "
                    f"{_inr_compact(core_loser_pnl)} from your result. "
                    f"Ask: do you still believe in them, or should size come down?"
                ),
                tone="negative",
            )
        )
    else:
        findings.append(
            _finding(
                "drawdown",
                "None right now",
                "None of your large holdings (5%+ each) is in loss. That is a clean base.",
                tone="positive",
            )
        )

    # Tiny stocks
    if tiny_count >= 3:
        findings.append(
            _finding(
                "structure",
                f"{tiny_count} small holdings",
                (
                    f"You have {tiny_count} stocks under 2% each — together only "
                    f"{tiny_weight:.0f}% of the portfolio. They rarely move the total, "
                    f"but they add noise. Merge into stronger ideas, or exit the clutter."
                ),
                tone="negative" if tiny_weight >= 15 else "neutral",
            )
        )

    # Winners vs losers
    if not winners.empty and not losers.empty:
        if abs(avg_loss) > avg_win:
            note = "Losses run deeper than wins — be careful adding more risk."
        else:
            note = "Wins are stronger than losses on average — protect the good ones."
        findings.append(
            _finding(
                "asymmetry",
                f"Up {avg_win:+.0f}% · Down {avg_loss:+.0f}%",
                (
                    f"On average, stocks in profit are {avg_win:+.1f}%, "
                    f"while stocks in loss are {avg_loss:+.1f}%. {note}"
                ),
                tone="negative" if abs(avg_loss) > avg_win * 1.2 else "positive",
            )
        )

    # Small stocks doing well
    if not small_winners.empty:
        gem = small_winners.iloc[0]
        gem_sym = str(gem["Symbol"])
        gem_ret = float(gem["Return %"])
        gem_wt = float(gem["Weight %"])
        findings.append(
            _finding(
                "satellite",
                f"{gem_sym} {gem_ret:+.0f}%",
                (
                    (
                        f"{len(small_winners)} smaller holdings (under 5% each) are up 15%+. "
                        if len(small_winners) > 1
                        else "One smaller holding (under 5%) is up 15%+. "
                    )
                    + f"Best so far: {gem_sym} at {gem_wt:.1f}% of the portfolio. "
                    + "Decide if it deserves a bigger seat — or if the run is just noise."
                ),
                tone="positive",
            )
        )

    # Industry mix
    if sector_df is not None and not sector_df.empty:
        known = sector_df[sector_df["Sector"] != "Unknown"].copy()
        if not known.empty:
            top_sec = known.iloc[0]
            by_ret = known.sort_values("Return %", ascending=False)
            best_sec = by_ret.iloc[0]
            worst_sec = by_ret.iloc[-1]
            top_wt = float(top_sec["Weight %"])
            heavy = top_wt >= 35
            findings.append(
                _finding(
                    "sector",
                    f"{top_sec['Sector']} {top_wt:.0f}%",
                    (
                        f"Your money is spread across {len(known)} industries. "
                        f"Largest slice: {top_sec['Sector']} ({top_wt:.0f}%). "
                        f"Best industry lately: {best_sec['Sector']} "
                        f"({float(best_sec['Return %']):+.1f}%). "
                        f"Weakest: {worst_sec['Sector']} "
                        f"({float(worst_sec['Return %']):+.1f}%)."
                        + (
                            " One industry is doing a lot of the heavy lifting — that adds industry risk."
                            if heavy
                            else ""
                        )
                    ),
                    tone="negative" if heavy else "neutral",
                )
            )
    else:
        findings.append(
            _finding(
                "sector",
                "Not loaded yet",
                "Click Load market data above to see which industries hold your money.",
                tone="neutral",
            )
        )

    # You vs Nifty
    if alpha is not None:
        if alpha > 1:
            note = "You beat the market in this window."
        elif alpha < -1:
            note = "The market did better than you — check your big losers first."
        else:
            note = "You moved roughly with the market."
        findings.append(
            _finding(
                "relative",
                f"{alpha:+.1f}% vs Nifty",
                (
                    f"Last ~{days} days: your portfolio {port_ret:+.1f}%, "
                    f"Nifty 50 {nifty_ret:+.1f}%. {note}"
                ),
                tone="positive" if alpha > 1 else "negative" if alpha < -1 else "neutral",
            )
        )

    priorities: list[dict] = []
    for check in checks:
        if check.get("status") not in ("fail", "warn"):
            continue
        detail = (check.get("headline") or check.get("detail") or "").strip()
        priorities.append(
            {
                "severity": check["status"],
                "title": CHECK_TITLES.get(check["name"], check["name"]),
                "detail": detail,
            }
        )
        if len(priorities) >= 5:
            break

    return {
        "headline": headline,
        "health": health,
        "summary": exec_summary,
        "kpis": kpis,
        "findings": findings,
        "priorities": priorities,
        "stats": {
            "capital_profit_pct": capital_profit_pct,
            "capital_loss_pct": capital_loss_pct,
            "core_loser_weight": core_loser_weight,
            "tiny_count": tiny_count,
            "overall_return_pct": ret,
            "total_pl": total_pl,
        },
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
