"""Altair charts for Insights — clean visuals for a personal portfolio review."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st


def _dark() -> bool:
    try:
        return st.context.theme.type == "dark"
    except Exception:
        return False


_LIGHT_PALETTE = {
    "colors": ["#4F46E5", "#047857", "#0E7490", "#BE123C", "#B45309", "#6D28D9", "#0F766E", "#94A3B8"],
    "green": "#047857",
    "red": "#BE123C",
    "blue": "#4F46E5",
    "amber": "#B45309",
    "gray": "#94A3B8",
    "muted": "#64748B",
    "title": "#0B1220",
    "grid": "#E3E8F0",
    "axis": "#CBD5E1",
    "stroke": "#FFFFFF",
    "soft_green": "#6EE7B7",
    "soft_red": "#FDA4AF",
}
_DARK_PALETTE = {
    "colors": ["#818CF8", "#34D399", "#38BDF8", "#FB7185", "#FBBF24", "#A78BFA", "#2DD4BF", "#94A3B8"],
    "green": "#34D399",
    "red": "#FB7185",
    "blue": "#818CF8",
    "amber": "#FBBF24",
    "gray": "#64748B",
    "muted": "#94A3B8",
    "title": "#E6EDF7",
    "grid": "#23304B",
    "axis": "#334155",
    "stroke": "#0B1120",
    "soft_green": "#0F766E",
    "soft_red": "#9F1239",
}


def _p() -> dict:
    return _DARK_PALETTE if _dark() else _LIGHT_PALETTE


def _c(name: str) -> str:
    return _p()[name]


# Readable type — charts sit beside metrics/tables; details live in tooltips.
FS_TITLE = 15
FS_SUBTITLE = 11.5
FS_AXIS = 11.5
FS_LEGEND = 11.5


HEADING_FONT = "Plus Jakarta Sans, Inter, sans-serif"
BODY_FONT = "Inter, sans-serif"


def _title(text: str, subtitle: str | None = None) -> alt.TitleParams:
    return alt.TitleParams(
        text=text,
        subtitle=subtitle or "",
        anchor="start",
        font=HEADING_FONT,
        fontSize=FS_TITLE,
        fontWeight=700,
        color=_c("title"),
        subtitleFont=BODY_FONT,
        subtitleColor=_c("muted"),
        subtitleFontSize=FS_SUBTITLE,
        offset=6,
        subtitlePadding=4,
    )


def _base_props(height: int = 280, top: int = 30) -> dict:
    # Top padding reserves room for the title block so it never clips.
    return {"height": height, "padding": {"left": 4, "right": 8, "top": top, "bottom": 6}}


def _style(chart: alt.Chart | alt.LayerChart) -> alt.Chart:
    return (
        chart.configure_view(strokeWidth=0)
        .configure_axis(
            labelFont=BODY_FONT,
            titleFont=BODY_FONT,
            labelColor=_c("muted"),
            titleColor=_c("muted"),
            gridColor=_c("grid"),
            gridOpacity=0.55,
            gridDash=[2, 4],
            domainColor=_c("axis"),
            tickColor=_c("axis"),
            labelFontSize=FS_AXIS,
            titleFontSize=FS_AXIS,
            titleFontWeight=500,
            titlePadding=8,
            labelPadding=4,
        )
        .configure_axisX(grid=False)
        .configure_axisY(domain=False, ticks=False)
        .configure_legend(
            labelFont=BODY_FONT,
            titleFont=BODY_FONT,
            labelColor=_c("muted"),
            titleColor=_c("muted"),
            labelFontSize=FS_LEGEND,
            symbolSize=90,
            symbolType="circle",
            orient="bottom",
            direction="horizontal",
            columns=3,
            padding=4,
            offset=8,
        )
        .configure_title(anchor="start")
    )


def _gain_loss(series: pd.Series) -> pd.Series:
    return series.apply(lambda v: "Gain" if v >= 0 else "Loss")


def _bar_height(n: int, row_px: int = 22, min_h: int = 220) -> int:
    return max(min_h, n * row_px + 48)


def prepare_allocation_data(merged: pd.DataFrame, top_n: int = 7) -> pd.DataFrame:
    if len(merged) <= top_n + 1:
        data = merged[["Symbol", "Current Value", "Weight %", "Return %"]].copy()
        data["Label"] = data["Symbol"]
        return data

    top = merged.head(top_n).copy()
    others_value = merged.iloc[top_n:]["Current Value"].sum()
    others_weight = merged.iloc[top_n:]["Weight %"].sum()
    others = pd.DataFrame(
        [
            {
                "Symbol": "Others",
                "Label": f"Others ({len(merged) - top_n})",
                "Current Value": others_value,
                "Weight %": others_weight,
                "Return %": merged.iloc[top_n:]["Return %"].mean(),
            }
        ]
    )
    top["Label"] = top["Symbol"]
    return pd.concat([top, others], ignore_index=True)


def _donut(
    data: pd.DataFrame,
    category: str,
    value_col: str,
    weight_col: str,
    title: str,
    subtitle: str | None = None,
    height: int = 280,
) -> alt.Chart:
    """Donut only — no center text (avoids legend overlap). Hover for weights."""
    chart = (
        alt.Chart(data)
        .mark_arc(
            innerRadius=62,
            outerRadius=108,
            stroke=_c("stroke"),
            strokeWidth=2,
            padAngle=0.012,
            cornerRadius=2,
        )
        .encode(
            theta=alt.Theta(f"{value_col}:Q", stack=True),
            color=alt.Color(
                f"{category}:N",
                scale=alt.Scale(range=_c("colors")),
                legend=alt.Legend(title=None, labelLimit=120),
            ),
            order=alt.Order(f"{weight_col}:Q", sort="descending"),
            tooltip=[
                alt.Tooltip(f"{category}:N", title="Name"),
                alt.Tooltip(f"{weight_col}:Q", format=".1f", title="Weight %"),
                alt.Tooltip(f"{value_col}:Q", format=",.0f", title="Value (₹)"),
            ],
        )
        .properties(**_base_props(height, top=56), title=_title(title, subtitle))
    )
    return _style(chart)


def allocation_donut(merged: pd.DataFrame) -> alt.Chart:
    return _donut(
        prepare_allocation_data(merged),
        "Label",
        "Current Value",
        "Weight %",
        "Holdings mix",
        "Top names by value — hover for %",
        height=300,
    )


def platform_donut(portfolio_df: pd.DataFrame) -> alt.Chart:
    data = (
        portfolio_df.groupby("Owner", as_index=False)["Current Value"]
        .sum()
        .rename(columns={"Owner": "Platform"})
    )
    data["Weight %"] = data["Current Value"] / data["Current Value"].sum() * 100
    return _donut(data, "Platform", "Current Value", "Weight %", "Broker split", height=260)


def sector_donut(sector_df: pd.DataFrame) -> alt.Chart:
    data = sector_df.copy()
    if len(data) > 8:
        top = data.head(7).copy()
        others = pd.DataFrame(
            [
                {
                    "Sector": "Others",
                    "Current Value": data.iloc[7:]["Current Value"].sum(),
                    "Weight %": data.iloc[7:]["Weight %"].sum(),
                }
            ]
        )
        data = pd.concat([top, others], ignore_index=True)
    return _donut(data, "Sector", "Current Value", "Weight %", "Sector mix", height=300)


def weight_bar_colored(merged: pd.DataFrame, limit: int = 12) -> alt.Chart:
    data = merged.head(limit).copy()
    data["Performance"] = _gain_loss(data["Return %"])
    return _style(
        alt.Chart(data)
        .mark_bar(cornerRadiusEnd=3, height=14)
        .encode(
            x=alt.X("Weight %:Q", title="Weight %", scale=alt.Scale(nice=True)),
            y=alt.Y("Symbol:N", sort="-x", title=None, axis=alt.Axis(labelLimit=80)),
            color=alt.Color(
                "Performance:N",
                scale=alt.Scale(domain=["Gain", "Loss"], range=[_c("green"), _c("red")]),
                legend=alt.Legend(title=None),
            ),
            tooltip=[
                alt.Tooltip("Symbol:N"),
                alt.Tooltip("Weight %:Q", format=".1f"),
                alt.Tooltip("Return %:Q", format="+.1f", title="Return %"),
                alt.Tooltip("Current Value:Q", format=",.0f", title="Value (₹)"),
            ],
        )
        .properties(
            height=_bar_height(len(data)),
            title=_title("Weight vs return", "Green = up · red = down"),
        )
    )


def stock_concentration_bars(merged: pd.DataFrame, limit: int = 8) -> alt.Chart:
    """Top holdings only — you already have the full table below."""
    data = merged.head(limit).copy()
    bars = (
        alt.Chart(data)
        .mark_bar(cornerRadiusEnd=3, height=14)
        .encode(
            x=alt.X("Weight %:Q", title="Weight %"),
            y=alt.Y("Symbol:N", sort="-x", title=None, axis=alt.Axis(labelLimit=80)),
            color=alt.Color(
                "Return %:Q",
                scale=alt.Scale(domainMid=0, range=[_c("red"), _c("muted"), _c("green")]),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("Symbol:N"),
                alt.Tooltip("Weight %:Q", format=".1f"),
                alt.Tooltip("Return %:Q", format="+.1f"),
                alt.Tooltip("Current Value:Q", format=",.0f", title="Value (₹)"),
            ],
        )
    )
    rule15 = (
        alt.Chart(pd.DataFrame({"x": [15], "note": ["15% line"]}))
        .mark_rule(strokeDash=[4, 4], color=_c("amber"), opacity=0.6)
        .encode(x="x:Q")
    )
    return _style(
        (bars + rule15).properties(
            height=_bar_height(len(data)),
            title=_title("Top holdings", "Dashed line = 15% single-stock guide"),
        )
    )


def cumulative_concentration(merged: pd.DataFrame) -> alt.Chart:
    data = merged[["Symbol", "Weight %", "Cumulative weight %"]].head(10).copy()
    symbol_order = data["Symbol"].tolist()

    line = (
        alt.Chart(data)
        .mark_line(color=_c("blue"), strokeWidth=2, point=alt.OverlayMarkDef(size=40, filled=True))
        .encode(
            x=alt.X(
                "Symbol:N",
                sort=symbol_order,
                title=None,
                axis=alt.Axis(labelAngle=-25, labelLimit=60),
            ),
            y=alt.Y(
                "Cumulative weight %:Q",
                title="Cumulative %",
                scale=alt.Scale(domain=[0, 100]),
            ),
            tooltip=[
                alt.Tooltip("Symbol:N"),
                alt.Tooltip("Weight %:Q", format=".1f", title="Stock %"),
                alt.Tooltip("Cumulative weight %:Q", format=".1f", title="Running total %"),
            ],
        )
    )
    rules = (
        alt.Chart(pd.DataFrame({"y": [50, 80]}))
        .mark_rule(strokeDash=[4, 4], color=_c("muted"), opacity=0.45)
        .encode(y="y:Q")
    )
    return _style(
        (line + rules).properties(
            **_base_props(240),
            title=_title("Concentration curve", "How much of the book the top names cover"),
        )
    )


def return_weight_scatter(merged: pd.DataFrame) -> alt.Chart:
    data = merged.copy()
    data["Bucket"] = data.apply(
        lambda r: (
            "Core winner"
            if r["Weight %"] >= 5 and r["Return %"] >= 0
            else "Core loser"
            if r["Weight %"] >= 5 and r["Return %"] < 0
            else "Small winner"
            if r["Return %"] >= 0
            else "Small loser"
        ),
        axis=1,
    )
    points = (
        alt.Chart(data)
        .mark_circle(opacity=0.85, stroke=_c("stroke"), strokeWidth=1)
        .encode(
            x=alt.X("Weight %:Q", title="Weight %", scale=alt.Scale(zero=True, nice=True)),
            y=alt.Y("Return %:Q", title="Return %", scale=alt.Scale(nice=True)),
            size=alt.Size("Current Value:Q", legend=None, scale=alt.Scale(range=[60, 700])),
            color=alt.Color(
                "Bucket:N",
                scale=alt.Scale(
                    domain=["Core winner", "Core loser", "Small winner", "Small loser"],
                    range=[_c("green"), _c("red"), _c("soft_green"), _c("soft_red")],
                ),
                legend=alt.Legend(title=None, labelLimit=100),
            ),
            tooltip=[
                alt.Tooltip("Symbol:N"),
                alt.Tooltip("Weight %:Q", format=".1f"),
                alt.Tooltip("Return %:Q", format="+.1f"),
                alt.Tooltip("P&L:Q", format=",.0f", title="P&L (₹)"),
            ],
        )
    )
    h_rule = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=_c("muted"), opacity=0.45).encode(y="y:Q")
    v_rule = (
        alt.Chart(pd.DataFrame({"x": [5, 15]}))
        .mark_rule(strokeDash=[4, 4], color=_c("muted"), opacity=0.3)
        .encode(x="x:Q")
    )
    return _style(
        (h_rule + v_rule + points).properties(
            **_base_props(280),
            title=_title("Risk map", "Size = value · hover a dot for details"),
        )
    )


def pnl_waterfall(merged: pd.DataFrame, limit: int = 8) -> alt.Chart:
    half = limit // 2
    gainers = merged.nlargest(half, "P&L")[["Symbol", "P&L", "Return %"]]
    losers = merged.nsmallest(half, "P&L")[["Symbol", "P&L", "Return %"]]
    data = pd.concat([losers, gainers], ignore_index=True)
    data["Direction"] = _gain_loss(data["P&L"])

    bars = (
        alt.Chart(data)
        .mark_bar(cornerRadiusEnd=3, height=14)
        .encode(
            x=alt.X("P&L:Q", title="P&L (₹)"),
            y=alt.Y("Symbol:N", sort="x", title=None, axis=alt.Axis(labelLimit=80)),
            color=alt.Color(
                "Direction:N",
                scale=alt.Scale(domain=["Loss", "Gain"], range=[_c("red"), _c("green")]),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("Symbol:N"),
                alt.Tooltip("P&L:Q", format=",.0f", title="P&L (₹)"),
                alt.Tooltip("Return %:Q", format="+.1f"),
            ],
        )
    )
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=_c("muted"), opacity=0.45).encode(x="x:Q")
    return _style(
        (zero + bars).properties(
            height=_bar_height(len(data)),
            title=_title("P&L drivers", "Biggest rupee movers"),
        )
    )


def profit_loss_split(metrics: dict) -> alt.Chart:
    data = pd.DataFrame(
        {
            "Outcome": ["In profit", "In loss", "Flat"],
            "Count": [metrics["in_profit"], metrics["in_loss"], metrics["flat"]],
        }
    )
    data = data[data["Count"] > 0]
    return _style(
        alt.Chart(data)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, size=36)
        .encode(
            x=alt.X("Outcome:N", title=None, sort=["In profit", "In loss", "Flat"]),
            y=alt.Y("Count:Q", title="Holdings"),
            color=alt.Color(
                "Outcome:N",
                scale=alt.Scale(
                    domain=["In profit", "In loss", "Flat"],
                    range=[_c("green"), _c("red"), _c("gray")],
                ),
                legend=None,
            ),
            tooltip=[alt.Tooltip("Outcome:N"), alt.Tooltip("Count:Q")],
        )
        .properties(**_base_props(200), title=_title("Win / loss count"))
    )


def sector_weight_bar(sector_df: pd.DataFrame) -> alt.Chart:
    data = sector_df.copy()
    return _style(
        alt.Chart(data)
        .mark_bar(cornerRadiusEnd=3, height=14)
        .encode(
            x=alt.X("Weight %:Q", title="Weight %"),
            y=alt.Y("Sector:N", sort="-x", title=None, axis=alt.Axis(labelLimit=100)),
            color=alt.Color(
                "Return %:Q",
                scale=alt.Scale(domainMid=0, range=[_c("red"), _c("muted"), _c("green")]),
                legend=alt.Legend(title="Return", orient="bottom", gradientLength=80),
            ),
            tooltip=[
                alt.Tooltip("Sector:N"),
                alt.Tooltip("Weight %:Q", format=".1f"),
                alt.Tooltip("Return %:Q", format="+.1f"),
                alt.Tooltip("Holdings:Q", title="Stocks"),
            ],
        )
        .properties(
            height=_bar_height(len(data)),
            title=_title("Sector weights", "Color = sector return"),
        )
    )


def sector_return_bar(sector_df: pd.DataFrame) -> alt.Chart:
    data = sector_df.copy()
    data["Direction"] = _gain_loss(data["Return %"])
    bars = (
        alt.Chart(data)
        .mark_bar(cornerRadiusEnd=3, height=14)
        .encode(
            x=alt.X("Return %:Q", title="Return %"),
            y=alt.Y("Sector:N", sort="-x", title=None, axis=alt.Axis(labelLimit=100)),
            color=alt.Color(
                "Direction:N",
                scale=alt.Scale(domain=["Gain", "Loss"], range=[_c("green"), _c("red")]),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("Sector:N"),
                alt.Tooltip("Return %:Q", format="+.1f"),
                alt.Tooltip("P&L:Q", format=",.0f", title="P&L (₹)"),
                alt.Tooltip("Weight %:Q", format=".1f"),
            ],
        )
    )
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=_c("muted"), opacity=0.45).encode(x="x:Q")
    return _style(
        (zero + bars).properties(
            height=_bar_height(len(data)),
            title=_title("Sector returns"),
        )
    )


def nifty_vs_portfolio_line(chart_df: pd.DataFrame) -> alt.Chart:
    data = chart_df.copy()
    data["Series"] = data["Series"].replace({"Portfolio": "Your portfolio"})
    data["date"] = pd.to_datetime(data["date"])

    area = (
        alt.Chart(data)
        .mark_area(opacity=0.1, interpolate="monotone")
        .encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("Indexed:Q", title="Indexed (100 = start)", scale=alt.Scale(zero=False, nice=True)),
            color=alt.Color(
                "Series:N",
                scale=alt.Scale(domain=["Your portfolio", "Nifty 50"], range=[_c("blue"), _c("amber")]),
                legend=None,
            ),
        )
    )
    lines = (
        alt.Chart(data)
        .mark_line(strokeWidth=2, interpolate="monotone")
        .encode(
            x="date:T",
            y="Indexed:Q",
            color=alt.Color(
                "Series:N",
                scale=alt.Scale(domain=["Your portfolio", "Nifty 50"], range=[_c("blue"), _c("amber")]),
                legend=alt.Legend(title=None),
            ),
            tooltip=[
                alt.Tooltip("date:T", title="Date", format="%d %b"),
                alt.Tooltip("Series:N"),
                alt.Tooltip("Indexed:Q", format=".1f"),
            ],
        )
    )
    baseline = (
        alt.Chart(pd.DataFrame({"y": [100]}))
        .mark_rule(strokeDash=[4, 4], color=_c("muted"), opacity=0.4)
        .encode(y="y:Q")
    )
    return _style(
        (area + baseline + lines).properties(
            **_base_props(260),
            title=_title("Vs Nifty 50", "~30-day indexed return"),
        )
    )


def benchmark_return_bars(comparison: pd.DataFrame) -> alt.Chart:
    data = comparison.dropna(subset=["Return %"]).copy()
    bars = (
        alt.Chart(data)
        .mark_bar(cornerRadiusEnd=4, size=40)
        .encode(
            x=alt.X("Benchmark:N", title=None, sort=["Your portfolio", "Nifty 50"]),
            y=alt.Y("Return %:Q", title="Return %"),
            color=alt.Color(
                "Benchmark:N",
                scale=alt.Scale(domain=["Your portfolio", "Nifty 50"], range=[_c("blue"), _c("amber")]),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("Benchmark:N"),
                alt.Tooltip("Return %:Q", format="+.2f"),
            ],
        )
    )
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=_c("muted"), opacity=0.45).encode(y="y:Q")
    return _style(
        (zero + bars).properties(
            **_base_props(220),
            title=_title("Period returns"),
        )
    )


def sector_vs_nifty_grouped(active_df: pd.DataFrame) -> alt.Chart:
    data = active_df.melt(
        id_vars=["Sector", "Active weight %"],
        value_vars=["Your portfolio %", "Nifty 50 %"],
        var_name="Source",
        value_name="Weight %",
    )
    top_sectors = (
        active_df.assign(_abs=active_df["Active weight %"].abs())
        .sort_values("_abs", ascending=False)
        .head(8)["Sector"]
        .tolist()
    )
    data = data[data["Sector"].isin(top_sectors)].copy()
    data["Source"] = data["Source"].replace(
        {"Your portfolio %": "You", "Nifty 50 %": "Nifty"}
    )
    return _style(
        alt.Chart(data)
        .mark_bar(cornerRadiusEnd=2, height=8)
        .encode(
            y=alt.Y("Sector:N", sort="-x", title=None, axis=alt.Axis(labelLimit=100)),
            x=alt.X("Weight %:Q", title="Weight %"),
            color=alt.Color(
                "Source:N",
                scale=alt.Scale(domain=["You", "Nifty"], range=[_c("blue"), _c("amber")]),
                legend=alt.Legend(title=None),
            ),
            yOffset="Source:N",
            tooltip=[
                alt.Tooltip("Sector:N"),
                alt.Tooltip("Source:N"),
                alt.Tooltip("Weight %:Q", format=".1f"),
                alt.Tooltip("Active weight %:Q", format="+.1f", title="Active"),
            ],
        )
        .properties(
            height=_bar_height(len(top_sectors), row_px=28),
            title=_title("Sector mix vs Nifty", "Blue = you · amber = Nifty"),
        )
    )


def active_weight_bar(active_df: pd.DataFrame) -> alt.Chart:
    data = (
        active_df.assign(_abs=active_df["Active weight %"].abs())
        .sort_values("_abs", ascending=False)
        .head(8)
        .copy()
    )
    data["Tilt"] = data["Active weight %"].apply(lambda v: "Over" if v >= 0 else "Under")
    bars = (
        alt.Chart(data)
        .mark_bar(cornerRadiusEnd=3, height=14)
        .encode(
            y=alt.Y("Sector:N", sort="-x", title=None, axis=alt.Axis(labelLimit=100)),
            x=alt.X("Active weight %:Q", title="You − Nifty (%)"),
            color=alt.Color(
                "Tilt:N",
                scale=alt.Scale(domain=["Over", "Under"], range=[_c("green"), _c("red")]),
                legend=alt.Legend(title=None),
            ),
            tooltip=[
                alt.Tooltip("Sector:N"),
                alt.Tooltip("Your portfolio %:Q", format=".1f", title="You %"),
                alt.Tooltip("Nifty 50 %:Q", format=".1f", title="Nifty %"),
                alt.Tooltip("Active weight %:Q", format="+.1f", title="Active"),
            ],
        )
    )
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=_c("muted"), opacity=0.45).encode(x="x:Q")
    return _style(
        (zero + bars).properties(
            height=_bar_height(len(data), row_px=28),
            title=_title("Active bets", "Over / under vs Nifty"),
        )
    )


def correlation_heatmap(corr: pd.DataFrame) -> alt.Chart | None:
    """Top-holdings return correlation matrix."""
    if corr is None or corr.empty or len(corr.columns) < 2:
        return None

    order = list(corr.columns)
    data = corr.stack().reset_index()
    data.columns = ["Symbol A", "Symbol B", "Correlation"]
    data = data[data["Symbol A"] != data["Symbol B"]]

    chart = (
        alt.Chart(data)
        .mark_rect(stroke=_c("stroke"), strokeWidth=1, cornerRadius=2)
        .encode(
            x=alt.X("Symbol A:N", sort=order, title=None, axis=alt.Axis(labelAngle=-35)),
            y=alt.Y("Symbol B:N", sort=order, title=None),
            color=alt.Color(
                "Correlation:Q",
                scale=alt.Scale(scheme="redblue", domain=[-1, 1]),
                legend=alt.Legend(title="ρ", orient="right"),
            ),
            tooltip=[
                alt.Tooltip("Symbol A:N"),
                alt.Tooltip("Symbol B:N"),
                alt.Tooltip("Correlation:Q", format=".2f"),
            ],
        )
        .properties(
            height=max(240, 28 * len(order)),
            title=_title("Return correlation", "Top holdings · 252d daily returns"),
        )
    )
    return _style(chart)
