import pandas as pd
import streamlit as st
from pathlib import Path

from dotenv import load_dotenv
from kiteconnect import KiteConnect
from kiteconnect.exceptions import KiteException

import portfolio_ai_display as paidisp
import portfolio_ai as pai
import portfolio_analysis as pan
import portfolio_benchmarks as pbench
import portfolio_charts as pchart
import portfolio_chat as pchat
import portfolio_forensics as pforensic
import portfolio_history as ph
import portfolio_market_data as pmd
import portfolio_research as presearch
import portfolio_risk as prisk
import portfolio_terminal as pterm
import portfolio_ui as pui
import screener_import as screener
import stock_analyzer as san
import symbol_resolver as sym
import transcript_auditor as taudit
import yahoo_client as yahoo
import zerodha_auth as zauth

load_dotenv(Path(__file__).resolve().parent / ".env")
yahoo.configure_yfinance()

st.set_page_config(
    page_title="Family Wealth Dashboard",
    page_icon=":material/account_balance_wallet:",
    layout="wide",
)

if "portfolio_df" not in st.session_state:
    st.session_state.portfolio_df = None
if "portfolio_summary" not in st.session_state:
    st.session_state.portfolio_summary = None
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "agent_analysis" not in st.session_state:
    st.session_state.agent_analysis = None
if "agent_analysis_ticker" not in st.session_state:
    st.session_state.agent_analysis_ticker = None
if "ai_portfolio_briefing" not in st.session_state:
    st.session_state.ai_portfolio_briefing = None
if "ai_holding_research" not in st.session_state:
    st.session_state.ai_holding_research = None
if "risk_snapshot" not in st.session_state:
    st.session_state.risk_snapshot = None
if "fmp_forensic_snapshot" not in st.session_state:
    st.session_state.fmp_forensic_snapshot = None
if "institutional_df" not in st.session_state:
    st.session_state.institutional_df = None
if "transcript_audit" not in st.session_state:
    st.session_state.transcript_audit = None


def get_secret(path: str, default: str = "") -> str:
    try:
        node = st.secrets
        for part in path.split("."):
            node = node[part]
        return str(node) if node else default
    except Exception:
        return default


if "zerodha_access_token" not in st.session_state:
    st.session_state.zerodha_access_token = ""
if "zerodha_api_secret" not in st.session_state:
    st.session_state.zerodha_api_secret = ""
if "zerodha_auth_error" not in st.session_state:
    st.session_state.zerodha_auth_error = ""
if "zerodha_auth_success" not in st.session_state:
    st.session_state.zerodha_auth_success = ""

if not st.session_state.zerodha_api_secret:
    st.session_state.zerodha_api_secret = get_secret("zerodha.api_secret")
if not st.session_state.zerodha_access_token:
    cached_token = zauth.load_cached_token()
    if cached_token:
        st.session_state.zerodha_access_token = cached_token
    else:
        secrets_token = get_secret("zerodha.access_token")
        if secrets_token:
            st.session_state.zerodha_access_token = secrets_token


def handle_zerodha_oauth_callback() -> None:
    request_token = st.query_params.get("request_token")
    if not request_token:
        return

    handled = st.session_state.setdefault("zerodha_handled_request_tokens", set())
    if request_token in handled:
        return
    handled.add(request_token)

    api_key = get_secret("zerodha.api_key")
    api_secret = zauth.resolve_api_secret(
        get_secret("zerodha.api_secret"),
        st.session_state.zerodha_api_secret,
    )
    stored_access_token = get_secret("zerodha.access_token")
    cred_error = zauth.validate_credentials(api_key, api_secret, stored_access_token)
    if cred_error:
        st.session_state.zerodha_auth_error = cred_error
        return

    status = st.query_params.get("status", "")
    if status and status != "success":
        st.session_state.zerodha_auth_error = f"Zerodha login failed ({status}). Try again."
        return

    try:
        st.session_state.zerodha_access_token = zauth.generate_access_token(
            api_key, api_secret, request_token
        )
        st.session_state.zerodha_auth_success = "Zerodha connected for today."
        st.session_state.zerodha_auth_error = ""
        st.query_params.clear()
        st.rerun()
    except Exception as exc:
        st.session_state.zerodha_auth_error = str(exc)


handle_zerodha_oauth_callback()


def get_smart_match(cols: list[str], keywords: list[str]) -> str | None:
    for kw in keywords:
        if kw in cols:
            return kw
    for col in cols:
        for kw in keywords:
            if kw in col:
                return col
    return None


def parse_groww_file(uploaded_file) -> list[dict]:
    if uploaded_file.name.endswith(".csv"):
        raw_df = pd.read_csv(uploaded_file, header=None)
    else:
        raw_df = pd.read_excel(uploaded_file, header=None)

    header_row_index = -1
    for i, row in raw_df.head(20).iterrows():
        row_values = [str(val).lower() for val in row.values]
        if any("qty" in v or "quantity" in v or "shares" in v for v in row_values) and any(
            "price" in v or "avg" in v or "current" in v or "close" in v for v in row_values
        ):
            header_row_index = i
            break

    if header_row_index == -1:
        st.sidebar.error("Could not locate the header row in the Groww file.")
        return []

    groww_df = raw_df.iloc[header_row_index + 1 :].copy()
    cols = [str(col).strip().lower() for col in raw_df.iloc[header_row_index].values]
    groww_df.columns = cols

    # Prefer human-readable name/ticker columns; ISIN is only a fallback.
    name_col = get_smart_match(cols, ["company name", "stock name", "security name", "name of company"])
    if not name_col:
        name_col = get_smart_match(cols, ["company", "stock", "instrument", "name"])
    ticker_col = get_smart_match(cols, ["trading symbol", "tradingsymbol", "symbol", "scrip", "ticker"])
    isin_col = get_smart_match(cols, ["isin", "isin number", "isinnumber"])
    qty_col = get_smart_match(cols, ["quantity", "qty", "shares", "balance"])
    price_col = get_smart_match(cols, ["average", "avg", "buy"])
    current_price_col = get_smart_match(
        cols, ["ltp", "close", "closing", "market price", "current price", "current"]
    )

    if not current_price_col:
        st.sidebar.warning(
            "Could not identify a current price column. Buy price will be used instead. "
            f"Columns found: {cols}"
        )

    if not ((name_col or ticker_col or isin_col) and qty_col and price_col):
        st.sidebar.error("Could not match standard columns (symbol/name, quantity, buy price).")
        return []

    # Warm ISIN map once so Groww ISIN rows resolve to NSE symbols.
    try:
        sym.load_isin_map()
    except Exception:
        pass

    holdings = []
    for _, row in groww_df.iterrows():
        try:
            raw_qty = str(row[qty_col]).replace(",", "").strip()
            raw_price = str(row[price_col]).replace("₹", "").replace(",", "").strip()
            if current_price_col and pd.notna(row[current_price_col]):
                raw_current = str(row[current_price_col]).replace("₹", "").replace(",", "").strip()
            else:
                raw_current = raw_price

            if not (
                raw_qty.replace(".", "", 1).isdigit() and raw_price.replace(".", "", 1).isdigit()
            ):
                continue

            company_name = str(row[name_col]).strip() if name_col and pd.notna(row[name_col]) else ""
            ticker_value = str(row[ticker_col]).strip() if ticker_col and pd.notna(row[ticker_col]) else ""
            isin_value = str(row[isin_col]).strip() if isin_col and pd.notna(row[isin_col]) else ""

            # Resolve in order: ticker → company name → ISIN.
            # Avoid using ISIN as the display label when a better value exists.
            candidates = []
            if ticker_value and not sym.looks_like_isin(ticker_value):
                candidates.append(ticker_value)
            if company_name and not sym.looks_like_isin(company_name):
                candidates.append(company_name)
            if isin_value:
                candidates.append(isin_value)
            if ticker_value:
                candidates.append(ticker_value)
            if company_name:
                candidates.append(company_name)

            if not candidates:
                continue

            resolved = None
            for candidate in candidates:
                resolved = sym.resolve_nse_symbol(candidate)
                if resolved and not sym.looks_like_isin(resolved):
                    break

            display_name = company_name or ticker_value or isin_value
            holdings.append(
                {
                    "Owner": "Groww",
                    "Symbol": resolved or candidates[0],
                    "Company Name": display_name,
                    "Quantity": float(raw_qty),
                    "Buy Price": float(raw_price),
                    "Current Price": float(raw_current),
                }
            )
        except Exception:
            continue

    return holdings


def fetch_zerodha_holdings(api_key: str, access_token: str) -> list[dict]:
    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
    try:
        live_holdings = kite.holdings()
    except KiteException as exc:
        if zauth.is_access_token_error(exc):
            zauth.clear_cached_token()
            st.session_state.zerodha_access_token = ""
            raise RuntimeError(zauth.access_token_error_message(exc)) from exc
        raise RuntimeError(zauth.token_error_message(exc)) from exc
    return [
        {
            "Owner": "Zerodha (Kite)",
            "Symbol": item["tradingsymbol"],
            "Company Name": item["tradingsymbol"],
            "Quantity": item["quantity"],
            "Buy Price": item["average_price"],
            "Current Price": item["last_price"],
        }
        for item in live_holdings
    ]


def enrich_portfolio(portfolio_df: pd.DataFrame) -> pd.DataFrame:
    df = portfolio_df.copy()
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["Buy Price"] = pd.to_numeric(df["Buy Price"], errors="coerce")
    df["Current Price"] = pd.to_numeric(df["Current Price"], errors="coerce")
    df["Invested Value"] = df["Quantity"] * df["Buy Price"]
    df["Current Value"] = df["Quantity"] * df["Current Price"]
    df["P&L"] = df["Current Value"] - df["Invested Value"]
    df["Return %"] = (df["P&L"] / df["Invested Value"].replace(0, pd.NA)) * 100
    return df


def build_summary(portfolio_df: pd.DataFrame) -> dict:
    total_invested = portfolio_df["Invested Value"].sum()
    total_current = portfolio_df["Current Value"].sum()
    platform_totals = portfolio_df.groupby("Owner")["Current Value"].sum().to_dict()
    return {
        "total_invested": total_invested,
        "total_current": total_current,
        "pl_amount": total_current - total_invested,
        "zerodha_current": platform_totals.get("Zerodha (Kite)", 0.0),
        "groww_current": platform_totals.get("Groww", 0.0),
        "holding_count": len(portfolio_df),
    }


def calculate_portfolio(api_key: str, access_token: str, groww_file) -> None:
    holdings = []

    if groww_file:
        try:
            holdings.extend(parse_groww_file(groww_file))
            if holdings:
                st.sidebar.success("Groww data loaded.")
        except Exception as exc:
            st.sidebar.error(f"Error reading the Groww file: {exc}")

    if api_key and access_token:
        try:
            holdings.extend(fetch_zerodha_holdings(api_key, access_token))
            st.sidebar.success("Zerodha holdings fetched.")
        except Exception as exc:
            st.sidebar.error(f"Failed to fetch Zerodha data: {exc}")
    elif api_key or access_token:
        st.sidebar.warning("Provide both Zerodha API key and access token.")

    if not holdings:
        st.session_state.portfolio_df = None
        st.session_state.portfolio_summary = None
        st.warning("No holdings found. Upload a Groww file and/or connect Zerodha.")
        return

    portfolio_df = sym.normalize_portfolio_symbols(enrich_portfolio(pd.DataFrame(holdings)))
    st.session_state.portfolio_df = portfolio_df
    st.session_state.portfolio_summary = build_summary(portfolio_df)
    pmd.clear_market_cache()


def render_portfolio_tab(portfolio_df: pd.DataFrame, summary: dict) -> None:
    total_return = (
        summary["pl_amount"] / summary["total_invested"] * 100 if summary["total_invested"] else 0
    )
    pui.kpi_row(
        [
            ("Total invested", pui.format_inr(summary["total_invested"]), None),
            (
                "Current value",
                pui.format_inr(summary["total_current"]),
                pui.format_inr(summary["pl_amount"]),
            ),
            ("Total return", pui.format_pct(total_return), f"{summary['holding_count']} holdings"),
            (
                "Zerodha",
                pui.format_inr(summary.get("zerodha_current", 0)),
                None,
            ),
            (
                "Groww",
                pui.format_inr(summary.get("groww_current", 0)),
                None,
            ),
        ]
    )

    pui.section("Platform breakdown", "Invested vs current by broker", icon="pie_chart")
    platforms = list(portfolio_df["Owner"].unique())
    plat_cols = st.columns(min(len(platforms), 2) or 1)
    for idx, platform in enumerate(platforms):
        plat_df = portfolio_df[portfolio_df["Owner"] == platform]
        plat_invested = plat_df["Invested Value"].sum()
        plat_current = plat_df["Current Value"].sum()
        plat_pl = plat_current - plat_invested
        plat_ret = (plat_pl / plat_invested * 100) if plat_invested else 0
        with plat_cols[idx % len(plat_cols)]:
            with st.container(border=True):
                st.markdown(f"**:material/store: {platform}**")
                with st.container(horizontal=True):
                    st.metric("Invested", pui.format_inr(plat_invested), border=True)
                    st.metric(
                        "Current",
                        pui.format_inr(plat_current),
                        pui.format_inr(plat_pl),
                        border=True,
                    )
                st.caption(f"{len(plat_df)} holdings · {pui.format_pct(plat_ret)} return")

    pui.section("Holdings", "Consolidated view across platforms", icon="table_chart")
    display_cols = [
        "Owner",
        "Symbol",
        "Quantity",
        "Buy Price",
        "Current Price",
        "Invested Value",
        "Current Value",
        "P&L",
        "Return %",
    ]
    with st.container(border=True):
        st.dataframe(
            portfolio_df[display_cols],
            width="stretch",
            hide_index=True,
            column_config={
                "Invested Value": st.column_config.NumberColumn(format="₹%d"),
                "Current Value": st.column_config.NumberColumn(format="₹%d"),
                "P&L": st.column_config.NumberColumn(format="₹%d"),
                "Return %": st.column_config.NumberColumn(format="%+.1f%%"),
                "Buy Price": st.column_config.NumberColumn(format="₹%.2f"),
                "Current Price": st.column_config.NumberColumn(format="₹%.2f"),
            },
        )

    with st.expander(":material/bug_report: Line-by-line debug", expanded=False):
        debug_df = portfolio_df.copy()
        debug_df["Line total"] = debug_df["Quantity"] * debug_df["Current Price"]
        st.dataframe(debug_df, width="stretch", hide_index=True)


def render_trends_tab() -> None:
    history = ph.load_snapshots()
    if history.empty:
        pui.empty_state(
            "No snapshots yet",
            "Save your first snapshot after calculating live wealth.",
            icon="photo_camera",
            hint="Use **Save today's snapshot** in the sidebar. Weekly saves give the best trend lines.",
        )
        return

    monthly = ph.get_monthly_summary(history)
    last_saved = history["date"].max().date()
    days_since = (pd.Timestamp.today().normalize() - pd.Timestamp(last_saved)).days

    pui.kpi_row(
        [
            ("Snapshots", str(len(history)), None),
            ("Last saved", last_saved.strftime("%d %b %Y"), None),
            ("Days since save", str(days_since), "Weekly saves recommended" if days_since > 7 else None),
            (
                "Latest value",
                pui.format_inr(float(history["total_current"].iloc[-1])),
                None,
            ),
        ]
    )

    if days_since > 7:
        pui.status_banner(
            "Last snapshot is over 7 days old — save a fresh one for accurate trends.",
            kind="warning",
        )

    pui.section("Wealth over time", icon="show_chart")
    value_chart_df = history.set_index("date")[["total_invested", "total_current"]]
    with st.container(border=True):
        st.line_chart(
            value_chart_df,
            color=[pui.CHART_LINE_INVESTED, pui.CHART_LINE_CURRENT],
        )
        st.caption(":gray[Gray = invested · Green = current value]")

    if len(monthly) >= 2:
        pui.section("Monthly growth", icon="bar_chart")
        growth_col1, growth_col2 = st.columns(2)
        with growth_col1:
            with st.container(border=True):
                st.markdown("**Monthly change (₹)**")
                change_df = monthly.set_index("date")[["monthly_change"]].dropna()
                st.bar_chart(change_df, color=pui.CHART_BAR_CHANGE)
        with growth_col2:
            with st.container(border=True):
                st.markdown("**Monthly growth (%)**")
                pct_df = monthly.set_index("date")[["monthly_growth_pct"]].dropna()
                st.bar_chart(pct_df, color=pui.CHART_BAR_GROWTH)
    else:
        st.caption("Save snapshots across at least two months to unlock monthly growth charts.")

    if "zerodha_current" in history.columns and history[["zerodha_current", "groww_current"]].sum().sum() > 0:
        pui.section("Platform split over time", icon="stacked_line_chart")
        platform_df = history.set_index("date")[["zerodha_current", "groww_current"]]
        with st.container(border=True):
            st.area_chart(
                platform_df,
                color=[pui.CHART_AREA_ZERODHA, pui.CHART_AREA_GROWW],
            )

    pui.section("Snapshot log", icon="history")
    display_history = history.copy()
    display_history["date"] = display_history["date"].dt.strftime("%Y-%m-%d")
    with st.container(border=True):
        st.dataframe(display_history, width="stretch", hide_index=True)


def get_fmp_api_key() -> str:
    return get_secret("fmp.api_key") or get_secret("FMP_API_KEY")


def analysis_extras() -> dict:
    return {
        "risk": st.session_state.get("risk_snapshot"),
        "fmp_snapshot": st.session_state.get("fmp_forensic_snapshot"),
        "inst_df": st.session_state.get("institutional_df"),
        "fmp_api_key": get_fmp_api_key(),
    }


def render_portfolio_risk_section(merged: pd.DataFrame) -> None:
    pui.section("Portfolio risk", "Beta, Sharpe, and correlation vs Nifty", icon="speed")

    load_risk = st.button("Compute risk metrics", key="load_risk_metrics")
    if load_risk:
        prisk.fetch_daily_returns.clear()

    if load_risk or st.session_state.risk_snapshot:
        if load_risk:
            with st.spinner("Downloading price history and computing risk..."):
                st.session_state.risk_snapshot = prisk.compute_portfolio_risk(merged)
        risk = st.session_state.risk_snapshot
    else:
        st.info("Click **Compute risk metrics** to load beta, Sharpe, and correlation data.")
        return

    if not risk or not risk.get("ok"):
        st.warning(risk.get("error", "Risk metrics unavailable.") if risk else "Risk metrics unavailable.")
        return

    with st.container(horizontal=True):
        beta = risk.get("portfolio_beta")
        sharpe = risk.get("sharpe_ratio")
        st.metric(
            "Portfolio beta",
            f"{beta:.2f}" if beta is not None else "—",
            "vs Nifty 50",
            border=True,
        )
        st.metric(
            "Sharpe ratio",
            f"{sharpe:.2f}" if sharpe is not None else "—",
            f"rf {risk.get('risk_free_rate_pct', 7):.0f}%",
            border=True,
        )
        st.metric(
            "Ann. return",
            f"{risk.get('annualized_return_pct', 0):+.1f}%",
            border=True,
        )
        st.metric(
            "Ann. volatility",
            f"{risk.get('annualized_vol_pct', 0):.1f}%",
            border=True,
        )

    betas = risk.get("stock_betas")
    if isinstance(betas, pd.DataFrame) and not betas.empty:
        with st.expander("Stock betas vs Nifty", expanded=False):
            st.dataframe(
                betas.round(2),
                width="stretch",
                hide_index=True,
                column_config={
                    "Weight %": st.column_config.NumberColumn(format="%.1f%%"),
                    "Beta vs Nifty": st.column_config.NumberColumn(format="%.2f"),
                },
            )

    corr = risk.get("correlation")
    heatmap = pchart.correlation_heatmap(corr) if isinstance(corr, pd.DataFrame) else None
    if heatmap is not None:
        with st.container(border=True):
            st.altair_chart(heatmap, width="stretch")


def render_forensic_section(merged: pd.DataFrame, enriched: pd.DataFrame | None) -> list[dict]:
    pui.section("Forensic checks", "Leverage, cash flow quality, and FMP scores", icon="fact_check")
    st.caption("D/E leverage, FMP Piotroski F-Score, and operating cash flow vs reported profit.")

    fmp_key = get_fmp_api_key()
    if not fmp_key:
        st.info("Add `[fmp] api_key` in `.streamlit/secrets.toml` to enable Piotroski and OCF checks.")
    else:
        st.caption("FMP key detected — load forensic data for your holdings.")

    load_forensic = st.button("Load forensic data", key="load_forensic_data")
    if load_forensic:
        pforensic.fetch_forensic_snapshot.clear()

    fmp_snapshot = st.session_state.get("fmp_forensic_snapshot")
    if load_forensic and fmp_key:
        symbols = tuple(merged["Symbol"].tolist())
        with st.spinner("Fetching FMP scores and cash-flow data..."):
            st.session_state.fmp_forensic_snapshot = pforensic.fetch_forensic_snapshot(symbols, fmp_key)
        fmp_snapshot = st.session_state.fmp_forensic_snapshot

    checks = pforensic.run_forensic_checks(merged, enriched, fmp_snapshot, api_key=fmp_key)
    if not checks and not fmp_key:
        return []
    if not load_forensic and not fmp_snapshot and fmp_key:
        st.caption("Click **Load forensic data** to run Piotroski and earnings-quality checks.")
    return checks


def render_screener_upload(merged: pd.DataFrame) -> list[dict]:
    pui.section("Institutional data", "Upload Screener.in or Trendlyne exports", icon="upload_file")
    st.caption("Upload a Screener.in or Trendlyne export (CSV/XLSX) for promoter pledge, FII/DII, and shareholding checks.")

    uploaded = st.file_uploader(
        "Screener / Trendlyne export",
        type=["csv", "xlsx", "xls"],
        key="institutional_upload",
    )
    if uploaded is not None:
        try:
            inst_df, fmt = screener.parse_institutional_file(uploaded)
            if inst_df.empty:
                st.error("Could not parse columns — export should include Symbol and shareholding fields.")
            else:
                st.session_state.institutional_df = inst_df
                st.success(f"Loaded {len(inst_df)} symbols ({fmt} format).")
        except Exception as exc:
            st.error(f"Could not read file: {exc}")

    inst_df = st.session_state.get("institutional_df")
    if inst_df is not None and not inst_df.empty:
        matched = merged["Symbol"].isin(inst_df["Symbol"]).sum()
        st.caption(f"Matched {matched}/{len(merged)} portfolio symbols.")
        show_cols = [c for c in inst_df.columns if c in ("Symbol", "Promoter %", "Pledged %", "FII %", "DII %")]
        st.dataframe(inst_df[show_cols].head(12), width="stretch", hide_index=True)
        return screener.run_institutional_checks(merged, inst_df)
    return []


def render_transcript_auditor(
    merged: pd.DataFrame,
    summary: dict,
    metrics: dict,
    portfolio_df: pd.DataFrame,
    keys: dict,
    openai_key: str,
) -> None:
    pui.section("Transcript auditor", "Forensic read on earnings call text", icon="record_voice_over")
    st.caption("Paste or upload an earnings call transcript — AI flags guidance gaps, tone, and red flags.")

    has_ai = bool(keys.get("gemini") or keys.get("xai") or keys.get("groq") or openai_key)
    with st.container(border=True):
        if not has_ai:
            st.caption("Configure Gemini, xAI, Groq, or OpenAI to run transcript audits.")
            return

        symbols = merged["Symbol"].tolist()
        default_index = 0
        largest = metrics.get("largest_symbol")
        if largest in symbols:
            default_index = symbols.index(largest)
        audit_symbol = st.selectbox("Stock", options=symbols, index=default_index, key="transcript_symbol")

        tab_paste, tab_upload = st.tabs(["Paste transcript", "Upload file"])
        transcript_text = ""
        with tab_paste:
            transcript_text = st.text_area(
                "Transcript text",
                height=200,
                placeholder="Paste the earnings call transcript here...",
                key="transcript_paste",
            )
        with tab_upload:
            transcript_file = st.file_uploader("Transcript (.txt)", type=["txt"], key="transcript_file")
            if transcript_file is not None:
                transcript_text = transcript_file.read().decode("utf-8", errors="ignore")

        if st.button("Audit transcript", key="audit_transcript"):
            row = merged.loc[merged["Symbol"] == audit_symbol].iloc[0].to_dict()
            excerpt = pai.build_rich_context(
                merged,
                summary,
                metrics,
                portfolio_df,
                include_technicals=False,
                **analysis_extras(),
            )
            with st.spinner(f"Auditing {audit_symbol} transcript..."):
                try:
                    result = taudit.audit_transcript(
                        transcript_text,
                        audit_symbol,
                        row,
                        excerpt,
                        gemini_key=keys.get("gemini") or "",
                        groq_key=keys.get("groq") or "",
                        xai_key=keys.get("xai") or "",
                        openai_key=openai_key,
                    )
                    st.session_state.transcript_audit = result
                except Exception as exc:
                    st.error(str(exc))

    if st.session_state.transcript_audit:
        audit = st.session_state.transcript_audit
        with st.container(border=True):
            st.markdown(f"**Transcript audit · {audit.get('symbol', '')}** · {audit.get('provider', '')}")
            paidisp.render_ai_response(audit.get("text", ""), show_meta=False)


def render_research_brief(brief: dict) -> None:
    health = brief.get("health", "mixed")
    health_badge = {
        "healthy": ":green-badge[Healthy]",
        "mixed": ":orange-badge[Mixed]",
        "needs attention": ":red-badge[Watch]",
    }.get(health, ":blue-badge[Overview]")

    with st.container(border=True):
        st.markdown(f"#### :material/lightbulb: At a glance · {health_badge}")
        st.markdown(f"**{brief['headline']}**")
        st.space("small")
        for item in brief.get("takeaways", []):
            with st.container(border=True):
                st.markdown(f"**{item['title']}**")
                st.caption(item["body"])


def render_portfolio_checks(checks: list[dict]) -> None:
    summary = pan.summarize_checks(checks)
    total = len(checks)
    health_score = round((summary["pass"] / total) * 100) if total else 0

    pui.section("Portfolio checks", "Risk and health signals", icon="health_and_safety")

    with st.container(border=True):
        with st.container(horizontal=True):
            st.metric("Health score", f"{health_score}%", border=True)
            st.metric("Passed", summary["pass"], border=True)
            st.metric("Watch", summary["warn"], border=True)
            st.metric("Action needed", summary["fail"], border=True)

        fail_checks = [c for c in checks if c["status"] == "fail"]
        warn_checks = [c for c in checks if c["status"] == "warn"]
        pass_checks = [c for c in checks if c["status"] == "pass"]

        if fail_checks:
            st.markdown("**:red[Needs attention]**")
            for check in fail_checks:
                with st.container(border=True):
                    title = presearch.CHECK_TITLES.get(check["name"], check["name"])
                    st.markdown(f"**:red-badge[{title}]**")
                    st.markdown(check["headline"])
                    if check["detail"]:
                        st.caption(check["detail"])
                    if check["table"] is not None and not check["table"].empty:
                        st.dataframe(check["table"], width="stretch", hide_index=True)

        if warn_checks:
            st.markdown("**:orange[Watchlist]**")
            cols = st.columns(2)
            for idx, check in enumerate(warn_checks):
                with cols[idx % 2]:
                    with st.container(border=True):
                        title = presearch.CHECK_TITLES.get(check["name"], check["name"])
                        st.markdown(f"**:orange-badge[{title}]**")
                        st.markdown(check["headline"])
                        if check["detail"]:
                            st.caption(check["detail"])

        with st.expander(f"All checks passed ({len(pass_checks)})", expanded=False):
            for check in pass_checks:
                title = presearch.CHECK_TITLES.get(check["name"], check["name"])
                st.markdown(f":green-badge[{title}] {check['headline']}")


def render_sector_section(enriched: pd.DataFrame, sector_df: pd.DataFrame, coverage: dict) -> None:
    pui.section(
        "Sector exposure",
        (
            f"Yahoo sectors · mapped {coverage['sector_mapped']}/{coverage['total']} "
            f"({coverage['sector_mapped_pct']}%) · prices {coverage['found']}/{coverage['total']}"
        ),
        icon="category",
    )

    if coverage["unknown_sector_symbols"]:
        st.warning(
            "Could not classify sector for: "
            + ", ".join(coverage["unknown_sector_symbols"])
            + ". These sit in **Unknown** until Yahoo has sector data or the symbol is added to our alias map."
        )

    if coverage["not_found_symbols"]:
        st.info(
            "Yahoo could not find these tickers: "
            + ", ".join(coverage["not_found_symbols"])
            + ". Check that Groww/Zerodha symbols match NSE trading symbols."
        )

    sec_left, sec_right = st.columns([1, 1])
    with sec_left:
        with st.container(border=True):
            st.altair_chart(pchart.sector_donut(sector_df), width="stretch")
    with sec_right:
        with st.container(border=True):
            st.altair_chart(pchart.sector_weight_bar(sector_df), width="stretch")

    with st.container(border=True):
        st.altair_chart(pchart.sector_return_bar(sector_df), width="stretch")

    st.dataframe(
        sector_df[
            ["Sector", "Weight %", "Current Value", "P&L", "Return %", "Holdings", "Stocks"]
        ].round(2),
        width="stretch",
        hide_index=True,
        column_config={
            "Sector": st.column_config.TextColumn("Sector"),
            "Weight %": st.column_config.ProgressColumn(
                "Weight",
                format="%.1f%%",
                min_value=0,
                max_value=max(30, float(sector_df["Weight %"].max()) + 5),
            ),
            "Current Value": st.column_config.NumberColumn("Current", format="₹%.0f"),
            "P&L": st.column_config.NumberColumn(format="₹%.0f"),
            "Return %": st.column_config.NumberColumn(format="%.1f%%"),
        },
    )


def render_stock_info_table(enriched: pd.DataFrame, coverage: dict) -> None:
    pui.section("Stock information", icon="info")
    st.caption("Live reference data from Yahoo Finance for each holding in your portfolio.")

    display = enriched[
        [
            "Symbol",
            "Company",
            "Sector",
            "Industry",
            "Weight %",
            "Current Value",
            "P&L",
            "Return %",
            "Yahoo price",
            "P/E",
            "52-week range",
            "Data source",
        ]
    ].copy()

    st.dataframe(
        display,
        width="stretch",
        hide_index=True,
        column_config={
            "Symbol": st.column_config.TextColumn("Symbol", width="small"),
            "Company": st.column_config.TextColumn("Company", width="medium"),
            "Sector": st.column_config.TextColumn("Sector"),
            "Industry": st.column_config.TextColumn("Industry"),
            "Weight %": st.column_config.NumberColumn(format="%.1f%%"),
            "Current Value": st.column_config.NumberColumn(format="₹%.0f"),
            "P&L": st.column_config.NumberColumn(format="₹%.0f"),
            "Return %": st.column_config.NumberColumn(format="%.1f%%"),
            "Yahoo price": st.column_config.NumberColumn(format="₹%.2f"),
            "P/E": st.column_config.NumberColumn(format="%.1f"),
            "52-week range": st.column_config.TextColumn("52-week range"),
            "Data source": st.column_config.TextColumn("Source", width="small"),
        },
    )

    if coverage["missing"] > 0:
        missing = enriched.loc[enriched["Data source"] != "Yahoo Finance", "Symbol"].tolist()
        st.caption(f"Missing Yahoo data for: {', '.join(missing)}")


def render_insights_analysis(
    merged: pd.DataFrame,
    metrics: dict,
    portfolio_df: pd.DataFrame,
    summary: dict,
) -> None:
    checks = pan.run_portfolio_checks(merged, portfolio_df, summary, metrics)
    quadrants = presearch.quadrant_labels(merged)

    cached_market = pmd.get_cached_market_context(merged, summary)
    enriched = None
    sector_df = None
    coverage = None

    load_col, refresh_col = st.columns([3, 1])
    with load_col:
        if cached_market is None:
            pui.status_banner(
                "Market data loads on demand — click **Load market data** to fetch Yahoo sectors and fundamentals.",
                kind="info",
            )
    with refresh_col:
        fetch_clicked = st.button(
            "Load market data" if cached_market is None else "Refresh market data",
            width="stretch",
        )

    if fetch_clicked or cached_market is not None:
        if fetch_clicked:
            pmd.fetch_market_info_for_symbols.clear()
            pmd.clear_market_cache()
        if fetch_clicked or cached_market is None:
            with st.spinner("Fetching Yahoo Finance data in parallel..."):
                enriched, sector_df, coverage = pmd.fetch_and_cache_market_context(merged, summary)
        else:
            enriched, sector_df, coverage = cached_market
        checks.extend(pan.run_sector_checks(sector_df))

    benchmark = None
    with st.spinner("Comparing portfolio vs Nifty 50..."):
        try:
            benchmark = pbench.build_nifty_vs_portfolio(merged, days=30)
        except Exception:
            benchmark = None

    brief = presearch.build_research_brief(
        merged, summary, metrics, checks, sector_df, benchmark=benchmark
    )

    pui.section("Overview", "Key numbers and research brief", icon="analytics")

    with st.container(horizontal=True):
        st.metric(
            "Portfolio value",
            f"₹{summary['total_current']:,.0f}",
            f"{metrics['overall_return_pct']:+.1f}%",
            border=True,
        )
        st.metric(
            "Total P&L",
            f"₹{summary['pl_amount']:,.0f}",
            f"{metrics['in_profit']} up / {metrics['in_loss']} down",
            border=True,
        )
        st.metric(
            "Top 3 weight",
            f"{metrics['top3_weight']:.1f}%",
            metrics["concentration_label"],
            border=True,
        )
        st.metric(
            "Holdings",
            metrics["holding_count"],
            f"HHI {metrics['hhi']:.3f}",
            border=True,
        )

    st.space("small")
    render_research_brief(brief)

    with st.container(border=True):
        st.markdown("**:material/auto_awesome: AI deep dive**")
        st.caption("Open **Insights → Research chat** for briefing, holding research, and agent Q&A.")

    pui.section("Allocation", "Stock weights, sectors, and concentration", icon="pie_chart")

    stock_tab, sector_tab = st.tabs(["By stock", "By sector"])

    with stock_tab:
        stock_left, stock_right = st.columns([1, 1])
        with stock_left:
            with st.container(border=True):
                st.altair_chart(pchart.allocation_donut(merged), width="stretch")
        with stock_right:
            with st.container(border=True):
                with st.container(horizontal=True):
                    st.metric(
                        "Largest",
                        metrics["largest_symbol"],
                        f"{metrics['largest_weight']:.1f}%",
                        border=True,
                    )
                    st.metric(
                        "Top 3",
                        f"{metrics['top3_weight']:.1f}%",
                        metrics["concentration_label"],
                        border=True,
                    )
                st.altair_chart(pchart.stock_concentration_bars(merged), width="stretch")

        with st.container(border=True):
            st.altair_chart(pchart.cumulative_concentration(merged), width="stretch")

        st.dataframe(
            merged[["Symbol", "Weight %", "Current Value", "P&L", "Return %", "Cumulative weight %"]]
            .head(20)
            .round(2),
            width="stretch",
            hide_index=True,
            column_config={
                "Weight %": st.column_config.ProgressColumn(
                    "Weight",
                    format="%.1f%%",
                    min_value=0,
                    max_value=max(25, float(merged["Weight %"].max()) + 5),
                ),
                "Current Value": st.column_config.NumberColumn(format="₹%.0f"),
                "P&L": st.column_config.NumberColumn(format="₹%.0f"),
                "Return %": st.column_config.NumberColumn(format="%.1f%%"),
                "Cumulative weight %": st.column_config.NumberColumn("Cumul. %", format="%.1f%%"),
            },
        )

    with sector_tab:
        if sector_df is None or sector_df.empty:
            st.info("Load market data above to see sector-wise allocation.")
        else:
            sec_left, sec_right = st.columns([1, 1])
            with sec_left:
                with st.container(border=True):
                    st.altair_chart(pchart.sector_donut(sector_df), width="stretch")
            with sec_right:
                with st.container(border=True):
                    st.altair_chart(pchart.sector_weight_bar(sector_df), width="stretch")

            known = sector_df[sector_df["Sector"] != "Unknown"]
            if not known.empty:
                top_sec = known.iloc[0]
                with st.container(horizontal=True):
                    st.metric(
                        "Largest sector",
                        top_sec["Sector"],
                        f"{top_sec['Weight %']:.1f}%",
                        border=True,
                    )
                    st.metric("Sectors held", len(known), border=True)
                    if coverage:
                        st.metric(
                            "Sector coverage",
                            f"{coverage['sector_mapped_pct']}%",
                            f"{coverage['sector_mapped']}/{coverage['total']} mapped",
                            border=True,
                        )

            st.dataframe(
                sector_df[
                    ["Sector", "Weight %", "Current Value", "P&L", "Return %", "Holdings", "Stocks"]
                ].round(2),
                width="stretch",
                hide_index=True,
                column_config={
                    "Weight %": st.column_config.ProgressColumn(
                        "Weight",
                        format="%.1f%%",
                        min_value=0,
                        max_value=max(30, float(sector_df["Weight %"].max()) + 5),
                    ),
                    "Current Value": st.column_config.NumberColumn(format="₹%.0f"),
                    "P&L": st.column_config.NumberColumn(format="₹%.0f"),
                    "Return %": st.column_config.NumberColumn(format="%.1f%%"),
                },
            )

    with st.expander("Broker split & win/loss", expanded=False):
        plat_left, plat_right = st.columns(2)
        with plat_left:
            st.altair_chart(pchart.platform_donut(portfolio_df), width="stretch")
        with plat_right:
            st.altair_chart(pchart.profit_loss_split(metrics), width="stretch")

    pui.section("Vs market", "Portfolio return compared with Nifty 50", icon="compare_arrows")

    cmp_left, cmp_right = st.columns([1.4, 1])
    with cmp_left:
        with st.container(border=True):
            if benchmark and not benchmark.get("chart_df", pd.DataFrame()).empty:
                st.altair_chart(pchart.nifty_vs_portfolio_line(benchmark["chart_df"]), width="stretch")
            else:
                st.info("Could not load Nifty comparison right now.")
    with cmp_right:
        with st.container(border=True):
            if benchmark and benchmark.get("portfolio_return") is not None:
                alpha = benchmark.get("alpha")
                st.metric(
                    "Your portfolio",
                    f"{benchmark['portfolio_return']:+.1f}%",
                    f"~{benchmark.get('days', 30)}d",
                    border=True,
                )
                st.metric(
                    "Nifty 50",
                    f"{benchmark['nifty_return']:+.1f}%"
                    if benchmark.get("nifty_return") is not None
                    else "—",
                    border=True,
                )
                st.metric(
                    "Difference",
                    f"{alpha:+.1f}%" if alpha is not None else "—",
                    border=True,
                )
            else:
                st.info("Return comparison unavailable.")

    if sector_df is not None and not sector_df.empty:
        sector_vs = pbench.build_sector_vs_nifty(sector_df)
        if not sector_vs.empty:
            st.markdown("#### Sector tilt vs Nifty")
            vs_left, vs_right = st.columns(2)
            with vs_left:
                with st.container(border=True):
                    st.altair_chart(pchart.sector_vs_nifty_grouped(sector_vs), width="stretch")
            with vs_right:
                with st.container(border=True):
                    st.altair_chart(pchart.active_weight_bar(sector_vs), width="stretch")

            st.dataframe(
                sector_vs.round(2),
                width="stretch",
                hide_index=True,
                column_config={
                    "Your portfolio %": st.column_config.NumberColumn(format="%.1f%%"),
                    "Nifty 50 %": st.column_config.NumberColumn(format="%.1f%%"),
                    "Active weight %": st.column_config.NumberColumn(format="%+.1f%%"),
                },
            )
    else:
        st.info("Load market data to compare your sector mix against Nifty 50.")

    render_portfolio_risk_section(merged)

    forensic_checks = render_forensic_section(merged, enriched)
    checks.extend(forensic_checks)

    inst_checks = render_screener_upload(merged)
    checks.extend(inst_checks)

    if enriched is not None and sector_df is not None and coverage is not None:
        with st.expander("Sector performance & stock info", expanded=False):
            with st.container(border=True):
                st.altair_chart(pchart.sector_return_bar(sector_df), width="stretch")
            render_stock_info_table(enriched, coverage)

    pui.section("Performance", "Returns, P&L drivers, and weight map", icon="query_stats")
    perf_left, perf_right = st.columns(2)
    with perf_left:
        with st.container(border=True):
            st.altair_chart(pchart.return_weight_scatter(merged), width="stretch")
    with perf_right:
        with st.container(border=True):
            st.altair_chart(pchart.pnl_waterfall(merged), width="stretch")

    show_quadrants = st.toggle("Show performance quadrants", value=False)
    if show_quadrants:
        quadrant_meta = [
            ("Core winners", quadrants["Core winners"], ":green-badge[Core]"),
            ("Core losers", quadrants["Core losers"], ":red-badge[Core]"),
            ("Small winners", quadrants["Small winners"], ":green-badge[Small]"),
            ("Low-impact losers", quadrants["Low-impact losers"], ":orange-badge[Small]"),
        ]
        quad_col1, quad_col2, quad_col3, quad_col4 = st.columns(4, border=True)
        for col, (title, symbols, _badge) in zip([quad_col1, quad_col2, quad_col3, quad_col4], quadrant_meta):
            with col:
                st.markdown(f"**{title}**")
                if symbols:
                    st.markdown(", ".join(f"**{s}**" for s in symbols))
                else:
                    st.caption("None in this bucket")

    pui.section("Full weightage", icon="table_chart")
    weight_table = pan.format_weight_table(merged)
    st.dataframe(
        weight_table,
        width="stretch",
        hide_index=True,
        column_config={
            "Symbol": st.column_config.TextColumn("Stock", width="small"),
            "Platforms": st.column_config.TextColumn("Platforms", width="medium"),
            "Weight %": st.column_config.ProgressColumn(
                "Weight",
                format="%.1f%%",
                min_value=0,
                max_value=max(20, float(weight_table["Weight %"].max()) + 2),
            ),
            "Cumulative weight %": st.column_config.NumberColumn("Cumulative %", format="%.1f"),
            "Invested Value": st.column_config.NumberColumn("Invested", format="₹%.0f"),
            "Current Value": st.column_config.NumberColumn("Current", format="₹%.0f"),
            "P&L": st.column_config.NumberColumn("P&L", format="₹%.0f"),
            "Return %": st.column_config.NumberColumn("Return", format="%.2f%%"),
            "P&L contribution %": st.column_config.NumberColumn("P&L share", format="%.1f%%"),
        },
    )

    top_left, top_mid, top_right = st.columns(3)
    perf_tables = [
        ("Top gainers", merged.nlargest(5, "Return %")),
        ("Top losers", merged.nsmallest(5, "Return %")),
        (
            "Largest P&L impact",
            merged.reindex(merged["P&L"].abs().sort_values(ascending=False).index),
        ),
    ]
    for col, (title, data) in zip([top_left, top_mid, top_right], perf_tables):
        with col:
            with st.container(border=True):
                st.markdown(f"**{title}**")
                st.dataframe(
                    data.head(5)[["Symbol", "P&L", "Return %", "Weight %"]],
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "P&L": st.column_config.NumberColumn(format="₹%.0f"),
                        "Return %": st.column_config.NumberColumn(format="%.1f%%"),
                        "Weight %": st.column_config.NumberColumn(format="%.1f%%"),
                    },
                )

    st.space("small")
    render_portfolio_checks(checks)


def resolve_agent_api_keys() -> dict[str, str | None]:
    """Prefer Streamlit secrets, then fall back to .env via stock_analyzer."""
    return san.resolve_api_keys(
        {
            "alpha": get_secret("alpha_vantage.api_key") or get_secret("ALPHA_VANTAGE_API_KEY"),
            "groq": get_secret("groq.api_key") or get_secret("GROQ_API_KEY"),
            "xai": get_secret("xai.api_key") or get_secret("XAI_API_KEY"),
            "gemini": get_secret("gemini.api_key") or get_secret("GEMINI_API_KEY"),
        }
    )


def _ai_badge(label: str, key: str | None, expected: str | None = None) -> None:
    provider = pai.detect_key_provider(key or "")
    if provider == "none":
        st.badge(label, icon=":material/close:", color="orange")
        return
    if expected and provider != expected and provider != "unknown":
        st.badge(f"{label} (wrong key type)", icon=":material/warning:", color="orange")
        return
    st.badge(label, icon=":material/check:", color="green")


def render_research_terminal(merged: pd.DataFrame) -> dict | None:
    """OpenBB research terminal — free Bloomberg-style fundamentals layer."""
    pui.section("Research terminal", "OpenBB fundamentals and news pulse", icon="candlestick_chart")
    if not pterm.is_available():
        reason = pterm.unavailability_reason()
        st.info(
            f"OpenBB research terminal is unavailable here. {reason} "
            "The rest of the dashboard works without it."
        )
        return None

    st.caption(
        "Powered by OpenBB (open-source) + Yahoo Finance. "
        "Adds P/E, PEG, margins, growth, ROE, D/E, Nifty pulse, and headlines into AI analysis."
    )
    load = st.button("Load terminal data", key="load_terminal_data")
    symbols = tuple(merged["Symbol"].tolist())
    news_symbols = tuple(merged.head(5)["Symbol"].tolist())

    if load:
        pterm.build_terminal_snapshot.clear()

    snapshot = None
    if load or st.session_state.get("terminal_snapshot_key") == symbols:
        with st.spinner("Fetching OpenBB fundamentals and market pulse..."):
            snapshot = pterm.build_terminal_snapshot(symbols, news_symbols)
        st.session_state.terminal_snapshot_key = symbols
        st.session_state.terminal_snapshot = snapshot
    elif "terminal_snapshot" in st.session_state:
        snapshot = st.session_state.terminal_snapshot

    if not snapshot or not snapshot.get("available"):
        st.caption("Click **Load terminal data** to pull fundamentals for your holdings.")
        return None

    pulse = snapshot.get("pulse") or {}
    if pulse.get("prev_close"):
        with st.container(horizontal=True):
            st.metric("Nifty 50", f"{pulse['prev_close']:,.0f}", border=True)
            st.metric("52w high", f"{pulse.get('year_high', 0):,.0f}", border=True)
            st.metric("52w low", f"{pulse.get('year_low', 0):,.0f}", border=True)
            st.metric("50d MA", f"{pulse.get('ma_50d', 0):,.0f}", border=True)

    df = pterm.metrics_dataframe(snapshot, merged)
    if not df.empty:
        show = df[
            [
                "Symbol",
                "Weight %",
                "pe_ratio",
                "forward_pe",
                "peg_ratio",
                "profit_margin",
                "revenue_growth",
                "earnings_growth",
                "return_on_equity",
                "debt_to_equity",
                "price_return_1y",
            ]
        ].copy()
        for col in ("profit_margin", "revenue_growth", "earnings_growth", "return_on_equity", "price_return_1y"):
            if col in show.columns:
                show[col] = show[col] * 100
        show = show.rename(
            columns={
                "pe_ratio": "P/E",
                "forward_pe": "Fwd P/E",
                "peg_ratio": "PEG",
                "profit_margin": "Margin",
                "revenue_growth": "Rev growth",
                "earnings_growth": "Earn growth",
                "return_on_equity": "ROE",
                "debt_to_equity": "D/E",
                "price_return_1y": "1y return",
            }
        )
        st.dataframe(
            show.round(3),
            width="stretch",
            hide_index=True,
            column_config={
                "Weight %": st.column_config.NumberColumn(format="%.1f%%"),
                "Margin": st.column_config.NumberColumn(format="%.1f%%"),
                "Rev growth": st.column_config.NumberColumn(format="%.1f%%"),
                "Earn growth": st.column_config.NumberColumn(format="%.1f%%"),
                "ROE": st.column_config.NumberColumn(format="%.1f%%"),
                "1y return": st.column_config.NumberColumn(format="%.1f%%"),
            },
        )

    news = snapshot.get("news") or {}
    if news:
        with st.expander("Recent headlines (top holdings)", expanded=False):
            for sym, headlines in news.items():
                for headline in headlines:
                    st.markdown(f"- **{sym}** — {headline}")

    return snapshot


def render_ai_research_studio(
    merged: pd.DataFrame,
    summary: dict,
    metrics: dict,
    portfolio_df: pd.DataFrame,
) -> None:
    pui.section(
        "AI research studio",
        "Briefings, deep holding notes, and transcript audit",
        icon="auto_awesome",
    )
    st.caption(
        "OpenBB + Yahoo + Nifty context feed every AI memo. Generate for the full book or one holding."
    )

    terminal_snapshot = render_research_terminal(merged)

    keys = resolve_agent_api_keys()
    openai_key = get_secret("openai.api_key")
    has_ai = bool(keys.get("gemini") or keys.get("xai") or keys.get("groq") or openai_key)

    status = st.container(horizontal=True)
    with status:
        _ai_badge("Gemini", keys.get("gemini"), expected="gemini")
        _ai_badge("xAI / Grok", keys.get("xai"), expected="xai")
        _ai_badge("Groq", keys.get("groq"), expected="groq")
        _ai_badge("OpenAI", openai_key, expected="openai")

    if not keys.get("groq"):
        st.info(
            "**Groq not configured yet.** Put your real Groq key (starts with `gsk_`) yourself into "
            "`.streamlit/secrets.toml` as `[groq] api_key = \"gsk_...\"` "
            "or into `.env` as `GROQ_API_KEY=gsk_...`. "
            "Do **not** paste keys in chat. Do **not** put an `xai-` key under Groq — that belongs in `[xai]`."
        )
    elif pai.detect_key_provider(keys.get("groq") or "") == "xai":
        st.warning(
            "Your Groq slot still has an **xAI** key (`xai-...`). "
            "Move it to `[xai]` and put a `gsk_...` key under `[groq]` in secrets.toml or `.env`."
        )
    elif pai.detect_key_provider(keys.get("groq") or "") == "groq":
        st.caption("Groq key looks valid (`gsk_`). Fallback order: Gemini Flash → xAI → Groq → OpenAI.")

    if keys.get("xai") and not keys.get("groq"):
        st.caption(
            "xAI/Grok is configured as a fallback. Note: new xAI teams often need credits before API calls work."
        )

    if not has_ai:
        st.warning(
            "Add at least one AI key in `.streamlit/secrets.toml`:\n\n"
            "`[gemini] api_key = ...` or `[xai] api_key = ...` or `[groq] api_key = \"gsk_...\"`"
        )
        return

    brief_col, research_col = st.columns(2)

    with brief_col:
        with st.container(border=True):
            st.markdown("**Portfolio briefing**")
            st.caption("Executive memo: health, risks, rebalancing ideas, watchlist.")
            if st.button("Generate AI briefing", type="primary", width="stretch", key="gen_ai_brief"):
                with st.spinner("Writing portfolio briefing with AI..."):
                    result = pai.generate_portfolio_briefing(
                        merged,
                        summary,
                        metrics,
                        portfolio_df,
                        gemini_key=keys.get("gemini") or "",
                        groq_key=keys.get("groq") or "",
                        xai_key=keys.get("xai") or "",
                        openai_key=openai_key,
                        terminal_snapshot=terminal_snapshot,
                        **analysis_extras(),
                    )
                st.session_state.ai_portfolio_briefing = result
                st.session_state.chat_messages.append(
                    {"role": "user", "content": "Generate a full AI portfolio briefing."}
                )
                st.session_state.chat_messages.append(
                    {
                        "role": "assistant",
                        "content": f"*AI briefing via {result['provider']}*\n\n{result['text']}",
                    }
                )

    with research_col:
        with st.container(border=True):
            st.markdown("**Deep holding research**")
            st.caption("Fundamentals, 52w positioning, portfolio role, and hold/trim/add with triggers.")
            symbols = merged["Symbol"].tolist()
            default_index = 0
            largest = metrics.get("largest_symbol")
            if largest in symbols:
                default_index = symbols.index(largest)
            research_symbol = st.selectbox(
                "Holding to research",
                options=symbols,
                index=default_index,
                key="ai_research_symbol",
            )
            include_tech = st.toggle("Include 30d price + RSI", value=True, key="ai_research_tech")
            if st.button("Research holding", width="stretch", key="gen_ai_holding"):
                with st.spinner(f"Researching {research_symbol} with AI..."):
                    result = pai.research_holding_deep(
                        research_symbol,
                        merged,
                        summary,
                        metrics,
                        portfolio_df=portfolio_df,
                        gemini_key=keys.get("gemini") or "",
                        groq_key=keys.get("groq") or "",
                        xai_key=keys.get("xai") or "",
                        openai_key=openai_key,
                        alpha_key=keys.get("alpha") or "",
                        include_technicals=include_tech,
                    )
                st.session_state.ai_holding_research = result
                st.session_state.chat_messages.append(
                    {"role": "user", "content": f"Deep research on {research_symbol}"}
                )
                st.session_state.chat_messages.append(
                    {
                        "role": "assistant",
                        "content": f"*AI research via {result['provider']}*\n\n{result['text']}",
                    }
                )

    if st.session_state.ai_portfolio_briefing:
        with st.container(border=True):
            provider = st.session_state.ai_portfolio_briefing["provider"]
            paidisp.render_ai_response(
                f"*Portfolio briefing · {provider}*\n\n{st.session_state.ai_portfolio_briefing['text']}"
            )

    if st.session_state.ai_holding_research:
        with st.container(border=True):
            note = st.session_state.ai_holding_research
            paidisp.render_ai_response(
                f"*Holding research · `{note['symbol']}` · {note['provider']}*\n\n{note['text']}"
            )

    st.divider()
    render_transcript_auditor(merged, summary, metrics, portfolio_df, keys, openai_key)


def render_multi_agent_analyzer(merged: pd.DataFrame) -> None:
    pui.section(
        "Stock analyzer",
        "Prices → RSI → sentiment → AI summary",
        icon="psychology",
    )
    st.caption("Pipeline uses yfinance, Alpha Vantage, Groq (`gsk_`), and Gemini.")

    keys = resolve_agent_api_keys()
    key_status = st.container(horizontal=True)
    with key_status:
        _ai_badge("Alpha Vantage", keys.get("alpha"), expected=None)
        _ai_badge("Gemini", keys.get("gemini"), expected="gemini")
        _ai_badge("xAI / Grok", keys.get("xai"), expected="xai")
        _ai_badge("Groq", keys.get("groq"), expected="groq")

    if not keys.get("groq"):
        st.caption(
            "Sentiment needs a real Groq key (`gsk_`) in `[groq]` or `GROQ_API_KEY`. "
            "xAI (`xai-`) can stand in if Groq is empty, but only when that account has credits."
        )

    symbols = merged["Symbol"].tolist()
    if not symbols:
        st.info("No holdings available to analyze.")
        return

    default_index = 0
    for preferred in ("RELIANCE", "GMDCLTD", "BEL"):
        if preferred in symbols:
            default_index = symbols.index(preferred)
            break

    pick_col, run_col = st.columns([3, 1])
    with pick_col:
        selected = st.selectbox(
            "Select a holding",
            options=symbols,
            index=default_index,
            key="agent_symbol_select",
        )
    with run_col:
        st.write("")
        st.write("")
        run_clicked = st.button("Run analysis", type="primary", width="stretch")

    if run_clicked:
        yahoo_ticker = san.to_yahoo_ticker(selected)
        with st.spinner(f"Running multi-agent pipeline for {yahoo_ticker}..."):
            result = san.analyze_stock(
                selected,
                alpha_key=keys.get("alpha"),
                # Prefer real Groq (gsk_); fall back to xAI only if Groq is missing.
                groq_key=keys.get("groq") or keys.get("xai"),
                gemini_key=keys.get("gemini"),
                quiet=True,
            )
        st.session_state.agent_analysis = result
        st.session_state.agent_analysis_ticker = selected

        # Also drop a summary into the chat history for continuity.
        report = san.format_analysis_markdown(result)
        st.session_state.chat_messages.append(
            {"role": "user", "content": f"Run multi-agent analysis on {selected}"}
        )
        st.session_state.chat_messages.append({"role": "assistant", "content": report})

    result = st.session_state.agent_analysis
    if result and st.session_state.agent_analysis_ticker:
        price = result.get("price_summary") or {}
        metric_row = st.container(horizontal=True)
        with metric_row:
            if price:
                st.metric(
                    "30d change",
                    f"{price.get('pct_change_30d', 0):+.2f}%",
                    price.get("trend", "—"),
                    border=True,
                )
                st.metric("Last close", f"₹{price.get('end_close', 0):,.2f}", border=True)
            rsi = result.get("rsi")
            st.metric("RSI (14)", f"{rsi:.1f}" if rsi is not None else "—", border=True)
            sentiment = result.get("sentiment")
            st.metric(
                "Sentiment",
                f"{sentiment:+.2f}" if sentiment is not None else "—",
                border=True,
            )

        analysis = result.get("gemini_analysis")
        if analysis:
            paidisp.render_ai_response(analysis, show_meta=False)
        else:
            st.caption("_No AI assessment generated._")

        warnings = result.get("warnings") or []
        if warnings:
            with st.expander("Pipeline notes"):
                for warning in warnings:
                    st.markdown(f"- {warning}")

        if result.get("headlines"):
            with st.expander("Headlines used for sentiment"):
                for headline in result["headlines"]:
                    st.markdown(f"- {headline}")


def render_insights_chat(
    merged: pd.DataFrame,
    summary: dict,
    metrics: dict,
    portfolio_df: pd.DataFrame,
) -> None:
    keys = resolve_agent_api_keys()
    openai_key = get_secret("openai.api_key")
    openai_model = get_secret("openai.model", "gpt-4o-mini")
    has_ai = bool(keys.get("gemini") or keys.get("xai") or keys.get("groq") or openai_key)

    render_ai_research_studio(merged, summary, metrics, portfolio_df)
    st.space("medium")
    render_multi_agent_analyzer(merged)
    st.space("medium")

    pui.section("Portfolio chat", "Ask questions — agent mode for complex research", icon="chat")
    if has_ai:
        if keys.get("gemini"):
            provider = "Gemini Flash (with fallbacks)"
        elif keys.get("xai"):
            provider = "xAI / Grok"
        elif keys.get("groq"):
            provider = "Groq"
        else:
            provider = f"OpenAI ({openai_model})"
        st.caption(
            f"AI chat enabled via {provider}. Complex questions use an **agent** "
            "(Groq plans tools → live data → Gemini/Groq answer). Simple questions stay fast."
        )
    else:
        st.caption(
            "Add keys in `.streamlit/secrets.toml`: `[gemini]`, `[xai]` (`xai-...`), "
            "and/or `[groq]` (`gsk_...`). Basic offline answers still work for a few questions."
        )

    if st.button("Clear chat"):
        st.session_state.chat_messages = []
        st.rerun()

    if not st.session_state.chat_messages:
        selected = st.pills(
            "Try asking",
            list(pchat.SUGGESTED_QUESTIONS.keys()),
            label_visibility="collapsed",
        )
        if selected:
            st.session_state.chat_messages.append(
                {"role": "user", "content": pchat.SUGGESTED_QUESTIONS[selected]}
            )
            st.rerun()

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                paidisp.render_chat_assistant(msg["content"])
            else:
                st.markdown(msg["content"])

    if prompt := st.chat_input(
        "Ask for briefing, risks, rebalancing ideas, or research a stock...",
        submit_mode="disable",
    ):
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Researching with portfolio agent..."):
                response = pchat.get_response(
                    prompt,
                    st.session_state.chat_messages,
                    merged,
                    summary,
                    metrics,
                    portfolio_df=portfolio_df,
                    gemini_key=keys.get("gemini") or "",
                    groq_key=keys.get("groq") or "",
                    xai_key=keys.get("xai") or "",
                    openai_key=openai_key,
                    openai_model=openai_model,
                    **analysis_extras(),
                )
            paidisp.render_chat_assistant(response)

        st.session_state.chat_messages.append({"role": "assistant", "content": response})


def render_insights_tab(portfolio_df: pd.DataFrame, summary: dict) -> None:
    total_current = summary["total_current"]
    if total_current <= 0:
        st.warning("Portfolio value is zero, so insights cannot be calculated.")
        return

    merged = pan.merge_holdings(portfolio_df, total_current)
    metrics = pan.compute_metrics(merged, summary)

    insight_view = st.segmented_control(
        "Insights view",
        ["Analysis", "Research chat"],
        default="Analysis",
        label_visibility="collapsed",
    )
    st.space("small")

    if insight_view == "Analysis":
        render_insights_analysis(merged, metrics, portfolio_df, summary)
    else:
        render_insights_chat(merged, summary, metrics, portfolio_df)


# --- Sidebar ---
pui.sidebar_block("Data sources", icon="cloud_sync")

with st.sidebar.container(border=True):
    st.markdown("**:material/link: Zerodha (live)**")
    default_api_key = get_secret("zerodha.api_key")
    default_api_secret = get_secret("zerodha.api_secret")
    default_redirect_url = get_secret("zerodha.redirect_url")
    kite_redirect = zauth.redirect_url(default_redirect_url)

    api_key = st.text_input("API key", value=default_api_key, type="password")
    st.text_input(
        "API secret",
        key="zerodha_api_secret",
        type="password",
        help="Permanent app secret from developers.kite.trade — not the daily access token.",
    )
    effective_secret = zauth.resolve_api_secret(default_api_secret, st.session_state.zerodha_api_secret)
    cred_error = zauth.validate_credentials(api_key, effective_secret)

    if st.session_state.zerodha_auth_success:
        st.success(st.session_state.zerodha_auth_success)
    if st.session_state.zerodha_auth_error:
        st.error(st.session_state.zerodha_auth_error)
    elif cred_error:
        st.error(cred_error)
    else:
        st.caption(f"Redirect URL in Kite Connect app: `{kite_redirect}`")
        if api_key.strip():
            st.link_button(
                "Connect Zerodha",
                zauth.login_url(api_key),
                use_container_width=True,
            )

    st.caption(zauth.token_status_caption(st.session_state.zerodha_access_token))

    with st.expander("Manual token exchange"):
        with st.form("zerodha_token_form", clear_on_submit=False):
            request_token = st.text_input("Request token or redirect URL")
            submitted = st.form_submit_button("Generate access token")
        if submitted:
            if not api_key.strip() or not effective_secret or not request_token.strip():
                st.error("API key, API secret, and request token are all required.")
            else:
                try:
                    st.session_state.zerodha_access_token = zauth.generate_access_token(
                        api_key, effective_secret, request_token
                    )
                    st.session_state.zerodha_auth_success = (
                        "Access token ready for today. Valid until 6:00 AM IST tomorrow."
                    )
                    st.session_state.zerodha_auth_error = ""
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))

    access_token = st.text_input(
        "Access token",
        key="zerodha_access_token",
        type="password",
    )

with st.sidebar.container(border=True):
    st.markdown("**:material/upload_file: Groww (CSV)**")
    groww_file = st.file_uploader(
        "Holdings export",
        type=["csv", "xlsx"],
        label_visibility="collapsed",
    )

st.sidebar.space("small")
if st.sidebar.button("Calculate live wealth", type="primary", use_container_width=True):
    calculate_portfolio(api_key, access_token, groww_file)

if st.sidebar.button("Save today's snapshot", use_container_width=True):
    if st.session_state.portfolio_summary is None:
        st.sidebar.warning("Calculate your portfolio first.")
    else:
        summary = st.session_state.portfolio_summary
        saved_date = ph.save_snapshot(
            total_invested=summary["total_invested"],
            total_current=summary["total_current"],
            pl_amount=summary["pl_amount"],
            zerodha_current=summary["zerodha_current"],
            groww_current=summary["groww_current"],
            holding_count=summary["holding_count"],
        )
        st.sidebar.success(f"Snapshot saved for {saved_date}.")

last_snapshot = ph.get_last_snapshot_date()
if last_snapshot:
    st.sidebar.caption(f"Last snapshot · {last_snapshot.strftime('%d %b %Y')}")

# --- Main navigation ---
main_view = st.segmented_control(
    "Dashboard view",
    ["Portfolio", "Trends", "Insights"],
    default="Portfolio",
    label_visibility="collapsed",
)

pui.page_header(main_view, st.session_state.portfolio_summary)
st.space("small")

if main_view == "Portfolio":
    if st.session_state.portfolio_df is None:
        pui.empty_state(
            "No portfolio loaded",
            "Connect **Zerodha** and/or upload a **Groww** file in the sidebar.",
            icon="upload",
            hint="Then click **Calculate live wealth**.",
        )
    else:
        render_portfolio_tab(st.session_state.portfolio_df, st.session_state.portfolio_summary)
elif main_view == "Trends":
    render_trends_tab()
else:
    if st.session_state.portfolio_df is None:
        pui.empty_state(
            "Insights need portfolio data",
            "Calculate live wealth first to unlock allocation, benchmarks, and AI research.",
            icon="insights",
        )
    else:
        render_insights_tab(st.session_state.portfolio_df, st.session_state.portfolio_summary)
