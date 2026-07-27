from datetime import date

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
import realized_pnl as rpnl
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
if "ai_engine_analysis" not in st.session_state:
    st.session_state.ai_engine_analysis = None
if "risk_snapshot" not in st.session_state:
    st.session_state.risk_snapshot = None
if "fmp_forensic_snapshot" not in st.session_state:
    st.session_state.fmp_forensic_snapshot = None
if "institutional_df" not in st.session_state:
    st.session_state.institutional_df = None
if "transcript_audit" not in st.session_state:
    st.session_state.transcript_audit = None
if "realized_pnl" not in st.session_state:
    st.session_state.realized_pnl = rpnl.load_realized_state()


def _summary_for_header(summary: dict | None) -> dict | None:
    if not summary:
        return summary
    out = dict(summary)
    realized_state = st.session_state.get("realized_pnl") or {}
    if realized_state.get("row_count") or realized_state.get("realized_total"):
        economic = rpnl.combine_economic_pnl(float(out.get("pl_amount") or 0), realized_state)
        out["has_realized_pnl"] = True
        out["realized_total"] = economic["realized"]
        out["booked_losses"] = economic["booked_losses"]
        out["economic_pl"] = economic["economic"]
    return out


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


def _query_param(name: str) -> str:
    """Read a single query param as a clean string (Streamlit may return list-like values)."""
    try:
        value = st.query_params.get(name)
    except Exception:
        value = None
    if value is None:
        try:
            raw = st.query_params.to_dict().get(name)
            value = raw
        except Exception:
            return ""
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        value = value[0] if value else ""
    return str(value).strip()


# Keep session token in sync with today's cache. After 6:00 AM IST the cache is
# cleared — also drop any leftover session token so Refresh does not keep sending
# yesterday's access token to Kite (which surfaces as "token expired").
# Never wipe the session while a Kite redirect is still in the URL.
_cached_zerodha_token = zauth.load_cached_token()
_oauth_token_in_url = bool(_query_param("request_token"))
if _cached_zerodha_token:
    st.session_state.zerodha_access_token = _cached_zerodha_token
elif st.session_state.zerodha_access_token and not _oauth_token_in_url:
    st.session_state.zerodha_access_token = ""


def handle_zerodha_oauth_callback() -> None:
    """
    Auto-read request_token from the browser URL after Kite redirects back.
    Claim the token immediately so Streamlit's double-run cannot burn it twice.
    """
    request_token = zauth.normalize_request_token(_query_param("request_token"))
    if not request_token:
        return

    st.session_state.zerodha_oauth_pending = True
    handled = st.session_state.setdefault("zerodha_handled_request_tokens", set())
    if request_token in handled:
        # Already claimed this run-cycle (or earlier). Prefer cached/session token.
        st.session_state.zerodha_oauth_pending = False
        cached = zauth.load_cached_token()
        if cached:
            st.session_state.zerodha_access_token = cached
            st.session_state.zerodha_auth_error = ""
            if not st.session_state.get("zerodha_auth_success"):
                st.session_state.zerodha_auth_success = (
                    "Zerodha connected. Click Refresh portfolio."
                )
        try:
            st.query_params.clear()
        except Exception:
            pass
        return

    # Claim BEFORE calling Kite — request_token is one-time use.
    handled.add(request_token)

    status = _query_param("status")
    if status and status != "success":
        st.session_state.zerodha_auth_error = (
            f"Zerodha login failed ({status}). You are back in Wealthapp — "
            "click Connect Zerodha again when ready."
        )
        st.session_state.zerodha_oauth_pending = False
        try:
            st.query_params.clear()
        except Exception:
            pass
        return

    api_key = get_secret("zerodha.api_key") or str(
        st.session_state.get("zerodha_api_key") or ""
    ).strip()
    api_secret = zauth.resolve_api_secret(
        get_secret("zerodha.api_secret"),
        st.session_state.zerodha_api_secret,
    )
    cred_error = zauth.validate_credentials(api_key, api_secret)
    if cred_error:
        st.session_state.zerodha_auth_error = (
            f"{cred_error} Login redirect was received, but the access token could not be created."
        )
        st.session_state.zerodha_oauth_pending = False
        return

    try:
        access_token = zauth.generate_access_token(api_key, api_secret, request_token)
        st.session_state.zerodha_access_token = access_token
        st.session_state.zerodha_auth_success = (
            "Zerodha connected automatically from the redirect URL. Click Refresh portfolio."
        )
        st.session_state.zerodha_auth_error = ""
        st.session_state.zerodha_oauth_pending = False
        try:
            st.query_params.clear()
        except Exception:
            pass
        st.rerun()
    except Exception as exc:
        st.session_state.zerodha_oauth_pending = False
        # If a twin rerun already saved the token, don't scare the user.
        cached = zauth.load_cached_token() or st.session_state.get("zerodha_access_token")
        if cached and zauth.has_active_token(str(cached)):
            st.session_state.zerodha_access_token = str(cached)
            st.session_state.zerodha_auth_error = ""
            st.session_state.zerodha_auth_success = (
                "Zerodha connected. Click Refresh portfolio."
            )
            try:
                st.query_params.clear()
            except Exception:
                pass
            return
        st.session_state.zerodha_auth_error = (
            f"Got the redirect URL, but token exchange failed: {exc}. "
            "Click Connect Zerodha again, or paste the redirect URL in Fallback below."
        )


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
    total_invested = float(portfolio_df["Invested Value"].sum())
    total_current = float(portfolio_df["Current Value"].sum())
    by_owner_current = portfolio_df.groupby("Owner")["Current Value"].sum().to_dict()
    by_owner_invested = portfolio_df.groupby("Owner")["Invested Value"].sum().to_dict()
    by_owner_count = portfolio_df.groupby("Owner").size().to_dict()
    z_n = int(by_owner_count.get("Zerodha (Kite)", 0))
    g_n = int(by_owner_count.get("Groww", 0))
    return {
        "total_invested": total_invested,
        "total_current": total_current,
        "pl_amount": total_current - total_invested,
        "zerodha_current": float(by_owner_current.get("Zerodha (Kite)", 0.0)),
        "groww_current": float(by_owner_current.get("Groww", 0.0)),
        "zerodha_invested": float(by_owner_invested.get("Zerodha (Kite)", 0.0)),
        "groww_invested": float(by_owner_invested.get("Groww", 0.0)),
        "zerodha_holdings": z_n,
        "groww_holdings": g_n,
        "holding_count": z_n + g_n,
    }


def calculate_portfolio(
    api_key: str,
    access_token: str,
    groww_file,
    *,
    groww_as_of: date | None = None,
) -> None:
    holdings = []
    had_groww = False
    had_zerodha = False

    if groww_file:
        try:
            groww_rows = parse_groww_file(groww_file)
            holdings.extend(groww_rows)
            had_groww = bool(groww_rows)
            if had_groww:
                as_of = groww_as_of or ph.yesterday_ist()
                st.sidebar.success(
                    f"Groww file loaded (treated as as-of {as_of.isoformat()})."
                )
        except Exception as exc:
            st.sidebar.error(f"Error reading the Groww file: {exc}")

    if api_key and access_token:
        try:
            z_rows = fetch_zerodha_holdings(api_key, access_token)
            holdings.extend(z_rows)
            had_zerodha = bool(z_rows)
            st.sidebar.success("Zerodha holdings fetched (live).")
            st.session_state.zerodha_auth_error = ""
        except Exception as exc:
            message = str(exc)
            if "Connect Zerodha" in message:
                st.session_state.zerodha_auth_error = message
            else:
                st.sidebar.error(f"Failed to fetch Zerodha data: {exc}")
    elif api_key and not access_token:
        st.sidebar.info(
            "Zerodha API key is set, but today's login is missing. "
            "Connect Zerodha, then Refresh — Groww can still finalise yesterday (T+1)."
        )
    elif access_token and not api_key:
        st.sidebar.warning(
            "Zerodha access token is ready, but the API key is missing. "
            "Add `api_key` under `[zerodha]` in secrets, or enter it in the sidebar."
        )

    if not holdings:
        st.session_state.portfolio_df = None
        st.session_state.portfolio_summary = None
        st.warning("No holdings found. Upload a Groww file and/or connect Zerodha.")
        return

    portfolio_df = sym.normalize_portfolio_symbols(enrich_portfolio(pd.DataFrame(holdings)))
    st.session_state.portfolio_df = portfolio_df
    summary = build_summary(portfolio_df)
    st.session_state.portfolio_summary = summary
    pmd.clear_market_cache()

    # History: live Zerodha for today + Groww T+1 as-of (default yesterday) + carries.
    try:
        result = ph.save_summary_snapshot(
            summary,
            had_zerodha=had_zerodha,
            had_groww=had_groww,
            groww_as_of=groww_as_of or (ph.yesterday_ist() if had_groww else None),
        )
        st.session_state.last_auto_snapshot = result
        if result["saved"]:
            st.sidebar.success(result["reason"])
        elif result.get("skipped_today"):
            st.sidebar.info(result["reason"])
        else:
            st.sidebar.warning(result["reason"])
    except Exception as exc:
        st.sidebar.warning(f"Portfolio loaded, but daily snapshot failed: {exc}")


def wealth_trend_series(column: str = "total_current", points: int = 12) -> list[float] | None:
    """Recent snapshot values for metric sparklines."""
    try:
        history = ph.load_snapshots()
    except Exception:
        return None
    if history.empty or column not in history.columns or len(history) < 3:
        return None
    values = history[column].tail(points).astype(float).tolist()
    return values if len(values) >= 3 else None


def render_portfolio_ai_engine(portfolio_df: pd.DataFrame) -> None:
    """Interactive trigger for the modular Portfolio Analysis Engine (Insights)."""
    pui.section(
        "Portfolio scorecard",
        "Objective health and risk scores from your holdings, plus drivers, risks, and suggested actions",
    )
    keys = resolve_agent_api_keys()
    openai_key = get_secret("openai.api_key")
    has_ai = bool(keys.get("gemini") or keys.get("xai") or keys.get("groq") or openai_key)

    st.caption(
        "Price technicals are fetched via OpenBB (Yahoo Finance fallback). "
        "An AI model fills the driver, risk, and action tables below the scores."
    )

    if not has_ai:
        st.info(
            "Add `GROQ_API_KEY` / `GEMINI_API_KEY` (or `[groq]` / `[gemini]` in secrets) "
            "to generate the written scorecard sections."
        )
        return

    run = st.button(
        "Generate scorecard",
        type="primary",
        icon=":material/analytics:",
        width="stretch",
        key="run_portfolio_ai_engine",
    )
    if run:
        with st.spinner("Fetching technicals and generating the scorecard..."):
            result = pai.analyze_portfolio_engine(
                portfolio_df,
                gemini_key=keys.get("gemini") or "",
                groq_key=keys.get("groq") or "",
                xai_key=keys.get("xai") or "",
                openai_key=openai_key or "",
            )
        st.session_state.ai_engine_analysis = result

    if st.session_state.ai_engine_analysis:
        paidisp.render_engine_analysis(
            st.session_state.ai_engine_analysis,
            show_payload=True,
        )


def render_decision_snapshot(portfolio_df: pd.DataFrame, summary: dict) -> None:
    """Dense, decision-oriented portfolio snapshot for the Portfolio page."""
    total_current = float(summary.get("total_current") or 0)
    total_invested = float(summary.get("total_invested") or 0)
    pl = float(summary.get("pl_amount") or 0)
    ret = (pl / total_invested * 100) if total_invested else 0.0

    merged = pan.merge_holdings(portfolio_df, total_current)
    metrics = pan.compute_metrics(merged, summary)

    winners = int((portfolio_df["P&L"] > 0).sum())
    losers = int((portfolio_df["P&L"] < 0).sum())
    flat = int((portfolio_df["P&L"] == 0).sum())
    n = len(portfolio_df)
    profit_share = (winners / n * 100) if n else 0.0

    best = portfolio_df.loc[portfolio_df["Return %"].idxmax()]
    worst = portfolio_df.loc[portfolio_df["Return %"].idxmin()]
    largest = portfolio_df.loc[portfolio_df["Current Value"].idxmax()]
    largest_weight = float(metrics.get("largest_weight") or 0)
    top3 = float(metrics.get("top3_weight") or 0)
    concentration = metrics.get("concentration_label") or "—"

    by_pnl = portfolio_df.sort_values("P&L")
    top_drags = by_pnl.head(5)

    trend = wealth_trend_series("total_current")
    pl_trend = wealth_trend_series("pl_amount")

    pui.section(
        "Decision snapshot",
        "Capital, concentration, and the holdings that most affect portfolio P&L",
    )

    # --- Capital ---
    st.markdown("**Capital position**")
    realized_state = st.session_state.get("realized_pnl") or rpnl.empty_realized_state()
    economic = rpnl.combine_economic_pnl(pl, realized_state)
    with st.container(horizontal=True):
        st.metric(
            "Total invested",
            pui.format_inr_compact(total_invested),
            help="Sum of buy value across all holdings",
            border=True,
        )
        st.metric(
            "Current value",
            pui.format_inr_compact(total_current),
            help="Current market value of all holdings",
            border=True,
            chart_data=trend,
            chart_type="area",
        )
        st.metric(
            "Unrealised P&L",
            pui.format_inr_compact(pl),
            pui.format_pct(ret),
            help="Open holdings only (Current − Invested). Sold positions are not included.",
            border=True,
            chart_data=pl_trend,
            chart_type="bar",
        )
        st.metric(
            "Holdings",
            str(n),
            f"{len(portfolio_df['Owner'].unique())} brokers",
            delta_color="off",
            border=True,
        )

    # Booked (realised) P&L — without this, sells vanish and the book looks too green.
    st.markdown("**Booked + economic P&L**")
    has_realized = bool(realized_state.get("row_count") or realized_state.get("realized_total"))
    with st.container(horizontal=True):
        st.metric(
            "Booked (realised) P&L",
            pui.format_inr_compact(economic["realized"]) if has_realized else "—",
            (
                f"{realized_state.get('loss_count', 0)} loss / "
                f"{realized_state.get('gain_count', 0)} gain closes"
                if has_realized
                else "Upload Console/Groww P&L"
            ),
            delta_color="off",
            help="From sold trades in your broker P&L export — includes booked losses.",
            border=True,
        )
        st.metric(
            "Booked losses",
            pui.format_inr_compact(economic["booked_losses"]) if has_realized else "—",
            help="Sum of realised losses only (negative).",
            border=True,
        )
        st.metric(
            "Economic P&L",
            pui.format_inr_compact(economic["economic"]) if has_realized else pui.format_inr_compact(pl),
            (
                "Unrealised + booked"
                if has_realized
                else "Unrealised only until P&L file is loaded"
            ),
            delta_color="off",
            help="Unrealised on open book plus realised P&L from sells.",
            border=True,
        )

    if has_realized and realized_state.get("top_losses"):
        with st.expander("Booked losses (from P&L file)", expanded=False):
            loss_df = pd.DataFrame(realized_state["top_losses"])
            show_cols = [c for c in ["symbol", "quantity", "buy_value", "sell_value", "realized_pnl", "sell_date"] if c in loss_df.columns]
            st.dataframe(
                loss_df[show_cols],
                hide_index=True,
                column_config={
                    "symbol": "Symbol",
                    "quantity": st.column_config.NumberColumn("Qty", format="%.0f"),
                    "buy_value": st.column_config.NumberColumn("Buy value", format="₹%d"),
                    "sell_value": st.column_config.NumberColumn("Sell value", format="₹%d"),
                    "realized_pnl": st.column_config.NumberColumn("Booked P&L", format="₹%d"),
                    "sell_date": "Sell date",
                },
            )
            st.caption(
                f"Source: {realized_state.get('source') or 'P&L file'} · "
                f"{realized_state.get('filename') or '—'} · "
                f"imported {realized_state.get('imported_at') or '—'}"
            )
    elif not has_realized:
        st.caption(
            "Selling a loser removes it from holdings, so unrealised P&L can look all green. "
            "Upload a **Realised P&L** file in the sidebar (Zerodha Console or Groww) to show booked losses."
        )

    # --- Structure ---
    st.markdown("**Portfolio structure**")
    with st.container(horizontal=True):
        st.metric(
            "Largest holding",
            str(largest["Symbol"]),
            f"{largest_weight:.1f}% of portfolio",
            delta_color="off",
            help="Stock with the highest current market value",
            border=True,
        )
        st.metric(
            "Top 3 weight",
            f"{top3:.1f}%",
            f"Concentration: {concentration}",
            delta_color="off",
            help="Combined portfolio weight of the three largest holdings",
            border=True,
        )
        st.metric(
            "Holdings in profit",
            str(winners),
            f"{profit_share:.0f}% of holdings",
            delta_color="normal" if profit_share >= 50 else "inverse",
            delta_arrow="off",
            help="Number of holdings with positive unrealised P&L",
            border=True,
        )
        st.metric(
            "Holdings at a loss",
            str(losers),
            f"{flat} flat" if flat else f"{100 - profit_share:.0f}% of holdings",
            delta_color="off",
            help="Number of holdings with negative unrealised P&L",
            border=True,
        )

    # --- Alerts (full width, compact) ---
    st.markdown("**Priority alerts**")
    st.caption("Concentration and drawdown flags — review these before making changes")
    flags: list[str] = []
    if largest_weight >= 20:
        flags.append(
            f":red-badge[High concentration] **{largest['Symbol']}** is "
            f"**{largest_weight:.1f}%** of the portfolio (above 20%)."
        )
    elif largest_weight >= 15:
        flags.append(
            f":orange-badge[Watch concentration] **{largest['Symbol']}** is "
            f"**{largest_weight:.1f}%** of the portfolio (above 15%)."
        )
    if top3 >= 50:
        flags.append(
            f":orange-badge[Top-heavy] Top 3 holdings are **{top3:.1f}%** of portfolio value."
        )
    if profit_share < 40 and n >= 5:
        flags.append(
            f":orange-badge[Weak breadth] Only **{profit_share:.0f}%** of holdings are in profit "
            f"({winners} of {n})."
        )
    if float(worst["Return %"]) <= -25:
        flags.append(
            f":red-badge[Deep drawdown] **{worst['Symbol']}** is "
            f"**{pui.format_pct(float(worst['Return %']))}** unrealised."
        )
    if not flags:
        flags.append(
            ":green-badge[No alerts] No concentration or drawdown flags from current holdings."
        )
    with st.container(border=True):
        for flag in flags:
            st.markdown(f"- {flag}")

    # --- Performance extremes (aligned 3-column tables) ---
    st.markdown("**Performance extremes**")
    st.caption("Largest unrealised losses and strongest / weakest returns by holding")

    wmap = (
        merged.drop_duplicates("Symbol").set_index("Symbol")["Weight %"]
        if not merged.empty and "Weight %" in merged.columns
        else pd.Series(dtype=float)
    )

    def _weight_for(symbol: str) -> float:
        try:
            return float(wmap.loc[symbol])
        except Exception:
            return 0.0

    hi_col, lo_col = st.columns(2, gap="medium")
    with hi_col:
        with st.container(border=True):
            st.caption("Highest return")
            st.markdown(
                f"**{best['Symbol']}** &nbsp;"
                f"{pui.tone_badge(float(best['Return %']), pui.format_pct(float(best['Return %'])))}"
            )
            with st.container(horizontal=True):
                st.metric("Portfolio weight", f"{_weight_for(str(best['Symbol'])):.1f}%")
                st.metric("Unrealised P&L", pui.format_inr_compact(float(best["P&L"])))
    with lo_col:
        with st.container(border=True):
            st.caption("Lowest return")
            st.markdown(
                f"**{worst['Symbol']}** &nbsp;"
                f"{pui.tone_badge(float(worst['Return %']), pui.format_pct(float(worst['Return %'])))}"
            )
            with st.container(horizontal=True):
                st.metric("Portfolio weight", f"{_weight_for(str(worst['Symbol'])):.1f}%")
                st.metric("Unrealised P&L", pui.format_inr_compact(float(worst["P&L"])))

    mover_cfg = {
        "Symbol": st.column_config.TextColumn("Stock"),
        "Invested": st.column_config.TextColumn("Invested"),
        "P&L": st.column_config.TextColumn("P&L"),
        "Return %": st.column_config.NumberColumn("Return", format="%+.1f%%"),
        "Weight %": st.column_config.NumberColumn("Weight", format="%.1f%%"),
    }

    def _mover_table(source: pd.DataFrame) -> pd.DataFrame:
        out = source[["Symbol", "Invested Value", "P&L", "Return %"]].copy()
        out["Invested"] = out["Invested Value"].map(
            lambda v: pui.format_inr_compact(float(v)) if pd.notna(v) else "—"
        )
        out["P&L"] = out["P&L"].map(
            lambda v: pui.format_inr_compact(float(v)) if pd.notna(v) else "—"
        )
        out["Weight %"] = out["Symbol"].map(wmap)
        return out[["Symbol", "Invested", "P&L", "Return %", "Weight %"]]

    show_drags = _mover_table(top_drags)
    gain_df = _mover_table(portfolio_df.nlargest(5, "Return %"))
    loss_df = _mover_table(portfolio_df.nsmallest(5, "Return %"))

    col_loss, col_gain, col_weak = st.columns(3, gap="medium")
    with col_loss:
        with st.container(border=True, height="stretch"):
            st.markdown("**Largest unrealised losses**")
            st.dataframe(show_drags, hide_index=True, column_config=mover_cfg)
    with col_gain:
        with st.container(border=True, height="stretch"):
            st.markdown("**Highest returns**")
            st.dataframe(gain_df, hide_index=True, column_config=mover_cfg)
    with col_weak:
        with st.container(border=True, height="stretch"):
            st.markdown("**Lowest returns**")
            st.dataframe(loss_df, hide_index=True, column_config=mover_cfg)

    st.caption(
        "Next: **Insights → Scorecard** for full risk analysis · "
        "**Research → Fundamentals** before changing a large position."
    )


def render_portfolio_tab(portfolio_df: pd.DataFrame, summary: dict) -> None:
    if portfolio_df.empty:
        return

    render_decision_snapshot(portfolio_df, summary)

    pui.section("Value by broker", "Invested capital and current market value on each platform")
    platforms = list(portfolio_df["Owner"].unique())
    plat_cols = st.columns(min(len(platforms), 2) or 1)
    for idx, platform in enumerate(platforms):
        plat_df = portfolio_df[portfolio_df["Owner"] == platform]
        plat_invested = plat_df["Invested Value"].sum()
        plat_current = plat_df["Current Value"].sum()
        plat_pl = plat_current - plat_invested
        plat_ret = (plat_pl / plat_invested * 100) if plat_invested else 0
        share = plat_current / summary["total_current"] * 100 if summary["total_current"] else 0
        with plat_cols[idx % len(plat_cols)]:
            with st.container(border=True, height="stretch"):
                st.markdown(
                    f"**{platform}** &nbsp;{pui.tone_badge(plat_ret, pui.format_pct(plat_ret))}"
                )
                with st.container(horizontal=True):
                    st.metric("Total invested", pui.format_inr_compact(plat_invested))
                    st.metric(
                        "Current value",
                        pui.format_inr_compact(plat_current),
                        pui.format_inr_compact(plat_pl),
                    )
                st.caption(f"{len(plat_df)} holdings · {share:.0f}% of total portfolio value")

    pui.section("All holdings", "Sorted by current market value — filter by broker or stock symbol")
    table = portfolio_df.copy()
    total_value = table["Current Value"].sum()
    table["Weight %"] = table["Current Value"] / total_value * 100 if total_value else 0
    table = table.sort_values("Current Value", ascending=False)

    platform_options = ["All brokers", *platforms]
    filt_left, filt_right = st.columns([2, 1], vertical_alignment="bottom")
    with filt_left:
        platform_pick = st.pills(
            "Broker",
            platform_options,
            default="All brokers",
            key="portfolio_platform_filter",
        )
    with filt_right:
        search = st.text_input(
            "Search stock symbol",
            placeholder="e.g. RELIANCE",
            key="portfolio_symbol_search",
        )
    if platform_pick and platform_pick != "All brokers":
        table = table[table["Owner"] == platform_pick]
    if search and search.strip():
        q = search.strip().upper()
        table = table[table["Symbol"].astype(str).str.upper().str.contains(q, na=False)]

    display = table[
        [
            "Symbol",
            "Owner",
            "Quantity",
            "Buy Price",
            "Current Price",
            "Invested Value",
            "Current Value",
            "P&L",
            "Return %",
            "Weight %",
        ]
    ].copy()
    display["Invested"] = display["Invested Value"].map(
        lambda v: pui.format_inr_compact(float(v)) if pd.notna(v) else "—"
    )
    display["Current"] = display["Current Value"].map(
        lambda v: pui.format_inr_compact(float(v)) if pd.notna(v) else "—"
    )
    display["P&L display"] = display["P&L"].map(
        lambda v: pui.format_inr_compact(float(v)) if pd.notna(v) else "—"
    )
    display = display[
        [
            "Symbol",
            "Owner",
            "Quantity",
            "Buy Price",
            "Current Price",
            "Invested",
            "Current",
            "P&L display",
            "Return %",
            "Weight %",
        ]
    ].rename(columns={"P&L display": "P&L"})
    st.caption(f"Showing {len(display)} of {len(portfolio_df)} holdings · amounts in ₹ / L / Cr")
    with st.container(border=True):
        st.dataframe(
            display,
            hide_index=True,
            height=min(520, 48 + 36 * max(len(display), 4)),
            column_config={
                "Symbol": st.column_config.TextColumn(pinned=True),
                "Owner": st.column_config.TextColumn("Platform"),
                "Weight %": st.column_config.ProgressColumn(
                    "Weight",
                    format="%.1f%%",
                    min_value=0,
                    max_value=float(max(table["Weight %"].max() if len(table) else 1, 1)),
                ),
                "Invested": st.column_config.TextColumn("Invested"),
                "Current": st.column_config.TextColumn("Current"),
                "P&L": st.column_config.TextColumn("P&L"),
                "Return %": st.column_config.NumberColumn("Return", format="%+.1f%%"),
                "Buy Price": st.column_config.NumberColumn("Avg buy", format="₹%.2f"),
                "Current Price": st.column_config.NumberColumn("LTP", format="₹%.2f"),
            },
        )

    with st.expander("Line-by-line debug", icon=":material/bug_report:", expanded=False):
        debug_df = portfolio_df.copy()
        debug_df["Line total"] = debug_df["Quantity"] * debug_df["Current Price"]
        st.dataframe(debug_df, hide_index=True)


def render_trends_tab() -> None:
    history = ph.load_snapshots()
    if history.empty:
        pui.empty_state(
            "No daily history yet",
            "Trends builds from one snapshot per day. Refreshing the portfolio now saves today automatically.",
            icon="photo_camera",
            steps=[
                "Connect Zerodha (live) and/or upload Groww (T+1 as-of)",
                "Click **Refresh portfolio** — Zerodha→today, Groww→as-of day + carry",
                "Repeat each day to grow the chart",
            ],
        )
        return

    colors = pui.chart_colors()
    status = ph.history_status(lookback_days=30)
    if not status["today_saved"]:
        if status["after_close"]:
            pui.status_banner(
                "Today is not in history yet — Refresh after market close to save today's point. "
                "A Groww upload still finalises its as-of day even before that.",
                kind="info",
            )
        else:
            pui.status_banner(
                "Before 3:30 PM IST, Refresh updates live holdings and can finalise Groww "
                "on its as-of day, but today's history point waits until after close "
                "(or use Force-save).",
                kind="info",
            )
    elif status["missing_recent"] > 0:
        pui.status_banner(
            f"History has {status['count']} day(s). "
            f"{status['missing_recent']} day(s) missing in the last month — "
            "one Refresh after close per day keeps Trends honest.",
            kind="warning",
        )
    else:
        st.caption(
            "Each row is one IST calendar day. Zerodha is live for that day; "
            f"Groww is T+1 (finalised when the file arrives). {status['count']} day(s) so far."
        )

    monthly = ph.get_monthly_summary(history)
    last_saved = history["date"].max().date()
    days_since = (pd.Timestamp(ph.today_ist()) - pd.Timestamp(last_saved)).days
    latest_value = float(history["total_current"].iloc[-1])
    first_value = float(history["total_current"].iloc[0])
    since_start = (latest_value - first_value) / first_value * 100 if first_value else 0
    trend = history["total_current"].tail(12).astype(float).tolist()

    pui.kpi_row(
        [
            {
                "label": "Latest portfolio value",
                "value": pui.format_inr(latest_value),
                "delta": pui.format_pct(since_start) + " since first snapshot",
                "chart": trend if len(trend) >= 3 else None,
                "chart_type": "area",
                "help": "Total market value from your most recent saved snapshot",
            },
            {
                "label": "Snapshots saved",
                "value": str(len(history)),
                "delta": f"since {history['date'].min().strftime('%d %b %Y')}",
                "delta_color": "off",
                "help": "Number of portfolio snapshots stored for trend charts",
            },
            {
                "label": "Last snapshot date",
                "value": last_saved.strftime("%d %b"),
                "delta": f"{days_since} days ago",
                "delta_color": "inverse" if days_since > 7 else "off",
            },
            {
                "label": "Unrealised P&L",
                "value": pui.format_inr_compact(float(history["pl_amount"].iloc[-1])),
                "chart": history["pl_amount"].tail(12).astype(float).tolist()
                if len(history) >= 3
                else None,
                "chart_type": "bar",
                "help": "Unrealised profit or loss from the latest snapshot",
            },
        ]
    )

    if days_since > 7:
        pui.status_banner(
            "Your last snapshot is more than 7 days old. Save a new one for up-to-date trends.",
            kind="warning",
        )

    pui.section("Portfolio value over time", "Total invested capital vs current market value at each snapshot")
    value_chart_df = history.set_index("date")[["total_invested", "total_current"]].rename(
        columns={"total_invested": "Invested", "total_current": "Current value"}
    )
    with st.container(border=True):
        st.area_chart(
            value_chart_df,
            color=[colors["current"], colors["invested"]],
            stack=False,
        )

    if len(monthly) >= 2:
        pui.section(
            "Monthly change",
            "Portfolio value change between consecutive month-end snapshots",
        )
        growth_col1, growth_col2 = st.columns(2)
        with growth_col1:
            with st.container(border=True, height="stretch"):
                st.markdown("**Change in rupees**")
                st.caption("Absolute ₹ change from the previous month-end")
                change_df = (
                    monthly.set_index("date")[["monthly_change"]]
                    .dropna()
                    .rename(columns={"monthly_change": "Change (₹)"})
                )
                st.bar_chart(change_df, color=colors["change"])
        with growth_col2:
            with st.container(border=True, height="stretch"):
                st.markdown("**Change in percent**")
                st.caption("Percentage change from the previous month-end")
                pct_df = (
                    monthly.set_index("date")[["monthly_growth_pct"]]
                    .dropna()
                    .rename(columns={"monthly_growth_pct": "Growth (%)"})
                )
                st.bar_chart(pct_df, color=colors["growth"])
    else:
        st.caption("Save snapshots in at least two different months to unlock monthly change charts.")

    if "zerodha_current" in history.columns and history[["zerodha_current", "groww_current"]].sum().sum() > 0:
        pui.section("Value by broker over time", "Zerodha vs Groww market value at each snapshot")
        platform_df = history.set_index("date")[["zerodha_current", "groww_current"]].rename(
            columns={"zerodha_current": "Zerodha", "groww_current": "Groww"}
        )
        with st.container(border=True):
            st.area_chart(platform_df, color=[colors["groww"], colors["zerodha"]])

    pui.section("Snapshot history", "One row per day — sleeves shown separately")
    display_history = history.sort_values("date", ascending=False).copy()
    with st.container(border=True):
        st.dataframe(
            display_history,
            hide_index=True,
            column_config={
                "date": st.column_config.DatetimeColumn("Date", format="DD MMM YYYY", pinned=True),
                "total_invested": st.column_config.NumberColumn("Invested", format="₹%d"),
                "total_current": st.column_config.NumberColumn("Current", format="₹%d"),
                "pl_amount": st.column_config.NumberColumn("P&L", format="₹%d"),
                "zerodha_current": st.column_config.NumberColumn("Zerodha", format="₹%d"),
                "groww_current": st.column_config.NumberColumn("Groww", format="₹%d"),
                "zerodha_invested": None,
                "groww_invested": None,
                "zerodha_holdings": st.column_config.NumberColumn("Z holdings"),
                "groww_holdings": st.column_config.NumberColumn("G holdings"),
                "holding_count": st.column_config.NumberColumn("Holdings"),
                "created_at": None,
            },
        )


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
    pui.section(
        "Portfolio risk metrics",
        "Beta vs Nifty 50, Sharpe ratio, annualised return, and volatility from price history",
    )

    load_risk = st.button("Calculate risk metrics", key="load_risk_metrics")
    if load_risk:
        prisk.fetch_daily_returns.clear()

    if load_risk or st.session_state.risk_snapshot:
        if load_risk:
            with st.spinner("Downloading price history and computing risk..."):
                st.session_state.risk_snapshot = prisk.compute_portfolio_risk(merged)
        risk = st.session_state.risk_snapshot
    else:
        st.info("Click **Calculate risk metrics** to load beta, Sharpe ratio, and correlation data.")
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
    pui.section(
        "Earnings quality checks",
        "Debt-to-equity, Piotroski F-Score, and operating cash flow vs reported profit",
    )
    st.caption("Uses Financial Modeling Prep (FMP) data when an API key is configured.")

    fmp_key = get_fmp_api_key()
    if not fmp_key:
        st.info("Add `[fmp] api_key` in `.streamlit/secrets.toml` to enable Piotroski and OCF checks.")
    else:
        st.caption("FMP key found — load scores for your holdings.")

    load_forensic = st.button("Load earnings quality data", key="load_forensic_data")
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
        st.caption("Click **Load earnings quality data** to run Piotroski and cash-flow checks.")
    return checks


def render_screener_upload(merged: pd.DataFrame) -> list[dict]:
    pui.section(
        "Shareholding data import",
        "Upload a Screener.in or Trendlyne export for promoter, pledge, and FII/DII checks",
    )
    st.caption("CSV or XLSX files are matched to stock symbols in your portfolio.")

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
    pui.section(
        "Earnings call review",
        "Paste a quarterly earnings transcript to flag tone shifts, guidance gaps, and risk language",
    )
    st.caption("Works best with a full earnings call transcript, not a short press release.")

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
    pui.render_diagnosis_brief(brief)


def render_portfolio_checks(checks: list[dict]) -> None:
    summary = pan.summarize_checks(checks)
    total = len(checks)
    health_score = round((summary["pass"] / total) * 100) if total else 0

    pui.section(
        "Portfolio health checks",
        "Rule-based alerts for concentration, drawdowns, and other portfolio-level risks",
    )

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
            st.markdown("**Needs attention**")
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
            st.markdown("**Watch closely**")
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
        "Sectors",
        (
            f"Mapped {coverage['sector_mapped']}/{coverage['total']} "
            f"({coverage['sector_mapped_pct']}%) · prices {coverage['found']}/{coverage['total']}"
        ),
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
    pui.section("Holding reference data")
    st.caption("Company, sector, valuation, and 52-week range from Yahoo Finance for each holding.")

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

    score_tab, alloc_tab, market_tab, deep_tab = st.tabs(
        [
            ":material/speed: Scorecard",
            ":material/pie_chart: Allocation",
            ":material/compare_arrows: Market & risk",
            ":material/fact_check: Quality checks",
        ],
        on_change="rerun",
        key="insights_main_tabs",
        default=":material/speed: Scorecard",
    )

    with score_tab:
        if score_tab.open:
            render_portfolio_ai_engine(portfolio_df)
            pui.section(
                "Portfolio diagnosis",
                "Plain read of where your money sits, what is dragging results, and what to do next",
            )
            render_research_brief(brief)
            render_portfolio_checks(checks)

            pui.section(
                "Return and P&L drivers",
                "Which holdings contributed most to gains and losses",
            )
            perf_left, perf_right = st.columns(2)
            with perf_left:
                with st.container(border=True):
                    st.altair_chart(pchart.return_weight_scatter(merged), width="stretch")
            with perf_right:
                with st.container(border=True):
                    st.altair_chart(pchart.pnl_waterfall(merged), width="stretch")

            show_quadrants = st.toggle("Show performance quadrants", value=False, key="show_quads")
            if show_quadrants:
                quadrant_meta = [
                    ("Core winners", quadrants["Core winners"]),
                    ("Core losers", quadrants["Core losers"]),
                    ("Small winners", quadrants["Small winners"]),
                    ("Low-impact losers", quadrants["Low-impact losers"]),
                ]
                quad_cols = st.columns(4, border=True)
                for col, (title, symbols) in zip(quad_cols, quadrant_meta):
                    with col:
                        st.markdown(f"**{title}**")
                        if symbols:
                            st.markdown(", ".join(f"**{s}**" for s in symbols))
                        else:
                            st.caption("None in this bucket")

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

    with alloc_tab:
        if alloc_tab.open:
            pui.section(
                "Portfolio allocation",
                "How portfolio weight is spread across stocks and sectors",
            )
            stock_tab, sector_tab = st.tabs(
                ["By stock", "By sector"], key="alloc_inner_tabs"
            )

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
                    merged[
                        ["Symbol", "Weight %", "Current Value", "P&L", "Return %", "Cumulative weight %"]
                    ]
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
                        "Cumulative weight %": st.column_config.NumberColumn(
                            "Cumul. %", format="%.1f%%"
                        ),
                    },
                )

                pui.section(
                    "Full holdings weight table",
                    "Every stock with portfolio weight, invested value, and P&L contribution",
                )
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
                        "Cumulative weight %": st.column_config.NumberColumn(
                            "Cumulative %", format="%.1f"
                        ),
                        "Invested Value": st.column_config.NumberColumn("Invested", format="₹%.0f"),
                        "Current Value": st.column_config.NumberColumn("Current", format="₹%.0f"),
                        "P&L": st.column_config.NumberColumn("P&L", format="₹%.0f"),
                        "Return %": st.column_config.NumberColumn("Return", format="%.2f%%"),
                        "P&L contribution %": st.column_config.NumberColumn(
                            "P&L share", format="%.1f%%"
                        ),
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
                            [
                                "Sector",
                                "Weight %",
                                "Current Value",
                                "P&L",
                                "Return %",
                                "Holdings",
                                "Stocks",
                            ]
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

    with market_tab:
        if market_tab.open:
            pui.section(
                "Portfolio vs Nifty 50",
                "Recent portfolio return compared with the Nifty 50 index",
            )
            cmp_left, cmp_right = st.columns([1.4, 1])
            with cmp_left:
                with st.container(border=True):
                    if benchmark and not benchmark.get("chart_df", pd.DataFrame()).empty:
                        st.altair_chart(
                            pchart.nifty_vs_portfolio_line(benchmark["chart_df"]), width="stretch"
                        )
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
                    st.markdown("#### Sector weights vs Nifty 50")
                    vs_left, vs_right = st.columns(2)
                    with vs_left:
                        with st.container(border=True):
                            st.altair_chart(
                                pchart.sector_vs_nifty_grouped(sector_vs), width="stretch"
                            )
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
                st.info("Load market data above to compare your sector mix against Nifty 50.")

            render_portfolio_risk_section(merged)

    with deep_tab:
        if deep_tab.open:
            forensic_checks = render_forensic_section(merged, enriched)
            checks.extend(forensic_checks)

            inst_checks = render_screener_upload(merged)
            checks.extend(inst_checks)

            if enriched is not None and sector_df is not None and coverage is not None:
                with st.expander("Sector performance & stock info", expanded=False):
                    with st.container(border=True):
                        st.altair_chart(pchart.sector_return_bar(sector_df), width="stretch")
                    render_stock_info_table(enriched, coverage)
            else:
                st.info("Load market data above for sector returns and name cards.")


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
    """OpenBB research terminal — visual fundamentals board (not a raw table)."""
    pui.section(
        "Company fundamentals",
        "Valuation, profitability, growth, and recent headlines for stocks you hold",
    )
    if not pterm.is_available():
        reason = pterm.unavailability_reason()
        st.info(
            f"Fundamental data is unavailable here. {reason} "
            "The rest of the dashboard still works."
        )
        return None

    st.caption("Data is loaded via OpenBB using Yahoo Finance for your portfolio symbols.")
    load = st.button(
        "Load company fundamentals",
        type="primary",
        icon=":material/download:",
        key="load_terminal_data",
    )
    symbols = tuple(merged["Symbol"].tolist())
    news_symbols = tuple(merged.head(5)["Symbol"].tolist())

    if load:
        pterm.build_terminal_snapshot.clear()

    snapshot = None
    if load or st.session_state.get("terminal_snapshot_key") == symbols:
        with st.spinner("Fetching company fundamentals and market pulse..."):
            snapshot = pterm.build_terminal_snapshot(symbols, news_symbols)
        st.session_state.terminal_snapshot_key = symbols
        st.session_state.terminal_snapshot = snapshot
    elif "terminal_snapshot" in st.session_state:
        snapshot = st.session_state.terminal_snapshot

    if not snapshot or not snapshot.get("available"):
        st.caption("Click **Load company fundamentals** to fetch data for your holdings.")
        return None

    _render_terminal_board(snapshot, merged)
    return snapshot


def _fmt_num(value, *, pct: bool = False, signed: bool = False, digits: int = 1) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    try:
        num = float(value)
    except (TypeError, ValueError):
        return "—"
    if pct:
        return f"{num:+.{digits}f}%" if signed else f"{num:.{digits}f}%"
    return f"{num:.{digits}f}"


def _prepare_terminal_display(snapshot: dict, merged: pd.DataFrame) -> pd.DataFrame:
    df = pterm.metrics_dataframe(snapshot, merged)
    if df.empty:
        return df
    cols = [
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
        "name",
    ]
    keep = [c for c in cols if c in df.columns]
    show = df[keep].copy()
    for col in ("profit_margin", "revenue_growth", "earnings_growth", "return_on_equity", "price_return_1y"):
        if col in show.columns:
            show[col] = pd.to_numeric(show[col], errors="coerce") * 100
    return show.rename(
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
            "name": "Name",
        }
    )


def _render_terminal_board(snapshot: dict, merged: pd.DataFrame) -> None:
    """Clean research workstation: scanner · focus · one comparison chart."""
    show = _prepare_terminal_display(snapshot, merged)
    if show.empty:
        st.warning("No fundamental rows returned for current holdings.")
        return

    pulse = snapshot.get("pulse") or {}
    if pulse.get("prev_close"):
        nifty = float(pulse["prev_close"])
        ma50 = pulse.get("ma_50d")
        high = pulse.get("year_high")
        low = pulse.get("year_low")
        bits = [f"**Nifty 50** {nifty:,.0f}"]
        if ma50:
            vs = (nifty / float(ma50) - 1) * 100
            bits.append(f"50d MA {float(ma50):,.0f} ({vs:+.1f}%)")
        if high and low:
            bits.append(f"52w {float(low):,.0f}–{float(high):,.0f}")
        st.caption(" · ".join(bits))

    symbols = show["Symbol"].tolist()
    default_sym = symbols[0]
    largest = None
    if "Weight %" in merged.columns and not merged.empty:
        try:
            largest = str(merged.sort_values("Weight %", ascending=False).iloc[0]["Symbol"])
        except Exception:
            largest = None
    if largest in symbols:
        default_sym = largest

    scan_col, focus_col = st.columns([1.15, 1], gap="large")

    with scan_col:
        st.markdown("**Holdings scanner**")
        st.caption("Sorted by portfolio weight — select a stock on the right for details")
        scanner = show[
            [c for c in ("Symbol", "Weight %", "P/E", "ROE", "1y return", "Rev growth") if c in show.columns]
        ].copy()
        st.dataframe(
            scanner.round(1),
            hide_index=True,
            height=min(420, 48 + 36 * max(len(scanner), 4)),
            column_config={
                "Symbol": st.column_config.TextColumn("Stock", pinned=True),
                "Weight %": st.column_config.ProgressColumn(
                    "Portfolio weight",
                    format="%.1f%%",
                    min_value=0,
                    max_value=float(max(scanner["Weight %"].max() if len(scanner) else 1, 1)),
                ),
                "P/E": st.column_config.NumberColumn("P/E", format="%.1f", help="Trailing price-to-earnings"),
                "ROE": st.column_config.NumberColumn("ROE", format="%.0f%%", help="Return on equity"),
                "1y return": st.column_config.NumberColumn("1y price return", format="%+.0f%%"),
                "Rev growth": st.column_config.NumberColumn("Revenue growth", format="%+.0f%%"),
            },
        )

    with focus_col:
        st.markdown("**Selected stock**")
        selected = st.selectbox(
            "Stock symbol",
            options=symbols,
            index=symbols.index(default_sym) if default_sym in symbols else 0,
            label_visibility="collapsed",
            key="terminal_focus_symbol",
        )
        row = show.loc[show["Symbol"] == selected].iloc[0]
        name = row.get("Name")
        wt = row.get("Weight %")
        header = f"### {selected}"
        st.markdown(header)
        meta = []
        if isinstance(name, str) and name.strip():
            meta.append(name.strip()[:60])
        if pd.notna(wt):
            meta.append(f"{float(wt):.1f}% of portfolio")
        if meta:
            st.caption(" · ".join(meta))

        m1, m2, m3 = st.columns(3)
        m1.metric("P/E", _fmt_num(row.get("P/E")), help="Trailing price-to-earnings ratio")
        m2.metric("Forward P/E", _fmt_num(row.get("Fwd P/E")), help="Forward price-to-earnings")
        m3.metric("PEG", _fmt_num(row.get("PEG"), digits=2), help="Price/earnings-to-growth")
        m4, m5, m6 = st.columns(3)
        m4.metric("ROE", _fmt_num(row.get("ROE"), pct=True, digits=0), help="Return on equity")
        m5.metric("Profit margin", _fmt_num(row.get("Margin"), pct=True, digits=0))
        m6.metric("Debt / equity", _fmt_num(row.get("D/E"), digits=1))
        m7, m8, m9 = st.columns(3)
        m7.metric(
            "Revenue growth",
            _fmt_num(row.get("Rev growth"), pct=True, signed=True, digits=0),
            help="Year-over-year revenue growth",
        )
        m8.metric(
            "Earnings growth",
            _fmt_num(row.get("Earn growth"), pct=True, signed=True, digits=0),
            help="Year-over-year earnings growth",
        )
        m9.metric(
            "1-year price return",
            _fmt_num(row.get("1y return"), pct=True, signed=True, digits=0),
        )

        news = (snapshot.get("news") or {}).get(selected) or []
        if news:
            st.markdown("**Recent headlines**")
            for headline in news[:3]:
                st.markdown(f"- {headline}")

    st.space("small")
    st.markdown("**Compare holdings**")
    st.caption("Switch the metric to compare stocks across your portfolio")
    metric = st.segmented_control(
        "Comparison metric",
        options=["P/E", "ROE", "Rev growth", "1y return"],
        default="P/E",
        format_func=lambda m: {
            "P/E": "P/E",
            "ROE": "ROE",
            "Rev growth": "Revenue growth",
            "1y return": "1y price return",
        }.get(m, m),
        key="terminal_compare_metric",
    )
    metric = metric or "P/E"
    chart_map = {
        "P/E": (pchart.terminal_pe_bars, False),
        "ROE": (pchart.terminal_roe_bars, False),
        "Rev growth": (pchart.terminal_growth_bars, False),
        "1y return": (pchart.terminal_return_bars, False),
    }
    chart_fn, _ = chart_map[metric]
    st.altair_chart(chart_fn(show), width="stretch")

    other_news = {
        sym: heads
        for sym, heads in (snapshot.get("news") or {}).items()
        if sym != selected and heads
    }
    if other_news:
        with st.expander("More headlines", icon=":material/newspaper:", expanded=False):
            for sym, headlines in other_news.items():
                st.markdown(f"**{sym}**")
                for headline in headlines:
                    st.caption(f"· {headline}")

    with st.expander("All metrics", icon=":material/table_chart:", expanded=False):
        table = show.drop(columns=["Name"], errors="ignore")
        st.dataframe(
            table.round(2),
            hide_index=True,
            column_config={
                "Weight %": st.column_config.NumberColumn(format="%.1f%%"),
                "Margin": st.column_config.NumberColumn(format="%.1f%%"),
                "Rev growth": st.column_config.NumberColumn(format="%+.1f%%"),
                "Earn growth": st.column_config.NumberColumn(format="%+.1f%%"),
                "ROE": st.column_config.NumberColumn(format="%.1f%%"),
                "1y return": st.column_config.NumberColumn(format="%+.1f%%"),
            },
        )


def render_written_research(
    merged: pd.DataFrame,
    summary: dict,
    metrics: dict,
    portfolio_df: pd.DataFrame,
    terminal_snapshot: dict | None,
) -> None:
    """Book brief, name deep-dive, and transcript tools."""
    pui.section(
        "Written research notes",
        "Generate a portfolio brief or a deeper research note on one stock",
    )

    keys = resolve_agent_api_keys()
    openai_key = get_secret("openai.api_key")
    has_ai = bool(keys.get("gemini") or keys.get("xai") or keys.get("groq") or openai_key)

    status = st.container(horizontal=True)
    with status:
        _ai_badge("Gemini", keys.get("gemini"), expected="gemini")
        _ai_badge("xAI / Grok", keys.get("xai"), expected="xai")
        _ai_badge("Groq", keys.get("groq"), expected="groq")
        _ai_badge("OpenAI", openai_key, expected="openai")

    groq_provider = pai.detect_key_provider(keys.get("groq") or "")
    if groq_provider == "xai":
        st.warning(
            "Your Groq slot has an **xAI** key (`xai-...`). Move it to `[xai]` and put a `gsk_...` key under `[groq]`."
        )
        return
    if not has_ai:
        st.warning(
            "Add at least one key in `.streamlit/secrets.toml`: `[gemini]`, `[xai]`, or `[groq]` (`gsk_...`)."
        )
        return

    st.caption("Tries Gemini first, then xAI, Groq, OpenAI.")

    brief_col, research_col = st.columns(2)
    with brief_col:
        with st.container(border=True):
            st.markdown("**Portfolio brief**")
            st.caption("Summary of portfolio health, key risks, rebalancing ideas, and watchlist.")
            if st.button("Generate portfolio brief", type="primary", width="stretch", key="gen_ai_brief"):
                with st.spinner("Writing portfolio brief..."):
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
                    {"role": "user", "content": "Write a full portfolio brief."}
                )
                st.session_state.chat_messages.append(
                    {
                        "role": "assistant",
                        "content": f"*Brief via {result['provider']}*\n\n{result['text']}",
                    }
                )

    with research_col:
        with st.container(border=True):
            st.markdown("**Single-stock research note**")
            st.caption(
                "Fundamentals, 52-week range, role in the portfolio, and hold / trim / add triggers."
            )
            symbols = merged["Symbol"].tolist()
            default_index = 0
            largest = metrics.get("largest_symbol")
            if largest in symbols:
                default_index = symbols.index(largest)
            research_symbol = st.selectbox(
                "Stock to research",
                options=symbols,
                index=default_index,
                key="ai_research_symbol",
            )
            include_tech = st.toggle(
                "Include 30-day price chart and RSI",
                value=True,
                key="ai_research_tech",
            )
            if st.button("Generate stock research", width="stretch", key="gen_ai_holding"):
                with st.spinner(f"Researching {research_symbol}..."):
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
                        "content": f"*Name note via {result['provider']}*\n\n{result['text']}",
                    }
                )

    if st.session_state.ai_portfolio_briefing:
        with st.container(border=True):
            provider = st.session_state.ai_portfolio_briefing["provider"]
            paidisp.render_ai_response(
                f"*Portfolio brief · {provider}*\n\n{st.session_state.ai_portfolio_briefing['text']}"
            )

    if st.session_state.ai_holding_research:
        with st.container(border=True):
            note = st.session_state.ai_holding_research
            paidisp.render_ai_response(
                f"*Stock research · `{note['symbol']}` · {note['provider']}*\n\n{note['text']}"
            )

    with st.expander("Earnings call transcript review", expanded=False):
        render_transcript_auditor(merged, summary, metrics, portfolio_df, keys, openai_key)


def render_portfolio_chat(
    merged: pd.DataFrame,
    summary: dict,
    metrics: dict,
    portfolio_df: pd.DataFrame,
) -> None:
    keys = resolve_agent_api_keys()
    openai_key = get_secret("openai.api_key")
    openai_model = get_secret("openai.model", "gpt-4o-mini")
    has_ai = bool(keys.get("gemini") or keys.get("xai") or keys.get("groq") or openai_key)

    pui.section(
        "Ask about your portfolio",
        "Ask questions about holdings, risk, allocation, or individual stocks",
    )
    if has_ai:
        if keys.get("gemini"):
            provider = "Gemini"
        elif keys.get("xai"):
            provider = "xAI / Grok"
        elif keys.get("groq"):
            provider = "Groq"
        else:
            provider = f"OpenAI ({openai_model})"
        st.caption(f"Answers use {provider} and are grounded in your current holdings data.")
    else:
        st.caption(
            "Add keys in `.streamlit/secrets.toml`: `[gemini]`, `[xai]`, and/or `[groq]`. "
            "A few offline answers still work without keys."
        )

    if st.button("Clear chat", key="clear_research_chat"):
        st.session_state.chat_messages = []
        st.rerun()

    if not st.session_state.chat_messages:
        selected = st.pills(
            "Try asking",
            list(pchat.SUGGESTED_QUESTIONS.keys()),
            label_visibility="collapsed",
            key="research_chat_pills",
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
        "Ask about risk, allocation, Nifty, or a specific stock…",
        submit_mode="disable",
        key="research_chat_input",
    ):
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analysing your portfolio..."):
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


def render_research_tab(
    portfolio_df: pd.DataFrame,
    summary: dict,
) -> None:
    """Narrative research: fundamentals, written notes, chat — one job per tab."""
    total_current = summary["total_current"]
    if total_current <= 0:
        st.warning("Portfolio value is zero, so research tools cannot run.")
        return

    merged = pan.merge_holdings(portfolio_df, total_current)
    metrics = pan.compute_metrics(merged, summary)

    fund_tab, notes_tab, chat_tab = st.tabs(
        [
            ":material/candlestick_chart: Fundamentals",
            ":material/edit_note: Research notes",
            ":material/chat: Q&A",
        ],
        on_change="rerun",
        key="research_main_tabs",
        default=":material/candlestick_chart: Fundamentals",
    )

    terminal_snapshot = st.session_state.get("terminal_snapshot")

    with fund_tab:
        if fund_tab.open:
            terminal_snapshot = render_research_terminal(merged)

    with notes_tab:
        if notes_tab.open:
            render_written_research(
                merged, summary, metrics, portfolio_df, terminal_snapshot
            )

    with chat_tab:
        if chat_tab.open:
            render_portfolio_chat(merged, summary, metrics, portfolio_df)


def render_insights_tab(portfolio_df: pd.DataFrame, summary: dict) -> None:
    """Quantitative analysis only — scorecard, allocation, risk, benchmarks."""
    total_current = summary["total_current"]
    if total_current <= 0:
        st.warning("Portfolio value is zero, so insights cannot be calculated.")
        return

    merged = pan.merge_holdings(portfolio_df, total_current)
    metrics = pan.compute_metrics(merged, summary)
    render_insights_analysis(merged, metrics, portfolio_df, summary)



def _exchange_zerodha_request_token(api_key: str, api_secret: str, raw_token: str) -> None:
    """Exchange request token / redirect URL and persist today's access token."""
    try:
        st.session_state.zerodha_access_token = zauth.generate_access_token(
            api_key, api_secret, raw_token
        )
        st.session_state.zerodha_auth_success = (
            "Zerodha connected for today. Click Refresh portfolio."
        )
        st.session_state.zerodha_auth_error = ""
        st.rerun()
    except Exception as exc:
        st.session_state.zerodha_auth_error = str(exc)


def _zerodha_connect_button(label: str, login_url: str) -> None:
    """
    Kite login entry points.

    Streamlit Cloud sometimes fails on st.link_button to external sites, so we
    always show a plain URL the user can open/copy.
    """
    st.link_button(label, login_url, type="primary", width="stretch")
    st.markdown(f"**Or open this Kite URL:** [{login_url}]({login_url})")
    st.code(login_url, language=None)
    st.caption("If the button does nothing, copy the URL above into a new browser tab.")


def render_zerodha_sidebar() -> tuple[str, str]:
    """Render Zerodha controls. Returns (api_key, access_token) for portfolio fetch."""
    import app_paths

    default_api_key = get_secret("zerodha.api_key")
    default_api_secret = get_secret("zerodha.api_secret")
    secrets_ready = bool(default_api_key.strip() and default_api_secret.strip())
    kite_redirect = zauth.redirect_url(get_secret("zerodha.redirect_url"))
    on_cloud = app_paths.is_streamlit_cloud()

    if secrets_ready and not st.session_state.zerodha_api_secret:
        st.session_state.zerodha_api_secret = default_api_secret

    if "zerodha_api_key" not in st.session_state:
        st.session_state.zerodha_api_key = default_api_key

    if secrets_ready:
        api_key = default_api_key.strip()
        effective_secret = default_api_secret.strip()
        st.caption("API key & secret loaded from app secrets.")
    else:
        st.text_input(
            "API key",
            key="zerodha_api_key",
            type="password",
            help="From developers.kite.trade — or add to Streamlit secrets so this is not asked again.",
        )
        st.text_input(
            "API secret",
            key="zerodha_api_secret",
            type="password",
            help="Permanent app secret from developers.kite.trade — not the daily access token.",
        )
        api_key = str(st.session_state.zerodha_api_key or "").strip()
        effective_secret = zauth.resolve_api_secret("", st.session_state.zerodha_api_secret)

    cred_error = zauth.validate_credentials(api_key, effective_secret)
    access_token = st.session_state.zerodha_access_token
    connected = zauth.has_active_token(access_token)

    if st.session_state.get("zerodha_oauth_pending"):
        st.info("Reading redirect URL and saving today's Zerodha login…")

    if st.session_state.zerodha_auth_success:
        st.success(st.session_state.zerodha_auth_success)
        st.session_state.zerodha_auth_success = ""
    if st.session_state.zerodha_auth_error:
        st.error(st.session_state.zerodha_auth_error)
    elif cred_error:
        st.error(cred_error)
    elif connected:
        st.success(zauth.token_status_caption(access_token))
    else:
        st.info(
            "1) Click **Connect Zerodha**  2) Finish Kite login  "
            "3) You return here automatically. If not connected, use Fallback below."
        )

    if not cred_error and api_key:
        label = "Reconnect Zerodha" if connected else "Connect Zerodha"
        _zerodha_connect_button(label, zauth.login_url(api_key))
        st.caption(
            f"After login, Kite must send you back to `{kite_redirect}`. "
            "Set that **exact** URL in https://developers.kite.trade → your app → Redirect URL "
            "and in Streamlit Cloud Secrets as `zerodha.redirect_url`."
        )
        if "127.0.0.1" in kite_redirect or "localhost" in kite_redirect:
            st.error(
                "This app thinks redirect is local (`127.0.0.1` / localhost). "
                "For Streamlit Cloud, both Kite developer console AND Cloud Secrets must use "
                "`https://wealthapp-ankur.streamlit.app` (no trailing slash). "
                "Kite allows one redirect URL per app — local and Cloud cannot differ."
            )
        elif on_cloud:
            st.warning(
                "On Kite developers.kite.trade, Redirect URL must be exactly "
                f"`{kite_redirect}` — if it still says 127.0.0.1, the Zerodha link fails after login."
            )

    if not connected and not cred_error and api_key:
        if st.button(
            "I finished Kite login — load token",
            width="stretch",
            type="secondary",
            help="Loads today's token from cache after Kite redirects back.",
        ):
            cached = zauth.load_cached_token()
            if cached:
                st.session_state.zerodha_access_token = cached
                st.session_state.zerodha_auth_success = (
                    "Found today's login. Click Refresh portfolio."
                )
                st.session_state.zerodha_auth_error = ""
                st.rerun()
            else:
                st.session_state.zerodha_auth_error = (
                    "No login saved yet. After Kite, your browser URL should look like "
                    f"`{kite_redirect}/?request_token=...&status=success`. "
                    "Paste that full URL in Fallback below."
                )

        with st.expander("Fallback: paste redirect URL", expanded=True):
            st.caption(
                "Copy the full address-bar URL after Kite login (must include request_token=) "
                f"and paste it here. Example: `{kite_redirect}/?request_token=...&status=success`."
            )
            with st.form("zerodha_token_form", clear_on_submit=False):
                request_token = st.text_input("Redirect URL or request_token")
                submitted = st.form_submit_button(
                    "Save today's access token",
                    width="stretch",
                    type="primary",
                )
            if submitted:
                if not request_token.strip():
                    st.error("Paste the redirect URL from the address bar.")
                else:
                    _exchange_zerodha_request_token(
                        api_key, effective_secret, request_token
                    )

        with st.expander("Connection diagnostics"):
            st.write(
                {
                    "cloud_runtime": on_cloud,
                    "api_key_present": bool(api_key),
                    "api_secret_present": bool(effective_secret),
                    "redirect_url": kite_redirect,
                    "token_cache_writable_path": str(zauth.token_cache_path()),
                    "cached_token_today": bool(zauth.load_cached_token()),
                    "session_connected": connected,
                }
            )

    if connected:
        with st.expander("Zerodha session"):
            st.caption(zauth.token_status_caption(access_token))
            if st.button("Clear today's login", width="stretch"):
                zauth.clear_cached_token()
                st.session_state.zerodha_access_token = ""
                st.session_state.zerodha_auth_success = ""
                st.session_state.zerodha_auth_error = ""
                st.rerun()
    elif not secrets_ready:
        def _persist_manual_access_token() -> None:
            token = str(st.session_state.get("zerodha_access_token") or "").strip()
            if token:
                zauth.save_cached_token(token)

        st.text_input(
            "Access token (auto-filled after Connect)",
            key="zerodha_access_token",
            type="password",
            help="Filled automatically after a successful redirect.",
            on_change=_persist_manual_access_token,
        )
        access_token = st.session_state.zerodha_access_token

    return api_key, access_token



# --- Sidebar ---
pui.sidebar_howto()
pui.sidebar_block("Data sources")

with st.sidebar.container(border=True):
    st.markdown("**1 · Zerodha (live holdings)**")
    api_key, access_token = render_zerodha_sidebar()

with st.sidebar.container(border=True):
    st.markdown("**2 · Groww (holdings file, T+1)**")
    st.caption(
        "Groww reports land a day late. On refresh, the file is applied to the "
        "**as-of** date below (default: yesterday), and that sleeve is carried into today "
        "until a newer file arrives."
    )
    groww_file = st.file_uploader(
        "Upload Groww holdings CSV or Excel",
        type=["csv", "xlsx"],
        label_visibility="collapsed",
    )
    groww_as_of = st.date_input(
        "Groww file as-of date",
        value=ph.yesterday_ist(),
        max_value=ph.today_ist(),
        help="Usually yesterday — the day the Groww report reflects.",
    )

with st.sidebar.container(border=True):
    st.markdown("**3 · Booked P&L (sells / realised)**")
    st.caption(
        "Holdings ignore sold stocks. Upload Zerodha Console → Reports → P&L "
        "(choose **Realised**) or Groww Tax/P&L CSV so booked losses show in the report."
    )
    realized_file = st.file_uploader(
        "Upload realised P&L CSV or Excel",
        type=["csv", "xlsx"],
        key="realized_pnl_upload",
        label_visibility="collapsed",
    )
    if realized_file is not None:
        try:
            parsed = rpnl.parse_realized_report(
                realized_file,
                filename=getattr(realized_file, "name", "") or "",
                source="Broker P&L upload",
            )
            rpnl.save_realized_state(parsed)
            st.session_state.realized_pnl = parsed
            st.sidebar.success(
                f"Booked P&L loaded: {pui.format_inr_compact(parsed['realized_total'])} "
                f"({parsed['loss_count']} losses, {parsed['gain_count']} gains)."
            )
        except Exception as exc:
            st.sidebar.error(f"Could not read realised P&L file: {exc}")

    current_realized = st.session_state.get("realized_pnl") or {}
    if current_realized.get("row_count"):
        st.caption(
            f"Loaded · realised {pui.format_inr_compact(current_realized.get('realized_total', 0))} · "
            f"booked losses {pui.format_inr_compact(current_realized.get('booked_losses', 0))}"
        )
        if st.button("Clear booked P&L", width="stretch"):
            rpnl.clear_realized_state()
            st.session_state.realized_pnl = rpnl.empty_realized_state()
            st.rerun()

st.sidebar.space("small")
st.sidebar.markdown("**4 · Update prices + daily history**")
st.sidebar.caption(
    "Zerodha → today's sleeve (live). Groww file → as-of day only (T+1), then carried forward. "
    "Today's history point auto-saves after 3:30 PM IST (Force-save anytime)."
)
if st.sidebar.button(
    "Refresh portfolio",
    type="primary",
    width="stretch",
    icon=":material/refresh:",
):
    calculate_portfolio(api_key, access_token, groww_file, groww_as_of=groww_as_of)

hist = ph.history_status(lookback_days=14)
if hist["today_saved"]:
    st.sidebar.success(hist["caption"])
else:
    st.sidebar.info(hist["caption"])

with st.sidebar.expander("Manual snapshot"):
    st.caption(
        "Force-save writes today's point even before market close. "
        "Use after you have Zerodha connected (Groww optional / T+1)."
    )
    if st.button("Force-save today's snapshot", width="stretch", icon=":material/photo_camera:"):
        if st.session_state.portfolio_summary is None:
            st.warning("Refresh the portfolio first.")
        else:
            summary = st.session_state.portfolio_summary
            result = ph.save_summary_snapshot(
                summary,
                force=True,
                had_zerodha=float(summary.get("zerodha_current") or 0) > 0,
                had_groww=float(summary.get("groww_current") or 0) > 0,
                groww_as_of=groww_as_of,
            )
            if result["saved"]:
                st.success(result["reason"])
            else:
                st.warning(result["reason"])

# --- Main navigation ---
NAV_ICONS = {
    "Portfolio": ":material/account_balance_wallet: Portfolio",
    "Insights": ":material/insights: Insights",
    "Research": ":material/menu_book: Research",
    "Trends": ":material/timeline: Trends",
}
main_view = st.segmented_control(
    "Go to",
    ["Portfolio", "Insights", "Research", "Trends"],
    default="Portfolio",
    format_func=lambda view: NAV_ICONS[view],
    label_visibility="collapsed",
    width="stretch",
    key="main_nav",
)
main_view = main_view or "Portfolio"

pui.page_header(main_view, _summary_for_header(st.session_state.portfolio_summary))
st.space("small")

if main_view == "Portfolio":
    if st.session_state.portfolio_df is None:
        pui.empty_state(
            "No portfolio loaded",
            "Connect a broker and refresh to view your holdings and portfolio value.",
            icon="upload",
            steps=[
                "Connect **Zerodha** and/or upload a **Groww** holdings file",
                "Click **Refresh portfolio**",
                "Review value by broker and the holdings table on this page",
            ],
        )
    else:
        render_portfolio_tab(st.session_state.portfolio_df, st.session_state.portfolio_summary)
elif main_view == "Trends":
    render_trends_tab()
elif main_view == "Research":
    if st.session_state.portfolio_df is None:
        pui.empty_state(
            "Portfolio required",
            "Stock research needs your current holdings to know which companies to analyse.",
            icon="menu_book",
            steps=[
                "Refresh the portfolio from the sidebar",
                "Open **Fundamentals** and load company data",
                "Use **Research notes** or **Q&A** for written analysis",
            ],
        )
    else:
        render_research_tab(st.session_state.portfolio_df, st.session_state.portfolio_summary)
else:
    if st.session_state.portfolio_df is None:
        pui.empty_state(
            "Portfolio required",
            "Portfolio analysis needs your current holdings.",
            icon="insights",
            steps=[
                "Refresh the portfolio from the sidebar",
                "Start on the **Scorecard** tab",
                "Open **Allocation** if a few stocks dominate portfolio weight",
            ],
        )
    else:
        render_insights_tab(st.session_state.portfolio_df, st.session_state.portfolio_summary)
