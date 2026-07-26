"""Resolve holdings and fetch live finance data for any stock question in chat."""

from __future__ import annotations

import re

import pandas as pd

import portfolio_earnings as pearn
import symbol_resolver as sym

# Portfolio-wide questions — live stock fetches usually not needed.
PORTFOLIO_WIDE_PATTERNS = (
    r"\bfull portfolio\b",
    r"\bportfolio briefing\b",
    r"\bexecutive summary\b",
    r"\brebalanc(e|ing)\b.*\bportfolio\b",
    r"\bsector allocation\b",
    r"\btop \d+ holdings\b",
    r"\bholdings by weight\b",
    r"\bbeating nifty\b",
    r"\bvs nifty\b",
    r"\bconcentration risk\b.*\bportfolio\b",
)

STOCK_FINANCE_PATTERNS = (
    r"\b(stock|share|company|counter|name|scrip)\b",
    r"\b(financial|finance|fundamental|valuation|metric|ratio)\b",
    r"\b(result|earnings|profit|revenue|eps|ebitda|margin)\b",
    r"\b(shareholding|promoter|pledge|fii|dii|institutional|ownership)\b",
    r"\b(outlook|thesis|view|catalyst|risk|quality)\b",
    r"\b(should i|hold|trim|add|sell|buy|accumulate|exit)\b",
    r"\bhow is\b",
    r"\btell me about\b",
    r"\bwhat about\b",
    r"\b(latest|recent|last quarter|quarterly)\b",
    r"\b(research|deep dive|analyse|analyze)\b",
    r"\b(pe|p/e|pb|p/b|dividend|roe|debt)\b",
)


def resolve_symbol_from_question(question: str, portfolio_symbols: list[str]) -> str | None:
    """Match any portfolio ticker or company name mentioned in the question."""
    if not portfolio_symbols:
        return None

    portfolio = set(portfolio_symbols)
    upper = question.upper()

    for symbol in sorted(portfolio_symbols, key=len, reverse=True):
        if re.search(rf"\b{re.escape(symbol.upper())}\b", upper):
            return symbol

    normalized = sym.normalize_label(question)
    name_hits: list[tuple[int, str]] = []
    for name, ticker in sym.NAME_TO_SYMBOL.items():
        if ticker not in portfolio:
            continue
        name_norm = sym.normalize_label(name)
        if name_norm in normalized or normalized in name_norm:
            name_hits.append((len(name_norm), ticker))
        elif name_norm.split()[0] in normalized.split():
            name_hits.append((len(name_norm.split()[0]), ticker))
    if name_hits:
        return max(name_hits, key=lambda item: item[0])[1]

    tokens = re.findall(r"[a-z0-9&]{3,}", question.lower())
    token_index: dict[str, str] = {}
    for symbol in portfolio_symbols:
        token_index[symbol.lower()] = symbol
        for name, ticker in sym.NAME_TO_SYMBOL.items():
            if ticker != symbol:
                continue
            token_index[name.lower()] = symbol
            first = name.split()[0].lower()
            if len(first) >= 3:
                token_index[first] = symbol
    for token in tokens:
        if token in token_index:
            return token_index[token]

    resolved = sym.resolve_nse_symbol(question.strip())
    if resolved in portfolio:
        return resolved

    if len(portfolio_symbols) == 1:
        return portfolio_symbols[0]
    return None


def is_portfolio_wide_question(question: str) -> bool:
    text = question.lower()
    return any(re.search(pattern, text) for pattern in PORTFOLIO_WIDE_PATTERNS)


def is_stock_finance_question(question: str) -> bool:
    text = question.lower()
    return any(re.search(pattern, text) for pattern in STOCK_FINANCE_PATTERNS)


def finance_data_needs(question: str, symbol: str | None) -> tuple[bool, bool]:
    """Return (fetch_earnings, fetch_shareholding) for a resolved symbol."""
    if not symbol:
        return False, False

    wants_earnings = pearn.is_earnings_question(question)
    from portfolio_shareholding import is_shareholding_question

    wants_shareholding = is_shareholding_question(question)
    if wants_earnings or wants_shareholding:
        return wants_earnings, wants_shareholding

    if is_portfolio_wide_question(question) and not is_stock_finance_question(question):
        return False, False

    if is_stock_finance_question(question):
        return True, True

    return False, False


def _merged_row(merged: pd.DataFrame, symbol: str) -> dict | None:
    if symbol not in merged["Symbol"].values:
        return None
    return merged.loc[merged["Symbol"] == symbol].iloc[0].to_dict()


def build_live_finance_context(
    question: str,
    merged: pd.DataFrame,
    *,
    fmp_api_key: str = "",
    inst_df: pd.DataFrame | None = None,
) -> str:
    """Fetch earnings and/or shareholding for any portfolio holding mentioned in chat."""
    symbols = merged["Symbol"].tolist()
    symbol = resolve_symbol_from_question(question, symbols)
    fetch_earnings, fetch_shareholding = finance_data_needs(question, symbol)
    if not symbol or (not fetch_earnings and not fetch_shareholding):
        return ""

    row = _merged_row(merged, symbol)
    parts: list[str] = []

    if fetch_earnings:
        data = pearn.fetch_quarterly_earnings(symbol, fmp_api_key)
        block = pearn.build_earnings_context(data, row)
        if block:
            parts.append(block)

    if fetch_shareholding:
        import portfolio_shareholding as pshare

        block = pshare.context_for_symbol(symbol, merged, inst_df)
        if block:
            parts.append(block)

    return "\n\n".join(parts)
