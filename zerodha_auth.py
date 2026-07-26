"""Zerodha Kite Connect login and daily access-token exchange."""

from __future__ import annotations

import hashlib
from urllib.parse import parse_qs, urlparse

from kiteconnect import KiteConnect
from kiteconnect.exceptions import KiteException

# Must match the redirect URL registered at https://developers.kite.trade
DEFAULT_REDIRECT_URL = "http://127.0.0.1:8501"


def normalize_request_token(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        return ""
    if "request_token=" in text:
        if text.startswith("http"):
            query = urlparse(text).query
        else:
            query = text.lstrip("?")
        values = parse_qs(query).get("request_token", [])
        if values:
            return values[0].strip()
        for part in text.split("&"):
            if part.startswith("request_token="):
                return part.split("=", 1)[1].strip()
    return text


def login_url(api_key: str) -> str:
    return KiteConnect(api_key=api_key.strip()).login_url()


def checksum(api_key: str, request_token: str, api_secret: str) -> str:
    payload = f"{api_key.strip()}{request_token.strip()}{api_secret.strip()}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_credentials(api_key: str, api_secret: str, access_token: str = "") -> str | None:
    if not api_key.strip():
        return "Missing Zerodha API key."
    if not api_secret.strip():
        return (
            "Missing Zerodha API secret. Add `api_secret` under `[zerodha]` in "
            "`.streamlit/secrets.toml` from https://developers.kite.trade (your app page). "
            "This is not the daily access token."
        )
    if access_token and api_secret.strip() == access_token.strip():
        return (
            "`api_secret` must be your Kite Connect app secret from developers.kite.trade, "
            "not the same value as `access_token`."
        )
    return None


def generate_access_token(api_key: str, api_secret: str, request_token: str) -> str:
    clean_key = api_key.strip()
    clean_secret = api_secret.strip()
    clean_request_token = normalize_request_token(request_token)
    cred_error = validate_credentials(clean_key, clean_secret)
    if cred_error:
        raise ValueError(cred_error)
    if not clean_request_token:
        raise ValueError("Request token is required.")

    kite = KiteConnect(api_key=clean_key)
    try:
        session = kite.generate_session(clean_request_token, api_secret=clean_secret)
    except KiteException as exc:
        raise RuntimeError(token_error_message(exc)) from exc
    access_token = session.get("access_token")
    if not access_token:
        raise RuntimeError("Kite did not return an access token.")
    return access_token


def token_error_message(exc: Exception) -> str:
    message = str(exc).strip()
    lower = message.lower()
    if "invalid checksum" in lower:
        return (
            "Invalid checksum: your `api_secret` does not match this `api_key`, or the "
            "request token is wrong, expired, or already used. "
            "Get the API secret from developers.kite.trade (not the access token), "
            "then log in again for a fresh request token."
        )
    if "token is invalid" in lower or "expired" in lower:
        return "Request token expired or already used. Log in to Kite again for a new one."
    return message
