"""Zerodha Kite Connect login and daily access-token exchange."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from zoneinfo import ZoneInfo

from kiteconnect import KiteConnect
from kiteconnect.exceptions import KiteException, TokenException

# Must match the redirect URL registered at https://developers.kite.trade
DEFAULT_REDIRECT_URL = "http://127.0.0.1:8501"
IST = ZoneInfo("Asia/Kolkata")
TOKEN_CACHE_PATH = Path(__file__).resolve().parent / "data" / "zerodha_token.json"


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


def redirect_url(configured: str = "") -> str:
    url = (configured or DEFAULT_REDIRECT_URL).strip()
    return url or DEFAULT_REDIRECT_URL


def resolve_api_secret(*sources: str) -> str:
    for source in sources:
        value = (source or "").strip()
        if value:
            return value
    return ""


def last_reset_time_ist(now: datetime | None = None) -> datetime:
    """Most recent 6:00 AM IST — tokens issued before this are expired."""
    now = now or datetime.now(IST)
    reset = now.replace(hour=6, minute=0, second=0, microsecond=0)
    if now < reset:
        reset -= timedelta(days=1)
    return reset


def next_expiry_ist(now: datetime | None = None) -> datetime:
    """Next 6:00 AM IST when the current access token expires."""
    now = now or datetime.now(IST)
    expiry = now.replace(hour=6, minute=0, second=0, microsecond=0)
    if now >= expiry:
        expiry += timedelta(days=1)
    return expiry


def token_cache_valid(saved_at: datetime, now: datetime | None = None) -> bool:
    now = now or datetime.now(IST)
    return saved_at >= last_reset_time_ist(now)


def load_cached_token() -> str:
    if not TOKEN_CACHE_PATH.exists():
        return ""
    try:
        payload = json.loads(TOKEN_CACHE_PATH.read_text(encoding="utf-8"))
        token = str(payload.get("access_token", "")).strip()
        saved_raw = payload.get("saved_at", "")
        if not token or not saved_raw:
            return ""
        saved_at = datetime.fromisoformat(saved_raw)
        if saved_at.tzinfo is None:
            saved_at = saved_at.replace(tzinfo=IST)
        if token_cache_valid(saved_at):
            return token
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        pass
    clear_cached_token()
    return ""


def save_cached_token(access_token: str) -> None:
    token = (access_token or "").strip()
    if not token:
        return
    TOKEN_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "access_token": token,
        "saved_at": datetime.now(IST).isoformat(),
    }
    TOKEN_CACHE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def clear_cached_token() -> None:
    try:
        if TOKEN_CACHE_PATH.exists():
            TOKEN_CACHE_PATH.unlink()
    except OSError:
        pass


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
            "Missing Zerodha API secret. Paste it in the sidebar below, or add `api_secret` under "
            "`[zerodha]` in `.streamlit/secrets.toml` from https://developers.kite.trade "
            "(your app page). This is your permanent app secret — not the daily access token."
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
    save_cached_token(access_token)
    return access_token


def is_access_token_error(exc: Exception) -> bool:
    if isinstance(exc, TokenException):
        return True
    message = str(exc).lower()
    return any(
        phrase in message
        for phrase in (
            "incorrect `api_key` or `access_token`",
            "invalid api key or access token",
            "token is invalid",
            "access token",
            "session expired",
        )
    )


def access_token_error_message(exc: Exception | None = None) -> str:
    expiry = next_expiry_ist()
    base = (
        "Zerodha access token expired. Kite resets tokens every day at 6:00 AM IST. "
        f"Next expiry: {expiry.strftime('%d %b %Y, %I:%M %p IST')}. "
        "Click Connect Zerodha in the sidebar for a fresh login."
    )
    if exc is None:
        return base
    detail = str(exc).strip()
    if detail and not is_access_token_error(exc):
        return f"{base} ({detail})"
    return base


def token_error_message(exc: Exception) -> str:
    message = str(exc).strip()
    lower = message.lower()
    if "invalid checksum" in lower:
        return (
            "Invalid checksum: your `api_secret` does not match this `api_key`, or the "
            "request token is wrong, expired, or already used. "
            "Copy the API secret from developers.kite.trade (not the access token), "
            "paste it in the sidebar, then click **Connect Zerodha** again for a fresh login."
        )
    if "token is invalid" in lower or "expired" in lower or "already used" in lower:
        return (
            "Request token expired or already used. Click **Connect Zerodha** again — "
            "do not reuse an old redirect URL or request token."
        )
    if is_access_token_error(exc):
        return access_token_error_message(exc)
    return message


def token_status_caption(access_token: str) -> str:
    expiry = next_expiry_ist()
    if not (access_token or "").strip():
        return f"Not connected · log in to sync live holdings until {expiry.strftime('%I:%M %p IST')}."
    return f"Live until {expiry.strftime('%d %b, %I:%M %p IST')} · resets daily at 6 AM IST."


def has_active_token(access_token: str) -> bool:
    return bool((access_token or "").strip())
