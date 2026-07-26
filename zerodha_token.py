#!/usr/bin/env python3
"""Generate today's Zerodha access token from the command line."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore

import zerodha_auth as zauth


def load_zerodha_secrets() -> tuple[str, str, str]:
    secrets_path = Path(__file__).resolve().parent / ".streamlit" / "secrets.toml"
    if not secrets_path.exists():
        return "", "", ""
    data = tomllib.loads(secrets_path.read_text(encoding="utf-8"))
    zerodha = data.get("zerodha", {})
    return (
        str(zerodha.get("api_key", "")).strip(),
        str(zerodha.get("api_secret", "")).strip(),
        str(zerodha.get("access_token", "")).strip(),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Exchange a Zerodha request token for an access token.")
    parser.add_argument(
        "request_token",
        nargs="?",
        help="Request token or full redirect URL containing request_token=",
    )
    parser.add_argument("--api-key", help="Kite API key (defaults to secrets.toml)")
    parser.add_argument("--api-secret", help="Kite API secret (defaults to secrets.toml)")
    args = parser.parse_args()

    default_key, default_secret, default_access_token = load_zerodha_secrets()
    api_key = (args.api_key or default_key).strip()
    api_secret = (args.api_secret or default_secret).strip()

    cred_error = zauth.validate_credentials(api_key, api_secret, default_access_token)
    if cred_error:
        print(cred_error, file=sys.stderr)
        return 1

    request_token = zauth.normalize_request_token(args.request_token or "")
    if not request_token:
        print("Open this URL, log in, then rerun with the request token from the redirect URL:")
        print(zauth.login_url(api_key))
        print(f"\nRegister this redirect URL in Kite Connect: {zauth.DEFAULT_REDIRECT_URL}")
        return 1

    try:
        access_token = zauth.generate_access_token(api_key, api_secret, request_token)
    except Exception as exc:
        print(f"Failed: {exc}", file=sys.stderr)
        return 1

    print(access_token)
    print("\nPaste this into the sidebar 'Zerodha access token' field.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
