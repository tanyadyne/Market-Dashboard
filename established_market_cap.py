"""Approved market-cap continuity records for established screener names.

The market-cap pipeline must distinguish a newly discovered security from a
constituent that has already proved itself in the screener.  New names require
a provider-supplied market cap.  Established names may use a retained,
approved share count times the current price if a provider temporarily omits
the cap field.
"""

import json
import math
import os


ESTABLISHED_MCAP_FILE = "leaders_established_mcap.json"
ESTABLISHED_RECORD_VERSION = 1


def positive_int(value):
    """Return a positive integer, or zero when a value is unusable."""
    try:
        value = int(value or 0)
    except (TypeError, ValueError):
        return 0
    return value if value > 0 else 0


def approved_shares(record):
    """Return shares only for an explicitly approved established record."""
    if not isinstance(record, dict) or record.get("status") != "approved":
        return 0
    return positive_int(record.get("shares"))


def estimate_from_established_shares(record, price, split_like_gap=False):
    """Return a guarded established-name cap estimate, or zero.

    A split-like price gap deliberately disables the fallback.  That requires a
    fresh provider value rather than combining a pre-event share count with a
    post-event quote.
    """
    if split_like_gap:
        return 0
    shares = approved_shares(record)
    try:
        price = float(price)
    except (TypeError, ValueError):
        return 0
    if not shares or not math.isfinite(price) or price <= 0:
        return 0
    return int(shares * price)


def load_established_market_cap_data(path=ESTABLISHED_MCAP_FILE):
    """Load the approved registry without treating a malformed file as trusted."""
    payload = {"version": ESTABLISHED_RECORD_VERSION, "tickers": {}}
    if not os.path.exists(path):
        return payload
    try:
        with open(path) as handle:
            loaded = json.load(handle)
    except Exception:
        return payload
    if not isinstance(loaded, dict):
        return payload
    tickers = loaded.get("tickers")
    if not isinstance(tickers, dict):
        tickers = {}
    payload.update(loaded)
    payload["tickers"] = tickers
    payload.setdefault("version", ESTABLISHED_RECORD_VERSION)
    return payload
