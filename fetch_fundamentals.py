#!/usr/bin/env python3
"""
Fetch static stock fundamentals from Yahoo Finance for the stock screener.

Outputs:
- fundamentals.json
- fundamentals.js

The browser reads the static JSON; this script is intended for a scheduled or
manual GitHub Actions refresh rather than live client-side API calls.
"""

from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import yfinance as yf

try:
    from curl_cffi import requests as cffi_requests
except ImportError:  # pragma: no cover - optional optimization only
    cffi_requests = None


ROOT = Path(__file__).resolve().parent
LEADERS_JSON = ROOT / "leaders.json"
SCREENER_TICKERS = ROOT / "screener_tickers.json"
OUT_JSON = ROOT / "fundamentals.json"
OUT_JS = ROOT / "fundamentals.js"
MAX_QUARTERS = 16
MAX_YEARS = 10


def _build_session():
    if cffi_requests is None:
        return None
    try:
        return cffi_requests.Session(impersonate="chrome")
    except Exception:
        return None


def _load_universe() -> list[str]:
    tickers: set[str] = set()

    if LEADERS_JSON.exists():
        try:
            with LEADERS_JSON.open() as f:
                data = json.load(f)
            for row in data.get("e", []):
                ticker = (row.get("t") or "").strip().upper()
                if ticker:
                    tickers.add(ticker)
        except Exception as exc:
            print(f"Warning: failed to read {LEADERS_JSON.name}: {exc}")

    if not tickers and SCREENER_TICKERS.exists():
        try:
            with SCREENER_TICKERS.open() as f:
                data = json.load(f)
            if isinstance(data, list):
                tickers.update(str(t).strip().upper() for t in data if str(t).strip())
            elif isinstance(data, dict):
                for value in data.values():
                    if isinstance(value, list):
                        tickers.update(str(t).strip().upper() for t in value if str(t).strip())
        except Exception as exc:
            print(f"Warning: failed to read {SCREENER_TICKERS.name}: {exc}")

    return sorted(tickers)


def _safe_number(value):
    try:
        if value is None:
            return None
        if hasattr(value, "item"):
            value = value.item()
        num = float(value)
        if not math.isfinite(num):
            return None
        return num
    except Exception:
        return None


def _period_label(value) -> str:
    try:
        if hasattr(value, "strftime"):
            return value.strftime("%Y-%m-%d")
        return str(value)[:10]
    except Exception:
        return str(value)


def _find_row(frame, names: tuple[str, ...]):
    if frame is None or getattr(frame, "empty", True):
        return None
    row_map = {str(idx).strip().lower(): idx for idx in frame.index}
    for name in names:
        idx = row_map.get(name.lower())
        if idx is not None:
            return frame.loc[idx]
    return None


def _series_from_income_statement(frame, row_names: tuple[str, ...], limit: int) -> list[dict]:
    row = _find_row(frame, row_names)
    if row is None:
        return []

    points = []
    for period, value in row.items():
        num = _safe_number(value)
        if num is None:
            continue
        points.append({"period": _period_label(period), "value": round(num, 4)})

    points.sort(key=lambda item: item["period"])
    return points[-limit:]


def _get_income_statement(ticker_obj, frequency: str):
    accessors = (
        "quarterly_income_stmt" if frequency == "quarterly" else "income_stmt",
        "quarterly_financials" if frequency == "quarterly" else "financials",
    )
    for attr in accessors:
        try:
            frame = getattr(ticker_obj, attr)
            if frame is not None and not frame.empty:
                return frame
        except Exception:
            pass

    try:
        frame = ticker_obj.get_income_stmt(freq=frequency)
        if frame is not None and not frame.empty:
            return frame
    except Exception:
        pass
    return None


def _fetch_one(ticker: str, session) -> dict:
    kwargs = {"session": session} if session is not None else {}
    ticker_obj = yf.Ticker(ticker, **kwargs)
    quarterly = _get_income_statement(ticker_obj, "quarterly")
    annual = _get_income_statement(ticker_obj, "yearly")

    return {
        "quarterly_revenue": _series_from_income_statement(
            quarterly,
            ("Total Revenue", "Operating Revenue"),
            MAX_QUARTERS,
        ),
        "annual_revenue": _series_from_income_statement(
            annual,
            ("Total Revenue", "Operating Revenue"),
            MAX_YEARS,
        ),
        "quarterly_eps": _series_from_income_statement(
            quarterly,
            ("Diluted EPS", "Basic EPS", "Diluted EPS Other Gains Losses"),
            MAX_QUARTERS,
        ),
        "annual_eps": _series_from_income_statement(
            annual,
            ("Diluted EPS", "Basic EPS", "Diluted EPS Other Gains Losses"),
            MAX_YEARS,
        ),
    }


def main() -> int:
    tickers = _load_universe()
    session = _build_session()
    payload = {
        "meta": {
            "updated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "count": 0,
            "universe": len(tickers),
            "max_quarters": MAX_QUARTERS,
            "max_years": MAX_YEARS,
        },
        "tickers": {},
    }

    print(f"Fetching fundamentals for {len(tickers)} stock(s)...")
    for i, ticker in enumerate(tickers, start=1):
        if i % 50 == 0:
            print(f"  {i}/{len(tickers)}...")
        try:
            data = _fetch_one(ticker, session)
            if any(data.values()):
                payload["tickers"][ticker] = data
        except Exception as exc:
            print(f"  {ticker}: failed ({exc})")
        time.sleep(0.05)

    payload["meta"]["count"] = len(payload["tickers"])
    with OUT_JSON.open("w") as f:
        json.dump(payload, f, separators=(",", ":"))
    with OUT_JS.open("w") as f:
        f.write("window.FUNDAMENTALS_DATA = ")
        json.dump(payload, f, separators=(",", ":"))
        f.write(";\n")

    print(f"Wrote {OUT_JSON.name}: {payload['meta']['count']}/{len(tickers)} stock(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
