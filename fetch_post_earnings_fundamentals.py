#!/usr/bin/env python3
"""
Refresh fundamentals only for stocks that just reported earnings.

This is a lightweight companion to the full weekly fundamentals refresh. It
uses the dashboard's earnings calendar, finds stock-screener tickers that
reported on the current US trading date, fetches fresh Yahoo Finance
fundamentals for those tickers, and merges them into fundamentals.json/js.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from fetch_fundamentals import (
    MAX_QUARTERS,
    MAX_YEARS,
    OUT_JS,
    OUT_JSON,
    _build_session,
    _fetch_one,
    _load_universe,
)


ROOT = Path(__file__).resolve().parent
CALENDAR_JSON = ROOT / "economic_calendar.json"
TZ_ET = ZoneInfo("America/New_York")


def _previous_weekday(day):
    while day.weekday() >= 5:
        day -= timedelta(days=1)
    return day


def _target_earnings_dates() -> list[str]:
    override = os.getenv("TARGET_EARNINGS_DATE", "").strip()
    if override:
        return [override]

    now_et = datetime.now(TZ_ET)
    # The scheduled 7am SGT run lands in the US evening, so the current ET date
    # is the session whose BMO/AMC reports should be rechecked.
    dates = []
    cursor = _previous_weekday(now_et.date())
    lookback = 2
    while len(dates) < lookback:
        dates.append(cursor.isoformat())
        cursor = _previous_weekday(cursor - timedelta(days=1))
    return dates


def _calendar_earnings_for_dates(target_dates: set[str], universe: set[str]) -> list[str]:
    if not CALENDAR_JSON.exists():
        print(f"{CALENDAR_JSON.name} not found; no post-earnings refresh candidates.")
        return []

    with CALENDAR_JSON.open(encoding="utf-8") as handle:
        calendar = json.load(handle)

    tickers: set[str] = set()
    for week in calendar.get("weeks", []):
        for event in week.get("earnings_events", []):
            ticker = str(event.get("ticker") or "").strip().upper()
            if event.get("date") not in target_dates or ticker not in universe:
                continue
            # Keep this deliberately broad. Yahoo may label some reports as TNS;
            # if the event date matches the finished US session, a quick refresh
            # is cheap and safer than missing a newly published quarter.
            tickers.add(ticker)

    return sorted(tickers)


def _load_existing_payload(universe_count: int) -> dict:
    if OUT_JSON.exists():
        try:
            with OUT_JSON.open(encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, dict) and isinstance(payload.get("tickers"), dict):
                payload.setdefault("meta", {})
                return payload
        except Exception as exc:
            print(f"Warning: failed to read {OUT_JSON.name}; starting fresh ({exc})")

    return {
        "meta": {
            "updated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "count": 0,
            "universe": universe_count,
            "max_quarters": MAX_QUARTERS,
            "max_years": MAX_YEARS,
        },
        "tickers": {},
    }


def _write_payload(payload: dict) -> None:
    with OUT_JSON.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"))
    with OUT_JS.open("w", encoding="utf-8") as handle:
        handle.write("window.FUNDAMENTALS_DATA = ")
        json.dump(payload, handle, separators=(",", ":"))
        handle.write(";\n")


def main() -> int:
    universe = set(_load_universe())
    target_dates = _target_earnings_dates()
    tickers = _calendar_earnings_for_dates(set(target_dates), universe)

    if not tickers:
        print(f"No screener-universe earnings found for {', '.join(target_dates)}; nothing to refresh.")
        return 0

    payload = _load_existing_payload(len(universe))
    payload.setdefault("tickers", {})
    session = _build_session()
    refreshed: list[str] = []
    failed: list[str] = []

    print(f"Post-earnings fundamentals refresh for {', '.join(target_dates)}: {len(tickers)} ticker(s)")
    for ticker in tickers:
        try:
            data = _fetch_one(ticker, session)
            if any(isinstance(value, list) and value for value in data.values()):
                payload["tickers"][ticker] = data
                refreshed.append(ticker)
                print(f"  {ticker}: refreshed")
            else:
                failed.append(ticker)
                print(f"  {ticker}: no fundamentals returned")
        except Exception as exc:
            failed.append(ticker)
            print(f"  {ticker}: failed ({exc})")
        time.sleep(0.05)

    if not refreshed:
        print("No fundamentals entries were updated.")
        return 0

    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    meta = payload.setdefault("meta", {})
    meta["updated"] = now_utc
    meta["count"] = len(payload.get("tickers", {}))
    meta["universe"] = len(universe)
    meta["max_quarters"] = MAX_QUARTERS
    meta["max_years"] = MAX_YEARS
    meta["last_post_earnings_update"] = now_utc
    meta["last_post_earnings_dates"] = target_dates
    meta.pop("last_post_earnings_date", None)
    meta["last_post_earnings_tickers"] = refreshed
    if failed:
        meta["last_post_earnings_failed"] = failed
    else:
        meta.pop("last_post_earnings_failed", None)

    _write_payload(payload)
    print(f"Wrote {OUT_JSON.name}: refreshed {len(refreshed)} ticker(s): {', '.join(refreshed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
