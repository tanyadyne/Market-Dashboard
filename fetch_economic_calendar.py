#!/usr/bin/env python3
"""
Fetch upcoming major US economic events from Yahoo Finance's free economic
calendar feed, normalize them into a compact weekly payload, and write:

- economic_calendar.json
- economic_calendar.js

The event scope is intentionally narrow and driven by a curated whitelist of
major releases the dashboard cares about.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import yfinance as yf

try:
    from curl_cffi import requests as cffi_requests
except ImportError:  # pragma: no cover - optional optimization only
    cffi_requests = None


ROOT = Path(__file__).resolve().parent
OUT_JSON = ROOT / "economic_calendar.json"
OUT_JS = ROOT / "economic_calendar.js"
TZ_ET = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class EventMatcher:
    key: str
    label: str
    patterns: tuple[str, ...]
    exclusions: tuple[str, ...] = ()


MAJOR_EVENTS: tuple[EventMatcher, ...] = (
    EventMatcher("retail_sales", "Retail sales", (r"\bretail sales mm\b", r"\bretail sales\b")),
    EventMatcher(
        "fomc_rate",
        "FOMC interest rate decision",
        (
            r"\bfed funds tgt rate\b",
            r"\bfed interest rate decision\b",
            r"\bfomc.*rate decision\b",
            r"\brate decision\b",
        ),
    ),
    EventMatcher(
        "fed_press",
        "Fed chair press conference",
        (r"\bfed press conference\b", r"\bchair press conference\b"),
    ),
    EventMatcher("cpi", "CPI inflation", (r"\bcpi\b",), (r"\bcore\b",)),
    EventMatcher("ppi", "PPI inflation", (r"\bppi\b",), (r"\bcore\b",)),
    EventMatcher(
        "uom_sentiment",
        "Consumer sentiment (UoM)",
        (r"\bconsumer sentiment\b", r"\bu mich sentiment\b", r"\buniv(?:ersity)? of michigan sentiment\b"),
    ),
    EventMatcher(
        "pce",
        "PCE inflation",
        (r"\bpce\b", r"\bpersonal consumption expenditures\b"),
        (r"\bcore\b",),
    ),
    EventMatcher(
        "jolts",
        "JOLTS Job openings",
        (r"\bjolts\b", r"\bjob openings\b"),
    ),
    EventMatcher("adp", "ADP employment", (r"\badp\b",)),
    EventMatcher("unemployment", "Unemployment rate", (r"\bunemployment rate\b",)),
    EventMatcher(
        "consumer_confidence",
        "Consumer confidence",
        (r"\bconsumer confidence\b",),
    ),
    EventMatcher(
        "gdp",
        "GDP (quarterly)",
        (r"^gdp\b", r"\bgross domestic product\b"),
        (r"\bdeflator\b", r"\bprice index\b", r"\bcons spending\b", r"\bconsumption\b", r"\bfinal sales\b"),
    ),
    EventMatcher("chicago_pmi", "Chicago PMI", (r"\bchicago pmi\b",)),
    EventMatcher(
        "sp_mfg_pmi",
        "S&P U.S. Manufacturing PMI",
        (r"\bs&p.*manufacturing pmi\b", r"\bmanufacturing pmi\b"),
        (r"\bism\b",),
    ),
    EventMatcher(
        "sp_services_pmi",
        "S&P U.S. Services PMI",
        (r"\bs&p.*services pmi\b", r"\bservices pmi\b"),
        (r"\bism\b",),
    ),
)


def _build_session():
    if cffi_requests is None:
        return None
    try:
        return cffi_requests.Session(impersonate="chrome")
    except Exception:
        return None


def _week_bounds(now_et: datetime) -> tuple[datetime, datetime]:
    mode = os.getenv("CALENDAR_WEEK_MODE", "auto").strip().lower()
    if mode == "next":
        days_until_monday = 7 - now_et.weekday() if now_et.weekday() < 5 else 7 - now_et.weekday()
        start = (now_et + timedelta(days=days_until_monday)).date()
    elif now_et.weekday() >= 5:
        days_until_monday = 7 - now_et.weekday()
        start = (now_et + timedelta(days=days_until_monday)).date()
    else:
        start = (now_et - timedelta(days=now_et.weekday())).date()
    end = start + timedelta(days=4)
    start_dt = datetime.combine(start, datetime.min.time(), tzinfo=TZ_ET)
    end_dt = datetime.combine(end, datetime.max.time().replace(microsecond=0), tzinfo=TZ_ET)
    return start_dt, end_dt


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip()).lower()


def _matcher_priority(matcher: EventMatcher, event_name: str) -> int:
    name = _normalize_text(event_name)
    if any(re.search(pattern, name, re.I) for pattern in matcher.exclusions):
        return -1
    for idx, pattern in enumerate(matcher.patterns):
        if re.search(pattern, name, re.I):
            return idx
    return -1


def _clean_number(value) -> str:
    if value is None:
        return "-"
    try:
        if math.isnan(value):
            return "-"
    except TypeError:
        pass
    if isinstance(value, float):
        text = f"{value:.2f}".rstrip("0").rstrip(".")
        return text or "0"
    return str(value)


def _period_text(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() == "none":
        return ""
    return text.upper()


def _day_label(dt_et: datetime) -> str:
    return dt_et.strftime("%A, %B ") + str(dt_et.day) + dt_et.strftime(", %Y")


def _range_label(days: list[dict]) -> str:
    if not days:
        return ""
    first = datetime.fromisoformat(days[0]["date"])
    last = datetime.fromisoformat(days[-1]["date"])
    return (
        first.strftime("%B ")
        + str(first.day)
        + first.strftime(", %Y")
        + " - "
        + last.strftime("%B ")
        + str(last.day)
        + last.strftime(", %Y")
    )


def _build_days(start_et_date: datetime.date) -> list[dict]:
    days = []
    for offset in range(5):
        dt = datetime.combine(start_et_date + timedelta(days=offset), datetime.min.time())
        days.append({"date": dt.date().isoformat(), "label": _day_label(dt)})
    return days


def _iter_calendar_rows(rows):
    if hasattr(rows, "iterrows"):
        return rows.iterrows()
    return iter(rows or [])


def _payload_from_row(key: str, label: str, source_event: str, event_time: datetime, row) -> dict:
    event_time_et = event_time.astimezone(TZ_ET)
    return {
        "key": key,
        "event": label,
        "source_event": source_event,
        "date": event_time_et.date().isoformat(),
        "day": _day_label(event_time_et),
        "time": event_time_et.strftime("%I:%M %p"),
        "period": _period_text(row.get("For")),
        "actual": _clean_number(row.get("Actual")),
        "consensus": _clean_number(row.get("Expected")),
        "previous": _clean_number(row.get("Last")),
        "forecast": _clean_number(row.get("Expected")),
        "revised": _clean_number(row.get("Revised")),
    }


def _synthetic_fomc_press(rate_payload: dict, rate_time_et: datetime) -> dict:
    press_time = rate_time_et + timedelta(minutes=30)
    return {
        "key": "fed_press",
        "event": "Fed chair press conference",
        "source_event": "Synthetic FOMC press conference",
        "date": press_time.date().isoformat(),
        "day": _day_label(press_time),
        "time": press_time.strftime("%I:%M %p"),
        "period": rate_payload.get("period", ""),
        "actual": "-",
        "consensus": "-",
        "previous": "-",
        "forecast": "-",
        "revised": "-",
    }


def _select_major_events(df) -> list[dict]:
    chosen: dict[str, tuple[tuple[datetime, int, str], dict]] = {}

    for event_name, row in _iter_calendar_rows(df):
        if str(row.get("Region", "")).upper() != "US":
            continue

        event_time = row.get("Event Time")
        if not isinstance(event_time, datetime):
            continue

        name = str(event_name).replace("*", "").strip()
        for matcher in MAJOR_EVENTS:
            priority = _matcher_priority(matcher, name)
            if priority < 0:
                continue

            event_time_et = event_time.astimezone(TZ_ET)
            payload = _payload_from_row(matcher.key, matcher.label, name, event_time, row)
            sort_key = (event_time_et, priority, name)
            current = chosen.get(matcher.key)
            if current is None or sort_key < current[0]:
                chosen[matcher.key] = (sort_key, payload)
            break

    if "fomc_rate" in chosen and "fed_press" not in chosen:
        rate_time_et = chosen["fomc_rate"][0][0]
        press_payload = _synthetic_fomc_press(chosen["fomc_rate"][1], rate_time_et)
        chosen["fed_press"] = ((rate_time_et + timedelta(minutes=30), 0, "Synthetic FOMC press conference"), press_payload)

    selected = [item[1] for item in sorted(chosen.values(), key=lambda item: item[0])]
    return selected


def _fetch_calendar_rows(start_et: datetime, end_et: datetime):
    session = _build_session()
    rows = []
    day = start_et.date()
    end_day = end_et.date()
    while day <= end_day:
        next_day = day + timedelta(days=1)
        calendars = yf.Calendars(
            start=day.isoformat(),
            end=next_day.isoformat(),
            session=session,
        )
        daily_rows = calendars.get_economic_events_calendar(limit=100, force=True)
        rows.extend(list(_iter_calendar_rows(daily_rows)))
        day = next_day
    return rows


def build_payload(now_et: datetime | None = None) -> dict:
    now_et = now_et or datetime.now(TZ_ET)
    week_start_et, week_end_et = _week_bounds(now_et)
    rows = _fetch_calendar_rows(week_start_et, week_end_et)
    events = _select_major_events(rows)

    days = _build_days(week_start_et.date())

    payload = {
        "generated_at_utc": datetime.now(TZ_ET).astimezone(ZoneInfo("UTC")).strftime("%Y-%m-%d %H:%M UTC"),
        "generated_at_et": now_et.strftime("%d/%m/%Y, %H:%M %Z"),
        "timezone": "America/New_York",
        "week_label": "This week",
        "range_label": _range_label(days),
        "days": days,
        "events": events,
        "event_keys": [event["key"] for event in events],
    }
    return payload


def write_outputs(payload: dict) -> None:
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    js = "window.ECONOMIC_CALENDAR_DATA = " + json.dumps(payload, indent=2) + ";\n"
    OUT_JS.write_text(js, encoding="utf-8")


def main() -> None:
    payload = build_payload()
    write_outputs(payload)
    print(f"Wrote {OUT_JSON.name} and {OUT_JS.name} with {len(payload['events'])} event(s)")
    missing = [matcher.label for matcher in MAJOR_EVENTS if matcher.key not in payload["event_keys"]]
    if missing:
        print("Missing this week:")
        for label in missing:
            print(f"  - {label}")


if __name__ == "__main__":
    main()
