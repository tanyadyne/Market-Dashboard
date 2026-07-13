#!/usr/bin/env python3
"""
Fetch major US economic events and stock-universe earnings from Yahoo Finance,
normalize them into a three-week payload, and write:

- economic_calendar.json
- economic_calendar.js

The event scope is intentionally narrow and driven by a curated whitelist of
major releases the dashboard cares about.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
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
SCREENER_TICKERS = ROOT / "screener_tickers.json"
LEADERS_JSON = ROOT / "leaders.json"
EARNINGS_STATE = ROOT / "earnings_calendar_state.json"
TZ_ET = ZoneInfo("America/New_York")

EARNINGS_STATE_VERSION = 1
ROLL_FORWARD_MAX_DAYS = 14
RECOVERY_STABLE_DAYS = 7
RECOVERY_MIN_LEAD_DAYS = 3
STATE_RETENTION_DAYS = 180


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


def _reference_monday(now_et: datetime) -> datetime.date:
    """Return this trading week's Monday, rolling weekends to next Monday."""
    if now_et.weekday() >= 5:
        return (now_et + timedelta(days=7 - now_et.weekday())).date()
    return (now_et - timedelta(days=now_et.weekday())).date()


def _week_bounds(now_et: datetime, offset: int = 0) -> tuple[datetime, datetime]:
    start = _reference_monday(now_et) + timedelta(days=offset * 7)
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
    chosen: dict[tuple[str, str], tuple[tuple[datetime, int, str], dict]] = {}

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
            identity = (matcher.key, event_time_et.date().isoformat())
            current = chosen.get(identity)
            if current is None or sort_key < current[0]:
                chosen[identity] = (sort_key, payload)
            break

    rate_items = [item for identity, item in chosen.items() if identity[0] == "fomc_rate"]
    press_dates = {identity[1] for identity in chosen if identity[0] == "fed_press"}
    for rate_sort, rate_payload in rate_items:
        rate_time_et = rate_sort[0]
        rate_date = rate_time_et.date().isoformat()
        if rate_date in press_dates:
            continue
        press_payload = _synthetic_fomc_press(rate_payload, rate_time_et)
        chosen[("fed_press", rate_date)] = (
            (rate_time_et + timedelta(minutes=30), 0, "Synthetic FOMC press conference"),
            press_payload,
        )

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


def _load_stock_metadata() -> tuple[set[str], dict[str, dict]]:
    with SCREENER_TICKERS.open(encoding="utf-8") as handle:
        universe_data = json.load(handle)
    universe = {str(ticker).upper() for ticker in universe_data.get("tickers", [])}

    metadata: dict[str, dict] = {}
    if LEADERS_JSON.exists():
        with LEADERS_JSON.open(encoding="utf-8") as handle:
            leaders = json.load(handle)
        for row in leaders.get("e", []):
            ticker = str(row.get("t", "")).upper()
            if ticker:
                metadata[ticker] = {
                    "company": row.get("n") or ticker,
                    "group": row.get("th") or "-",
                }
    return universe, metadata


def _row_value(row, *names):
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip().lower() not in {"", "nan", "nat", "none"}:
            return value
    return None


def _as_datetime(value) -> datetime | None:
    if value is None:
        return None
    if hasattr(value, "to_pydatetime"):
        value = value.to_pydatetime()
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=TZ_ET)
        return value.astimezone(TZ_ET)
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=TZ_ET)
    return parsed.astimezone(TZ_ET)


def _as_date(value) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


def _empty_earnings_state() -> dict:
    return {"version": EARNINGS_STATE_VERSION, "updated": None, "tickers": {}}


def _load_earnings_state(observed_on: date) -> dict:
    if EARNINGS_STATE.exists():
        try:
            with EARNINGS_STATE.open(encoding="utf-8") as handle:
                state = json.load(handle)
            if isinstance(state.get("tickers"), dict):
                state["version"] = EARNINGS_STATE_VERSION
                return state
        except (OSError, ValueError, TypeError):
            pass

    state = _empty_earnings_state()
    if not OUT_JSON.exists():
        return state

    try:
        with OUT_JSON.open(encoding="utf-8") as handle:
            previous_payload = json.load(handle)
    except (OSError, ValueError, TypeError):
        return state

    for week in previous_payload.get("weeks", []):
        for event in week.get("earnings_events", []):
            ticker = str(event.get("ticker", "")).upper()
            event_date = _as_date(event.get("date"))
            if not ticker or event_date is None:
                continue
            state["tickers"][ticker] = {
                "last_date": event_date.isoformat(),
                "stable_since": observed_on.isoformat(),
                "last_seen": observed_on.isoformat(),
                "roll_count": 0,
                "quarantined": False,
            }
    return state


def _save_earnings_state(state: dict, observed_on: date) -> None:
    state["version"] = EARNINGS_STATE_VERSION
    state["updated"] = observed_on.isoformat()
    EARNINGS_STATE.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def _public_earnings_event(event: dict) -> dict:
    return {key: value for key, value in event.items() if not key.startswith("_")}


def _candidate_is_reported(event: dict) -> bool:
    return bool(event.get("_reported"))


def _apply_earnings_reliability_guard(
    earnings: list[dict], state: dict, observed_on: date
) -> tuple[list[dict], list[dict]]:
    """Suppress Yahoo dates that roll forward without an earnings report."""
    ticker_state = state.setdefault("tickers", {})
    allowed: list[dict] = []
    suppressed: list[dict] = []
    pending: dict[str, dict] = {}

    for event in earnings:
        ticker = event["ticker"]
        if _candidate_is_reported(event):
            allowed.append(_public_earnings_event(event))
            record = ticker_state.setdefault(ticker, {})
            record.update({
                "last_reported_date": event["date"],
                "last_seen": observed_on.isoformat(),
                "quarantined": False,
            })
            record.pop("reason", None)
            record.pop("quarantined_at", None)
            continue

        current = pending.get(ticker)
        if current is None or event["date"] < current["date"]:
            pending[ticker] = event

    for ticker, event in sorted(pending.items()):
        event_date = _as_date(event["date"])
        if event_date is None:
            continue

        record = ticker_state.setdefault(ticker, {})
        previous_date = _as_date(record.get("last_date"))
        stable_since = _as_date(record.get("stable_since"))
        confirmed_date = _as_date(record.get("confirmed_date"))
        date_changed = previous_date is not None and event_date != previous_date

        if confirmed_date == event_date:
            record.update({
                "last_date": event["date"],
                "stable_since": record.get("stable_since") or observed_on.isoformat(),
                "last_seen": observed_on.isoformat(),
                "quarantined": False,
            })
            record.pop("reason", None)
            record.pop("quarantined_at", None)
            allowed.append(_public_earnings_event(event))
            continue

        is_roll_forward = (
            date_changed
            and event_date > previous_date
            and previous_date <= observed_on
            and (event_date - previous_date).days <= ROLL_FORWARD_MAX_DAYS
        )

        if date_changed:
            record["stable_since"] = observed_on.isoformat()
            stable_since = observed_on
        elif stable_since is None:
            record["stable_since"] = observed_on.isoformat()
            stable_since = observed_on

        if is_roll_forward:
            record["quarantined"] = True
            record["reason"] = "rolling_unreported_date"
            record["quarantined_at"] = record.get("quarantined_at") or observed_on.isoformat()
            record["roll_count"] = int(record.get("roll_count", 0)) + 1

        record["last_date"] = event["date"]
        record["last_seen"] = observed_on.isoformat()
        record.setdefault("roll_count", 0)

        stable_days = (observed_on - stable_since).days
        lead_days = (event_date - observed_on).days
        can_recover = (
            record.get("quarantined")
            and stable_days >= RECOVERY_STABLE_DAYS
            and lead_days >= RECOVERY_MIN_LEAD_DAYS
        )
        if can_recover:
            record["quarantined"] = False
            record["recovered_at"] = observed_on.isoformat()
            record.pop("reason", None)
            record.pop("quarantined_at", None)

        if record.get("quarantined"):
            suppressed.append({
                "ticker": ticker,
                "date": event["date"],
                "reason": record.get("reason", "unconfirmed_date"),
            })
        else:
            allowed.append(_public_earnings_event(event))

    retention_cutoff = observed_on - timedelta(days=STATE_RETENTION_DAYS)
    for ticker in list(ticker_state):
        last_seen = _as_date(ticker_state[ticker].get("last_seen"))
        if (
            last_seen is not None
            and last_seen < retention_cutoff
            and not ticker_state[ticker].get("quarantined")
        ):
            del ticker_state[ticker]

    allowed.sort(key=lambda item: (item["date"], item["time"], item["ticker"]))
    return allowed, suppressed


def _earnings_time(row, event_time: datetime) -> str:
    raw = _row_value(row, "Timing", "Earnings Call Time", "Call Time", "Time")
    text = str(raw or "").strip().upper()
    aliases = {
        "BEFORE MARKET OPEN": "BMO",
        "BEFORE OPEN": "BMO",
        "PRE-MARKET": "BMO",
        "AFTER MARKET CLOSE": "AMC",
        "AFTER CLOSE": "AMC",
        "POST-MARKET": "AMC",
        "TIME NOT SUPPLIED": "TNS",
        "NOT SUPPLIED": "TNS",
    }
    if text in {"BMO", "AMC", "TNS"}:
        return text
    if text in aliases:
        return aliases[text]
    if event_time.hour < 12:
        return "BMO"
    if event_time.hour >= 16:
        return "AMC"
    return "TNS"


def _fetch_earnings_rows(start_et: datetime, end_et: datetime):
    session = _build_session()
    calendar = yf.Calendars(start=start_et.date(), end=end_et.date() + timedelta(days=1), session=session)
    rows = []
    limit = 100
    max_pages = 20
    for page in range(max_pages):
        batch = calendar.get_earnings_calendar(
            filter_most_active=False,
            limit=limit,
            offset=page * limit,
            force=page == 0,
        )
        page_rows = list(_iter_calendar_rows(batch))
        rows.extend(page_rows)
        if len(page_rows) < limit:
            break
    return rows


def _select_universe_earnings(rows, universe: set[str], metadata: dict[str, dict]) -> list[dict]:
    selected: dict[tuple[str, str], dict] = {}
    for index, row in rows:
        ticker = str(_row_value(row, "Symbol", "Ticker") or index or "").strip().upper()
        if ticker not in universe:
            continue
        event_time = _as_datetime(
            _row_value(row, "Event Start Date", "Earnings Date", "Event Time", "Start Date")
        )
        if event_time is None:
            continue
        profile = metadata.get(ticker, {})
        company = _row_value(row, "Company", "Company Name") or profile.get("company") or ticker
        selected[(ticker, event_time.date().isoformat())] = {
            "date": event_time.date().isoformat(),
            "day": _day_label(event_time),
            "ticker": ticker,
            "company": str(company),
            "group": str(profile.get("group") or "-"),
            "time": _earnings_time(row, event_time),
            "_reported": _row_value(row, "Reported EPS", "EPS Actual") is not None,
        }
    return sorted(selected.values(), key=lambda item: (item["date"], item["time"], item["ticker"]))


def _week_payload(key: str, label: str, start_et: datetime, events: list[dict], earnings: list[dict]) -> dict:
    days = _build_days(start_et.date())
    dates = {day["date"] for day in days}
    week_events = [event for event in events if event["date"] in dates]
    week_earnings = [event for event in earnings if event["date"] in dates]
    return {
        "key": key,
        "week_label": label,
        "range_label": _range_label(days),
        "days": days,
        "events": week_events,
        "earnings_events": week_earnings,
        "event_keys": [event["key"] for event in week_events],
    }


def build_payload(now_et: datetime | None = None) -> tuple[dict, dict]:
    now_et = now_et or datetime.now(TZ_ET)
    previous_start, _ = _week_bounds(now_et, -1)
    current_start, _ = _week_bounds(now_et, 0)
    next_start, next_end = _week_bounds(now_et, 1)

    rows = _fetch_calendar_rows(previous_start, next_end)
    events = _select_major_events(rows)
    universe, metadata = _load_stock_metadata()
    earnings_rows = _fetch_earnings_rows(previous_start, next_end)
    earnings = _select_universe_earnings(earnings_rows, universe, metadata)
    earnings_state = _load_earnings_state(now_et.date())
    earnings, suppressed = _apply_earnings_reliability_guard(
        earnings, earnings_state, now_et.date()
    )

    weeks = [
        _week_payload("previous", "Last week", previous_start, events, earnings),
        _week_payload("current", "This week", current_start, events, earnings),
        _week_payload("next", "Next week", next_start, events, earnings),
    ]
    current_week = weeks[1]

    payload = {
        "generated_at_utc": datetime.now(TZ_ET).astimezone(ZoneInfo("UTC")).strftime("%Y-%m-%d %H:%M UTC"),
        "generated_at_et": now_et.strftime("%d/%m/%Y, %H:%M %Z"),
        "timezone": "America/New_York",
        "default_week_index": 1,
        "weeks": weeks,
        # Keep current-week fields for the dashboard landing page.
        "week_label": current_week["week_label"],
        "range_label": current_week["range_label"],
        "days": current_week["days"],
        "events": current_week["events"],
        "earnings_events": current_week["earnings_events"],
        "event_keys": current_week["event_keys"],
        "earnings_guard": {
            "suppressed_count": len(suppressed),
            "suppressed": suppressed,
        },
    }
    return payload, earnings_state


def write_outputs(payload: dict) -> None:
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    js = "window.ECONOMIC_CALENDAR_DATA = " + json.dumps(payload, indent=2) + ";\n"
    OUT_JS.write_text(js, encoding="utf-8")


def main() -> None:
    now_et = datetime.now(TZ_ET)
    payload, earnings_state = build_payload(now_et)
    write_outputs(payload)
    _save_earnings_state(earnings_state, now_et.date())
    total_economic = sum(len(week["events"]) for week in payload["weeks"])
    total_earnings = sum(len(week["earnings_events"]) for week in payload["weeks"])
    print(
        f"Wrote {OUT_JSON.name} and {OUT_JS.name} with "
        f"{total_economic} economic event(s) and {total_earnings} earnings event(s)"
    )
    suppressed = payload["earnings_guard"]["suppressed"]
    if suppressed:
        print("Suppressed unreliable Yahoo earnings dates:")
        for event in suppressed:
            print(f"  - {event['ticker']} on {event['date']} ({event['reason']})")
    missing = [matcher.label for matcher in MAJOR_EVENTS if matcher.key not in payload["event_keys"]]
    if missing:
        print("Missing this week:")
        for label in missing:
            print(f"  - {label}")


if __name__ == "__main__":
    main()
