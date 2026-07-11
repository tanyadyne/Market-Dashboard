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
import argparse
import copy
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from calendar import monthrange
from pathlib import Path

try:
    import yfinance as yf
except ImportError:  # Recovery mode only needs local JSON files.
    yf = None

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
PROFILE_QUARTERLY_EPS_ACTUALS = 5
PROFILE_QUARTERLY_EPS_ESTIMATES = 2
PROFILE_ANNUAL_EPS_ACTUALS = 4
PROFILE_ANNUAL_EPS_ESTIMATES = 1
FX_RATE_CACHE: dict[tuple[str, str], float | None] = {}
OWNERSHIP_HOLDER_EXCLUDE_TERMS = (
    "blackrock",
    "vanguard",
    "state street",
    "ssga",
    "jpmorgan",
    "jp morgan",
    "morgan stanley",
    "ubs",
    "bank of america",
    "bofa",
    "bnp paribas",
    "t. rowe",
    "t.rowe",
    "t rowe",
    "price (t.rowe",
    "invesco",
    "fmr",
    "jane street",
    "first trust",
    "ishares",
    "spdr",
    "qqq",
    "index",
    "etf",
    "fidelity",
    "charles schwab",
    "schwab",
    "northern trust",
    "geode",
    "bny",
    "mellon",
    "deutsche bank",
    "goldman sachs",
    "citigroup",
    "wells fargo",
    "barclays",
    "hsbc",
    "credit suisse",
    "nomura",
    "mizuho",
    "societe generale",
)


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


def _parse_ticker_filter(value: str | None) -> set[str] | None:
    if not value:
        return None
    tickers = {item.strip().upper() for item in value.split(",") if item.strip()}
    return tickers or None


def _filter_tickers(tickers: list[str], ticker_filter: set[str] | None) -> list[str]:
    if not ticker_filter:
        return tickers
    return [ticker for ticker in tickers if ticker in ticker_filter]


def _empty_payload(ticker_count: int) -> dict:
    return {
        "meta": {
            "updated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "count": 0,
            "universe": ticker_count,
            "max_quarters": MAX_QUARTERS,
            "max_years": MAX_YEARS,
        },
        "tickers": {},
    }


def _load_existing_payload(ticker_count: int) -> dict:
    if OUT_JSON.exists():
        try:
            with OUT_JSON.open() as f:
                payload = json.load(f)
            if isinstance(payload, dict) and isinstance(payload.get("tickers"), dict):
                payload.setdefault("meta", {})
                payload["meta"]["universe"] = ticker_count
                payload["meta"].setdefault("max_quarters", MAX_QUARTERS)
                payload["meta"].setdefault("max_years", MAX_YEARS)
                return payload
        except Exception as exc:
            print(f"Warning: failed to read existing {OUT_JSON.name}: {exc}")
    return _empty_payload(ticker_count)


def _ownership_record_count(payload: dict) -> int:
    return sum(
        1
        for record in payload.get("tickers", {}).values()
        if isinstance(record, dict)
        and isinstance(record.get("ownership"), dict)
        and bool(record["ownership"])
    )


def _merge_cached_ownership(fresh: dict, cached: dict | None) -> dict:
    """Merge fresh fields over cached ownership without losing missing subfields."""
    if not isinstance(cached, dict) or not cached:
        return fresh
    merged = copy.deepcopy(cached)
    current = fresh.get("ownership")
    if isinstance(current, dict):
        for key, value in current.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key].update({k: v for k, v in value.items() if v is not None})
            elif value is not None:
                merged[key] = value
    fresh["ownership"] = merged
    return fresh


def _write_payload(payload: dict, ticker_count: int) -> None:
    payload.setdefault("meta", {})
    payload.setdefault("tickers", {})
    payload["meta"]["updated"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    payload["meta"]["count"] = len(payload["tickers"])
    payload["meta"]["universe"] = ticker_count
    payload["meta"]["max_quarters"] = MAX_QUARTERS
    payload["meta"]["max_years"] = MAX_YEARS

    json_tmp = OUT_JSON.with_name(f"{OUT_JSON.name}.tmp")
    js_tmp = OUT_JS.with_name(f"{OUT_JS.name}.tmp")
    with json_tmp.open("w") as f:
        json.dump(payload, f, separators=(",", ":"))
    with js_tmp.open("w") as f:
        f.write("window.FUNDAMENTALS_DATA = ")
        json.dump(payload, f, separators=(",", ":"))
        f.write(";\n")
    os.replace(json_tmp, OUT_JSON)
    os.replace(js_tmp, OUT_JS)


def _restore_ownership_from_snapshot(snapshot_path: Path, ticker_count: int) -> int:
    payload = _load_existing_payload(ticker_count)
    try:
        with snapshot_path.open(encoding="utf-8-sig") as f:
            snapshot = json.load(f)
    except Exception as exc:
        print(f"ERROR: unable to read ownership snapshot {snapshot_path}: {exc}")
        return 1

    snapshot_tickers = snapshot.get("tickers", {}) if isinstance(snapshot, dict) else {}
    restored = 0
    retained = 0
    unavailable = 0
    for ticker, record in payload.get("tickers", {}).items():
        current = record.get("ownership") if isinstance(record, dict) else None
        if isinstance(current, dict) and current:
            retained += 1
            continue
        cached_record = snapshot_tickers.get(ticker, {})
        cached = cached_record.get("ownership") if isinstance(cached_record, dict) else None
        if isinstance(cached, dict) and cached:
            record["ownership"] = copy.deepcopy(cached)
            restored += 1
        else:
            unavailable += 1

    total = _ownership_record_count(payload)
    payload.setdefault("meta", {})
    payload["meta"].update({
        "ownership_recovered": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "ownership_recovery_source": snapshot_path.name,
        "ownership_recovery_restored": restored,
        "ownership_count": total,
    })
    _write_payload(payload, ticker_count)
    print(
        f"Recovered ownership for {restored} stock(s); retained {retained}; "
        f"unavailable {unavailable}; total ownership records {total}"
    )
    return 0


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


def _median(values: list[float]) -> float | None:
    clean = sorted(abs(v) for v in values if math.isfinite(v) and v != 0)
    if not clean:
        return None
    mid = len(clean) // 2
    if len(clean) % 2:
        return clean[mid]
    return (clean[mid - 1] + clean[mid]) / 2


def _period_label(value) -> str:
    try:
        if hasattr(value, "strftime"):
            return value.strftime("%Y-%m-%d")
        return str(value)[:10]
    except Exception:
        return str(value)


def _period_month(value) -> int | None:
    label = _period_label(value)
    try:
        return int(label.split("-")[1])
    except Exception:
        return None


def _period_year_month(period: str) -> tuple[int, int] | None:
    try:
        year_s, month_s, *_ = period.split("-")
        return int(year_s), int(month_s)
    except Exception:
        return None


def _add_months(period: str, months: int) -> str | None:
    parsed = _period_year_month(period)
    if parsed is None:
        return None
    year, month = parsed
    day = 1
    try:
        day = int(period.split("-")[2])
    except Exception:
        pass
    month_index = (year * 12 + month - 1) + months
    new_year = month_index // 12
    new_month = month_index % 12 + 1
    new_day = min(day, monthrange(new_year, new_month)[1])
    return f"{new_year:04d}-{new_month:02d}-{new_day:02d}"


def _reported_period_from_earnings_date(period: str, fiscal_year_end_month: int) -> str:
    parsed = _period_year_month(period)
    if parsed is None:
        return period
    year, month = parsed
    quarter_end_months = sorted(((fiscal_year_end_month - 9 - 1) % 12 + 1,
                                 (fiscal_year_end_month - 6 - 1) % 12 + 1,
                                 (fiscal_year_end_month - 3 - 1) % 12 + 1,
                                 fiscal_year_end_month))
    prev_month = None
    for q_month in quarter_end_months:
        if q_month < month:
            prev_month = q_month
    if prev_month is None:
        prev_month = quarter_end_months[-1]
        year -= 1
    day = monthrange(year, prev_month)[1]
    return f"{year:04d}-{prev_month:02d}-{day:02d}"


def _infer_fiscal_year_end_month(annual_frame) -> int:
    if annual_frame is None or getattr(annual_frame, "empty", True):
        return 12
    counts: dict[int, int] = {}
    for period in annual_frame.columns:
        month = _period_month(period)
        if month is not None:
            counts[month] = counts.get(month, 0) + 1
    if not counts:
        return 12
    return sorted(counts.items(), key=lambda item: (-item[1], -item[0]))[0][0]


def _fiscal_label(period: str, annual: bool, fiscal_year_end_month: int) -> str:
    parsed = _period_year_month(period)
    if parsed is None:
        return period
    year, month = parsed
    fiscal_year = year if month <= fiscal_year_end_month else year + 1
    if annual:
        return f"FY{str(fiscal_year)[-2:]}"
    fiscal_start_month = (fiscal_year_end_month % 12) + 1
    months_since_start = (month - fiscal_start_month + 12) % 12
    quarter = months_since_start // 3 + 1
    return f"{str(fiscal_year)[-2:]}Q{quarter}"


def _find_row(frame, names: tuple[str, ...]):
    if frame is None or getattr(frame, "empty", True):
        return None
    row_map = {str(idx).strip().lower(): idx for idx in frame.index}
    for name in names:
        idx = row_map.get(name.lower())
        if idx is not None:
            return frame.loc[idx]
    return None


def _series_from_income_statement(
    frame,
    row_names: tuple[str, ...],
    limit: int,
    *,
    annual: bool,
    fiscal_year_end_month: int,
) -> list[dict]:
    row = _find_row(frame, row_names)
    if row is None:
        return []

    points = []
    for period, value in row.items():
        num = _safe_number(value)
        if num is None:
            continue
        period_label = _period_label(period)
        points.append({
            "period": period_label,
            "label": _fiscal_label(period_label, annual, fiscal_year_end_month),
            "value": round(num, 4),
        })

    points.sort(key=lambda item: item["period"])
    return points[-limit:]


def _get_estimate_frame(ticker_obj, attr: str):
    try:
        frame = getattr(ticker_obj, attr)
        if frame is not None and not frame.empty:
            return frame
    except Exception:
        pass
    getter = f"get_{attr}"
    try:
        func = getattr(ticker_obj, getter)
        frame = func()
        if frame is not None and not frame.empty:
            return frame
    except Exception:
        pass
    return None


def _get_earnings_dates(ticker_obj, limit: int = 24):
    try:
        frame = ticker_obj.get_earnings_dates(limit=limit)
        if frame is not None and not frame.empty:
            return frame
    except Exception:
        pass
    return None


def _estimate_avg_column(frame):
    if frame is None or getattr(frame, "empty", True):
        return None
    normalized = {str(col).strip().lower(): col for col in frame.columns}
    for key in ("avg", "average", "mean"):
        if key in normalized:
            return normalized[key]
    for key, col in normalized.items():
        if "avg" in key or "average" in key:
            return col
    return None


def _quarterly_estimates(
    frame,
    actual_points: list[dict],
    *,
    fiscal_year_end_month: int,
    limit: int = PROFILE_QUARTERLY_EPS_ESTIMATES,
) -> list[dict]:
    if frame is None or getattr(frame, "empty", True) or not actual_points:
        return []
    avg_col = _estimate_avg_column(frame)
    if avg_col is None:
        return []

    latest_period = actual_points[-1]["period"]
    estimates = []
    for row_key, row in frame.iterrows():
        key = str(row_key).strip().lower()
        is_quarter = "q" in key or "quarter" in key
        is_year = "y" in key or "year" in key
        if not is_quarter or is_year:
            continue
        num = _safe_number(row.get(avg_col))
        if num is None:
            continue
        period = _add_months(latest_period, 3 * (len(estimates) + 1))
        if period is None:
            continue
        estimates.append({
            "period": period,
            "label": f"{_fiscal_label(period, False, fiscal_year_end_month)} (E)",
            "value": round(num, 4),
            "estimate": True,
        })
        if len(estimates) >= limit:
            break
    return estimates


def _annual_estimates(
    frame,
    actual_points: list[dict],
    *,
    fiscal_year_end_month: int,
    limit: int = PROFILE_ANNUAL_EPS_ESTIMATES,
) -> list[dict]:
    if frame is None or getattr(frame, "empty", True) or not actual_points:
        return []
    avg_col = _estimate_avg_column(frame)
    if avg_col is None:
        return []

    latest_period = actual_points[-1]["period"]
    estimates = []
    for row_key, row in frame.iterrows():
        key = str(row_key).strip().lower()
        is_quarter = "q" in key or "quarter" in key
        is_year = "y" in key or "year" in key
        if not is_year or is_quarter:
            continue
        num = _safe_number(row.get(avg_col))
        if num is None:
            continue
        period = _add_months(latest_period, 12 * (len(estimates) + 1))
        if period is None:
            continue
        estimates.append({
            "period": period,
            "label": f"{_fiscal_label(period, True, fiscal_year_end_month)} (E)",
            "value": round(num, 4),
            "estimate": True,
        })
        if len(estimates) >= limit:
            break
    return estimates


def _series_from_earnings_history(
    frame,
    limit: int,
    *,
    fiscal_year_end_month: int,
    reported_dates: bool = False,
) -> list[dict]:
    if frame is None or getattr(frame, "empty", True):
        return []
    normalized = {str(col).strip().lower(): col for col in frame.columns}
    actual_col = None
    for key in ("epsactual", "eps actual", "reported eps", "actual"):
        if key in normalized:
            actual_col = normalized[key]
            break
    if actual_col is None:
        for key, col in normalized.items():
            if "eps" in key and "actual" in key:
                actual_col = col
                break
    if actual_col is None:
        return []

    period_col = None
    for key in ("period", "quarter", "date"):
        if key in normalized:
            period_col = normalized[key]
            break

    points = []
    for row_key, row in frame.iterrows():
        num = _safe_number(row.get(actual_col))
        if num is None:
            continue
        period_value = row.get(period_col) if period_col is not None else row_key
        period_label = _period_label(period_value)
        if reported_dates:
            period_label = _reported_period_from_earnings_date(period_label, fiscal_year_end_month)
        if not period_label or len(period_label) < 7:
            continue
        points.append({
            "period": period_label,
            "label": _fiscal_label(period_label, False, fiscal_year_end_month),
            "value": round(num, 4),
        })

    points.sort(key=lambda item: item["period"])
    return points[-limit:]


def _annual_eps_from_quarters(points: list[dict], *, fiscal_year_end_month: int, limit: int) -> list[dict]:
    grouped: dict[str, dict] = {}
    for point in points:
        if point.get("estimate"):
            continue
        period = point.get("period")
        value = _safe_number(point.get("value"))
        if not period or value is None:
            continue
        label = _fiscal_label(period, True, fiscal_year_end_month)
        bucket = grouped.setdefault(label, {"period": period, "label": label, "value": 0.0, "count": 0})
        bucket["value"] += value
        bucket["count"] += 1
        if period > bucket["period"]:
            bucket["period"] = period

    annual = [
        {"period": item["period"], "label": item["label"], "value": round(item["value"], 4)}
        for item in grouped.values()
        if item["count"] >= 4
    ]
    annual.sort(key=lambda item: item["period"])
    return annual[-limit:]


def _merge_missing_periods(primary: list[dict], fallback: list[dict], limit: int) -> list[dict]:
    merged = {point.get("period"): dict(point) for point in primary if point.get("period")}
    for point in fallback:
        period = point.get("period")
        if period and period not in merged:
            merged[period] = dict(point)
    points = sorted(merged.values(), key=lambda item: item["period"])
    return points[-limit:]


def _profile_eps_series(
    actual_points: list[dict],
    estimate_points: list[dict],
    *,
    actual_limit: int,
    estimate_limit: int,
) -> list[dict]:
    actual = [dict(point) for point in actual_points if not point.get("estimate")]
    estimates = [dict(point) for point in estimate_points if point.get("estimate")]
    actual.sort(key=lambda item: item.get("period", ""))
    estimates.sort(key=lambda item: item.get("period", ""))
    return actual[-actual_limit:] + estimates[:estimate_limit]


def _get_ticker_info(ticker_obj) -> dict:
    try:
        info = ticker_obj.get_info()
    except Exception:
        try:
            info = ticker_obj.info
        except Exception:
            info = {}
    return info if isinstance(info, dict) else {}


def _get_ticker_currencies(ticker_obj) -> tuple[str | None, str | None]:
    quote_currency = None
    financial_currency = None

    try:
        fast_info = ticker_obj.fast_info
        if fast_info:
            quote_currency = str(fast_info.get("currency") or "").upper() or None
    except Exception:
        pass

    info = _get_ticker_info(ticker_obj)
    quote_currency = str(info.get("currency") or quote_currency or "").upper() or None
    financial_currency = str(info.get("financialCurrency") or "").upper() or None

    return financial_currency, quote_currency


def _recent_fx_close(symbol: str, session) -> float | None:
    kwargs = {"session": session} if session is not None else {}
    try:
        hist = yf.Ticker(symbol, **kwargs).history(period="5d")
        if hist is None or hist.empty or "Close" not in hist:
            return None
        closes = hist["Close"].dropna()
        if closes.empty:
            return None
        value = _safe_number(closes.iloc[-1])
        return value if value and value > 0 else None
    except Exception:
        return None


def _fx_rate(from_currency: str | None, to_currency: str | None, session) -> float | None:
    if not from_currency or not to_currency:
        return None
    from_currency = from_currency.upper()
    to_currency = to_currency.upper()
    if from_currency == to_currency:
        return 1.0

    key = (from_currency, to_currency)
    if key in FX_RATE_CACHE:
        return FX_RATE_CACHE[key]

    attempts: list[tuple[str, bool]] = [
        (f"{from_currency}{to_currency}=X", False),
        (f"{to_currency}{from_currency}=X", True),
    ]
    # Yahoo commonly exposes USD cross rates like TWD=X as USD/TWD.
    if to_currency == "USD":
        attempts.append((f"{from_currency}=X", True))
    if from_currency == "USD":
        attempts.append((f"{to_currency}=X", False))

    for symbol, invert in attempts:
        close = _recent_fx_close(symbol, session)
        if close:
            rate = 1 / close if invert else close
            FX_RATE_CACHE[key] = rate
            return rate

    FX_RATE_CACHE[key] = None
    return None


def _looks_like_eps_currency_mismatch(actual_points: list[dict], estimate_points: list[dict]) -> bool:
    actual = _median([float(p["value"]) for p in actual_points[-4:] if not p.get("estimate")])
    estimates = _median([float(p["value"]) for p in estimate_points if p.get("estimate")])
    if actual is None or estimates is None or estimates <= 0:
        return False
    ratio = actual / estimates
    return ratio >= 8


def _convert_eps_points(points: list[dict], rate: float, from_currency: str, to_currency: str) -> list[dict]:
    converted = []
    for point in points:
        item = dict(point)
        if not item.get("estimate"):
            item["value"] = round(float(item["value"]) * rate, 4)
            item["converted_from_currency"] = from_currency
            item["currency"] = to_currency
        converted.append(item)
    return converted


def _safe_percent(value):
    num = _safe_number(value)
    if num is None:
        return None
    pct = num * 100 if abs(num) <= 1 else num
    if not math.isfinite(pct) or pct < 0:
        return None
    return round(pct, 4)


def _series_value(row, names: tuple[str, ...]):
    if row is None:
        return None
    normalized = {str(key).strip().lower(): key for key in getattr(row, "index", [])}
    for name in names:
        key = normalized.get(name.lower())
        if key is not None:
            try:
                return row.get(key)
            except Exception:
                return row[key]
    return None


def _major_holder_value(frame, key: str):
    if frame is None or getattr(frame, "empty", True):
        return None
    try:
        if key in frame.index:
            row = frame.loc[key]
            if hasattr(row, "iloc"):
                return row.iloc[0]
            return row
    except Exception:
        pass
    return None


def _is_filtered_institutional_holder(name: str) -> bool:
    clean = " ".join(str(name or "").lower().replace("&", " and ").split())
    if not clean:
        return True
    return any(term in clean for term in OWNERSHIP_HOLDER_EXCLUDE_TERMS)


def _get_major_holders(ticker_obj):
    try:
        frame = ticker_obj.get_major_holders()
        if frame is not None and not frame.empty:
            return frame
    except Exception:
        pass
    try:
        frame = ticker_obj.major_holders
        if frame is not None and not frame.empty:
            return frame
    except Exception:
        pass
    return None


def _get_institutional_holders(ticker_obj):
    try:
        frame = ticker_obj.get_institutional_holders()
        if frame is not None and not frame.empty:
            return frame
    except Exception:
        pass
    try:
        frame = ticker_obj.institutional_holders
        if frame is not None and not frame.empty:
            return frame
    except Exception:
        pass
    return None


def _institutional_holder_rows(frame) -> list[dict]:
    if frame is None or getattr(frame, "empty", True):
        return []
    rows = []
    for _, row in frame.iterrows():
        holder = _series_value(row, ("Holder", "holder"))
        holder = str(holder or "").strip()
        if _is_filtered_institutional_holder(holder):
            continue
        shares = _safe_number(_series_value(row, ("Shares", "shares")))
        pct_out = _safe_percent(_series_value(row, ("pctHeld", "pct held", "% Out", "% out")))
        value = _safe_number(_series_value(row, ("Value", "value")))
        date_reported = _series_value(row, ("Date Reported", "date reported", "Date", "date"))
        item = {"holder": holder}
        if shares is not None:
            item["shares"] = int(round(shares))
        if pct_out is not None:
            item["pct_out"] = pct_out
        if value is not None:
            item["value"] = int(round(value))
        if date_reported is not None:
            item["date_reported"] = _period_label(date_reported)
        rows.append(item)
    rows.sort(key=lambda item: item.get("shares", 0), reverse=True)
    return rows[:12]


def _ownership_float_metrics(info: dict) -> dict:
    reported_shares_outstanding = _safe_number(info.get("sharesOutstanding"))
    implied_shares_outstanding = _safe_number(info.get("impliedSharesOutstanding"))
    public_float = _safe_number(info.get("floatShares"))
    shares_outstanding = reported_shares_outstanding
    if implied_shares_outstanding is not None and (
        shares_outstanding is None
        or (
            public_float is not None
            and shares_outstanding < public_float
            and implied_shares_outstanding >= public_float
        )
    ):
        shares_outstanding = implied_shares_outstanding
    if (
        shares_outstanding is not None
        and public_float is not None
        and shares_outstanding > 0
        and public_float > shares_outstanding
    ):
        if public_float <= shares_outstanding * 1.05:
            public_float = shares_outstanding
        else:
            public_float = None
    short_pct_float = _safe_percent(info.get("shortPercentOfFloat"))
    shares_short = _safe_number(info.get("sharesShort"))
    if short_pct_float is None and public_float and public_float > 0 and shares_short is not None:
        short_pct_float = round(max(0, shares_short) / public_float * 100, 4)

    metrics: dict = {}
    if shares_outstanding is not None:
        metrics["shares_outstanding"] = int(round(shares_outstanding))
    if public_float is not None:
        metrics["public_float"] = int(round(public_float))
    if shares_outstanding and shares_outstanding > 0 and public_float is not None:
        metrics["float_pct_of_tso"] = round(max(0, public_float) / shares_outstanding * 100, 4)
    if short_pct_float is not None:
        metrics["short_pct_float"] = short_pct_float
    return metrics


def _fetch_ownership(ticker_obj) -> dict:
    ownership: dict = {}

    major = _get_major_holders(ticker_obj)
    institutional_pct = _safe_percent(_major_holder_value(major, "institutionsPercentHeld"))
    insider_pct = _safe_percent(_major_holder_value(major, "insidersPercentHeld"))
    institutions_count = _safe_number(_major_holder_value(major, "institutionsCount"))
    float_metrics = _ownership_float_metrics(_get_ticker_info(ticker_obj))
    if institutional_pct is not None or insider_pct is not None or float_metrics:
        breakdown: dict = {}
        if institutional_pct is not None:
            breakdown["institutional"] = institutional_pct
        if insider_pct is not None:
            breakdown["insider"] = insider_pct
        if institutional_pct is not None or insider_pct is not None:
            breakdown["other"] = round(
                max(0, 100 - (institutional_pct or 0) - (insider_pct or 0)),
                4,
            )
        if institutions_count is not None:
            breakdown["institutions_count"] = int(round(institutions_count))
        breakdown.update(float_metrics)
        ownership["breakdown"] = breakdown

    holder_rows = _institutional_holder_rows(_get_institutional_holders(ticker_obj))
    if holder_rows:
        ownership["institutional_holders"] = holder_rows

    return ownership


def _ownership_has_breakdown(ownership: dict) -> bool:
    breakdown = ownership.get("breakdown") if isinstance(ownership, dict) else None
    return isinstance(breakdown, dict) and (
        breakdown.get("institutional") is not None
        or breakdown.get("insider") is not None
    )


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


def _fetch_one(ticker: str, session, include_ownership: bool = True) -> dict:
    kwargs = {"session": session} if session is not None else {}
    ticker_obj = yf.Ticker(ticker, **kwargs)
    quarterly = _get_income_statement(ticker_obj, "quarterly")
    annual = _get_income_statement(ticker_obj, "yearly")
    fiscal_year_end_month = _infer_fiscal_year_end_month(annual)
    revenue_estimate = _get_estimate_frame(ticker_obj, "revenue_estimate")
    earnings_estimate = _get_estimate_frame(ticker_obj, "earnings_estimate")
    earnings_dates = _get_earnings_dates(ticker_obj, max(MAX_QUARTERS + 4, 24))
    earnings_history = _get_estimate_frame(ticker_obj, "earnings_history")
    ownership = _fetch_ownership(ticker_obj) if include_ownership else {}

    quarterly_revenue = _series_from_income_statement(
        quarterly,
        ("Total Revenue", "Operating Revenue"),
        MAX_QUARTERS,
        annual=False,
        fiscal_year_end_month=fiscal_year_end_month,
    )
    quarterly_eps_gaap = _series_from_income_statement(
        quarterly,
        ("Diluted EPS", "Basic EPS", "Diluted EPS Other Gains Losses"),
        MAX_QUARTERS,
        annual=False,
        fiscal_year_end_month=fiscal_year_end_month,
    )
    reported_eps = _series_from_earnings_history(
        earnings_dates,
        MAX_QUARTERS,
        fiscal_year_end_month=fiscal_year_end_month,
        reported_dates=True,
    )
    history_eps = _series_from_earnings_history(
        earnings_history,
        MAX_QUARTERS,
        fiscal_year_end_month=fiscal_year_end_month,
    )
    non_gaap_uses_gaap_fallback = not reported_eps and not history_eps
    quarterly_eps_non_gaap = reported_eps or history_eps or quarterly_eps_gaap
    quarterly_eps_estimates = _quarterly_estimates(
        earnings_estimate,
        quarterly_eps_non_gaap,
        fiscal_year_end_month=fiscal_year_end_month,
    )
    annual_eps_gaap = _series_from_income_statement(
        annual,
        ("Diluted EPS", "Basic EPS", "Diluted EPS Other Gains Losses"),
        MAX_YEARS,
        annual=True,
        fiscal_year_end_month=fiscal_year_end_month,
    )
    annual_eps_non_gaap = _annual_eps_from_quarters(
        quarterly_eps_non_gaap,
        fiscal_year_end_month=fiscal_year_end_month,
        limit=MAX_YEARS,
    ) or annual_eps_gaap
    annual_eps_estimates = _annual_estimates(
        earnings_estimate,
        annual_eps_non_gaap or annual_eps_gaap,
        fiscal_year_end_month=fiscal_year_end_month,
    )

    if _looks_like_eps_currency_mismatch(quarterly_eps_gaap, quarterly_eps_estimates):
        financial_currency, quote_currency = _get_ticker_currencies(ticker_obj)
        rate = _fx_rate(financial_currency, quote_currency, session)
        if rate and rate > 0:
            quarterly_eps_gaap = _convert_eps_points(quarterly_eps_gaap, rate, financial_currency or "", quote_currency or "")
            annual_eps_gaap = _convert_eps_points(annual_eps_gaap, rate, financial_currency or "", quote_currency or "")
            if non_gaap_uses_gaap_fallback:
                quarterly_eps_non_gaap = quarterly_eps_gaap
                annual_eps_non_gaap = annual_eps_gaap

    if reported_eps:
        quarterly_eps_gaap = _merge_missing_periods(quarterly_eps_gaap, reported_eps, MAX_QUARTERS)

    quarterly_eps_gaap_profile = _profile_eps_series(
        quarterly_eps_gaap,
        quarterly_eps_estimates,
        actual_limit=PROFILE_QUARTERLY_EPS_ACTUALS,
        estimate_limit=PROFILE_QUARTERLY_EPS_ESTIMATES,
    )
    quarterly_eps_non_gaap_profile = _profile_eps_series(
        quarterly_eps_non_gaap,
        quarterly_eps_estimates,
        actual_limit=PROFILE_QUARTERLY_EPS_ACTUALS,
        estimate_limit=PROFILE_QUARTERLY_EPS_ESTIMATES,
    )
    annual_eps_gaap_profile = _profile_eps_series(
        annual_eps_gaap,
        annual_eps_estimates,
        actual_limit=PROFILE_ANNUAL_EPS_ACTUALS,
        estimate_limit=PROFILE_ANNUAL_EPS_ESTIMATES,
    )
    annual_eps_non_gaap_profile = _profile_eps_series(
        annual_eps_non_gaap,
        annual_eps_estimates,
        actual_limit=PROFILE_ANNUAL_EPS_ACTUALS,
        estimate_limit=PROFILE_ANNUAL_EPS_ESTIMATES,
    )

    result = {
        "quarterly_revenue": quarterly_revenue + _quarterly_estimates(
            revenue_estimate,
            quarterly_revenue,
            fiscal_year_end_month=fiscal_year_end_month,
        ),
        "annual_revenue": _series_from_income_statement(
            annual,
            ("Total Revenue", "Operating Revenue"),
            MAX_YEARS,
            annual=True,
            fiscal_year_end_month=fiscal_year_end_month,
        ),
        "quarterly_eps": quarterly_eps_non_gaap_profile,
        "annual_eps": annual_eps_non_gaap_profile,
        "quarterly_eps_gaap": quarterly_eps_gaap_profile,
        "annual_eps_gaap": annual_eps_gaap_profile,
        "quarterly_eps_non_gaap": quarterly_eps_non_gaap_profile,
        "annual_eps_non_gaap": annual_eps_non_gaap_profile,
    }
    if ownership:
        result["ownership"] = ownership
    return result


def _record_has_data(data: dict) -> bool:
    return any(
        (isinstance(value, list) and value)
        or (isinstance(value, dict) and value)
        for value in data.values()
    )


def _fetch_ownership_for_ticker(
    ticker: str,
    session=None,
    attempts: int = 1,
    retry_delay: float = 1.0,
) -> tuple[str, dict, Exception | None]:
    last_error: Exception | None = None
    for attempt in range(max(1, attempts)):
        try:
            active_session = session if attempt == 0 else _build_session()
            kwargs = {"session": active_session} if active_session is not None else {}
            ownership = _fetch_ownership(yf.Ticker(ticker, **kwargs))
            if _ownership_has_breakdown(ownership):
                return ticker, ownership, None
            last_error = RuntimeError("Yahoo returned no ownership breakdown")
        except Exception as exc:
            last_error = exc
        if attempt + 1 < max(1, attempts):
            time.sleep(max(0.0, retry_delay) * (2 ** attempt))
    return ticker, {}, last_error


def _fetch_ownership_metrics_for_ticker(ticker: str, session=None) -> tuple[str, dict, Exception | None]:
    try:
        kwargs = {"session": session} if session is not None else {}
        info = _get_ticker_info(yf.Ticker(ticker, **kwargs))
        return ticker, _ownership_float_metrics(info), None
    except Exception as exc:
        return ticker, {}, exc


def _refresh_ownership_metrics_only(tickers: list[str], session, universe_count: int, workers: int) -> int:
    payload = _load_existing_payload(universe_count)
    updated = 0
    skipped = 0

    print(f"Refreshing ownership float metrics for {len(tickers)} stock(s)...")
    workers = max(1, int(workers or 1))

    def store(ticker: str, metrics: dict, exc: Exception | None) -> None:
        nonlocal updated, skipped
        if exc is not None:
            skipped += 1
            print(f"  {ticker}: ownership metrics failed ({exc})")
            return
        if not metrics:
            skipped += 1
            return
        rec = payload["tickers"].setdefault(ticker, {})
        ownership = rec.setdefault("ownership", {})
        breakdown = ownership.setdefault("breakdown", {})
        breakdown.update(metrics)
        updated += 1

    if workers == 1:
        for i, ticker in enumerate(tickers, start=1):
            if i % 50 == 0:
                print(f"  {i}/{len(tickers)}...")
            ticker, metrics, exc = _fetch_ownership_metrics_for_ticker(ticker, session)
            store(ticker, metrics, exc)
            time.sleep(0.05)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_fetch_ownership_metrics_for_ticker, ticker, None): ticker for ticker in tickers}
            for i, future in enumerate(as_completed(futures), start=1):
                if i % 50 == 0:
                    print(f"  {i}/{len(tickers)}...")
                ticker, metrics, exc = future.result()
                store(ticker, metrics, exc)

    payload.setdefault("meta", {})
    payload["meta"]["ownership_metrics_updated"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    payload["meta"]["ownership_count"] = sum(
        1 for rec in payload.get("tickers", {}).values()
        if isinstance(rec, dict) and isinstance(rec.get("ownership"), dict) and rec["ownership"]
    )
    payload["meta"]["ownership_metrics_count"] = sum(
        1 for rec in payload.get("tickers", {}).values()
        if isinstance(rec, dict)
        and isinstance(rec.get("ownership"), dict)
        and isinstance(rec["ownership"].get("breakdown"), dict)
        and any(
            key in rec["ownership"]["breakdown"]
            for key in ("shares_outstanding", "public_float", "float_pct_of_tso", "short_pct_float")
        )
    )
    _write_payload(payload, universe_count)
    print(
        f"Wrote {OUT_JSON.name}: ownership metrics updated for {updated}/{len(tickers)} stock(s); "
        f"skipped {skipped}; total metric records {payload['meta']['ownership_metrics_count']}"
    )
    return 0


def _refresh_ownership_only(
    tickers: list[str],
    session,
    universe_count: int,
    workers: int,
    retries: int,
    retry_delay: float,
    request_delay: float,
    min_success_ratio: float,
) -> int:
    payload = _load_existing_payload(universe_count)
    previous_count = _ownership_record_count(payload)
    updated = 0
    skipped = 0

    print(f"Refreshing ownership for {len(tickers)} stock(s)...")
    workers = max(1, int(workers or 1))

    def store(ticker: str, ownership: dict, exc: Exception | None) -> None:
        nonlocal updated, skipped
        if exc is not None:
            skipped += 1
            print(f"  {ticker}: ownership failed ({exc})")
            return
        if ownership:
            rec = payload["tickers"].setdefault(ticker, {})
            rec["ownership"] = _merge_cached_ownership(
                {"ownership": ownership},
                rec.get("ownership"),
            )["ownership"]
            updated += 1
        else:
            skipped += 1

    if workers == 1:
        for i, ticker in enumerate(tickers, start=1):
            if i % 50 == 0:
                print(f"  {i}/{len(tickers)}...")
            ticker, ownership, exc = _fetch_ownership_for_ticker(
                ticker,
                session,
                attempts=retries,
                retry_delay=retry_delay,
            )
            store(ticker, ownership, exc)
            if i < len(tickers):
                time.sleep(max(0.0, request_delay))
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    _fetch_ownership_for_ticker,
                    ticker,
                    None,
                    retries,
                    retry_delay,
                ): ticker
                for ticker in tickers
            }
            for i, future in enumerate(as_completed(futures), start=1):
                if i % 50 == 0:
                    print(f"  {i}/{len(tickers)}...")
                ticker, ownership, exc = future.result()
                store(ticker, ownership, exc)

    success_ratio = updated / len(tickers) if tickers else 1.0
    if success_ratio < min_success_ratio:
        print(
            f"ERROR: ownership refresh succeeded for only {updated}/{len(tickers)} stock(s) "
            f"({success_ratio:.1%}); required {min_success_ratio:.1%}. "
            f"Existing {previous_count} ownership records were left untouched."
        )
        return 1

    payload.setdefault("meta", {})
    payload["meta"]["ownership_updated"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    payload["meta"]["ownership_count"] = _ownership_record_count(payload)
    payload["meta"]["ownership_refresh_succeeded"] = updated
    payload["meta"]["ownership_refresh_failed"] = skipped
    payload["meta"]["ownership_refresh_success_ratio"] = round(success_ratio, 4)
    _write_payload(payload, universe_count)
    print(
        f"Wrote {OUT_JSON.name}: ownership updated for {updated}/{len(tickers)} stock(s); "
        f"skipped {skipped}; total ownership records {payload['meta']['ownership_count']}"
    )
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch static stock fundamentals from Yahoo Finance.")
    parser.add_argument(
        "--ownership-only",
        action="store_true",
        help="Refresh ownership fields only and preserve existing fundamentals data.",
    )
    parser.add_argument(
        "--ownership-metrics-only",
        action="store_true",
        help="Refresh shares outstanding, public float, and short-float metrics only.",
    )
    parser.add_argument(
        "--skip-ownership",
        action="store_true",
        help="Skip ownership fetching during the full fundamentals refresh.",
    )
    parser.add_argument(
        "--tickers",
        help="Optional comma-separated ticker filter, useful for targeted refreshes like BE,AMD.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel workers for --ownership-only. Use 1 for sequential fetching.",
    )
    parser.add_argument(
        "--ownership-retries",
        type=int,
        default=3,
        help="Attempts per ticker during --ownership-only.",
    )
    parser.add_argument(
        "--ownership-retry-delay",
        type=float,
        default=1.0,
        help="Initial exponential-backoff delay in seconds for ownership retries.",
    )
    parser.add_argument(
        "--ownership-delay",
        type=float,
        default=0.25,
        help="Delay between sequential ownership requests.",
    )
    parser.add_argument(
        "--min-ownership-success-ratio",
        type=float,
        default=0.9,
        help="Abort without writing when ownership refresh success falls below this ratio.",
    )
    parser.add_argument(
        "--restore-ownership-from",
        type=Path,
        help="Restore only missing ownership records from a known-good fundamentals JSON snapshot.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    all_tickers = _load_universe()
    tickers = _filter_tickers(all_tickers, _parse_ticker_filter(args.tickers))
    session = _build_session()

    if args.restore_ownership_from:
        return _restore_ownership_from_snapshot(args.restore_ownership_from, len(all_tickers))

    if yf is None:
        print("ERROR: yfinance is required for network refresh modes")
        return 1

    if args.ownership_metrics_only:
        return _refresh_ownership_metrics_only(tickers, session, len(all_tickers), args.workers)

    if args.ownership_only:
        return _refresh_ownership_only(
            tickers,
            session,
            len(all_tickers),
            args.workers,
            max(1, args.ownership_retries),
            max(0.0, args.ownership_retry_delay),
            max(0.0, args.ownership_delay),
            min(1.0, max(0.0, args.min_ownership_success_ratio)),
        )

    existing = _load_existing_payload(len(all_tickers))
    active_tickers = set(all_tickers)
    payload = _empty_payload(len(all_tickers))
    payload["meta"].update(copy.deepcopy(existing.get("meta", {})))
    payload["tickers"] = {
        ticker: copy.deepcopy(record)
        for ticker, record in existing.get("tickers", {}).items()
        if ticker in active_tickers and isinstance(record, dict)
    }

    print(f"Fetching fundamentals for {len(tickers)} stock(s)...")
    for i, ticker in enumerate(tickers, start=1):
        if i % 50 == 0:
            print(f"  {i}/{len(tickers)}...")
        try:
            data = _fetch_one(ticker, session, include_ownership=not args.skip_ownership)
            if _record_has_data(data):
                cached = payload["tickers"].get(ticker, {}).get("ownership")
                data = _merge_cached_ownership(data, cached)
                payload["tickers"][ticker] = data
        except Exception as exc:
            print(f"  {ticker}: failed ({exc})")
        time.sleep(0.05)

    _write_payload(payload, len(all_tickers))
    print(f"Wrote {OUT_JSON.name}: {payload['meta']['count']}/{len(tickers)} stock(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
