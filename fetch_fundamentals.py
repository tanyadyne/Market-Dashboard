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
from calendar import monthrange
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
FX_RATE_CACHE: dict[tuple[str, str], float | None] = {}


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
    limit: int = 2,
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


def _get_ticker_currencies(ticker_obj) -> tuple[str | None, str | None]:
    quote_currency = None
    financial_currency = None

    try:
        fast_info = ticker_obj.fast_info
        if fast_info:
            quote_currency = str(fast_info.get("currency") or "").upper() or None
    except Exception:
        pass

    try:
        info = ticker_obj.get_info()
    except Exception:
        try:
            info = ticker_obj.info
        except Exception:
            info = {}

    if isinstance(info, dict):
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
    fiscal_year_end_month = _infer_fiscal_year_end_month(annual)
    revenue_estimate = _get_estimate_frame(ticker_obj, "revenue_estimate")
    earnings_estimate = _get_estimate_frame(ticker_obj, "earnings_estimate")

    quarterly_revenue = _series_from_income_statement(
        quarterly,
        ("Total Revenue", "Operating Revenue"),
        MAX_QUARTERS,
        annual=False,
        fiscal_year_end_month=fiscal_year_end_month,
    )
    quarterly_eps = _series_from_income_statement(
        quarterly,
        ("Diluted EPS", "Basic EPS", "Diluted EPS Other Gains Losses"),
        MAX_QUARTERS,
        annual=False,
        fiscal_year_end_month=fiscal_year_end_month,
    )
    quarterly_eps_estimates = _quarterly_estimates(
        earnings_estimate,
        quarterly_eps,
        fiscal_year_end_month=fiscal_year_end_month,
    )
    annual_eps = _series_from_income_statement(
        annual,
        ("Diluted EPS", "Basic EPS", "Diluted EPS Other Gains Losses"),
        MAX_YEARS,
        annual=True,
        fiscal_year_end_month=fiscal_year_end_month,
    )

    if _looks_like_eps_currency_mismatch(quarterly_eps, quarterly_eps_estimates):
        financial_currency, quote_currency = _get_ticker_currencies(ticker_obj)
        rate = _fx_rate(financial_currency, quote_currency, session)
        if rate and rate > 0:
            quarterly_eps = _convert_eps_points(quarterly_eps, rate, financial_currency or "", quote_currency or "")
            annual_eps = _convert_eps_points(annual_eps, rate, financial_currency or "", quote_currency or "")

    return {
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
        "quarterly_eps": quarterly_eps + quarterly_eps_estimates,
        "annual_eps": annual_eps,
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
            if any(isinstance(value, list) and value for value in data.values()):
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
