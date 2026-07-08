#!/usr/bin/env python3
"""
Fetch quarterly shares outstanding history from SEC EDGAR companyfacts.

This builds the ownership tab's shares outstanding tile from public SEC data:
  - map screener tickers to SEC CIKs via company_tickers.json
  - fetch/cached companyfacts JSON for each CIK
  - extract dei:EntityCommonStockSharesOutstanding facts from 2011 onward
  - collapse duplicate facts to one point per reported period
  - emit ownership_shares_outstanding_sec.json keyed by ticker
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
DEFAULT_UNIVERSE = Path("screener_tickers.json")
DEFAULT_OUTPUT = Path("ownership_shares_outstanding_sec.json")
DEFAULT_CACHE_DIR = Path(".sec_companyfacts_cache")
DEFAULT_USER_AGENT = "WIN_MarketDashboard/1.0 contact:tanya@example.com"
SHARES_CONCEPT = "EntityCommonStockSharesOutstanding"
MANUAL_CIK_OVERRIDES = {
    # The screener uses the widely recognized/common-market symbols, while SEC's
    # current ticker map now lists these issuers under newer/alternate symbols.
    "BK": {"cik": "0001390777", "title": "Bank of New York Mellon Corp"},
    "VSCO": {"cik": "0001856437", "title": "Victoria's Secret & Co."},
}
PERIODIC_FORMS = {
    "10-K",
    "10-K/A",
    "10-Q",
    "10-Q/A",
    "20-F",
    "20-F/A",
    "40-F",
    "40-F/A",
    "6-K",
    "6-K/A",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_env_file(path: Path = Path(".env.local")) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip().lstrip("\ufeff"), value.strip().strip('"').strip("'"))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_universe(path: Path) -> list[str]:
    data = read_json(path)
    if isinstance(data, dict):
        values = data.get("tickers", data)
        if isinstance(values, dict):
            return sorted(str(ticker).upper() for ticker in values if ticker)
        if isinstance(values, list):
            return sorted(str(ticker).upper() for ticker in values if ticker)
    if isinstance(data, list):
        return sorted(str(ticker).upper() for ticker in data if ticker)
    raise ValueError(f"Unsupported universe file shape: {path}")


def parse_int(value: Any) -> int | None:
    try:
        if value in ("", None):
            return None
        parsed = int(round(float(value)))
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if parsed > 0 else None


def date_key(value: Any) -> tuple[int, int, int]:
    try:
        dt = datetime.strptime(str(value or ""), "%Y-%m-%d")
        return dt.year, dt.month, dt.day
    except ValueError:
        return 0, 0, 0


def filed_key(value: Any) -> tuple[int, int, int]:
    return date_key(value)


def cik_padded(cik: Any) -> str:
    return str(cik).strip().zfill(10)


def ticker_variants(ticker: str) -> list[str]:
    raw = str(ticker or "").upper().strip()
    variants = [raw]
    variants.append(raw.replace(".", "-"))
    variants.append(raw.replace("-", "."))
    variants.append(raw.replace("/", "-"))
    out: list[str] = []
    for item in variants:
        if item and item not in out:
            out.append(item)
    return out


def request_json(url: str, user_agent: str, timeout: int) -> Any:
    req = Request(url, headers={"User-Agent": user_agent, "Accept": "application/json"})
    with urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def cached_json(url: str, cache_path: Path, user_agent: str, timeout: int, refresh: bool = False) -> Any:
    if cache_path.exists() and not refresh:
        return read_json(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    data = request_json(url, user_agent, timeout)
    write_json(cache_path, data)
    return data


def load_company_tickers(cache_dir: Path, user_agent: str, timeout: int, refresh: bool) -> dict[str, dict[str, Any]]:
    raw = cached_json(COMPANY_TICKERS_URL, cache_dir / "company_tickers.json", user_agent, timeout, refresh)
    mapping: dict[str, dict[str, Any]] = {}
    values = raw.values() if isinstance(raw, dict) else raw
    for row in values:
        if not isinstance(row, dict):
            continue
        ticker = str(row.get("ticker") or "").upper().strip()
        cik = row.get("cik_str")
        if ticker and cik:
            mapping[ticker] = {"cik": cik_padded(cik), "title": row.get("title")}
    return mapping


def lookup_cik(ticker: str, company_tickers: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    override = MANUAL_CIK_OVERRIDES.get(str(ticker or "").upper().strip())
    if override:
        return override
    for variant in ticker_variants(ticker):
        found = company_tickers.get(variant)
        if found:
            return found
    return None


def period_key(fact: dict[str, Any]) -> str:
    frame = str(fact.get("frame") or "").upper()
    match = re.match(r"CY(\d{4})Q([1-4])I?$", frame)
    if match:
        return f"{match.group(1)}Q{match.group(2)}"
    fy = fact.get("fy")
    fp = str(fact.get("fp") or "").upper()
    if fy and fp:
        return f"{fy}{fp}"
    year, month, _ = date_key(fact.get("end"))
    quarter = ((month - 1) // 3 + 1) if month else 0
    return f"{year}Q{quarter}" if year and quarter else str(fact.get("end") or "")


def period_label(fact: dict[str, Any]) -> str:
    key = period_key(fact)
    match = re.match(r"(\d{4})Q([1-4])$", key)
    if match:
        return f"Q{match.group(2)} {match.group(1)}"
    match = re.match(r"(\d{4})FY$", key)
    if match:
        return f"FY {match.group(1)}"
    return key


def fact_sort_key(fact: dict[str, Any]) -> tuple[tuple[int, int, int], str, int]:
    return filed_key(fact.get("filed")), str(fact.get("accn") or ""), parse_int(fact.get("val")) or 0


def extract_share_points(companyfacts: dict[str, Any], start_year: int) -> list[dict[str, Any]]:
    concept = (
        companyfacts.get("facts", {})
        .get("dei", {})
        .get(SHARES_CONCEPT, {})
        .get("units", {})
        .get("shares", [])
    )
    if not isinstance(concept, list):
        return []

    by_period: dict[str, dict[str, Any]] = {}
    for fact in concept:
        if not isinstance(fact, dict):
            continue
        form = str(fact.get("form") or "").upper()
        if form not in PERIODIC_FORMS:
            continue
        end = str(fact.get("end") or "")
        year, _, _ = date_key(end)
        if year < start_year:
            continue
        value = parse_int(fact.get("val"))
        if not value:
            continue
        key = period_key(fact)
        current = by_period.get(key)
        if current is None or fact_sort_key(fact) >= fact_sort_key(current):
            by_period[key] = fact

    points: list[dict[str, Any]] = []
    for key, fact in by_period.items():
        value = parse_int(fact.get("val"))
        end = str(fact.get("end") or "")
        if not value or not end:
            continue
        points.append(
            {
                "date": end,
                "period": period_label(fact),
                "value": value,
            }
        )
    points.sort(key=lambda item: (date_key(item["date"]), str(item.get("period") or "")))
    return points


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    load_env_file()
    user_agent = args.user_agent or os.environ.get("SEC_USER_AGENT") or DEFAULT_USER_AGENT
    tickers = read_universe(args.universe)
    company_tickers = load_company_tickers(args.cache_dir, user_agent, args.timeout, args.refresh_tickers)
    counters: Counter[str] = Counter()
    errors: list[dict[str, str]] = []
    missing_cik: list[str] = []
    tickers_out: dict[str, dict[str, Any]] = {}

    cik_cache: dict[str, dict[str, Any] | None] = {}
    for index, ticker in enumerate(tickers, start=1):
        cik_row = lookup_cik(ticker, company_tickers)
        if not cik_row:
            missing_cik.append(ticker)
            counters["missing_cik"] += 1
            tickers_out[ticker] = {"ticker": ticker, "points": []}
            continue

        cik = cik_row["cik"]
        companyfacts = cik_cache.get(cik)
        if cik not in cik_cache:
            cache_path = args.cache_dir / "companyfacts" / f"CIK{cik}.json"
            try:
                companyfacts = cached_json(
                    COMPANYFACTS_URL.format(cik=cik),
                    cache_path,
                    user_agent,
                    args.timeout,
                    args.refresh_facts,
                )
                counters["companyfacts_loaded"] += 1
            except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
                errors.append({"ticker": ticker, "cik": cik, "error": str(exc)[:240]})
                counters["companyfacts_failed"] += 1
                companyfacts = None
            cik_cache[cik] = companyfacts
            if args.request_delay and index < len(tickers):
                time.sleep(args.request_delay)

        points = extract_share_points(companyfacts or {}, args.start_year)
        if points:
            counters["tickers_with_points"] += 1
        else:
            counters["tickers_without_points"] += 1
        tickers_out[ticker] = {
            "ticker": ticker,
            "cik": cik,
            "company": cik_row.get("title"),
            "points": points,
        }

        if args.progress_every and index % args.progress_every == 0:
            print(f"Processed {index}/{len(tickers)} tickers; with_points={counters['tickers_with_points']}")

    meta = {
        "updated": now_iso(),
        "source": "SEC EDGAR companyfacts: dei:EntityCommonStockSharesOutstanding",
        "start_year": args.start_year,
        "universe_file": str(args.universe),
        "universe_count": len(tickers),
        "company_tickers_url": COMPANY_TICKERS_URL,
        "companyfacts_endpoint": COMPANYFACTS_URL,
        "concept": f"dei:{SHARES_CONCEPT}",
        "mapped_cik_count": len(tickers) - len(missing_cik),
        "missing_cik_count": len(missing_cik),
        "missing_cik_sample": missing_cik[:50],
        "tickers_with_points": counters["tickers_with_points"],
        "tickers_without_points": counters["tickers_without_points"],
        "errors_count": len(errors),
        "errors_sample": errors[:50],
        "processing_counts": dict(counters),
    }
    return {"meta": meta, "tickers": tickers_out}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch SEC shares outstanding history for screener tickers.")
    parser.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE, help="Ticker universe JSON.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output JSON.")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR, help="SEC response cache directory.")
    parser.add_argument("--start-year", type=int, default=2011, help="Earliest fact end-year to include.")
    parser.add_argument("--request-delay", type=float, default=0.12, help="Delay after uncached SEC companyfacts requests.")
    parser.add_argument("--timeout", type=int, default=30, help="HTTP timeout in seconds.")
    parser.add_argument("--user-agent", default=None, help="SEC User-Agent. Defaults to SEC_USER_AGENT or project default.")
    parser.add_argument("--refresh-tickers", action="store_true", help="Refresh cached SEC ticker map.")
    parser.add_argument("--refresh-facts", action="store_true", help="Refresh cached companyfacts files.")
    parser.add_argument("--progress-every", type=int, default=100, help="Print progress every N tickers; 0 disables.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_dataset(args)
    write_json(args.output, payload)
    print(json.dumps(payload["meta"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
