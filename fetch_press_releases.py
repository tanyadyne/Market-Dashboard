#!/usr/bin/env python3
"""Fetch Yahoo Finance press releases for stock screener profiles."""

from __future__ import annotations

import argparse
import html
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import yfinance as yf


ROOT = Path(__file__).resolve().parent
LEADERS_JSON = ROOT / "leaders.json"
SCREENER_TICKERS = ROOT / "screener_tickers.json"
OUT_DIR = ROOT / "press_releases"
INDEX_JSON = OUT_DIR / "index.json"
PRESS_RELEASE_TAB = "press releases"
DEFAULT_COUNT = 30
SUMMARY_LIMIT = 420
HARD_EXCLUDED_TICKERS = {
    "AHR",
    "BAND",
    "CWAN",
    "GOOG",
    "LILA",
    "LILAK",
    "MRX",
    "OZK",
    "SVC",
    "TECH",
    "UPBD",
    "XMTR",
}

# yfinance otherwise converts malformed/throttled Yahoo responses into an empty
# list. Let those errors surface so a failed refresh cannot masquerade as valid
# "no press releases" data and overwrite the cache.
yf.config.debug.hide_exceptions = False


def load_universe() -> list[str]:
    tickers: set[str] = set()
    if LEADERS_JSON.exists():
        data = json.loads(LEADERS_JSON.read_text(encoding="utf-8"))
        tickers.update(
            str(row.get("t") or "").strip().upper()
            for row in data.get("e", [])
            if str(row.get("t") or "").strip()
        )
    if not tickers and SCREENER_TICKERS.exists():
        data = json.loads(SCREENER_TICKERS.read_text(encoding="utf-8"))
        values = data.get("tickers", []) if isinstance(data, dict) else data
        tickers.update(str(value).strip().upper() for value in values if str(value).strip())
    return sorted(ticker for ticker in tickers if ticker not in HARD_EXCLUDED_TICKERS)


def parse_ticker_filter(value: str | None) -> set[str] | None:
    if not value:
        return None
    parsed = {part.strip().upper() for part in value.split(",") if part.strip()}
    return parsed or None


def shard_key(ticker: str) -> str:
    first = str(ticker or "_").strip().upper()[:1]
    return first if first and first.isalnum() else "_"


def clean_text(value: object, limit: int | None = None) -> str:
    text = html.unescape(re.sub(r"<[^>]+>", " ", str(value or "")))
    text = re.sub(r"\s+", " ", text).strip()
    if limit and len(text) > limit:
        text = text[: limit - 3].rsplit(" ", 1)[0].rstrip() + "..."
    return text


def nested_url(value: object) -> str:
    if isinstance(value, dict):
        return clean_text(value.get("url"))
    return clean_text(value)


def published_at(content: dict) -> str:
    value = content.get("pubDate") or content.get("displayTime") or content.get("providerPublishTime")
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, timezone.utc).isoformat().replace("+00:00", "Z")
    return clean_text(value)


def normalize_release(item: object) -> dict | None:
    if not isinstance(item, dict):
        return None
    content = item.get("content") if isinstance(item.get("content"), dict) else item
    title = clean_text(content.get("title"))
    if not title:
        return None
    provider_value = content.get("provider")
    provider = (
        clean_text(provider_value.get("displayName"))
        if isinstance(provider_value, dict)
        else clean_text(provider_value)
    )
    url = (
        nested_url(content.get("clickThroughUrl"))
        or nested_url(content.get("canonicalUrl"))
        or nested_url(content.get("link"))
    )
    return {
        "id": clean_text(content.get("id") or item.get("id")),
        "title": title,
        "summary": clean_text(content.get("summary") or content.get("description"), SUMMARY_LIMIT),
        "published_at": published_at(content),
        "provider": provider,
        "url": url,
    }


def normalize_releases(items: object, count: int = DEFAULT_COUNT) -> list[dict]:
    if not isinstance(items, list):
        return []
    releases: list[dict] = []
    seen: set[str] = set()
    for item in items:
        release = normalize_release(item)
        if not release:
            continue
        identity = release["id"] or release["url"] or f'{release["title"]}|{release["published_at"]}'
        if identity in seen:
            continue
        seen.add(identity)
        releases.append(release)
    releases.sort(key=lambda row: row.get("published_at") or "", reverse=True)
    return releases[:count]


def fetch_ticker(ticker: str, count: int, retries: int, retry_delay: float) -> tuple[str, list[dict] | None, str | None]:
    last_error = "Unknown Yahoo Finance error"
    for attempt in range(retries):
        try:
            raw = yf.Ticker(ticker).get_news(count=count, tab=PRESS_RELEASE_TAB)
            return ticker, normalize_releases(raw, count), None
        except Exception as exc:  # Yahoo errors vary by transport and response type.
            last_error = str(exc)
            if attempt + 1 < retries:
                time.sleep(retry_delay * (2**attempt))
    return ticker, None, last_error


def load_existing() -> dict[str, dict]:
    records: dict[str, dict] = {}
    if not OUT_DIR.exists():
        return records
    for path in OUT_DIR.glob("*.json"):
        if path.name == INDEX_JSON.name:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            for ticker, record in payload.get("tickers", {}).items():
                if isinstance(record, dict) and isinstance(record.get("items"), list):
                    records[str(ticker).upper()] = record
        except Exception as exc:
            print(f"Warning: failed to read {path.name}: {exc}")
    return records


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    temp.replace(path)


def write_dataset(records: dict[str, dict], meta: dict) -> None:
    grouped: dict[str, dict[str, dict]] = {}
    for ticker in sorted(records):
        grouped.setdefault(shard_key(ticker), {})[ticker] = records[ticker]
    expected = set()
    for shard, tickers in grouped.items():
        path = OUT_DIR / f"{shard}.json"
        expected.add(path.name)
        write_json(path, {"tickers": tickers})
    if OUT_DIR.exists():
        for path in OUT_DIR.glob("*.json"):
            if path.name != INDEX_JSON.name and path.name not in expected:
                path.unlink()
    meta["shards"] = sorted(expected)
    write_json(INDEX_JSON, meta)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch Yahoo Finance press releases for screener profiles.")
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--tickers", help="Optional comma-separated ticker filter.")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=1.0)
    parser.add_argument("--request-delay", type=float, default=0.05)
    parser.add_argument("--min-success-ratio", type=float, default=0.5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    count = max(1, min(100, args.count))
    all_tickers = load_universe()
    ticker_filter = parse_ticker_filter(args.tickers)
    tickers = [ticker for ticker in all_tickers if not ticker_filter or ticker in ticker_filter]
    active = set(all_tickers)
    records = {ticker: record for ticker, record in load_existing().items() if ticker in active}
    succeeded = 0
    failed: list[str] = []
    preserved = 0
    changed = 0

    print(f"Fetching up to {count} press releases for {len(tickers)} stock(s)...")
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {}
        for ticker in tickers:
            futures[pool.submit(fetch_ticker, ticker, count, max(1, args.retries), max(0.0, args.retry_delay))] = ticker
            if args.request_delay > 0:
                time.sleep(args.request_delay)
        for index, future in enumerate(as_completed(futures), start=1):
            ticker, items, error = future.result()
            if error is not None:
                failed.append(ticker)
                if ticker in records:
                    preserved += 1
                print(f"  {ticker}: failed ({error})")
            else:
                succeeded += 1
                old_items = records.get(ticker, {}).get("items", [])
                # Empty Yahoo responses can be transient; never erase a populated cache.
                if items or not old_items:
                    new_record = {"items": items}
                    if records.get(ticker) != new_record:
                        records[ticker] = new_record
                        changed += 1
            if index % 100 == 0:
                print(f"  {index}/{len(tickers)}...")

    if tickers and succeeded == 0:
        print("ERROR: no press-release requests succeeded; existing dataset left untouched")
        return 1

    success_ratio = succeeded / len(tickers) if tickers else 1.0
    meta = {
        "updated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "source": "yfinance.Ticker.get_news",
        "tab": PRESS_RELEASE_TAB,
        "requested_per_ticker": count,
        "universe": len(all_tickers),
        "records": len(records),
        "attempted": len(tickers),
        "succeeded": succeeded,
        "failed": len(failed),
        "preserved_on_failure": preserved,
        "changed": changed,
        "success_ratio": round(success_ratio, 4),
        "failed_tickers": sorted(failed),
    }
    write_dataset(records, meta)
    print(
        f"Wrote {OUT_DIR.name}: {succeeded}/{len(tickers)} succeeded; "
        f"{len(failed)} failed; {preserved} cached record(s) preserved"
    )
    return 0 if success_ratio >= max(0.0, min(1.0, args.min_success_ratio)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
