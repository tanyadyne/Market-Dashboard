#!/usr/bin/env python3
"""
Build a local ticker -> CUSIP map for the stock screener universe.

Primary source: EODHD ID Mapping API.

The script caches raw EODHD pages under .cusip_cache/ so interrupted runs can
resume without re-consuming API calls.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen


API_BASE = "https://eodhd.com/api/id-mapping"
DEFAULT_CACHE_DIR = Path(".cusip_cache")
DEFAULT_UNIVERSE = Path("screener_tickers.json")
DEFAULT_OUTPUT = Path("ticker_cusip_map.json")
DEFAULT_UNMAPPED = Path("ticker_cusip_unmapped.json")
DEFAULT_DUPLICATES = Path("ticker_cusip_duplicates.json")


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


def load_universe(path: Path) -> list[str]:
    payload = read_json(path)
    if isinstance(payload, dict) and isinstance(payload.get("tickers"), list):
        tickers = payload["tickers"]
    elif isinstance(payload, list):
        tickers = payload
    else:
        raise ValueError(f"Unsupported universe file shape: {path}")
    return sorted({str(t).strip().upper() for t in tickers if str(t).strip()})


def request_json(params: dict[str, Any], token: str, retries: int = 3) -> dict[str, Any]:
    query = dict(params)
    query["api_token"] = token
    query["fmt"] = "json"
    url = f"{API_BASE}?{urlencode(query)}"
    headers = {
        "User-Agent": "WIN_MarketDashboard/1.0 (ticker-cusip-map)",
        "Accept": "application/json",
    }
    for attempt in range(retries):
        try:
            with urlopen(Request(url, headers=headers), timeout=45) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError("unreachable")


def cache_name(prefix: str, params: dict[str, Any]) -> str:
    safe = "_".join(f"{k}-{str(v).replace('/', '-')}" for k, v in sorted(params.items()))
    return f"{prefix}_{safe}.json"


def cached_request(
    cache_dir: Path,
    cache_prefix: str,
    params: dict[str, Any],
    token: str,
    refresh: bool,
) -> dict[str, Any]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / cache_name(cache_prefix, params)
    if path.exists() and not refresh:
        return read_json(path)
    payload = request_json(params, token)
    write_json(path, payload)
    return payload


def base_symbol(eodhd_symbol: str) -> str:
    symbol = str(eodhd_symbol or "").strip().upper()
    if symbol.endswith(".US"):
        symbol = symbol[:-3]
    return symbol


def isin_identifier(row: dict[str, Any]) -> str | None:
    isin = str(row.get("isin") or "").upper()
    if len(isin) >= 11:
        return isin[2:11]
    return None


def isin_country(row: dict[str, Any]) -> str:
    return str(row.get("isin") or "").upper()[:2]


def effective_cusip(row: dict[str, Any]) -> str | None:
    cusip = str(row.get("cusip") or "").strip().upper()
    isin_cusip = isin_identifier(row)
    if isin_country(row) == "US" and cusip and isin_cusip and cusip != isin_cusip:
        return isin_cusip
    if cusip:
        return cusip
    return isin_cusip


def effective_cusip_source(row: dict[str, Any]) -> str | None:
    cusip = str(row.get("cusip") or "").strip().upper()
    isin_cusip = isin_identifier(row)
    if isin_country(row) == "US" and cusip and isin_cusip and cusip != isin_cusip:
        return "isin_override_us_mismatch"
    if cusip:
        return "eodhd"
    if isin_cusip:
        return "isin_derived"
    return None


def cusip_matches_isin(row: dict[str, Any]) -> bool:
    isin = str(row.get("isin") or "").upper()
    cusip = str(row.get("cusip") or "").upper()
    return len(isin) >= 11 and bool(cusip) and isin[2:11] == cusip


def score_row(row: dict[str, Any]) -> tuple[int, int, int]:
    return (
        1 if cusip_matches_isin(row) else 0,
        1 if row.get("cik") else 0,
        1 if row.get("figi") else 0,
    )


def choose_best_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    valid = [row for row in rows if effective_cusip(row)]
    if not valid:
        return None
    return sorted(valid, key=score_row, reverse=True)[0]


def compact_row(row: dict[str, Any]) -> dict[str, Any]:
    raw_cusip = str(row.get("cusip") or "").strip().upper()
    cusip = effective_cusip(row)
    cusip_source = effective_cusip_source(row)
    return {
        "cusip": cusip,
        "cusip_source": cusip_source,
        "isin": row.get("isin"),
        "figi": row.get("figi"),
        "lei": row.get("lei"),
        "cik": row.get("cik"),
        "eodhd_symbol": row.get("symbol"),
        **({"raw_cusip": raw_cusip} if raw_cusip and raw_cusip != cusip else {}),
        "source": "EODHD ID Mapping API",
    }


def fetch_bulk_us(
    token: str,
    universe: set[str],
    cache_dir: Path,
    refresh: bool,
    page_limit: int,
    max_pages: int | None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    by_ticker: dict[str, list[dict[str, Any]]] = {}
    pages_read = 0
    records_seen = 0
    total = None
    offset = 0
    error = None

    while True:
        params = {
            "filter[ex]": "US",
            "page[limit]": page_limit,
            "page[offset]": offset,
        }
        try:
            payload = cached_request(cache_dir, "eodhd_us", params, token, refresh)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            break
        data = payload.get("data") or []
        meta = payload.get("meta") or {}
        pages_read += 1
        records_seen += len(data)
        total = meta.get("total", total)

        for row in data:
            ticker = base_symbol(row.get("symbol"))
            if ticker in universe:
                by_ticker.setdefault(ticker, []).append(row)

        if universe.issubset(by_ticker.keys()):
            break
        if not data or not (payload.get("links") or {}).get("next"):
            break
        if max_pages is not None and pages_read >= max_pages:
            break

        offset += page_limit

    stats = {
        "pages_read": pages_read,
        "records_seen": records_seen,
        "reported_total": total,
        "matched_from_bulk": len(by_ticker),
    }
    if error:
        stats["error"] = error
    return by_ticker, stats


def fetch_direct_symbol(
    token: str,
    ticker: str,
    cache_dir: Path,
    refresh: bool,
) -> list[dict[str, Any]]:
    params = {"filter[symbol]": f"{ticker}.US"}
    payload = cached_request(cache_dir, "eodhd_symbol", params, token, refresh)
    return payload.get("data") or []


def build_map(args: argparse.Namespace) -> dict[str, Any]:
    load_env_file()
    token = os.environ.get("EODHD_API_TOKEN")
    if not token:
        raise RuntimeError("EODHD_API_TOKEN is not set. Add it to .env.local or the environment.")

    universe = load_universe(args.universe)
    universe_set = set(universe)
    bulk_rows, bulk_stats = fetch_bulk_us(
        token=token,
        universe=universe_set,
        cache_dir=args.cache_dir,
        refresh=args.refresh,
        page_limit=args.page_limit,
        max_pages=args.max_pages,
    )

    direct_calls = 0
    if args.direct_retries:
        missing = [ticker for ticker in universe if ticker not in bulk_rows]
        for ticker in missing[: args.direct_limit]:
            rows = fetch_direct_symbol(token, ticker, args.cache_dir, args.refresh)
            direct_calls += 1
            if rows:
                bulk_rows.setdefault(ticker, []).extend(rows)

    mapped: dict[str, dict[str, Any]] = {}
    duplicates: dict[str, list[dict[str, Any]]] = {}
    for ticker in universe:
        rows = bulk_rows.get(ticker, [])
        best = choose_best_row(rows)
        if best:
            mapped[ticker] = compact_row(best)
        unique_cusips = sorted({str(effective_cusip(row)) for row in rows if effective_cusip(row)})
        if len(unique_cusips) > 1:
            duplicates[ticker] = [compact_row(row) for row in rows if effective_cusip(row)]

    unmapped = [ticker for ticker in universe if ticker not in mapped]
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    result = {
        "updated": now,
        "source": "EODHD ID Mapping API",
        "universe_file": str(args.universe),
        "universe_count": len(universe),
        "mapped_count": len(mapped),
        "unmapped_count": len(unmapped),
        "duplicate_cusip_tickers_count": len(duplicates),
        "bulk_stats": bulk_stats,
        "direct_symbol_calls": direct_calls,
        "tickers": mapped,
    }
    write_json(args.output, result)
    write_json(args.unmapped_output, {"updated": now, "count": len(unmapped), "tickers": unmapped})
    write_json(args.duplicates_output, {"updated": now, "count": len(duplicates), "tickers": duplicates})
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--unmapped-output", type=Path, default=DEFAULT_UNMAPPED)
    parser.add_argument("--duplicates-output", type=Path, default=DEFAULT_DUPLICATES)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--page-limit", type=int, default=1000)
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--direct-retries", action="store_true")
    parser.add_argument("--direct-limit", type=int, default=100)
    parser.add_argument("--refresh", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build_map(args)
    print(
        "mapped={mapped_count}/{universe_count} unmapped={unmapped_count} "
        "bulk_pages={pages_read} records={records_seen} duplicates={duplicate_cusip_tickers_count}".format(
            pages_read=result["bulk_stats"]["pages_read"],
            records_seen=result["bulk_stats"]["records_seen"],
            **result,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
