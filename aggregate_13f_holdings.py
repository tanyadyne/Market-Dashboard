#!/usr/bin/env python3
"""
Aggregate raw SEC 13F rows into manager/ticker positions.

This is step 3 of the ownership tile pipeline:
  - read raw matched 13F holdings from fetch_13f_holdings.py
  - ignore option and non-equity rows by default
  - sum duplicate rows inside the same filing
  - keep the latest filing for each manager/ticker/report period
  - emit one row per manager + ticker + report period
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_INPUT = Path("13f_holdings_2026_q2_raw.ndjson")
DEFAULT_OUTPUT = Path("13f_holdings_2026_q2_aggregated.ndjson")
DEFAULT_PRICE_MAP = Path("leaders_mcap.json")

EQUITY_TITLE_MARKERS = (
    "SHS",
    "SHARE",
    "ORDINARY",
    "COMMON",
    "CLASS A",
    "CLASS B",
    "CLASS C",
)


@dataclass
class FilingPosition:
    ticker: str
    manager_cik: str
    manager_name: str
    report_period: str
    accession: str
    filing_date: str
    form_type: str
    shares: int = 0
    value_usd: int = 0
    row_count: int = 0
    issuers: set[str] = field(default_factory=set)
    titles: set[str] = field(default_factory=set)
    cusips: set[str] = field(default_factory=set)
    value_multipliers: set[int] = field(default_factory=set)
    value_normalization_sources: set[str] = field(default_factory=set)


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_ndjson(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            yield line_no, json.loads(line)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def summary_path_for(output: Path) -> Path:
    return output.with_suffix(".summary.json")


def parse_date_key(value: Any) -> tuple[int, int, int]:
    text = str(value or "")
    try:
        dt = datetime.strptime(text, "%Y-%m-%d")
        return dt.year, dt.month, dt.day
    except ValueError:
        return 0, 0, 0


def filing_rank_key(position: FilingPosition) -> tuple[tuple[int, int, int], int, str]:
    amendment_priority = 1 if str(position.form_type).endswith("/A") else 0
    return parse_date_key(position.filing_date), amendment_priority, position.accession


def as_positive_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        number = int(float(str(value).replace(",", "")))
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def as_positive_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def clean_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def is_equity_like_position(row: dict[str, Any]) -> bool:
    """Keep malformed SEC principal rows when their title identifies equity."""
    title = clean_text(row.get("title"))
    if not title:
        return False
    normalized = title.upper()
    return any(marker in normalized for marker in EQUITY_TITLE_MARKERS)


def add_text(target: set[str], value: Any) -> None:
    text = clean_text(value)
    if text:
        target.add(text)


def load_price_map(path: Path | None) -> dict[str, float]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}

    prices: dict[str, float] = {}
    caps = payload.get("caps")
    shares = payload.get("shares")
    if isinstance(caps, dict) and isinstance(shares, dict):
        for ticker, cap_value in caps.items():
            cap = as_positive_float(cap_value)
            share_count = as_positive_float(shares.get(ticker))
            if cap is not None and share_count is not None:
                prices[str(ticker).upper()] = cap / share_count
        return prices

    for ticker, value in payload.items():
        price = as_positive_float(value)
        if price is not None:
            prices[str(ticker).upper()] = price
    return prices


def infer_value_multiplier(
    ticker: str,
    value_reported: int,
    shares: int,
    prices: dict[str, float],
) -> tuple[int, str]:
    reported_per_share = value_reported / shares
    price = prices.get(ticker.upper())
    if price is not None:
        candidates = (
            (1, reported_per_share),
            (1000, reported_per_share * 1000),
        )
        multiplier, _ = min(candidates, key=lambda item: abs(math.log(max(item[1], 1e-9) / price)))
        return multiplier, "price_map"
    return (1000, "ratio_threshold") if reported_per_share < 2.0 else (1, "ratio_threshold")


def aggregate_raw_rows(
    input_path: Path,
    prices: dict[str, float],
    include_options: bool,
    include_non_share_types: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_accession: dict[tuple[str, str, str, str], FilingPosition] = {}
    malformed_rows: list[dict[str, Any]] = []
    skipped = defaultdict(int)
    raw_rows = 0

    for line_no, row in read_ndjson(input_path):
        raw_rows += 1
        ticker = clean_text(row.get("ticker"))
        manager_cik = clean_text(row.get("manager_cik"))
        report_period = clean_text(row.get("report_period"))
        accession = clean_text(row.get("accession"))
        if not ticker or not manager_cik or not report_period or not accession:
            skipped["missing_required_key"] += 1
            if len(malformed_rows) < 25:
                malformed_rows.append({"line": line_no, "reason": "missing_required_key"})
            continue

        if row.get("put_call") and not include_options:
            skipped["option_rows"] += 1
            continue

        share_type = clean_text(row.get("share_type"))
        if share_type and share_type.upper() != "SH" and not include_non_share_types:
            if is_equity_like_position(row):
                skipped["equity_like_principal_rows"] += 1
            else:
                skipped["non_share_type_rows"] += 1
                continue

        shares = as_positive_int(row.get("shares"))
        if shares is None:
            skipped["invalid_shares"] += 1
            continue

        value_reported = as_positive_int(row.get("value_reported"))
        if value_reported is None:
            value_reported = as_positive_int(row.get("value_usd"))
        if value_reported is None:
            skipped["invalid_value"] += 1
            continue

        value_multiplier, normalization_source = infer_value_multiplier(ticker, value_reported, shares, prices)
        value_usd = value_reported * value_multiplier
        key = (manager_cik, ticker, report_period, accession)
        position = by_accession.get(key)
        if position is None:
            position = FilingPosition(
                ticker=ticker,
                manager_cik=manager_cik,
                manager_name=clean_text(row.get("manager_name")) or manager_cik,
                report_period=report_period,
                accession=accession,
                filing_date=clean_text(row.get("filing_date")) or "",
                form_type=clean_text(row.get("form_type")) or "",
            )
            by_accession[key] = position

        position.shares += shares
        position.value_usd += value_usd
        position.row_count += 1
        add_text(position.issuers, row.get("issuer"))
        add_text(position.titles, row.get("title"))
        add_text(position.cusips, row.get("cusip"))
        position.value_multipliers.add(value_multiplier)
        position.value_normalization_sources.add(normalization_source)

    latest_by_group: dict[tuple[str, str, str], FilingPosition] = {}
    superseded_by_group: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for position in by_accession.values():
        group_key = (position.manager_cik, position.ticker, position.report_period)
        current = latest_by_group.get(group_key)
        if current is None or filing_rank_key(position) > filing_rank_key(current):
            if current is not None:
                superseded_by_group[group_key].append(current.accession)
            latest_by_group[group_key] = position
        else:
            superseded_by_group[group_key].append(position.accession)

    output_rows: list[dict[str, Any]] = []
    for group_key, position in latest_by_group.items():
        superseded = sorted(set(superseded_by_group.get(group_key, [])))
        output_rows.append(
            {
                "ticker": position.ticker,
                "manager_cik": position.manager_cik,
                "manager_name": position.manager_name,
                "report_period": position.report_period,
                "filing_date": position.filing_date,
                "form_type": position.form_type,
                "accession": position.accession,
                "shares": position.shares,
                "value_usd": position.value_usd,
                "source_row_count": position.row_count,
                "issuer_names": sorted(position.issuers),
                "titles": sorted(position.titles),
                "cusips": sorted(position.cusips),
                "value_multipliers": sorted(position.value_multipliers),
                "value_normalization_sources": sorted(position.value_normalization_sources),
                "superseded_accessions": superseded,
                "superseded_accession_count": len(superseded),
            }
        )

    output_rows.sort(
        key=lambda item: (
            item["ticker"],
            item["report_period"],
            -int(item.get("value_usd") or 0),
            item["manager_name"],
        )
    )

    summary = {
        "input": str(input_path),
        "raw_rows": raw_rows,
        "accession_level_positions": len(by_accession),
        "aggregated_positions": len(output_rows),
        "skipped": dict(sorted(skipped.items())),
        "malformed_sample": malformed_rows,
        "unique_tickers": len({row["ticker"] for row in output_rows}),
        "unique_managers": len({row["manager_cik"] for row in output_rows}),
        "report_periods": sorted({row["report_period"] for row in output_rows}),
        "positions_with_superseded_filings": sum(1 for row in output_rows if row["superseded_accession_count"]),
        "value_multiplier_counts": dict(
            sorted(Counter(multiplier for row in output_rows for multiplier in row["value_multipliers"]).items())
        ),
        "value_normalization_sources": dict(
            sorted(Counter(source for row in output_rows for source in row["value_normalization_sources"]).items())
        ),
        "total_shares": sum(int(row["shares"]) for row in output_rows),
        "total_value_usd": sum(int(row["value_usd"]) for row in output_rows),
        "include_options": include_options,
        "include_non_share_types": include_non_share_types,
    }
    return output_rows, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--price-map", type=Path, default=DEFAULT_PRICE_MAP)
    parser.add_argument("--no-price-map", action="store_true")
    parser.add_argument("--include-options", action="store_true")
    parser.add_argument("--include-non-share-types", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Missing raw 13F input: {args.input}")

    prices = {} if args.no_price_map else load_price_map(args.price_map)
    rows, summary = aggregate_raw_rows(
        args.input,
        prices=prices,
        include_options=args.include_options,
        include_non_share_types=args.include_non_share_types,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")

    summary.update(
        {
            "updated": now_iso(),
            "output": str(args.output),
            "price_map": None if args.no_price_map else str(args.price_map),
            "price_map_tickers": len(prices),
            "summary": str(summary_path_for(args.output)),
            "note": "Step 4 should apply manager inclusion/exclusion rules to this aggregated file.",
        }
    )
    write_json(summary_path_for(args.output), summary)

    print(
        "raw_rows={raw_rows} aggregated_positions={aggregated_positions} "
        "tickers={unique_tickers} managers={unique_managers}".format(**summary)
    )
    print(f"summary={summary_path_for(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
