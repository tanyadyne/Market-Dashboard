#!/usr/bin/env python3
"""
Rank top included 13F holders for the ownership tab.

This is step 5 of the ownership tile pipeline:
  - read holder_status=include positions from classify_13f_holders.py
  - select a 13F report period
  - rank each ticker's holders by shares held
  - calculate percent of shares outstanding
  - emit a compact per-ticker JSON dataset for the ownership table
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_INPUT = Path("13f_holdings_2026_q2_included.ndjson")
DEFAULT_SHARES = Path("leaders_mcap.json")
DEFAULT_UNIVERSE = Path("screener_tickers.json")
DEFAULT_OUTPUT = Path("ownership_top_holders_13f.json")

GENERIC_HOLDER_TOKENS = {
    "ADVISER",
    "ADVISERS",
    "ADVISOR",
    "ADVISORS",
    "ADVISORY",
    "AND",
    "ASSET",
    "CAPITAL",
    "CO",
    "COMPANY",
    "CORP",
    "CORPORATION",
    "FINANCIAL",
    "FAMILY",
    "FIRST",
    "FUND",
    "FUNDS",
    "GLOBAL",
    "GROUP",
    "INC",
    "INVESTMENT",
    "INVESTMENTS",
    "LEGACY",
    "LLC",
    "LLP",
    "LP",
    "LTD",
    "MANAGEMENT",
    "MANAGER",
    "MANAGERS",
    "OFFICE",
    "PARTNER",
    "PARTNERS",
    "PLANNING",
    "PRESERVATION",
    "PRIVATE",
    "SERVICE",
    "SERVICES",
    "STRATEGIES",
    "STRATEGY",
    "THE",
    "WEALTH",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_ndjson(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def summary_path_for(output: Path) -> Path:
    return output.with_suffix(".summary.json")


def parse_int(value: Any) -> int | None:
    try:
        if value in ("", None):
            return None
        return int(float(value))
    except (TypeError, ValueError, OverflowError):
        return None


def parse_period(value: Any) -> tuple[int, int, int]:
    text = str(value or "")
    try:
        dt = datetime.strptime(text, "%Y-%m-%d")
        return dt.year, dt.month, dt.day
    except ValueError:
        return 0, 0, 0


def pretty_period(value: str) -> str:
    try:
        dt = datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        return value
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    return f"{dt.day} {months[dt.month - 1]} {dt.year}"


def compact_number(value: int | float | None, unit: str = "") -> str | None:
    if value is None:
        return None
    amount = float(value)
    sign = "-" if amount < 0 else ""
    amount = abs(amount)
    for threshold, suffix in ((1_000_000_000_000, "T"), (1_000_000_000, "B"), (1_000_000, "M"), (1_000, "K")):
        if amount >= threshold:
            text = f"{amount / threshold:.2f}".rstrip("0").rstrip(".")
            return f"{sign}{unit}{text}{suffix}"
    if amount == int(amount):
        return f"{sign}{unit}{int(amount):,}"
    return f"{sign}{unit}{amount:,.2f}"


def holder_tokens(name: Any) -> set[str]:
    text = str(name or "").upper().replace("&", " AND ")
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    tokens = set()
    for token in text.split():
        if len(token) <= 2:
            continue
        if token in GENERIC_HOLDER_TOKENS:
            continue
        tokens.add(token)
    return tokens


def holders_look_related(left: Any, right: Any) -> bool:
    left_tokens = holder_tokens(left)
    right_tokens = holder_tokens(right)
    if not left_tokens or not right_tokens:
        return False
    return bool(left_tokens & right_tokens)


def load_universe(path: Path) -> list[str]:
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


def load_shares_outstanding(path: Path) -> dict[str, int]:
    data = read_json(path)
    raw = data.get("shares") if isinstance(data, dict) else None
    if not isinstance(raw, dict):
        raise ValueError(f"Could not find shares map in {path}")
    shares: dict[str, int] = {}
    for ticker, value in raw.items():
        parsed = parse_int(value)
        if parsed and parsed > 0:
            shares[str(ticker).upper()] = parsed
    return shares


def select_report_period(rows: list[dict[str, Any]], requested: str) -> str:
    periods = sorted({str(row.get("report_period") or "") for row in rows if row.get("report_period")}, key=parse_period)
    if not periods:
        raise ValueError("No report_period values found in input")
    if requested.lower() in {"auto", "latest", "max"}:
        return periods[-1]
    if requested not in periods:
        raise ValueError(f"Requested report period {requested!r} not found. Available: {', '.join(periods)}")
    return requested


def position_key(row: dict[str, Any]) -> tuple[str, str, str]:
    ticker = str(row.get("ticker") or "").upper()
    manager_id = str(row.get("manager_cik") or row.get("manager_name") or "")
    report_period = str(row.get("report_period") or "")
    return ticker, manager_id, report_period


def filing_key(row: dict[str, Any]) -> tuple[tuple[int, int, int], str]:
    return parse_period(row.get("filing_date")), str(row.get("accession") or "")


def rank_positions(
    rows: list[dict[str, Any]],
    *,
    report_period: str,
    shares_outstanding: dict[str, int],
    top_n: int,
    rank_by: str,
) -> tuple[dict[str, list[dict[str, Any]]], Counter[str], list[dict[str, Any]]]:
    counters: Counter[str] = Counter()
    duplicate_sample: list[dict[str, Any]] = []
    latest_by_position: dict[tuple[str, str, str], dict[str, Any]] = {}

    for row in rows:
        if str(row.get("holder_status") or "include") != "include":
            counters["skipped_not_included"] += 1
            continue
        if str(row.get("report_period") or "") != report_period:
            counters["skipped_other_report_period"] += 1
            continue
        ticker = str(row.get("ticker") or "").upper()
        if not ticker:
            counters["skipped_missing_ticker"] += 1
            continue
        shares = parse_int(row.get("shares"))
        if not shares or shares <= 0:
            counters["skipped_invalid_shares"] += 1
            continue
        value_usd = parse_int(row.get("value_usd")) or 0

        normalized = dict(row)
        normalized["ticker"] = ticker
        normalized["shares"] = shares
        normalized["value_usd"] = value_usd

        key = position_key(normalized)
        existing = latest_by_position.get(key)
        if existing is None or filing_key(normalized) >= filing_key(existing):
            latest_by_position[key] = normalized
        else:
            counters["skipped_superseded_duplicate"] += 1

    by_ticker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in latest_by_position.values():
        ticker = row["ticker"]
        shares = int(row["shares"])
        value_usd = int(row["value_usd"])
        shares_out = shares_outstanding.get(ticker)
        pct_out = (shares / shares_out * 100) if shares_out else None

        by_ticker[ticker].append(
            {
                "holder": str(row.get("manager_name") or "").strip(),
                "manager_cik": str(row.get("manager_cik") or "").strip() or None,
                "shares": shares,
                "shares_display": compact_number(shares),
                "pct_out": round(pct_out, 4) if pct_out is not None else None,
                "pct_out_display": f"{pct_out:.2f}%" if pct_out is not None else None,
                "value_usd": value_usd,
                "value_display": compact_number(value_usd, "$"),
                "filing_date": row.get("filing_date"),
                "accession": row.get("accession"),
                "report_period": row.get("report_period"),
                "classification_rule": row.get("classification_rule"),
                "classification_match": row.get("classification_match"),
            }
        )

    rank_field = "shares" if rank_by == "shares" else "value_usd"
    ranked: dict[str, list[dict[str, Any]]] = {}
    for ticker, positions in by_ticker.items():
        positions.sort(key=lambda item: (item.get(rank_field) or 0, item.get("value_usd") or 0, item.get("shares") or 0), reverse=True)
        deduped: list[dict[str, Any]] = []
        for position in positions:
            duplicate_of = None
            for kept in deduped:
                same_position_size = position.get("shares") == kept.get("shares") and position.get("value_usd") == kept.get("value_usd")
                if same_position_size and holders_look_related(position.get("holder"), kept.get("holder")):
                    duplicate_of = kept
                    break
            if duplicate_of:
                counters["equivalent_holder_duplicates_dropped"] += 1
                if len(duplicate_sample) < 50:
                    duplicate_sample.append(
                        {
                            "ticker": ticker,
                            "kept_holder": duplicate_of.get("holder"),
                            "dropped_holder": position.get("holder"),
                            "shares": position.get("shares"),
                            "value_usd": position.get("value_usd"),
                        }
                    )
                continue
            deduped.append(position)
        ranked[ticker] = deduped[:top_n]

    counters["positions_considered"] = len(latest_by_position)
    counters["tickers_with_positions"] = len(ranked)
    return ranked, counters, duplicate_sample


def build_output(
    *,
    rows: list[dict[str, Any]],
    universe: list[str],
    shares_outstanding: dict[str, int],
    report_period: str,
    top_n: int,
    rank_by: str,
    input_path: Path,
    shares_path: Path,
    universe_path: Path,
) -> dict[str, Any]:
    ranked, counters, duplicate_sample = rank_positions(
        rows,
        report_period=report_period,
        shares_outstanding=shares_outstanding,
        top_n=top_n,
        rank_by=rank_by,
    )

    tickers: dict[str, dict[str, Any]] = {}
    missing_shares_outstanding: list[str] = []
    for ticker in universe:
        shares_out = shares_outstanding.get(ticker)
        if shares_out is None:
            missing_shares_outstanding.append(ticker)
        holders = ranked.get(ticker, [])
        tickers[ticker] = {
            "ticker": ticker,
            "report_period": report_period,
            "shares_outstanding": shares_out,
            "shares_outstanding_display": compact_number(shares_out),
            "holders": holders,
        }

    meta = {
        "updated": now_iso(),
        "source": "SEC 13F-HR filings filtered by holder classification rules",
        "input": str(input_path),
        "shares_outstanding_file": str(shares_path),
        "universe_file": str(universe_path),
        "report_period": report_period,
        "report_period_label": f"Latest 13F filings, quarter ended {pretty_period(report_period)}",
        "rank_by": rank_by,
        "top_n": top_n,
        "universe_count": len(universe),
        "tickers_with_holders": sum(1 for ticker in universe if ranked.get(ticker)),
        "positions_used": counters["positions_considered"],
        "missing_shares_outstanding_count": len(missing_shares_outstanding),
        "missing_shares_outstanding_sample": missing_shares_outstanding[:50],
        "equivalent_holder_duplicate_sample": duplicate_sample,
        "processing_counts": dict(counters),
    }

    return {"meta": meta, "tickers": tickers}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank top included 13F holders by ticker.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Included-only 13F NDJSON from classify_13f_holders.py.")
    parser.add_argument("--shares", type=Path, default=DEFAULT_SHARES, help="JSON file containing shares outstanding by ticker.")
    parser.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE, help="Ticker universe JSON.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output JSON for ownership top holders.")
    parser.add_argument("--report-period", default="latest", help="13F report period to use, or latest/auto.")
    parser.add_argument("--top-n", type=int, default=5, help="Number of holders per ticker.")
    parser.add_argument("--rank-by", choices=("shares", "value"), default="shares", help="Ranking metric.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = list(read_ndjson(args.input))
    report_period = select_report_period(rows, args.report_period)
    universe = load_universe(args.universe)
    shares_outstanding = load_shares_outstanding(args.shares)

    payload = build_output(
        rows=rows,
        universe=universe,
        shares_outstanding=shares_outstanding,
        report_period=report_period,
        top_n=args.top_n,
        rank_by=args.rank_by,
        input_path=args.input,
        shares_path=args.shares,
        universe_path=args.universe,
    )

    write_json(args.output, payload)
    write_json(summary_path_for(args.output), payload["meta"])
    print(json.dumps(payload["meta"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
