#!/usr/bin/env python3
"""
Parse SEC 13F holdings by CUSIP for the stock screener universe.

This is step 2 of the ownership tile pipeline:
  - enumerate SEC 13F filings from the quarterly EDGAR form index
  - download/cache each submission
  - parse the information table
  - emit matched holdings whose CUSIP/CINS appears in ticker_cusip_map.json
    or in the SEC-verified alternate identifier map

Large raw output is written as NDJSON so later aggregation can stream it.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.request import Request, urlopen


SEC_ARCHIVES = "https://www.sec.gov/Archives"
DEFAULT_CACHE_DIR = Path(".sec_13f_cache")
DEFAULT_CUSIP_MAP = Path("ticker_cusip_map.json")
DEFAULT_CUSIP_ALIASES = Path("ticker_cusip_aliases.json")
DEFAULT_PRICE_MAP = Path("leaders_mcap.json")
DEFAULT_FORM_TYPES = ("13F-HR",)
DEFAULT_USER_AGENT = "WIN_MarketDashboard/1.0 contact:tanya@example.com"


@dataclass(frozen=True)
class Filing:
    form_type: str
    manager_name: str
    manager_cik: str
    filing_date: str
    file_name: str

    @property
    def accession(self) -> str:
        return Path(self.file_name).name.replace(".txt", "")

    @property
    def accession_nodash(self) -> str:
        return self.accession.replace("-", "")

    @property
    def url(self) -> str:
        return f"{SEC_ARCHIVES}/{self.file_name}"


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


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def normalize_cusip(value: Any) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value or "").upper())


def normalize_report_period(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%m-%d-%Y", "%Y%m%d"):
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue
    return text


def infer_report_period(filing_year: int, filing_quarter: int) -> str:
    # Most 13F-HR filings in a calendar quarter report the prior quarter end.
    if filing_quarter == 1:
        return f"{filing_year - 1}-12-31"
    if filing_quarter == 2:
        return f"{filing_year}-03-31"
    if filing_quarter == 3:
        return f"{filing_year}-06-30"
    return f"{filing_year}-09-30"


def quarter_label(report_period: str) -> str:
    parts = [int(x) for x in report_period.split("-")]
    dt = date(parts[0], parts[1], parts[2])
    month_to_q = {3: "Q1", 6: "Q2", 9: "Q3", 12: "Q4"}
    return f"{month_to_q.get(dt.month, 'Q?')} {dt.year}"


def sec_request_text(url: str, user_agent: str, rate_limit_seconds: float) -> str:
    time.sleep(max(0.0, rate_limit_seconds))
    headers = {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate", "Host": "www.sec.gov"}
    with urlopen(Request(url, headers=headers), timeout=60) as response:
        encoding = (response.headers.get("Content-Encoding") or "").lower()
        data = response.read()
    if "gzip" in encoding:
        data = gzip.decompress(data)
    return data.decode("utf-8", "replace")


def cache_path_for(cache_dir: Path, filing: Filing) -> Path:
    cik_dir = cache_dir / "submissions" / filing.manager_cik
    return cik_dir / f"{filing.accession}.txt.gz"


def fetch_submission(
    filing: Filing,
    cache_dir: Path,
    user_agent: str,
    rate_limit_seconds: float,
    refresh: bool,
) -> tuple[str, bool]:
    path = cache_path_for(cache_dir, filing)
    if path.exists() and not refresh:
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
            return fh.read(), True
    text = sec_request_text(filing.url, user_agent, rate_limit_seconds)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write(text)
    return text, False


def load_form_index(year: int, quarter: int, cache_dir: Path, user_agent: str, refresh: bool) -> str:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "indexes" / f"{year}_QTR{quarter}_form.idx"
    if path.exists() and not refresh:
        return path.read_text(encoding="utf-8", errors="replace")
    url = f"{SEC_ARCHIVES}/edgar/full-index/{year}/QTR{quarter}/form.idx"
    text = sec_request_text(url, user_agent, 0.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text


def parse_form_index(text: str, form_types: set[str]) -> list[Filing]:
    filings: list[Filing] = []
    for line in text.splitlines():
        form_type = line[:12].strip()
        if form_type not in form_types:
            continue
        try:
            company_and_tail = line[12:].rstrip()
            company, cik, filing_date, file_name = company_and_tail.rsplit(None, 3)
        except ValueError:
            continue
        filings.append(
            Filing(
                form_type=form_type,
                manager_name=company.strip(),
                manager_cik=cik.strip(),
                filing_date=filing_date.strip(),
                file_name=file_name.strip(),
            )
        )
    return filings


def load_cusip_map(path: Path, aliases_path: Path | None = None) -> dict[str, list[str]]:
    payload = read_json(path)
    if not payload or "tickers" not in payload:
        raise ValueError(f"Invalid CUSIP map: {path}")
    by_cusip: dict[str, list[str]] = {}

    def add_mapping(cusip: Any, ticker: str) -> None:
        normalized = normalize_cusip(cusip)
        if normalized and ticker not in by_cusip.setdefault(normalized, []):
            by_cusip[normalized].append(ticker)

    for ticker, item in payload["tickers"].items():
        add_mapping(item.get("cusip"), ticker)

    if aliases_path and aliases_path.exists():
        aliases_payload = read_json(aliases_path, {}) or {}
        for ticker, entry in (aliases_payload.get("tickers", {}) or {}).items():
            alias_values = entry.get("cusips", []) if isinstance(entry, dict) else entry
            for cusip in alias_values or []:
                add_mapping(cusip, ticker)
    return by_cusip


def as_positive_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def load_price_map(path: Path | None) -> dict[str, float]:
    if path is None or not path.exists():
        return {}
    payload = read_json(path, {})
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


def infer_row_value_multiplier(
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


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def text_of(elem: ET.Element | None) -> str | None:
    if elem is None or elem.text is None:
        return None
    text = elem.text.strip()
    return text or None


def child_text(elem: ET.Element, *names: str) -> str | None:
    targets = {name.lower() for name in names}
    for child in list(elem):
        if local_name(child.tag).lower() in targets:
            return text_of(child)
    return None


def nested_child(elem: ET.Element, *names: str) -> ET.Element | None:
    targets = {name.lower() for name in names}
    for child in elem.iter():
        if child is elem:
            continue
        if local_name(child.tag).lower() in targets:
            return child
    return None


def parse_number(value: Any) -> float | int | None:
    text = str(value or "").replace(",", "").strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return int(number) if number.is_integer() else number


def xml_blocks(submission_text: str) -> Iterable[str]:
    yield from re.findall(r"<XML>\s*(.*?)\s*</XML>", submission_text, flags=re.I | re.S)


def parse_period_of_report(blocks: list[str]) -> str | None:
    for block in blocks:
        if "periodOfReport" not in block and "periodofreport" not in block.lower():
            continue
        try:
            root = ET.fromstring(block.encode("utf-8"))
        except ET.ParseError:
            continue
        for elem in root.iter():
            if local_name(elem.tag).lower() == "periodofreport":
                return text_of(elem)
    return None


def parse_info_tables(blocks: list[str]) -> tuple[list[ET.Element], int]:
    tables: list[ET.Element] = []
    parse_failures = 0
    for block in blocks:
        lower = block.lower()
        if "infotable" not in lower and "informationtable" not in lower:
            continue
        try:
            root = ET.fromstring(block.encode("utf-8"))
        except ET.ParseError:
            parse_failures += 1
            continue
        if local_name(root.tag).lower() == "infotable":
            tables.append(root)
        else:
            tables.extend(elem for elem in root.iter() if local_name(elem.tag).lower() == "infotable")
    return tables, parse_failures


def holding_from_info_table(elem: ET.Element) -> dict[str, Any]:
    shrs = nested_child(elem, "shrsOrPrnAmt", "shrsOrPrn")
    voting = nested_child(elem, "votingAuthority")
    value_reported = parse_number(child_text(elem, "value"))
    shares = parse_number(child_text(shrs, "sshPrnamt", "shrsPrnAmount") if shrs is not None else None)
    row: dict[str, Any] = {
        "issuer": child_text(elem, "nameOfIssuer"),
        "title": child_text(elem, "titleOfClass"),
        "cusip": normalize_cusip(child_text(elem, "cusip")),
        "value_reported": int(value_reported) if isinstance(value_reported, (int, float)) else None,
        "value_usd": None,
        "shares": int(shares) if isinstance(shares, (int, float)) else None,
        "share_type": child_text(shrs, "sshPrnamtType", "shrsPrnAmountType") if shrs is not None else None,
        "put_call": child_text(elem, "putCall"),
        "investment_discretion": child_text(elem, "investmentDiscretion"),
        "other_manager": child_text(elem, "otherManager"),
        "voting_sole": parse_number(child_text(voting, "Sole")) if voting is not None else None,
        "voting_shared": parse_number(child_text(voting, "Shared")) if voting is not None else None,
        "voting_none": parse_number(child_text(voting, "None")) if voting is not None else None,
    }
    return row


def process_filing(
    filing: Filing,
    cusip_to_tickers: dict[str, list[str]],
    prices: dict[str, float],
    cache_dir: Path,
    user_agent: str,
    rate_limit_seconds: float,
    refresh: bool,
    fallback_report_period: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    submission, from_cache = fetch_submission(filing, cache_dir, user_agent, rate_limit_seconds, refresh)
    blocks = list(xml_blocks(submission))
    report_period = normalize_report_period(parse_period_of_report(blocks)) or fallback_report_period
    info_tables, parse_failures = parse_info_tables(blocks)

    rows: list[dict[str, Any]] = []
    parsed_rows: list[dict[str, Any]] = []
    info_count = 0
    matched_info_count = 0
    for info in info_tables:
        info_count += 1
        parsed_rows.append(holding_from_info_table(info))

    value_multiplier_counts: dict[int, int] = {}
    value_normalization_sources: dict[str, int] = {}
    for base in parsed_rows:
        value_reported = base.get("value_reported")
        shares = base.get("shares")
        cusip = base.get("cusip")
        if not cusip or cusip not in cusip_to_tickers:
            continue
        matched_info_count += 1
        for ticker in cusip_to_tickers[cusip]:
            row_base = dict(base)
            if isinstance(value_reported, (int, float)) and isinstance(shares, (int, float)) and shares > 0:
                multiplier, normalization_source = infer_row_value_multiplier(
                    ticker,
                    int(value_reported),
                    int(shares),
                    prices,
                )
                row_base["value_usd"] = int(value_reported * multiplier)
                row_base["value_multiplier"] = multiplier
                row_base["value_normalization_source"] = normalization_source
                value_multiplier_counts[multiplier] = value_multiplier_counts.get(multiplier, 0) + 1
                value_normalization_sources[normalization_source] = value_normalization_sources.get(
                    normalization_source, 0
                ) + 1
            rows.append(
                {
                    "ticker": ticker,
                    "manager_name": filing.manager_name,
                    "manager_cik": filing.manager_cik,
                    "form_type": filing.form_type,
                    "accession": filing.accession,
                    "filing_date": filing.filing_date,
                    "report_period": report_period,
                    **row_base,
                }
            )

    stats = {
        "accession": filing.accession,
        "manager_name": filing.manager_name,
        "manager_cik": filing.manager_cik,
        "form_type": filing.form_type,
        "filing_date": filing.filing_date,
        "report_period": report_period,
        "from_cache": from_cache,
        "xml_blocks": len(blocks),
        "info_table_rows": info_count,
        "matched_info_table_rows": matched_info_count,
        "emitted_rows": len(rows),
        "parse_failures": parse_failures,
        "value_multiplier_counts": {str(key): value for key, value in sorted(value_multiplier_counts.items())},
        "value_normalization_sources": dict(sorted(value_normalization_sources.items())),
    }
    return rows, stats


def summary_path_for(output: Path) -> Path:
    return output.with_suffix(".summary.json")


def progress_path_for(output: Path) -> Path:
    return output.with_suffix(".progress.json")


def load_progress(path: Path) -> dict[str, Any]:
    return read_json(path, {"processed": {}, "errors": {}})


def save_progress(path: Path, progress: dict[str, Any]) -> None:
    write_json(path, progress)


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    write_json(path, payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--quarter", type=int, default=2, choices=(1, 2, 3, 4))
    parser.add_argument("--report-period", default=None)
    parser.add_argument("--forms", default=",".join(DEFAULT_FORM_TYPES))
    parser.add_argument("--cusip-map", type=Path, default=DEFAULT_CUSIP_MAP)
    parser.add_argument("--cusip-aliases", type=Path, default=DEFAULT_CUSIP_ALIASES)
    parser.add_argument("--price-map", type=Path, default=DEFAULT_PRICE_MAP)
    parser.add_argument("--no-price-map", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-filings", type=int, default=None)
    parser.add_argument("--start-at", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--rate-limit-seconds", type=float, default=0.12)
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_env_file()
    user_agent = os.environ.get("SEC_USER_AGENT", DEFAULT_USER_AGENT)
    form_types = {item.strip() for item in args.forms.split(",") if item.strip()}
    report_period = args.report_period or infer_report_period(args.year, args.quarter)
    output = args.output or Path(f"13f_holdings_{args.year}_q{args.quarter}_raw.ndjson")
    summary_path = summary_path_for(output)
    progress_path = progress_path_for(output)

    cusip_to_tickers = load_cusip_map(
        args.cusip_map,
        args.cusip_aliases if args.cusip_aliases.exists() else None,
    )
    prices = {} if args.no_price_map else load_price_map(args.price_map)
    index_text = load_form_index(args.year, args.quarter, args.cache_dir, user_agent, args.refresh)
    filings = parse_form_index(index_text, form_types)
    selected = filings[args.start_at :]
    if args.max_filings is not None:
        selected = selected[: args.max_filings]

    if args.resume:
        progress = load_progress(progress_path)
        mode = "a"
    else:
        progress = {"processed": {}, "errors": {}}
        mode = "w"

    processed = progress.setdefault("processed", {})
    errors = progress.setdefault("errors", {})
    totals = {
        "filings_available": len(filings),
        "filings_selected": len(selected),
        "filings_processed_this_run": 0,
        "filings_skipped_already_processed": 0,
        "filings_with_matches": 0,
        "info_table_rows": 0,
        "matched_info_table_rows": 0,
        "emitted_rows": 0,
        "parse_failures": 0,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open(mode, encoding="utf-8") as out:
        for idx, filing in enumerate(selected, start=args.start_at):
            if args.resume and filing.accession in processed:
                totals["filings_skipped_already_processed"] += 1
                continue
            try:
                rows, stats = process_filing(
                    filing,
                    cusip_to_tickers,
                    prices,
                    args.cache_dir,
                    user_agent,
                    args.rate_limit_seconds,
                    args.refresh,
                    report_period,
                )
                for row in rows:
                    out.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")
                out.flush()
                processed[filing.accession] = stats
                totals["filings_processed_this_run"] += 1
                totals["filings_with_matches"] += 1 if stats["emitted_rows"] else 0
                totals["info_table_rows"] += stats["info_table_rows"]
                totals["matched_info_table_rows"] += stats["matched_info_table_rows"]
                totals["emitted_rows"] += stats["emitted_rows"]
                totals["parse_failures"] += stats["parse_failures"]
            except Exception as exc:
                errors[filing.accession] = {
                    "manager_name": filing.manager_name,
                    "manager_cik": filing.manager_cik,
                    "filing_date": filing.filing_date,
                    "error": f"{type(exc).__name__}: {exc}",
                }

            done = totals["filings_processed_this_run"] + totals["filings_skipped_already_processed"]
            if done % max(1, args.progress_every) == 0:
                progress["updated"] = now_iso()
                progress["year"] = args.year
                progress["quarter"] = args.quarter
                progress["report_period"] = report_period
                save_progress(progress_path, progress)

    progress["updated"] = now_iso()
    progress["year"] = args.year
    progress["quarter"] = args.quarter
    progress["report_period"] = report_period
    save_progress(progress_path, progress)

    processed_stats = list(processed.values())
    cumulative = {
        "filings_processed_total": len(processed_stats),
        "errors_total": len(errors),
        "filings_with_matches_total": sum(1 for item in processed_stats if item.get("emitted_rows", 0) > 0),
        "info_table_rows_total": sum(int(item.get("info_table_rows", 0)) for item in processed_stats),
        "matched_info_table_rows_total": sum(int(item.get("matched_info_table_rows", 0)) for item in processed_stats),
        "emitted_rows_total": sum(int(item.get("emitted_rows", 0)) for item in processed_stats),
        "cache_hits_total": sum(1 for item in processed_stats if item.get("from_cache")),
    }
    summary = {
        "updated": now_iso(),
        "source": "SEC EDGAR 13F-HR filings",
        "year": args.year,
        "quarter": args.quarter,
        "form_types": sorted(form_types),
        "report_period": report_period,
        "report_period_label": quarter_label(report_period),
        "cusip_map": str(args.cusip_map),
        "cusip_aliases": str(args.cusip_aliases) if args.cusip_aliases.exists() else None,
        "price_map": None if args.no_price_map else str(args.price_map),
        "price_map_tickers": len(prices),
        "output": str(output),
        "progress": str(progress_path),
        "filings_available": len(filings),
        "filings_selected_this_run": len(selected),
        "run_totals": totals,
        "cumulative_totals": cumulative,
        "note": "Raw NDJSON output is ignored by git; rerun with --resume to continue safely.",
    }
    write_summary(summary_path, summary)

    print(
        "processed={filings_processed_this_run} skipped={filings_skipped_already_processed} "
        "emitted={emitted_rows} cumulative_emitted={emitted_rows_total} errors={errors_total}".format(
            **totals,
            **cumulative,
        )
    )
    print(f"summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
