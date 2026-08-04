#!/usr/bin/env python3
"""Restore established screener rows after a temporary market-cap metadata outage.

This is intentionally a fast repair, not a replacement for ``fetch_leaders``.
It restores only entries that existed in the audited pre-provenance snapshot,
meet today's market-cap floor, and have an approved share-count record.  The
following intraday reranker then refreshes price, market-cap estimate and rank.
"""

import argparse
import copy
import json
import subprocess
import sys
from datetime import datetime, timezone

from established_market_cap import (
    ESTABLISHED_MCAP_FILE,
    ESTABLISHED_RECORD_VERSION,
    positive_int,
)
from fetch_leaders import (
    ESTABLISHED_MCAP_FALLBACK_BLOCKLIST,
    HARD_EXCLUDE,
    INTRADAY_BASELINES_FILE,
    min_mcap_for_industry,
)


SEED_COMMIT = "c9b275e915d99391567dea18c301c28f67a90847"
LEADERS_FILE = "leaders.json"
SCREENER_TICKERS_FILE = "screener_tickers.json"


def git_json(commit, filename):
    """Read a versioned JSON artifact without checking out or mutating Git."""
    result = subprocess.run(
        ["git", "show", f"{commit}:{filename}"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(
            f"Could not read restoration seed {commit}:{filename}: "
            f"{result.stderr.strip() or 'unknown git error'}"
        )
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Restoration seed {commit}:{filename} is not valid JSON") from exc


def approved_record_from_snapshot(entry, ticker):
    """Build the only kind of record allowed to bypass a provider outage."""
    if ticker in ESTABLISHED_MCAP_FALLBACK_BLOCKLIST:
        return None
    cap = positive_int(entry.get("mc"))
    try:
        price = float(entry.get("p"))
    except (TypeError, ValueError):
        return None
    if not cap or price <= 0:
        return None
    shares = int(round(cap / price))
    if shares <= 0:
        return None
    return {
        "status": "approved",
        "shares": shares,
        "reference_cap": cap,
        "reference_price": price,
        "approved_on": "2026-08-03",
        "provenance": "audited_pre_provenance_snapshot",
    }


def eligible_for_rank_restore(entry, record, industry):
    """Keep the current cap policy intact while repairing only a metadata loss."""
    if not record:
        return False
    return positive_int(record.get("reference_cap")) >= min_mcap_for_industry(industry or entry.get("th"))


def prepare_restoration(seed_entries, current_entries, seed_baselines, industries):
    """Return restored rows/baselines plus the approved registry payload.

    Kept pure so the admission boundaries can be unit-tested without invoking
    Git or a market-data client.
    """
    current_tickers = {row.get("t") for row in current_entries if row.get("t")}
    seed_by_ticker = {row.get("t"): row for row in seed_entries if row.get("t")}
    registry = {}
    restored_entries = []
    restored_baselines = {}
    skipped_floor = []
    skipped_baseline = []

    for ticker in sorted(seed_by_ticker):
        if ticker in HARD_EXCLUDE:
            continue
        entry = seed_by_ticker[ticker]
        record = approved_record_from_snapshot(entry, ticker)
        if record:
            registry[ticker] = record
        if ticker in current_tickers or ticker in ESTABLISHED_MCAP_FALLBACK_BLOCKLIST:
            continue
        industry = (industries or {}).get(ticker, "")
        if not eligible_for_rank_restore(entry, record, industry):
            skipped_floor.append(ticker)
            continue
        baseline = (seed_baselines or {}).get(ticker)
        if not isinstance(baseline, dict):
            skipped_baseline.append(ticker)
            continue
        restored = copy.deepcopy(entry)
        restored.pop("rh", None)
        restored["mc"] = record["reference_cap"]
        restored_entries.append(restored)
        restored_baselines[ticker] = copy.deepcopy(baseline)

    return {
        "registry": registry,
        "entries": restored_entries,
        "baselines": restored_baselines,
        "skipped_floor": skipped_floor,
        "skipped_baseline": skipped_baseline,
    }


def load_json(filename):
    with open(filename) as handle:
        return json.load(handle)


def write_json(filename, payload):
    with open(filename, "w") as handle:
        json.dump(payload, handle, separators=(",", ":"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-commit", default=SEED_COMMIT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    try:
        seed_leaders = git_json(args.seed_commit, LEADERS_FILE)
        seed_baseline_payload = git_json(args.seed_commit, INTRADAY_BASELINES_FILE)
        current_leaders = load_json(LEADERS_FILE)
        current_baseline_payload = load_json(INTRADAY_BASELINES_FILE)
        market_caps = load_json("leaders_mcap.json")
    except (OSError, RuntimeError) as exc:
        print(f"ERROR: {exc}")
        return 1

    seed_entries = seed_leaders.get("e") or []
    current_entries = current_leaders.get("e") or []
    seed_baselines = seed_baseline_payload.get("d") or {}
    current_baselines = current_baseline_payload.get("d") or {}
    industries = market_caps.get("industries") or {}
    prepared = prepare_restoration(
        seed_entries,
        current_entries,
        seed_baselines,
        industries,
    )

    registry_payload = {
        "version": ESTABLISHED_RECORD_VERSION,
        "updated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "seed_commit": args.seed_commit,
        "tickers": prepared["registry"],
    }
    try:
        existing_registry = load_json(ESTABLISHED_MCAP_FILE)
        existing_records = existing_registry.get("tickers") or {}
        if isinstance(existing_records, dict):
            registry_payload["tickers"].update(existing_records)
    except (OSError, json.JSONDecodeError):
        pass

    restored_entries = prepared["entries"]
    restored_baselines = prepared["baselines"]
    print(
        "Established-universe repair: "
        f"{len(restored_entries)} rank-eligible rows to restore; "
        f"{len(prepared['skipped_floor'])} remain below the current cap floor; "
        f"{len(prepared['skipped_baseline'])} missing a baseline"
    )
    if prepared["skipped_floor"]:
        print(f"  Current-policy exclusions: {prepared['skipped_floor']}")
    if args.dry_run:
        print("Dry run only; no files written.")
        return 0

    current_entries.extend(restored_entries)
    current_entries.sort(key=lambda row: row.get("t", ""))
    current_baselines.update(restored_baselines)
    current_leaders["e"] = current_entries
    current_baseline_payload["d"] = current_baselines
    current_baseline_payload.setdefault("meta", {})["count"] = len(current_baselines)
    current_baseline_payload["meta"]["updated"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    current_leaders.setdefault("meta", {})["count"] = len(current_entries)
    current_leaders["meta"]["universe"] = len(current_entries)
    current_leaders["meta"]["established_mcap_restored"] = len(restored_entries)

    screener_tickers = sorted({row.get("t") for row in current_entries if row.get("t")})
    write_json(LEADERS_FILE, current_leaders)
    write_json(INTRADAY_BASELINES_FILE, current_baseline_payload)
    write_json(ESTABLISHED_MCAP_FILE, registry_payload)
    write_json(
        SCREENER_TICKERS_FILE,
        {
            "updated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "count": len(screener_tickers),
            "tickers": screener_tickers,
        },
    )
    print(f"Restored: {[row['t'] for row in restored_entries]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
