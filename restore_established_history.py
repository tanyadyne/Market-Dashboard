#!/usr/bin/env python3
"""Restore a complete historical rank snapshot for approved constituents.

The market-cap metadata outage removed valid 3 August ranks from a subset of
established tickers.  Filling only those blank cells would mix ranks produced
against two different universes.  This repair rebuilds the validated
cross-section from the audited pre-guard snapshot, excluding unapproved
tickers and densely reindexing the remaining ranks and denominator.
"""

import argparse
import copy
import json
import subprocess
import sys

from established_market_cap import ESTABLISHED_MCAP_FILE
from rank_history import WEEKLY_RANK_TOTALS_FIELD, infer_ranked_total
from repair_history import DATE_ALIGNED_ENTRY_FIELDS


HISTORY_FILE = "leaders_score_history.json"
SEED_COMMIT = "c9b275e915d99391567dea18c301c28f67a90847"
TARGET_DATE = "2026-08-03"


def git_json(commit, filename):
    """Read an immutable JSON artifact without changing the checkout."""
    result = subprocess.run(
        ["git", "show", f"{commit}:{filename}"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(
            f"Could not read history seed {commit}:{filename}: "
            f"{result.stderr.strip() or 'unknown git error'}"
        )
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"History seed {commit}:{filename} is not valid JSON") from exc


def target_index(history, target_date):
    dates = history.get("dates") if isinstance(history, dict) else None
    if not isinstance(dates, list) or target_date not in dates:
        raise ValueError(f"History does not contain target session {target_date}")
    return dates.index(target_date)


def approved_tickers(registry):
    records = registry.get("tickers") if isinstance(registry, dict) else {}
    if not isinstance(records, dict):
        return set()
    return {
        ticker
        for ticker, record in records.items()
        if isinstance(record, dict) and record.get("status") == "approved"
    }


def set_aligned_value(entry, field, index, value, date_count):
    """Set one date slot without disturbing the surrounding history."""
    values = entry.get(field)
    if not isinstance(values, list):
        values = []
        entry[field] = values
    while len(values) < date_count:
        values.append(None)
    values[index] = copy.deepcopy(value)


def restore_snapshot(current, seed, registry, target_date=TARGET_DATE):
    """Restore and reindex the approved historical cross-section.

    Returns repair metadata.  Raises ``ValueError`` before mutating anything if
    the source snapshot is incomplete or date alignment is unsafe.
    """
    current_index = target_index(current, target_date)
    seed_index = target_index(seed, target_date)
    current_scores = current.get("d") if isinstance(current.get("d"), dict) else {}
    seed_scores = seed.get("d") if isinstance(seed.get("d"), dict) else {}
    source_denominator = infer_ranked_total(seed_scores, seed_index)
    if source_denominator is None:
        raise ValueError(f"Seed {target_date} ranks are incomplete; refusing to mix universes")

    approved = approved_tickers(registry)
    date_count = len(current.get("dates") or [])
    ranked_rows = []
    for ticker in sorted(approved):
        source_entry = seed_scores.get(ticker)
        if not isinstance(source_entry, dict):
            continue
        source_ranks = source_entry.get("wr")
        if not isinstance(source_ranks, list) or seed_index >= len(source_ranks):
            continue
        source_rank = source_ranks[seed_index]
        if source_rank is None:
            continue
        if isinstance(source_rank, bool):
            raise ValueError(f"Seed {target_date} rank for {ticker} is invalid")
        try:
            numeric_rank = float(source_rank)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Seed {target_date} rank for {ticker} is invalid") from exc
        if not numeric_rank.is_integer() or numeric_rank < 1:
            raise ValueError(f"Seed {target_date} rank for {ticker} is invalid")
        source_rank = int(numeric_rank)
        ranked_rows.append((source_rank, ticker, source_entry))

    ranked_rows.sort()
    source_ranks = [row[0] for row in ranked_rows]
    if len(source_ranks) != len(set(source_ranks)):
        raise ValueError(
            f"Approved {target_date} ranks contain a duplicate; refusing unstable reindex"
        )

    restored = []
    for rank, (_, ticker, source_entry) in enumerate(ranked_rows, start=1):
        target_entry = current_scores.setdefault(ticker, {})
        for field in DATE_ALIGNED_ENTRY_FIELDS:
            source_values = source_entry.get(field)
            if isinstance(source_values, list) and seed_index < len(source_values):
                set_aligned_value(
                    target_entry,
                    field,
                    current_index,
                    rank if field == "wr" else source_values[seed_index],
                    date_count,
                )
        # ``wr`` must exist for every restored ranked row even if a legacy seed
        # lacks another aligned field.
        set_aligned_value(target_entry, "wr", current_index, rank, date_count)
        restored.append(ticker)

    restored_set = set(restored)
    excluded = []
    # Remove every remaining target-date rank outside the reconstructed
    # cross-section. A partial snapshot can contain a ticker that is absent
    # from the audited seed, and its rank cannot be reconciled against the
    # complete universe. The output date must remain an exact cross-section.
    for ticker, target_entry in current_scores.items():
        if ticker in restored_set or not isinstance(target_entry, dict):
            continue
        target_values = target_entry.get("wr")
        if not isinstance(target_values, list) or current_index >= len(target_values):
            continue
        if target_values[current_index] is None:
            continue
        for field in DATE_ALIGNED_ENTRY_FIELDS:
            set_aligned_value(target_entry, field, current_index, None, date_count)
        excluded.append(ticker)

    totals = current.get(WEEKLY_RANK_TOTALS_FIELD)
    if not isinstance(totals, list):
        totals = []
        current[WEEKLY_RANK_TOTALS_FIELD] = totals
    while len(totals) < date_count:
        totals.append(None)
    totals[current_index] = len(restored)
    return {
        "target_date": target_date,
        "source_denominator": source_denominator,
        "denominator": len(restored),
        "restored": restored,
        "excluded": sorted(excluded),
    }


def load_json(filename):
    with open(filename, encoding="utf-8") as handle:
        return json.load(handle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-commit", default=SEED_COMMIT)
    parser.add_argument("--target-date", default=TARGET_DATE)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    try:
        current = load_json(HISTORY_FILE)
        registry = load_json(ESTABLISHED_MCAP_FILE)
        seed = git_json(args.seed_commit, HISTORY_FILE)
        repaired = restore_snapshot(current, seed, registry, args.target_date)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1

    print(
        f"Established history repair: {len(repaired['restored'])} rows reindexed "
        f"for {repaired['target_date']} from source denominator "
        f"{repaired['source_denominator']} to {repaired['denominator']}"
    )
    if args.dry_run:
        print("Dry run only; no files written.")
        return 0
    with open(HISTORY_FILE, "w", encoding="utf-8") as handle:
        json.dump(current, handle, separators=(",", ":"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
