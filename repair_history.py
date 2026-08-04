"""One-off repairs for ``leaders_score_history.json``.

Modes:
  python repair_history.py
      Remove weekend/future ghost entries.
  python repair_history.py --strip-last
      Remove the most recent entry.
  python repair_history.py --repair-misdated-snapshot
      Repair a stale-date snapshot inserted after a newer trading session.
      The snapshot is copied onto the preceding newer date, then its duplicate
      date slot is removed.  This never fetches data or recalculates ranks.
"""

import json
import sys
from datetime import date, datetime, timedelta, timezone


DATE_ALIGNED_ENTRY_FIELDS = {
    "r",
    "s",
    "seen",
    "tm",
    "tz",
    "vcp_coil_1",
    "vcp_coil_2",
    "wr",
}
DATE_ALIGNED_ROOT_FIELDS = {"wr_total"}


def _parse_iso_date(value):
    try:
        return date.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


def find_misdated_snapshot_index(dates):
    """Find a stale duplicate inserted after a newer session, if one exists.

    A normal history is strictly chronological. We only repair an older date
    that duplicates an earlier date, which keeps this deliberately conservative.
    """
    if not isinstance(dates, list):
        return None
    parsed = [_parse_iso_date(value) for value in dates]
    for index in range(1, len(parsed)):
        current = parsed[index]
        previous = parsed[index - 1]
        if current is None or previous is None or current >= previous:
            continue
        if dates[index] in dates[:index]:
            return index
    return None


def _copy_then_remove(values, target_index, source_index):
    """Move one aligned history observation, when this row contains it."""
    if not isinstance(values, list) or len(values) <= source_index:
        return values
    values[target_index] = values[source_index]
    del values[source_index]
    return values


def repair_misdated_snapshot(history):
    """Move the stale duplicate snapshot to its immediately preceding session.

    Returns metadata describing the repair, or ``None`` when no safe repair is
    available. Only aligned arrays are touched; live data remains untouched.
    """
    if not isinstance(history, dict):
        return None
    dates = history.get("dates")
    source_index = find_misdated_snapshot_index(dates)
    if source_index is None:
        return None

    target_index = source_index - 1
    source_date = dates[source_index]
    target_date = dates[target_index]

    scores = history.get("d")
    if isinstance(scores, dict):
        for entry in scores.values():
            if not isinstance(entry, dict):
                continue
            for field in DATE_ALIGNED_ENTRY_FIELDS:
                if field in entry:
                    _copy_then_remove(entry[field], target_index, source_index)

    for field in DATE_ALIGNED_ROOT_FIELDS:
        if field in history:
            _copy_then_remove(history[field], target_index, source_index)

    del dates[source_index]
    return {
        "source_date": source_date,
        "target_date": target_date,
        "source_index": source_index,
        "target_index": target_index,
    }


def _trim_history(history, keep_idx):
    """Rebuild known aligned history arrays with only ``keep_idx`` slots."""
    scores = history.get("d") or {}
    for entry in scores.values():
        if not isinstance(entry, dict):
            continue
        for field in DATE_ALIGNED_ENTRY_FIELDS:
            values = entry.get(field)
            if isinstance(values, list):
                entry[field] = [values[index] for index in keep_idx if index < len(values)]

    for field in DATE_ALIGNED_ROOT_FIELDS:
        values = history.get(field)
        if isinstance(values, list):
            history[field] = [values[index] for index in keep_idx if index < len(values)]


def main():
    path = "leaders_score_history.json"
    with open(path, encoding="utf-8") as handle:
        history = json.load(handle)

    dates_old = list(history.get("dates", []))
    strip_last = "--strip-last" in sys.argv
    repair_misdated = "--repair-misdated-snapshot" in sys.argv

    if strip_last and repair_misdated:
        raise SystemExit("Use only one repair mode at a time.")

    if repair_misdated:
        repaired = repair_misdated_snapshot(history)
        if repaired is None:
            print("No misdated duplicate snapshot found - nothing to repair.")
            return
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(history, handle, separators=(",", ":"))
        print(
            "Repaired misdated snapshot: "
            f"moved {repaired['source_date']} slot onto {repaired['target_date']} "
            f"({len(dates_old)} dates -> {len(history['dates'])} dates)"
        )
        return

    if strip_last:
        if not dates_old:
            print("History is empty - nothing to strip.")
            return
        removed = dates_old[-1]
        dates_new = dates_old[:-1]
        keep_idx = list(range(len(dates_new)))
        print(f"  stripping last entry: {removed}")
    else:
        # Determine today in US Eastern (approx UTC-4).
        now_utc = datetime.now(timezone.utc)
        et_now = now_utc.astimezone(timezone(timedelta(hours=-4)))
        today_et = et_now.date()

        keep_idx = []
        dates_new = []
        for index, value in enumerate(dates_old):
            current = _parse_iso_date(value)
            if current is None:
                print(f"  skipping unparseable date: {value}")
                continue
            if current.weekday() >= 5:
                print(f"  removing weekend ghost: {value}")
                continue
            if current > today_et:
                print(f"  removing future ghost: {value}")
                continue
            keep_idx.append(index)
            dates_new.append(value)

        if len(keep_idx) == len(dates_old):
            print("No changes needed - history is clean.")
            return

    _trim_history(history, keep_idx)
    history["dates"] = dates_new
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(history, handle, separators=(",", ":"))
    print(f"Repaired: {len(dates_old)} dates -> {len(dates_new)} dates")


if __name__ == "__main__":
    main()
