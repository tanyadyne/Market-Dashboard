"""
One-off repair script for leaders_score_history.json.

Problem: before the ET-date fix, the history file accumulated ghost entries —
a Sunday (2026-04-19) from a run that interpreted UTC Saturday/Sunday as today,
and a 2026-04-22 entry from a post-close run that crossed UTC midnight.

This script:
  1. Loads leaders_score_history.json
  2. Keeps only dates that are weekdays (Monday–Friday)
  3. For each ticker, keeps only the array positions corresponding to those dates
  4. Additionally strips any date that's strictly in the future relative to US Eastern

Run once from the repo root:
    python repair_history.py
Then commit the updated leaders_score_history.json.
"""
import json
from datetime import date, datetime, timezone, timedelta


def main():
    path = "leaders_score_history.json"
    with open(path) as f:
        h = json.load(f)

    dates_old = h.get("dates", [])
    scores = h.get("d", {})

    # Determine today in US Eastern (approx UTC-4; precision doesn't matter for date-level filter)
    now_utc = datetime.now(timezone.utc)
    et_now = now_utc.astimezone(timezone(timedelta(hours=-4)))
    today_et = et_now.date()

    # Which indices to keep? Only weekdays, and not strictly after today-ET.
    keep_idx = []
    dates_new = []
    for i, d in enumerate(dates_old):
        try:
            dt = date.fromisoformat(d)
        except Exception:
            print(f"  skipping unparseable date: {d}")
            continue
        if dt.weekday() >= 5:  # 5=Sat, 6=Sun
            print(f"  removing weekend ghost: {d}")
            continue
        if dt > today_et:
            print(f"  removing future ghost: {d}")
            continue
        keep_idx.append(i)
        dates_new.append(d)

    if len(keep_idx) == len(dates_old):
        print("No changes needed — history is clean.")
        return

    # Rebuild each ticker's arrays keeping only the positions in keep_idx
    for tk, entry in scores.items():
        for key in ("s", "r", "wr", "tz"):
            if key in entry and isinstance(entry[key], list):
                arr = entry[key]
                new_arr = []
                for i in keep_idx:
                    if i < len(arr):
                        new_arr.append(arr[i])
                entry[key] = new_arr

    h["dates"] = dates_new
    with open(path, "w") as f:
        json.dump(h, f, separators=(",", ":"))
    print(f"Repaired: {len(dates_old)} dates → {len(dates_new)} dates")


if __name__ == "__main__":
    main()
