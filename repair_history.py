"""
One-off repair script for leaders_score_history.json.

Modes:
  python repair_history.py           — remove weekend/future ghost entries
  python repair_history.py --strip-last  — remove the most recent entry (e.g. a ghost
                                           date created by a post-midnight-ET run)
"""
import json
import sys
from datetime import date, datetime, timezone, timedelta


def main():
    path = "leaders_score_history.json"
    with open(path) as f:
        h = json.load(f)

    dates_old = h.get("dates", [])
    scores = h.get("d", {})

    strip_last = "--strip-last" in sys.argv

    if strip_last:
        if not dates_old:
            print("History is empty — nothing to strip.")
            return
        removed = dates_old[-1]
        dates_new = dates_old[:-1]
        keep_idx = list(range(len(dates_new)))
        print(f"  stripping last entry: {removed}")
    else:
        # Determine today in US Eastern (approx UTC-4)
        now_utc = datetime.now(timezone.utc)
        et_now = now_utc.astimezone(timezone(timedelta(hours=-4)))
        today_et = et_now.date()

        keep_idx = []
        dates_new = []
        for i, d in enumerate(dates_old):
            try:
                dt = date.fromisoformat(d)
            except Exception:
                print(f"  skipping unparseable date: {d}")
                continue
            if dt.weekday() >= 5:
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
