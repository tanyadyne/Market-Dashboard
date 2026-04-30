"""One-off script: overwrite the most recent date entry in
leaders_score_history.json with the current values from leaders.json.

Use case: when the RS scoring methodology has changed and you want the most
recent history entry to reflect the new logic. Only touches the LAST entry —
older history is preserved as-is.

The history file now stores ONLY the weekly rank (`wr`). Older daily fields
(`s`, `r`, `tz`) are stripped automatically by fetch_leaders.py on each run.

Run after a fresh run of fetch_leaders.py has produced an updated leaders.json.
"""
import json
import os

HIST_PATH = "leaders_score_history.json"
LEADERS_PATH = "leaders.json"

# Daily-RS fields removed from the dashboard. Strip on every resnap.
OBSOLETE_HIST_KEYS = {"s", "r", "tz"}


def main():
    if not os.path.exists(HIST_PATH):
        print(f"No {HIST_PATH} found.")
        return
    if not os.path.exists(LEADERS_PATH):
        print(f"No {LEADERS_PATH} found.")
        return

    with open(HIST_PATH) as f:
        hist = json.load(f)
    with open(LEADERS_PATH) as f:
        leaders = json.load(f)

    dates = hist.get("dates", [])
    scores = hist.get("d", {})

    if not dates:
        print("History is empty — nothing to overwrite.")
        return

    last_date = dates[-1]
    print(f"Overwriting history entry for {last_date} with current leaders.json values...")

    # Strip obsolete daily fields from existing history entries
    stripped = 0
    for tk, entry in scores.items():
        if isinstance(entry, dict):
            for k in OBSOLETE_HIST_KEYS:
                if k in entry:
                    del entry[k]
                    stripped += 1
    if stripped:
        print(f"  Cleaned up {stripped} obsolete daily-RS field arrays.")

    # Build lookup from current leaders.json (key 'e' contains the stock list)
    leaders_list = leaders.get("e", [])
    if not leaders_list:
        print("Could not find stocks in leaders.json (expected key 'e').")
        return

    lookup = {r["t"]: r for r in leaders_list if "t" in r}
    print(f"Found {len(lookup)} stocks in leaders.json")

    updated = 0
    new_tickers = 0
    for tk, r in lookup.items():
        if tk not in scores:
            # New ticker that wasn't in history before — add with Nones for prior dates
            scores[tk] = {
                "wr": [None] * (len(dates) - 1) + [r.get("w_rk")],
            }
            new_tickers += 1
            continue

        rec = scores[tk]
        wr_arr = rec.get("wr", [])
        # Pad with None if shorter than dates array
        while len(wr_arr) < len(dates):
            wr_arr.append(None)
        # Overwrite the last index
        wr_arr[-1] = r.get("w_rk")
        rec["wr"] = wr_arr
        updated += 1

    # Pad tickers no longer in leaders.json so their arrays stay aligned
    for tk in list(scores.keys()):
        if tk not in lookup:
            rec = scores[tk]
            wr_arr = rec.get("wr", [])
            while len(wr_arr) < len(dates):
                wr_arr.append(None)
            rec["wr"] = wr_arr

    hist["dates"] = dates
    hist["d"] = scores

    with open(HIST_PATH, "w") as f:
        json.dump(hist, f, separators=(",", ":"))

    print(f"Done. Updated {updated} existing tickers, added {new_tickers} new tickers.")
    print(f"Most recent history entry ({last_date}) now matches current leaders.json.")


if __name__ == "__main__":
    main()
