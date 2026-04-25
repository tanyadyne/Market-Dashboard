"""One-off script: overwrite the most recent date entry in
leaders_score_history.json with the current values from leaders.json.

Use case: when the RS scoring methodology has changed and you want the most
recent history entry to reflect the new logic (so the daily-rank list matches
the top of the history table). Only touches the LAST entry — older history is
preserved as-is.

Run after a fresh run of fetch_leaders.py has produced an updated leaders.json.
"""
import json
import os

HIST_PATH = "leaders_score_history.json"
LEADERS_PATH = "leaders.json"


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
                "s": [None] * (len(dates) - 1) + [r.get("rs")],
                "r": [None] * (len(dates) - 1) + [r.get("rk")],
                "wr": [None] * (len(dates) - 1) + [r.get("w_rk")],
                "tz": [None] * (len(dates) - 1) + [r.get("tz")],
            }
            new_tickers += 1
            continue

        rec = scores[tk]
        # Ensure every array exists and has the right length; pad if needed
        for key, src in (("s", "rs"), ("r", "rk"), ("wr", "w_rk"), ("tz", "tz")):
            arr = rec.get(key, [])
            # Pad with None if shorter than dates array
            while len(arr) < len(dates):
                arr.append(None)
            # Overwrite the last index
            arr[-1] = r.get(src)
            rec[key] = arr
        updated += 1

    # Trim entries for tickers no longer in leaders.json (data hygiene)
    removed = 0
    for tk in list(scores.keys()):
        if tk not in lookup:
            # Just leave the existing data alone but still pad if needed; don't add a fresh value
            rec = scores[tk]
            for key in ("s", "r", "wr", "tz"):
                arr = rec.get(key, [])
                while len(arr) < len(dates):
                    arr.append(None)
                rec[key] = arr

    hist["dates"] = dates
    hist["d"] = scores

    with open(HIST_PATH, "w") as f:
        json.dump(hist, f, separators=(",", ":"))

    print(f"Done. Updated {updated} existing tickers, added {new_tickers} new tickers.")
    print(f"Most recent history entry ({last_date}) now matches current leaders.json.")


if __name__ == "__main__":
    main()
