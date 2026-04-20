"""Scrape NAAIM Exposure Index table and save as naaim.json.

Runs weekly. The NAAIM site updates every Thursday.

Output schema:
{
  "updated": "2026-04-20T07:00:00Z",
  "latest": 79.49,
  "q1_avg": 82.00,
  "rows": [
    {"date": "04/15/2026", "mean": 79.49, "bearish": -200, "q1": 50.00, "q2": 90.00, "q3": 100.00, "bullish": 200, "deviation": 68.87},
    ...
  ]
}
"""

import json
import re
import sys
from datetime import datetime, timezone

try:
    import requests
except ImportError:
    print("ERROR: pip install requests", file=sys.stderr)
    sys.exit(1)


URL = "https://naaim.org/programs/naaim-exposure-index/"
OUTPUT = "naaim.json"
MAX_ROWS = 15


def scrape():
    print(f"Fetching {URL}...")
    headers = {
        "User-Agent": "Mozilla/5.0 (Market-Dashboard NAAIM Scraper)"
    }
    resp = requests.get(URL, headers=headers, timeout=30)
    resp.raise_for_status()
    html = resp.text

    # Pull out all tables, find the one with "NAAIM Number Mean"
    tables = re.findall(r"<table[^>]*>(.*?)</table>", html, re.DOTALL | re.IGNORECASE)
    target = None
    for t in tables:
        if "NAAIM Number" in t or "Bearish" in t:
            target = t
            break
    if target is None:
        print("ERROR: NAAIM table not found in page", file=sys.stderr)
        sys.exit(1)

    # Extract rows
    rows_html = re.findall(r"<tr[^>]*>(.*?)</tr>", target, re.DOTALL | re.IGNORECASE)
    parsed = []
    for row in rows_html:
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.DOTALL | re.IGNORECASE)
        # Strip HTML tags and whitespace
        cells = [re.sub(r"<[^>]+>", "", c).strip() for c in cells]
        if len(cells) != 8:
            continue
        # Skip header row
        if "Date" in cells[0] or "NAAIM" in cells[1]:
            continue
        # Skip empty
        if not cells[0]:
            continue
        # Parse numeric cells
        try:
            def parse_num(s):
                s = s.replace(",", "").strip()
                return float(s) if s else None
            parsed.append({
                "date":      cells[0],
                "mean":      parse_num(cells[1]),
                "bearish":   parse_num(cells[2]),
                "q1":        parse_num(cells[3]),
                "q2":        parse_num(cells[4]),
                "q3":        parse_num(cells[5]),
                "bullish":   parse_num(cells[6]),
                "deviation": parse_num(cells[7]),
            })
        except Exception as e:
            print(f"  skip malformed row: {cells} ({e})", file=sys.stderr)
            continue

    # Dedupe (the source sometimes has duplicate rows — drop exact date duplicates)
    seen = set()
    uniq = []
    for r in parsed:
        key = r["date"]
        if key in seen:
            continue
        seen.add(key)
        uniq.append(r)

    rows = uniq[:MAX_ROWS]
    if not rows:
        print("ERROR: no rows parsed", file=sys.stderr)
        sys.exit(1)

    # Latest value / Q1 average from page text
    latest = rows[0]["mean"]
    # The page text looks like: "Last Quarter Average (Q1) 82.00"
    # Skip past the "(Q1)" token before grabbing the number.
    q1_match = re.search(
        r"Last\s+Quarter\s+Average.*?\(Q1\)\s*</[^>]+>\s*<[^>]+>\s*([\-0-9.]+)",
        html, re.IGNORECASE | re.DOTALL)
    if not q1_match:
        # Fallback: generic "Last Quarter Average [anything] 82.00"
        q1_match = re.search(
            r"Last\s+Quarter\s+Average[^\n]*?\)\s*[^0-9\-]*([\-0-9.]+)",
            html, re.IGNORECASE)
    q1_avg = float(q1_match.group(1)) if q1_match else None

    out = {
        "updated": datetime.now(timezone.utc).isoformat(),
        "latest": latest,
        "q1_avg": q1_avg,
        "rows": rows,
    }

    with open(OUTPUT, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"✓ Saved {OUTPUT} ({len(rows)} rows, latest {latest})")


if __name__ == "__main__":
    scrape()
