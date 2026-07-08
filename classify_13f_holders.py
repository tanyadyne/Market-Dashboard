#!/usr/bin/env python3
"""
Classify aggregated 13F positions by holder inclusion rules.

This is step 4 of the ownership tile pipeline:
  - classify each unique manager using holder_classification_rules.json
  - annotate every aggregated manager/ticker position
  - write an included-only file for top-holder ranking
  - write a manager decision audit file for review and rule tuning
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_INPUT = Path("13f_holdings_2026_q2_aggregated.ndjson")
DEFAULT_RULES = Path("holder_classification_rules.json")
DEFAULT_CLASSIFIED_OUTPUT = Path("13f_holdings_2026_q2_classified.ndjson")
DEFAULT_INCLUDED_OUTPUT = Path("13f_holdings_2026_q2_included.ndjson")
DEFAULT_MANAGER_OUTPUT = Path("13f_holder_manager_classifications.json")


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


def write_ndjson(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")
            count += 1
    return count


def normalize_name(value: Any) -> str:
    text = str(value or "").upper()
    text = text.replace("&", " AND ")
    text = re.sub(r"\bL\s*/\s*P\b", " LP ", text)
    text = re.sub(r"\bL\s*/\s*L\s*/\s*C\b", " LLC ", text)
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    return f" {' '.join(text.split())} "


def list_rules(rules: dict[str, Any], key: str) -> list[str]:
    values = rules.get(key, [])
    if not isinstance(values, list):
        return []
    return [str(value) for value in values if str(value).strip()]


def find_match(normalized_name: str, patterns: list[str]) -> str | None:
    for pattern in patterns:
        normalized_pattern = normalize_name(pattern)
        if normalized_pattern.strip() and normalized_pattern in normalized_name:
            return pattern
    return None


def classify_manager(manager_name: str, rules: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_name(manager_name)

    checks = (
        ("include", "allowed_sovereign", "allowed_sovereign_contains"),
        ("include", "explicit_include", "explicit_include_contains"),
        ("exclude", "explicit_exclude", "explicit_exclude_contains"),
        ("exclude", "market_maker", "market_maker_contains"),
        ("exclude", "structural_exclude", "structural_exclude_contains"),
        ("include", "structural_include", "structural_include_contains"),
    )
    for status, rule, key in checks:
        match = find_match(normalized, list_rules(rules, key))
        if match:
            return {
                "holder_status": status,
                "holder_included": status == "include",
                "classification_rule": rule,
                "classification_match": match,
                "classification_reason": f"{status} via {rule}: {match}",
            }

    return {
        "holder_status": "review",
        "holder_included": False,
        "classification_rule": "review",
        "classification_match": None,
        "classification_reason": "No deterministic inclusion or exclusion rule matched.",
    }


def classify_rows(
    rows: list[dict[str, Any]],
    rules: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    manager_names: dict[str, str] = {}
    for row in rows:
        manager_cik = str(row.get("manager_cik") or "").strip()
        manager_name = str(row.get("manager_name") or "").strip()
        if manager_cik and manager_cik not in manager_names:
            manager_names[manager_cik] = manager_name or manager_cik

    decisions: dict[str, dict[str, Any]] = {}
    for manager_cik, manager_name in sorted(manager_names.items(), key=lambda item: item[1].upper()):
        decisions[manager_cik] = {
            "manager_cik": manager_cik,
            "manager_name": manager_name,
            **classify_manager(manager_name, rules),
        }

    classified_rows: list[dict[str, Any]] = []
    for row in rows:
        manager_cik = str(row.get("manager_cik") or "").strip()
        decision = decisions.get(manager_cik)
        if decision is None:
            decision = {
                "holder_status": "review",
                "holder_included": False,
                "classification_rule": "review",
                "classification_match": None,
                "classification_reason": "Missing manager CIK.",
            }
        classified_rows.append(
            {
                **row,
                "holder_status": decision["holder_status"],
                "holder_included": decision["holder_included"],
                "classification_rule": decision["classification_rule"],
                "classification_match": decision["classification_match"],
                "classification_reason": decision["classification_reason"],
            }
        )
    return classified_rows, decisions


def summarize(
    input_path: Path,
    rules_path: Path,
    classified_output: Path,
    included_output: Path,
    manager_output: Path,
    rows: list[dict[str, Any]],
    decisions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    manager_status_counts = Counter(item["holder_status"] for item in decisions.values())
    row_status_counts = Counter(row["holder_status"] for row in rows)
    rule_counts = Counter(item["classification_rule"] for item in decisions.values())
    report_periods = sorted({str(row.get("report_period")) for row in rows if row.get("report_period")})
    included_rows = [row for row in rows if row.get("holder_included")]
    included_managers = {row["manager_cik"] for row in included_rows}
    review_sample = [
        item
        for item in decisions.values()
        if item["holder_status"] == "review"
    ][:50]
    exclude_sample = [
        item
        for item in decisions.values()
        if item["holder_status"] == "exclude"
    ][:50]
    include_sample = [
        item
        for item in decisions.values()
        if item["holder_status"] == "include"
    ][:50]

    return {
        "updated": now_iso(),
        "input": str(input_path),
        "rules": str(rules_path),
        "classified_output": str(classified_output),
        "included_output": str(included_output),
        "manager_output": str(manager_output),
        "positions_total": len(rows),
        "positions_included": len(included_rows),
        "positions_excluded_or_review": len(rows) - len(included_rows),
        "managers_total": len(decisions),
        "managers_included": len(included_managers),
        "manager_status_counts": dict(sorted(manager_status_counts.items())),
        "position_status_counts": dict(sorted(row_status_counts.items())),
        "manager_rule_counts": dict(sorted(rule_counts.items())),
        "unique_tickers_total": len({row.get("ticker") for row in rows}),
        "unique_tickers_included": len({row.get("ticker") for row in included_rows}),
        "report_periods": report_periods,
        "review_sample": review_sample,
        "include_sample": include_sample,
        "exclude_sample": exclude_sample,
        "note": "Only holder_status=include rows should feed Step 5 top-holder ranking.",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES)
    parser.add_argument("--classified-output", type=Path, default=DEFAULT_CLASSIFIED_OUTPUT)
    parser.add_argument("--included-output", type=Path, default=DEFAULT_INCLUDED_OUTPUT)
    parser.add_argument("--manager-output", type=Path, default=DEFAULT_MANAGER_OUTPUT)
    parser.add_argument("--summary", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Missing aggregated 13F input: {args.input}")
    if not args.rules.exists():
        raise FileNotFoundError(f"Missing holder classification rules: {args.rules}")

    rules = read_json(args.rules)
    rows = list(read_ndjson(args.input))
    classified_rows, decisions = classify_rows(rows, rules)
    included_rows = [row for row in classified_rows if row["holder_included"]]

    write_ndjson(args.classified_output, classified_rows)
    write_ndjson(args.included_output, included_rows)
    write_json(
        args.manager_output,
        {
            "updated": now_iso(),
            "rules": str(args.rules),
            "managers": sorted(decisions.values(), key=lambda item: item["manager_name"].upper()),
        },
    )

    summary_path = args.summary or args.classified_output.with_suffix(".summary.json")
    summary = summarize(
        args.input,
        args.rules,
        args.classified_output,
        args.included_output,
        args.manager_output,
        classified_rows,
        decisions,
    )
    summary["summary"] = str(summary_path)
    write_json(summary_path, summary)

    print(
        "positions={positions_total} included={positions_included} "
        "managers={managers_total} included_managers={managers_included}".format(**summary)
    )
    print(f"summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
