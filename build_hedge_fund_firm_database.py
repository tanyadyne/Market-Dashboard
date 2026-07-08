#!/usr/bin/env python3
"""
Build the hedge-fund firm mini-database used by holder classification.

The firm include list is sourced from hedge-fund firm directories. Product or
fund-name files are retained for audit context but are not bulk-added as 13F
manager include rules.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_RULES = Path("holder_classification_rules.json")
DEFAULT_DATABASE = Path("hedge_fund_firm_database.json")


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def normalize_name(value: Any) -> str:
    text = str(value or "").upper()
    text = text.replace("&", " AND ")
    text = re.sub(r"\bL\s*/\s*P\b", " LP ", text)
    text = re.sub(r"\bL\s*/\s*L\s*/\s*C\b", " LLC ", text)
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    return " ".join(text.split())


def normalized_contains(name: str, pattern: str) -> bool:
    needle = normalize_name(pattern)
    if not needle:
        return False
    return f" {needle} " in f" {normalize_name(name)} "


def list_rules(rules: dict[str, Any], key: str) -> list[str]:
    values = rules.get(key, [])
    if not isinstance(values, list):
        return []
    return [str(value).strip() for value in values if str(value).strip()]


def find_match(name: str, patterns: list[str]) -> str | None:
    for pattern in patterns:
        if normalized_contains(name, pattern):
            return pattern
    return None


def clean_name(value: Any) -> str | None:
    name = str(value or "").strip()
    if not name or name.lower() == "nan":
        return None
    return re.sub(r"\s+", " ", name)


def read_hedge_funds_list(path: Path) -> list[dict[str, Any]]:
    df = pd.read_csv(path)
    rows: list[dict[str, Any]] = []
    for idx, row in df.iterrows():
        name = clean_name(row.get("Firm_Name_name"))
        if not name:
            continue
        rows.append(
            {
                "name": name,
                "source": path.name,
                "source_type": "firm_list",
                "source_row": int(idx) + 2,
                "city": clean_name(row.get("Firm_Name_City")),
                "country": clean_name(row.get("Firm_Name_Country")),
                "aum_millions": clean_name(row.get("Firm_Name_AUM_in_millions")),
                "strategies": clean_name(row.get("Firm_Name_Strategies")),
            }
        )
    return rows


def read_wsp_top_hedge_funds(path: Path) -> list[dict[str, Any]]:
    df = pd.read_excel(path, sheet_name="Hedge Fund List", header=2)
    rows: list[dict[str, Any]] = []
    for idx, row in df.iterrows():
        name = clean_name(row.get("Firm Name"))
        if not name:
            continue
        rows.append(
            {
                "name": name,
                "source": path.name,
                "source_type": "firm_list",
                "source_row": int(idx) + 4,
                "rank": int(row["Rank"]) if pd.notna(row.get("Rank")) else None,
                "city": clean_name(row.get("City")),
                "country": clean_name(row.get("Country")),
                "aum_millions": float(row["AUM ($mm)"]) if pd.notna(row.get("AUM ($mm)")) else None,
            }
        )
    return rows


def read_fund_names(path: Path) -> list[dict[str, Any]]:
    df = pd.read_csv(path, header=None, names=["name"])
    rows: list[dict[str, Any]] = []
    for idx, row in df.iterrows():
        name = clean_name(row.get("name"))
        if not name:
            continue
        rows.append(
            {
                "name": name,
                "source": path.name,
                "source_type": "fund_or_product_name",
                "source_row": int(idx) + 1,
            }
        )
    return rows


def exclusion_patterns(rules: dict[str, Any]) -> dict[str, list[str]]:
    explicit = [pattern for pattern in list_rules(rules, "explicit_exclude_contains") if normalize_name(pattern) != "CAPITAL GROUP"]
    explicit.extend(
        [
            "CAPITAL GROUP INVESTMENT MANAGEMENT",
            "CAPITAL GROUP PRIVATE CLIENT SERVICES",
            "CAPITAL INTERNATIONAL INC",
            "CAPITAL INTERNATIONAL INVESTORS",
            "CAPITAL INTERNATIONAL LTD",
            "CAPITAL INTERNATIONAL SARL",
            "CAPITAL RESEARCH GLOBAL INVESTORS",
        ]
    )
    return {
        "explicit_exclude": sorted(set(explicit), key=lambda value: normalize_name(value)),
        "market_maker": list_rules(rules, "market_maker_contains"),
        "structural_exclude": list_rules(rules, "structural_exclude_contains"),
    }


def classify_candidate(name: str, exclusions: dict[str, list[str]]) -> dict[str, Any]:
    for rule, patterns in exclusions.items():
        match = find_match(name, patterns)
        if match:
            return {
                "included": False,
                "exclusion_rule": rule,
                "exclusion_match": match,
            }
    return {
        "included": True,
        "exclusion_rule": None,
        "exclusion_match": None,
    }


def merge_firm_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for row in rows:
        normalized = normalize_name(row["name"])
        if not normalized:
            continue
        item = merged.setdefault(
            normalized,
            {
                "name": row["name"],
                "normalized_name": normalized,
                "sources": [],
                "source_rows": [],
            },
        )
        if row["source"] not in item["sources"]:
            item["sources"].append(row["source"])
        item["source_rows"].append(
            {
                key: value
                for key, value in row.items()
                if key not in {"name"} and value is not None
            }
        )
    return sorted(merged.values(), key=lambda item: item["normalized_name"])


def build_database(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    rules = read_json(args.rules)
    exclusions = exclusion_patterns(rules)

    firm_rows: list[dict[str, Any]] = []
    firm_rows.extend(read_hedge_funds_list(args.hedge_funds_list))
    firm_rows.extend(read_wsp_top_hedge_funds(args.wsp_top_hedge_funds))

    supplemental_rows = read_fund_names(args.fund_names)
    firms = merge_firm_rows(firm_rows)

    included_firms: list[dict[str, Any]] = []
    excluded_candidates: list[dict[str, Any]] = []
    for firm in firms:
        decision = classify_candidate(firm["name"], exclusions)
        payload = {**firm, **decision}
        if decision["included"]:
            included_firms.append(payload)
        else:
            excluded_candidates.append(payload)

    supplemental = []
    for row in supplemental_rows:
        decision = classify_candidate(row["name"], exclusions)
        supplemental.append(
            {
                "name": row["name"],
                "normalized_name": normalize_name(row["name"]),
                "source": row["source"],
                "source_row": row["source_row"],
                **decision,
            }
        )

    updated = now_iso()
    database = {
        "meta": {
            "description": "Mini-database of hedge-fund firm names used as explicit include patterns for 13F holder classification.",
            "updated": updated,
            "source_files": [
                args.hedge_funds_list.name,
                args.wsp_top_hedge_funds.name,
                args.fund_names.name,
            ],
            "firm_source_unique_count": len(firms),
            "included_firm_count": len(included_firms),
            "excluded_firm_candidate_count": len(excluded_candidates),
            "supplemental_fund_or_product_name_count": len(supplemental),
            "notes": [
                "Only firm-list source rows are bulk-added to holder_classification_rules.json.",
                "fund_names.csv is retained as supplemental audit data because it contains fund/product names rather than clean 13F manager firm names.",
                "Explicit ETF, index-fund, bank, market-maker, quant-exclusion, and structural exclusions are applied before inclusion.",
                "The broad CAPITAL GROUP exclusion is replaced with specific Capital Group/Capital Research/Capital International sponsor patterns to avoid excluding unrelated hedge funds named ... Capital Group.",
            ],
        },
        "included_firms": included_firms,
        "excluded_firm_candidates": excluded_candidates,
        "supplemental_fund_or_product_names": supplemental,
    }

    existing_includes = list_rules(rules, "explicit_include_contains")
    include_patterns = sorted(
        {*(pattern.strip() for pattern in existing_includes if pattern.strip()), *(firm["name"] for firm in included_firms)},
        key=lambda value: normalize_name(value),
    )

    updated_rules = dict(rules)
    updated_meta = dict(updated_rules.get("meta", {}))
    notes = list(updated_meta.get("notes", []))
    new_note = f"Hedge-fund explicit include database updated {updated}; included_firm_count={len(included_firms)}."
    if new_note not in notes:
        notes.append(new_note)
    updated_meta["notes"] = notes
    updated_rules["meta"] = updated_meta
    updated_rules["explicit_include_contains"] = include_patterns
    updated_rules["explicit_exclude_contains"] = exclusions["explicit_exclude"]

    return database, updated_rules


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hedge-funds-list", type=Path, required=True)
    parser.add_argument("--fund-names", type=Path, required=True)
    parser.add_argument("--wsp-top-hedge-funds", type=Path, required=True)
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--write-rules", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    database, updated_rules = build_database(args)
    write_json(args.database, database)
    if args.write_rules:
        write_json(args.rules, updated_rules)
    print(f"database={args.database}")
    print(f"included_firm_count={database['meta']['included_firm_count']}")
    print(f"excluded_firm_candidate_count={database['meta']['excluded_firm_candidate_count']}")
    print(f"rules_updated={args.write_rules}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
