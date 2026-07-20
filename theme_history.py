"""Helpers for aligned per-day theme snapshots in stock rank history."""

import json
import os


THEME_HISTORY_VERSION = 1


def load_theme_snapshot_index(path="data.json"):
    """Return theme name -> [name, weekly rank, status, universe count]."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return {}

    themes = payload.get("e") or []
    total = len(themes)
    snapshots = {}
    for theme in themes:
        name = theme.get("n")
        rank = theme.get("w_rk")
        if not name:
            continue
        status = None
        if rank is not None and total:
            percentile = rank / total
            status = "Leader" if percentile <= 0.25 else "Laggard" if percentile > 0.75 else "Mid-Range"
        snapshots[name] = [name, rank, status, total]
    return snapshots


def snapshot_for_stock(stock, snapshots):
    theme_name = stock.get("thm") or stock.get("th") or ""
    if not theme_name:
        return None
    return snapshots.get(theme_name, [theme_name, None, None, len(snapshots)])


def prepare_theme_history(history):
    """Discard pre-feature theme metadata once; legacy dates stay unsnapshotted."""
    if history.get("theme_history_version") == THEME_HISTORY_VERSION:
        return history
    for entry in (history.get("d") or {}).values():
        if isinstance(entry, dict):
            entry.pop("tm", None)
    history["theme_history_version"] = THEME_HISTORY_VERSION
    return history


def set_theme_snapshot(entry, date_count, snapshot):
    """Set latest aligned snapshot, padding legacy dates with nulls."""
    snapshots = entry.setdefault("tm", [])
    while len(snapshots) < date_count - 1:
        snapshots.append(None)
    if len(snapshots) == date_count:
        snapshots[-1] = snapshot
    else:
        snapshots.append(snapshot)
