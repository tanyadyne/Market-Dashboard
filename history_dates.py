"""Choose safe trading-session dates for rank-history snapshots."""

from collections import Counter
from datetime import date


def _to_iso_date(value):
    """Return a YYYY-MM-DD string for a bar timestamp, or ``None``."""
    if value is None:
        return None
    try:
        if hasattr(value, "date"):
            value = value.date()
        if isinstance(value, date):
            return value.isoformat()
        return date.fromisoformat(str(value)[:10]).isoformat()
    except (TypeError, ValueError):
        return None


def select_history_session_date(results, daily_data):
    """Choose the consensus most-recent daily-bar date across result tickers.

    One ticker can be stale while the rest of the universe has the current bar.
    Selecting the most common date prevents that single stale ticker from
    relabeling an entire daily snapshot. A tie goes to the later date.
    """
    counts = Counter()
    for result in results or []:
        ticker = result.get("t") if isinstance(result, dict) else None
        frame = daily_data.get(ticker) if ticker and isinstance(daily_data, dict) else None
        if frame is None:
            continue
        try:
            if len(frame) == 0:
                continue
            session_date = _to_iso_date(frame.index[-1])
        except (AttributeError, IndexError, TypeError):
            continue
        if session_date:
            counts[session_date] += 1

    if not counts:
        return None
    return max(counts, key=lambda value: (counts[value], value))


def latest_history_session_date(dates):
    """Return the latest valid date already recorded in history."""
    valid_dates = [_to_iso_date(value) for value in dates or []]
    valid_dates = [value for value in valid_dates if value]
    return max(valid_dates) if valid_dates else None
