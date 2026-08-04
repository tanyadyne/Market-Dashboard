"""Helpers for immutable per-session weekly-rank universe sizes."""

WEEKLY_RANK_TOTALS_FIELD = "wr_total"


def _positive_integer(value):
    """Return a positive integer value, or None when it is not usable."""
    if isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not numeric.is_integer() or numeric < 1:
        return None
    return int(numeric)


def infer_ranked_total(scores, date_index):
    """Recover a legacy session's universe size only when its ranks are complete.

    Pre-wr_total history has no stored denominator. A complete rank sequence
    must contain each integer from 1 through its maximum, so its maximum is an
    exact historical universe size rather than a guess.
    """
    if not isinstance(scores, dict) or date_index < 0:
        return None

    ranks = set()
    for entry in scores.values():
        if not isinstance(entry, dict):
            continue
        weekly_ranks = entry.get("wr")
        if not isinstance(weekly_ranks, list) or date_index >= len(weekly_ranks):
            continue
        rank = _positive_integer(weekly_ranks[date_index])
        if rank is not None:
            ranks.add(rank)

    if not ranks:
        return None
    maximum = max(ranks)
    return maximum if len(ranks) == maximum else None


def aligned_ranked_totals(history, dates, scores):
    """Return one weekly-rank universe size per history date.

    Existing values remain authoritative. Missing legacy values are recovered
    only when the stored rank sequence proves the exact denominator.
    """
    date_count = len(dates) if isinstance(dates, list) else 0
    raw_totals = history.get(WEEKLY_RANK_TOTALS_FIELD, []) if isinstance(history, dict) else []
    totals = list(raw_totals) if isinstance(raw_totals, list) else []

    if len(totals) > date_count:
        totals = totals[-date_count:] if date_count else []
    elif len(totals) < date_count:
        totals.extend([None] * (date_count - len(totals)))

    for index, value in enumerate(totals):
        total = _positive_integer(value)
        totals[index] = total if total is not None else infer_ranked_total(scores, index)
    return totals


def set_latest_ranked_total(totals, date_count, ranked_total):
    """Align a total series to date_count and replace its latest value."""
    if date_count <= 0:
        return []

    values = list(totals) if isinstance(totals, list) else []
    if len(values) > date_count:
        values = values[-date_count:]
    elif len(values) < date_count:
        values.extend([None] * (date_count - len(values)))
    values[-1] = _positive_integer(ranked_total)
    return values
