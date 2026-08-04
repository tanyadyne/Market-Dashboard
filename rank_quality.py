"""Data-quality and admission rules for the stock RS ranking.

Keep these rules independent from market-data clients so they can be tested without
network access.  A suspect price series is retained for profile visibility but is
never allowed into the cross-sectional RS calculation.
"""

import math
import statistics


NEW_ENTRANT_PROBATION_SESSIONS = 5
QUALITY_LOOKBACK_BARS = 25

# These are deliberately review thresholds, not trading limits.  A genuine move
# beyond them is rare and should be manually confirmed before it can influence
# every other stock's percentile rank.
MAX_ADJUSTED_DAILY_RETURN = 3.00       # 300%
MAX_ADJUSTED_WEEKLY_RETURN = 4.00      # 400%
MAX_ADJUSTED_MONTHLY_RETURN = 6.00     # 600%
OUTLIER_VOLATILITY_MULTIPLE = 30.0
OUTLIER_MIN_RETURN = 0.75              # 75%
LIVE_PRICE_SCALE_RETURN = 3.00         # 300%
LIVE_PRICE_SCALE_ATR_MULTIPLE = 25.0


def _positive_number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _simple_return(current, previous):
    current = _positive_number(current)
    previous = _positive_number(previous)
    if current is None or previous is None:
        return None
    return current / previous - 1.0


def technical_price_from_quote(raw_price, price_scale=None):
    """Put a live/raw quote onto the adjusted-price scale used by RS math."""
    raw_price = _positive_number(raw_price)
    if raw_price is None:
        return None
    scale = _positive_number(price_scale)
    return raw_price * (scale if scale is not None else 1.0)


def _trailing_abs_returns(closes, end, length=60):
    start = max(1, end - length)
    values = []
    for index in range(start, end):
        change = _simple_return(closes[index], closes[index - 1])
        if change is not None:
            values.append(abs(change))
    return values


def adjusted_price_quality_flags(adjusted_closes, raw_closes=None):
    """Return stable reason codes for implausible adjusted-price moves.

    A correctly adjusted split has a large raw move but a normal adjusted move.
    Therefore this deliberately judges the adjusted series.  It holds malformed
    or unadjusted corporate-action jumps without pretending to repair an event
    that has not been independently confirmed.
    """
    if adjusted_closes is None:
        return ["missing_adjusted_history"]

    try:
        count = len(adjusted_closes)
    except TypeError:
        return ["missing_adjusted_history"]
    if count < 2:
        return ["insufficient_adjusted_history"]

    flags = []
    start = max(1, count - QUALITY_LOOKBACK_BARS)
    for index in range(start, count):
        change = _simple_return(adjusted_closes[index], adjusted_closes[index - 1])
        if change is None:
            flags.append("invalid_adjusted_close")
            break
        absolute_change = abs(change)
        if absolute_change >= MAX_ADJUSTED_DAILY_RETURN:
            flags.append("extreme_adjusted_daily_move")
            break
        prior_moves = _trailing_abs_returns(adjusted_closes, index)
        baseline = statistics.median(prior_moves) if prior_moves else None
        if (
            baseline is not None
            and baseline > 0
            and absolute_change >= OUTLIER_MIN_RETURN
            and absolute_change / baseline >= OUTLIER_VOLATILITY_MULTIPLE
        ):
            flags.append("adjusted_volatility_outlier")
            break

    period_limits = ((5, MAX_ADJUSTED_WEEKLY_RETURN, "extreme_adjusted_weekly_move"),
                     (21, MAX_ADJUSTED_MONTHLY_RETURN, "extreme_adjusted_monthly_move"))
    for bars, limit, code in period_limits:
        if count <= bars:
            continue
        change = _simple_return(adjusted_closes[-1], adjusted_closes[-1 - bars])
        if change is not None and abs(change) >= limit:
            flags.append(code)

    # A split is only safe when adjustment removed the raw discontinuity.  This
    # makes the failure mode explicit in diagnostics and tests.
    if raw_closes is not None:
        try:
            raw_count = min(count, len(raw_closes))
        except TypeError:
            raw_count = 0
        for index in range(max(1, raw_count - QUALITY_LOOKBACK_BARS), raw_count):
            raw_change = _simple_return(raw_closes[index], raw_closes[index - 1])
            adjusted_change = _simple_return(adjusted_closes[index], adjusted_closes[index - 1])
            if (
                raw_change is not None
                and adjusted_change is not None
                and abs(raw_change) >= MAX_ADJUSTED_DAILY_RETURN
                and abs(adjusted_change) >= MAX_ADJUSTED_DAILY_RETURN
            ):
                flags.append("unrepaired_price_scale_jump")
                break

    # Preserve insertion order while removing duplicated reasons.
    return list(dict.fromkeys(flags))


def live_price_quality_flags(live_price, previous_price, atr_dollars=None):
    """Detect a bad intraday quote before it can rerank the full universe."""
    change = _simple_return(live_price, previous_price)
    if change is None:
        return ["invalid_live_price"]
    if abs(change) < LIVE_PRICE_SCALE_RETURN:
        return []
    atr = _positive_number(atr_dollars)
    absolute_dollars = abs(float(live_price) - float(previous_price))
    if atr is None or absolute_dollars / atr >= LIVE_PRICE_SCALE_ATR_MULTIPLE:
        return ["live_price_scale_jump"]
    return []


def prior_observation_sessions(history_entry):
    """Count prior completed EOD appearances for one stock.

    ``seen`` is an aligned history field introduced with the admission gate.
    Older histories have only ranked observations; those stocks are established
    already, so their non-null rank count is a safe migration fallback.
    """
    if not isinstance(history_entry, dict):
        return 0
    seen = history_entry.get("seen")
    seen_count = 0
    if isinstance(seen, list):
        seen_count = sum(1 for value in seen if value is True)
    ranks = history_entry.get("wr")
    if isinstance(ranks, list):
        # During rollout, legacy ranked rows gain `seen` only on their next
        # refresh.  Retain their existing maturity instead of restarting them.
        return max(seen_count, sum(1 for value in ranks if value is not None))
    return seen_count


def append_observation(history_entry, history_length, observed=True):
    """Write one aligned EOD observation, safely handling same-day reruns."""
    if not isinstance(history_entry, dict) or history_length <= 0:
        return
    seen = history_entry.setdefault("seen", [])
    if not isinstance(seen, list):
        seen = []
        history_entry["seen"] = seen
    while len(seen) < history_length - 1:
        seen.append(False)
    if len(seen) == history_length:
        seen[-1] = bool(observed)
    else:
        seen.append(bool(observed))


def probation_complete(prior_sessions):
    try:
        return int(prior_sessions) >= NEW_ENTRANT_PROBATION_SESSIONS - 1
    except (TypeError, ValueError):
        return False


def trend_confirmed(price, sma50, sma200):
    """Require a confirmed Stage-2-like trend for main RS-rank eligibility."""
    price = _positive_number(price)
    sma50 = _positive_number(sma50)
    sma200 = _positive_number(sma200)
    return bool(price and sma50 and sma200 and price >= sma50 and price >= sma200 and sma50 >= sma200)


def rank_hold_code(quality_flags, prior_sessions, price, sma50, sma200):
    """Return public admission status, or ``None`` for a rankable stock."""
    if quality_flags:
        return "data"
    if not probation_complete(prior_sessions):
        return "probation"
    if not trend_confirmed(price, sma50, sma200):
        return "trend"
    return None
