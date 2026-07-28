"""Canonical VCP coil definitions shared by stock and ETF pipelines.

Definition 1 is the dashboard's existing normalized tightness score.
Definition 2 mirrors the attached Pine Script's ``coil_detected`` output when
``Coil Sensitivity`` is ``Balanced``:

    balanced_coil or strict_coil

Definition 2 intentionally does not include the Pine Script's stateful box,
breakout, or inside-box outputs. Those are separate signals in the source
indicator.
"""

import math

import numpy as np


def set_aligned_vcp_history_value(entry, key, date_count, value):
    """Set today's VCP value while keeping its series aligned to history dates."""
    if date_count <= 0:
        return
    values = entry.get(key)
    if not isinstance(values, list):
        values = []
        entry[key] = values
    if len(values) > date_count:
        del values[date_count:]
    while len(values) < date_count:
        values.append(None)
    values[date_count - 1] = value


def compute_vcp_coil_definition_1_score(
    closes,
    highs,
    lows,
    length=10,
    adr_len=21,
    baseline=100,
):
    """Return Definition 1 tightness score; ``<= 10`` is a VCP coil."""
    try:
        closes = np.asarray(closes, dtype=float)
        highs = np.asarray(highs, dtype=float)
        lows = np.asarray(lows, dtype=float)
    except Exception:
        return None
    n = len(closes)
    if n < max(length, adr_len) or len(highs) != n or len(lows) != n:
        return None

    with np.errstate(divide="ignore", invalid="ignore"):
        bar_range_pct = np.where(
            lows != 0,
            ((highs - lows) / lows) * 100,
            np.nan,
        )

    combined = np.full(n, np.nan)
    for i in range(n):
        if i + 1 < max(length, adr_len):
            continue
        adr_window = bar_range_pct[i - adr_len + 1:i + 1]
        adr_pct = float(np.nanmean(adr_window))
        if not np.isfinite(adr_pct):
            continue
        safe_adr = 0.0001 if adr_pct == 0 else adr_pct

        close_window = closes[i - length + 1:i + 1]
        high_window = highs[i - length + 1:i + 1]
        low_window = lows[i - length + 1:i + 1]
        if not (
            np.isfinite(close_window).any()
            and np.isfinite(high_window).any()
            and np.isfinite(low_window).any()
        ):
            continue
        hi_c = float(np.nanmax(close_window))
        lo_c = float(np.nanmin(close_window))
        hi_p = float(np.nanmax(high_window))
        lo_p = float(np.nanmin(low_window))
        if lo_c <= 0 or lo_p <= 0:
            continue

        close_spread_pct = ((hi_c - lo_c) / lo_c) * 100
        price_spread_pct = ((hi_p - lo_p) / lo_p) * 100
        combined[i] = (
            (close_spread_pct / safe_adr)
            + (price_spread_pct / safe_adr)
        ) / 2

    current = combined[-1]
    if not np.isfinite(current):
        return None
    window = combined[max(0, n - baseline):]
    window = window[np.isfinite(window)]
    if len(window) == 0:
        return None
    lowest_ratio = float(np.min(window))
    highest_ratio = float(np.max(window))
    range_ratio = highest_ratio - lowest_ratio
    if range_ratio == 0:
        return 0.0
    return ((float(current) - lowest_ratio) / range_ratio) * 100


def compute_vcp_coil_definition_1(closes, highs, lows):
    """Return dashboard Definition 1 as ``True``, ``False``, or ``None``."""
    score = compute_vcp_coil_definition_1_score(closes, highs, lows)
    return None if score is None else bool(score <= 10)


def _pine_ema_endpoint(values, length):
    """Match Pine ``ta.ema`` endpoint semantics (first-value seed)."""
    alpha = 2.0 / (length + 1)
    ema = float(values[0])
    for value in values[1:]:
        ema = alpha * float(value) + (1.0 - alpha) * ema
    return ema


def _pine_rma_series(values, length):
    """Match Pine ``ta.rma`` series semantics used by ``ta.atr``."""
    values = np.asarray(values, dtype=float)
    out = np.full(len(values), np.nan)
    if len(values) < length:
        return out
    out[length - 1] = float(np.mean(values[:length]))
    alpha = 1.0 / length
    for i in range(length, len(values)):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return out


def _pine_atr_series(highs, lows, closes, length):
    """Match Pine ``ta.atr(length)`` using true range and Wilder RMA."""
    true_ranges = np.empty(len(closes), dtype=float)
    true_ranges[0] = highs[0] - lows[0]
    for i in range(1, len(closes)):
        true_ranges[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
    return _pine_rma_series(true_ranges, length)


def _pine_wma_endpoint(values, length):
    """Match Pine ``ta.wma`` with greatest weight on current value."""
    window = np.asarray(values[-length:], dtype=float)
    weights = np.arange(1, length + 1, dtype=float)
    return float(np.dot(window, weights) / np.sum(weights))


def _pine_hma_endpoint(values, length):
    """Match the source script's explicit Hull MA expression."""
    half_length = length // 2
    root_length = math.floor(math.sqrt(length))
    transformed = []
    for end in range(len(values) - root_length + 1, len(values) + 1):
        prefix = values[:end]
        transformed.append(
            2 * _pine_wma_endpoint(prefix, half_length)
            - _pine_wma_endpoint(prefix, length)
        )
    return _pine_wma_endpoint(transformed, root_length)


def compute_vcp_coil_definition_2(opens, highs, lows, closes):
    """Return Pine Balanced ``coil_detected`` as ``True``, ``False``, or ``None``.

    This is a direct daily-bar translation of the supplied Pine Script. Under
    Balanced sensitivity, its ``coil_detected`` variable is:

        balanced_coil or strict_coil
    """
    try:
        opens = np.asarray(opens, dtype=float)
        highs = np.asarray(highs, dtype=float)
        lows = np.asarray(lows, dtype=float)
        closes = np.asarray(closes, dtype=float)
    except Exception:
        return None

    n = len(closes)
    if (
        n < 63
        or len(opens) != n
        or len(highs) != n
        or len(lows) != n
        or not all(
            np.isfinite(values).all()
            for values in (opens, highs, lows, closes)
        )
    ):
        return None

    atr3 = _pine_atr_series(highs, lows, closes, 3)
    atr5 = _pine_atr_series(highs, lows, closes, 5)
    atr8 = _pine_atr_series(highs, lows, closes, 8)
    atr14 = _pine_atr_series(highs, lows, closes, 14)

    atr = float(atr14[-1])
    atr_avg = float(np.mean(atr14[-50:]))
    if not np.isfinite(atr) or not np.isfinite(atr_avg) or atr <= 0:
        return None

    body_range = abs(closes[-1] - opens[-1])
    candle_range = abs(highs[-1] - lows[-1])
    atr_contracted_strict = (
        body_range < atr_avg * 0.6
        and candle_range < atr_avg * 0.85
    )
    atr_contracted_balanced = candle_range < atr_avg * 0.8

    ema5 = _pine_ema_endpoint(closes, 5)
    ema9 = _pine_ema_endpoint(closes, 9)
    ema21 = _pine_ema_endpoint(closes, 21)
    sma50 = float(np.mean(closes[-50:]))
    sma200 = float(np.mean(closes[-200:])) if n >= 200 else None
    hma40 = _pine_hma_endpoint(lows, 40)

    spread_short = abs(ema9 - ema21) / atr * 100
    spread_long = abs(ema21 - hma40) / atr * 100
    ma_tight_balanced = spread_short < 50

    atr_buffer = 0.5 * atr
    price_near_ema9 = abs(closes[-1] - ema9) <= atr_buffer
    price_near_ema21 = abs(closes[-1] - ema21) <= atr_buffer
    price_near_short_ema = price_near_ema9 or price_near_ema21

    short_average = (atr3 + atr5 + atr8) / 3
    short_window = short_average[-30:]
    if not np.isfinite(short_window).all():
        return None
    highest_short_average = float(np.max(short_window))
    lowest_short_average = float(np.min(short_window))
    rmv = (
        (float(short_average[-1]) - lowest_short_average)
        / max(highest_short_average - lowest_short_average, 0.001)
        * 100
    )

    bar_ranges = highs - lows
    previous_ranges = bar_ranges[-22:-1]
    range_min = float(np.min(previous_ranges))
    range_max = float(np.max(previous_ranges))
    denominator = 1e-6 if range_max - range_min == 0 else range_max - range_min
    rmv_raw = 100 * (float(bar_ranges[-1]) - range_min) / denominator
    rmv_alt = min(100, max(0, rmv_raw))

    trend_ok = ema21 > sma50 or (
        sma200 is not None and sma50 > sma200
    )

    strict_coil = (
        atr_contracted_strict
        and trend_ok
        and (rmv < 21 or rmv_alt < 10)
        and price_near_short_ema
        and (
            (spread_short < 50 and spread_long < 50)
            or closes[-1] > sma50
            or ema5 > ema21
        )
    )

    balanced_coil = (
        atr_contracted_balanced
        and trend_ok
        and (rmv < 25 or rmv_alt < 20)
        and ma_tight_balanced
        and price_near_short_ema
    )

    return bool(balanced_coil or strict_coil)
