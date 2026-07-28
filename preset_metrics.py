"""Compact daily technical flags used by stock-screener presets."""

import numpy as np


PRESET_SMA10_ABOVE_SMA50 = 1
PRESET_SMA10_BELOW_SMA50 = 2
PRESET_PRIOR_TWO_HAS_RED = 4
PRESET_PRIOR_TWO_HAS_GREEN = 8
PRESET_NEW_52W_HIGH_LAST_3 = 16
PRESET_NEW_52W_LOW_LAST_3 = 32
PRESET_SMA10_DECLINING = 64
PRESET_SMA50_DECLINING = 128
PRESET_EMA9_ABOVE_EMA21 = 256
PRESET_EMA9_BELOW_EMA21 = 512


def json_numpy_scalar(value):
    """Convert NumPy values at JSON boundaries without masking other type errors."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(
        f"Object of type {value.__class__.__name__} is not JSON serializable"
    )


def _finite_float(value):
    try:
        value = float(value)
        return value if np.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def _ema_value(values, period):
    values = np.asarray(values, dtype=float)
    if len(values) == 0 or not np.isfinite(values).all():
        return None
    if len(values) < period:
        return float(np.mean(values))
    multiplier = 2.0 / (period + 1)
    ema = float(np.mean(values[:period]))
    for value in values[period:]:
        ema = (float(value) - ema) * multiplier + ema
    return ema


def _is_rolling_extreme(values, index, *, lookback=252, find_high=True):
    if index < 0 or index >= len(values):
        return False
    value = _finite_float(values[index])
    if value is None:
        return False
    start = max(0, index - lookback + 1)
    window = np.asarray(values[start:index + 1], dtype=float)
    if not np.isfinite(window).any():
        return False
    extreme = np.nanmax(window) if find_high else np.nanmin(window)
    return bool(value >= extreme if find_high else value <= extreme)


def _finite_extreme(values, *, find_high=True):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return None
    return float(np.max(finite) if find_high else np.min(finite))


def _has_new_rolling_extreme(values, *, recent_bars=3, lookback=252, find_high=True):
    """Return whether any recent bar set its trailing-lookback high or low."""
    n = len(values)
    for index in range(max(0, n - recent_bars), n):
        if _is_rolling_extreme(
            values,
            index,
            lookback=lookback,
            find_high=find_high,
        ):
            return True
    return False


def compute_preset_flags(opens, highs, lows, closes, ema9=None, ema21=None):
    """Return bitmask for preset-only daily conditions.

    Moving-average slope means the current SMA is strictly below its value one
    daily bar earlier. "Prior two candles" excludes the current bar.
    """
    try:
        opens = np.asarray(opens, dtype=float)
        highs = np.asarray(highs, dtype=float)
        lows = np.asarray(lows, dtype=float)
        closes = np.asarray(closes, dtype=float)
    except Exception:
        return 0

    n = len(closes)
    if n == 0 or len(opens) != n or len(highs) != n or len(lows) != n:
        return 0

    flags = 0

    if n >= 50 and np.isfinite(closes[-50:]).all():
        sma10 = float(np.mean(closes[-10:]))
        sma50 = float(np.mean(closes[-50:]))
        if sma10 > sma50:
            flags |= PRESET_SMA10_ABOVE_SMA50
        elif sma10 < sma50:
            flags |= PRESET_SMA10_BELOW_SMA50

        if n >= 51 and np.isfinite(closes[-51:-1]).all():
            sma10_previous = float(np.mean(closes[-11:-1]))
            sma50_previous = float(np.mean(closes[-51:-1]))
            if sma10 < sma10_previous:
                flags |= PRESET_SMA10_DECLINING
            if sma50 < sma50_previous:
                flags |= PRESET_SMA50_DECLINING

    if n >= 3:
        prior_indexes = (-2, -3)
        if any(
            np.isfinite(closes[i])
            and np.isfinite(opens[i])
            and closes[i] < opens[i]
            for i in prior_indexes
        ):
            flags |= PRESET_PRIOR_TWO_HAS_RED
        if any(
            np.isfinite(closes[i])
            and np.isfinite(opens[i])
            and closes[i] > opens[i]
            for i in prior_indexes
        ):
            flags |= PRESET_PRIOR_TWO_HAS_GREEN

    if _has_new_rolling_extreme(highs, find_high=True):
        flags |= PRESET_NEW_52W_HIGH_LAST_3
    if _has_new_rolling_extreme(lows, find_high=False):
        flags |= PRESET_NEW_52W_LOW_LAST_3

    try:
        ema9_value = float(ema9)
        ema21_value = float(ema21)
        if np.isfinite(ema9_value) and np.isfinite(ema21_value):
            if ema9_value > ema21_value:
                flags |= PRESET_EMA9_ABOVE_EMA21
            elif ema9_value < ema21_value:
                flags |= PRESET_EMA9_BELOW_EMA21
    except (TypeError, ValueError):
        pass

    return flags


def build_preset_intraday_baseline(
    opens,
    highs,
    lows,
    closes,
    *,
    last_date,
):
    """Build compact EOD inputs for exact preset refreshes during the next session."""
    try:
        opens = np.asarray(opens, dtype=float)
        highs = np.asarray(highs, dtype=float)
        lows = np.asarray(lows, dtype=float)
        closes = np.asarray(closes, dtype=float)
    except Exception:
        return {}

    n = len(closes)
    if (
        n < 3
        or len(opens) != n
        or len(highs) != n
        or len(lows) != n
        or not last_date
    ):
        return {}

    close_tail = closes[-51:]
    if len(close_tail) == 0 or not np.isfinite(close_tail).all():
        return {}

    def candle_is_red(index):
        return bool(
            np.isfinite(opens[index])
            and np.isfinite(closes[index])
            and closes[index] < opens[index]
        )

    def candle_is_green(index):
        return bool(
            np.isfinite(opens[index])
            and np.isfinite(closes[index])
            and closes[index] > opens[index]
        )

    high_flags = [
        _is_rolling_extreme(highs, index, find_high=True)
        for index in range(max(0, n - 3), n)
    ]
    low_flags = [
        _is_rolling_extreme(lows, index, find_high=False)
        for index in range(max(0, n - 3), n)
    ]
    while len(high_flags) < 3:
        high_flags.insert(0, False)
        low_flags.insert(0, False)

    return {
        "_preset_last_date": str(last_date),
        "_preset_closes": [float(value) for value in close_tail],
        "_preset_ema9_before_last": _ema_value(closes[:-1], 9),
        "_preset_ema21_before_last": _ema_value(closes[:-1], 21),
        "_preset_prior_red_same": candle_is_red(-2) or candle_is_red(-3),
        "_preset_prior_green_same": candle_is_green(-2) or candle_is_green(-3),
        "_preset_prior_red_next": candle_is_red(-1) or candle_is_red(-2),
        "_preset_prior_green_next": candle_is_green(-1) or candle_is_green(-2),
        "_preset_high_prior_same": high_flags[0] or high_flags[1],
        "_preset_high_prior_next": high_flags[1] or high_flags[2],
        "_preset_low_prior_same": low_flags[0] or low_flags[1],
        "_preset_low_prior_next": low_flags[1] or low_flags[2],
        "_preset_high_threshold_same": _finite_extreme(
            highs[max(0, n - 252):n - 1],
            find_high=True,
        ),
        "_preset_high_threshold_next": _finite_extreme(
            highs[max(0, n - 251):n],
            find_high=True,
        ),
        "_preset_low_threshold_same": _finite_extreme(
            lows[max(0, n - 252):n - 1],
            find_high=False,
        ),
        "_preset_low_threshold_next": _finite_extreme(
            lows[max(0, n - 251):n],
            find_high=False,
        ),
        "_preset_day_high_same": _finite_float(highs[-1]),
        "_preset_day_low_same": _finite_float(lows[-1]),
    }


def compute_intraday_preset_state(
    baseline,
    live_price,
    *,
    session_date,
    day_high=None,
    day_low=None,
):
    """Recompute preset flags and short MAs from a live intraday quote."""
    live_price = _finite_float(live_price)
    baseline_date = str(baseline.get("_preset_last_date") or "")
    session_date = str(session_date or "")
    if (
        live_price is None
        or live_price <= 0
        or not baseline_date
        or not session_date
        or session_date < baseline_date
    ):
        return None
    price_scale = _finite_float(baseline.get("_preset_price_scale"))
    if price_scale is None or price_scale <= 0:
        price_scale = 1.0
    live_price *= price_scale

    try:
        closes = [
            float(value)
            for value in baseline.get("_preset_closes", [])
            if _finite_float(value) is not None
        ]
    except Exception:
        return None
    if not closes:
        return None

    same_session = session_date == baseline_date
    if same_session:
        current_closes = closes.copy()
        current_closes[-1] = live_price
        previous_closes = closes[:-1]
        ema9_base = _finite_float(baseline.get("_preset_ema9_before_last"))
        ema21_base = _finite_float(baseline.get("_preset_ema21_before_last"))
        suffix = "same"
    else:
        current_closes = closes + [live_price]
        previous_closes = closes
        ema9_base = _finite_float(baseline.get("_ma_ema9"))
        ema21_base = _finite_float(baseline.get("_ma_ema21"))
        suffix = "next"

    sma10 = (
        float(np.mean(current_closes[-10:]))
        if len(current_closes) >= 10
        else None
    )
    sma50 = (
        float(np.mean(current_closes[-50:]))
        if len(current_closes) >= 50
        else None
    )
    previous_sma10 = (
        float(np.mean(previous_closes[-10:]))
        if len(previous_closes) >= 10
        else None
    )
    previous_sma50 = (
        float(np.mean(previous_closes[-50:]))
        if len(previous_closes) >= 50
        else None
    )
    ema9 = (
        (live_price - ema9_base) * (2.0 / 10.0) + ema9_base
        if ema9_base is not None
        else None
    )
    ema21 = (
        (live_price - ema21_base) * (2.0 / 22.0) + ema21_base
        if ema21_base is not None
        else None
    )

    flags = 0
    if sma10 is not None and sma50 is not None:
        if sma10 > sma50:
            flags |= PRESET_SMA10_ABOVE_SMA50
        elif sma10 < sma50:
            flags |= PRESET_SMA10_BELOW_SMA50
    if (
        sma10 is not None
        and previous_sma10 is not None
        and sma10 < previous_sma10
    ):
        flags |= PRESET_SMA10_DECLINING
    if (
        sma50 is not None
        and previous_sma50 is not None
        and sma50 < previous_sma50
    ):
        flags |= PRESET_SMA50_DECLINING
    if ema9 is not None and ema21 is not None:
        if ema9 > ema21:
            flags |= PRESET_EMA9_ABOVE_EMA21
        elif ema9 < ema21:
            flags |= PRESET_EMA9_BELOW_EMA21

    if baseline.get(f"_preset_prior_red_{suffix}"):
        flags |= PRESET_PRIOR_TWO_HAS_RED
    if baseline.get(f"_preset_prior_green_{suffix}"):
        flags |= PRESET_PRIOR_TWO_HAS_GREEN

    observed_high = _finite_float(day_high)
    observed_low = _finite_float(day_low)
    if observed_high is not None:
        observed_high *= price_scale
    if observed_low is not None:
        observed_low *= price_scale
    if same_session:
        baseline_day_high = _finite_float(baseline.get("_preset_day_high_same"))
        baseline_day_low = _finite_float(baseline.get("_preset_day_low_same"))
        if baseline_day_high is not None:
            observed_high = max(observed_high or live_price, baseline_day_high)
        if baseline_day_low is not None:
            observed_low = min(observed_low or live_price, baseline_day_low)
    observed_high = max(live_price, observed_high or live_price)
    observed_low = min(live_price, observed_low or live_price)
    high_threshold = _finite_float(
        baseline.get(f"_preset_high_threshold_{suffix}")
    )
    low_threshold = _finite_float(
        baseline.get(f"_preset_low_threshold_{suffix}")
    )
    if (
        baseline.get(f"_preset_high_prior_{suffix}")
        or (
            high_threshold is not None
            and observed_high >= high_threshold
        )
    ):
        flags |= PRESET_NEW_52W_HIGH_LAST_3
    if (
        baseline.get(f"_preset_low_prior_{suffix}")
        or (
            low_threshold is not None
            and observed_low <= low_threshold
        )
    ):
        flags |= PRESET_NEW_52W_LOW_LAST_3

    return {
        "flags": flags,
        "same_session": same_session,
        "technical_price": live_price,
        "ma_values": {
            "ema9": ema9,
            "ema21": ema21,
            "sma10": sma10,
            "sma50": sma50,
        },
    }
