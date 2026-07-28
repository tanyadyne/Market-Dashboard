import math
import unittest

import numpy as np

from vcp_coils import (
    compute_vcp_coil_definition_1,
    compute_vcp_coil_definition_1_score,
    compute_vcp_coil_definition_2,
    set_aligned_vcp_history_value,
)


def _reference_rma(values, length):
    out = np.full(len(values), np.nan)
    out[length - 1] = np.mean(values[:length])
    for i in range(length, len(values)):
        out[i] = values[i] / length + out[i - 1] * (length - 1) / length
    return out


def _reference_atr(highs, lows, closes, length):
    true_range = np.empty(len(closes))
    true_range[0] = highs[0] - lows[0]
    for i in range(1, len(closes)):
        true_range[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
    return _reference_rma(true_range, length)


def _reference_ema(values, length):
    alpha = 2 / (length + 1)
    result = values[0]
    for value in values[1:]:
        result = alpha * value + (1 - alpha) * result
    return result


def _reference_wma(values, length):
    weights = np.arange(1, length + 1)
    return np.dot(values[-length:], weights) / weights.sum()


def _reference_hma(values, length):
    root_length = math.floor(math.sqrt(length))
    transformed = []
    for end in range(len(values) - root_length + 1, len(values) + 1):
        prefix = values[:end]
        transformed.append(
            2 * _reference_wma(prefix, length // 2)
            - _reference_wma(prefix, length)
        )
    return _reference_wma(np.asarray(transformed), root_length)


def _reference_balanced_coil_detected(opens, highs, lows, closes):
    atr3 = _reference_atr(highs, lows, closes, 3)
    atr5 = _reference_atr(highs, lows, closes, 5)
    atr8 = _reference_atr(highs, lows, closes, 8)
    atr14 = _reference_atr(highs, lows, closes, 14)
    atr = atr14[-1]
    atr_avg = np.mean(atr14[-50:])

    ema5 = _reference_ema(closes, 5)
    ema9 = _reference_ema(closes, 9)
    ema21 = _reference_ema(closes, 21)
    sma50 = np.mean(closes[-50:])
    sma200 = np.mean(closes[-200:]) if len(closes) >= 200 else None
    hma40 = _reference_hma(lows, 40)

    spread_short = abs(ema9 - ema21) / atr * 100
    spread_long = abs(ema21 - hma40) / atr * 100
    near_ema = (
        abs(closes[-1] - ema9) <= 0.5 * atr
        or abs(closes[-1] - ema21) <= 0.5 * atr
    )

    short_average = (atr3 + atr5 + atr8) / 3
    short_window = short_average[-30:]
    rmv = (
        (short_average[-1] - np.min(short_window))
        / max(np.max(short_window) - np.min(short_window), 0.001)
        * 100
    )

    bar_ranges = highs - lows
    previous_ranges = bar_ranges[-22:-1]
    range_min = np.min(previous_ranges)
    range_max = np.max(previous_ranges)
    denominator = 1e-6 if range_max - range_min == 0 else range_max - range_min
    rmv_alt = min(
        100,
        max(0, 100 * (bar_ranges[-1] - range_min) / denominator),
    )

    trend_ok = ema21 > sma50 or (
        sma200 is not None and sma50 > sma200
    )
    strict = (
        abs(closes[-1] - opens[-1]) < atr_avg * 0.6
        and abs(highs[-1] - lows[-1]) < atr_avg * 0.85
        and trend_ok
        and (rmv < 21 or rmv_alt < 10)
        and near_ema
        and (
            (spread_short < 50 and spread_long < 50)
            or closes[-1] > sma50
            or ema5 > ema21
        )
    )
    balanced = (
        abs(highs[-1] - lows[-1]) < atr_avg * 0.8
        and trend_ok
        and (rmv < 25 or rmv_alt < 20)
        and spread_short < 50
        and near_ema
    )
    return bool(balanced or strict)


def _fixture(final_width=0.15):
    count = 260
    index = np.arange(count, dtype=float)
    closes = 100 + 0.05 * index + 0.15 * np.sin(index / 4)
    opens = closes + 0.02 * np.cos(index)
    widths = np.full(count, 2.0)
    widths[-30:] = np.linspace(1.0, final_width, 30)
    highs = np.maximum(opens, closes) + widths / 2
    lows = np.minimum(opens, closes) - widths / 2
    return opens, highs, lows, closes


class VcpCoilTests(unittest.TestCase):
    def test_definition_1_boolean_uses_existing_score_threshold(self):
        _, highs, lows, closes = _fixture()
        score = compute_vcp_coil_definition_1_score(closes, highs, lows)

        self.assertIsNotNone(score)
        self.assertEqual(
            compute_vcp_coil_definition_1(closes, highs, lows),
            score <= 10,
        )

    def test_definition_2_matches_balanced_pine_reference(self):
        for final_width in (0.15, 0.6, 2.5, 5.0):
            bars = _fixture(final_width)
            self.assertEqual(
                compute_vcp_coil_definition_2(*bars),
                _reference_balanced_coil_detected(*bars),
            )

    def test_definition_2_can_signal_without_sma200_when_ema21_exceeds_sma50(self):
        bars = tuple(values[-100:] for values in _fixture())

        self.assertTrue(_reference_balanced_coil_detected(*bars))
        self.assertEqual(
            compute_vcp_coil_definition_2(*bars),
            _reference_balanced_coil_detected(*bars),
        )

    def test_definition_2_returns_none_before_atr_average_is_available(self):
        bars = tuple(values[:62] for values in _fixture())

        self.assertIsNone(compute_vcp_coil_definition_2(*bars))

    def test_history_series_backfills_and_updates_in_place(self):
        entry = {"wr": [3, 2, 1]}

        set_aligned_vcp_history_value(entry, "vcp_coil_1", 3, True)
        set_aligned_vcp_history_value(entry, "vcp_coil_2", 3, False)
        self.assertEqual([None, None, True], entry["vcp_coil_1"])
        self.assertEqual([None, None, False], entry["vcp_coil_2"])

        set_aligned_vcp_history_value(entry, "vcp_coil_1", 3, False)
        self.assertEqual([None, None, False], entry["vcp_coil_1"])


if __name__ == "__main__":
    unittest.main()
