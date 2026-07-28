import unittest

import numpy as np

from preset_metrics import (
    PRESET_EMA9_ABOVE_EMA21,
    PRESET_EMA9_BELOW_EMA21,
    PRESET_NEW_52W_HIGH_LAST_3,
    PRESET_NEW_52W_LOW_LAST_3,
    PRESET_PRIOR_TWO_HAS_GREEN,
    PRESET_PRIOR_TWO_HAS_RED,
    PRESET_SMA10_ABOVE_SMA50,
    PRESET_SMA10_BELOW_SMA50,
    PRESET_SMA10_DECLINING,
    PRESET_SMA50_DECLINING,
    build_preset_intraday_baseline,
    compute_intraday_preset_state,
    compute_preset_flags,
)


def bars(closes):
    closes = np.asarray(closes, dtype=float)
    opens = closes.copy()
    highs = closes + 1
    lows = closes - 1
    return opens, highs, lows, closes


class PresetMetricTests(unittest.TestCase):
    def test_leader_structure_and_prior_red_candle(self):
        opens, highs, lows, closes = bars(np.linspace(50, 100, 260))
        opens[-2] = closes[-2] + 2

        flags = compute_preset_flags(
            opens,
            highs,
            lows,
            closes,
            ema9=99,
            ema21=97,
        )

        self.assertTrue(flags & PRESET_SMA10_ABOVE_SMA50)
        self.assertTrue(flags & PRESET_PRIOR_TWO_HAS_RED)
        self.assertTrue(flags & PRESET_EMA9_ABOVE_EMA21)
        self.assertFalse(flags & PRESET_EMA9_BELOW_EMA21)

    def test_laggard_structure_slopes_and_prior_green_candle(self):
        opens, highs, lows, closes = bars(np.linspace(100, 50, 260))
        opens[-3] = closes[-3] - 2

        flags = compute_preset_flags(
            opens,
            highs,
            lows,
            closes,
            ema9=51,
            ema21=53,
        )

        self.assertTrue(flags & PRESET_SMA10_BELOW_SMA50)
        self.assertTrue(flags & PRESET_SMA10_DECLINING)
        self.assertTrue(flags & PRESET_SMA50_DECLINING)
        self.assertTrue(flags & PRESET_PRIOR_TWO_HAS_GREEN)
        self.assertTrue(flags & PRESET_EMA9_BELOW_EMA21)
        self.assertFalse(flags & PRESET_EMA9_ABOVE_EMA21)

    def test_new_high_detected_when_it_occurred_two_bars_ago(self):
        opens, highs, lows, closes = bars(np.full(260, 100.0))
        highs[-2] = 120
        highs[-1] = 110

        flags = compute_preset_flags(opens, highs, lows, closes)

        self.assertTrue(flags & PRESET_NEW_52W_HIGH_LAST_3)

    def test_new_low_detected_when_it_occurred_three_bars_ago(self):
        opens, highs, lows, closes = bars(np.full(260, 100.0))
        lows[-3] = 80
        lows[-2] = 90
        lows[-1] = 95

        flags = compute_preset_flags(opens, highs, lows, closes)

        self.assertTrue(flags & PRESET_NEW_52W_LOW_LAST_3)

    def test_extreme_four_bars_ago_is_outside_recent_window(self):
        opens, highs, lows, closes = bars(np.full(260, 100.0))
        highs[-4] = 120
        highs[-3:] = [115, 114, 113]
        lows[-4] = 80
        lows[-3:] = [85, 86, 87]

        flags = compute_preset_flags(opens, highs, lows, closes)

        self.assertFalse(flags & PRESET_NEW_52W_HIGH_LAST_3)
        self.assertFalse(flags & PRESET_NEW_52W_LOW_LAST_3)

    def test_current_candle_is_not_one_of_the_prior_two(self):
        opens, highs, lows, closes = bars(np.full(260, 100.0))
        opens[-1] = closes[-1] + 2

        flags = compute_preset_flags(opens, highs, lows, closes)

        self.assertFalse(flags & PRESET_PRIOR_TWO_HAS_RED)

    def test_intraday_next_session_updates_mas_candles_and_new_high(self):
        opens, highs, lows, closes = bars(np.linspace(50, 100, 260))
        opens[-1] = closes[-1] + 2
        baseline = build_preset_intraday_baseline(
            opens,
            highs,
            lows,
            closes,
            last_date="2026-07-27",
        )
        baseline["_ma_ema9"] = 99
        baseline["_ma_ema21"] = 97

        state = compute_intraday_preset_state(
            baseline,
            105,
            session_date="2026-07-28",
            day_high=110,
            day_low=104,
        )

        self.assertIsNotNone(state)
        self.assertFalse(state["same_session"])
        self.assertTrue(state["flags"] & PRESET_SMA10_ABOVE_SMA50)
        self.assertTrue(state["flags"] & PRESET_EMA9_ABOVE_EMA21)
        self.assertTrue(state["flags"] & PRESET_PRIOR_TWO_HAS_RED)
        self.assertTrue(state["flags"] & PRESET_NEW_52W_HIGH_LAST_3)

    def test_intraday_same_session_excludes_current_candle(self):
        opens, highs, lows, closes = bars(np.full(260, 100.0))
        opens[-1] = closes[-1] + 2
        baseline = build_preset_intraday_baseline(
            opens,
            highs,
            lows,
            closes,
            last_date="2026-07-28",
        )

        state = compute_intraday_preset_state(
            baseline,
            99,
            session_date="2026-07-28",
            day_high=101,
            day_low=98,
        )

        self.assertIsNotNone(state)
        self.assertTrue(state["same_session"])
        self.assertFalse(state["flags"] & PRESET_PRIOR_TWO_HAS_RED)

    def test_intraday_preserves_yesterdays_new_high_in_three_day_window(self):
        opens, highs, lows, closes = bars(np.full(260, 100.0))
        highs[-1] = 120
        baseline = build_preset_intraday_baseline(
            opens,
            highs,
            lows,
            closes,
            last_date="2026-07-27",
        )
        baseline["_ma_ema9"] = 100
        baseline["_ma_ema21"] = 100

        state = compute_intraday_preset_state(
            baseline,
            105,
            session_date="2026-07-28",
            day_high=110,
            day_low=104,
        )

        self.assertTrue(state["flags"] & PRESET_NEW_52W_HIGH_LAST_3)

    def test_intraday_quote_is_scaled_to_adjusted_history(self):
        opens, highs, lows, closes = bars(np.full(260, 100.0))
        baseline = build_preset_intraday_baseline(
            opens,
            highs,
            lows,
            closes,
            last_date="2026-07-27",
        )
        baseline["_ma_ema9"] = 100
        baseline["_ma_ema21"] = 100
        baseline["_preset_price_scale"] = 0.5

        state = compute_intraday_preset_state(
            baseline,
            202,
            session_date="2026-07-28",
            day_high=204,
            day_low=200,
        )

        self.assertEqual(state["technical_price"], 101)
        self.assertAlmostEqual(state["ma_values"]["sma50"], 100.02)

    def test_intraday_new_high_updates_for_short_history_stock(self):
        opens, highs, lows, closes = bars(np.linspace(90, 100, 20))
        baseline = build_preset_intraday_baseline(
            opens,
            highs,
            lows,
            closes,
            last_date="2026-07-27",
        )

        state = compute_intraday_preset_state(
            baseline,
            105,
            session_date="2026-07-28",
            day_high=110,
            day_low=104,
        )

        self.assertIsNotNone(state)
        self.assertTrue(state["flags"] & PRESET_NEW_52W_HIGH_LAST_3)
        self.assertIsNone(state["ma_values"]["sma50"])

    def test_intraday_extremes_fail_closed_without_threshold_history(self):
        opens, highs, lows, closes = bars(np.linspace(90, 100, 20))
        baseline = build_preset_intraday_baseline(
            opens,
            highs,
            lows,
            closes,
            last_date="2026-07-27",
        )
        for direction in ("high", "low"):
            baseline[f"_preset_{direction}_prior_next"] = False
            baseline[f"_preset_{direction}_threshold_next"] = None

        state = compute_intraday_preset_state(
            baseline,
            105,
            session_date="2026-07-28",
            day_high=110,
            day_low=80,
        )

        self.assertFalse(state["flags"] & PRESET_NEW_52W_HIGH_LAST_3)
        self.assertFalse(state["flags"] & PRESET_NEW_52W_LOW_LAST_3)

    def test_equal_moving_averages_set_neither_direction(self):
        opens, highs, lows, closes = bars(np.full(260, 100.0))

        flags = compute_preset_flags(
            opens,
            highs,
            lows,
            closes,
            ema9=100,
            ema21=100,
        )

        self.assertFalse(flags & PRESET_SMA10_ABOVE_SMA50)
        self.assertFalse(flags & PRESET_SMA10_BELOW_SMA50)
        self.assertFalse(flags & PRESET_SMA10_DECLINING)
        self.assertFalse(flags & PRESET_SMA50_DECLINING)
        self.assertFalse(flags & PRESET_EMA9_ABOVE_EMA21)
        self.assertFalse(flags & PRESET_EMA9_BELOW_EMA21)


if __name__ == "__main__":
    unittest.main()
