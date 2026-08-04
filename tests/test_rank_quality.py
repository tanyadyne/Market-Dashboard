import unittest

from rank_quality import (
    adjusted_price_quality_flags,
    append_observation,
    prior_observation_sessions,
    probation_complete,
    rank_hold_code,
    technical_price_from_quote,
    trend_confirmed,
)


class RankQualityTests(unittest.TestCase):
    def test_unrepaired_tenfold_price_jump_is_held_before_ranking(self):
        closes = [3.8, 3.75, 3.9, 3.85, 3.8, 46.0]
        flags = adjusted_price_quality_flags(closes, closes)

        self.assertIn("extreme_adjusted_daily_move", flags)
        self.assertIn("unrepaired_price_scale_jump", flags)
        self.assertEqual(
            rank_hold_code(flags, 20, 46.0, 20.0, 15.0),
            "data",
        )

    def test_adjusted_reverse_split_does_not_trigger_data_hold(self):
        # Raw price jumps 10x. Correct adjusted history remains continuous.
        raw = [0.38, 0.39, 0.40, 4.0, 4.1]
        adjusted = [3.8, 3.9, 4.0, 4.0, 4.1]

        self.assertEqual(adjusted_price_quality_flags(adjusted, raw), [])

    def test_new_entrant_requires_five_completed_eod_sessions(self):
        self.assertFalse(probation_complete(3))
        self.assertTrue(probation_complete(4))
        self.assertEqual(
            rank_hold_code([], 3, 120.0, 110.0, 100.0),
            "probation",
        )
        self.assertIsNone(rank_hold_code([], 4, 120.0, 110.0, 100.0))

    def test_history_seen_field_handles_same_day_refresh_and_legacy_rows(self):
        new_entry = {"wr": []}
        append_observation(new_entry, 3)
        self.assertEqual(new_entry["seen"], [False, False, True])
        append_observation(new_entry, 3)
        self.assertEqual(new_entry["seen"], [False, False, True])
        self.assertEqual(prior_observation_sessions(new_entry), 1)

        legacy_entry = {"seen": [True], "wr": [12, 10, 8]}
        self.assertEqual(prior_observation_sessions(legacy_entry), 3)

    def test_stage_four_trend_remains_rank_eligible(self):
        self.assertFalse(trend_confirmed(3.85, 5.3, 9.5))
        self.assertIsNone(rank_hold_code([], 10, 3.85, 5.3, 9.5))
        self.assertTrue(trend_confirmed(120.0, 110.0, 100.0))

    def test_live_raw_quote_is_scaled_before_adjusted_return_math(self):
        # Raw $3.85 quote corresponds to adjusted $0.385 history.
        self.assertAlmostEqual(technical_price_from_quote(3.85, 0.1), 0.385)


if __name__ == "__main__":
    unittest.main()
