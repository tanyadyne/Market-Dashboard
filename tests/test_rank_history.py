import unittest

from rank_history import (
    WEEKLY_RANK_TOTALS_FIELD,
    aligned_ranked_totals,
    infer_ranked_total,
    set_latest_ranked_total,
)


class RankHistoryTests(unittest.TestCase):
    def test_legacy_complete_rank_series_recovers_exact_totals(self):
        scores = {
            "AAA": {"wr": [1, 3]},
            "BBB": {"wr": [2, 1]},
            "CCC": {"wr": [3, 2]},
        }
        history = {"dates": ["2026-08-03", "2026-08-04"], "d": scores}

        self.assertEqual(aligned_ranked_totals(history, history["dates"], scores), [3, 3])

    def test_incomplete_legacy_rank_series_is_not_guessed(self):
        scores = {
            "AAA": {"wr": [1]},
            "CCC": {"wr": [3]},
        }

        self.assertIsNone(infer_ranked_total(scores, 0))

    def test_stored_total_is_preserved_when_present(self):
        scores = {
            "AAA": {"wr": [1]},
            "BBB": {"wr": [2]},
            "CCC": {"wr": [3]},
        }
        history = {
            "dates": ["2026-08-03"],
            "d": scores,
            WEEKLY_RANK_TOTALS_FIELD: [5],
        }

        self.assertEqual(aligned_ranked_totals(history, history["dates"], scores), [5])

    def test_latest_total_is_aligned_and_replaced_in_place(self):
        totals = set_latest_ranked_total([1265], 2, 1255)
        self.assertEqual(totals, [1265, 1255])
        self.assertEqual(set_latest_ranked_total(totals, 2, 1267), [1265, 1267])


if __name__ == "__main__":
    unittest.main()
