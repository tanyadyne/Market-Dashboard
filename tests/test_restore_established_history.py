import unittest

from restore_established_history import restore_snapshot


class RestoreEstablishedHistoryTests(unittest.TestCase):
    def test_reindexes_the_full_approved_cross_section_and_denominator(self):
        current = {
            "dates": ["2026-07-31", "2026-08-03", "2026-08-04"],
            "wr_total": [1265, 1020, 1089],
            "d": {
                "SNDK": {"wr": [4, None, 2], "tm": [["Old"], None, ["New"]]},
                "AAPL": {"wr": [3, 4, 1], "seen": [True, True, True]},
                "MVIS": {"wr": [None, None, None]},
            },
        }
        seed = {
            "dates": ["2026-07-31", "2026-08-03", "2026-08-04"],
            "d": {
                "SNDK": {"wr": [4, 3, 2], "tm": [["Old"], ["Semiconductors"], ["New"]]},
                "AAPL": {"wr": [3, 2, 1], "seen": [True, False, True]},
                "MVIS": {"wr": [1, 1, 1]},
                "OTHER": {"wr": [4, 4, 4]},
            },
        }
        registry = {
            "tickers": {
                "SNDK": {"status": "approved"},
                "AAPL": {"status": "approved"},
                "MVIS": {"status": "blocked"},
                "OTHER": {"status": "blocked"},
            }
        }

        repaired = restore_snapshot(current, seed, registry)

        self.assertEqual(repaired["source_denominator"], 4)
        self.assertEqual(repaired["denominator"], 2)
        self.assertEqual(repaired["restored"], ["AAPL", "SNDK"])
        self.assertEqual(repaired["excluded"], ["MVIS"])
        self.assertEqual(current["wr_total"], [1265, 2, 1089])
        self.assertEqual(current["d"]["SNDK"]["wr"], [4, 2, 2])
        self.assertEqual(current["d"]["SNDK"]["tm"][1], ["Semiconductors"])
        self.assertEqual(current["d"]["AAPL"]["wr"], [3, 1, 1])
        self.assertFalse(current["d"]["AAPL"]["seen"][1])
        self.assertIsNone(current["d"]["MVIS"]["wr"][1])

    def test_refuses_an_incomplete_seed_snapshot(self):
        current = {"dates": ["2026-08-03"], "d": {}, "wr_total": [1020]}
        seed = {"dates": ["2026-08-03"], "d": {"SNDK": {"wr": [2]}}}
        registry = {"tickers": {"SNDK": {"status": "approved"}}}

        with self.assertRaisesRegex(ValueError, "incomplete"):
            restore_snapshot(current, seed, registry)

    def test_reindexes_after_excluding_a_duplicate_ranked_ticker(self):
        current = {
            "dates": ["2026-08-03"],
            "wr_total": [4],
            "d": {
                "CAKE": {"wr": [2]},
                "TDC": {"wr": [3]},
                "SKYT": {"wr": [3]},
                "LAST": {"wr": [4]},
            },
        }
        seed = {
            "dates": ["2026-08-03"],
            "d": {
                "MVIS": {"wr": [1]},
                "CAKE": {"wr": [2]},
                "TDC": {"wr": [3]},
                "SKYT": {"wr": [3]},
                "LAST": {"wr": [4]},
            },
        }
        registry = {
            "tickers": {
                "CAKE": {"status": "approved"},
                "TDC": {"status": "approved"},
                "LAST": {"status": "approved"},
            }
        }

        repaired = restore_snapshot(current, seed, registry)

        self.assertEqual(repaired["denominator"], 3)
        self.assertEqual(current["wr_total"], [3])
        self.assertEqual(current["d"]["CAKE"]["wr"], [1])
        self.assertEqual(current["d"]["TDC"]["wr"], [2])
        self.assertEqual(current["d"]["LAST"]["wr"], [3])
        self.assertIsNone(current["d"]["SKYT"]["wr"][0])


if __name__ == "__main__":
    unittest.main()
