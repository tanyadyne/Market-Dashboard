import unittest

from restore_established_history import restore_snapshot


class RestoreEstablishedHistoryTests(unittest.TestCase):
    def test_replaces_the_full_approved_cross_section_and_denominator(self):
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

        self.assertEqual(repaired["denominator"], 4)
        self.assertEqual(repaired["restored"], ["AAPL", "SNDK"])
        self.assertEqual(current["wr_total"], [1265, 4, 1089])
        self.assertEqual(current["d"]["SNDK"]["wr"], [4, 3, 2])
        self.assertEqual(current["d"]["SNDK"]["tm"][1], ["Semiconductors"])
        self.assertEqual(current["d"]["AAPL"]["wr"], [3, 2, 1])
        self.assertFalse(current["d"]["AAPL"]["seen"][1])
        self.assertIsNone(current["d"]["MVIS"]["wr"][1])

    def test_refuses_an_incomplete_seed_snapshot(self):
        current = {"dates": ["2026-08-03"], "d": {}, "wr_total": [1020]}
        seed = {"dates": ["2026-08-03"], "d": {"SNDK": {"wr": [2]}}}
        registry = {"tickers": {"SNDK": {"status": "approved"}}}

        with self.assertRaisesRegex(ValueError, "incomplete"):
            restore_snapshot(current, seed, registry)


if __name__ == "__main__":
    unittest.main()
