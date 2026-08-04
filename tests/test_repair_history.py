import copy
import unittest

from repair_history import find_misdated_snapshot_index, repair_misdated_snapshot


class RepairHistoryTests(unittest.TestCase):
    def test_repairs_duplicate_stale_snapshot_in_the_middle_of_history(self):
        history = {
            "dates": ["2026-07-31", "2026-08-03", "2026-07-31", "2026-08-04"],
            "wr_total": [1265, 1265, 1020, 1020],
            "d": {
                "AAPL": {
                    "wr": [548, None, 454, 665],
                    "seen": [True, True, True, True],
                    "tm": [["Tech"], ["Tech"], ["Software"], ["Software"]],
                    "vcp_coil_1": [False, False, True, False],
                },
                "NEW": {"wr": [10]},
            },
        }

        result = repair_misdated_snapshot(history)

        self.assertEqual(result["source_date"], "2026-07-31")
        self.assertEqual(result["target_date"], "2026-08-03")
        self.assertEqual(history["dates"], ["2026-07-31", "2026-08-03", "2026-08-04"])
        self.assertEqual(history["wr_total"], [1265, 1020, 1020])
        self.assertEqual(history["d"]["AAPL"]["wr"], [548, 454, 665])
        self.assertEqual(history["d"]["AAPL"]["seen"], [True, True, True])
        self.assertEqual(history["d"]["AAPL"]["tm"], [["Tech"], ["Software"], ["Software"]])
        self.assertEqual(history["d"]["AAPL"]["vcp_coil_1"], [False, True, False])
        self.assertEqual(history["d"]["NEW"]["wr"], [10])

    def test_refuses_to_change_a_chronological_history(self):
        history = {"dates": ["2026-08-03", "2026-08-04"], "d": {"A": {"wr": [1, 2]}}}
        before = copy.deepcopy(history)

        self.assertIsNone(find_misdated_snapshot_index(history["dates"]))
        self.assertIsNone(repair_misdated_snapshot(history))
        self.assertEqual(history, before)


if __name__ == "__main__":
    unittest.main()
