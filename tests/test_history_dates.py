import unittest

from history_dates import latest_history_session_date, select_history_session_date


class _Frame:
    def __init__(self, values):
        self.index = values

    def __len__(self):
        return len(self.index)


class HistoryDateTests(unittest.TestCase):
    def test_uses_consensus_not_the_first_stale_ticker(self):
        results = [{"t": "A"}, {"t": "B"}, {"t": "C"}, {"t": "D"}]
        daily_data = {
            "A": _Frame(["2026-07-31"]),
            "B": _Frame(["2026-08-03"]),
            "C": _Frame(["2026-08-03"]),
            "D": _Frame(["2026-08-03"]),
        }

        self.assertEqual(select_history_session_date(results, daily_data), "2026-08-03")

    def test_tie_prefers_the_later_session(self):
        results = [{"t": "A"}, {"t": "B"}]
        daily_data = {"A": _Frame(["2026-07-31"]), "B": _Frame(["2026-08-03"])}

        self.assertEqual(select_history_session_date(results, daily_data), "2026-08-03")

    def test_finds_latest_valid_existing_history_date(self):
        dates = ["2026-07-31", "bad", "2026-08-03", "2026-07-31"]

        self.assertEqual(latest_history_session_date(dates), "2026-08-03")


if __name__ == "__main__":
    unittest.main()
