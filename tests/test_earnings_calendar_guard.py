import unittest
from datetime import date

import fetch_economic_calendar as calendar


def event(ticker, event_date, reported=False):
    return {
        "date": event_date,
        "day": event_date,
        "ticker": ticker,
        "company": ticker,
        "group": "Test",
        "time": "AMC",
        "_reported": reported,
        "_event_name": "Earnings Announcement",
    }


class EarningsCalendarGuardTests(unittest.TestCase):
    def test_roll_forward_is_quarantined(self):
        state = {
            "tickers": {
                "ROLL": {
                    "last_date": "2026-07-12",
                    "stable_since": "2026-07-10",
                    "last_seen": "2026-07-12",
                    "quarantined": False,
                    "roll_count": 0,
                }
            }
        }

        allowed, suppressed = calendar._apply_earnings_reliability_guard(
            [event("ROLL", "2026-07-13")], state, date(2026, 7, 13)
        )

        self.assertEqual([], allowed)
        self.assertEqual("ROLL", suppressed[0]["ticker"])
        self.assertEqual("rolling_unreported_date", suppressed[0]["reason"])
        self.assertTrue(state["tickers"]["ROLL"]["quarantined"])

    def test_new_imminent_unreported_date_is_allowed_until_it_rolls(self):
        state = {"tickers": {}}

        allowed, suppressed = calendar._apply_earnings_reliability_guard(
            [event("FRESH", "2026-07-14")], state, date(2026, 7, 13)
        )

        self.assertEqual([], suppressed)
        self.assertEqual("FRESH", allowed[0]["ticker"])

    def test_new_future_date_is_allowed(self):
        state = {"tickers": {}}

        allowed, suppressed = calendar._apply_earnings_reliability_guard(
            [event("FUTURE", "2026-07-20")], state, date(2026, 7, 13)
        )

        self.assertEqual([], suppressed)
        self.assertEqual("FUTURE", allowed[0]["ticker"])
        self.assertNotIn("_reported", allowed[0])

    def test_stable_future_date_recovers_from_quarantine(self):
        state = {
            "tickers": {
                "STABLE": {
                    "last_date": "2026-07-20",
                    "stable_since": "2026-07-01",
                    "last_seen": "2026-07-12",
                    "quarantined": True,
                    "reason": "rolling_unreported_date",
                    "roll_count": 2,
                }
            }
        }

        allowed, suppressed = calendar._apply_earnings_reliability_guard(
            [event("STABLE", "2026-07-20")], state, date(2026, 7, 13)
        )

        self.assertEqual([], suppressed)
        self.assertEqual("STABLE", allowed[0]["ticker"])
        self.assertFalse(state["tickers"]["STABLE"]["quarantined"])

    def test_reported_event_clears_quarantine(self):
        state = {
            "tickers": {
                "DONE": {
                    "last_date": "2026-07-13",
                    "quarantined": True,
                    "reason": "rolling_unreported_date",
                }
            }
        }

        allowed, suppressed = calendar._apply_earnings_reliability_guard(
            [event("DONE", "2026-07-13", reported=True)], state, date(2026, 7, 13)
        )

        self.assertEqual([], suppressed)
        self.assertEqual("DONE", allowed[0]["ticker"])
        self.assertFalse(state["tickers"]["DONE"]["quarantined"])

    def test_manually_confirmed_date_clears_quarantine(self):
        state = {
            "tickers": {
                "CONFIRMED": {
                    "last_date": "2026-07-13",
                    "confirmed_date": "2026-08-04",
                    "quarantined": True,
                    "reason": "rolling_unreported_date",
                }
            }
        }

        allowed, suppressed = calendar._apply_earnings_reliability_guard(
            [event("CONFIRMED", "2026-08-04")], state, date(2026, 7, 13)
        )

        self.assertEqual([], suppressed)
        self.assertEqual("CONFIRMED", allowed[0]["ticker"])
        self.assertFalse(state["tickers"]["CONFIRMED"]["quarantined"])


if __name__ == "__main__":
    unittest.main()
