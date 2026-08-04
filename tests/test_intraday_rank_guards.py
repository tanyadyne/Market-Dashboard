import sys
import types
import unittest
from unittest.mock import patch

# CI installs yfinance, but keep this unit test focused on local math and usable
# in lightweight environments too.
if "yfinance" not in sys.modules:
    sys.modules["yfinance"] = types.SimpleNamespace(Ticker=type("Ticker", (), {}))

import fetch_leaders
import fetch_leaders_intraday


class IntradayRankGuardTests(unittest.TestCase):
    def test_overlay_uses_adjusted_live_price_for_period_returns(self):
        row = {
            "t": "SCALE",
            "p": 3.80,  # raw/display price
            "dv": 100,
            "_preset_price_scale": 0.1,
            "_5b": 0.35,
            "_20b": 0.40,
            "_yb": 0.50,
            "_atr": 0.03,
            "_ma_ema9": 0.38,
            "_ma_ema21": 0.37,
            "_ma_sma50": 0.36,
            "_ma_ema65": 0.35,
            "_ma_sma200": 0.34,
        }
        quotes = {"SCALE": (3.85, 3.80, None, None, None)}
        with patch.object(fetch_leaders, "is_us_market_open_or_recently_closed", return_value=True), \
             patch.object(fetch_leaders, "fetch_live_quotes_bulk", return_value=quotes):
            fetch_leaders.apply_intraday_overlay([row])

        self.assertEqual(row["p"], 3.85)  # display price stays raw
        self.assertAlmostEqual(row["c5"], 10.0)  # 0.385 / 0.35 - 1
        self.assertAlmostEqual(row["c20"], -3.75)

    def test_intraday_gate_compares_adjusted_quote_to_adjusted_history(self):
        entry = {"p": 3.85, "ma": 20}
        baseline = {
            "_preset_price_scale": 0.1,
            "_pc": 0.38,
            "_atr": 0.03,
            "_ma_sma50": 0.36,
            "_ma_sma200": 0.34,
        }
        self.assertIsNone(fetch_leaders_intraday.intraday_rank_hold(entry, baseline))


if __name__ == "__main__":
    unittest.main()
