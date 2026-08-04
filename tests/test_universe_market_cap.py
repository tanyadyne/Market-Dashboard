import sys
import types
import unittest

# The policy is pure local logic; avoid requiring Yahoo's runtime package just
# to import the pipeline module in lightweight test environments.
if "yfinance" not in sys.modules:
    sys.modules["yfinance"] = types.SimpleNamespace(Ticker=type("Ticker", (), {}))

from fetch_leaders import (
    CAP_ONLY_MIN_MCAP,
    STANDARD_MIN_MCAP,
    min_mcap_for_industry,
)


class UniverseMarketCapTests(unittest.TestCase):
    def test_semiconductor_theme_industries_require_two_billion(self):
        for industry in (
            "Semiconductors",
            "Semiconductor Equipment & Materials",
            "Electronic Components",
            "Computer Hardware",
            "Scientific & Technical Instruments",
        ):
            self.assertEqual(min_mcap_for_industry(industry), STANDARD_MIN_MCAP)

    def test_other_cap_only_industries_keep_their_existing_floor(self):
        self.assertEqual(
            min_mcap_for_industry("Information Technology Services"),
            CAP_ONLY_MIN_MCAP,
        )

    def test_standard_industries_require_two_billion(self):
        self.assertEqual(min_mcap_for_industry("Banks - Regional"), STANDARD_MIN_MCAP)


if __name__ == "__main__":
    unittest.main()
