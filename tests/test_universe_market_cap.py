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
    recent_split_like_close_gap,
    select_market_cap,
    shares_reconcile_with_market_cap,
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

    def test_direct_market_cap_wins_over_pre_split_share_cache(self):
        # MVIS: $3.85 on the new 1-for-15 split scale. A stale 344.6M
        # pre-split share count would incorrectly fabricate a $1.326B cap.
        direct_cap = 89_898_000
        selected = select_market_cap(
            market_cap=direct_cap,
            market_cap_source="yahoo_quote",
            market_cap_refreshed_at="2026-08-04",
            shares=344_645_965,
            shares_source="yahoo_quote",
            shares_refreshed_at="2026-07-02",
            price=3.85,
            today="2026-08-04",
            raw_closes=[0.26, 3.85],
            established_record={"status": "approved", "shares": 344_645_965},
        )
        self.assertEqual(selected, direct_cap)

    def test_legacy_calculated_cap_is_not_trusted_without_provenance(self):
        self.assertEqual(
            select_market_cap(
                market_cap=1_326_886_932,
                market_cap_source="",
                market_cap_refreshed_at="2026-08-03",
                shares=0,
                shares_source="",
                shares_refreshed_at="",
                price=3.85,
                today="2026-08-04",
                raw_closes=[0.26, 3.85],
            ),
            0,
        )

    def test_approved_established_record_can_estimate_from_current_price(self):
        selected = select_market_cap(
            market_cap=190_744_055_335,
            market_cap_source="",
            market_cap_refreshed_at="2026-08-03",
            shares=0,
            shares_source="",
            shares_refreshed_at="",
            price=1_397.65,
            today="2026-08-04",
            raw_closes=[1_288.03, 1_397.65],
            established_record={"status": "approved", "shares": 148_089_758},
        )
        self.assertEqual(selected, int(148_089_758 * 1_397.65))

    def test_established_record_cannot_bypass_a_split_guard(self):
        self.assertEqual(
            select_market_cap(
                market_cap=0,
                market_cap_source="",
                market_cap_refreshed_at="",
                shares=0,
                shares_source="",
                shares_refreshed_at="",
                price=3.85,
                today="2026-08-04",
                raw_closes=[0.26, 3.85],
                established_record={"status": "approved", "shares": 344_645_965},
            ),
            0,
        )

    def test_pre_split_direct_cache_is_held_until_refreshed(self):
        self.assertEqual(
            select_market_cap(
                market_cap=89_898_000,
                market_cap_source="yahoo_quote",
                market_cap_refreshed_at="2026-08-03",
                shares=0,
                shares_source="",
                shares_refreshed_at="",
                price=3.85,
                today="2026-08-04",
                raw_closes=[0.26, 3.85],
            ),
            0,
        )

    def test_recent_reverse_split_blocks_share_times_price_fallback(self):
        self.assertTrue(recent_split_like_close_gap([0.26, 3.85]))
        self.assertEqual(
            select_market_cap(
                market_cap=0,
                market_cap_source="",
                market_cap_refreshed_at="",
                shares=344_645_965,
                shares_source="yahoo_quote",
                shares_refreshed_at="2026-08-04",
                price=3.85,
                today="2026-08-04",
                raw_closes=[0.26, 3.85],
            ),
            0,
        )

    def test_share_reconciliation_rejects_pre_split_share_count(self):
        self.assertFalse(
            shares_reconcile_with_market_cap(89_898_000, 344_645_965, 3.85)
        )
        self.assertTrue(
            shares_reconcile_with_market_cap(89_898_000, 23_350_000, 3.85)
        )


if __name__ == "__main__":
    unittest.main()
