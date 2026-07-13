import unittest
from unittest.mock import patch

import fetch_press_releases as press_releases


class PressReleaseTests(unittest.TestCase):
    @patch("fetch_press_releases.yf.Ticker")
    def test_fetches_only_thirty_press_releases(self, ticker_factory):
        ticker_factory.return_value.get_news.return_value = [
            {"id": "release", "title": "Company update"}
        ]

        ticker, items, error = press_releases.fetch_ticker("BE", 30, 1, 0)

        ticker_factory.assert_called_once_with("BE")
        ticker_factory.return_value.get_news.assert_called_once_with(
            count=30,
            tab="press releases",
        )
        self.assertEqual(ticker, "BE")
        self.assertEqual(len(items), 1)
        self.assertIsNone(error)

    def test_normalizes_nested_yfinance_item(self):
        item = {
            "id": "outer-id",
            "content": {
                "id": "release-id",
                "title": " Example &amp; Company Announces Results ",
                "summary": "<p>Quarterly   update.</p>",
                "pubDate": "2026-07-13T09:00:00Z",
                "provider": {"displayName": "GlobeNewswire"},
                "clickThroughUrl": {"url": "https://example.com/release"},
            },
        }

        result = press_releases.normalize_release(item)

        self.assertEqual(result["id"], "release-id")
        self.assertEqual(result["title"], "Example & Company Announces Results")
        self.assertEqual(result["summary"], "Quarterly update.")
        self.assertEqual(result["provider"], "GlobeNewswire")
        self.assertEqual(result["url"], "https://example.com/release")

    def test_deduplicates_sorts_and_caps_results(self):
        items = [
            {"id": "older", "title": "Older", "pubDate": "2026-07-11T09:00:00Z"},
            {"id": "newer", "title": "Newer", "pubDate": "2026-07-13T09:00:00Z"},
            {"id": "newer", "title": "Duplicate", "pubDate": "2026-07-12T09:00:00Z"},
        ]

        result = press_releases.normalize_releases(items, count=1)

        self.assertEqual([row["id"] for row in result], ["newer"])

    def test_shard_key_uses_first_ticker_character(self):
        self.assertEqual(press_releases.shard_key("aapl"), "A")
        self.assertEqual(press_releases.shard_key("1TEST"), "1")
        self.assertEqual(press_releases.shard_key("-"), "_")


if __name__ == "__main__":
    unittest.main()
