import sys
import types
import unittest

if "yfinance" not in sys.modules:
    sys.modules["yfinance"] = types.SimpleNamespace(Ticker=type("Ticker", (), {}))

from restore_established_universe import prepare_restoration


class RestoreEstablishedUniverseTests(unittest.TestCase):
    def test_restores_only_trusted_records_that_meet_current_floor(self):
        seed_entries = [
            {"t": "SNDK", "mc": 190_744_055_335, "p": 1288.03, "th": "Computer Hardware"},
            {"t": "MVIS", "mc": 1_326_886_932, "p": 3.85, "th": "Scientific & Technical Instruments"},
            {"t": "AIP", "mc": 1_390_238_376, "p": 30.11, "th": "Semiconductors"},
        ]
        prepared = prepare_restoration(
            seed_entries,
            current_entries=[],
            seed_baselines={ticker: {"_pc": 1} for ticker in ("SNDK", "MVIS", "AIP")},
            industries={
                "SNDK": "Computer Hardware",
                "MVIS": "Scientific & Technical Instruments",
                "AIP": "Semiconductors",
            },
        )

        self.assertEqual([row["t"] for row in prepared["entries"]], ["SNDK"])
        self.assertIn("SNDK", prepared["registry"])
        self.assertNotIn("MVIS", prepared["registry"])
        self.assertEqual(prepared["skipped_floor"], ["AIP"])
        self.assertEqual(prepared["registry"]["SNDK"]["status"], "approved")


if __name__ == "__main__":
    unittest.main()
