#!/usr/bin/env python3
"""
Stock Screener — scans S&P 500, Dow, Nasdaq-100, Russell 2000/3000 for VCP coils.

Replicates the Pine Script VCP/coil detection logic:
  - ATR contraction detection (Strict/Balanced/Loose)
  - MA spread tightness (EMA9 vs EMA21, EMA21 vs HMA40)
  - Price proximity to key EMAs
  - RMV (Relative Magnitude of Volatility) calculation
  - Coil box tracking with breakout detection
  - Inside-box detection (price in bottom 80% of coil range)

Outputs screener.json with results for each screener mode.
"""

import json
import os
import sys
import time
from datetime import datetime, timedelta
import numpy as np
import yfinance as yf

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


# ─── Fetch stock universe ────────────────────────────────────────────────────

def get_sp500_tickers():
    """Fetch S&P 500 tickers from Wikipedia."""
    try:
        import pandas as pd
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        tickers = tables[0]["Symbol"].tolist()
        return [t.replace(".", "-") for t in tickers]
    except Exception as e:
        print(f"  Failed to fetch S&P 500 list: {e}")
        return []


def get_nasdaq100_tickers():
    """Fetch Nasdaq-100 tickers from Wikipedia."""
    try:
        import pandas as pd
        url = "https://en.wikipedia.org/wiki/Nasdaq-100"
        tables = pd.read_html(url)
        for t in tables:
            if "Ticker" in t.columns:
                return t["Ticker"].tolist()
            if "Symbol" in t.columns:
                return t["Symbol"].tolist()
        return []
    except Exception as e:
        print(f"  Failed to fetch Nasdaq-100 list: {e}")
        return []


def get_russell_tickers():
    """Fetch Russell 2000/3000 tickers via FMP API if available."""
    fmp_key = os.environ.get("FMP_API_KEY", "")
    if not fmp_key or not HAS_REQUESTS:
        return []
    try:
        # FMP index constituents endpoint
        url = f"https://financialmodelingprep.com/api/v3/russell2000_constituent?apikey={fmp_key}"
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            tickers = [d["symbol"] for d in data if "symbol" in d]
            print(f"  Fetched {len(tickers)} Russell 2000 tickers from FMP")
            return tickers
    except Exception as e:
        print(f"  Failed to fetch Russell tickers: {e}")
    return []


def get_dow_tickers():
    """Dow Jones 30 tickers."""
    return [
        "AAPL", "AMGN", "AMZN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS",
        "GS", "HD", "HON", "IBM", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK",
        "MSFT", "NKE", "NVDA", "PG", "SHW", "TRV", "UNH", "V", "VZ", "WMT"
    ]


# ─── Technical helpers ────────────────────────────────────────────────────────

def compute_ema(closes, period):
    if len(closes) < period:
        return [None] * len(closes)
    mult = 2 / (period + 1)
    out = [None] * len(closes)
    out[period - 1] = np.mean(closes[:period])
    for i in range(period, len(closes)):
        out[i] = (closes[i] - out[i-1]) * mult + out[i-1]
    return out


def compute_sma(closes, period):
    out = [None] * len(closes)
    for i in range(period - 1, len(closes)):
        out[i] = np.mean(closes[i - period + 1:i + 1])
    return out


def compute_wma(values, period):
    out = [None] * len(values)
    for i in range(period - 1, len(values)):
        window = values[i - period + 1:i + 1]
        if None in window:
            continue
        weights = list(range(1, period + 1))
        out[i] = sum(w * v for w, v in zip(weights, window)) / sum(weights)
    return out


def compute_hma(closes, period):
    """Hull Moving Average."""
    half = compute_wma(closes, period // 2)
    full = compute_wma(closes, period)
    diff = [None] * len(closes)
    for i in range(len(closes)):
        if half[i] is not None and full[i] is not None:
            diff[i] = 2 * half[i] - full[i]
    sqrt_period = int(np.sqrt(period))
    return compute_wma(diff, sqrt_period)


def compute_atr_series(highs, lows, closes, period=14):
    n = len(closes)
    out = [None] * n
    trs = []
    for i in range(1, n):
        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
        trs.append(tr)
        if len(trs) <= period:
            out[i] = np.mean(trs)
        else:
            out[i] = (out[i-1] * (period - 1) + tr) / period
    return out


def compute_atr_avg(atr_series, period=50):
    """SMA of ATR series."""
    return compute_sma(atr_series, period)


# ─── VCP Coil Detection ──────────────────────────────────────────────────────

def scan_ticker(closes, highs, lows, opens, volumes):
    """
    Run VCP coil detection on a single ticker.
    Returns dict with booleans for each screener condition.
    """
    n = len(closes)
    if n < 210:  # Need enough data for SMA200
        return None

    # Moving averages
    ema3 = compute_ema(closes, 3)
    ema5 = compute_ema(closes, 5)
    ema9 = compute_ema(closes, 9)
    ema21 = compute_ema(closes, 21)
    sma10 = compute_sma(closes, 10)
    sma40 = compute_sma(closes, 40)
    sma50 = compute_sma(closes, 50)
    sma200 = compute_sma(closes, 200)
    hma40 = compute_hma(closes, 40)

    # ATR
    atr_series = compute_atr_series(highs, lows, closes, 14)
    atr_avg = compute_atr_avg(atr_series, 50)

    # Current bar values (last bar)
    i = n - 1
    if any(v is None for v in [ema9[i], ema21[i], sma50[i], sma200[i], hma40[i], atr_series[i], atr_avg[i]]):
        return None

    close = closes[i]
    high = highs[i]
    low = lows[i]
    opn = opens[i]
    atr = atr_series[i]
    atr_a = atr_avg[i]

    body_range = abs(close - opn)
    candle_range = abs(high - low)
    atr_buffer = 0.5 * atr

    # ATR contraction
    atr_contracted_str = (body_range < atr_a * 0.6) and (candle_range < atr_a * 0.85)
    atr_contracted_bal = candle_range < atr_a * 0.8
    atr_contracted_loo = body_range < atr_a * 1.5

    # MA spread
    spread_short = abs(ema9[i] - ema21[i]) / atr * 100 if atr > 0 else 999
    spread_long = abs(ema21[i] - hma40[i]) / atr * 100 if atr > 0 else 999
    ma_tight = spread_short < 50  # Balanced mode

    # Price proximity
    price_near_ema9 = abs(close - ema9[i]) <= atr_buffer
    price_near_ema21 = abs(close - ema21[i]) <= atr_buffer

    # RMV Calc #1
    atr3 = compute_atr_series(highs, lows, closes, 3)
    atr5 = compute_atr_series(highs, lows, closes, 5)
    atr8 = compute_atr_series(highs, lows, closes, 8)
    if atr3[i] is not None and atr5[i] is not None and atr8[i] is not None:
        short_avg = (atr3[i] + atr5[i] + atr8[i]) / 3
        # Look back 30 bars for highest/lowest
        short_avgs = []
        for j in range(max(0, i - 29), i + 1):
            if atr3[j] is not None and atr5[j] is not None and atr8[j] is not None:
                short_avgs.append((atr3[j] + atr5[j] + atr8[j]) / 3)
        if short_avgs:
            h_sa = max(short_avgs)
            l_sa = min(short_avgs)
            rmv = (short_avg - l_sa) / max(h_sa - l_sa, 0.001) * 100
        else:
            rmv = 50
    else:
        rmv = 50

    # RMV Calc #2 (alt)
    bar_ranges = [highs[j] - lows[j] for j in range(n)]
    if i >= 22:
        prev_ranges = bar_ranges[i-21:i]
        r_min = min(prev_ranges)
        r_max = max(prev_ranges)
        den = max(r_max - r_min, 1e-6)
        rmv_alt = max(0, min(100, 100 * (bar_ranges[i] - r_min) / den))
    else:
        rmv_alt = 50

    # ─── Coil conditions (Balanced mode — also includes strict) ───
    strict_coil = (atr_contracted_str
        and ((ema21[i] > sma50[i]) or (sma50[i] > sma200[i]))
        and (rmv < 21 or rmv_alt < 10)
        and (price_near_ema9 or price_near_ema21)
        and ((spread_short < 50 and spread_long < 50) or close > sma50[i] or ema5[i] > ema21[i]))

    balanced_coil = (atr_contracted_bal
        and ((ema21[i] > sma50[i]) or (sma50[i] > sma200[i]))
        and (rmv < 25 or rmv_alt < 20)
        and ma_tight
        and (price_near_ema9 or price_near_ema21))

    coil_detected = balanced_coil or strict_coil

    # ─── Coil box tracking (simplified — look back for recent coil start) ───
    # Find the most recent coil box by scanning backwards
    coil_high = None
    coil_low = None
    coil_active = False

    for j in range(i, max(i - 10, 0), -1):
        if j < 1:
            break
        bj = abs(closes[j] - opens[j])
        cj = abs(highs[j] - lows[j])
        aa = atr_avg[j] if atr_avg[j] is not None else atr_a

        contracted = cj < aa * 0.8
        e9 = ema9[j] if ema9[j] is not None else 0
        e21 = ema21[j] if ema21[j] is not None else 0
        s50 = sma50[j] if sma50[j] is not None else 0
        s200 = sma200[j] if sma200[j] is not None else 0
        atr_j = atr_series[j] if atr_series[j] is not None else atr

        sp = abs(e9 - e21) / atr_j * 100 if atr_j > 0 else 999
        near9 = abs(closes[j] - e9) <= 0.5 * atr_j
        near21 = abs(closes[j] - e21) <= 0.5 * atr_j
        rmv_ok = True  # simplified for lookback

        bal_j = contracted and ((e21 > s50) or (s50 > s200)) and sp < 50 and (near9 or near21)
        if bal_j:
            if coil_high is None:
                coil_high = highs[j]
                coil_low = lows[j]
            else:
                coil_high = max(coil_high, highs[j])
                coil_low = min(coil_low, lows[j])
            coil_active = True
        elif coil_active:
            break  # End of contiguous coil region

    # ─── Breakout detection (daily close > coil high, no 15m needed) ───
    breakout = False
    if coil_active and coil_high is not None:
        breakout = (close > coil_high) and (close > sma50[i]) and (close > sma200[i])

    # ─── Inside box detection ───
    inside_box = False
    if coil_active and coil_high is not None and coil_low is not None:
        box_80 = coil_low + 0.80 * (coil_high - coil_low)
        open_in = (opn >= coil_low) and (opn <= coil_high)
        close_in_80 = (close >= coil_low) and (close <= box_80)
        inside_box = open_in and close_in_80

    # ─── Filter conditions ───
    above_50sma = close > sma50[i]
    above_200sma = close > sma200[i]
    above_21ema = close > ema21[i]

    return {
        "coil": bool(coil_detected),
        "breakout": bool(breakout),
        "inside": bool(inside_box),
        "a50": bool(above_50sma),
        "a200": bool(above_200sma),
        "a21": bool(above_21ema),
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Stock Screener — VCP Coil Detection")
    print("=" * 60)

    # Gather universe
    print("\nStep 1: Building stock universe...")
    sp500 = get_sp500_tickers()
    print(f"  S&P 500: {len(sp500)} tickers")
    nasdaq = get_nasdaq100_tickers()
    print(f"  Nasdaq-100: {len(nasdaq)} tickers")
    dow = get_dow_tickers()
    print(f"  Dow 30: {len(dow)} tickers")
    russell = get_russell_tickers()
    print(f"  Russell: {len(russell)} tickers")

    all_tickers = sorted(set(sp500 + nasdaq + dow + russell))
    print(f"\n  Total unique tickers: {len(all_tickers)}")

    if not all_tickers:
        print("ERROR: No tickers found")
        sys.exit(1)

    # Download data
    print(f"\nStep 2: Downloading price data for {len(all_tickers)} tickers...")
    end = datetime.now() + timedelta(days=1)
    start = end - timedelta(days=365)

    raw = yf.download(
        all_tickers,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        group_by="ticker",
        auto_adjust=True,
        threads=True,
    )

    if raw.empty:
        print("ERROR: No data returned")
        sys.exit(1)

    def get_ohlcv(ticker):
        try:
            if len(all_tickers) == 1:
                df = raw.dropna(subset=["Close"])
            else:
                df = raw[ticker].dropna(subset=["Close"])
            if len(df) < 210:
                return None
            return {
                "c": df["Close"].values.tolist(),
                "h": df["High"].values.tolist(),
                "l": df["Low"].values.tolist(),
                "o": df["Open"].values.tolist(),
                "v": df["Volume"].values.tolist(),
            }
        except Exception:
            return None

    # Scan
    print(f"\nStep 3: Scanning {len(all_tickers)} tickers for VCP coils...")
    results = {"coil": [], "breakout": [], "inside": []}
    scanned = 0
    for i, tk in enumerate(all_tickers):
        if (i + 1) % 200 == 0 or i == 0:
            print(f"  {i+1}/{len(all_tickers)}...")
        data = get_ohlcv(tk)
        if data is None:
            continue
        scanned += 1
        scan = scan_ticker(data["c"], data["h"], data["l"], data["o"], data["v"])
        if scan is None:
            continue

        entry = {"t": tk, "a50": scan["a50"], "a200": scan["a200"], "a21": scan["a21"]}

        if scan["coil"]:
            results["coil"].append(entry)
        if scan["breakout"]:
            results["breakout"].append(entry)
        if scan["inside"]:
            results["inside"].append(entry)

    print(f"\nResults:")
    print(f"  Scanned: {scanned}")
    print(f"  Coil Active: {len(results['coil'])}")
    print(f"  Coil Breakout: {len(results['breakout'])}")
    print(f"  Inside Box: {len(results['inside'])}")

    output = {
        "results": results,
        "meta": {
            "updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
            "scanned": scanned,
            "universe": len(all_tickers),
        }
    }

    with open("screener.json", "w") as f:
        json.dump(output, f, separators=(",", ":"))

    print(f"\nWritten screener.json")


if __name__ == "__main__":
    main()
