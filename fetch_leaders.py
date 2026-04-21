#!/usr/bin/env python3
"""
Liquid Leaders — Stock-level relative strength tracker.
Computes VARS (daily + weekly) for ~1100 individual stocks vs SPY.
Outputs leaders.json (current snapshot) and leaders_history.json (rolling 30-day history).
"""

import json, os, sys, time, io
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta, date
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# Use curl_cffi to bypass Yahoo Finance rate limiting (impersonates Chrome)
try:
    from curl_cffi import requests as cffi_requests
    _session = cffi_requests.Session(impersonate="chrome")
    yf.Ticker._session = _session
except ImportError:
    _session = None

# Import shared functions from fetch_data.py
from fetch_data import ETF_INFO, compute_atr, compute_atr_series, percentrank_inc

LOOKBACK = 50      # daily: compute deltas for last 50 bars
MA_LENGTH = 20     # daily: SMA of deltas (smoothing)
LOOKBACK_W = 20    # weekly: compute deltas for last 20 weeks
MA_LENGTH_W = 8    # weekly: SMA of deltas (smoothing)
ATR_PERIOD = 14
MAX_HISTORY_DAYS = 30

# Extra tickers from CSV not in any ETF holding
CSV_EXTRAS = [
    "ACN","AEHR","ALAB","APO","ARES","ARM","AXTI","BIRD","BP","BRK-B","CAR",
    "GOOG","LNG","NOK","NVO","OWL","RDDT","RIG","SMCI","SNAP","SNDK","STX","WDC",
]

# Yahoo Finance industry → our theme name mapping
INDUSTRY_TO_THEME = {
    # Tech
    "Semiconductors": "Semiconductors",
    "Semiconductor Equipment & Materials": "Semiconductors",
    "Electronic Components": "Semiconductors",
    "Software—Application": "Software",
    "Software - Application": "Software",
    "Software—Infrastructure": "Software",
    "Software - Infrastructure": "Software",
    "Information Technology Services": "Software",
    "Computer Hardware": "Software",
    "Communication Equipment": "Telecom",
    "Consumer Electronics": "Software",
    "Scientific & Technical Instruments": "Semiconductors",
    "Electronic Gaming & Multimedia": "Esports & Gaming",
    # Aerospace/Defense
    "Aerospace & Defense": "Aerospace & Defense",
    # Healthcare
    "Biotechnology": "Biotechnology",
    "Drug Manufacturers—General": "Pharmaceuticals",
    "Drug Manufacturers - General": "Pharmaceuticals",
    "Drug Manufacturers—Specialty & Generic": "Pharmaceuticals",
    "Drug Manufacturers - Specialty & Generic": "Pharmaceuticals",
    "Medical Devices": "Healthcare Equipment",
    "Medical Instruments & Supplies": "Healthcare Equipment",
    "Diagnostics & Research": "Healthcare",
    "Healthcare Plans": "Healthcare Services",
    "Health Information Services": "Healthcare Services",
    "Medical Care Facilities": "Medical/Nursing Services",
    "Medical Distribution": "Healthcare Services",
    "Pharmaceutical Retailers": "Healthcare Services",
    # Financials
    "Banks—Diversified": "Banks",
    "Banks - Diversified": "Banks",
    "Banks—Regional": "Regional Banks",
    "Banks - Regional": "Regional Banks",
    "Capital Markets": "Capital Markets",
    "Asset Management": "Capital Markets",
    "Financial Data & Stock Exchanges": "Capital Markets",
    "Credit Services": "Digital Payments",
    "Insurance—Brokers": "Insurance",
    "Insurance - Brokers": "Insurance",
    "Insurance—Diversified": "Insurance",
    "Insurance - Diversified": "Insurance",
    "Insurance—Life": "Insurance",
    "Insurance - Life": "Insurance",
    "Insurance—Property & Casualty": "Insurance",
    "Insurance - Property & Casualty": "Insurance",
    "Insurance—Reinsurance": "Insurance",
    "Insurance - Reinsurance": "Insurance",
    "Insurance—Specialty": "Insurance",
    "Insurance - Specialty": "Insurance",
    "Mortgage Finance": "Financials",
    "Financial Conglomerates": "Financials",
    # Consumer
    "Internet Retail": "Online Retail",
    "Discount Stores": "Consumer Staples",
    "Grocery Stores": "Consumer Staples",
    "Department Stores": "Retail",
    "Specialty Retail": "Retail",
    "Footwear & Accessories": "Consumer Discretionary",
    "Apparel Retail": "Retail",
    "Apparel Manufacturing": "Consumer Discretionary",
    "Luxury Goods": "Consumer Discretionary",
    "Furnishings, Fixtures & Appliances": "Consumer Discretionary",
    "Household & Personal Products": "Consumer Staples",
    "Personal Services": "Consumer Discretionary",
    "Restaurants": "Consumer Discretionary",
    "Lodging": "Leisure & Ent",
    "Resorts & Casinos": "Casinos & Gaming",
    "Gambling": "Casinos & Gaming",
    "Travel Services": "Airlines & Travel",
    "Leisure": "Leisure & Ent",
    "Recreational Vehicles": "Consumer Discretionary",
    "Auto Manufacturers": "Consumer Discretionary",
    "Auto Parts": "Consumer Discretionary",
    "Auto & Truck Dealerships": "Consumer Discretionary",
    # Food/Beverage
    "Beverages—Brewers": "Food & Beverage",
    "Beverages - Brewers": "Food & Beverage",
    "Beverages—Non-Alcoholic": "Food & Beverage",
    "Beverages - Non-Alcoholic": "Food & Beverage",
    "Beverages—Wineries & Distilleries": "Food & Beverage",
    "Beverages - Wineries & Distilleries": "Food & Beverage",
    "Confectioners": "Food & Beverage",
    "Packaged Foods": "Food & Beverage",
    "Farm Products": "Food & Beverage",
    "Food Distribution": "Food & Beverage",
    "Tobacco": "Consumer Staples",
    "Agricultural Inputs": "Chemicals (Agricultural)",
    # Energy
    "Oil & Gas Drilling": "Oil & Gas Equipment",
    "Oil & Gas E&P": "Oil Refining/Exploration",
    "Oil & Gas Equipment & Services": "Oil & Gas Equipment",
    "Oil & Gas Integrated": "Oil Refining/Exploration",
    "Oil & Gas Midstream": "Energy Infrastructure",
    "Oil & Gas Refining & Marketing": "Oil Refining/Exploration",
    "Thermal Coal": "Metals & Mining",
    "Coking Coal": "Metals & Mining",
    "Uranium": "Uranium / Nuclear",
    "Solar": "Solar Energy",
    # Industrial
    "Conglomerates": "Industrials",
    "Industrial Distribution": "Industrials",
    "Engineering & Construction": "Infrastructure Dev",
    "Farm & Heavy Construction Machinery": "Industrials",
    "Building Products & Equipment": "Home Construction",
    "Building Materials": "Basic Materials",
    "Residential Construction": "Home Construction",
    "Specialty Industrial Machinery": "Industrials",
    "Specialty Business Services": "Industrials",
    "Tools & Accessories": "Industrials",
    "Electrical Equipment & Parts": "Industrials",
    "Pollution & Treatment Controls": "Industrials",
    "Waste Management": "Industrials",
    "Rental & Leasing Services": "Industrials",
    "Staffing & Employment Services": "Industrials",
    "Consulting Services": "Industrials",
    "Education & Training Services": "Industrials",
    "Security & Protection Services": "Industrials",
    "Business Equipment & Supplies": "Industrials",
    "Metal Fabrication": "Metals & Mining",
    # Transport
    "Airlines": "Airlines & Travel",
    "Trucking": "Transportation",
    "Railroads": "Transportation",
    "Marine Shipping": "Maritime & Shipping",
    "Integrated Freight & Logistics": "Transportation",
    # Materials
    "Specialty Chemicals": "Chemicals (Specialty)",
    "Chemicals": "Chemicals (Specialty)",
    "Aluminum": "Metals & Mining",
    "Copper": "Copper Miners",
    "Gold": "Gold Miners",
    "Silver": "Silver Miners",
    "Other Industrial Metals & Mining": "Metals & Mining",
    "Other Precious Metals & Mining": "Silver Miners",
    "Steel": "Steel",
    "Paper & Paper Products": "Timber & Forestry",
    "Lumber & Wood Production": "Timber & Forestry",
    "Packaging & Containers": "Basic Materials",
    # Telecom/Media
    "Telecom Services": "Telecom",
    "Internet Content & Information": "Telecom",
    "Entertainment": "Telecom",
    "Broadcasting": "Telecom",
    "Publishing": "Telecom",
    "Advertising Agencies": "Telecom",
    # Real Estate / REITs
    "Real Estate—Development": "US Real Estate",
    "Real Estate - Development": "US Real Estate",
    "Real Estate—Diversified": "US Real Estate",
    "Real Estate - Diversified": "US Real Estate",
    "Real Estate Services": "US Real Estate",
    "REIT—Diversified": "US Real Estate",
    "REIT - Diversified": "US Real Estate",
    "REIT—Healthcare Facilities": "US Real Estate",
    "REIT - Healthcare Facilities": "US Real Estate",
    "REIT—Hotel & Motel": "US Real Estate",
    "REIT - Hotel & Motel": "US Real Estate",
    "REIT—Industrial": "US Real Estate",
    "REIT - Industrial": "US Real Estate",
    "REIT—Mortgage": "US Real Estate",
    "REIT - Mortgage": "US Real Estate",
    "REIT—Office": "US Real Estate",
    "REIT - Office": "US Real Estate",
    "REIT—Residential": "US Real Estate",
    "REIT - Residential": "US Real Estate",
    "REIT—Retail": "US Real Estate",
    "REIT - Retail": "US Real Estate",
    "REIT—Specialty": "US Real Estate",
    "REIT - Specialty": "US Real Estate",
    # Utilities
    "Utilities—Diversified": "Utilities",
    "Utilities - Diversified": "Utilities",
    "Utilities—Independent Power Producers": "Utilities",
    "Utilities - Independent Power Producers": "Utilities",
    "Utilities—Regulated Electric": "Utilities",
    "Utilities - Regulated Electric": "Utilities",
    "Utilities—Regulated Gas": "Utilities",
    "Utilities - Regulated Gas": "Utilities",
    "Utilities—Regulated Water": "Water Infrastructure",
    "Utilities - Regulated Water": "Water Infrastructure",
    "Utilities—Renewable": "Clean Energy",
    "Utilities - Renewable": "Clean Energy",
}


def map_industry_to_theme(industry):
    """Map Yahoo Finance industry to one of our themes."""
    if not industry:
        return None
    # Direct match
    if industry in INDUSTRY_TO_THEME:
        return INDUSTRY_TO_THEME[industry]
    # Try normalized (handle dash variants)
    normalized = industry.replace("—", " - ").replace("–", " - ")
    if normalized in INDUSTRY_TO_THEME:
        return INDUSTRY_TO_THEME[normalized]
    # Substring fallback
    industry_lower = industry.lower()
    for key, theme in INDUSTRY_TO_THEME.items():
        if key.lower() in industry_lower or industry_lower in key.lower():
            return theme
    return None



def fetch_ishares_holdings(etf_id, name="ETF"):
    """Fetch holdings list from iShares official CSV endpoint."""
    if not HAS_REQUESTS:
        return []
    url = f"https://www.ishares.com/us/products/{etf_id}/ishares-{name.lower()}/1467271812596.ajax?fileType=csv&fileName={name}_holdings&dataType=fund"
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
        resp = requests.get(url, timeout=30, headers=headers)
        if resp.status_code != 200:
            print(f"  iShares {name} HTTP {resp.status_code}")
            return []
        lines = resp.text.splitlines()
        # iShares CSV has metadata header rows before the actual data table
        header_idx = None
        for i, line in enumerate(lines):
            if line.startswith("Ticker,") or line.startswith('"Ticker"'):
                header_idx = i
                break
        if header_idx is None:
            print(f"  iShares {name}: couldn't find Ticker header")
            return []
        import csv as csvlib
        reader = csvlib.reader(io.StringIO("\n".join(lines[header_idx:])))
        header = next(reader)
        try:
            tk_col = header.index("Ticker")
        except ValueError:
            return []
        ac_col = header.index("Asset Class") if "Asset Class" in header else -1

        tickers = []
        for row in reader:
            if len(row) <= tk_col:
                continue
            tk = row[tk_col].strip().replace(".", "-")
            if not tk or tk in ("-", "USD", "Cash"):
                continue
            if ac_col >= 0 and len(row) > ac_col:
                ac = row[ac_col].strip()
                if ac and "Equity" not in ac:
                    continue
            if len(tk) > 6 or "/" in tk:
                continue
            tickers.append(tk)
        return tickers
    except Exception as e:
        print(f"  iShares {name} fetch failed: {e}")
        return []


def get_index_universe():
    """Fetch IWV (Russell 3000) holdings — covers all of SPY, QQQ, DIA, IWM.
    Cached for 7 days. Falls back to cache or empty list on failure.
    """
    cache_file = "leaders_index_universe.json"
    cache = {"refreshed": "", "tickers": []}
    if os.path.exists(cache_file):
        try:
            with open(cache_file) as f:
                cache = json.load(f)
        except Exception:
            pass

    try:
        if cache.get("refreshed") and cache.get("tickers"):
            days_old = (date.today() - date.fromisoformat(cache["refreshed"])).days
            if days_old < 7:
                print(f"  Using cached index universe ({len(cache['tickers'])} tickers, {days_old}d old)")
                return cache["tickers"]
    except Exception:
        pass

    print("  Fetching Russell 3000 (IWV) holdings from iShares...")
    tickers = fetch_ishares_holdings("239714", "IWV")
    if not tickers:
        print("  IWV fetch failed, trying IWM (Russell 2000)...")
        tickers = fetch_ishares_holdings("239710", "IWM")
    if not tickers:
        print(f"  All fetches failed, using cached version ({len(cache.get('tickers', []))} tickers)")
        return cache.get("tickers", [])

    print(f"  Fetched {len(tickers)} tickers from iShares")
    cache = {"refreshed": date.today().isoformat(), "tickers": tickers}
    with open(cache_file, "w") as f:
        json.dump(cache, f, separators=(",", ":"))
    return tickers


def build_universe():
    """Build stock universe + collect all ETF assignments per stock.
    Universe sources:
      1. All stocks from ETF_INFO holdings (themed)
      2. CSV_EXTRAS (manually added)
      3. IWV (Russell 3000) holdings via iShares — covers SPY/QQQ/DIA/IWM
    Returns sorted stock list and stock_to_etfs dict.
    """
    stocks = set()
    stock_to_etfs = {}  # stock -> [(theme_name, holdings_count), ...]
    for info in ETF_INFO:
        h_str = info.get("h", "")
        name = info.get("n", "")
        if not h_str:
            continue
        holdings = [h.strip() for h in h_str.split(",") if h.strip()]
        count = len(holdings)
        for h in holdings:
            stocks.add(h)
            stock_to_etfs.setdefault(h, []).append((name, count))
    for t in CSV_EXTRAS:
        stocks.add(t)

    # Add Russell 3000 constituents (covers SPY/QQQ/DIA/IWM)
    print("Loading index universe (SPY/QQQ/DIA/IWM via Russell 3000)...")
    index_tickers = get_index_universe()
    added = 0
    for t in index_tickers:
        if t not in stocks:
            stocks.add(t)
            added += 1
    print(f"  Added {added} new tickers from Russell 3000 (total universe before mcap filter: {len(stocks)})")

    # Diagnostic: check if specific tickers are present in the starting universe
    DEBUG_TICKERS_UNIV = {"RIG"}
    for tk in DEBUG_TICKERS_UNIV:
        if tk in stocks:
            etfs_for_tk = stock_to_etfs.get(tk, [])
            print(f"  [DEBUG] {tk}: in starting universe (ETF baskets: {[n for n,_ in etfs_for_tk] or 'Russell3000/CSV only'})")
        else:
            print(f"  [DEBUG] {tk}: NOT in starting universe — will never appear in rankings")

    return sorted(stocks), stock_to_etfs


# Niche themes Yahoo Finance can't properly classify — these baskets take priority over Yahoo industry
PROTECTED_BASKETS = {
    "Magnificent Seven",
    "Crypto Miners / Data Centers",
    "AI & Power Infra",
    "Agentic AI",
    "Quantum",
    "Photonics",
    "LiDAR",
    "HVAC / Cooling",
    "Drones",
    "Space",
    "Hydrogen",
    "Bitcoin",
    "Blockchain",
    "Cybersecurity",
    "Cloud Computing",
    "Cannabis",
    "Genomics",
    "China Large-Cap",
    "Emerging Markets",
    "Lithium & Battery",
    "Rare Earth Metals",
    "Clean Energy",
    "Solar Energy",
    "Uranium / Nuclear",
    "Fintech Innovation",
    "Digital Payments",
    "Esports & Gaming",
    "Casinos & Gaming",
}


def resolve_theme(ticker, stock_to_etfs, industry_cache):
    """
    Resolve theme for a stock with this priority:
    1. If stock is in any PROTECTED_BASKETS ETF, use the most specific (fewest holdings)
    2. Else if Yahoo industry maps to a theme, use Yahoo
    3. Else use the ETF with fewest holdings (any theme)
    4. Else "General"
    """
    etfs = stock_to_etfs.get(ticker, [])
    # Step 1: protected baskets win
    protected = [(n, c) for n, c in etfs if n in PROTECTED_BASKETS]
    if protected:
        protected.sort(key=lambda x: x[1])
        return protected[0][0]
    # Step 2: Yahoo industry
    industry = industry_cache.get(ticker, "")
    yahoo_theme = map_industry_to_theme(industry)
    if yahoo_theme:
        return yahoo_theme
    # Step 3: any ETF (fewest holdings)
    if etfs:
        etfs_sorted = sorted(etfs, key=lambda x: x[1])
        return etfs_sorted[0][0]
    # Step 4: general
    return "General"


def compute_ema_value(closes, period):
    """Compute the final EMA value of a closes array using standard EMA formula."""
    n = len(closes)
    if n == 0:
        return 0.0
    if n < period:
        return float(np.mean(closes))
    multiplier = 2.0 / (period + 1)
    ema = float(np.mean(closes[:period]))  # Seed with SMA
    for i in range(period, n):
        ema = (closes[i] - ema) * multiplier + ema
    return ema


def compute_trend_zone(c, h, l, spy_closes, spy_ts_map, df_index):
    """Compute the smooth_trend score from the Pine Script 'Custom CMI' indicator.

    Returns a string zone name:
      "bull_strong"  : smooth_trend > 30
      "bull_light"   : 10 < smooth_trend <= 30
      "neutral"      : -10 <= smooth_trend <= 10
      "bear_light"   : -30 <= smooth_trend < -10
      "bear_strong"  : smooth_trend < -30

    trend_score is composed of:
      25% RSI(14) momentum score = (RSI - 50) * 2
      35% MA score: avg of 3 EMA divergences:
           - daily EMA(5) vs EMA(20)
           - weekly EMA(3) vs EMA(10)   → approximated with 15/50 daily bars
           - intraday EMA(10) vs EMA(21) → approximated with 10/21 daily bars
      20% Bollinger Band(20, 2) position score = (bbpos - 50) * 2
      20% Relative strength vs SPY (1D change difference * 10)

    Then smooth_trend = EMA(trend_score, 5).
    """
    c_arr = np.asarray([x for x in c if x is not None], dtype=float)
    h_arr = np.asarray([x for x in h if x is not None], dtype=float)
    l_arr = np.asarray([x for x in l if x is not None], dtype=float)
    n = len(c_arr)
    if n < 55:  # need at least 50 bars for longer EMAs + 5-bar smoothing
        return "neutral"

    def ema(arr, span):
        if len(arr) < span:
            return None
        k = 2.0 / (span + 1)
        e = float(arr[0])
        for v in arr[1:]:
            e = float(v) * k + e * (1 - k)
        return e

    def ema_series(arr, span):
        if len(arr) < span:
            return None
        k = 2.0 / (span + 1)
        out = [float(arr[0])]
        for v in arr[1:]:
            out.append(float(v) * k + out[-1] * (1 - k))
        return np.array(out)

    def rsi(arr, length=14):
        if len(arr) < length + 1:
            return None
        deltas = np.diff(arr)
        gains = np.maximum(deltas, 0)
        losses = np.maximum(-deltas, 0)
        # Wilder smoothing
        avg_gain = np.mean(gains[:length])
        avg_loss = np.mean(losses[:length])
        for i in range(length, len(deltas)):
            avg_gain = (avg_gain * (length - 1) + gains[i]) / length
            avg_loss = (avg_loss * (length - 1) + losses[i]) / length
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - 100.0 / (1.0 + rs)

    def rsi_series(arr, length=14):
        """RSI at every bar (Wilder smoothing)."""
        if len(arr) < length + 1:
            return None
        deltas = np.diff(arr)
        gains = np.maximum(deltas, 0)
        losses = np.maximum(-deltas, 0)
        avg_gain = np.mean(gains[:length])
        avg_loss = np.mean(losses[:length])
        rsi_vals = [50.0] * (length)  # pad until we have enough data
        rs = avg_gain / avg_loss if avg_loss > 0 else float('inf')
        rsi_vals.append(100.0 - 100.0 / (1.0 + rs) if avg_loss > 0 else 100.0)
        for i in range(length, len(deltas)):
            avg_gain = (avg_gain * (length - 1) + gains[i]) / length
            avg_loss = (avg_loss * (length - 1) + losses[i]) / length
            if avg_loss == 0:
                rsi_vals.append(100.0)
            else:
                rs = avg_gain / avg_loss
                rsi_vals.append(100.0 - 100.0 / (1.0 + rs))
        return np.array(rsi_vals)

    # Compute component scores across the last N bars so we can then smooth via EMA(5).
    # We only need the last ~10 bars of trend_score for EMA(5) smoothing to be stable.
    WINDOW = 10
    if n < 55:
        return "neutral"

    # 1) RSI score series
    rsi_s = rsi_series(c_arr, 14)
    if rsi_s is None or len(rsi_s) < WINDOW:
        return "neutral"

    # 2) MA score = avg of 3 EMA divergence scores, per bar
    # We'll compute EMAs as series
    ema_f_intra_s = ema_series(c_arr, 10)
    ema_s_intra_s = ema_series(c_arr, 21)
    ema_f_day_s   = ema_series(c_arr, 5)
    ema_s_day_s   = ema_series(c_arr, 20)
    # Weekly EMA(3) and EMA(10) on weekly bars ≈ EMA(15) and EMA(50) on daily bars
    ema_f_wk_s    = ema_series(c_arr, 15)
    ema_s_wk_s    = ema_series(c_arr, 50)

    if ema_s_wk_s is None or ema_s_day_s is None or ema_s_intra_s is None:
        return "neutral"

    # 3) Bollinger position series
    bb_len = 20
    bb_mult = 2.0
    bb_score_s = np.zeros(n)
    for i in range(bb_len - 1, n):
        window = c_arr[i - bb_len + 1:i + 1]
        basis = np.mean(window)
        std = np.std(window, ddof=0)
        upper = basis + bb_mult * std
        lower = basis - bb_mult * std
        if upper - lower > 0:
            pos = (c_arr[i] - lower) / (upper - lower) * 100
            bb_score_s[i] = (pos - 50) * 2
        else:
            bb_score_s[i] = 0

    # 4) RS vs SPY score series: (stock daily change - SPY daily change) * 10
    # Align with SPY bars
    rs_score_s = np.zeros(n)
    for i, ts in enumerate(df_index):
        if i < 1:
            continue
        stock_chg = (c[i] / c[i - 1] - 1) * 100 if c[i] is not None and c[i - 1] else 0
        spy_chg = 0
        if ts in spy_ts_map:
            si = spy_ts_map[ts]
            if si > 0 and spy_closes[si] and spy_closes[si - 1]:
                spy_chg = (spy_closes[si] / spy_closes[si - 1] - 1) * 100
        rs_score_s[i] = max(-100, min(100, (stock_chg - spy_chg) * 10))

    # Compute trend_score for each of the last WINDOW bars
    trend_scores = []
    for offset in range(WINDOW, 0, -1):
        idx = n - offset
        if idx < 50:
            continue
        # RSI score
        rsi_val = rsi_s[idx] if idx < len(rsi_s) else 50
        rsi_score = (rsi_val - 50) * 2

        # MA scores
        if ema_s_intra_s[idx] > 0:
            md_intra = (ema_f_intra_s[idx] - ema_s_intra_s[idx]) / ema_s_intra_s[idx] * 100
            ma_intra = max(-100, min(100, md_intra * 5))
        else:
            ma_intra = 0
        if ema_s_day_s[idx] > 0:
            md_day = (ema_f_day_s[idx] - ema_s_day_s[idx]) / ema_s_day_s[idx] * 100
            ma_day = max(-100, min(100, md_day * 5))
        else:
            ma_day = 0
        if ema_s_wk_s[idx] > 0:
            md_wk = (ema_f_wk_s[idx] - ema_s_wk_s[idx]) / ema_s_wk_s[idx] * 100
            ma_wk = max(-100, min(100, md_wk * 5))
        else:
            ma_wk = 0
        ma_score = (ma_intra + ma_day + ma_wk) / 3

        # BB score
        bb_score = bb_score_s[idx] if idx < len(bb_score_s) else 0
        # RS vs SPY
        rs_score = rs_score_s[idx] if idx < len(rs_score_s) else 0

        # Composite trend score
        ts_score = (rsi_score * 0.25) + (ma_score * 0.35) + (bb_score * 0.20) + (rs_score * 0.20)
        trend_scores.append(ts_score)

    if not trend_scores:
        return "neutral"

    # Apply EMA(5) smoothing to the series to get smooth_trend
    k = 2.0 / (5 + 1)
    smooth = trend_scores[0]
    for v in trend_scores[1:]:
        smooth = v * k + smooth * (1 - k)

    # Classify into zones
    if smooth > 30:
        return "bull_strong"
    elif smooth > 10:
        return "bull_light"
    elif smooth < -30:
        return "bear_strong"
    elif smooth < -10:
        return "bear_light"
    else:
        return "neutral"


def compute_setup_adjustment(c, h, l, n):
    """Evaluate technical setup criteria and return a score adjustment (in percentile points).

    Gold criteria (+1.5 each met, -2.0 each NOT met):
      G1: 21EMA > 50SMA
      G2: Price > 21EMA
      G3: Price > 50SMA
      G4: 50SMA rising (positive slope)

    Silver criteria (+0.75 each met, -1.0 each NOT met):
      S1: Price > 100SMA
      S2: 50SMA > 100SMA
      S3: 200SMA rising (positive slope)

    Bronze criteria (+0.5 each met, no penalty):
      B1: 9EMA and 21EMA coiled within 0.5x ADR
      B2: Price > 200SMA
    """
    price = float(c[-1])

    # Compute MAs
    ema9 = compute_ema_value(c, 9)
    ema21 = compute_ema_value(c, 21)
    sma50 = float(np.mean(c[-50:])) if n >= 50 else float(np.mean(c))
    sma100 = float(np.mean(c[-100:])) if n >= 100 else None
    sma200 = float(np.mean(c[-200:])) if n >= 200 else None

    # 50SMA slope: compare current vs 5 bars ago
    sma50_prev = float(np.mean(c[-55:-5])) if n >= 55 else None
    sma50_rising = (sma50 > sma50_prev) if sma50_prev is not None else None

    # 200SMA slope: compare current vs 5 bars ago
    sma200_prev = float(np.mean(c[-205:-5])) if n >= 205 else None
    sma200_rising = (sma200 > sma200_prev) if sma200_prev is not None else None

    # ADR (Average Daily Range) over last 14 bars
    adr = float(np.mean([h[i] - l[i] for i in range(max(0, n - 14), n)])) if n > 0 else 0

    # ─── Evaluate criteria ────────────────────────────────────
    adj = 0.0
    flags = 0  # Bitmask: G1=1, G2=2, G3=4, G4=8, S1=16, S2=32, S3=64, B1=128, B2=256

    # Gold criteria
    GOLD_BONUS = 1.5
    GOLD_PENALTY = -2.0

    # G1: 21EMA > 50SMA
    g1 = ema21 > sma50
    adj += GOLD_BONUS if g1 else GOLD_PENALTY
    if g1: flags |= 1

    # G2: Price > 21EMA
    g2 = price > ema21
    adj += GOLD_BONUS if g2 else GOLD_PENALTY
    if g2: flags |= 2

    # G3: Price > 50SMA
    g3 = price > sma50
    adj += GOLD_BONUS if g3 else GOLD_PENALTY
    if g3: flags |= 4

    # G4: 50SMA rising
    if sma50_rising is not None:
        adj += GOLD_BONUS if sma50_rising else GOLD_PENALTY
        if sma50_rising: flags |= 8
    # If insufficient data, no bonus or penalty

    # Silver criteria
    SILVER_BONUS = 0.75
    SILVER_PENALTY = -1.0

    # S1: Price > 100SMA
    if sma100 is not None:
        s1 = price > sma100
        adj += SILVER_BONUS if s1 else SILVER_PENALTY
        if s1: flags |= 16

    # S2: 50SMA > 100SMA
    if sma100 is not None:
        s2 = sma50 > sma100
        adj += SILVER_BONUS if s2 else SILVER_PENALTY
        if s2: flags |= 32

    # S3: 200SMA rising
    if sma200_rising is not None:
        adj += SILVER_BONUS if sma200_rising else SILVER_PENALTY
        if sma200_rising: flags |= 64

    # Bronze criteria (bonus only, no penalty)
    BRONZE_BONUS = 0.5

    # B1: 9EMA and 21EMA coiled within 0.5x ADR
    if adr > 0:
        coiled = abs(ema9 - ema21) < (0.5 * adr)
        if coiled:
            adj += BRONZE_BONUS
            flags |= 128

    # B2: Price > 200SMA
    if sma200 is not None:
        if price > sma200:
            adj += BRONZE_BONUS
            flags |= 256

    return round(adj, 2), flags


def process_stock(ticker, df, spy_closes, spy_highs, spy_lows, spy_atr_series, spy_ts_map):
    """Process daily metrics for one stock. Returns dict or None.
    Liquidity/delisted/flat-price filters are applied upstream in main().
    """
    if df is None or len(df) < 10:
        return None
    c = df["Close"].values
    h = df["High"].values
    l = df["Low"].values
    v = df["Volume"].values
    n = len(c)

    price = float(c[-1])
    change = (c[-1] / c[-2] - 1) * 100 if n >= 2 else 0
    c5 = (c[-1] / c[-6] - 1) * 100 if n >= 6 else None
    c20 = (c[-1] / c[-21] - 1) * 100 if n >= 21 else None

    # YTD: find the last bar of the previous year (= first bar of current year - 1)
    current_year = date.today().year
    ytd = None
    ytd_base = None  # baseline close (last bar of previous year, or first of current year)
    try:
        for i, ts in enumerate(df.index):
            bar_year = ts.year if hasattr(ts, 'year') else int(str(ts)[:4])
            if bar_year >= current_year:
                if i > 0 and c[i - 1] is not None and c[-1] is not None and c[i - 1] != 0:
                    ytd_base = float(c[i - 1])
                    ytd = (c[-1] / ytd_base - 1) * 100
                elif c[i] is not None and c[-1] is not None and c[i] != 0:
                    # Fallback: data starts in current year (IPO etc.) — use first available bar
                    ytd_base = float(c[i])
                    ytd = (c[-1] / ytd_base - 1) * 100
                break
    except Exception:
        ytd = None
        ytd_base = None

    # ─── Intraday overlay baseline selection ───────────────────────
    # During the overlay, `live` is substituted as "today's price". We want the 5/20-trading-day
    # baselines to span the conventional distance from TODAY, not from c[-1].
    #   • If c[-1] is today's (possibly partial) bar: overlay replaces c[-1] → 5d back = c[-6]
    #   • If c[-1] is yesterday (df not yet updated): overlay adds today via live → 5d back = c[-5]
    try:
        from datetime import timezone as _tz
        _now_utc = datetime.now(_tz.utc)
        _today_et = (_now_utc - timedelta(hours=4)).date()  # EDT approx; workflow gates on ET anyway
        _last_ts = df.index[-1]
        _last_date = _last_ts.date() if hasattr(_last_ts, 'date') else None
        _c1_is_today = (_last_date == _today_et)
    except Exception:
        _c1_is_today = True  # default to "EOD semantics" on error
    _b5_off  = 6  if _c1_is_today else 5
    _b20_off = 21 if _c1_is_today else 20

    # ATR Ext = gainPct / atrPct (matches Pine Script exactly)
    # gainPct = (price - sma50) / sma50 * 100
    # atrPct  = atr / price * 100
    # atrMultiple = gainPct / atrPct = ((price - sma50) / sma50) / (atr / price)
    trs = []
    for i in range(max(1, n - 14), n):
        if c[i] is None or c[i-1] is None or h[i] is None or l[i] is None:
            continue
        tr = max(h[i] - l[i], abs(h[i] - c[i-1]), abs(l[i] - c[i-1]))
        trs.append(tr)
    atr = float(np.mean(trs)) if trs else 0
    valid_c = [x for x in c if x is not None]
    sma50 = np.mean(valid_c[-50:]) if len(valid_c) >= 50 else np.mean(valid_c)
    if atr > 0 and sma50 > 0:
        gain_pct = (c[-1] - sma50) / sma50 * 100
        atr_pct  = atr / c[-1] * 100
        atr_ext  = gain_pct / atr_pct  # can be negative (below SMA)
    else:
        atr_ext = 0
    atr_pct_for_mult = (atr / c[-2] * 100) if n >= 2 and c[-2] != 0 else 1
    atr_mult = abs(change) / atr_pct_for_mult if atr_pct_for_mult > 0 else 0

    vols = [x for x in v if x is not None and x > 0]
    rvol = (vols[-1] / np.mean(vols[:-1][-20:])) if len(vols) > 1 and np.mean(vols[:-1][-20:]) > 0 else None

    # ─── Pine Script RS calculation (4-quarter weighted performance vs SPY) ──
    # Quarter 1 (most recent) = 40% weight, Q2/Q3/Q4 = 20% each
    # rs_stock = 0.4*(px/px[-63]) + 0.2*(px/px[-126]) + 0.2*(px/px[-189]) + 0.2*(px/px[-252])
    # rs_ref   = same for SPY
    # totalRsScore = (rs_stock / rs_ref) * 100
    # Align stock and SPY bars by timestamp
    common = []
    for idx, ts in enumerate(df.index):
        if ts in spy_ts_map and c[idx] is not None and spy_closes[spy_ts_map[ts]] is not None:
            common.append((idx, spy_ts_map[ts]))

    # Need at least 63 common bars for Q1 calculation
    if len(common) < 63:
        _sa, _sf = compute_setup_adjustment(c, h, l, n)
        return {"rv": round(rvol * 100) if rvol else None, "am": round(atr_mult * 100),
                "ax": round(atr_ext * 100), "ch": round(change, 2),
                "c5": round(c5, 2) if c5 is not None else None,
                "c20": round(c20, 2) if c20 is not None else None,
                "ytd": round(ytd, 2) if ytd is not None else None,
                "rs": None, "rf": 0, "ra": 0, "p": round(price, 2), "fr": None, "vs": None,
                "sa": _sa, "sf": _sf, "tz": "neutral",
                # Internal baselines for intraday overlay (stripped before output)
                "_pc": float(c[-2]) if n >= 2 else None,
                "_5b": float(c[-_b5_off]) if n >= _b5_off else None,
                "_20b": float(c[-_b20_off]) if n >= _b20_off else None,
                "_yb": ytd_base}

    # Handle IPOs with <252 bars: use actual bar count (mirrors Pine's "n63/n126/n189/n252" logic)
    avail = len(common)
    n63  = min(63,  avail - 1)
    n126 = min(126, avail - 1)
    n189 = min(189, avail - 1)
    n252 = min(252, avail - 1)

    # Latest stock/SPY closes at common[-1]
    cur_stock_idx, cur_spy_idx = common[-1]
    stock_now = c[cur_stock_idx]
    spy_now   = spy_closes[cur_spy_idx]

    # Lookback stock/SPY closes
    def stock_at(offset):
        idx = common[-1 - offset][0]
        return c[idx]
    def spy_at(offset):
        idx = common[-1 - offset][1]
        return spy_closes[idx]

    perfT63  = stock_now / stock_at(n63)  if stock_at(n63)  else 1
    perfT126 = stock_now / stock_at(n126) if stock_at(n126) else 1
    perfT189 = stock_now / stock_at(n189) if stock_at(n189) else 1
    perfT252 = stock_now / stock_at(n252) if stock_at(n252) else 1

    perfS63  = spy_now / spy_at(n63)  if spy_at(n63)  else 1
    perfS126 = spy_now / spy_at(n126) if spy_at(n126) else 1
    perfS189 = spy_now / spy_at(n189) if spy_at(n189) else 1
    perfS252 = spy_now / spy_at(n252) if spy_at(n252) else 1

    rs_stock = 0.4 * perfT63 + 0.2 * perfT126 + 0.2 * perfT189 + 0.2 * perfT252
    rs_ref   = 0.4 * perfS63 + 0.2 * perfS126 + 0.2 * perfS189 + 0.2 * perfS252
    final_rs = (rs_stock / rs_ref) * 100 if rs_ref > 0 else 100

    # Build a lightweight VARS-like sparkline: rolling 20-bar weighted RS score
    # (for visual trend reference — kept for compatibility with frontend sparklines)
    sma_series = []
    if avail >= 63:
        # Compute totalRsScore at each of the last min(30, avail-63) bars
        span = min(30, avail - 63)
        for k in range(span):
            bar_offset = span - 1 - k  # oldest first → newest last
            idx_end = avail - 1 - bar_offset
            if idx_end < 63:
                continue
            s_idx, sp_idx = common[idx_end]
            s_now = c[s_idx]; sp_now = spy_closes[sp_idx]
            n63_k  = min(63, idx_end)
            n126_k = min(126, idx_end)
            n189_k = min(189, idx_end)
            n252_k = min(252, idx_end)
            s_63 = c[common[idx_end - n63_k][0]]
            s_126 = c[common[idx_end - n126_k][0]]
            s_189 = c[common[idx_end - n189_k][0]]
            s_252 = c[common[idx_end - n252_k][0]]
            p_63 = spy_closes[common[idx_end - n63_k][1]]
            p_126 = spy_closes[common[idx_end - n126_k][1]]
            p_189 = spy_closes[common[idx_end - n189_k][1]]
            p_252 = spy_closes[common[idx_end - n252_k][1]]
            if s_63 and s_126 and s_189 and s_252 and p_63 and p_126 and p_189 and p_252:
                rss = 0.4*(s_now/s_63) + 0.2*(s_now/s_126) + 0.2*(s_now/s_189) + 0.2*(s_now/s_252)
                rsr = 0.4*(sp_now/p_63) + 0.2*(sp_now/p_126) + 0.2*(sp_now/p_189) + 0.2*(sp_now/p_252)
                if rsr > 0:
                    sma_series.append(round((rss/rsr)*100, 2))

    if not sma_series:
        sma_series = [round(final_rs, 2)]

    # Advancing/declining streaks on the smoothed series
    adv_streak = 0
    for j in range(len(sma_series) - 1, 0, -1):
        if sma_series[j] > sma_series[j - 1]: adv_streak += 1
        else: break
    dec_streak = 0
    for j in range(len(sma_series) - 1, 0, -1):
        if sma_series[j] < sma_series[j - 1]: dec_streak += 1
        else: break

    # Setup quality adjustment (applied to RS percentrank in main)
    setup_adj, setup_flags = compute_setup_adjustment(c, h, l, n)

    # Trend zone (from Pine Script CMI "smooth_trend" — used as multiplier on RS)
    trend_zone = compute_trend_zone(c, h, l, spy_closes, spy_ts_map, df.index)

    return {
        "rv": round(rvol * 100) if rvol else None,
        "am": round(atr_mult * 100), "ax": round(atr_ext * 100),
        "ch": round(change, 2),
        "c5": round(c5, 2) if c5 is not None else None,
        "c20": round(c20, 2) if c20 is not None else None,
        "ytd": round(ytd, 2) if ytd is not None else None,
        "rs": None,  # Will be set cross-sectionally in main()
        "rf": dec_streak, "ra": adv_streak,
        "p": round(price, 2), "fr": round(final_rs, 4), "vs": sma_series,
        "sa": setup_adj, "sf": setup_flags, "tz": trend_zone,
        # Internal baselines for intraday overlay (stripped before output)
        "_pc": float(c[-2]) if n >= 2 else None,
        "_5b": float(c[-_b5_off]) if n >= _b5_off else None,
        "_20b": float(c[-_b20_off]) if n >= _b20_off else None,
        "_yb": ytd_base,
    }


def process_stock_weekly(ticker, df_w, spy_w_closes, spy_w_atr_series, spy_w_ts_map):
    """Process weekly metrics using SMA-based Real Relative Strength."""
    null_result = {"w_rv": None, "w_am": None, "w_rs": None, "w_rf": 0, "w_ra": 0, "w_vs": None}
    if df_w is None or len(df_w) < 5:
        return null_result

    c = df_w["Close"].values
    h = df_w["High"].values
    l = df_w["Low"].values
    v = df_w["Volume"].values
    n = len(c)

    atr = compute_atr(h, l, c, ATR_PERIOD)
    week_change = (c[-1] / c[-2] - 1) * 100 if n >= 2 else 0
    atr_pct_w = (atr / c[-2] * 100) if n >= 2 and c[-2] != 0 else 1
    atr_mult = abs(week_change) / atr_pct_w if atr_pct_w > 0 else 0

    vols = [x for x in v if x is not None and x > 0]
    rvol = (vols[-1] / np.mean(vols[:-1][-10:])) if len(vols) > 1 and np.mean(vols[:-1][-10:]) > 0 else None

    common = []
    for idx, ts in enumerate(df_w.index):
        if ts in spy_w_ts_map and c[idx] is not None and spy_w_closes[spy_w_ts_map[ts]] is not None:
            common.append((idx, spy_w_ts_map[ts]))

    if len(common) < LOOKBACK_W:
        return {"w_rv": round(rvol * 100) if rvol else None, "w_am": round(atr_mult * 100),
                "w_rs": None, "w_rf": 0, "w_ra": 0, "w_vs": None, "w_fr": None}

    etf_atr_series = compute_atr_series(h, l, c, ATR_PERIOD)
    extended = common[-(LOOKBACK_W + 1):]

    # Per-week deltas
    deltas = []
    for k in range(1, len(extended)):
        ei_prev, si_prev = extended[k - 1]
        ei, si = extended[k]
        stock_chg = c[ei] - c[ei_prev]
        spy_chg = spy_w_closes[si] - spy_w_closes[si_prev]
        stock_atr = etf_atr_series[ei] if etf_atr_series[ei] > 0 else 1
        spy_atr = spy_w_atr_series[si] if si < len(spy_w_atr_series) and spy_w_atr_series[si] > 0 else 1
        spy_pi = spy_chg / spy_atr
        expected = spy_pi * stock_atr
        rrs = (stock_chg - expected) / stock_atr
        deltas.append(rrs)

    # SMA of deltas
    sma_series = []
    for i in range(MA_LENGTH_W - 1, len(deltas)):
        window = deltas[i - MA_LENGTH_W + 1:i + 1]
        sma_series.append(round(np.mean(window), 4))

    if not sma_series:
        return {"w_rv": round(rvol * 100) if rvol else None, "w_am": round(atr_mult * 100),
                "w_rs": None, "w_rf": 0, "w_ra": 0, "w_vs": None, "w_fr": None}

    final_rs = sma_series[-1]

    adv_streak = 0
    for j in range(len(sma_series) - 1, 0, -1):
        if sma_series[j] > sma_series[j - 1]: adv_streak += 1
        else: break
    dec_streak = 0
    for j in range(len(sma_series) - 1, 0, -1):
        if sma_series[j] < sma_series[j - 1]: dec_streak += 1
        else: break

    return {
        "w_rv": round(rvol * 100) if rvol else None, "w_am": round(atr_mult * 100),
        "w_rs": None,  # Set cross-sectionally in main()
        "w_rf": dec_streak, "w_ra": adv_streak, "w_vs": sma_series, "w_fr": round(final_rs, 4),
    }


def fetch_live_quote(ticker):
    """Fetch live (intraday) quote for a single ticker via fast_info.
    Returns (last_price, prev_close) or (None, None) on failure.
    """
    try:
        ti = yf.Ticker(ticker, session=_session)
        fi = ti.fast_info
        last = fi.get("lastPrice", fi.get("last_price"))
        prev = fi.get("previousClose", fi.get("previous_close"))
        last = float(last) if last not in (None, 0) else None
        prev = float(prev) if prev not in (None, 0) else None
        return last, prev
    except Exception:
        return None, None


def fetch_live_quotes_bulk(tickers, max_workers=20):
    """Parallel-fetch live quotes for many tickers. Returns {ticker: (last, prev)}."""
    out = {}
    if not tickers:
        return out
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(fetch_live_quote, t): t for t in tickers}
        for fut in as_completed(futs):
            tk = futs[fut]
            try:
                last, prev = fut.result(timeout=15)
                if last is not None and prev is not None:
                    out[tk] = (last, prev)
            except Exception:
                pass
    return out


def is_us_market_open():
    """Return True if US equity market is currently in regular session (9:30am–4:00pm ET, Mon–Fri).
    Uses naive ET wall clock; ignores holidays (the script's GitHub workflow already gates on
    market hours, so this is just a defensive check for the overlay step).
    """
    try:
        from datetime import timezone
        # Approximate ET as UTC-4 (EDT). Workflow already gates by ET hour, so close-enough.
        now_utc = datetime.now(timezone.utc)
        et_hour = (now_utc.hour - 4) % 24
        et_min  = now_utc.minute
        et_min_of_day = et_hour * 60 + et_min
        # Mon=0..Sun=6
        if now_utc.weekday() >= 5:
            return False
        # 9:30am = 570 min; 4:00pm = 960 min
        return 570 <= et_min_of_day <= 960
    except Exception:
        return False


def apply_intraday_overlay(results):
    """For each result, fetch live price via fast_info and recompute ch/c5/c20/ytd/p so the
    output reflects the current intraday move (not just the last completed daily bar).

    Only runs if US market is open. RS scores are NOT recomputed (they remain anchored to
    the last EOD bar — recomputing intraday would require re-running the full VARS pipeline).

    Each result must carry internal baseline fields (_pc, _5b, _20b, _yb) which were set
    by process_stock(); these are stripped before output.
    """
    if not is_us_market_open():
        print("  [Intraday overlay] Skipping — US market not open.")
        return
    tickers = [r["t"] for r in results]
    print(f"  [Intraday overlay] Fetching live quotes for {len(tickers)} stocks...")
    t0 = time.time()
    quotes = fetch_live_quotes_bulk(tickers)
    print(f"  [Intraday overlay] Got {len(quotes)} live quotes in {time.time()-t0:.1f}s")
    refreshed = 0
    for r in results:
        q = quotes.get(r["t"])
        if not q:
            continue
        live, prev_close = q
        if live <= 0 or prev_close <= 0:
            continue
        # Sanity check: live should be within ±50% of the EOD close. Reject wild values
        # (could be a stale fast_info or ticker mismatch).
        eod = r.get("p")
        if eod and (live > eod * 1.5 or live < eod * 0.5):
            continue
        # Apply overlay
        r["p"] = round(live, 2)
        # 1D change uses live's own previousClose for accuracy (handles dividends/splits
        # adjustments yfinance applies to fast_info)
        r["ch"] = round((live / prev_close - 1) * 100, 2)
        # 1W/1M/YTD: substitute live for the "current" leg, keep historical baselines
        if r.get("_5b") and r["_5b"] > 0:
            r["c5"] = round((live / r["_5b"] - 1) * 100, 2)
        if r.get("_20b") and r["_20b"] > 0:
            r["c20"] = round((live / r["_20b"] - 1) * 100, 2)
        if r.get("_yb") and r["_yb"] > 0:
            r["ytd"] = round((live / r["_yb"] - 1) * 100, 2)
        refreshed += 1
    print(f"  [Intraday overlay] Refreshed {refreshed}/{len(results)} entries with live prices")


def strip_internal_fields(results):
    """Remove _-prefixed internal fields before serializing to leaders.json."""
    for r in results:
        for k in list(r.keys()):
            if k.startswith("_"):
                del r[k]


def get_position_label(rank, total):
    """Map rank to position label."""
    if rank <= 50: return "Strong Leader"
    if rank <= 200: return "Leader Zone"
    pct = rank / total
    if pct <= 0.5: return "Mid-Range"
    if pct <= 0.75: return "Laggard Zone"
    return "Deep Laggard"


def get_theme_status(theme_name, etf_data):
    """Determine if a theme is Leader/Neutral/Laggard based on ETF RS data."""
    if not etf_data or "e" not in etf_data:
        return "Neutral"
    etfs = etf_data["e"]
    total = len(etfs)
    if total == 0:
        return "Neutral"
    # Find the theme's ETF entry
    for e in etfs:
        if e.get("n") == theme_name:
            rk = e.get("rk", total)
            w_rk = e.get("w_rk", total)
            avg_rk = (rk + w_rk) / 2
            q = avg_rk / total
            if q <= 0.25: return "Leader"
            if q <= 0.75: return "Neutral"
            return "Laggard"
    return "Neutral"


def main():
    print("=" * 60)
    print("Liquid Leaders — Stock RS Tracker")
    print("=" * 60)

    all_tickers, stock_to_etfs = build_universe()
    print(f"Universe: {len(all_tickers)} stocks, {len(set(name for etfs in stock_to_etfs.values() for name, _ in etfs))} unique ETF themes")

    # ─── Download SPY (daily + weekly) ────────────────────────
    end = datetime.now() + timedelta(days=1)
    start = end - timedelta(days=400)  # ~280 trading days — enough for 252-bar Pine Script RS + SMA200
    w_start = end - timedelta(days=365)

    print("\nDownloading SPY (daily + weekly)...")
    # auto_adjust=False so historical closes are UNADJUSTED (raw) — matches Yahoo Finance's
    # quote-page % changes. Dividend back-adjustment was inflating YTD for stocks that
    # paid divs between Dec 31 and today.
    spy_df = yf.download("SPY", session=_session, start=start.strftime("%Y-%m-%d"),
                          end=end.strftime("%Y-%m-%d"), auto_adjust=False, progress=False)
    if isinstance(spy_df.columns, type(spy_df.columns)) and hasattr(spy_df.columns, 'levels'):
        spy_df.columns = spy_df.columns.droplevel(1) if spy_df.columns.nlevels > 1 else spy_df.columns
    spy_df = spy_df.dropna(subset=["Close"])
    print(f"  SPY daily: {len(spy_df)} bars")

    spy_df_w = yf.download("SPY", session=_session, start=w_start.strftime("%Y-%m-%d"),
                            end=end.strftime("%Y-%m-%d"), interval="1wk",
                            auto_adjust=False, progress=False)
    if isinstance(spy_df_w.columns, type(spy_df_w.columns)) and hasattr(spy_df_w.columns, 'levels'):
        spy_df_w.columns = spy_df_w.columns.droplevel(1) if spy_df_w.columns.nlevels > 1 else spy_df_w.columns
    spy_df_w = spy_df_w.dropna(subset=["Close"])
    print(f"  SPY weekly: {len(spy_df_w)} bars")

    if len(spy_df) < LOOKBACK + 1:
        print(f"ERROR: SPY has only {len(spy_df)} bars, need {LOOKBACK + 1}"); sys.exit(1)

    # ─── Bulk download prices for entire universe (chunked) ──
    # yfinance's bulk download silently fails for many tickers when given >500 at once.
    # Chunk into batches of 200 with retries.
    print(f"\nDownloading daily data for {len(all_tickers)} tickers (chunked)...")
    import pandas as pd

    def chunked_download(tickers, **kwargs):
        """Download in chunks of 200 with proper MultiIndex handling for yfinance 1.0+."""
        CHUNK = 200
        all_dfs = {}

        def extract_ticker(df, tk):
            """Extract single-ticker DataFrame from bulk download."""
            cols = df.columns
            # yfinance 1.0+: MultiIndex with named levels ('Price', 'Ticker')
            if hasattr(cols, 'names') and cols.names and 'Ticker' in cols.names:
                try:
                    sub = df.xs(tk, axis=1, level='Ticker')
                    return sub.dropna(subset=['Close']) if not sub.empty else None
                except KeyError:
                    return None
            # yfinance 0.2.51+: unnamed 2-level MultiIndex
            if hasattr(cols, 'nlevels') and cols.nlevels == 2:
                try:
                    sub = df.xs(tk, axis=1, level=1)
                    return sub.dropna(subset=['Close']) if not sub.empty else None
                except KeyError:
                    return None
            # Old format: df[tk] gives sub-DataFrame
            try:
                sub = df[tk]
                if hasattr(sub, 'columns') and hasattr(sub.columns, 'nlevels') and sub.columns.nlevels > 1:
                    sub.columns = [c[0] if isinstance(c, tuple) else c for c in sub.columns]
                return sub.dropna(subset=['Close']) if not sub.empty else None
            except (KeyError, TypeError, AttributeError):
                return None

        def flatten_single(df):
            """Flatten a single-ticker download."""
            if hasattr(df.columns, 'names') and df.columns.names and 'Ticker' in df.columns.names:
                df = df.droplevel('Ticker', axis=1)
            elif hasattr(df.columns, 'nlevels') and df.columns.nlevels == 2:
                df = df.droplevel(1, axis=1)
            return df

        for i in range(0, len(tickers), CHUNK):
            batch = tickers[i:i+CHUNK]
            print(f"  Batch {i//CHUNK + 1}/{(len(tickers)+CHUNK-1)//CHUNK}: {i+1}-{min(i+CHUNK, len(tickers))}...")
            attempt = 0
            while attempt < 3:
                try:
                    df = yf.download(batch, group_by="ticker", auto_adjust=False,
                                     session=_session, threads=False, progress=False, **kwargs)
                    if df.empty:
                        print(f"    Empty result, retrying...")
                        attempt += 1; time.sleep(5); continue
                    if len(batch) == 1:
                        flat = flatten_single(df.copy()).dropna(subset=['Close'])
                        if not flat.empty:
                            all_dfs[batch[0]] = flat
                    else:
                        if i == 0:
                            print(f"    Column format: nlevels={df.columns.nlevels}, names={df.columns.names}")
                        for tk in batch:
                            tk_df = extract_ticker(df, tk)
                            if tk_df is not None and not tk_df.empty:
                                all_dfs[tk] = tk_df
                    break
                except Exception as e:
                    print(f"    Batch error: {e}, retrying...")
                    attempt += 1; time.sleep(5)
            time.sleep(1)
        return all_dfs


    daily_data = chunked_download(all_tickers,
                                   start=start.strftime("%Y-%m-%d"),
                                   end=end.strftime("%Y-%m-%d"))
    print(f"  Successfully downloaded daily data for {len(daily_data)}/{len(all_tickers)} tickers")

    if not daily_data:
        print("ERROR: No daily data downloaded"); sys.exit(1)

    print(f"\nDownloading weekly data (chunked)...")
    weekly_data = chunked_download(all_tickers,
                                    start=w_start.strftime("%Y-%m-%d"),
                                    end=end.strftime("%Y-%m-%d"),
                                    interval="1wk")
    print(f"  Successfully downloaded weekly data for {len(weekly_data)}/{len(all_tickers)} tickers")

    def get_df(ticker):
        return daily_data.get(ticker)

    def get_df_w(ticker):
        return weekly_data.get(ticker)

    # ─── Pre-filter: dollar volume + delisted + acquisition-limbo ──
    # Uses already-downloaded data — zero API cost.
    # Narrows universe before expensive mcap/industry lookups.
    MIN_DOLLAR_VOL = 70_000_000
    print(f"\nPre-filtering by liquidity (price × avg_vol_10d >= ${MIN_DOLLAR_VOL/1e6:.0f}M)...")

    # Tickers to log verbosely through each filter (for debugging why a stock is missing)
    DEBUG_TICKERS = {"RIG"}

    liquid_tickers = []
    excluded = {"no_data": 0, "stale": 0, "flat": 0, "illiquid": 0}
    for tk in all_tickers:
        df = get_df(tk)
        if df is None or len(df) < 10:
            excluded["no_data"] += 1
            if tk in DEBUG_TICKERS:
                print(f"  [DEBUG] {tk}: EXCLUDED — no_data (df={'None' if df is None else f'{len(df)} bars'})")
            continue
        # Stale data (likely delisted)
        try:
            last_bar = df.index[-1]
            last_date = last_bar.date() if hasattr(last_bar, "date") else date.fromisoformat(str(last_bar)[:10])
            if (date.today() - last_date).days > 7:
                excluded["stale"] += 1
                if tk in DEBUG_TICKERS:
                    print(f"  [DEBUG] {tk}: EXCLUDED — stale (last bar {last_date}, {(date.today()-last_date).days} days ago)")
                continue
        except Exception:
            pass
        c = df["Close"].values
        v = df["Volume"].values
        n = len(c)
        # ─── Acquisition-limbo detection ───────────────────────
        # Two patterns:
        # 1. Stock has been flat for a long time (slow-grinding pre-merger arb)
        # 2. Stock gapped recently and now sits pinned at deal price (post-announcement)
        # Use the LAST 10 BARS — short enough to catch post-gap pinning even when older
        # data is volatile. Real stocks (even low-vol staples/utilities) have wider
        # 10-day ranges than acquisition-pinned stocks.
        if n >= 10:
            recent = c[-10:]
            avg_price = np.mean(recent)
            if avg_price > 0:
                price_range_pct = (np.max(recent) - np.min(recent)) / avg_price
                returns = np.diff(recent) / recent[:-1]
                return_std = float(np.std(returns)) if len(returns) > 0 else 0
                # Acquisition-pinned: range under 2% AND return std under 0.5% over 10 days.
                # For comparison, even the lowest-vol utilities (e.g. SO, DUK) typically
                # have 10-day ranges of 3-5% and return std of 0.7-1.2%.
                if price_range_pct < 0.02 and return_std < 0.005:
                    excluded["flat"] += 1
                    if tk in DEBUG_TICKERS:
                        print(f"  [DEBUG] {tk}: EXCLUDED — acquisition-limbo (10d range {price_range_pct*100:.2f}%, std {return_std*100:.2f}%)")
                    continue
        # Dollar volume check (price × avg_vol_10d)
        last_price = float(c[-1])
        avg_vol_10d = float(np.mean(v[-10:]))
        dollar_vol = last_price * avg_vol_10d
        if dollar_vol < MIN_DOLLAR_VOL:
            excluded["illiquid"] += 1
            if tk in DEBUG_TICKERS:
                print(f"  [DEBUG] {tk}: EXCLUDED — illiquid (price ${last_price:.2f} × avg_vol_10d {avg_vol_10d/1e6:.2f}M = ${dollar_vol/1e6:.1f}M < ${MIN_DOLLAR_VOL/1e6:.0f}M)")
            continue
        liquid_tickers.append(tk)
        if tk in DEBUG_TICKERS:
            print(f"  [DEBUG] {tk}: PASSED pre-filter (price ${last_price:.2f}, dollar_vol ${dollar_vol/1e6:.1f}M)")
    print(f"  Pre-filter: {len(all_tickers)} → {len(liquid_tickers)} (excluded: {excluded['no_data']} no data, {excluded['stale']} delisted, {excluded['flat']} acquisition-limbo, {excluded['illiquid']} illiquid)")

    # ─── Market cap + industry lookup (ONLY on liquid survivors) ──
    MIN_MCAP = 2_000_000_000
    CACHE_VERSION = 5  # Bumped: adds names + descriptions
    mcap_data = {"refreshed": "", "caps": {}, "industries": {}, "names": {}, "descs": {}, "version": 0, "refresh_in_progress": False}
    if os.path.exists("leaders_mcap.json"):
        try:
            with open("leaders_mcap.json") as f:
                mcap_data = json.load(f)
                if "industries" not in mcap_data:
                    mcap_data["industries"] = {}
                if "names" not in mcap_data:
                    mcap_data["names"] = {}
                if "descs" not in mcap_data:
                    mcap_data["descs"] = {}
                if "version" not in mcap_data:
                    mcap_data["version"] = 0
                if "refresh_in_progress" not in mcap_data:
                    mcap_data["refresh_in_progress"] = False
        except Exception:
            mcap_data = {"refreshed": "", "caps": {}, "industries": {}, "names": {}, "descs": {}, "version": 0, "refresh_in_progress": False}

    mcap_cache = mcap_data.get("caps", {})
    industry_cache = mcap_data.get("industries", {})
    name_cache = mcap_data.get("names", {})
    desc_cache = mcap_data.get("descs", {})
    last_refreshed = mcap_data.get("refreshed", "")
    cache_version = mcap_data.get("version", 0)
    refresh_in_progress = mcap_data.get("refresh_in_progress", False)
    # Per-version refresh checkpoint: tracks which tickers have been re-verified at the current schema
    refreshed_at_v = mcap_data.get("refreshed_at_v", {})  # {ticker: version_int}

    # Auto-refresh if: cache older than 7 days OR schema version changed OR refresh still in progress
    needs_full_refresh = False
    if cache_version < CACHE_VERSION or refresh_in_progress:
        needs_full_refresh = True
        if cache_version < CACHE_VERSION:
            print(f"  Cache schema v{cache_version} → v{CACHE_VERSION} — re-verifying all entries (multi-run)")
        else:
            print(f"  Resuming in-progress refresh from previous run")
    else:
        try:
            if not last_refreshed:
                needs_full_refresh = True
            else:
                days_old = (date.today() - date.fromisoformat(last_refreshed)).days
                if days_old >= 7:
                    needs_full_refresh = True
                    print(f"  Cache is {days_old} days old — refreshing")
        except Exception:
            needs_full_refresh = True

    # Determine which liquid tickers need fetching this run
    if needs_full_refresh:
        # Only re-fetch tickers NOT yet verified at current schema version
        tickers_to_check = [t for t in liquid_tickers if refreshed_at_v.get(t, 0) < CACHE_VERSION]
    else:
        tickers_to_check = []
        for t in liquid_tickers:
            if t not in mcap_cache or t not in industry_cache:
                tickers_to_check.append(t)
            elif mcap_cache.get(t, 0) == 0:
                tickers_to_check.append(t)  # Self-heal failed mcap
            elif not industry_cache.get(t, ""):
                tickers_to_check.append(t)  # Self-heal missing industry
            elif not name_cache.get(t, ""):
                tickers_to_check.append(t)  # Self-heal missing name
            elif not desc_cache.get(t, ""):
                tickers_to_check.append(t)  # Self-heal missing description

    if tickers_to_check:
        MAX_PER_RUN = 500
        remaining_after_run = max(0, len(tickers_to_check) - MAX_PER_RUN)
        if len(tickers_to_check) > MAX_PER_RUN:
            print(f"  {len(tickers_to_check)} tickers need refresh — processing {MAX_PER_RUN}/run, {remaining_after_run} remaining")
            tickers_to_check = tickers_to_check[:MAX_PER_RUN]
        total = len(tickers_to_check)
        print(f"  Fetching market cap + industry for {total} liquid tickers...")
        failed = 0
        for i, tk in enumerate(tickers_to_check):
            mc = 0
            industry = ""
            name = ""
            desc = ""
            try:
                ticker_obj = yf.Ticker(tk, session=_session)
                fi = ticker_obj.fast_info
                try:
                    mc = int(fi.get("marketCap", 0) or fi.get("market_cap", 0) or 0)
                except Exception:
                    mc = 0
                if mc == 0:
                    try:
                        shares = fi.get("shares", 0) or 0
                        last_price = fi.get("lastPrice", fi.get("last_price", 0)) or 0
                        if shares and last_price:
                            mc = int(shares * last_price)
                    except Exception:
                        pass
                try:
                    info = ticker_obj.info
                    industry = info.get("industry", "") or ""
                    name = info.get("shortName", "") or info.get("longName", "") or ""
                    desc = info.get("longBusinessSummary", "") or ""
                    if mc == 0:
                        info_mc = info.get("marketCap", 0) or 0
                        if info_mc:
                            mc = int(info_mc)
                except Exception:
                    pass
            except Exception:
                pass
            if mc == 0:
                failed += 1
                # DO NOT overwrite cache with empty values on failure.
                # Keep any previously-cached data so transient fetch failures don't erase good entries.
                # If this ticker has never been successfully fetched, leave it unmarked so we retry next run.
                # If it has good cached data, mark as verified at current schema.
                if mcap_cache.get(tk, 0) > 0:
                    refreshed_at_v[tk] = CACHE_VERSION
            else:
                # Successful fetch — update all fields
                mcap_cache[tk] = mc
                industry_cache[tk] = industry
                name_cache[tk] = name
                desc_cache[tk] = desc
                refreshed_at_v[tk] = CACHE_VERSION
            time.sleep(0.2)
            if (i + 1) % 100 == 0:
                print(f"    {i+1}/{total}  (failed so far: {failed})")
                time.sleep(2)
        if failed > 0:
            print(f"  {failed} tickers failed lookup (will retry next run)")

        # Determine if refresh is now complete
        still_needs_refresh = remaining_after_run > 0
        mcap_data = {
            "refreshed": date.today().isoformat() if not still_needs_refresh else last_refreshed,
            "caps": mcap_cache,
            "industries": industry_cache,
            "names": name_cache,
            "descs": desc_cache,
            "version": CACHE_VERSION if not still_needs_refresh else cache_version,
            "refresh_in_progress": still_needs_refresh,
            "refreshed_at_v": refreshed_at_v,
        }
        with open("leaders_mcap.json", "w") as f:
            json.dump(mcap_data, f, separators=(",", ":"))
        if still_needs_refresh:
            print(f"  Cache partially updated ({len(mcap_cache)} entries, {remaining_after_run} more in next runs)")
        else:
            print(f"  Cache fully refreshed ({len(mcap_cache)} entries, version {CACHE_VERSION})")

    # Final filter: market cap >= $2B (unknowns excluded)
    all_tickers_before_mcap = list(liquid_tickers)
    all_tickers = [t for t in liquid_tickers if mcap_cache.get(t, 0) >= MIN_MCAP]
    removed = len(liquid_tickers) - len(all_tickers)
    print(f"  Market cap filter: {removed} removed (< ${MIN_MCAP/1e9:.0f}B), {len(all_tickers)} remaining")
    # Diagnostic: log DEBUG_TICKERS that got filtered here
    for tk in DEBUG_TICKERS:
        if tk in all_tickers_before_mcap and tk not in all_tickers:
            cached_mc = mcap_cache.get(tk, 0)
            print(f"  [DEBUG] {tk}: EXCLUDED by market cap filter (mc=${cached_mc/1e9:.2f}B, need >= ${MIN_MCAP/1e9:.0f}B)")
        elif tk in all_tickers:
            cached_mc = mcap_cache.get(tk, 0)
            print(f"  [DEBUG] {tk}: PASSED market cap filter (mc=${cached_mc/1e9:.2f}B)")

    # ─── Resolve themes: two fields per stock ──────────────────
    # th  = display label (Yahoo industry verbatim, shown in frontend column)
    # thm = mapped sector basket name (used for Leader/Neutral/Laggard status)
    theme_map = {}       # ticker -> mapped sector name
    industry_label = {}  # ticker -> Yahoo industry verbatim
    protected_count = yahoo_count = etf_count = general_count = 0
    for tk in all_tickers:
        # Display label: always use Yahoo's raw industry if available
        raw_industry = industry_cache.get(tk, "")
        industry_label[tk] = raw_industry if raw_industry else "General"

        # Mapped sector: priority = protected basket > Yahoo mapping > ETF fallback
        etfs = stock_to_etfs.get(tk, [])
        protected = [(n, c) for n, c in etfs if n in PROTECTED_BASKETS]
        if protected:
            protected.sort(key=lambda x: x[1])
            theme_map[tk] = protected[0][0]
            protected_count += 1
        else:
            yahoo_theme = map_industry_to_theme(raw_industry)
            if yahoo_theme:
                theme_map[tk] = yahoo_theme
                yahoo_count += 1
            elif etfs:
                etfs_sorted = sorted(etfs, key=lambda x: x[1])
                theme_map[tk] = etfs_sorted[0][0]
                etf_count += 1
            else:
                theme_map[tk] = "General"
                general_count += 1
    print(f"  Theme mapping: {protected_count} protected, {yahoo_count} via Yahoo, {etf_count} via ETF, {general_count} general · {len(set(theme_map.values()))} active themes")

    # ─── Pharma/biotech mcap filter: require >= $20B for these themes ──
    PHARMA_BIOTECH_THEMES = {"Pharmaceuticals", "Biotechnology"}
    PHARMA_BIOTECH_MIN_MCAP = 20_000_000_000
    before = len(all_tickers)
    all_tickers = [
        t for t in all_tickers
        if theme_map.get(t) not in PHARMA_BIOTECH_THEMES
        or mcap_cache.get(t, 0) >= PHARMA_BIOTECH_MIN_MCAP
    ]
    pharma_removed = before - len(all_tickers)
    print(f"  Pharma/Biotech filter: {pharma_removed} removed (< ${PHARMA_BIOTECH_MIN_MCAP/1e9:.0f}B), {len(all_tickers)} remaining")
    # Diagnostic for DEBUG_TICKERS
    for tk in DEBUG_TICKERS:
        if tk in all_tickers:
            mc = mcap_cache.get(tk, 0)
            theme = theme_map.get(tk, "?")
            industry = industry_label.get(tk, "?")
            print(f"  [DEBUG] {tk}: IN FINAL UNIVERSE (mc=${mc/1e9:.2f}B, theme='{theme}', industry='{industry}')")
        else:
            # Check if it ever made it this far
            if tk not in all_tickers_before_mcap:
                pass  # already logged as excluded earlier
            else:
                mc = mcap_cache.get(tk, 0)
                theme = theme_map.get(tk, "?")
                if theme in PHARMA_BIOTECH_THEMES and mc < PHARMA_BIOTECH_MIN_MCAP:
                    print(f"  [DEBUG] {tk}: EXCLUDED by pharma filter (mc=${mc/1e9:.2f}B, theme='{theme}')")

    # ─── SPY baselines ────────────────────────────────────────
    spy_closes = spy_df["Close"].values
    spy_highs = spy_df["High"].values
    spy_lows = spy_df["Low"].values
    spy_atr_series = compute_atr_series(spy_highs, spy_lows, spy_closes, ATR_PERIOD)
    spy_ts_map = {ts: i for i, ts in enumerate(spy_df.index)}

    spy_w_closes = spy_df_w["Close"].values if spy_df_w is not None and len(spy_df_w) > 0 else np.array([])
    spy_w_highs = spy_df_w["High"].values if spy_df_w is not None and len(spy_df_w) > 0 else np.array([])
    spy_w_lows = spy_df_w["Low"].values if spy_df_w is not None and len(spy_df_w) > 0 else np.array([])
    spy_w_atr_series = compute_atr_series(spy_w_highs, spy_w_lows, spy_w_closes, ATR_PERIOD) if len(spy_w_closes) > ATR_PERIOD else []
    spy_w_ts_map = {ts: i for i, ts in enumerate(spy_df_w.index)} if spy_df_w is not None and len(spy_df_w) > 0 else {}

    # ─── Process all stocks ───────────────────────────────────
    print(f"\nProcessing {len(all_tickers)} stocks...")
    results = []
    processed = 0
    for i, tk in enumerate(all_tickers):
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(all_tickers)}...")
        try:
            df = get_df(tk)
            d_metrics = process_stock(tk, df, spy_closes, spy_highs, spy_lows, spy_atr_series, spy_ts_map)
            w_metrics = process_stock_weekly(tk, get_df_w(tk), spy_w_closes, spy_w_atr_series, spy_w_ts_map)
            if d_metrics is None:
                continue
            processed += 1
            entry = {"t": tk, "n": name_cache.get(tk, ""), "d": desc_cache.get(tk, ""), "th": industry_label.get(tk, "General"), "thm": theme_map.get(tk, "General"), **d_metrics, **w_metrics}
            results.append(entry)
        except Exception:
            continue

    print(f"  Processed: {processed}/{len(all_tickers)}")

    # ─── Intraday overlay: replace EOD price with live quote when market is open ──
    # Updates p / ch / c5 / c20 / ytd to reflect the current intraday move. RS scores
    # remain anchored to the last completed daily bar (recomputing intraday would require
    # re-running the full VARS pipeline, which is expensive).
    apply_intraday_overlay(results)

    # ─── Cross-sectional percentrank (compare each stock vs ALL others) ──
    all_d_rs = [r["fr"] for r in results if r.get("fr") is not None]
    all_w_rs = [r.get("w_fr") for r in results if r.get("w_fr") is not None]

    for r in results:
        if r.get("fr") is not None and len(all_d_rs) > 1:
            raw_pctrank = round(percentrank_inc(all_d_rs, r["fr"]) * 100)
            # Apply setup quality adjustment (bonus/penalty based on MA criteria)
            sa = r.get("sa", 0)
            adjusted = max(0, min(100, raw_pctrank + sa))
            # Apply trend-zone multiplier from Pine Script CMI smooth_trend
            # bear_strong → ×0.50, bear_light → ×0.75, neutral → ×1.00,
            # bull_light → ×1.05, bull_strong → ×1.10
            tz = r.get("tz", "neutral")
            tz_mult = {
                "bear_strong": 0.50,
                "bear_light":  0.75,
                "neutral":     1.00,
                "bull_light":  1.05,
                "bull_strong": 1.10,
            }.get(tz, 1.0)
            adjusted = max(0, min(100, adjusted * tz_mult))
            r["rs"] = round(adjusted)
        if r.get("w_fr") is not None and len(all_w_rs) > 1:
            r["w_rs"] = round(percentrank_inc(all_w_rs, r["w_fr"]) * 100)

    # ─── Rank stocks (by adjusted RS score, then raw RS as tiebreak) ──
    results.sort(key=lambda x: (x["rs"] if x["rs"] is not None else -999,
                                 x["fr"] if x["fr"] is not None else -999), reverse=True)
    for i, r in enumerate(results):
        r["rk"] = i + 1

    # Weekly rank (by raw weekly RS value desc, then c5 desc)
    w_sorted = sorted(results, key=lambda x: (x.get("w_fr") if x.get("w_fr") is not None else -999,
                                                x["c5"] if x["c5"] is not None else -999), reverse=True)
    w_rank_map = {r["t"]: i + 1 for i, r in enumerate(w_sorted)}
    for r in results:
        r["w_rk"] = w_rank_map.get(r["t"], len(results))

    # ─── Rank history (rolling weekly snapshots) ──────────────
    today_str = date.today().isoformat()
    today_week = date.today().isocalendar()
    week_key = f"{today_week[0]}-W{today_week[1]:02d}"

    rank_history = {"weeks": []}
    if os.path.exists("leaders_rank_history.json"):
        try:
            with open("leaders_rank_history.json") as f:
                rank_history = json.load(f)
        except Exception:
            rank_history = {"weeks": []}

    current_d_ranks = {r["t"]: r["rk"] for r in results}
    current_w_ranks = {r["t"]: r["w_rk"] for r in results}
    weeks = rank_history.get("weeks", [])
    if weeks and weeks[-1].get("wk") == week_key:
        weeks[-1] = {"wk": week_key, "date": today_str, "d": current_d_ranks, "w": current_w_ranks}
    else:
        weeks.append({"wk": week_key, "date": today_str, "d": current_d_ranks, "w": current_w_ranks})
    if len(weeks) > 9:
        weeks = weeks[-9:]
    rank_history["weeks"] = weeks

    with open("leaders_rank_history.json", "w") as f:
        json.dump(rank_history, f, separators=(",", ":"))

    # ─── Score history (rolling daily snapshots for search) ───
    score_history = {"dates": [], "d": {}}
    if os.path.exists("leaders_score_history.json"):
        try:
            with open("leaders_score_history.json") as f:
                score_history = json.load(f)
        except Exception:
            score_history = {"dates": [], "d": {}}

    dates = score_history.get("dates", [])
    scores = score_history.get("d", {})

    if not dates or dates[-1] != today_str:
        dates.append(today_str)
        for r in results:
            tk = r["t"]
            if tk not in scores:
                scores[tk] = {"s": [], "r": [], "wr": []}
            scores[tk]["s"].append(r.get("rs"))
            scores[tk]["r"].append(r.get("rk"))
            scores[tk]["wr"].append(r.get("w_rk"))
        # Trim to MAX_HISTORY_DAYS
        if len(dates) > MAX_HISTORY_DAYS:
            trim = len(dates) - MAX_HISTORY_DAYS
            dates = dates[trim:]
            for tk in scores:
                for key in ["s", "r", "wr"]:
                    if len(scores[tk][key]) > MAX_HISTORY_DAYS:
                        scores[tk][key] = scores[tk][key][trim:]
    else:
        # Update today's entry
        for r in results:
            tk = r["t"]
            if tk not in scores:
                scores[tk] = {"s": [], "r": [], "wr": []}
            if len(scores[tk]["s"]) == len(dates):
                scores[tk]["s"][-1] = r.get("rs")
                scores[tk]["r"][-1] = r.get("rk")
                scores[tk]["wr"][-1] = r.get("w_rk")
            else:
                scores[tk]["s"].append(r.get("rs"))
                scores[tk]["r"].append(r.get("rk"))
                scores[tk]["wr"].append(r.get("w_rk"))

    score_history = {"dates": dates, "d": scores}
    with open("leaders_score_history.json", "w") as f:
        json.dump(score_history, f, separators=(",", ":"))

    # ─── Load ETF data for theme status ───────────────────────
    etf_data = None
    if os.path.exists("data.json"):
        try:
            with open("data.json") as f:
                etf_data = json.load(f)
        except Exception:
            pass

    # Build theme status map
    theme_status = {}
    if etf_data:
        for e in etf_data.get("e", []):
            name = e.get("n", "")
            if name and name not in theme_status:
                theme_status[name] = get_theme_status(name, etf_data)

    # Add theme status to each result
    for r in results:
        r["ts"] = theme_status.get(r.get("thm", ""), "Neutral")

    # ─── Prepare rank history for frontend ────────────────────
    rh_d = [snap.get("d", {}) for snap in weeks[:-1]]
    rh_w = [snap.get("w", {}) for snap in weeks[:-1]]

    # ─── Output leaders.json ──────────────────────────────────
    # Strip internal _-prefixed baseline fields used by the intraday overlay so they
    # don't bloat the output JSON.
    strip_internal_fields(results)

    data = {
        "e": results,
        "meta": {
            "updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
            "count": len(results),
            "universe": len(all_tickers),
        },
        "rh": {"d": rh_d, "w": rh_w},
    }

    with open("leaders.json", "w") as f:
        json.dump(data, f, separators=(",", ":"))

    print(f"\nOutput: leaders.json ({len(results)} stocks)")
    print(f"  leaders_rank_history.json ({len(weeks)} weekly snapshots)")
    print(f"  leaders_score_history.json ({len(dates)} daily snapshots)")


if __name__ == "__main__":
    main()
