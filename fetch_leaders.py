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

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

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
    "GOOG","LNG","NOK","NVO","OWL","RDDT","SMCI","SNAP","SNDK","STX","WDC",
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
    "Auto Manufacturers": "EV & Mobility",
    "Auto Parts": "EV & Mobility",
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

    return sorted(stocks), stock_to_etfs


# Niche themes Yahoo Finance can't properly classify — these baskets take priority over Yahoo industry
PROTECTED_BASKETS = {
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
    "EV & Mobility",
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


def process_stock(ticker, df, spy_closes, spy_highs, spy_lows, spy_atr_series, spy_ts_map):
    """Process daily metrics for one stock. Returns dict or None.
    Returns None if stock appears delisted (stale data) or in acquisition limbo (flat price).
    """
    if df is None or len(df) < 10:
        return None
    c = df["Close"].values
    h = df["High"].values
    l = df["Low"].values
    v = df["Volume"].values
    n = len(c)

    # ─── Delisted check: last bar must be recent ─────────────
    try:
        last_bar = df.index[-1]
        last_date = last_bar.date() if hasattr(last_bar, "date") else date.fromisoformat(str(last_bar)[:10])
        days_stale = (date.today() - last_date).days
        if days_stale > 7:  # More than a week with no data → likely delisted
            return None
    except Exception:
        pass

    # ─── Acquisition-limbo check: extremely flat price ───────
    # Acquired stocks trade near the deal price with near-zero volatility for weeks before delisting.
    # A normal large-cap has daily return std >= 0.4% and 20-day price range >= 2%.
    if n >= 20:
        recent = c[-20:]
        avg_price = np.mean(recent)
        if avg_price > 0:
            price_range = (np.max(recent) - np.min(recent)) / avg_price
            returns = np.diff(recent) / recent[:-1]
            return_std = float(np.std(returns)) if len(returns) > 0 else 0
            # Flag only if BOTH conditions met (avoids false positives on low-vol staples/utilities)
            if price_range < 0.015 and return_std < 0.0025:
                return None

    price = float(c[-1])
    change = (c[-1] / c[-2] - 1) * 100 if n >= 2 else 0
    c5 = (c[-1] / c[-6] - 1) * 100 if n >= 6 else None
    c20 = (c[-1] / c[-21] - 1) * 100 if n >= 21 else None

    # ─── Liquidity filter: Price × Avg Volume (10D) >= $70M ──
    if n >= 10:
        avg_vol_10d = float(np.mean(v[-10:]))
        dollar_vol = price * avg_vol_10d
        if dollar_vol < 70_000_000:
            return None

    atr = compute_atr(h, l, c, ATR_PERIOD)
    valid_c = [x for x in c if x is not None]
    sma50 = np.mean(valid_c[-50:]) if len(valid_c) >= 50 else np.mean(valid_c)
    atr_ext = abs(c[-1] - sma50) / atr if atr > 0 else 0
    atr_pct = (atr / c[-2] * 100) if n >= 2 and c[-2] != 0 else 1
    atr_mult = abs(change) / atr_pct if atr_pct > 0 else 0

    vols = [x for x in v if x is not None and x > 0]
    rvol = (vols[-1] / np.mean(vols[:-1][-20:])) if len(vols) > 1 and np.mean(vols[:-1][-20:]) > 0 else None

    # Real Relative Strength (Reddit method):
    # Per-bar delta = (stock_change/stock_ATR) - (spy_change/spy_ATR)
    # Then SMA(MA_LENGTH) of those deltas = smoothed RS
    # Sparkline shows rolling SMA values over lookback
    common = []
    for idx, ts in enumerate(df.index):
        if ts in spy_ts_map and c[idx] is not None and spy_closes[spy_ts_map[ts]] is not None:
            common.append((idx, spy_ts_map[ts]))

    if len(common) < LOOKBACK:
        return {"rv": round(rvol * 100) if rvol else None, "am": round(atr_mult * 100),
                "ax": round(atr_ext * 100), "ch": round(change, 2),
                "c5": round(c5, 2) if c5 is not None else None,
                "c20": round(c20, 2) if c20 is not None else None,
                "rs": None, "rf": 0, "ra": 0, "p": round(price, 2), "fr": None, "vs": None}

    etf_atr_series = compute_atr_series(h, l, c, ATR_PERIOD)
    extended = common[-(LOOKBACK + 1):]

    # Compute per-bar deltas
    deltas = []
    for k in range(1, len(extended)):
        ei_prev, si_prev = extended[k - 1]
        ei, si = extended[k]
        stock_chg = c[ei] - c[ei_prev]
        spy_chg = spy_closes[si] - spy_closes[si_prev]
        stock_atr = etf_atr_series[ei] if etf_atr_series[ei] > 0 else 1
        spy_atr = spy_atr_series[si] if spy_atr_series[si] > 0 else 1
        # SPY Power Index: how many ATRs did SPY move?
        spy_pi = spy_chg / spy_atr
        # Expected stock change = SPY_PI * stock_ATR
        expected = spy_pi * stock_atr
        # Real RS = (actual - expected) / stock_ATR
        rrs = (stock_chg - expected) / stock_atr
        deltas.append(rrs)

    # Compute rolling SMA(MA_LENGTH) of deltas → smoothed RS series
    sma_series = []
    for i in range(MA_LENGTH - 1, len(deltas)):
        window = deltas[i - MA_LENGTH + 1:i + 1]
        sma_series.append(round(np.mean(window), 4))

    if not sma_series:
        return {"rv": round(rvol * 100) if rvol else None, "am": round(atr_mult * 100),
                "ax": round(atr_ext * 100), "ch": round(change, 2),
                "c5": round(c5, 2) if c5 is not None else None,
                "c20": round(c20, 2) if c20 is not None else None,
                "rs": None, "rf": 0, "ra": 0, "p": round(price, 2), "fr": None, "vs": None}

    final_rs = sma_series[-1]
    # RS percentrank computed cross-sectionally in main() after all stocks processed

    # Advancing/declining streaks on the smoothed series
    adv_streak = 0
    for j in range(len(sma_series) - 1, 0, -1):
        if sma_series[j] > sma_series[j - 1]: adv_streak += 1
        else: break
    dec_streak = 0
    for j in range(len(sma_series) - 1, 0, -1):
        if sma_series[j] < sma_series[j - 1]: dec_streak += 1
        else: break

    return {
        "rv": round(rvol * 100) if rvol else None,
        "am": round(atr_mult * 100), "ax": round(atr_ext * 100),
        "ch": round(change, 2),
        "c5": round(c5, 2) if c5 is not None else None,
        "c20": round(c20, 2) if c20 is not None else None,
        "rs": None,  # Will be set cross-sectionally in main()
        "rf": dec_streak, "ra": adv_streak,
        "p": round(price, 2), "fr": round(final_rs, 4), "vs": sma_series,
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

    # ─── Filter by market cap >= $1B + fetch industry (cached, weekly auto-refresh) ──
    MIN_MCAP = 1_000_000_000
    CACHE_VERSION = 2  # Bump this when mcap/industry fetching logic changes to force a refresh
    mcap_data = {"refreshed": "", "caps": {}, "industries": {}, "version": 0}
    if os.path.exists("leaders_mcap.json"):
        try:
            with open("leaders_mcap.json") as f:
                mcap_data = json.load(f)
                if "industries" not in mcap_data:
                    mcap_data["industries"] = {}
                if "version" not in mcap_data:
                    mcap_data["version"] = 0
        except Exception:
            mcap_data = {"refreshed": "", "caps": {}, "industries": {}, "version": 0}

    mcap_cache = mcap_data.get("caps", {})
    industry_cache = mcap_data.get("industries", {})
    last_refreshed = mcap_data.get("refreshed", "")
    cache_version = mcap_data.get("version", 0)

    # Auto-refresh if: cache is older than 7 days OR schema version changed
    needs_full_refresh = False
    if cache_version < CACHE_VERSION:
        needs_full_refresh = True
        print(f"  Cache schema v{cache_version} → v{CACHE_VERSION} — forcing full refresh")
    else:
        try:
            if not last_refreshed:
                needs_full_refresh = True
            else:
                days_old = (date.today() - date.fromisoformat(last_refreshed)).days
                if days_old >= 7:
                    needs_full_refresh = True
                    print(f"  Cache is {days_old} days old — refreshing all market caps + industries")
        except Exception:
            needs_full_refresh = True

    # Refresh tickers missing either market cap OR industry
    if needs_full_refresh:
        tickers_to_check = all_tickers
    else:
        tickers_to_check = [t for t in all_tickers if t not in mcap_cache or t not in industry_cache]

    if tickers_to_check:
        print(f"  Fetching market cap + industry for {len(tickers_to_check)} tickers{' (full refresh)' if needs_full_refresh else ' (new only)'}...")
        for i, tk in enumerate(tickers_to_check):
            mc = 0
            industry = ""
            try:
                ticker_obj = yf.Ticker(tk)
                fi = ticker_obj.fast_info
                # Try fast_info.marketCap first
                try:
                    mc = int(fi.get("marketCap", 0) or fi.get("market_cap", 0) or 0)
                except Exception:
                    mc = 0
                # Fallback: shares outstanding × last price
                if mc == 0:
                    try:
                        shares = fi.get("shares", 0) or 0
                        last_price = fi.get("lastPrice", fi.get("last_price", 0)) or 0
                        if shares and last_price:
                            mc = int(shares * last_price)
                    except Exception:
                        pass
                # Fetch industry (and try one more market cap source from info)
                try:
                    info = ticker_obj.info
                    industry = info.get("industry", "") or ""
                    if mc == 0:
                        info_mc = info.get("marketCap", 0) or 0
                        if info_mc:
                            mc = int(info_mc)
                except Exception:
                    pass
            except Exception:
                pass
            mcap_cache[tk] = mc  # 0 if all attempts failed → will be filtered out
            industry_cache[tk] = industry
            if (i + 1) % 50 == 0:
                print(f"    {i+1}/{len(tickers_to_check)}...")
                time.sleep(1)
        mcap_data = {"refreshed": date.today().isoformat(), "caps": mcap_cache, "industries": industry_cache, "version": CACHE_VERSION}
        with open("leaders_mcap.json", "w") as f:
            json.dump(mcap_data, f, separators=(",", ":"))
        print(f"  Saved {len(mcap_cache)} market caps + {len(industry_cache)} industries (refreshed {date.today().isoformat()})")

    # Filter by market cap
    filtered = [t for t in all_tickers if mcap_cache.get(t, 0) >= MIN_MCAP]
    removed = len(all_tickers) - len(filtered)
    print(f"  Market cap filter: {removed} stocks removed (< $1B), {len(filtered)} remaining")
    all_tickers = filtered

    # ─── Resolve each stock's theme using protected basket > Yahoo industry > fallback ──
    theme_map = {}
    protected_count = yahoo_count = etf_count = general_count = 0
    for tk in all_tickers:
        etfs = stock_to_etfs.get(tk, [])
        protected = [(n, c) for n, c in etfs if n in PROTECTED_BASKETS]
        if protected:
            protected.sort(key=lambda x: x[1])
            theme_map[tk] = protected[0][0]
            protected_count += 1
        else:
            industry = industry_cache.get(tk, "")
            yahoo_theme = map_industry_to_theme(industry)
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
    print(f"  Theme mapping: {protected_count} protected baskets, {yahoo_count} via Yahoo industry, {etf_count} via ETF fallback, {general_count} general")
    print(f"  Active themes: {len(set(theme_map.values()))}")

    # ─── Download daily data ──────────────────────────────────
    end = datetime.now() + timedelta(days=1)
    start = end - timedelta(days=180)  # Need ~84 trading days for LOOKBACK=50 + MA=20 + ATR=14
    w_start = end - timedelta(days=365)

    # Download SPY separately first (guaranteed)
    print("\nDownloading SPY (daily + weekly) separately...")
    spy_df = yf.download("SPY", start=start.strftime("%Y-%m-%d"),
                          end=end.strftime("%Y-%m-%d"), auto_adjust=True, progress=False)
    if isinstance(spy_df.columns, type(spy_df.columns)) and hasattr(spy_df.columns, 'levels'):
        spy_df.columns = spy_df.columns.droplevel(1) if spy_df.columns.nlevels > 1 else spy_df.columns
    spy_df = spy_df.dropna(subset=["Close"])
    print(f"  SPY daily: {len(spy_df)} bars")

    spy_df_w = yf.download("SPY", start=w_start.strftime("%Y-%m-%d"),
                            end=end.strftime("%Y-%m-%d"), interval="1wk",
                            auto_adjust=True, progress=False)
    if isinstance(spy_df_w.columns, type(spy_df_w.columns)) and hasattr(spy_df_w.columns, 'levels'):
        spy_df_w.columns = spy_df_w.columns.droplevel(1) if spy_df_w.columns.nlevels > 1 else spy_df_w.columns
    spy_df_w = spy_df_w.dropna(subset=["Close"])
    print(f"  SPY weekly: {len(spy_df_w)} bars")

    if len(spy_df) < LOOKBACK + 1:
        print(f"ERROR: SPY has only {len(spy_df)} bars, need {LOOKBACK + 1}"); sys.exit(1)

    print(f"\nDownloading daily data for {len(all_tickers)} stock tickers...")
    raw = yf.download(all_tickers, start=start.strftime("%Y-%m-%d"),
                      end=end.strftime("%Y-%m-%d"), group_by="ticker",
                      auto_adjust=True, threads=True, progress=False)
    if raw.empty:
        print("ERROR: No daily data"); sys.exit(1)

    print(f"Downloading weekly data...")
    raw_w = yf.download(all_tickers, start=w_start.strftime("%Y-%m-%d"),
                        end=end.strftime("%Y-%m-%d"), interval="1wk",
                        group_by="ticker", auto_adjust=True, threads=True, progress=False)

    def get_df(ticker):
        try:
            df = raw[ticker].dropna(subset=["Close"]) if len(all_tickers) > 1 else raw.dropna(subset=["Close"])
            return df
        except Exception:
            return None

    def get_df_w(ticker):
        try:
            if raw_w.empty: return None
            df = raw_w[ticker].dropna(subset=["Close"]) if len(all_tickers) > 1 else raw_w.dropna(subset=["Close"])
            return df
        except Exception:
            return None

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
            entry = {"t": tk, "th": theme_map.get(tk, "General"), **d_metrics, **w_metrics}
            results.append(entry)
        except Exception:
            continue

    print(f"  Processed: {processed}/{len(all_tickers)}")

    # ─── Cross-sectional percentrank (compare each stock vs ALL others) ──
    all_d_rs = [r["fr"] for r in results if r.get("fr") is not None]
    all_w_rs = [r.get("w_fr") for r in results if r.get("w_fr") is not None]

    for r in results:
        if r.get("fr") is not None and len(all_d_rs) > 1:
            r["rs"] = round(percentrank_inc(all_d_rs, r["fr"]) * 100)
        if r.get("w_fr") is not None and len(all_w_rs) > 1:
            r["w_rs"] = round(percentrank_inc(all_w_rs, r["w_fr"]) * 100)

    # ─── Rank stocks ──────────────────────────────────────────
    # Daily rank (by raw RS value desc, then 1D change desc)
    results.sort(key=lambda x: (x["fr"] if x["fr"] is not None else -999,
                                 x["ch"] if x["ch"] is not None else -999), reverse=True)
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
        r["ts"] = theme_status.get(r.get("th", ""), "Neutral")

    # ─── Prepare rank history for frontend ────────────────────
    rh_d = [snap.get("d", {}) for snap in weeks[:-1]]
    rh_w = [snap.get("w", {}) for snap in weeks[:-1]]

    # ─── Output leaders.json ──────────────────────────────────
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
