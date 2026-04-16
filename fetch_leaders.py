#!/usr/bin/env python3
"""
Liquid Leaders — Stock-level relative strength tracker.
Computes VARS (daily + weekly) for ~1100 individual stocks vs SPY.
Outputs leaders.json (current snapshot) and leaders_history.json (rolling 30-day history).
"""

import json, os, sys, time
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta, date

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
    "ACN","ALAB","APO","ARES","ARM","AXTI","BIRD","BP","BRK-B","CAR",
    "GOOG","LNG","NOK","NVO","OWL","RDDT","SMCI","SNAP","SNDK","STX","WDC",
]


def build_universe():
    """Build stock universe + theme map from ETF_INFO."""
    stocks = set()
    theme_map = {}
    for info in ETF_INFO:
        h_str = info.get("h", "")
        name = info.get("n", "")
        if not h_str:
            continue
        for h in h_str.split(","):
            h = h.strip()
            if h:
                stocks.add(h)
                if h not in theme_map:
                    theme_map[h] = name
    for t in CSV_EXTRAS:
        stocks.add(t)
        if t not in theme_map:
            theme_map[t] = "General"
    return sorted(stocks), theme_map


def process_stock(ticker, df, spy_closes, spy_highs, spy_lows, spy_atr_series, spy_ts_map):
    """Process daily metrics for one stock. Returns dict or None."""
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

    all_tickers, theme_map = build_universe()
    print(f"Universe: {len(all_tickers)} stocks, {len(set(theme_map.values()))} themes")

    # ─── Filter by market cap >= $1B (cached, auto-refreshes weekly) ──
    MIN_MCAP = 1_000_000_000
    mcap_data = {"refreshed": "", "caps": {}}
    if os.path.exists("leaders_mcap.json"):
        try:
            with open("leaders_mcap.json") as f:
                mcap_data = json.load(f)
        except Exception:
            mcap_data = {"refreshed": "", "caps": {}}

    mcap_cache = mcap_data.get("caps", {})
    last_refreshed = mcap_data.get("refreshed", "")

    # Auto-refresh if cache is older than 7 days
    needs_full_refresh = False
    try:
        if not last_refreshed:
            needs_full_refresh = True
        else:
            days_old = (date.today() - date.fromisoformat(last_refreshed)).days
            if days_old >= 7:
                needs_full_refresh = True
                print(f"  Market cap cache is {days_old} days old — refreshing all")
    except Exception:
        needs_full_refresh = True

    tickers_to_check = all_tickers if needs_full_refresh else [t for t in all_tickers if t not in mcap_cache]

    if tickers_to_check:
        print(f"  Fetching market caps for {len(tickers_to_check)} tickers{'(full refresh)' if needs_full_refresh else ' (new only)'}...")
        for i, tk in enumerate(tickers_to_check):
            try:
                fi = yf.Ticker(tk).fast_info
                mc = fi.get("marketCap", fi.get("market_cap", 0)) or 0
                mcap_cache[tk] = int(mc)
            except Exception:
                mcap_cache[tk] = MIN_MCAP  # Assume valid if can't check
            if (i + 1) % 50 == 0:
                print(f"    {i+1}/{len(tickers_to_check)}...")
                time.sleep(1)
        mcap_data = {"refreshed": date.today().isoformat(), "caps": mcap_cache}
        with open("leaders_mcap.json", "w") as f:
            json.dump(mcap_data, f, separators=(",", ":"))
        print(f"  Saved {len(mcap_cache)} market caps (refreshed {date.today().isoformat()})")

    # Filter
    filtered = [t for t in all_tickers if mcap_cache.get(t, MIN_MCAP) >= MIN_MCAP]
    removed = len(all_tickers) - len(filtered)
    print(f"  Market cap filter: {removed} stocks removed (< $1B), {len(filtered)} remaining")
    all_tickers = filtered

    # ─── Download daily data ──────────────────────────────────
    end = datetime.now() + timedelta(days=1)
    start = end - timedelta(days=180)  # Need ~84 trading days for LOOKBACK=50 + MA=20 + ATR=14

    dl_tickers = list(set(["SPY"] + all_tickers))
    print(f"\nDownloading daily data for {len(dl_tickers)} tickers...")
    raw = yf.download(dl_tickers, start=start.strftime("%Y-%m-%d"),
                      end=end.strftime("%Y-%m-%d"), group_by="ticker",
                      auto_adjust=True, threads=True)
    if raw.empty:
        print("ERROR: No daily data"); sys.exit(1)

    # Download weekly data
    w_start = end - timedelta(days=365)
    print(f"Downloading weekly data...")
    raw_w = yf.download(dl_tickers, start=w_start.strftime("%Y-%m-%d"),
                        end=end.strftime("%Y-%m-%d"), interval="1wk",
                        group_by="ticker", auto_adjust=True, threads=True)

    def get_df(ticker):
        try:
            df = raw[ticker].dropna(subset=["Close"]) if len(dl_tickers) > 1 else raw.dropna(subset=["Close"])
            return df
        except Exception:
            return None

    def get_df_w(ticker):
        try:
            if raw_w.empty: return None
            df = raw_w[ticker].dropna(subset=["Close"]) if len(dl_tickers) > 1 else raw_w.dropna(subset=["Close"])
            return df
        except Exception:
            return None

    # ─── SPY baselines ────────────────────────────────────────
    spy_df = get_df("SPY")
    if spy_df is None or len(spy_df) < LOOKBACK + 1:
        print("ERROR: Insufficient SPY data"); sys.exit(1)
    spy_closes = spy_df["Close"].values
    spy_highs = spy_df["High"].values
    spy_lows = spy_df["Low"].values
    spy_atr_series = compute_atr_series(spy_highs, spy_lows, spy_closes, ATR_PERIOD)
    spy_ts_map = {ts: i for i, ts in enumerate(spy_df.index)}

    spy_df_w = get_df_w("SPY")
    spy_w_closes = spy_df_w["Close"].values if spy_df_w is not None else np.array([])
    spy_w_highs = spy_df_w["High"].values if spy_df_w is not None else np.array([])
    spy_w_lows = spy_df_w["Low"].values if spy_df_w is not None else np.array([])
    spy_w_atr_series = compute_atr_series(spy_w_highs, spy_w_lows, spy_w_closes, ATR_PERIOD) if len(spy_w_closes) > ATR_PERIOD else []
    spy_w_ts_map = {ts: i for i, ts in enumerate(spy_df_w.index)} if spy_df_w is not None else {}

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
